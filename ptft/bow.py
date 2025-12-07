import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from transformers import AutoTokenizer

from util import load_eval_data  # 现在只需要这个

from torchmetrics import Accuracy, F1Score

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class BoWDataset(Dataset):
    """
    在 load_eval_data(dataname, 'target') 上构造 BoW 特征，
    标签是 4 类：
      0: Y
      1: B
      2: Z
      3: N

    兼容数据集：
      - mimir
      - oloma
      - wikimia
    """
    def __init__(self, tokenizer, dataname: str):
        self.tokenizer = tokenizer

        # 旧的 3-class Pile 实验，这里直接禁止
        if dataname == "pile":
            raise ValueError("BoW 这个脚本是给 mimir / oloma / wikimia 四分类用的，别传 'pile' 进来。")

        # === 关键：用和 cascade_mia 一样的 eval 集合 ===
        # 对 mimir/oloma/wikimia：严格是 [Y, B, Z, N] 拼起来
        self.dataset = load_eval_data(dataname, "target")

        # 4 类 Y/B/Z/N
        self.num_classes = 4
        # 从长度倒推每类样本数，保证兼容 group_size 的动态调整
        self.num_per_class = len(self.dataset) // self.num_classes
        self.pre_tokenized = False

        # 构建 BoW 词表（简单地把 target eval 集全部扫一遍）
        self.vocab = {}
        for sentence in self.dataset["text"]:
            tokens = self.tokenizer.tokenize(
                sentence,
                max_length=128,
                truncation=True,
            )
            for tok in tokens:
                if tok not in self.vocab:
                    self.vocab[tok] = len(self.vocab)

        self.VOCAB_SIZE = len(self.vocab)
        print(
            f"[BoW] dataname={dataname}, vocab_size={self.VOCAB_SIZE}, "
            f"num_samples={len(self)}, num_per_class={self.num_per_class}"
        )

    def make_bow_vector(self, sample):
        """
        sample:
          - 如果 pre_tokenized=True：直接是一串 token
          - 否则：是 HF dataset 的一条 {'text': ...}
        """
        vec = torch.zeros(self.VOCAB_SIZE, dtype=torch.float32)
        if self.pre_tokenized:
            tokens = sample
        else:
            tokens = self.tokenizer.tokenize(
                sample["text"],
                max_length=128,
                truncation=True,
            )
        for tok in tokens:
            idx = self.vocab.get(tok)
            if idx is not None:
                vec[idx] += 1.0
        return vec

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        bow_vec = self.make_bow_vector(self.dataset[idx])
        # 利用严格 [Y, B, Z, N] 拼接的结构
        label = idx // self.num_per_class
        return bow_vec, label


class BoWClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes):
        super().__init__()
        self.linear = nn.Linear(vocab_size, num_classes)

    def forward(self, bow_vec):
        return self.linear(bow_vec)


def compute_bit_auc(probs: torch.Tensor, y_true: torch.Tensor) -> float:
    """
    probs: [N, 4]，softmax 后的四类概率
    y_true: [N]，0/1/2/3 = Y/B/Z/N

    bits 定义和 cascade_mia 一致：
      bit1 (pretrain member): Y,B=1；Z,N=0
      bit2 (update  member): Y,Z=1；B,N=0

    AUC = (AUC_bit1 + AUC_bit2) / 2
    """
    probs = probs.detach().cpu()
    y_true = y_true.detach().cpu()

    # 每类概率
    p_y = probs[:, 0]
    p_b = probs[:, 1]
    p_z = probs[:, 2]
    p_n = probs[:, 3]

    # bit1: pretrain member → P(Y)+P(B)
    score_b1 = p_y + p_b
    b1_true = ((y_true == 0) | (y_true == 1)).numpy().astype(int)

    # bit2: update member → P(Y)+P(Z)
    score_b2 = p_y + p_z
    b2_true = ((y_true == 0) | (y_true == 2)).numpy().astype(int)

    try:
        auc_b1 = roc_auc_score(b1_true, score_b1.numpy())
        auc_b2 = roc_auc_score(b2_true, score_b2.numpy())
        return 0.5 * (auc_b1 + auc_b2)
    except ValueError:
        # 某个 bit 只有一个类的时候 roc_auc_score 会炸
        return float("nan")


def train_one_epoch(model, train_loader, loss_fn, optimizer, acc_metric):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        logits = model(inputs)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        acc_metric.update(logits, labels)

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = acc_metric.compute().item()
    acc_metric.reset()
    return epoch_loss, epoch_acc


def evaluate(model, data_loader, loss_fn, acc_metric, f1_metric):
    model.eval()
    all_logits = []
    all_labels = []
    running_loss = 0.0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(inputs)
            loss = loss_fn(logits, labels)
            running_loss += loss.item()

            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

            acc_metric.update(logits, labels)
            f1_metric.update(logits, labels)

    logits = torch.cat(all_logits, dim=0)      # [N, 4]，logits
    probs = torch.softmax(logits, dim=-1)      # [N, 4]，概率
    y_true = torch.cat(all_labels, dim=0)      # [N]

    acc_value = acc_metric.compute().item()
    f1_value = f1_metric.compute().item()
    acc_metric.reset()
    f1_metric.reset()

    # 4-class test loss
    avg_loss = running_loss / len(data_loader)

    # bit-level AUC（两 bit 的平均）
    auc_bits = compute_bit_auc(probs, y_true)

    # 4-class ACC / macro F1 / AUC(bits) —— 打印成和 attacker / cascade 一样的格式
    print(f"ACC      (4-class)   = {acc_value:.4f}")
    print(f"macro F1   (4-class)       = {f1_value:.4f}")
    print(f"AUC(bit1+bit2)/2 (2 bits) = {auc_bits:.4f}")

    return acc_value, f1_value, auc_bits, avg_loss


def main():
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <dataname>")
        print("  dataname in {mimir, oloma, wikimia}")
        sys.exit(1)

    dataname = sys.argv[1].lower()
    if dataname not in {"mimir", "oloma", "wikimia"}:
        raise ValueError(f"BoW 目前只支持 dataname in {{mimir, oloma, wikimia}}，但你给的是 {dataname!r}")

    print(f"[BoW] dataname = {dataname}")

    # 1. tokenizer（随便挑一个 HF tokenizer 就行）
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125m")
    tokenizer.pad_token = tokenizer.eos_token

    # 2. dataset
    dataset = BoWDataset(tokenizer, dataname)
    NUM_CLASSES = dataset.num_classes

    # 3. train/test split（这里按样本随机 8:2 划分，依然是 membership 四分类）
    indices = list(range(len(dataset)))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, random_state=1111, shuffle=True
    )
    train_dataset = Subset(dataset, train_idx)
    test_dataset = Subset(dataset, test_idx)

    train_loader = DataLoader(train_dataset, batch_size=50, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=50, shuffle=False)

    # 4. 模型 & 优化器 & metric
    model = BoWClassifier(dataset.VOCAB_SIZE, NUM_CLASSES).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr=0.01)

    acc_metric = Accuracy(task="multiclass", num_classes=NUM_CLASSES).to(DEVICE)
    f1_metric = F1Score(
        task="multiclass",
        num_classes=NUM_CLASSES,
        average="macro"
    ).to(DEVICE)

    best_acc = 0.0

    # 5. 训练若干 epoch
    for epoch in range(30):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, loss_fn, optimizer, acc_metric
        )
        acc, f1_val, auc_bits, test_loss = evaluate(
            model, test_loader, loss_fn, acc_metric, f1_metric
        )

        if acc >= best_acc:
            best_acc = acc

        print(
            f"[Epoch {epoch:02d}] "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"test_loss={test_loss:.4f}, test_acc={acc:.4f}, "
            f"best_acc={best_acc:.4f}"
        )
        print("-" * 60)

    print("[BoW] Final best ACC(4-class) =", best_acc)


if __name__ == "__main__":
    main()
