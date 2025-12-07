# train_attacker.py
from util import *
from transformer_util import *
from tqdm import tqdm
import os
from torchmetrics.classification import Accuracy
import torch
import sys
import random
import numpy as np
import sklearn.metrics as sk_metrics  # 新增：用于算 AUC

# ===== 解析命令行参数 =====
attack = sys.argv[1]
target_model_name = model_dict[sys.argv[2]]
shadow_model_name = model_dict[sys.argv[3]]
dataname = sys.argv[4]
pt_access = sys.argv[5]
ft_access = sys.argv[6]
feature_dim = int(sys.argv[7])

# 可选：第 8 个参数为随机种子；默认 3407
if len(sys.argv) > 8:
    seed = int(sys.argv[8])
else:
    seed = 3407


def set_global_seed(seed: int):
    """统一控制 python / numpy / torch 的随机性"""
    print(f"[train_attacker] Using seed={seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # 让 cudnn 更可复现（会稍微慢一点）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_global_seed(seed)

shadowfilepath = SAVE_PATH + f'output/{shadow_model_name}/{dataname}/{attack}_{pt_access}_{ft_access}_shadow_{feature_dim}.pt'
targetfilepath = SAVE_PATH + f'output/{target_model_name}/{dataname}/{attack}_{pt_access}_{ft_access}_target_{feature_dim}.pt'

# ===== 每类样本数 =====
if dataname == 'oloma':
    # oloma 仍然用固定的 group size
    num_per_class = OLOMA_GROUP_SIZE
elif dataname in ['mimir', 'wikimia']:
    # mimir / wikimia: 动态根据特征文件长度推断，每类占 1/4
    raw_preds = torch.load(shadowfilepath)
    total_len = len(raw_preds)
    num_per_class = total_len // 4
    if num_per_class == 0:
        raise ValueError(f"{dataname} shadow features too few examples: total_len={total_len}")
else:
    num_per_class = 2500

# 重新构建 dataset，使用正确的 num_per_class
trainset = shadowDataset(shadowfilepath, num_per_class=num_per_class)
testset = shadowDataset(targetfilepath, num_per_class=num_per_class)

# 自动推断序列长度（L-1）
context_size = trainset[0][0].shape[0]

training_loader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True)
test_loader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=True)
loss_fn = torch.nn.CrossEntropyLoss()

# 四分类：Y/B/Z/N （这里对应 A/B/C/D 四类）
model = Transformer(
    context_size=context_size,
    d_model=feature_dim,
    d_ff=16,
    num_heads=2,
    n_blocks=2,
    n_classes=4
).to('cuda')
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
metric = Accuracy(task="multiclass", num_classes=4).to('cuda')
best_acc = 0.0

# attacker 模型按 seed 单独存
attacker_dir = os.path.join(SAVE_PATH, "output", shadow_model_name)
os.makedirs(attacker_dir, exist_ok=True)
attacker_model_path = os.path.join(
    attacker_dir,
    f"{dataname}_{attack}_{pt_access}_{ft_access}_{feature_dim}_seed{seed}.pth"
)


def train_one_epoch():
    running_loss = 0.0
    for (inputs, labels) in training_loader:
        inputs, labels = inputs.to('cuda'), labels.to('cuda')
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        metric(outputs, labels)
    all_acc = metric.compute().item()  # 转成 float
    metric.reset()
    return running_loss / len(training_loader), all_acc


def evaluate_accuracy(dataloader):
    """在目标集上评估：返回 test_acc, test_loss, test_auc(bits)"""
    all_probs = []
    all_labels = []
    with torch.no_grad():
        running_loss = 0.0
        for (inputs, labels) in dataloader:
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
            outputs = model(inputs)                     # [B, 4]
            loss = loss_fn(outputs, labels)
            running_loss += loss.item()
            metric(outputs, labels)

            probs = torch.softmax(outputs, dim=-1)     # [B, 4]
            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())

        test_acc = metric.compute().item()
        metric.reset()

    probs = torch.cat(all_probs, dim=0).numpy()        # [N, 4]
    y_true = torch.cat(all_labels, dim=0).numpy()      # [N]

    # bit1: pretrain member -> Y,B=1; Z,N=0
    bit1_true = ((y_true == 0) | (y_true == 1)).astype(int)
    # bit2: finetune member -> Y,Z=1; B,N=0
    bit2_true = ((y_true == 0) | (y_true == 2)).astype(int)

    # 分数定义和 evaluate_attacker.py 一致
    bit1_score = probs[:, 0] + probs[:, 1]   # P(Y)+P(B)
    bit2_score = probs[:, 0] + probs[:, 2]   # P(Y)+P(Z)

    try:
        auc_bit1 = sk_metrics.roc_auc_score(bit1_true, bit1_score)
        auc_bit2 = sk_metrics.roc_auc_score(bit2_true, bit2_score)
        auc_bits = 0.5 * (auc_bit1 + auc_bit2)
    except ValueError:
        # 某个 bit 只有一个类时 AUC 会炸
        auc_bits = float("nan")

    avg_loss = running_loss / len(dataloader)
    return test_acc, avg_loss, auc_bits


for epoch in range(50):
    model.train()
    train_loss, train_acc = train_one_epoch()
    model.eval()
    test_acc, test_loss, test_auc = evaluate_accuracy(test_loader)
    if test_acc >= best_acc:
        best_acc = float(test_acc)
        torch.save(model, attacker_model_path)
    print(
        f"Epoch:{epoch} "
        f"loss: {train_loss:.4f}, acc:{train_acc:.4f} "
        f"test loss: {test_loss:.4f}, test acc: {test_acc:.4f}, "
        f"test auc(bits): {test_auc:.4f}, best acc: {best_acc:.4f}"
    )
