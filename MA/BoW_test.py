import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer
from util import *
import sys
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset, Dataset
from torchmetrics.classification import Accuracy
from torchmetrics import Precision, Recall, F1Score, AUROC
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from MADataset import MADataset, MADataset_FT_FT
from evaluate_attacker import compute_auc

# pt_ft imports for extra datasets
sys.path.append(os.path.join(os.path.dirname(__file__), "MA"))
try:
    from pt_ft_util import load_eval_data as ptft_load_eval_data
except ImportError:
    ptft_load_eval_data = None


class BoWDatasetClassic(Dataset):
    """Original BoW dataset for pile/fineweb splits."""
    def __init__(self, tokenizer, dataname):
        if dataname == 'pile':
            self.dataset = MADataset(dataname=dataname, mode='target').dataset
            self.num_classes = 3
            self.num_per_class = 1000
            self.pre_tokenized = True
            self.vocab = {}
            for tokens in self.dataset:
                for token in tokens:
                    if token not in self.vocab:
                        self.vocab[token] = len(self.vocab)
        elif dataname == 'fineweb':
            self.dataset = MADataset_FT_FT(dataname=dataname, mode='target').dataset
            self.num_classes = 4
            self.num_per_class = 2500
            self.pre_tokenized = False
            self.vocab = {}
            for sentence in self.dataset:
                tokens = tokenizer.tokenize(sentence['text'], max_length=128, truncation=True)
                for token in tokens:
                    if token not in self.vocab:
                        self.vocab[token] = len(self.vocab)
        else:
            raise ValueError("Unsupported dataname for classic BoW")
        self.VOCAB_SIZE = len(self.vocab)
        self.tokenizer = tokenizer

    def make_bow_vector(self, sample):
        vec = torch.zeros(self.VOCAB_SIZE)
        tokens = sample if self.pre_tokenized else self.tokenizer.tokenize(sample['text'], max_length=128, truncation=True)
        for token in tokens:
            if token in self.vocab:
                vec[self.vocab[token]] += 1
        return vec

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.make_bow_vector(self.dataset[idx]), idx//self.num_per_class


class BoWDatasetPtFt(Dataset):
    """BoW for pt_ft datasets (mimir/oloma/wikimia)."""
    def __init__(self, tokenizer, dataname: str):
        if ptft_load_eval_data is None:
            raise ImportError("pt_ft_util not available for pt_ft BoW")
        self.dataset = ptft_load_eval_data(dataname, "target")
        self.num_classes = 4
        self.num_per_class = len(self.dataset) // self.num_classes
        self.vocab = {}
        for sentence in self.dataset["text"]:
            tokens = tokenizer.tokenize(
                sentence,
                max_length=128,
                truncation=True,
            )
            for tok in tokens:
                if tok not in self.vocab:
                    self.vocab[tok] = len(self.vocab)
        self.VOCAB_SIZE = len(self.vocab)
        self.tokenizer = tokenizer

    def make_bow_vector(self, sample):
        vec = torch.zeros(self.VOCAB_SIZE, dtype=torch.float32)
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
        label = idx // self.num_per_class
        return bow_vec, label


class BoWClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes):
        super().__init__()
        self.linear = nn.Linear(vocab_size, num_classes)

    def forward(self, bow_vec):
        return self.linear(bow_vec)


def train_one_epoch(model, training_loader, loss_fn, metric):
    running_loss = 0.
    for inputs, labels in training_loader:
        inputs, labels = inputs.to('cuda'), labels.to('cuda')
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        metric(outputs, labels)
    all_acc = metric.compute()
    metric.reset()
    return running_loss/len(training_loader), all_acc


def evaluate_accuracy(model, dataloader, loss_fn, metric, precision, recall, f1, num_classes):
    with torch.no_grad():
        running_loss = 0.
        preds, y_trues = [], []
        for inputs, labels in dataloader:
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
            outputs = model(inputs)
            pred = torch.nn.functional.softmax(outputs, -1)
            preds.append(pred.cpu())
            y_trues.append(labels.cpu())
            loss = loss_fn(outputs, labels)
            running_loss += loss
            metric(outputs, labels)
            precision(outputs, labels)
            recall(outputs, labels)
            f1(outputs, labels)
        test_acc = metric.compute()
        pre_value = precision.compute()
        rec_value = recall.compute()
        f1_value = f1.compute()
        auc_value = compute_auc(torch.concat(preds), torch.concat(y_trues), num_classes)
        metric.reset(); precision.reset(); recall.reset(); f1.reset()
    return test_acc, running_loss/len(dataloader), pre_value, rec_value, f1_value, auc_value


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python BoW_test.py <dataname>")
        sys.exit(1)
    dataname = sys.argv[1]

    tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neo-125m')
    tokenizer.pad_token = tokenizer.eos_token

    ptft_datasets = {"mimir", "oloma", "wikimia"}
    if dataname in ptft_datasets:
        dataset = BoWDatasetPtFt(tokenizer, dataname)
    else:
        dataset = BoWDatasetClassic(tokenizer, dataname)

    train_indices, test_indices = train_test_split(list(range(len(dataset))), test_size=0.2, random_state=2023)
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)
    training_loader = DataLoader(train_dataset, batch_size=50)
    test_loader = DataLoader(test_dataset, batch_size=50)
    NUM_CLASSES = dataset.num_classes

    model = BoWClassifier(dataset.VOCAB_SIZE, NUM_CLASSES).to('cuda')
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    metric = Accuracy(task="multiclass", num_classes=NUM_CLASSES).to('cuda')
    precision = Precision(task="multiclass", average='none', num_classes=NUM_CLASSES).to('cuda')
    recall = Recall(task="multiclass", average='none',num_classes=NUM_CLASSES).to('cuda')
    f1 = F1Score(task="multiclass", average='macro', num_classes=NUM_CLASSES).to('cuda')

    best_acc = 0
    for epoch in range(30):
        model.train()
        train_loss, train_acc = train_one_epoch(model, training_loader, loss_fn, metric)
        model.eval()
        test_acc, test_loss, pre_value, rec_value, f1_value, auc_value = evaluate_accuracy(
            model, test_loader, loss_fn, metric, precision, recall, f1, NUM_CLASSES
        )
        if test_acc >= best_acc:
            best_acc = test_acc
        print(f"Epoch:{epoch} loss: {train_loss}, acc:{train_acc} test loss: {test_loss}, test acc: {test_acc}, best acc: {best_acc}")
        print(f'{test_acc=}, {pre_value=}, {rec_value=}, {f1_value=}, {auc_value=}')
