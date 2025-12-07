# evaluate_attacker.py
from util import *
from tqdm import tqdm
import os
import sklearn.metrics as metrics
from torchmetrics import Accuracy, F1Score, AUROC
import sys
import numpy as np
import torch

attack = sys.argv[1]
target_model_name = model_dict[sys.argv[2]]
shadow_model_name = model_dict[sys.argv[3]]
dataname = sys.argv[4]
pt_access = sys.argv[5]
ft_access = sys.argv[6]
feature_dim = int(sys.argv[7])

# 可选：第 8 个参数为 seed，用于选择对应的 attacker pth
if len(sys.argv) > 8:
    seed = int(sys.argv[8])
else:
    seed = None

# ===== 每类样本数（与 train_attacker 对齐） =====
if dataname == 'oloma':
    num_per_class = OLOMA_GROUP_SIZE
elif dataname in ['mimir', 'wikimia']:
    targetfilepath_tmp = SAVE_PATH + f'output/{target_model_name}/{dataname}/{attack}_{pt_access}_{ft_access}_target_{feature_dim}.pt'
    raw_preds = torch.load(targetfilepath_tmp)
    total_len = len(raw_preds)
    num_per_class = total_len // 4
    if num_per_class == 0:
        raise ValueError(f"{dataname} target features too few examples: total_len={total_len}")
else:
    num_per_class = 2500

# 读取 target 特征作为测试集
targetfilepath = SAVE_PATH + f'output/{target_model_name}/{dataname}/{attack}_{pt_access}_{ft_access}_target_{feature_dim}.pt'
testset = shadowDataset(targetfilepath, num_per_class=num_per_class)
test_loader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# 读取攻击模型（按 seed 区分）
if seed is None:
    # 兼容老版本：没给 seed 时，用旧路由
    model_path = SAVE_PATH + f'output/{shadow_model_name}/{dataname}_{attack}_{pt_access}_{ft_access}_{feature_dim}.pth'
else:
    model_path = SAVE_PATH + f'output/{shadow_model_name}/{dataname}_{attack}_{pt_access}_{ft_access}_{feature_dim}_seed{seed}.pth'

print(f"[evaluate_attacker] loading attacker from {model_path}")
model = torch.load(model_path, map_location='cuda', weights_only=False)
model.eval()

all_probs = []
all_labels = []

with torch.no_grad():
    for (inputs, labels) in test_loader:
        inputs = inputs.to('cuda')
        logits = model(inputs)
        probs = torch.nn.functional.softmax(logits, dim=-1)  # [B, 4]
        all_probs.append(probs.cpu())
        all_labels.append(labels)

probs = torch.cat(all_probs, dim=0)    # [N, 4]
y_true = torch.cat(all_labels, dim=0)  # [N]

# 取预测类别
y_pred = torch.argmax(probs, dim=-1)   # [N]
# ------- 4x4 混淆矩阵 -------
y_true_np = y_true.numpy()
y_pred_np = y_pred.numpy()

cm = metrics.confusion_matrix(
    y_true_np,
    y_pred_np,
    labels=[0, 1, 2, 3]
)  # 4x4，cm[i,j] = # true=i, pred=j

# 按行归一化：每行变成条件概率 P(pred=j | true=i)
row_sums = cm.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1
rate_matrix = cm / row_sums

print("4x4 预测率表（行 = 真实类别 y, 列 = 预测类别 ŷ，数值 = P(ŷ=j | y=i) ）")
header = "      " + "  ".join([f"pred={j}" for j in range(4)])
print(header)
for i in range(4):
    row_str = "  ".join(f"{rate_matrix[i, j]:.4f}" for j in range(4))
    print(f"true={i}  {row_str}")

# ------- bit1 / bit2 正确率 -------
# 约定：
#   类 0,1: pretrain_bit = 1；类 2,3: pretrain_bit = 0
#   类 0,2: finetune_bit = 1；类 1,3: finetune_bit = 0
true_bit1 = ((y_true == 0) | (y_true == 1)).int()   # [N]
pred_bit1 = ((y_pred == 0) | (y_pred == 1)).int()

true_bit2 = ((y_true == 0) | (y_true == 2)).int()
pred_bit2 = ((y_pred == 0) | (y_pred == 2)).int()

bit1_acc = (true_bit1 == pred_bit1).float().mean().item()
bit2_acc = (true_bit2 == pred_bit2).float().mean().item()

# ------- 4-class ACC / BalancedAcc / F1 -------
accuracy_metric = Accuracy(task="multiclass", num_classes=4)
f1_metric = F1Score(task="multiclass", average='macro', num_classes=4)

acc_value = accuracy_metric(probs, y_true).item()
f1_value = f1_metric(probs, y_true).item()

# BalancedAcc = 平均 per-class recall
from torchmetrics import Recall
recall_metric = Recall(task="multiclass", average='none', num_classes=4)
rec_per_class = recall_metric(probs, y_true).cpu().numpy()  # [4]
bal_acc = float(rec_per_class.mean())

# ------- bit-level AUC: (AUC_bit1 + AUC_bit2) / 2 -------
probs_np = probs.numpy()
bit1_true_np = true_bit1.numpy()
bit2_true_np = true_bit2.numpy()

# bit1=1 -> pretrain member (Y 或 B)；score = P(Y)+P(B)
bit1_score = probs_np[:, 0] + probs_np[:, 1]

# bit2=1 -> finetune member (Y 或 Z)；score = P(Y)+P(Z)
bit2_score = probs_np[:, 0] + probs_np[:, 2]

auc_bit1 = metrics.roc_auc_score(bit1_true_np, bit1_score)
auc_bit2 = metrics.roc_auc_score(bit2_true_np, bit2_score)
auc_bits = 0.5 * (auc_bit1 + auc_bit2)

print()
print(f'bit1 (pretrain membership) accuracy = {bit1_acc:.4f}')
print(f'bit2 (finetune  membership) accuracy = {bit2_acc:.4f}')

print(f'ACC      (4-class)   = {acc_value:.4f}')
print(f'BalancedAcc(4-class)       = {bal_acc:.4f}')
print(f'macro F1   (4-class)       = {f1_value:.4f}')
print(f'AUC(bit1+bit2)/2 (2 bits) = {auc_bits:.4f}')
