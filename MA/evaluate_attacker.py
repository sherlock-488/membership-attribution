import os
import sys
import numpy as np
import sklearn.metrics as metrics
import torch
from torchmetrics import Accuracy, Precision, Recall, F1Score, AUROC
from util import *

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.append(ROOT_DIR)
from model_snapshot import model_dict

# pt_ft imports
try:
    sys.path.append(os.path.dirname(__file__))
    from pt_ft_util import shadowDataset as ptft_shadowDataset, SAVE_PATH as PTFT_SAVE_PATH, OLOMA_GROUP_SIZE, MIMIR_GROUP_SIZE, WIKIMIA_GROUP_SIZE
except ImportError:
    ptft_shadowDataset = None
    PTFT_SAVE_PATH = SAVE_PATH

def get_TP_FP(preds, y_test, FP):
    preds = preds.numpy()

    fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
    roc_auc = metrics.auc(fpr, tpr)
    acc_max = np.argmax(1-(fpr+(1-tpr))/2)
    return threshold[np.where(fpr<=FP)[0][-1]], threshold[acc_max]

def probs_to_bit_probs(probs: torch.Tensor, mat) -> torch.Tensor:
    mat = mat.to(probs.device)
    return probs @ mat

def compute_auc(pred, y_test, num_classes):
    if num_classes == 4:
        BIT_PROB_MATRIX = torch.tensor(
            [[1, 0],  # class 0 → bit0=1, bit1=0
            [1, 1],  # class 1 → bit0=1, bit1=1
            [0, 1],  # class 2 → bit0=0, bit1=1
            [0, 0]], # class 3 → bit0=0, bit1=0
            dtype=torch.float32,
        )
    elif num_classes == 3:
        BIT_PROB_MATRIX= torch.tensor(
            [[1, 0],  # class 0 → bit0=1, bit1=0
            [0, 1],  # class 1 → bit0=0, bit1=1
            [0, 0]], # class 2 → bit0=0, bit1=0
            dtype=torch.float32,
        )
    bit_probs = probs_to_bit_probs(pred, BIT_PROB_MATRIX)
    bit_labels = BIT_PROB_MATRIX.to(y_test.device)[y_test]  # shape: (N, 2)
    auroc_bin = AUROC(task="binary")

    bit0_auc = auroc_bin(bit_probs[:, 0], bit_labels[:, 0].long())
    bit1_auc = auroc_bin(bit_probs[:, 1], bit_labels[:, 1].long())
    print(f"bit0 AUC = {bit0_auc:.4f}, bit1 AUC = {bit1_auc:.4f}")
    return (bit0_auc + bit1_auc) / 2

def eval_ptft():
    attack = sys.argv[1]
    target_model_name = model_dict[sys.argv[2]]
    shadow_model_name = model_dict[sys.argv[3]]
    dataname = sys.argv[4]
    pt_access = sys.argv[5]
    ft_access = sys.argv[6]
    feature_dim = int(sys.argv[7])
    seed = int(sys.argv[8]) if len(sys.argv) > 8 else 3407

    if ptft_shadowDataset is None:
        raise ImportError("pt_ft_util not available for pt_ft evaluation path")

    targetfilepath = PTFT_SAVE_PATH + f'output/{target_model_name}/{dataname}/{attack}_{pt_access}_{ft_access}_target_{feature_dim}.pt'
    model_path = os.path.join(
        PTFT_SAVE_PATH,
        "output",
        shadow_model_name,
        f"{dataname}_{attack}_{pt_access}_{ft_access}_{feature_dim}_seed{seed}.pth"
    )

    if dataname == 'oloma':
        num_per_class = OLOMA_GROUP_SIZE
    elif dataname in ['mimir', 'wikimia']:
        raw_preds = torch.load(targetfilepath)
        num_per_class = len(raw_preds) // 4
        if num_per_class == 0:
            raise ValueError(f"{dataname} target features too few examples: total_len={len(raw_preds)}")
    else:
        num_per_class = 2500

    testset = ptft_shadowDataset(targetfilepath, num_per_class=num_per_class)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)
    model = torch.load(model_path, map_location='cuda', weights_only=False).to('cuda')
    model.eval()

    accuracy = Accuracy(task="multiclass", num_classes=4).to('cuda')
    precision = Precision(task="multiclass", average='none', num_classes=4).to('cuda')
    recall = Recall(task="multiclass", average='none', num_classes=4).to('cuda')
    f1 = F1Score(task="multiclass", average='macro', num_classes=4).to('cuda')
    auc_macro = AUROC(task="multiclass", average='macro', num_classes=4).to('cuda')

    preds = []
    y_test = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to('cuda')
            outputs = torch.nn.functional.softmax(model(inputs), -1)
            preds.append(outputs.cpu())
            y_test.append(labels)

    pred = torch.concat(preds)
    y_true = torch.concat(y_test)

    acc_value = accuracy(pred, y_true)
    pre_value = precision(pred, y_true)
    rec_value = recall(pred, y_true)
    f1_value = f1(pred, y_true)
    auc_value = auc_macro(pred, y_true)
    auc_bits = compute_auc(pred, y_true, 4)

    print(f'{acc_value=}, {pre_value=}, {rec_value=}, {f1_value=}, {auc_bits=}')


def eval_legacy():
    attack = sys.argv[1]
    target_model_name = model_dict[sys.argv[2]]
    shadow_model_name = model_dict[sys.argv[3]]
    target_mode = sys.argv[4]
    dataname = sys.argv[5]
    pt_access = 'open'
    ft_access = 'open'
    feature_dim = 32

    targetfilepath = SAVE_PATH+f'output/{target_mode}/{target_model_name}/{dataname}/{attack}_{pt_access}_{ft_access}_target_{feature_dim}.pt'

    testset = shadowDataset(targetfilepath, target_mode)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)
    model = torch.load(SAVE_PATH+f'output/{target_mode}/{shadow_model_name}/{dataname}_{attack}_{pt_access}_{ft_access}_{feature_dim}.pth')

    pred = []
    y_test = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to('cuda')
            outputs = torch.nn.functional.softmax(model(inputs), -1)
            pred.append(outputs.cpu())
            y_test.append(labels)

    pred = torch.concat(pred)
    y_test = torch.concat(y_test)

    accuracy = Accuracy(task="multiclass", num_classes=testset.num_classes)
    precision = Precision(task="multiclass", average='none', num_classes=testset.num_classes)
    recall = Recall(task="multiclass", average='none', num_classes=testset.num_classes)
    f1 = F1Score(task="multiclass", average='macro', num_classes=testset.num_classes)
    auc = AUROC(task="multiclass", average='macro', num_classes=testset.num_classes)

    acc_value = accuracy(pred, y_test)
    pre_value = precision(pred, y_test)
    rec_value = recall(pred, y_test)
    f1_value = f1(pred, y_test)
    auc_value = auc(pred, y_test)
    auc_bits = compute_auc(pred, y_test, testset.num_classes)

    print(f'{acc_value=}, {pre_value=}, {rec_value=}, {f1_value=}, {auc_bits=}')


if __name__ == '__main__':
    if len(sys.argv) >= 8 and sys.argv[4] not in ["pt_pt", "ft_ft"]:
        eval_ptft()
    else:
        eval_legacy()
    
