import pickle
import torch
import numpy as np
import sklearn.metrics as metrics
from torchmetrics import Accuracy, Precision, Recall, F1Score, AUROC
import sys
from TargetModel import model_dict

attack = sys.argv[1]
dataname = sys.argv[2]
target_model_name = model_dict[sys.argv[3]]
shadow_model_name = model_dict[sys.argv[4]]
target_mode = 'ft_ft'

def load_scores(mode):
    
    if mode == 'target': model_name = target_model_name
    else: model_name = shadow_model_name
    print(f"{mode}: {model_name}")
    file = f'output/{target_mode}/{model_name}/{dataname}/{attack}_first_{mode}.pt'
    scores_first = torch.load(file)
    file = f'output/{target_mode}/{model_name}/{dataname}/loss_first_{mode}.pt'
    scores_loss_first = torch.load(file)

    file = f'output/{target_mode}/{model_name}/{dataname}/loss_second_{mode}.pt'
    scores_second = torch.load(file)
    scores_MIU = [score_first - score_second for score_first, score_second in zip(scores_loss_first, scores_second)]
    return scores_first, scores_MIU, scores_second

scores_first_target, scores_MIU_target, scores_second_target = load_scores('target')
scores_first_shadow, scores_MIU_shadow, scores_second_shadow = load_scores('shadow')

def find_thresholds_first(scores_first_shadow):
    scores_first_shadow = -np.array(scores_first_shadow)
    scores_first_shadow = np.nan_to_num(scores_first_shadow)
    y_true = np.concatenate((np.ones(5000), np.zeros(5000)))
    fpr, tpr, thresholds = metrics.roc_curve(y_true, scores_first_shadow)
    youden_index = tpr - fpr
    optimal_idx = np.argmax(youden_index)
    optimal_threshold = thresholds[optimal_idx]
    y_pred = (scores_first_shadow >= optimal_threshold).astype(int)
    acc = np.mean(y_pred == y_true)
    print(f"Optimal threshold: {optimal_threshold}, Accuracy: {acc}")
    return optimal_threshold

def find_thresholds_second(scores_MIU):
    scores = np.nan_to_num(scores_MIU)
    y_true = np.concatenate((np.zeros(2500), np.ones(2500), np.ones(2500), np.zeros(2500)))
    fpr, tpr, thresholds = metrics.roc_curve(y_true, scores)
    youden_index = tpr - fpr
    optimal_idx = np.argmax(youden_index)
    optimal_threshold = thresholds[optimal_idx]
    y_pred = (scores >= optimal_threshold).astype(int)
    acc = np.mean(y_pred == y_true)
    print(f"Optimal threshold: {optimal_threshold}, Accuracy: {acc}")
    return optimal_threshold

def evaluate(scores_first, scores_MIU, threshold_first, threshold_second):
    preds = np.full(len(scores_first), 0)
    scores_first_arr = np.nan_to_num(np.array(scores_first))
    scores_first_pred = -scores_first_arr
    scores_delta = np.nan_to_num(np.array(scores_MIU))
    BIT_TO_CLASSES = {'10': 0, '11': 1, '01': 2, '00': 3}
    for i in range(len(scores_first_arr)):
        pred = ''
        if scores_first_pred[i] >= threshold_first:
            pred += '1'
        else:
            pred += '0'
        if scores_MIU[i] >= threshold_second:
            pred += '1'
        else:
            pred += '0'
        preds[i] = BIT_TO_CLASSES[pred]

    y_true = np.concatenate((np.zeros(2500), np.ones(2500), np.full(2500, 2), np.full(2500, 3)))
    accuracy = np.mean(preds == y_true)
    print(f"Evaluation Accuracy: {accuracy}")

    f1 = metrics.f1_score(y_true, preds, average='macro', labels=[0, 1, 2, 3])
    print(f"F1 Score (macro, 4 classes): {f1}")


def compute_macro_auc(scores_first, scores_MIU, scores_second):
    scores_first_arr = np.nan_to_num(np.array(scores_first))
    scores_delta = np.nan_to_num(np.array(scores_MIU))
    scores_second_arr = np.nan_to_num(np.array(scores_second))

    labels_first = np.concatenate((np.ones(5000), np.zeros(5000)))
    labels_second = np.concatenate((np.zeros(2500), np.ones(2500), np.ones(2500), np.zeros(2500)))
    # labels_third = np.concatenate((np.zeros(2000), np.ones(1000)))

    auc_first = metrics.roc_auc_score(labels_first, -scores_first_arr)
    auc_second = metrics.roc_auc_score(labels_second, scores_delta)
    # auc_third = metrics.roc_auc_score(labels_third, scores_second_arr)
    macro_auc = np.mean([auc_first, auc_second])
    print(f"AUC <1,0>: {auc_first}, AUC <0,1>: {auc_second}, Macro AUC: {macro_auc}")
    print("--------------------------------------")

threshold_first = find_thresholds_first(scores_first_shadow)
threshold_second = find_thresholds_second(scores_MIU_shadow)
evaluate(scores_first_target, scores_MIU_target, threshold_first, threshold_second)
compute_macro_auc(scores_first_target, scores_MIU_target, scores_second_target)
