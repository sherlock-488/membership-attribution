import os
import sys
import random
import numpy as np
import torch
from torchmetrics.classification import Accuracy
import sklearn.metrics as sk_metrics
from util import *
from transformer_util import *

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


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_ptft_args(argv):
    attack = argv[1]
    target_model_name = model_dict[argv[2]]
    shadow_model_name = model_dict[argv[3]]
    dataname = argv[4]
    pt_access = argv[5]
    ft_access = argv[6]
    feature_dim = int(argv[7])
    seed = int(argv[8]) if len(argv) > 8 else 3407
    return attack, target_model_name, shadow_model_name, dataname, pt_access, ft_access, feature_dim, seed


def load_classic_args(argv):
    attack = argv[1]
    target_model_name = model_dict[argv[2]]
    shadow_model_name = model_dict[argv[3]]
    target_mode = argv[4]
    dataname = argv[5]
    return attack, target_model_name, shadow_model_name, target_mode, dataname


def main():
    # Detect CLI style: legacy (5 args) vs pt_ft (>=8 args)
    if len(sys.argv) >= 8 and sys.argv[4] not in ["pt_pt", "ft_ft"]:
        # pt_ft style
        attack, target_model_name, shadow_model_name, dataname, pt_access, ft_access, feature_dim, seed = load_ptft_args(sys.argv)
        set_global_seed(seed)

        if ptft_shadowDataset is None:
            raise ImportError("pt_ft_util not available for pt_ft training path")

        shadowfilepath = PTFT_SAVE_PATH + f'output/{shadow_model_name}/{dataname}/{attack}_{pt_access}_{ft_access}_shadow_{feature_dim}.pt'
        targetfilepath = PTFT_SAVE_PATH + f'output/{target_model_name}/{dataname}/{attack}_{pt_access}_{ft_access}_target_{feature_dim}.pt'

        # num_per_class inference
        if dataname == 'oloma':
            num_per_class = OLOMA_GROUP_SIZE
        elif dataname in ['mimir', 'wikimia']:
            raw_preds = torch.load(shadowfilepath)
            num_per_class = len(raw_preds) // 4
            if num_per_class == 0:
                raise ValueError(f"{dataname} shadow features too few examples: total_len={len(raw_preds)}")
        else:
            num_per_class = 2500

        trainset = ptft_shadowDataset(shadowfilepath, num_per_class=num_per_class)
        testset = ptft_shadowDataset(targetfilepath, num_per_class=num_per_class)
        training_loader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=True)
        loss_fn = torch.nn.CrossEntropyLoss()

        context_size = trainset[0][0].shape[0]
        model = Transformer(context_size=context_size, d_model=feature_dim, d_ff=16, num_heads=2, n_blocks=2, n_classes=4, use_pos_embedding=True).to('cuda')
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        metric = Accuracy(task="multiclass", num_classes=4).to('cuda')
        best_acc = 0.0

        def train_one_epoch():
            running_loss = 0.0
            for inputs, labels in training_loader:
                inputs, labels = inputs.to('cuda'), labels.to('cuda')
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                metric(outputs, labels)
            all_acc = metric.compute().item()
            metric.reset()
            return running_loss / len(training_loader), all_acc

        def evaluate_accuracy(dataloader):
            all_probs = []
            all_labels = []
            with torch.no_grad():
                running_loss = 0.0
                for inputs, labels in dataloader:
                    inputs, labels = inputs.to('cuda'), labels.to('cuda')
                    outputs = model(inputs)
                    loss = loss_fn(outputs, labels)
                    running_loss += loss.item()
                    metric(outputs, labels)
                    probs = torch.softmax(outputs, dim=-1)
                    all_probs.append(probs.cpu())
                    all_labels.append(labels.cpu())

                test_acc = metric.compute().item()
                metric.reset()

            probs_np = torch.cat(all_probs, dim=0).numpy()
            y_np = torch.cat(all_labels, dim=0).numpy()

            bit1_true = ((y_np == 0) | (y_np == 1)).astype(int)
            bit2_true = ((y_np == 0) | (y_np == 2)).astype(int)
            bit1_score = probs_np[:, 0] + probs_np[:, 1]
            bit2_score = probs_np[:, 0] + probs_np[:, 2]

            try:
                auc_bit1 = sk_metrics.roc_auc_score(bit1_true, bit1_score)
                auc_bit2 = sk_metrics.roc_auc_score(bit2_true, bit2_score)
                auc_bits = 0.5 * (auc_bit1 + auc_bit2)
            except ValueError:
                auc_bits = float("nan")

            return test_acc, running_loss/len(dataloader), auc_bits

        attacker_dir = os.path.join(PTFT_SAVE_PATH, "output", shadow_model_name)
        os.makedirs(attacker_dir, exist_ok=True)
        attacker_model_path = os.path.join(
            attacker_dir,
            f"{dataname}_{attack}_{pt_access}_{ft_access}_{feature_dim}_seed{seed}.pth"
        )

        for epoch in range(50):
            model.train()
            train_loss, train_acc = train_one_epoch()
            model.eval()
            test_acc, test_loss, test_auc = evaluate_accuracy(test_loader)
            if test_acc >= best_acc:
                best_acc = test_acc
                torch.save(model, attacker_model_path)
            print(
                f"Epoch:{epoch} "
                f"loss: {train_loss:.4f}, acc:{train_acc:.4f} "
                f"test loss: {test_loss:.4f}, test acc: {test_acc:.4f}, "
                f"test auc(bits): {test_auc:.4f}, best acc: {best_acc:.4f}"
            )

    else:
        # legacy style
        attack, target_model_name, shadow_model_name, target_mode, dataname = load_classic_args(sys.argv)
        pt_access = "open"
        ft_access = "open"
        feature_dim = 32
        shadowfilepath = SAVE_PATH + f'output/{target_mode}/{shadow_model_name}/{dataname}/{attack}_{pt_access}_{ft_access}_shadow_{feature_dim}.pt'
        targetfilepath = SAVE_PATH + f'output/{target_mode}/{target_model_name}/{dataname}/{attack}_{pt_access}_{ft_access}_target_{feature_dim}.pt'

        trainset = shadowDataset(shadowfilepath, target_mode)
        testset = shadowDataset(targetfilepath, target_mode)
        training_loader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=True)
        loss_fn = torch.nn.CrossEntropyLoss()

        model = Transformer(context_size=255, d_model=feature_dim, d_ff=16, num_heads=4, n_blocks=4, n_classes=trainset.num_classes).to('cuda')
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  
        metric = Accuracy(task="multiclass", num_classes=trainset.num_classes).to('cuda')
        best_acc = 0

        def train_one_epoch():
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

        def evaluate_accuracy(dataloader):
            with torch.no_grad():
                running_loss = 0.
                for inputs, labels in dataloader:
                    inputs, labels = inputs.to('cuda'), labels.to('cuda')
                    outputs = model(inputs)
                    loss = loss_fn(outputs, labels)
                    running_loss += loss
                    metric(outputs, labels)
                test_acc = metric.compute()
                metric.reset()
            return test_acc, running_loss/len(dataloader)

        for epoch in range(50):
            model.train()
            train_loss, train_acc = train_one_epoch()
            model.eval()
            test_acc, test_loss = evaluate_accuracy(test_loader)
            if test_acc >= best_acc:
                best_acc = test_acc
                torch.save(model, SAVE_PATH+f'output/{target_mode}/{shadow_model_name}/{dataname}_{attack}_{pt_access}_{ft_access}_{feature_dim}.pth')
            print(f"Epoch:{epoch} loss: {train_loss}, acc:{train_acc} test loss: {test_loss}, test acc: {test_acc}, best acc: {best_acc}")


if __name__ == "__main__":
    main()
