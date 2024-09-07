import copy
import time
from collections import defaultdict
import torch.nn as nn
import pandas as pd
import torch.optim as optim
import numpy as np
import torch
from torcheval.metrics.functional import binary_auroc
from tqdm import tqdm
import gc

from example_utils import valid_one_epoch
from pAUC import score
from pytorch_baseline.p_baseline_constants import CONFIG
from pytorch_baseline.p_baseline_utils import weighted_bce_loss, ISICDataset_for_Train, ISICDataset, bce_loss, VICReg, \
    custom_VICReg, just_covariance
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler


def train_one_epoch(model, optimizer, scheduler, dataloader, device, epoch, eval_every):
    model.train()

    dataset_size = 0
    running_loss = 0.0
    running_auroc = 0.0

    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:
        images = data['image'].to(device, dtype=torch.float)
        targets = data['target'].to(device, dtype=torch.float)

        batch_size = images.size(0)

        outputs = model(images).squeeze()
        loss = weighted_bce_loss(outputs, targets)

        # zero the parameter gradients

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        auroc = binary_auroc(input=outputs.squeeze(), target=targets).item()

        running_loss += (loss.item() * batch_size)
        running_auroc += (auroc * batch_size)
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size
        epoch_auroc = running_auroc / dataset_size

        bar.set_postfix(Epoch=epoch, Train_Loss=epoch_loss, Train_Auroc=epoch_auroc, LR=optimizer.param_groups[0]['lr'])

    gc.collect()

    return epoch_loss, epoch_auroc


def manual_roc_curve(y_true, y_scores):
    """
    Manually calculate the ROC curve.

    Args:
        y_true (np.array): True binary labels (0 or 1).
        y_scores (np.array): Predicted scores.

    Returns:
        fpr (np.array): False Positive Rate.
        tpr (np.array): True Positive Rate.
        thresholds (np.array): Thresholds used for calculating FPR and TPR.
    """
    # Sort scores and corresponding true labels in descending order
    desc_score_indices = np.argsort(-y_scores)
    y_scores = y_scores[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # Initialize variables to calculate TPR and FPR
    tpr = []
    fpr = []
    thresholds = []

    P = sum(y_true)
    N = len(y_true) - P

    tp = 0
    fp = 0

    for i in range(len(y_scores)):
        if y_true[i] == 1:
            tp += 1
        else:
            fp += 1

        # Calculate TPR and FPR at this threshold
        tpr.append(tp / P)
        fpr.append(fp / N)
        thresholds.append(y_scores[i])

    return np.array(fpr), np.array(tpr), np.array(thresholds)


# @torch.inference_mode()
# def valid_one_epoch(model, dataloader, device, epoch, optimizer):
#     model.eval()
#
#     dataset_size = 0
#     running_loss = 0.0
#     running_auroc = 0.0
#
#     bar = tqdm(enumerate(dataloader), total=len(dataloader))
#     for step, data in bar:
#         images = data['image'].to(device, dtype=torch.float)
#         targets = data['target'].to(device, dtype=torch.float)
#
#         batch_size = images.size(0)
#
#         outputs = model(images).squeeze()
#         loss = criterion(outputs, targets)
#
#         auroc = binary_auroc(input=outputs.squeeze(), target=targets).item()
#         running_loss += (loss.item() * batch_size)
#         running_auroc += (auroc * batch_size)
#         dataset_size += batch_size
#
#         epoch_loss = running_loss / dataset_size
#         epoch_auroc = running_auroc / dataset_size
#
#         bar.set_postfix(Epoch=epoch, Valid_Loss=epoch_loss, Valid_Auroc=epoch_auroc,
#                         LR=optimizer.param_groups[0]['lr'])
#
#     gc.collect()
#
#     return epoch_loss, epoch_auroc


@torch.inference_mode()
def valid_pAUC_one_epoch(model, valid_loader, device=CONFIG['device'],
                                                          epoch=None, optimizer=None, criterion=None):
    model.eval()

    dataset_size = 0
    running_loss = 0.0

    bar = tqdm(enumerate(valid_loader), total=len(valid_loader))
    all_preds = []
    all_targets = []
    for step, data in bar:
        images = data['image'].float().cuda()
        targets = data['target'].float().cuda()
        images_2 = data['image_2'].float().cuda()
        images_3 = data['image_3'].float().cuda()
        images_4 = data['image_4'].float().cuda()

        outputs_a, _ = model(images)
        outputs_b, _ = model(images_2)
        outputs_c, _ = model(images_3)
        outputs_d, _ = model(images_4)

        outputs = (outputs_a + outputs_b + outputs_c + outputs_d) / 4

        loss = bce_loss(outputs.squeeze(), targets)

        running_loss += loss.item()

        epoch_loss = running_loss / (step + 1)

        all_preds.append(outputs.cpu().detach().numpy())
        all_targets.append(targets.cpu().detach().numpy())

        bar.set_postfix(Epoch=epoch, Valid_Loss=epoch_loss)

    concat_preds = np.concatenate(all_preds)
    concat_targets = np.concatenate(all_targets)

    #pAUC_loss = binary_auroc(input=torch.tensor(concat_preds), target=torch.tensor(concat_targets))
    pAUC_loss = score(concat_targets, concat_preds)
    print(f"pAUC: {pAUC_loss}")

    gc.collect()

    return epoch_loss, pAUC_loss


def run_training(model, optimizer, scheduler, num_epochs, train_loader, valid_loader, CONFIG, fold):
    if torch.cuda.is_available():
        print("[INFO] Using GPU: {}\n".format(torch.cuda.get_device_name()))

    start = time.time()
    best_epoch_auroc = -np.inf
    history = defaultdict(list)
    last_epoch_saved = 0

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    PATH = f"models/pretrain_vcreg_params_{params}_b_{CONFIG['train_batch_size']}_dp_{CONFIG['dropout']}_fold_{fold}.bin"

    for epoch in range(1, num_epochs + 1):
        model.train()

        dataset_size = 0
        running_loss = 0.0
        running_vicreg = 0.0
        iters = 0

        for batch_idx, data in enumerate(train_loader):
            targets = data['target'].float().cuda()

            images = data['image'].cuda()
            images_2 = data['image_2'].cuda()

            outputs_a, z_a = model(images)
            outputs_b, z_b = model(images_2)

            outputs = (outputs_a + outputs_b) / 2

            bce_loss_batch = bce_loss(outputs.squeeze(), targets)
            vicreg_batch = VICReg(z_a, z_b)

            loss = 75 * bce_loss_batch + vicreg_batch

            # zero the parameter gradients

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            running_loss += bce_loss_batch.item()
            running_vicreg += vicreg_batch.item()
            iters += 1

            if batch_idx % CONFIG['eval_every'] == 0:

                model.eval()
                val_epoch_loss, val_epoch_auroc = valid_pAUC_one_epoch(model, valid_loader, device=CONFIG['device'],
                                                                  epoch=epoch, optimizer=optimizer, criterion=bce_loss)

                # deep copy the model
                if best_epoch_auroc <= val_epoch_auroc:
                    print(f"Validation AUROC Improved ({best_epoch_auroc} ---> {val_epoch_auroc})")
                    best_epoch_auroc = val_epoch_auroc
                    torch.save(model.state_dict(), PATH)
                    last_epoch_saved = epoch
                    # Save a model file from the current directory
                    print(f"Model Saved", "fold:", fold, "epoch:", epoch, "lr:", optimizer.param_groups[0]['lr'], "train_loss:", running_loss / iters, "vicreg:", running_vicreg / iters)
                else:
                    print(f"last, {last_epoch_saved}, best pAUC {best_epoch_auroc}, fold:", fold, "epoch:", epoch, "lr:", optimizer.param_groups[0]['lr'], "train_loss:", running_loss / iters, "vicreg:", running_vicreg / iters)

                model.train()
                running_loss = 0.0
                running_vicreg = 0.0
                iters = 0
                print()

    end = time.time()
    time_elapsed = end - start
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
    print("Best AUROC: {:.4f}".format(best_epoch_auroc))

    # load best model weights
    model.load_state_dict(torch.load(PATH))

    return model, history


def prepare_loaders(df, fold, data_transforms, CONFIG):
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    df_valid_positive = df_valid[df_valid.target == 1].reset_index(drop=True)
    df_valid_negative = df_valid[df_valid.target == 0].reset_index(drop=True)

    df_valid_negative = df_valid_negative.sample(df_valid_positive.shape[0] * CONFIG['positive_ratio_valid'])
    df_valid = pd.concat([df_valid_positive, df_valid_negative]).reset_index(drop=True)

    train_dataset = ISICDataset_for_Train(df_train, CONFIG, transforms=data_transforms["train"])
    valid_dataset = ISICDataset(df_valid, transforms=data_transforms["valid"])

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['train_batch_size'],
                              shuffle=True, pin_memory=True, drop_last=True)

    valid_loader = DataLoader(valid_dataset, batch_size=CONFIG['valid_batch_size'],
                              shuffle=False, pin_memory=True)

    return train_loader, valid_loader


def fetch_scheduler(optimizer):
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['T_max'], eta_min=CONFIG['min_lr'])
   # scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=CONFIG['T_0'], eta_min=CONFIG['min_lr'])

    return scheduler


def downsample(val_data, positive_ratio=1):
    df_valid_positive = val_data[val_data.target == 1].reset_index(drop=True)
    df_valid_negative = val_data[val_data.target == 0].reset_index(drop=True)

    df_valid_negative = df_valid_negative.sample(df_valid_positive.shape[0] * positive_ratio)
    val_data = pd.concat([df_valid_positive, df_valid_negative]).reset_index(drop=True)

    return val_data


def RankBoostAUC(y_pred, y_true):
    y_pred_diff = y_pred.unsqueeze(1) - y_pred.unsqueeze(0)
    y_true_diff = y_true.unsqueeze(1) - y_true.unsqueeze(0)
    loss_matrix = 0.5 * (1 - torch.sign(y_true_diff) * torch.tanh(y_pred_diff))
    loss = loss_matrix.mean()
    return loss


def SmoothAUC(y_pred, y_true, smooth_factor=10):
    y_pred_diff = y_pred.unsqueeze(1) - y_pred.unsqueeze(0)
    y_true_diff = y_true.unsqueeze(1) - y_true.unsqueeze(0)
    smooth_approx = torch.sigmoid(smooth_factor * y_pred_diff) * (y_true_diff > 0).float()
    loss = smooth_approx.mean()
    return -loss


def Smooth_pAUC(y_pred, y_true, smooth_factor=10, tpr_threshold=0.8):
    sorted_indices = torch.argsort(y_pred, descending=True)
    y_pred_sorted = y_pred[sorted_indices]
    y_true_sorted = y_true[sorted_indices]

    # Calculate TPR and FPR
    tpr = torch.cumsum(y_true_sorted, dim=0) / y_true_sorted.sum()

    # Mask to consider only points above the TPR threshold
    mask = tpr >= tpr_threshold
    if mask.sum() == 0:
        return torch.tensor(0.0, requires_grad=True)

    # Apply mask to the sorted predictions and true labels
    y_pred_filtered = y_pred_sorted[mask]
    y_true_filtered = y_true_sorted[mask]

    # Compute pairwise differences for partial AUC
    y_pred_diff = y_pred_filtered.unsqueeze(1) - y_pred_filtered.unsqueeze(0)
    y_true_diff = y_true_filtered.unsqueeze(1) - y_true_filtered.unsqueeze(0)

    # Smooth approximation for partial AUC
    smooth_approx = torch.sigmoid(smooth_factor * y_pred_diff) * (y_true_diff > 0).float()

    # Calculate the mean of the smooth approximation
    partial_auc = smooth_approx.mean()

    # We minimize negative pAUC to maximize it
    loss = -partial_auc

    return loss




