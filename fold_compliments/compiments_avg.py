import time

import numpy as np
import torch

from p_baseline_constants import CONFIG, data_transforms
from p_baseline_utils import bce_loss, ISICDataset
from train_p_baseline_utils import run_training, valid_pAUC_one_epoch
from torch.utils.data import DataLoader


def run_training_compliments(df, fold, components, valid_loader):

    print("num components:", len(components))
    fold_auc = 0

    models = []
    model_num = 0
    for model, train_loader, optimizer, scheduler in components:
        model.train()

        model, history = run_training(model, optimizer, scheduler, CONFIG['epochs'], train_loader, valid_loader,
                                      CONFIG, fold)

        models.append(model)
        df_valid = df[df.kfold == fold].reset_index(drop=True)
        valid_dataset = ISICDataset(df_valid, transforms=data_transforms["valid"])
        valid_loader_big = DataLoader(valid_dataset, batch_size=CONFIG['valid_batch_size'], shuffle=False, pin_memory=True)
        val_epoch_loss, val_epoch_auroc = valid_pAUC_one_epoch(model, valid_loader_big, device=CONFIG['device'])

        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        val_epoch_auroc = round(val_epoch_auroc, 4)
        fold_auc += val_epoch_auroc
        PATH = f"fold_models/vcreg_{model_num}_eff_net_pso_{CONFIG['positive_sampling_ratio']}_po_{CONFIG['positive_ratio_train']}_params_{params}_b_{CONFIG['train_batch_size']}_dp_{CONFIG['dropout']}_fold_{fold}_pAUC_{val_epoch_auroc}.bin"
        torch.save(model.state_dict(), PATH)
        print("fold:", fold, "done!", "model_num:", model_num)
        model_num += 1

    print("fold:", fold, "cv_auc:", fold_auc / len(components))
    return fold_auc, models

