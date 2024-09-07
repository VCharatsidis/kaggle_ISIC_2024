
import torch.nn.functional as F

from architectures.multimodal_soft_glu_mlp_late_concat import Multimodal_Soft_GLUMLP_late_concat


def epoch_update_gamma(y_true, y_pred, epoch=-1, delta=2):
    """
    Calculate gamma from last epoch's targets and predictions.
    Gamma is updated at the end of each epoch.
    y_true: `Tensor`. Targets (labels).  Float either 0.0 or 1.0 .
    y_pred: `Tensor` . Predictions.
    """
    sub_sample_size = 2000.0
    pos = y_pred[y_true == 1]
    neg = y_pred[y_true == 0]  # yo pytorch, no boolean tensors or operators?  Wassap?
    # subsample the training set for performance
    cap_pos = pos.shape[0]
    cap_neg = neg.shape[0]
    pos = pos[torch.rand_like(pos) < sub_sample_size / cap_pos]
    neg = neg[torch.rand_like(neg) < sub_sample_size / cap_neg]
    ln_pos = pos.shape[0]
    ln_neg = neg.shape[0]
    pos_expand = pos.view(-1, 1).expand(-1, ln_neg).reshape(-1)
    neg_expand = neg.repeat(ln_pos)
    diff = neg_expand - pos_expand
    Lp = diff[diff > 0]  # because we're taking positive diffs, we got pos and neg flipped.
    ln_Lp = Lp.shape[0] - 1
    diff_neg = -1.0 * diff[diff < 0]
    diff_neg = diff_neg.sort()[0]
    ln_neg = diff_neg.shape[0] - 1
    ln_neg = max([ln_neg, 0])
    left_wing = int(ln_Lp * delta)
    left_wing = max([0, left_wing])
    left_wing = min([ln_neg, left_wing])
    default_gamma = torch.tensor(0.2, dtype=torch.float).cuda()
    if diff_neg.shape[0] > 0:
        gamma = diff_neg[left_wing]
    else:
        gamma = default_gamma  # default=torch.tensor(0.2, dtype=torch.float).cuda() #zoink
    L1 = diff[diff > -1.0 * gamma]
    if epoch > -1:
        return gamma
    else:
        return default_gamma


def roc_star_loss(_y_true, y_pred, gamma, _epoch_true, epoch_pred):
    """
    Nearly direct loss function for AUC.
    See article,
    C. Reiss, "Roc-star : An objective function for ROC-AUC that actually works."
    https://github.com/iridiumblue/articles/blob/master/roc_star.md
        _y_true: `Tensor`. Targets (labels).  Float either 0.0 or 1.0 .
        y_pred: `Tensor` . Predictions.
        gamma  : `Float` Gamma, as derived from last epoch.
        _epoch_true: `Tensor`.  Targets (labels) from last epoch.
        epoch_pred : `Tensor`.  Predicions from last epoch.
    """

    # _y_true = _y_true[y_pred >= 0.8]
    # _epoch_true = _epoch_true[epoch_pred >= 0.8]
    # y_pred = y_pred[y_pred >= 0.8]
    # epoch_pred = epoch_pred[epoch_pred >= 0.8]

    # convert labels to boolean
    y_true = (_y_true >= 0.50)
    epoch_true = (_epoch_true >= 0.50)

    # if batch is either all true or false return small random stub value.
    if torch.sum(y_true) == 0 or torch.sum(y_true) == y_true.shape[0]: return torch.sum(y_pred) * 1e-8

    pos = y_pred[y_true]
    neg = y_pred[~y_true]

    epoch_pos = epoch_pred[epoch_true]
    epoch_neg = epoch_pred[~epoch_true]

    # Take random subsamples of the training set, both positive and negative.
    max_pos = 1000  # Max number of positive training samples
    max_neg = 1000  # Max number of negative training samples
    cap_pos = epoch_pos.shape[0]
    epoch_pos = epoch_pos[torch.rand_like(epoch_pos) < max_pos / cap_pos]
    epoch_neg = epoch_neg[torch.rand_like(epoch_neg) < max_neg / cap_pos]

    ln_pos = pos.shape[0]
    ln_neg = neg.shape[0]

    # sum positive batch elements against (subsampled) negative elements
    if ln_pos > 0:
        pos_expand = pos.view(-1, 1).expand(-1, epoch_neg.shape[0]).reshape(-1)
        neg_expand = epoch_neg.repeat(ln_pos)

        diff2 = neg_expand - pos_expand + gamma
        l2 = diff2[diff2 > 0]
        m2 = l2 * l2
    else:
        m2 = torch.tensor([0], dtype=torch.float).cuda()

    # Similarly, compare negative batch elements against (subsampled) positive elements
    if ln_neg > 0:
        pos_expand = epoch_pos.view(-1, 1).expand(-1, ln_neg).reshape(-1)
        neg_expand = neg.repeat(epoch_pos.shape[0])

        diff3 = neg_expand - pos_expand + gamma
        l3 = diff3[diff3 > 0]
        m3 = l3 * l3
    else:
        m3 = torch.tensor([0], dtype=torch.float).cuda()

    if (torch.sum(m2) + torch.sum(m3)) != 0:
        res2 = torch.sum(m2) / max_pos + torch.sum(m3) / max_neg
    else:
        res2 = torch.sum(m2) + torch.sum(m3)

    res2 = torch.where(torch.isnan(res2), torch.zeros_like(res2), res2)

    return res2


def train_model(model, x_train, train_loader, valid_loader, optimizer, n_epochs=5):

    print(len(x_train))

    for epoch in range(n_epochs):
        start_time = time.time()
        model.train()

        whole_y_pred = np.array([])
        whole_y_t = np.array([])

        for i, data in enumerate(train_loader):
            x_batch = data[:-1][0]
            y_batch = data[-1]

            y_pred = model(x_batch)

            if epoch > 0:
                if i == 0:
                    print('*Using Loss Roc-star')
                loss = roc_star_loss(y_batch, y_pred, epoch_gamma, last_whole_y_t, last_whole_y_pred)

            else:
                if i == 0:
                    print('*Using Loss BxE')
                loss = F.binary_cross_entropy(y_pred, 1.0 * y_batch)

            optimizer.zero_grad()
            loss.backward()
            # To prevent gradient explosions resulting in NaNs
            # https://discuss.pytorch.org/t/nan-loss-in-rnn-model/655/8
            # https://github.com/pytorch/examples/blob/master/word_language_model/main.py
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

            optimizer.step()

            whole_y_pred = np.append(whole_y_pred, y_pred.clone().detach().cpu().numpy())
            whole_y_t = np.append(whole_y_t, y_batch.clone().detach().cpu().numpy())

        model.eval()
        last_whole_y_t = torch.tensor(whole_y_t).cuda()
        last_whole_y_pred = torch.tensor(whole_y_pred).cuda()

        all_valid_preds = np.array([])
        all_valid_t = np.array([])
        for i, valid_data in enumerate(valid_loader):
            x_batch = valid_data[:-1]
            y_batch = valid_data[-1]

            y_pred = model(*x_batch).detach().cpu().numpy()
            y_t = y_batch.detach().cpu().numpy()

            all_valid_preds = np.concatenate([all_valid_preds, y_pred], axis=0)
            all_valid_t = np.concatenate([all_valid_t, y_t], axis=0)

        epoch_gamma = epoch_update_gamma(last_whole_y_t, last_whole_y_pred, epoch)

        elapsed_time = time.time() - start_time

        print()


import time

import numpy as np
import pandas as pd
import pandas.api.types
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.ensemble import VotingClassifier

import catboost as cb
import lightgbm as lgb
import xgboost as xgb
from tqdm.auto import tqdm
import gc

from ISIC_tabular_dataset import ISIC_tabular_dataset_for_Train, ISIC_tabular_dataset, \
    ISIC_multimodal_dataset_for_Train, ISIC_multimodal_dataset
from architectures.multimodal_glu_mlp import Multimodal_GLUMLP
from architectures.multimodal_soft_glu_mlp import Multimodal_Soft_GLUMLP
from architectures.multimodal_transformer import MultimodalEncoder
from architectures.transformer_encoder import SimpleEncoder
from example_utils import prepare_loaders
from pAUC import score
from p_baseline_constants import CONFIG, b_, TRAIN_DIR, ROOT_DIR, data_transforms
from p_baseline_utils import bce_loss, set_seed, get_train_file_path, VICReg, BCEWithPolarizationPenaltyLoss
from tabular_utils import feature_engineering, valid_pAUC_tabular, ManualOneHotEncoder
from torch.utils.data import Dataset, DataLoader
import glob

import sklearn
print("sklearn version:", sklearn.__version__)


df_train = pd.read_csv("../isic-2024-challenge/train-metadata.csv")
# df_test = pd.read_csv("../isic-2024-challenge/test-metadata.csv")

set_seed(CONFIG['seed'])


num_cols = [
    'age_approx', 'clin_size_long_diam_mm', 'tbp_lv_A', 'tbp_lv_Aext', 'tbp_lv_B', 'tbp_lv_Bext',
    'tbp_lv_C', 'tbp_lv_Cext', 'tbp_lv_H', 'tbp_lv_Hext', 'tbp_lv_L',
    'tbp_lv_Lext', 'tbp_lv_areaMM2', 'tbp_lv_area_perim_ratio', 'tbp_lv_color_std_mean',
    'tbp_lv_deltaA', 'tbp_lv_deltaB', 'tbp_lv_deltaL', 'tbp_lv_deltaLB',
    'tbp_lv_deltaLBnorm', 'tbp_lv_eccentricity', 'tbp_lv_minorAxisMM',
    'tbp_lv_nevi_confidence', 'tbp_lv_norm_border', 'tbp_lv_norm_color',
    'tbp_lv_perimeterMM', 'tbp_lv_radial_color_std_max', 'tbp_lv_stdL',
    'tbp_lv_stdLExt', 'tbp_lv_symm_2axis', 'tbp_lv_symm_2axis_angle',
    'tbp_lv_x', 'tbp_lv_y', 'tbp_lv_z',
]

df_train[num_cols] = df_train[num_cols].fillna(df_train[num_cols].median())
df_train, new_num_cols, new_cat_cols = feature_engineering(df_train.copy())
# df_test, _, _ = feature_engineering(df_test.copy())
num_cols += new_num_cols
# anatom_site_general
cat_cols = ["sex", "tbp_tile_type", "tbp_lv_location", "tbp_lv_location_simple"] + new_cat_cols
train_cols = num_cols

gc.collect()

# category_encoder = OrdinalEncoder(
#     categories='auto',
#     dtype=int,
#     handle_unknown='use_encoded_value',
#     unknown_value=-2,
#     encoded_missing_value=-1,
# )
#
# X_cat = category_encoder.fit_transform(df_train[cat_cols])
# for c, cat_col in enumerate(cat_cols):
#     df_train[cat_col] = X_cat[:, c]

from sklearn.preprocessing import OneHotEncoder
# Initialize the OneHotEncoder
encoder = ManualOneHotEncoder()

# Fit and transform the categorical columns
encoded_data = encoder.create_category_maps(df_train[cat_cols], cat_cols)
new_column_names = encoder.get_new_cat_cols()

df = encoder.transform(df_train)

print(list(df.columns.values))
print(df.shape)

columns_with_nan = df[train_cols].isna().any()
columns_with_nan = columns_with_nan[columns_with_nan].index.tolist()
print("Columns with NaN values:", columns_with_nan)

# Calculate the mean and standard deviation for the selected columns


EPS = 1e-8
# Normalize the selected columns


train_cols = train_cols + new_column_names


means = df[train_cols].mean()
stds = df[train_cols].std()
df[train_cols] = (df[train_cols] - means) / (stds + EPS)

print("train_cols:", train_cols)
print(len(train_cols))

print(df.columns.values)
print(df.shape)
print(len(new_column_names))


train_images_paths = sorted(glob.glob(f"{TRAIN_DIR}/*.jpg"))
train_images_paths = [path.replace('\\', '/') for path in train_images_paths]
df['file_path'] = df['isic_id'].apply(get_train_file_path)
df = df[df["file_path"].isin(train_images_paths)].reset_index(drop=True)

N_SPLITS = 5
gkf = GroupKFold(n_splits=N_SPLITS)

df["fold"] = -1
for idx, (train_idx, val_idx) in enumerate(gkf.split(df, df["target"], groups=df["patient_id"])):
    df.loc[val_idx, "fold"] = idx


def downsample(val_data):
    df_valid_positive = val_data[val_data.target == 1].reset_index(drop=True)
    df_valid_negative = val_data[val_data.target == 0].reset_index(drop=True)

    df_valid_negative = df_valid_negative.sample(df_valid_positive.shape[0] * CONFIG['positive_ratio_valid']).reset_index(drop=True)
    val_data = pd.concat([df_valid_positive, df_valid_negative]).reset_index(drop=True)

    return val_data


def add_percentage_noise(tensor, min_pct=0.0, max_pct=0.05):
    # Generate random percentages between min_pct and max_pct
    random_factors = torch.empty(tensor.size()).uniform_(min_pct, max_pct)
    # Generate random sign (-1 or 1) for each factor
    random_signs = torch.randint(0, 2, tensor.size()).float() * 2 - 1
    # Apply the random perturbations
    noise = tensor * random_factors.cuda() * random_signs.cuda()
    return tensor + noise


def train(df, means, stds):
    cv_auc = 0
    folds = 5

    means = torch.tensor(means.values, dtype=torch.float32).cuda()
    stds = torch.tensor(stds.values, dtype=torch.float32).cuda()

    reset_color = "\033[0m"
    for fold in range(folds):
        train_data = df[df["fold"] != fold].reset_index(drop=True)
        val_data = df[df["fold"] == fold].reset_index(drop=True)

        # X_train = train_data[train_cols].values
        # y_train = train_data["target"].values
        # X_val = val_data[train_cols].values
        # y_val = val_data["target"].values
        #
        # X_train = torch.tensor(X_train, dtype=torch.float32)
        # y_train = torch.tensor(y_train, dtype=torch.int8).unsqueeze(1)
        #
        # X_val = torch.tensor(X_val, dtype=torch.float32)
        # y_val = torch.tensor(y_val, dtype=torch.int8).unsqueeze(1)

        # train_data = downsample(train_data)
        # val_data = downsample(val_data)

        train_dataset = ISIC_multimodal_dataset_for_Train(train_data, pos_ratio=CONFIG['positive_ratio_train'],
                                                          train_cols=train_cols,
                                                          transforms=data_transforms['train'],
                                                          CONFIG=CONFIG)

        train_loader = DataLoader(train_dataset, batch_size=CONFIG['train_batch_size'],
                                  shuffle=True, pin_memory=True, drop_last=True)

        # valid

        df_valid_positive = val_data[val_data.target == 1].reset_index(drop=True)
        df_valid_negative = val_data[val_data.target == 0].reset_index(drop=True)

        df_valid_negative = df_valid_negative.sample(df_valid_positive.shape[0] * CONFIG['positive_ratio_valid'])
        df_valid = pd.concat([df_valid_positive, df_valid_negative]).reset_index(drop=True)

        val_dataset = ISIC_multimodal_dataset(df_valid, train_cols=train_cols, transforms=data_transforms['valid'])
        valid_loader = DataLoader(val_dataset, batch_size=CONFIG['valid_batch_size'],
                                  shuffle=False, pin_memory=True)

        seq_length = 245
        feature_dim = 1
        d_model = 64
        nhead = 1
        num_encoder_layers = 2
        dim_feedforward = 128
        output_dim = 1
        dropout = 0.1

        #model = MultimodalEncoder(seq_length, feature_dim, d_model, nhead, num_encoder_layers, dim_feedforward, output_dim, dropout).cuda()

        model = Multimodal_Soft_GLUMLP_late_concat(input_dim=125, hidden_dim=125, output_dim=1).cuda()
        print(
            f'The efficient_net has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters')

        print("num params:", sum(p.numel() for p in model.parameters() if p.requires_grad))
        print(f'The model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters')

        positive_num = train_loader.dataset.get_positive_num()
        print("positive examples:", positive_num)

        CONFIG['T_max'] = CONFIG['epochs'] * (
                    positive_num * CONFIG['positive_ratio_train'] // CONFIG['train_batch_size'])
        print('T_max', CONFIG['T_max'])

        optimizer = optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'])
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['T_max'], eta_min=CONFIG['min_lr'])

        start = time.time()
        best_epoch_auroc = -np.inf

        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        PATH = f"models/diff_roc_softmax_glu_{params}_b_{CONFIG['train_batch_size']}_dp_{CONFIG['dropout']}_fold_{fold}.bin"
        epoch_gamma = 0
        last_whole_y_t, last_whole_y_pred = 0, 0

        for epoch in range(1, CONFIG['epochs'] + 1):
            gc.collect()
            model.train()

            running_loss_bce = 0.0
            running_loss_vcreg = 0
            iters = 0

            whole_y_pred = np.array([])
            whole_y_t = np.array([])

            train_preds = []
            train_targets = []

            for batch_idx, data in enumerate(train_loader):
                tabular_data = data['input_data'].float().cuda()
                noisy_data = add_percentage_noise(tabular_data)

                targets = data['target'].float().cuda()
                images_1 = data['images_1'].float().cuda()
                images_2 = data['images_2'].float().cuda()

                outputs_a, z_a, i_a = model(tabular_data, images_1)
                outputs_b, z_b, i_b = model(noisy_data, images_2)

                outputs = (outputs_a + outputs_b) / 2
                # outputs = outputs_a
                train_preds.append(outputs.cpu().detach().numpy())
                train_targets.append(targets.cpu().detach().numpy())

                if epoch > 1:
                    bce_loss_batch = roc_star_loss(targets, outputs.squeeze(), epoch_gamma, last_whole_y_t, last_whole_y_pred)
                else:
                    bce_loss_batch = bce_loss(outputs.squeeze(), targets)

                vicreg_batch = VICReg(z_a, z_b)
                vicreg_image = VICReg(i_a, i_b)

                loss = 75 * bce_loss_batch + vicreg_batch + vicreg_image

                # zero the parameter gradients

                loss.backward()

                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

                whole_y_pred = np.append(whole_y_pred, outputs.clone().detach().cpu().numpy())
                whole_y_t = np.append(whole_y_t, targets.clone().detach().cpu().numpy())

                running_loss_bce += bce_loss_batch.item()
                running_loss_vcreg += vicreg_batch.item()
                iters += 1

                if batch_idx % CONFIG['eval_every'] == 0:

                    train_pAUC_loss = score(np.concatenate(train_targets), np.concatenate(train_preds))

                    model.eval()
                    val_epoch_loss, val_epoch_auroc = valid_pAUC_tabular(model, valid_loader, epoch=epoch)
                    print("train pAUC:", train_pAUC_loss)

                    # deep copy the model
                    if best_epoch_auroc <= val_epoch_auroc:
                        print(f"Validation AUROC Improved ({best_epoch_auroc} ---> {val_epoch_auroc})")
                        best_epoch_auroc = val_epoch_auroc
                        torch.save(model.state_dict(), PATH)
                        # Save a model file from the current directory
                        print(f"Model Saved", "fold:", fold, "epoch:", epoch, "lr:",
                              optimizer.param_groups[0]['lr'], "train_loss:", running_loss_bce / iters, "vcreg:",  running_loss_vcreg/iters,b_)

                    else:
                        print(f"fold:", fold, "epoch:", epoch, "lr:", optimizer.param_groups[0]['lr'],
                              "train_loss:", running_loss_bce / iters, running_loss_vcreg/iters, reset_color)

                    model.train()
                    running_loss_bce = 0.0
                    running_loss_vcreg = 0
                    iters = 0
                    print()

            last_whole_y_t = torch.tensor(whole_y_t).cuda()
            last_whole_y_pred = torch.tensor(whole_y_pred).cuda()
            epoch_gamma = epoch_update_gamma(last_whole_y_t, last_whole_y_pred, 1)

        val_data = df[df["fold"] == fold].reset_index(drop=True)
        val_dataset = ISIC_multimodal_dataset(val_data, train_cols=train_cols, transforms=data_transforms['valid'])

        valid_loader = DataLoader(val_dataset, batch_size=CONFIG['valid_batch_size'], shuffle=False, pin_memory=True)
        val_epoch_loss, val_epoch_auroc = valid_pAUC_tabular(model, valid_loader)

        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        val_epoch_auroc = round(val_epoch_auroc, 4)
        cv_auc += val_epoch_auroc
        PATH = f"cv_models/diff_roc_softmax_glu_ratio_{CONFIG['positive_ratio_train']}_psr_{CONFIG['positive_sampling_ratio']}_params_{params}_b_{CONFIG['train_batch_size']}_dp_{CONFIG['dropout']}_fold_{fold}_pAUC_{val_epoch_auroc}.bin"
        torch.save(model.state_dict(), PATH)
        print("fold:", fold, "done!")

    print("cv score:", cv_auc / CONFIG['n_fold'])


train(df, means, stds)





