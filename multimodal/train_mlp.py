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

from ISIC_tabular_dataset import ISIC_tabular_dataset_for_Train, ISIC_tabular_dataset
from architectures.glu_mlp import GLUMLP
from architectures.relu_mlp import TabularMLP
from architectures.transformer_encoder import SimpleEncoder
from example_utils import prepare_loaders
from p_baseline_constants import CONFIG, b_
from p_baseline_utils import bce_loss, set_seed
from tabular_utils import feature_engineering, valid_pAUC_tabular
from torch.utils.data import Dataset, DataLoader

df_train = pd.read_csv("../isic-2024-challenge/train-metadata.csv")
df_test = pd.read_csv("../isic-2024-challenge/test-metadata.csv")

print(df_test.columns.values)
print(df_test.shape)

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

df_test_cols = df_test.columns.values
for i in range(len(num_cols)):
    if num_cols[i] not in df_test_cols:
        print("Error col not in test:", num_cols[i])
        input()

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
encoder = OneHotEncoder(sparse_output=False, drop='first')

# Fit and transform the categorical columns
encoded_data = encoder.fit_transform(df_train[cat_cols])
new_column_names = encoder.get_feature_names_out(cat_cols)

import joblib
# Save the fitted encoder
joblib.dump(encoder, 'onehot_encoder.pkl')

# Create a DataFrame with the encoded columns
encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(cat_cols))

# Concatenate the encoded columns with the original DataFrame
df = pd.concat([df_train.drop(cat_cols, axis=1), encoded_df], axis=1)
print(list(df.columns.values))
print(df.shape)

columns_with_nan = df[train_cols].isna().any()
columns_with_nan = columns_with_nan[columns_with_nan].index.tolist()
print("Columns with NaN values:", columns_with_nan)

# Calculate the mean and standard deviation for the selected columns
means = df[train_cols].mean()
stds = df[train_cols].std()

EPS = 1e-8
# Normalize the selected columns
df[train_cols] = (df[train_cols] - means) / (stds + EPS)

train_cols = train_cols + new_column_names.tolist()


N_SPLITS = 5
gkf = GroupKFold(n_splits=N_SPLITS)

df["fold"] = -1
for idx, (train_idx, val_idx) in enumerate(gkf.split(df, df["iddx_4"], groups=df["patient_id"])):
    df.loc[val_idx, "fold"] = idx


def train(df):
    cv_auc = 0
    folds = 5

    reset_color = "\033[0m"
    for fold in range(folds):
        train_data = df[df["fold"] != fold]
        val_data = df[df["fold"] == fold]

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

        # train_data = downsample(train_data, 50)
        # val_data = downsample(val_data, 50)

        train_dataset = ISIC_tabular_dataset_for_Train(train_data, pos_ratio=CONFIG['positive_ratio_train'], train_cols=train_cols)
        val_dataset = ISIC_tabular_dataset(val_data, train_cols=train_cols)

        train_loader = DataLoader(train_dataset, batch_size=CONFIG['train_batch_size'],
                                  shuffle=True, pin_memory=True, drop_last=True)

        valid_loader = DataLoader(val_dataset, batch_size=CONFIG['valid_batch_size'],
                                  shuffle=False, pin_memory=True)

        input_dim = 120
        hidden_dim = [64, 32, 16]
        output_dim = 1
        num_layers = 5
        model = TabularMLP(input_dim, hidden_dim, output_dim).cuda()

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
        PATH = f"models/mlp_params_{params}_b_{CONFIG['train_batch_size']}_dp_{CONFIG['dropout']}_fold_{CONFIG['fold']}.bin"

        for epoch in range(1, CONFIG['epochs'] + 1):
            gc.collect()
            model.train()

            dataset_size = 0
            running_loss = 0.0

            for batch_idx, data in enumerate(train_loader):
                tabular_data = data['input_data'].float().cuda()
                targets = data['target'].float().cuda()

                batch_size = tabular_data.size(0)

                outputs = model(tabular_data).squeeze()

                loss = bce_loss(outputs, targets)

                # zero the parameter gradients

                loss.backward()

                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

                running_loss += (loss.item() * batch_size)
                dataset_size += batch_size

                if batch_idx % CONFIG['eval_every'] == 0:

                    model.eval()
                    val_epoch_loss, val_epoch_auroc = valid_pAUC_tabular(model, valid_loader, epoch=epoch)

                    # deep copy the model
                    if best_epoch_auroc <= val_epoch_auroc:
                        print(f"Validation AUROC Improved ({best_epoch_auroc} ---> {val_epoch_auroc})")
                        best_epoch_auroc = val_epoch_auroc
                        torch.save(model.state_dict(), PATH)
                        # Save a model file from the current directory
                        print(f"Model Saved", "fold:", fold, "epoch:", epoch, "lr:",
                              optimizer.param_groups[0]['lr'], "train_loss:", running_loss / dataset_size, b_)

                    else:
                        print(f"fold:", fold, "epoch:", epoch, "lr:", optimizer.param_groups[0]['lr'],
                              "train_loss:", running_loss / dataset_size, reset_color)

                    model.train()
                    running_loss = 0.0
                    print()

        val_data = df[df["fold"] == fold].reset_index(drop=True)
        val_dataset = ISIC_tabular_dataset(val_data, train_cols=train_cols)

        valid_loader = DataLoader(val_dataset, batch_size=CONFIG['valid_batch_size'], shuffle=False, pin_memory=True)
        val_epoch_loss, val_epoch_auroc = valid_pAUC_tabular(model, valid_loader, epoch=0, )

        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        val_epoch_auroc = round(val_epoch_auroc, 4)
        cv_auc += val_epoch_auroc
        PATH = f"cv_models/transformer_positive_ratio_{CONFIG['positive_ratio_train']}_params_{params}_b_{CONFIG['train_batch_size']}_dp_{CONFIG['dropout']}_fold_{CONFIG['fold']}_pAUC_{val_epoch_auroc}.bin"
        torch.save(model.state_dict(), PATH)
        print("fold:", CONFIG['fold'], "done!")

    print("cv score:", cv_auc / CONFIG['n_fold'])

train(df)




