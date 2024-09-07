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
from architectures.multimodal_soft_glu_mlp_late_concat import Multimodal_Soft_GLUMLP_late_concat
from architectures.multimodal_transformer import MultimodalEncoder
from architectures.only_tabular_mlp import GELU_MLP
from architectures.transformer_encoder import SimpleEncoder
from example_utils import prepare_loaders
from pAUC import score
from p_baseline_constants import CONFIG, b_, TRAIN_DIR, ROOT_DIR, data_transforms, to_exclude
from p_baseline_utils import bce_loss, set_seed, get_train_file_path, VICReg, BCEWithPolarizationPenaltyLoss, SAM
from tabular_utils import feature_engineering, valid_pAUC_tabular, ManualOneHotEncoder, valid_pAUC_only_tabular
from torch.utils.data import Dataset, DataLoader
import glob

import sklearn
print("sklearn version:", sklearn.__version__)


df_train = pd.read_csv("../isic-2024-challenge/train-metadata.csv")
# df_test = pd.read_csv("../isic-2024-challenge/test-metadata.csv")

df_train = df_train[~df_train['isic_id'].isin(to_exclude)]
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

print("original num_cols:", len(num_cols))
df_train[num_cols] = df_train[num_cols].fillna(df_train[num_cols].median())
df_train, new_num_cols = feature_engineering(df_train.copy())

num_cols += new_num_cols

cat_cols = ["sex", "tbp_tile_type", "tbp_lv_location", "tbp_lv_location_simple"]
df_train[cat_cols] = df_train[cat_cols].fillna('Unknown')
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
new_cat_cols = encoder.get_new_cat_cols()

df = encoder.transform(df_train)

print(list(df.columns.values))
print(df.shape)

columns_with_nan = df[num_cols].isna().any()
columns_with_nan = columns_with_nan[columns_with_nan].index.tolist()
print("Columns with NaN values:", columns_with_nan)

# Calculate the mean and standard deviation for the selected columns


EPS = 1e-8
# Normalize the selected columns

public_features = ['age_approx', 'clin_size_long_diam_mm', 'tbp_lv_A', 'tbp_lv_Aext', 'tbp_lv_B', 'tbp_lv_Bext', 'tbp_lv_C', 'tbp_lv_Cext', 'tbp_lv_H', 'tbp_lv_Hext', 'tbp_lv_L', 'tbp_lv_Lext', 'tbp_lv_areaMM2', 'tbp_lv_area_perim_ratio', 'tbp_lv_color_std_mean', 'tbp_lv_deltaA', 'tbp_lv_deltaB', 'tbp_lv_deltaL', 'tbp_lv_deltaLB', 'tbp_lv_deltaLBnorm', 'tbp_lv_eccentricity', 'tbp_lv_minorAxisMM', 'tbp_lv_nevi_confidence', 'tbp_lv_norm_border', 'tbp_lv_norm_color', 'tbp_lv_perimeterMM', 'tbp_lv_radial_color_std_max', 'tbp_lv_stdL', 'tbp_lv_stdLExt', 'tbp_lv_symm_2axis', 'tbp_lv_symm_2axis_angle', 'tbp_lv_x', 'tbp_lv_y', 'tbp_lv_z', 'lesion_size_ratio', 'lesion_shape_index', 'hue_contrast', 'luminance_contrast', 'lesion_color_difference', 'border_complexity', 'color_uniformity', 'position_distance_3d', 'perimeter_to_area_ratio', 'area_to_perimeter_ratio', 'lesion_visibility_score', 'symmetry_border_consistency', 'consistency_symmetry_border', 'color_consistency', 'consistency_color', 'size_age_interaction', 'hue_color_std_interaction', 'lesion_severity_index', 'shape_complexity_index', 'color_contrast_index', 'log_lesion_area', 'normalized_lesion_size', 'mean_hue_difference', 'std_dev_contrast', 'color_shape_composite_index', 'lesion_orientation_3d', 'overall_color_difference', 'symmetry_perimeter_interaction', 'comprehensive_lesion_index', 'color_variance_ratio', 'border_color_interaction', 'border_color_interaction_2', 'size_color_contrast_ratio', 'age_normalized_nevi_confidence', 'age_normalized_nevi_confidence_2', 'color_asymmetry_index', 'volume_approximation_3d', 'color_range', 'shape_color_consistency', 'border_length_ratio', 'age_size_symmetry_index', 'index_age_size_symmetry', 'age_approx_patient_norm', 'clin_size_long_diam_mm_patient_norm', 'tbp_lv_A_patient_norm', 'tbp_lv_Aext_patient_norm', 'tbp_lv_B_patient_norm', 'tbp_lv_Bext_patient_norm', 'tbp_lv_C_patient_norm', 'tbp_lv_Cext_patient_norm', 'tbp_lv_H_patient_norm', 'tbp_lv_Hext_patient_norm', 'tbp_lv_L_patient_norm', 'tbp_lv_Lext_patient_norm', 'tbp_lv_areaMM2_patient_norm', 'tbp_lv_area_perim_ratio_patient_norm', 'tbp_lv_color_std_mean_patient_norm', 'tbp_lv_deltaA_patient_norm', 'tbp_lv_deltaB_patient_norm', 'tbp_lv_deltaL_patient_norm', 'tbp_lv_deltaLB_patient_norm', 'tbp_lv_deltaLBnorm_patient_norm', 'tbp_lv_eccentricity_patient_norm', 'tbp_lv_minorAxisMM_patient_norm', 'tbp_lv_nevi_confidence_patient_norm', 'tbp_lv_norm_border_patient_norm', 'tbp_lv_norm_color_patient_norm', 'tbp_lv_perimeterMM_patient_norm', 'tbp_lv_radial_color_std_max_patient_norm', 'tbp_lv_stdL_patient_norm', 'tbp_lv_stdLExt_patient_norm', 'tbp_lv_symm_2axis_patient_norm', 'tbp_lv_symm_2axis_angle_patient_norm', 'tbp_lv_x_patient_norm', 'tbp_lv_y_patient_norm', 'tbp_lv_z_patient_norm', 'lesion_size_ratio_patient_norm', 'lesion_shape_index_patient_norm', 'hue_contrast_patient_norm', 'luminance_contrast_patient_norm', 'lesion_color_difference_patient_norm', 'border_complexity_patient_norm', 'color_uniformity_patient_norm', 'position_distance_3d_patient_norm', 'perimeter_to_area_ratio_patient_norm', 'area_to_perimeter_ratio_patient_norm', 'lesion_visibility_score_patient_norm', 'symmetry_border_consistency_patient_norm', 'consistency_symmetry_border_patient_norm', 'color_consistency_patient_norm', 'consistency_color_patient_norm', 'size_age_interaction_patient_norm', 'hue_color_std_interaction_patient_norm', 'lesion_severity_index_patient_norm', 'shape_complexity_index_patient_norm', 'color_contrast_index_patient_norm', 'log_lesion_area_patient_norm', 'normalized_lesion_size_patient_norm', 'mean_hue_difference_patient_norm', 'std_dev_contrast_patient_norm', 'color_shape_composite_index_patient_norm', 'lesion_orientation_3d_patient_norm', 'overall_color_difference_patient_norm', 'symmetry_perimeter_interaction_patient_norm', 'comprehensive_lesion_index_patient_norm', 'color_variance_ratio_patient_norm', 'border_color_interaction_patient_norm', 'border_color_interaction_2_patient_norm', 'size_color_contrast_ratio_patient_norm', 'age_normalized_nevi_confidence_patient_norm', 'age_normalized_nevi_confidence_2_patient_norm', 'color_asymmetry_index_patient_norm', 'volume_approximation_3d_patient_norm', 'color_range_patient_norm', 'shape_color_consistency_patient_norm', 'border_length_ratio_patient_norm', 'age_size_symmetry_index_patient_norm', 'index_age_size_symmetry_patient_norm', 'count_per_patient', 'onehot_0', 'onehot_1', 'onehot_2', 'onehot_3', 'onehot_4', 'onehot_5', 'onehot_6', 'onehot_7', 'onehot_8', 'onehot_9', 'onehot_10', 'onehot_11', 'onehot_12', 'onehot_13', 'onehot_14', 'onehot_15', 'onehot_16', 'onehot_17', 'onehot_18', 'onehot_19', 'onehot_20', 'onehot_21', 'onehot_22', 'onehot_23', 'onehot_24', 'onehot_25', 'onehot_26', 'onehot_27', 'onehot_28', 'onehot_29', 'onehot_30', 'onehot_31', 'onehot_32', 'onehot_33', 'onehot_34', 'onehot_35', 'onehot_36', 'onehot_37', 'onehot_38', 'onehot_39', 'onehot_40', 'onehot_41', 'onehot_42', 'onehot_43', 'onehot_44', 'onehot_45', 'onehot_46']
train_cols = num_cols + new_cat_cols

# for pub_f in public_features:
#     if pub_f not in train_cols:
#         print(pub_f)
#
# print("stop")
# input()

means = df[num_cols].mean()
stds = df[num_cols].std()

np.save('means.npy', means.values)
np.save('stds.npy', stds.values)

df[num_cols] = (df[num_cols] - means) / (stds + EPS)

print("train_cols:", train_cols)
print(len(train_cols))

print(df.columns.values)
print(df.shape)
print("new_cat_cols", len(new_cat_cols))


train_images_paths = sorted(glob.glob(f"{TRAIN_DIR}/*.jpg"))
train_images_paths = [path.replace('\\', '/') for path in train_images_paths]
df['file_path'] = df['isic_id'].apply(get_train_file_path)
df = df[df["file_path"].isin(train_images_paths)].reset_index(drop=True)

N_SPLITS = 5
gkf = GroupKFold(n_splits=N_SPLITS)

df["fold"] = -1
for idx, (train_idx, val_idx) in enumerate(gkf.split(df, df["target"], groups=df["patient_id"])):
    df.loc[val_idx, "fold"] = idx


df_1 = df[df["fold"] == 0].reset_index(drop=True)
df_1.to_csv("df_1.csv", index=False)


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


def train(df):
    cv_auc = 0

    reset_color = "\033[0m"
    for fold in range(N_SPLITS):
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

        ambiguous = df_valid_negative[df_valid_negative['iddx_full'] != 'Benign']
        df_diff = pd.concat([ambiguous, df_valid]).drop_duplicates(keep=False)
        df_valid = pd.concat([df_valid, df_diff]).reset_index(drop=True)

        val_dataset = ISIC_multimodal_dataset(df_valid, train_cols=train_cols, transforms=data_transforms['valid'])
        valid_loader = DataLoader(val_dataset, batch_size=CONFIG['valid_batch_size'],
                                  shuffle=False, pin_memory=True)

        seq_length = 125
        feature_dim = 1
        d_model = 64
        nhead = 1
        num_encoder_layers = 2
        dim_feedforward = 128
        output_dim = 1
        dropout = 0.1

        #model = MultimodalEncoder(seq_length, feature_dim, d_model, nhead, num_encoder_layers, dim_feedforward, output_dim, dropout).cuda()

        model = GELU_MLP(hidden_dim=104, output_dim=1).cuda()
        print(
            f'The efficient_net has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters')

        print("num params:", sum(p.numel() for p in model.parameters() if p.requires_grad))
        print(f'The model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters')

        positive_num = train_loader.dataset.get_positive_num()
        print("positive examples:", positive_num)

        CONFIG['T_max'] = CONFIG['epochs'] * (
                    positive_num * CONFIG['positive_ratio_train'] // CONFIG['train_batch_size'])
        print('T_max', CONFIG['T_max'])

        base_optimizer = optim.AdamW
        optimizer = SAM(model.parameters(), base_optimizer, lr=CONFIG['learning_rate'], adaptive=True, rho=2)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['T_max'], eta_min=CONFIG['min_lr'])

        start = time.time()
        best_epoch_auroc = -np.inf

        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        PATH = f"Only_tabular_batch_norm_SAM_adaptive_lr_{CONFIG['learning_rate']}_conf_loss_vcreg_just_GELU_{params}_b_{CONFIG['train_batch_size']}_dp_{CONFIG['dropout']}_fold_{fold}"

        for epoch in range(1, CONFIG['epochs'] + 1):
            gc.collect()
            model.train()

            running_loss_bce = 0.0
            running_loss_vcreg = 0
            iters = 0

            train_preds = []
            train_targets = []

            for batch_idx, data in enumerate(train_loader):
                tabular_data = data['input_data'].float().cuda()
                noisy_data = add_percentage_noise(tabular_data)

                targets = data['target'].float().cuda()

                outputs_a, loss_pred_a, z_a = model(tabular_data)
                outputs_b, loss_pred_b, z_b = model(noisy_data)

                mae_a = torch.abs(outputs_a.squeeze() - targets)
                mae_b = torch.abs(outputs_b.squeeze() - targets)

                loss_pred_loss_a = torch.abs(mae_a - loss_pred_a).mean()
                loss_pred_loss_b = torch.abs(mae_b - loss_pred_b).mean()

                loss_pred = (loss_pred_loss_a + loss_pred_loss_b) / 2

                outputs = (outputs_a + outputs_b) / 2
                # outputs = outputs_a
                train_preds.append(outputs.cpu().detach().numpy())
                train_targets.append(targets.cpu().detach().numpy())

                bce_loss_batch = bce_loss(outputs.squeeze(), targets)

                vicreg_batch = VICReg(z_a, z_b)

                loss = 75 * bce_loss_batch + 75 * loss_pred + vicreg_batch

                # zero the parameter gradients

                loss.backward()
                optimizer.first_step(zero_grad=True)

                # second forward-backward step

                outputs_a, loss_pred_a, z_a = model(tabular_data)
                outputs_b, loss_pred_b, z_b = model(noisy_data)

                mae_a = torch.abs(outputs_a.squeeze() - targets)
                mae_b = torch.abs(outputs_b.squeeze() - targets)

                loss_pred_loss_a = torch.abs(mae_a - loss_pred_a).mean()
                loss_pred_loss_b = torch.abs(mae_b - loss_pred_b).mean()

                loss_pred = (loss_pred_loss_a + loss_pred_loss_b) / 2

                outputs = (outputs_a + outputs_b) / 2
                # outputs = outputs_a
                train_preds.append(outputs.cpu().detach().numpy())
                train_targets.append(targets.cpu().detach().numpy())

                bce_loss_batch = bce_loss(outputs.squeeze(), targets)

                vicreg_batch = VICReg(z_a, z_b)

                loss = 75 * bce_loss_batch + 75 * loss_pred + vicreg_batch
                loss.backward()
                optimizer.second_step(zero_grad=True)

                optimizer.zero_grad()
                scheduler.step()

                running_loss_bce += bce_loss_batch.item()
                running_loss_vcreg += vicreg_batch.item()
                iters += 1

                if (batch_idx % CONFIG['eval_every'] == 0):# and (epoch > 1):

                    train_pAUC_loss = score(np.concatenate(train_targets), np.concatenate(train_preds))

                    model.eval()
                    val_epoch_loss, val_epoch_auroc = valid_pAUC_only_tabular(model, valid_loader, epoch=epoch)
                    print("train pAUC:", train_pAUC_loss)

                    # deep copy the model
                    if best_epoch_auroc <= val_epoch_auroc:
                        print(f"Validation AUROC Improved ({best_epoch_auroc} ---> {val_epoch_auroc})")
                        best_epoch_auroc = val_epoch_auroc
                        torch.save(model.state_dict(), f"models/{PATH}.bin")
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

        val_data = df[df["fold"] == fold].reset_index(drop=True)
        val_dataset = ISIC_multimodal_dataset(val_data, train_cols=train_cols, transforms=data_transforms['valid'])

        valid_loader = DataLoader(val_dataset, batch_size=CONFIG['valid_batch_size'], shuffle=False, pin_memory=True)
        val_epoch_loss, val_epoch_auroc = valid_pAUC_only_tabular(model, valid_loader)

        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        val_epoch_auroc = round(val_epoch_auroc, 4)
        cv_auc += val_epoch_auroc
        torch.save(model.state_dict(), f"cv_models/{PATH}_{val_epoch_auroc}.bin")

        print("fold:", fold, "done!")

    print("cv score:", cv_auc / CONFIG['n_fold'])
    print(PATH)
    with open('cv_scores.txt', 'a') as file:
        # Write the data to the file with a newline character at the end
        file.write(f"{PATH}_{cv_auc / CONFIG['n_fold']}\n")


train(df)




