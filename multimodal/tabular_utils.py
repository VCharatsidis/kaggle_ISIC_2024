import numpy as np
import pandas as pd
import pandas.api.types
import matplotlib.pyplot as plt
import torch

from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.ensemble import VotingClassifier

import catboost as cb
import lightgbm as lgb
import xgboost as xgb
from tqdm.auto import tqdm
import gc

from pAUC import score
from p_baseline_utils import bce_loss, VICReg


def feature_engineering(df):
    # New features to try...
    df["lesion_size_ratio"] = df["tbp_lv_minorAxisMM"] / df["clin_size_long_diam_mm"]
    df["lesion_shape_index"] = df["tbp_lv_areaMM2"] / (df["tbp_lv_perimeterMM"] ** 2)
    df["hue_contrast"] = (df["tbp_lv_H"] - df["tbp_lv_Hext"]).abs()
    df["luminance_contrast"] = (df["tbp_lv_L"] - df["tbp_lv_Lext"]).abs()
    df["lesion_color_difference"] = np.sqrt(
        df["tbp_lv_deltaA"] ** 2 + df["tbp_lv_deltaB"] ** 2 + df["tbp_lv_deltaL"] ** 2)
    df["border_complexity"] = df["tbp_lv_norm_border"] + df["tbp_lv_symm_2axis"]
    df["color_uniformity"] = df["tbp_lv_color_std_mean"] / (df["tbp_lv_radial_color_std_max"] + 1e-8)
    df["3d_position_distance"] = np.sqrt(df["tbp_lv_x"] ** 2 + df["tbp_lv_y"] ** 2 + df["tbp_lv_z"] ** 2)
    df["perimeter_to_area_ratio"] = df["tbp_lv_perimeterMM"] / df["tbp_lv_areaMM2"]
    df["lesion_visibility_score"] = df["tbp_lv_deltaLBnorm"] + df["tbp_lv_norm_color"]

    df["symmetry_border_consistency"] = df["tbp_lv_symm_2axis"] * df["tbp_lv_norm_border"]
    df["color_consistency"] = df["tbp_lv_stdL"] / df["tbp_lv_Lext"]

    df["size_age_interaction"] = df["clin_size_long_diam_mm"] * df["age_approx"]
    df["hue_color_std_interaction"] = df["tbp_lv_H"] * df["tbp_lv_color_std_mean"]
    df["lesion_severity_index"] = (df["tbp_lv_norm_border"] + df["tbp_lv_norm_color"] + df["tbp_lv_eccentricity"]) / 3
    df["shape_complexity_index"] = df["border_complexity"] + df["lesion_shape_index"]
    df["color_contrast_index"] = df["tbp_lv_deltaA"] + df["tbp_lv_deltaB"] + df["tbp_lv_deltaL"] + df[
        "tbp_lv_deltaLBnorm"]
    df["log_lesion_area"] = np.log(df["tbp_lv_areaMM2"] + 1)
    df["normalized_lesion_size"] = df["clin_size_long_diam_mm"] / df["age_approx"]
    df["mean_hue_difference"] = (df["tbp_lv_H"] + df["tbp_lv_Hext"]) / 2
    df["std_dev_contrast"] = np.sqrt(
        (df["tbp_lv_deltaA"] ** 2 + df["tbp_lv_deltaB"] ** 2 + df["tbp_lv_deltaL"] ** 2) / 3)
    df["color_shape_composite_index"] = (df["tbp_lv_color_std_mean"] + df["tbp_lv_area_perim_ratio"] + df[
        "tbp_lv_symm_2axis"]) / 3
    df["3d_lesion_orientation"] = np.arctan2(df["tbp_lv_y"], df["tbp_lv_x"])
    df["overall_color_difference"] = (df["tbp_lv_deltaA"] + df["tbp_lv_deltaB"] + df["tbp_lv_deltaL"]) / 3
    df["symmetry_perimeter_interaction"] = df["tbp_lv_symm_2axis"] * df["tbp_lv_perimeterMM"]
    df["comprehensive_lesion_index"] = (df["tbp_lv_area_perim_ratio"] + df["tbp_lv_eccentricity"] + df[
        "tbp_lv_norm_color"] + df["tbp_lv_symm_2axis"]) / 4

    # Taken from: https://www.kaggle.com/code/dschettler8845/isic-detect-skin-cancer-let-s-learn-together
    df["color_variance_ratio"] = df["tbp_lv_color_std_mean"] / df["tbp_lv_stdLExt"]
    df["border_color_interaction"] = df["tbp_lv_norm_border"] * df["tbp_lv_norm_color"]
    df["size_color_contrast_ratio"] = df["clin_size_long_diam_mm"] / df["tbp_lv_deltaLBnorm"]
    df["age_normalized_nevi_confidence"] = df["tbp_lv_nevi_confidence"] / df["age_approx"]
    df["color_asymmetry_index"] = df["tbp_lv_radial_color_std_max"] * df["tbp_lv_symm_2axis"]
    df["3d_volume_approximation"] = df["tbp_lv_areaMM2"] * np.sqrt(
        df["tbp_lv_x"] ** 2 + df["tbp_lv_y"] ** 2 + df["tbp_lv_z"] ** 2)
    df["color_range"] = (df["tbp_lv_L"] - df["tbp_lv_Lext"]).abs() + (df["tbp_lv_A"] - df["tbp_lv_Aext"]).abs() + (
                df["tbp_lv_B"] - df["tbp_lv_Bext"]).abs()
    df["shape_color_consistency"] = df["tbp_lv_eccentricity"] * df["tbp_lv_color_std_mean"]
    df["border_length_ratio"] = df["tbp_lv_perimeterMM"] / (2 * np.pi * np.sqrt(df["tbp_lv_areaMM2"] / np.pi))
    df["age_size_symmetry_index"] = df["age_approx"] * df["clin_size_long_diam_mm"] * df["tbp_lv_symm_2axis"]
    # Until here.

    #df['count_per_patient'] = df.groupby('patient_id')['isic_id'].transform('count')

    new_num_cols = [#'count_per_patient',
        "lesion_size_ratio", "lesion_shape_index", "hue_contrast",
        "luminance_contrast", "lesion_color_difference", "border_complexity",
        "color_uniformity", "3d_position_distance", "perimeter_to_area_ratio",
        "lesion_visibility_score", "symmetry_border_consistency", "color_consistency",

        "size_age_interaction", "hue_color_std_interaction", "lesion_severity_index",
        "shape_complexity_index", "color_contrast_index", "log_lesion_area",
        "normalized_lesion_size", "mean_hue_difference", "std_dev_contrast",
        "color_shape_composite_index", "3d_lesion_orientation", "overall_color_difference",
        "symmetry_perimeter_interaction", "comprehensive_lesion_index",

        "color_variance_ratio", "border_color_interaction", "size_color_contrast_ratio",
        "age_normalized_nevi_confidence", "color_asymmetry_index", "3d_volume_approximation",
        "color_range", "shape_color_consistency", "border_length_ratio", "age_size_symmetry_index",
    ]

    return df, new_num_cols


def comp_score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str, min_tpr: float=0.80):
    v_gt = abs(np.asarray(solution.values)-1)
    v_pred = np.array([1.0 - x for x in submission.values])
    max_fpr = abs(1-min_tpr)
    partial_auc_scaled = roc_auc_score(v_gt, v_pred, max_fpr=max_fpr)
    # change scale from [0.5, 1.0] to [0.5 * max_fpr**2, max_fpr]
    # https://math.stackexchange.com/questions/914823/shift-numbers-into-a-different-range
    partial_auc = 0.5 * max_fpr**2 + (max_fpr - 0.5 * max_fpr**2) / (1.0 - 0.5) * (partial_auc_scaled - 0.5)
    return partial_auc


def custom_lgbm_metric(y_true, y_hat):
    # TODO: Refactor with the above.
    min_tpr = 0.80
    v_gt = abs(y_true-1)
    v_pred = np.array([1.0 - x for x in y_hat])
    max_fpr = abs(1-min_tpr)
    partial_auc_scaled = roc_auc_score(v_gt, v_pred, max_fpr=max_fpr)
    # change scale from [0.5, 1.0] to [0.5 * max_fpr**2, max_fpr]
    # https://math.stackexchange.com/questions/914823/shift-numbers-into-a-different-range
    partial_auc = 0.5 * max_fpr**2 + (max_fpr - 0.5 * max_fpr**2) / (1.0 - 0.5) * (partial_auc_scaled - 0.5)
    return "pauc80", partial_auc, True

@torch.inference_mode()
def valid_pAUC_tabular(model, valid_loader,  epoch=None):
    model.eval()

    dataset_size = 0
    running_loss = 0.0

    bar = tqdm(enumerate(valid_loader), total=len(valid_loader))
    all_preds = []
    all_targets = []
    for step, data in bar:
        images_1 = data['images_1'].float().cuda()
        images_2 = data['images_2'].float().cuda()
        images_3 = data['images_3'].float().cuda()
        images_4 = data['images_4'].float().cuda()

        tab_data = data['input_data'].float().cuda()
        targets = data['target'].float().cuda()

        outputs_1, _, _, _ = model(tab_data, images_1)
        outputs_2, _, _, _ = model(tab_data, images_2)
        outputs_3, _, _, _ = model(tab_data, images_3)
        outputs_4, _, _, _ = model(tab_data, images_4)

        outputs = (outputs_1 + outputs_2 + outputs_3 + outputs_4) / 4

        loss = bce_loss(outputs.squeeze(), targets)

        batch_size = images_1.size(0)
        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size

        all_preds.append(outputs.cpu().detach().numpy())
        all_targets.append(targets.cpu().detach().numpy())

        bar.set_postfix(Epoch=epoch, Valid_Loss=epoch_loss)

    # Step 1: Count the number of values over 0.8
    preds = np.concatenate(all_preds)

    count_over_08 = np.sum(preds > 0.8)
    count_over_05 = np.sum(preds > 0.5)

    # Step 2: Calculate the total number of values in the array
    total_count = preds.shape[0]

    # Step 3: Calculate the percentage
    percentage_over_08 = round((count_over_08 / total_count), 4) * 100
    percentage_over_05 = round((count_over_05 / total_count), 4) * 100

    print(f"Percentage of values over 0.8: {percentage_over_08}%", f"Percentage of values over 0.5: {percentage_over_05}%")
    pAUC_loss = score(np.concatenate(all_targets), preds)
    print(f"pAUC: {pAUC_loss}")

    return epoch_loss, pAUC_loss


@torch.inference_mode()
def valid_ssl(model, valid_loader,  epoch=None):
    model.eval()

    dataset_size = 0
    running_loss = 0.0

    bar = tqdm(enumerate(valid_loader), total=len(valid_loader))

    for step, data in bar:
        images_1 = data['images_1'].float().cuda()
        images_2 = data['images_2'].float().cuda()

        outputs_1 = model(images_1)
        outputs_2 = model(images_2)

        loss = VICReg(outputs_1, outputs_2)

        batch_size = images_1.size(0)
        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size

        bar.set_postfix(Epoch=epoch, Valid_Loss=epoch_loss)

    final_loss = running_loss / dataset_size

    return final_loss


@torch.inference_mode()
def valid_pAUC_only_tabular(model, valid_loader,  epoch=None):
    model.eval()

    dataset_size = 0
    running_loss = 0.0

    bar = tqdm(enumerate(valid_loader), total=len(valid_loader))
    all_preds = []
    all_targets = []
    for step, data in bar:
        tab_data = data['input_data'].float().cuda()
        targets = data['target'].float().cuda()

        outputs_1, _, _ = model(tab_data)

        loss = bce_loss(outputs_1.squeeze(), targets)

        batch_size = outputs_1.size(0)
        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size

        all_preds.append(outputs_1.cpu().detach().numpy())
        all_targets.append(targets.cpu().detach().numpy())

        bar.set_postfix(Epoch=epoch, Valid_Loss=epoch_loss)

    # Step 1: Count the number of values over 0.8
    preds = np.concatenate(all_preds)

    count_over_08 = np.sum(preds > 0.8)
    count_over_05 = np.sum(preds > 0.5)

    # Step 2: Calculate the total number of values in the array
    total_count = preds.shape[0]

    # Step 3: Calculate the percentage
    percentage_over_08 = round((count_over_08 / total_count), 4) * 100
    percentage_over_05 = round((count_over_05 / total_count), 4) * 100

    print(f"Percentage of values over 0.8: {percentage_over_08}%", f"Percentage of values over 0.5: {percentage_over_05}%")
    pAUC_loss = score(np.concatenate(all_targets), preds)
    print(f"pAUC: {pAUC_loss}")

    return epoch_loss, pAUC_loss


class ManualOneHotEncoder:
    def __init__(self, category_maps={}):
        self.category_maps = category_maps

    def get_new_cat_cols(self):
        new_cat_cols = [new_col for new_col, (col, val) in self.category_maps.items()]
        return new_cat_cols

    def create_category_maps(self, df, cat_cols):
        for col in cat_cols:
            unique_values = df[col].unique()
            print(unique_values)
            for uv in unique_values:
                col_name = f"{col}_{uv}"
                self.category_maps[col_name] = (col, uv)

        print(self.category_maps)

    def transform(self, df):
        for new_col, (col, val) in self.category_maps.items():
            df[new_col] = (df[col] == val).astype(int)

        columns_to_drop = {col for col, _ in self.category_maps.values()}
        #df.drop(columns=columns_to_drop, inplace=True)

        return df

