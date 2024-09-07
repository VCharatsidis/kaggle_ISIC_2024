import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from optuna.samplers import TPESampler

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.model_selection import cross_val_score

from sklearn.metrics import roc_auc_score
from sklearn.ensemble import VotingClassifier

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline

import lightgbm as lgb
import catboost as cb
import xgboost as xgb

from sklearn.utils import resample

import optuna
from tqdm import tqdm

from ISIC_tabular_dataset import ISIC_multimodal_ssl_valid
from architectures.ssl_encoder import EfficientNet_pretrained, EfficientNet_pretrained_linear
from p_baseline_constants import TRAIN_DIR, CONFIG, data_transforms
from p_baseline_utils import get_train_file_path
from public_utils import custom_metric, preprocess, read_data
from tunning import lgb_objective, cb_objective, xgb_objective

import torch
import glob
from torch.utils.data import DataLoader

import warnings
warnings.filterwarnings("ignore")

root = Path('../isic-2024-challenge')

train_path = root / 'train-metadata.csv'
test_path = root / 'test-metadata.csv'
subm_path = root / 'sample_submission.csv'

id_col = 'isic_id'
target_col = 'target'
group_col = 'patient_id'

err = 1e-5
sampling_ratio = 0.01
seed = 6

num_cols = [
    'age_approx',                        # Approximate age of patient at time of imaging.
    'clin_size_long_diam_mm',            # Maximum diameter of the lesion (mm).+
    'tbp_lv_A',                          # A inside  lesion.+
    'tbp_lv_Aext',                       # A outside lesion.+
    'tbp_lv_B',                          # B inside  lesion.+
    'tbp_lv_Bext',                       # B outside lesion.+
    'tbp_lv_C',                          # Chroma inside  lesion.+
    'tbp_lv_Cext',                       # Chroma outside lesion.+
    'tbp_lv_H',                          # Hue inside the lesion; calculated as the angle of A* and B* in LAB* color space. Typical values range from 25 (red) to 75 (brown).+
    'tbp_lv_Hext',                       # Hue outside lesion.+
    'tbp_lv_L',                          # L inside lesion.+
    'tbp_lv_Lext',                       # L outside lesion.+
    'tbp_lv_areaMM2',                    # Area of lesion (mm^2).+
    'tbp_lv_area_perim_ratio',           # Border jaggedness, the ratio between lesions perimeter and area. Circular lesions will have low values; irregular shaped lesions will have higher values. Values range 0-10.+
    'tbp_lv_color_std_mean',             # Color irregularity, calculated as the variance of colors within the lesion's boundary.
    'tbp_lv_deltaA',                     # Average A contrast (inside vs. outside lesion).+
    'tbp_lv_deltaB',                     # Average B contrast (inside vs. outside lesion).+
    'tbp_lv_deltaL',                     # Average L contrast (inside vs. outside lesion).+
    'tbp_lv_deltaLB',                    #
    'tbp_lv_deltaLBnorm',                # Contrast between the lesion and its immediate surrounding skin. Low contrast lesions tend to be faintly visible such as freckles; high contrast lesions tend to be those with darker pigment. Calculated as the average delta LB of the lesion relative to its immediate background in LAB* color space. Typical values range from 5.5 to 25.+
    'tbp_lv_eccentricity',               # Eccentricity.+
    'tbp_lv_minorAxisMM',                # Smallest lesion diameter (mm).+
    'tbp_lv_nevi_confidence',            # Nevus confidence score (0-100 scale) is a convolutional neural network classifier estimated probability that the lesion is a nevus. The neural network was trained on approximately 57,000 lesions that were classified and labeled by a dermatologist.+,++
    'tbp_lv_norm_border',                # Border irregularity (0-10 scale); the normalized average of border jaggedness and asymmetry.+
    'tbp_lv_norm_color',                 # Color variation (0-10 scale); the normalized average of color asymmetry and color irregularity.+
    'tbp_lv_perimeterMM',                # Perimeter of lesion (mm).+
    'tbp_lv_radial_color_std_max',       # Color asymmetry, a measure of asymmetry of the spatial distribution of color within the lesion. This score is calculated by looking at the average standard deviation in LAB* color space within concentric rings originating from the lesion center. Values range 0-10.+
    'tbp_lv_stdL',                       # Standard deviation of L inside  lesion.+
    'tbp_lv_stdLExt',                    # Standard deviation of L outside lesion.+
    'tbp_lv_symm_2axis',                 # Border asymmetry; a measure of asymmetry of the lesion's contour about an axis perpendicular to the lesion's most symmetric axis. Lesions with two axes of symmetry will therefore have low scores (more symmetric), while lesions with only one or zero axes of symmetry will have higher scores (less symmetric). This score is calculated by comparing opposite halves of the lesion contour over many degrees of rotation. The angle where the halves are most similar identifies the principal axis of symmetry, while the second axis of symmetry is perpendicular to the principal axis. Border asymmetry is reported as the asymmetry value about this second axis. Values range 0-10.+
    'tbp_lv_symm_2axis_angle',           # Lesion border asymmetry angle.+
    'tbp_lv_x',                          # X-coordinate of the lesion on 3D TBP.+
    'tbp_lv_y',                          # Y-coordinate of the lesion on 3D TBP.+
    'tbp_lv_z',                          # Z-coordinate of the lesion on 3D TBP.+
]


new_num_cols = [
    'lesion_size_ratio',             # tbp_lv_minorAxisMM      / clin_size_long_diam_mm
    'lesion_shape_index',            # tbp_lv_areaMM2          / tbp_lv_perimeterMM **2
    'hue_contrast',                  # tbp_lv_H                - tbp_lv_Hext              abs
    'luminance_contrast',            # tbp_lv_L                - tbp_lv_Lext              abs
    'lesion_color_difference',       # tbp_lv_deltaA **2       + tbp_lv_deltaB **2 + tbp_lv_deltaL **2  sqrt
    'border_complexity',             # tbp_lv_norm_border      + tbp_lv_symm_2axis
    'color_uniformity',              # tbp_lv_color_std_mean   / tbp_lv_radial_color_std_max

    'position_distance_3d',          # tbp_lv_x **2 + tbp_lv_y **2 + tbp_lv_z **2  sqrt
    'perimeter_to_area_ratio',       # tbp_lv_perimeterMM      / tbp_lv_areaMM2
    'area_to_perimeter_ratio',       # tbp_lv_areaMM2          / tbp_lv_perimeterMM
    'lesion_visibility_score',       # tbp_lv_deltaLBnorm      + tbp_lv_norm_color
    'symmetry_border_consistency',   # tbp_lv_symm_2axis       * tbp_lv_norm_border
    'consistency_symmetry_border',   # tbp_lv_symm_2axis       * tbp_lv_norm_border / (tbp_lv_symm_2axis + tbp_lv_norm_border)

    'color_consistency',             # tbp_lv_stdL             / tbp_lv_Lext
    'consistency_color',             # tbp_lv_stdL*tbp_lv_Lext / tbp_lv_stdL + tbp_lv_Lext
    'size_age_interaction',          # clin_size_long_diam_mm  * age_approx
    'hue_color_std_interaction',     # tbp_lv_H                * tbp_lv_color_std_mean
    'lesion_severity_index',         # tbp_lv_norm_border      + tbp_lv_norm_color + tbp_lv_eccentricity / 3
    'shape_complexity_index',        # border_complexity       + lesion_shape_index
    'color_contrast_index',          # tbp_lv_deltaA + tbp_lv_deltaB + tbp_lv_deltaL + tbp_lv_deltaLBnorm

    'log_lesion_area',               # tbp_lv_areaMM2          + 1  np.log
    'normalized_lesion_size',        # clin_size_long_diam_mm  / age_approx
    'mean_hue_difference',           # tbp_lv_H                + tbp_lv_Hext    / 2
    'std_dev_contrast',              # tbp_lv_deltaA **2 + tbp_lv_deltaB **2 + tbp_lv_deltaL **2   / 3  np.sqrt
    'color_shape_composite_index',   # tbp_lv_color_std_mean   + bp_lv_area_perim_ratio + tbp_lv_symm_2axis   / 3
    'lesion_orientation_3d',         # tbp_lv_y                , tbp_lv_x  np.arctan2
    'overall_color_difference',      # tbp_lv_deltaA           + tbp_lv_deltaB + tbp_lv_deltaL   / 3

    'symmetry_perimeter_interaction',# tbp_lv_symm_2axis       * tbp_lv_perimeterMM
    'comprehensive_lesion_index',    # tbp_lv_area_perim_ratio + tbp_lv_eccentricity + bp_lv_norm_color + tbp_lv_symm_2axis   / 4
    'color_variance_ratio',          # tbp_lv_color_std_mean   / tbp_lv_stdLExt
    'border_color_interaction',      # tbp_lv_norm_border      * tbp_lv_norm_color
    'border_color_interaction_2',
    'size_color_contrast_ratio',     # clin_size_long_diam_mm  / tbp_lv_deltaLBnorm
    'age_normalized_nevi_confidence',# tbp_lv_nevi_confidence  / age_approx
    'age_normalized_nevi_confidence_2',
    'color_asymmetry_index',         # tbp_lv_symm_2axis       * tbp_lv_radial_color_std_max

    'volume_approximation_3d',       # tbp_lv_areaMM2          * sqrt(tbp_lv_x**2 + tbp_lv_y**2 + tbp_lv_z**2)
    'color_range',                   # abs(tbp_lv_L - tbp_lv_Lext) + abs(tbp_lv_A - tbp_lv_Aext) + abs(tbp_lv_B - tbp_lv_Bext)
    'shape_color_consistency',       # tbp_lv_eccentricity     * tbp_lv_color_std_mean
    'border_length_ratio',           # tbp_lv_perimeterMM      / pi * sqrt(tbp_lv_areaMM2 / pi)
    'age_size_symmetry_index',       # age_approx              * clin_size_long_diam_mm * tbp_lv_symm_2axis
    'index_age_size_symmetry',       # age_approx              * tbp_lv_areaMM2 * tbp_lv_symm_2axis
]


cat_cols = ['sex', 'anatom_site_general', 'tbp_tile_type', 'tbp_lv_location', 'tbp_lv_location_simple', 'attribution']
norm_cols = [f'{col}_patient_norm' for col in num_cols + new_num_cols]
min_max_cols = [f'{col}_patient_min_max' for col in num_cols + new_num_cols]
special_cols = ['count_per_patient']

pred_cols = [f'pred_{i}' for i in range(60)]

columns_to_normalize = num_cols + new_num_cols + special_cols + pred_cols

future_norm_cols = [f'{col}_future_norm' for col in columns_to_normalize]
past_norm_cols = [f'{col}_past_norm' for col in columns_to_normalize]
future_count_cols = ['count_future']
past_count_cols = ['count_past']

feature_cols = num_cols + new_num_cols + norm_cols + special_cols + min_max_cols
feature_cols += [f'{col}_patient_norm' for col in pred_cols]
feature_cols += [f'{col}_patient_min_max_norm' for col in pred_cols]

df_train = read_data(train_path, err, num_cols, cat_cols, new_num_cols)
df_test = read_data(test_path, err, num_cols, cat_cols, new_num_cols)
df_subm = pd.read_csv(subm_path, index_col=id_col)

df_train, df_test, new_cat_cols = preprocess(df_train, df_test, cat_cols)

feature_cols += new_cat_cols
feature_cols += pred_cols

feature_cols += future_norm_cols
feature_cols += past_norm_cols
feature_cols += future_count_cols
feature_cols += past_count_cols

feature_cols += [f'{col}_basel_norm' for col in columns_to_normalize]
# feature_cols += [f'{col}_xp_norm' for col in columns_to_normalize]

print("feature_cols", len(feature_cols))

# [I 2024-08-18 18:40:45,589] Trial 197 finished with value: 0.17817860453038978 and
# parameters: {'lambda_l1': 0.0011200671615842846, 'lambda_l2': 0.08147808383576136, 'learning_rate': 0.0436390574151692, 'max_depth': 8, 'num_leaves': 103, 'colsample_bytree': 0.836997981508387, 'colsample_bynode': 0.6766189266332212, 'bagging_fraction': 0.8274832451785119, 'bagging_freq': 1, 'min_data_in_leaf': 19, 'scale_pos_weight': 1.084998841082701}
# . Best is trial 197 with value: 0.17817860453038978

lgb_avoid = ['tbp_lv_norm_color_past_norm', 'color_asymmetry_index_future_norm', 'color_range_future_norm', 'tbp_lv_norm_border_past_norm', 'pred_2_future_norm', 'shape_complexity_index_past_norm', 'border_length_ratio_future_norm', 'pred_8_future_norm', 'shape_complexity_index', 'pred_7_future_norm', 'tbp_lv_radial_color_std_max_past_norm', 'border_complexity_patient_norm', 'log_lesion_area', 'tbp_lv_stdL_past_norm', 'tbp_lv_stdLExt_past_norm', 'hue_color_std_interaction', 'tbp_lv_deltaA_patient_norm', 'comprehensive_lesion_index', 'tbp_lv_Lext_patient_norm', 'pred_33_future_norm', 'pred_57_future_norm', 'pred_35_future_norm', 'tbp_lv_A_patient_norm', 'symmetry_border_consistency', 'pred_54_future_norm', 'pred_37_future_norm', 'tbp_lv_areaMM2', 'tbp_lv_eccentricity_patient_norm', 'pred_41_future_norm', 'tbp_lv_deltaLB_patient_norm', 'tbp_lv_area_perim_ratio_patient_norm', 'std_dev_contrast_past_norm', 'pred_57_past_norm', 'border_length_ratio', 'pred_31_future_norm', 'pred_14_future_norm', 'pred_23_future_norm', 'color_variance_ratio', 'pred_20_future_norm', 'tbp_lv_deltaB_past_norm', 'tbp_lv_deltaA_past_norm', 'lesion_visibility_score', 'tbp_lv_x_patient_norm', 'tbp_lv_areaMM2_past_norm', 'pred_30_future_norm', 'border_color_interaction_2', 'tbp_lv_C_past_norm', 'tbp_lv_symm_2axis_patient_norm', 'tbp_lv_stdL_patient_norm', 'tbp_lv_Bext_past_norm', 'perimeter_to_area_ratio_past_norm', 'pred_20_patient_min_max_norm', 'tbp_lv_color_std_mean_future_norm', 'consistency_color', 'color_shape_composite_index_patient_min_max', 'pred_45_patient_min_max_norm', 'tbp_lv_norm_color', 'pred_4', 'pred_2_patient_norm', 'border_length_ratio_patient_min_max', 'comprehensive_lesion_index_patient_min_max', 'pred_18', 'pred_19', 'pred_44_patient_min_max_norm', 'pred_54_patient_norm', 'border_color_interaction_2_future_norm', 'tbp_lv_nevi_confidence', 'log_lesion_area_patient_min_max', 'pred_30', 'size_age_interaction_patient_min_max', 'pred_34', 'pred_57_patient_norm', 'symmetry_border_consistency_patient_min_max', 'pred_9_patient_norm', 'pred_10_patient_norm', 'pred_45', 'pred_35_patient_norm', 'pred_43_patient_norm', 'pred_59_patient_min_max_norm', 'onehot_9', 'onehot_10', 'pred_22_past_norm', 'pred_37_patient_norm', 'tbp_lv_stdL', 'pred_36_patient_norm', 'onehot_20', 'pred_12_patient_norm', 'pred_32_patient_norm', 'pred_27_patient_norm', 'pred_45_patient_norm', 'pred_48_patient_norm', 'pred_23_patient_norm', 'pred_21_patient_norm', 'pred_53_patient_norm', 'pred_13_patient_norm', 'border_complexity_patient_min_max', 'tbp_lv_deltaLB', 'pred_31_past_norm', 'shape_complexity_index_future_norm', 'pred_33_patient_min_max_norm', 'symmetry_perimeter_interaction_patient_norm', 'color_variance_ratio_past_norm', 'std_dev_contrast_patient_norm', 'consistency_symmetry_border_future_norm', 'comprehensive_lesion_index_past_norm', 'symmetry_perimeter_interaction_past_norm', 'hue_color_std_interaction_future_norm', 'color_contrast_index_future_norm', 'pred_41_patient_min_max_norm', 'normalized_lesion_size_future_norm', 'pred_36_past_norm', 'pred_54_past_norm', 'consistency_color_patient_norm', 'pred_27_patient_min_max_norm', 'symmetry_perimeter_interaction_future_norm', 'color_consistency_patient_norm', 'tbp_lv_L', 'luminance_contrast_future_norm',
             'tbp_lv_symm_2axis_future_norm', 'tbp_lv_stdLExt_future_norm', 'size_color_contrast_ratio_past_norm', 'pred_47_past_norm', 'lesion_size_ratio_patient_min_max', 'tbp_lv_symm_2axis_angle_patient_min_max', 'tbp_lv_deltaL', 'clin_size_long_diam_mm_future_norm', 'pred_48_past_norm', 'pred_11_patient_min_max_norm', 'pred_23_patient_min_max_norm', 'tbp_lv_deltaA_future_norm', 'color_asymmetry_index_past_norm', 'luminance_contrast', 'tbp_lv_minorAxisMM_future_norm', 'tbp_lv_nevi_confidence_future_norm', 'border_complexity', 'tbp_lv_norm_color_future_norm', 'tbp_lv_radial_color_std_max_future_norm', 'onehot_5', 'lesion_color_difference_past_norm', 'luminance_contrast_past_norm', 'color_shape_composite_index_past_norm', 'pred_27_past_norm', 'pred_37_past_norm', 'normalized_lesion_size_past_norm', 'log_lesion_area_past_norm', 'pred_55_past_norm', 'border_color_interaction_2_past_norm', 'color_range_past_norm', 'shape_color_consistency_past_norm', 'consistency_color_past_norm', 'border_length_ratio_past_norm', 'consistency_symmetry_border_past_norm', 'pred_39_past_norm', 'count_per_patient_past_norm', 'pred_4_past_norm', 'area_to_perimeter_ratio_past_norm', 'pred_12_past_norm', 'pred_21_past_norm', 'color_consistency_past_norm', 'onehot_41', 'tbp_lv_y_past_norm', 'onehot_12', 'pred_31_patient_min_max_norm', 'pred_35_patient_min_max_norm', 'pred_56_patient_min_max_norm', 'onehot_1', 'onehot_2', 'onehot_3', 'onehot_4', 'onehot_6', 'onehot_7', 'onehot_8', 'onehot_13', 'pred_13_patient_min_max_norm', 'onehot_14', 'onehot_16', 'onehot_18', 'onehot_19',
             'onehot_21', 'onehot_22', 'onehot_23', 'onehot_24', 'onehot_25', 'onehot_26', 'pred_18_patient_min_max_norm', 'pred_41_patient_norm', 'tbp_lv_symm_2axis_past_norm', 'tbp_lv_deltaL_patient_min_max', 'border_color_interaction', 'age_approx_patient_norm', 'tbp_lv_areaMM2_patient_norm', 'tbp_lv_deltaL_patient_norm', 'tbp_lv_deltaLBnorm_patient_norm', 'luminance_contrast_patient_norm', 'shape_complexity_index_patient_norm', 'color_shape_composite_index_patient_norm', 'border_color_interaction_2_patient_norm', 'border_length_ratio_patient_norm', 'tbp_lv_deltaLB_patient_min_max', 'pred_30_patient_norm', 'tbp_lv_norm_border_patient_min_max', 'lesion_shape_index_patient_min_max', 'luminance_contrast_patient_min_max', 'color_uniformity_patient_min_max', 'lesion_visibility_score_patient_min_max', 'consistency_symmetry_border_patient_min_max', 'consistency_color_patient_min_max', 'shape_color_consistency_patient_min_max', 'pred_5_patient_norm', 'pred_8_patient_norm', 'onehot_27', 'onehot_28', 'onehot_29', 'pred_24_future_norm', 'log_lesion_area_future_norm', 'color_variance_ratio_future_norm', 'size_color_contrast_ratio_future_norm', 'age_normalized_nevi_confidence_future_norm', 'volume_approximation_3d_future_norm',
             'shape_color_consistency_future_norm', 'index_age_size_symmetry_future_norm', 'count_per_patient_future_norm', 'pred_11_future_norm', 'pred_21_future_norm', 'pred_27_future_norm', 'onehot_30', 'pred_32_future_norm', 'pred_43_future_norm', 'pred_44_future_norm', 'pred_53_future_norm', 'pred_55_future_norm', 'tbp_lv_A_past_norm', 'tbp_lv_B_past_norm', 'tbp_lv_area_perim_ratio_past_norm', 'tbp_lv_deltaL_past_norm', 'tbp_lv_minorAxisMM_past_norm', 'consistency_color_future_norm', 'color_consistency_future_norm', 'perimeter_to_area_ratio_future_norm', 'border_complexity_future_norm', 'onehot_31', 'onehot_32', 'onehot_33', 'onehot_34', 'onehot_35', 'onehot_36', 'onehot_37', 'onehot_38', 'onehot_39', 'onehot_42', 'onehot_43', 'onehot_44', 'pred_35', 'pred_54', 'tbp_lv_Bext_future_norm', 'tbp_lv_color_std_mean', 'tbp_lv_deltaL_future_norm', 'tbp_lv_eccentricity_future_norm', 'tbp_lv_perimeterMM_future_norm', 'tbp_lv_stdL_future_norm', 'lesion_color_difference_future_norm', 'onehot_17',
             'clin_size_long_diam_mm_patient_min_max', 'pred_0_past_norm', 'pred_42_past_norm', 'onehot_0',
             'tbp_lv_color_std_mean_patient_min_max', 'pred_38_past_norm', 'pred_39_patient_norm', 'pred_10_past_norm',
             'pred_19_patient_norm', 'pred_18_past_norm', 'pred_14_patient_norm', 'pred_20_past_norm',
             'pred_23_past_norm', 'tbp_lv_minorAxisMM_patient_min_max', 'volume_approximation_3d_patient_min_max',
             'border_color_interaction_2_patient_min_max', 'pred_25_patient_norm',
             'lesion_color_difference_patient_min_max', 'pred_32_patient_min_max_norm',
             'border_color_interaction_patient_norm', 'tbp_lv_symm_2axis_angle_patient_norm', 'pred_45_future_norm',
             'tbp_lv_norm_color_patient_norm', 'pred_0_future_norm', 'overall_color_difference_future_norm',
             'symmetry_border_consistency_future_norm', 'hue_color_std_interaction_patient_norm',
             'tbp_lv_deltaLB_future_norm', 'tbp_lv_Cext_past_norm', 'tbp_lv_B_future_norm', 'tbp_lv_Aext_future_norm',
             'tbp_lv_symm_2axis', 'pred_50', 'tbp_lv_eccentricity_past_norm', 'pred_48', 'pred_50_past_norm',
             'lesion_size_ratio', 'normalized_lesion_size_patient_norm', 'pred_9', 'pred_7', 'onehot_40',
             'hue_color_std_interaction_patient_min_max', 'pred_51_patient_min_max_norm',
             'color_contrast_index_past_norm', 'tbp_lv_areaMM2_patient_min_max', 'pred_38_future_norm',
             'age_size_symmetry_index_past_norm', 'shape_color_consistency', 'pred_25', 'pred_19_future_norm',
             'pred_4_future_norm', 'age_size_symmetry_index_future_norm', 'pred_49_future_norm', 'tbp_lv_z_future_norm',
             'tbp_lv_norm_border_future_norm', 'tbp_lv_deltaLB_past_norm', 'tbp_lv_perimeterMM_past_norm',
             'symmetry_border_consistency_past_norm', 'age_normalized_nevi_confidence_patient_min_max',
             'pred_38_patient_min_max_norm', 'pred_34_patient_min_max_norm', 'pred_8_patient_min_max_norm',
             'pred_55_patient_norm', 'tbp_lv_area_perim_ratio', 'pred_40_patient_norm', 'pred_8_past_norm',
             'pred_7_patient_norm', 'tbp_lv_perimeterMM_patient_norm', 'color_variance_ratio_patient_norm', 'pred_58_future_norm',
             'lesion_shape_index_patient_norm', 'tbp_lv_stdLExt_patient_norm', 'size_color_contrast_ratio_patient_norm', 'lesion_size_ratio_future_norm', 'color_shape_composite_index', 'tbp_lv_norm_border_patient_norm', 'pred_34_patient_norm', 'shape_complexity_index_patient_min_max', 'lesion_visibility_score_past_norm', 'pred_23', 'pred_3', 'pred_31_patient_norm', 'pred_50_patient_min_max_norm', 'mean_hue_difference_past_norm', 'tbp_lv_Aext_past_norm', 'pred_13_past_norm', 'pred_19_patient_min_max_norm', 'mean_hue_difference_future_norm', 'pred_14_patient_min_max_norm', 'overall_color_difference_past_norm', 'pred_4_patient_min_max_norm', 'index_age_size_symmetry_past_norm', 'pred_44_patient_norm', 'pred_32', 'pred_5_past_norm', 'age_normalized_nevi_confidence_patient_norm', 'pred_40_past_norm', 'hue_color_std_interaction_past_norm',
             'tbp_lv_stdL_patient_min_max', 'pred_5', 'std_dev_contrast_patient_min_max', 'pred_8',
             'pred_26_future_norm', 'symmetry_perimeter_interaction', 'consistency_symmetry_border_patient_norm',
             'pred_40_future_norm', 'tbp_lv_minorAxisMM_patient_norm', 'pred_56_future_norm', 'pred_25_future_norm',
             'pred_24_patient_norm', 'pred_30_past_norm', 'pred_24_past_norm', 'tbp_lv_areaMM2_future_norm',
             'tbp_lv_color_std_mean_past_norm', 'perimeter_to_area_ratio', 'pred_7_past_norm', 'tbp_lv_perimeterMM',
             'pred_15_future_norm', 'tbp_lv_Cext_patient_min_max', 'size_age_interaction_past_norm',
             'area_to_perimeter_ratio_future_norm', 'pred_0', 'pred_30_patient_min_max_norm',
             'tbp_lv_A_patient_min_max', 'pred_17_patient_min_max_norm',
             'pred_34_future_norm', 'pred_10_future_norm', 'perimeter_to_area_ratio_patient_norm',
             'color_range_patient_min_max', 'tbp_lv_x_future_norm', 'pred_44', 'age_size_symmetry_index_patient_norm',
             'tbp_lv_Hext_patient_norm', 'pred_4_patient_norm', 'lesion_shape_index_past_norm', 'pred_42_patient_norm',
             'tbp_lv_Lext_patient_min_max', 'border_color_interaction_patient_min_max', 'tbp_lv_deltaLBnorm_past_norm', 'pred_28_patient_norm', 'pred_39',
             'tbp_lv_area_perim_ratio_future_norm', 'pred_47_future_norm', 'pred_22_patient_norm',
             'tbp_lv_A_future_norm', 'shape_color_consistency_patient_norm', 'consistency_symmetry_border',
             'pred_46_future_norm', 'tbp_lv_Bext_patient_norm', 'pred_36_future_norm', 'pred_22_future_norm',
             'tbp_lv_symm_2axis_patient_min_max', 'pred_28_past_norm',
             'volume_approximation_3d', 'pred_38', 'color_asymmetry_index', 'lesion_severity_index',
             'lesion_severity_index_patient_min_max', 'hue_contrast_patient_norm',
             'size_color_contrast_ratio_patient_min_max', 'pred_45_past_norm', 'pred_33_past_norm', 'tbp_lv_stdLExt',
             'pred_42', 'pred_2_past_norm', 'std_dev_contrast', 'tbp_lv_perimeterMM_patient_min_max',
             'symmetry_perimeter_interaction_patient_min_max', 'lesion_size_ratio_past_norm',
             'index_age_size_symmetry_patient_norm', 'lesion_shape_index', 'pred_52_future_norm', 'pred_51_future_norm',
             'pred_50_future_norm', 'tbp_lv_symm_2axis_angle', 'pred_24', 'lesion_severity_index_past_norm',
             'tbp_lv_deltaB_future_norm', 'pred_39_future_norm',
             'mean_hue_difference_patient_min_max', 'pred_9_patient_min_max_norm', 'pred_28_patient_min_max_norm',
             'pred_42_future_norm', 'tbp_lv_stdLExt_patient_min_max', 'std_dev_contrast_future_norm',
             'pred_9_future_norm', 'pred_25_past_norm', 'pred_29_future_norm',
             'tbp_lv_deltaB_patient_norm', 'pred_58', 'pred_52_patient_min_max_norm', 'pred_55_patient_min_max_norm',
             'pred_50_patient_norm', 'pred_17_past_norm', 'pred_14_past_norm', 'pred_41', 'pred_43_past_norm',
             'onehot_11', 'pred_53_past_norm', 'pred_16_past_norm', 'tbp_lv_Lext_past_norm', 'pred_1_future_norm',
             'hue_contrast_future_norm', 'pred_40', 'border_complexity_past_norm', 'pred_31', 'pred_22',
             'area_to_perimeter_ratio_patient_norm', 'pred_0_patient_min_max_norm', 'pred_2_patient_min_max_norm',
             'pred_48_future_norm', 'pred_49_patient_norm',
             'pred_53', 'tbp_lv_Cext', 'color_range', 'pred_43', 'pred_39_patient_min_max_norm',
             'pred_6_patient_norm', 'pred_6_patient_min_max_norm', 'border_color_interaction_past_norm',
             'pred_41_past_norm', 'pred_35_past_norm', 'pred_32_past_norm', 'pred_37_patient_min_max_norm',
             'tbp_lv_color_std_mean_patient_norm', 'age_normalized_nevi_confidence_2_past_norm', 'pred_17_patient_norm',
             'pred_17_future_norm', 'size_age_interaction_patient_norm', 'tbp_lv_Cext_future_norm',
             'pred_49_patient_min_max_norm', 'pred_58_patient_min_max_norm', 'color_contrast_index_patient_min_max', 'pred_59_past_norm', 'lesion_shape_index_future_norm', 'pred_54_patient_min_max_norm',
             'tbp_lv_nevi_confidence_patient_min_max', 'tbp_lv_Lext_future_norm',
             'volume_approximation_3d_patient_norm', 'position_distance_3d_past_norm', 'onehot_15',
             'tbp_lv_x_past_norm', 'tbp_lv_z', 'pred_15_patient_min_max_norm', 'tbp_lv_x_patient_min_max',
             'tbp_lv_L_past_norm', 'tbp_lv_z_patient_min_max', 'lesion_severity_index_future_norm',
             'pred_7_patient_min_max_norm', 'tbp_lv_B', 'comprehensive_lesion_index_patient_norm', 'pred_56',
             'volume_approximation_3d_past_norm', 'tbp_lv_deltaB_patient_min_max',
             'tbp_lv_radial_color_std_max_patient_min_max', 'pred_51', 'clin_size_long_diam_mm_past_norm',
             'color_consistency_patient_min_max', 'pred_18_future_norm', 'pred_33_patient_norm', 'tbp_lv_H_past_norm',
             'log_lesion_area_patient_norm', 'tbp_lv_z_patient_norm', 'pred_46_patient_norm',
             'age_normalized_nevi_confidence_past_norm', 'index_age_size_symmetry',
             'age_size_symmetry_index_patient_min_max', 'normalized_lesion_size_patient_min_max', 'age_size_symmetry_index', 'pred_12_future_norm', 'pred_24_patient_min_max_norm', 'onehot_46', 'pred_13', 'pred_15_past_norm',
             'tbp_lv_Cext_patient_norm', 'pred_53_patient_min_max_norm', 'tbp_lv_eccentricity',
             'pred_36_patient_min_max_norm', 'pred_52', 'pred_10_patient_min_max_norm', 'pred_51_past_norm',
             'pred_5_patient_min_max_norm', 'tbp_lv_y_future_norm', 'tbp_lv_area_perim_ratio_patient_min_max',
             'pred_3_patient_min_max_norm',
             'pred_5_future_norm', 'pred_20_patient_norm', 'pred_26_past_norm', 'pred_49', 'pred_17', 'tbp_lv_C_patient_min_max', 'pred_6_future_norm', 'overall_color_difference', 'pred_2', 'count_future',
             'pred_1_patient_min_max_norm', 'tbp_lv_radial_color_std_max', 'pred_13_future_norm', 'pred_59',
             'lesion_visibility_score_future_norm', 'area_to_perimeter_ratio', 'pred_19_past_norm',
             'pred_29_past_norm', 'pred_11', 'tbp_lv_nevi_confidence_past_norm', 'pred_3_future_norm',
             'pred_56_patient_norm', 'pred_22_patient_min_max_norm', 'pred_28_future_norm',
             'mean_hue_difference_patient_norm', 'tbp_lv_symm_2axis_angle_future_norm',
             'pred_34_past_norm', 'tbp_lv_L_patient_norm', 'pred_49_past_norm',
             'perimeter_to_area_ratio_patient_min_max', 'lesion_orientation_3d', 'pred_59_future_norm',
             'pred_16_future_norm', 'pred_9_past_norm', 'pred_21_patient_min_max_norm', 'pred_1_past_norm',
             'pred_42_patient_min_max_norm', 'pred_47_patient_norm', 'pred_51_patient_norm',
             'color_uniformity_past_norm',
             'color_range_patient_norm', 'pred_48_patient_min_max_norm', 'mean_hue_difference', 'tbp_lv_deltaLBnorm_future_norm',
             'tbp_lv_Aext_patient_min_max', 'border_color_interaction_future_norm', 'lesion_size_ratio_patient_norm',
             'tbp_lv_Hext_future_norm', 'color_asymmetry_index_patient_norm', 'lesion_orientation_3d_patient_norm',
             'lesion_orientation_3d_patient_min_max', 'tbp_lv_B_patient_norm', 'tbp_lv_C_future_norm',
             'pred_15_patient_norm', 'symmetry_border_consistency_patient_norm', 'normalized_lesion_size',
             'pred_46_past_norm', 'pred_0_patient_norm', 'color_consistency', 'color_contrast_index', 'lesion_color_difference',
             'tbp_lv_Bext',
             'tbp_lv_deltaLBnorm_patient_min_max', 'pred_11_past_norm', 'tbp_lv_A',
             'size_color_contrast_ratio', 'pred_27', 'pred_14', 'hue_contrast_patient_min_max',
             'pred_20', 'pred_36', 'pred_43_patient_min_max_norm', 'age_approx',
             'tbp_lv_Bext_patient_min_max', 'pred_52_patient_norm',
'tbp_lv_x_basel_norm', 'tbp_lv_Cext_xp_norm', 'tbp_lv_z_xp_norm', 'tbp_lv_symm_2axis_xp_norm', 'tbp_lv_stdLExt_xp_norm', 'pred_36_xp_norm', 'lesion_color_difference_basel_norm', 'color_range_basel_norm', 'tbp_lv_area_perim_ratio_xp_norm', 'tbp_lv_eccentricity_xp_norm', 'color_uniformity_xp_norm', 'lesion_size_ratio_basel_norm', 'pred_51_xp_norm', 'pred_3_past_norm', 'pred_16_basel_norm', 'tbp_lv_x_xp_norm', 'pred_53_basel_norm', 'border_color_interaction_2_xp_norm', 'mean_hue_difference_xp_norm', 'tbp_lv_deltaA', 'pred_44_basel_norm', 'color_asymmetry_index_basel_norm', 'pred_53_xp_norm', 'pred_58_xp_norm', 'border_color_interaction_2_basel_norm', 'pred_55_basel_norm', 'age_normalized_nevi_confidence_basel_norm', 'volume_approximation_3d_xp_norm', 'pred_55', 'pred_37_basel_norm', 'tbp_lv_L_basel_norm', 'std_dev_contrast_xp_norm', 'lesion_shape_index_xp_norm', 'lesion_size_ratio_xp_norm', 'tbp_lv_Hext', 'symmetry_border_consistency_basel_norm', 'age_size_symmetry_index_basel_norm', 'comprehensive_lesion_index_xp_norm', 'pred_34_xp_norm', 'perimeter_to_area_ratio_xp_norm', 'tbp_lv_y_xp_norm', 'symmetry_perimeter_interaction_basel_norm', 'pred_32_basel_norm', 'tbp_lv_norm_color_basel_norm', 'pred_19_xp_norm', 'pred_50_xp_norm', 'pred_38_basel_norm', 'tbp_lv_color_std_mean_basel_norm', 'pred_40_basel_norm', 'tbp_lv_deltaL_basel_norm', 'tbp_lv_eccentricity_basel_norm', 'tbp_lv_A_basel_norm', 'pred_17_xp_norm', 'shape_complexity_index_basel_norm', 'pred_18_xp_norm', 'tbp_lv_stdL_xp_norm', 'pred_39_xp_norm', 'pred_59_basel_norm', 'pred_54_xp_norm', 'tbp_lv_Bext_xp_norm', 'pred_29', 'pred_8_basel_norm', 'pred_6_xp_norm', 'onehot_45', 'pred_12_patient_min_max_norm', 'pred_31_xp_norm', 'index_age_size_symmetry_xp_norm', 'pred_9_xp_norm', 'pred_25_xp_norm', 'pred_4_xp_norm', 'pred_0_xp_norm', 'pred_5_basel_norm', 'pred_59_xp_norm', 'perimeter_to_area_ratio_basel_norm', 'tbp_lv_deltaLB_xp_norm', 'tbp_lv_stdLExt_basel_norm', 'pred_34_basel_norm', 'pred_31_basel_norm', 'pred_30_basel_norm', 'pred_29_basel_norm', 'area_to_perimeter_ratio_basel_norm', 'tbp_lv_norm_color_xp_norm', 'pred_9_basel_norm', 'overall_color_difference_xp_norm', 'border_length_ratio_basel_norm', 'shape_color_consistency_basel_norm', 'border_color_interaction_basel_norm', 'consistency_color_basel_norm', 'tbp_lv_norm_border_xp_norm', 'tbp_lv_B_xp_norm', 'tbp_lv_perimeterMM_basel_norm', 'tbp_lv_Cext_basel_norm', 'color_shape_composite_index_xp_norm', 'color_uniformity', 'pred_10', 'pred_21', 'pred_37', 'age_approx_basel_norm', 'lesion_visibility_score_xp_norm', 'tbp_lv_norm_border_basel_norm', 'color_shape_composite_index_basel_norm', 'border_complexity_xp_norm', 'luminance_contrast_xp_norm', 'tbp_lv_area_perim_ratio_basel_norm', 'tbp_lv_symm_2axis_basel_norm', 'tbp_lv_deltaLB_basel_norm', 'pred_38_xp_norm', 'pred_42_xp_norm', 'pred_35_xp_norm', 'tbp_lv_stdL_basel_norm', 'consistency_symmetry_border_basel_norm', 'lesion_severity_index_basel_norm', 'pred_45_xp_norm', 'pred_48_xp_norm', 'tbp_lv_norm_border', 'luminance_contrast_basel_norm', 'pred_22_xp_norm', 'log_lesion_area_basel_norm', 'age_size_symmetry_index_xp_norm', 'tbp_lv_perimeterMM_xp_norm', 'shape_color_consistency_xp_norm', 'pred_50_basel_norm', 'pred_43_basel_norm', 'pred_35_basel_norm', 'consistency_symmetry_border_xp_norm', 'consistency_color_xp_norm', 'hue_color_std_interaction_xp_norm', 'pred_23_basel_norm', 'pred_21_basel_norm', 'pred_19_basel_norm', 'pred_18_basel_norm', 'border_length_ratio_xp_norm', 'shape_complexity_index_xp_norm', 'log_lesion_area_xp_norm', 'border_color_interaction_xp_norm', 'tbp_lv_areaMM2_basel_norm', 'tbp_lv_color_std_mean_xp_norm', 'tbp_lv_symm_2axis_angle_xp_norm', 'tbp_lv_radial_color_std_max_xp_norm', 'hue_color_std_interaction_basel_norm', 'pred_15_xp_norm', 'tbp_lv_symm_2axis_angle_basel_norm', 'pred_25_basel_norm', 'pred_23_xp_norm', 'comprehensive_lesion_index_basel_norm', 'pred_32_xp_norm', 'symmetry_perimeter_interaction_xp_norm',
             'color_consistency_xp_norm', 'pred_13_basel_norm', 'pred_24_xp_norm', 'lesion_visibility_score_basel_norm',
             'pred_0_basel_norm', 'tbp_lv_minorAxisMM_xp_norm', 'normalized_lesion_size_xp_norm', 'pred_39_basel_norm',
             'pred_45_basel_norm', 'color_variance_ratio_xp_norm', 'color_variance_ratio_basel_norm', 'pred_40_xp_norm',
             'tbp_lv_y', 'volume_approximation_3d_basel_norm', 'pred_8_xp_norm', 'tbp_lv_areaMM2_xp_norm',
             'pred_7_xp_norm', 'pred_42_basel_norm', 'pred_49_basel_norm', 'size_age_interaction_basel_norm',
             'pred_54_basel_norm', 'pred_13_xp_norm', 'tbp_lv_radial_color_std_max_basel_norm', 'pred_48_basel_norm',
             'pred_3_basel_norm', 'pred_4_basel_norm', 'pred_41_basel_norm', 'pred_57_xp_norm', 'pred_44_xp_norm',
             'pred_47_patient_min_max_norm',
             'pred_22_basel_norm', 'tbp_lv_deltaB_basel_norm', 'pred_1_xp_norm', 'pred_58_basel_norm', 'tbp_lv_C',
             'pred_7_basel_norm',
             'color_uniformity_future_norm', 'size_age_interaction', 'tbp_lv_H_xp_norm', 'pred_52_basel_norm',
             'tbp_lv_Aext', 'lesion_severity_index_xp_norm', 'tbp_lv_nevi_confidence_xp_norm', 'pred_28_basel_norm',
             'tbp_lv_z_basel_norm', 'normalized_lesion_size_basel_norm', 'pred_30_xp_norm', 'tbp_lv_deltaB',
             'hue_contrast', 'tbp_lv_deltaB_xp_norm', 'tbp_lv_H_basel_norm', 'pred_20_basel_norm', 'pred_24_basel_norm',
             'pred_56_basel_norm', 'area_to_perimeter_ratio_xp_norm', 'position_distance_3d_xp_norm', 'pred_37_xp_norm',
             'pred_20_xp_norm', 'pred_46_patient_min_max_norm', 'pred_6', 'pred_11_xp_norm', 'pred_52_xp_norm',
             'lesion_orientation_3d_basel_norm', 'pred_27_basel_norm', 'tbp_lv_nevi_confidence_basel_norm',
             'color_range_xp_norm', 'lesion_orientation_3d_xp_norm', 'size_color_contrast_ratio_xp_norm',
             'tbp_lv_deltaA_basel_norm', 'pred_12_xp_norm', 'tbp_lv_B_basel_norm', 'tbp_lv_symm_2axis_angle_past_norm',
             'lesion_color_difference_xp_norm', 'pred_41_xp_norm', 'pred_2_basel_norm',
             'tbp_lv_Lext', 'pred_46_basel_norm', 'pred_33_xp_norm', 'pred_33', 'pred_26', 'pred_12_basel_norm', 'pred_46', 'mean_hue_difference_basel_norm',
             'color_contrast_index_basel_norm', 'color_contrast_index_xp_norm', 'pred_58_patient_norm',
             'tbp_lv_deltaA_xp_norm', 'pred_3_xp_norm', 'pred_26_patient_min_max_norm',
             'index_age_size_symmetry_basel_norm', 'tbp_lv_A_xp_norm', 'tbp_lv_C_basel_norm', 'pred_57_basel_norm',
             'age_approx_xp_norm', 'lesion_shape_index_basel_norm', 'pred_49_xp_norm', 'tbp_lv_deltaLBnorm', 'pred_47_basel_norm', 'tbp_lv_nevi_confidence_patient_norm', 'overall_color_difference_basel_norm', 'tbp_lv_deltaL_xp_norm',
             'pred_40_patient_min_max_norm', 'tbp_lv_Aext_xp_norm', 'tbp_lv_H_patient_min_max',
             'color_contrast_index_patient_norm', 'pred_1_basel_norm', 'pred_33_basel_norm', 'pred_56_xp_norm',
             'tbp_lv_Hext_xp_norm', 'comprehensive_lesion_index_future_norm', 'pred_25_patient_min_max_norm', 'pred_16',
             'pred_16_patient_norm', 'color_consistency_basel_norm', 'position_distance_3d_basel_norm', 'tbp_lv_Lext_basel_norm', 'tbp_lv_Aext_basel_norm']

cat_avoid = ['pred_50_future_norm', 'pred_53_past_norm', 'pred_43_future_norm', 'pred_42_past_norm', 'clin_size_long_diam_mm_past_norm', 'pred_47_past_norm', 'pred_45_future_norm', 'pred_58_past_norm', 'pred_55_future_norm', 'pred_56_future_norm', 'age_approx_past_norm', 'pred_47_future_norm', 'pred_49_future_norm', 'pred_45_past_norm', 'pred_54_future_norm', 'consistency_color_past_norm', 'pred_41_past_norm', 'lesion_shape_index_past_norm', 'tbp_lv_deltaLB', 'count_per_patient_past_norm', 'tbp_lv_symm_2axis_past_norm', 'age_size_symmetry_index_past_norm', 'tbp_lv_x_past_norm', 'shape_color_consistency_past_norm', 'color_range_past_norm', 'tbp_lv_area_perim_ratio', 'color_variance_ratio_past_norm', 'luminance_contrast_past_norm', 'pred_2_past_norm', 'tbp_lv_color_std_mean', 'symmetry_perimeter_interaction_past_norm',
             'border_complexity_past_norm', 'std_dev_contrast_past_norm', 'normalized_lesion_size_past_norm', 'color_contrast_index_past_norm', 'hue_color_std_interaction_past_norm', 'symmetry_border_consistency_past_norm', 'size_age_interaction_past_norm', 'tbp_lv_eccentricity', 'pred_4_past_norm', 'pred_37_past_norm', 'pred_23_past_norm', 'tbp_lv_B_past_norm', 'pred_33_past_norm', 'tbp_lv_Hext_past_norm', 'pred_32_past_norm', 'pred_30_past_norm', 'pred_28_past_norm', 'pred_27_past_norm', 'tbp_lv_area_perim_ratio_past_norm', 'tbp_lv_color_std_mean_past_norm', 'pred_22_past_norm', 'pred_7_past_norm', 'tbp_lv_deltaL_past_norm', 'pred_20_past_norm', 'pred_19_past_norm', 'tbp_lv_deltaLBnorm_past_norm', 'pred_17_past_norm', 'tbp_lv_norm_border_past_norm', 'pred_11_past_norm', 'pred_10_past_norm', 'tbp_lv_Hext', 'pred_41_future_norm', 'volume_approximation_3d_patient_norm',
             'pred_39_future_norm', 'tbp_lv_symm_2axis_patient_norm', 'pred_31_patient_min_max_norm', 'pred_30_patient_min_max_norm', 'pred_24_patient_min_max_norm', 'pred_23_patient_min_max_norm', 'pred_22_patient_min_max_norm', 'tbp_lv_stdL_patient_norm', 'tbp_lv_stdLExt_patient_norm', 'pred_7_patient_min_max_norm', 'tbp_lv_norm_border_patient_norm', 'tbp_lv_z_patient_norm', 'pred_4_patient_min_max_norm', 'hue_contrast_patient_norm', 'pred_56_patient_norm', 'pred_55_patient_norm', 'luminance_contrast_patient_norm', 'pred_50_patient_norm', 'pred_32_patient_min_max_norm', 'pred_39_patient_min_max_norm', 'lesion_color_difference_patient_norm', 'onehot_2', 'onehot_14', 'onehot_13', 'onehot_12', 'onehot_7', 'onehot_6', 'onehot_4', 'onehot_3', 'onehot_1', 'pred_40_patient_min_max_norm', 'onehot_0', 'pred_56_patient_min_max_norm',
             'tbp_lv_deltaLB_patient_norm', 'pred_49_patient_min_max_norm', 'pred_48_patient_min_max_norm', 'tbp_lv_eccentricity_patient_norm', 'pred_45_patient_min_max_norm', 'pred_49_patient_norm', 'pred_44_patient_norm', 'pred_38_future_norm', 'comprehensive_lesion_index_patient_norm', 'size_age_interaction_patient_min_max', 'consistency_color_patient_min_max', 'color_consistency_patient_min_max', 'std_dev_contrast_patient_norm', 'color_uniformity_patient_min_max', 'lesion_shape_index_patient_min_max', 'tbp_lv_stdL_patient_min_max', 'tbp_lv_norm_color_patient_min_max', 'shape_complexity_index_patient_min_max', 'tbp_lv_norm_border_patient_min_max', 'tbp_lv_deltaB_patient_min_max', 'border_color_interaction_2_patient_norm', 'tbp_lv_Hext_patient_min_max', 'tbp_lv_Cext_patient_min_max',
             'age_approx_patient_min_max', 'border_length_ratio_patient_norm', 'hue_color_std_interaction_patient_min_max', 'normalized_lesion_size_patient_min_max', 'pred_39_patient_norm', 'pred_10_patient_norm', 'pred_37_patient_norm', 'pred_35_patient_norm', 'pred_34_patient_norm', 'pred_32_patient_norm', 'pred_31_patient_norm', 'pred_30_patient_norm', 'pred_22_patient_norm', 'pred_8_patient_norm', 'std_dev_contrast_patient_min_max', 'pred_7_patient_norm', 'pred_0_patient_norm', 'shape_color_consistency_patient_min_max', 'color_consistency_patient_norm', 'border_color_interaction_2_patient_min_max', 'lesion_severity_index_patient_norm', 'shape_complexity_index_patient_norm', 'onehot_16', 'onehot_18', 'onehot_21', 'shape_complexity_index_future_norm', 'border_color_interaction_2_future_norm',
             'border_complexity', 'symmetry_perimeter_interaction_future_norm', 'std_dev_contrast_future_norm', 'mean_hue_difference_future_norm', 'normalized_lesion_size_future_norm', 'perimeter_to_area_ratio', 'hue_color_std_interaction_future_norm', 'age_size_symmetry_index_future_norm', 'symmetry_border_consistency', 'luminance_contrast_future_norm', 'consistency_color', 'tbp_lv_symm_2axis_angle_future_norm', 'tbp_lv_symm_2axis_future_norm', 'tbp_lv_stdL_future_norm', 'tbp_lv_norm_border_future_norm', 'color_asymmetry_index_future_norm', 'count_per_patient_future_norm', 'onehot_22', 'pred_24_future_norm', 'pred_36_future_norm', 'pred_35_future_norm', 'tbp_lv_stdL', 'pred_31_future_norm', 'pred_30_future_norm', 'pred_28_future_norm', 'pred_27_future_norm', 'tbp_lv_symm_2axis_angle', 'pred_0_future_norm', 'pred_20_future_norm',
             'pred_10_future_norm', 'pred_9_future_norm', 'pred_7_future_norm', 'pred_6_future_norm', 'pred_4_future_norm', 'pred_1_future_norm', 'tbp_lv_minorAxisMM_future_norm', 'tbp_lv_deltaLB_future_norm', 'hue_color_std_interaction', 'onehot_33', 'onehot_42', 'onehot_39', 'onehot_38', 'onehot_37', 'onehot_36', 'onehot_35', 'onehot_34', 'onehot_31', 'tbp_lv_areaMM2_future_norm', 'onehot_30', 'onehot_29', 'onehot_28', 'onehot_26', 'onehot_25', 'onehot_24', 'onehot_23', 'pred_1', 'pred_2', 'pred_3', 'age_approx_patient_norm', 'shape_complexity_index', 'tbp_lv_Cext_future_norm', 'pred_56', 'pred_53', 'pred_40', 'pred_39', 'pred_35', 'pred_31', 'pred_30', 'pred_23', 'pred_21', 'pred_19', 'pred_11', 'pred_9', 'pred_8', 'onehot_17',
             'age_size_symmetry_index_patient_norm', 'onehot_41', 'pred_56_past_norm', 'pred_50', 'onehot_44',
             'pred_54', 'pred_55_past_norm', 'symmetry_perimeter_interaction_patient_norm', 'tbp_lv_nevi_confidence',
             'border_complexity_patient_norm', 'pred_4_patient_norm', 'pred_49_past_norm', 'pred_34',
             'pred_14_past_norm', 'pred_43_past_norm', 'tbp_lv_symm_2axis_angle_patient_norm', 'pred_38_past_norm',
             'lesion_size_ratio_patient_norm', 'pred_35_past_norm', 'pred_34_past_norm', 'pred_21_past_norm',
             'lesion_shape_index_patient_norm', 'pred_3_patient_norm', 'clin_size_long_diam_mm_future_norm',
             'pred_11_future_norm', 'age_size_symmetry_index_patient_min_max', 'lesion_shape_index',
             'log_lesion_area_patient_min_max', 'tbp_lv_Aext_patient_norm', 'pred_54_patient_min_max_norm',
             'pred_5_patient_min_max_norm', 'tbp_lv_A_past_norm', 'consistency_symmetry_border_future_norm',
             'pred_9_patient_min_max_norm', 'pred_52_future_norm', 'size_color_contrast_ratio',
             'lesion_color_difference_patient_min_max', 'lesion_severity_index', 'pred_29_future_norm',
             'pred_18_patient_min_max_norm', 'pred_23_future_norm', 'luminance_contrast_patient_min_max',
             'pred_33_patient_min_max_norm', 'color_range_future_norm', 'pred_0_patient_min_max_norm',
             'tbp_lv_areaMM2_past_norm', 'pred_27_patient_norm', 'tbp_lv_deltaB_past_norm',
             'size_color_contrast_ratio_past_norm', 'onehot_27', 'comprehensive_lesion_index_past_norm',
             'pred_36_patient_norm', 'onehot_20', 'log_lesion_area_past_norm', 'shape_complexity_index_past_norm',
             'color_consistency_past_norm', 'tbp_lv_Lext_patient_min_max', 'tbp_lv_area_perim_ratio_patient_min_max',
             'age_normalized_nevi_confidence_patient_min_max', 'size_color_contrast_ratio_patient_min_max',
             'tbp_lv_color_std_mean_patient_min_max', 'tbp_lv_z_future_norm', 'tbp_lv_Bext_patient_norm',
             'lesion_color_difference_future_norm', 'border_complexity_future_norm',
             'consistency_symmetry_border_past_norm',
             'border_complexity_patient_min_max', 'tbp_lv_nevi_confidence_past_norm', 'tbp_lv_deltaL_patient_min_max',
             'pred_2_patient_norm', 'luminance_contrast', 'tbp_lv_A_future_norm', 'tbp_lv_area_perim_ratio_future_norm',
             'tbp_lv_color_std_mean_future_norm', 'pred_5', 'pred_9_patient_norm', 'size_age_interaction_patient_norm',
             'pred_31_past_norm', 'onehot_40', 'onehot_19', 'pred_29_patient_norm', 'tbp_lv_deltaL',
             'pred_48_past_norm', 'pred_41_patient_min_max_norm', 'pred_35_patient_min_max_norm',
             'pred_34_patient_min_max_norm', 'tbp_lv_A', 'pred_20_patient_min_max_norm', 'pred_57_patient_norm',
             'lesion_size_ratio_future_norm', 'perimeter_to_area_ratio_future_norm',
             'consistency_symmetry_border_patient_norm', 'tbp_lv_deltaA_past_norm', 'color_variance_ratio',
             'tbp_lv_radial_color_std_max_past_norm', 'tbp_lv_stdL_past_norm', 'tbp_lv_Lext_past_norm',
             'clin_size_long_diam_mm_patient_norm', 'tbp_lv_Bext_past_norm', 'color_contrast_index',
             'lesion_size_ratio_past_norm', 'pred_44_future_norm', 'pred_42_future_norm', 'pred_37_future_norm',
             'tbp_lv_Lext_patient_norm', 'pred_25_future_norm', 'tbp_lv_area_perim_ratio_patient_norm',
             'pred_18_future_norm', 'tbp_lv_deltaL_patient_norm', 'mean_hue_difference_past_norm',
             'border_color_interaction_future_norm', 'border_color_interaction_past_norm',
             'age_normalized_nevi_confidence_past_norm', 'pred_13_patient_min_max_norm',
             'color_asymmetry_index_past_norm', 'color_consistency_future_norm',
             'symmetry_border_consistency_future_norm', 'area_to_perimeter_ratio_future_norm',
             'consistency_color_future_norm',
             'tbp_lv_nevi_confidence_future_norm', 'tbp_lv_deltaLBnorm_future_norm', 'tbp_lv_deltaL_future_norm',
             'pred_44_past_norm', 'lesion_visibility_score', 'lesion_size_ratio', 'tbp_lv_C_future_norm',
             'pred_50_past_norm', 'pred_51_past_norm', 'tbp_lv_areaMM2_patient_norm', 'pred_41', 'pred_21_patient_norm',
             'tbp_lv_Bext_future_norm', 'tbp_lv_color_std_mean_patient_norm', 'pred_33_future_norm',
             'tbp_lv_Cext_past_norm', 'color_shape_composite_index_patient_norm', 'tbp_lv_deltaLB_past_norm',
             'tbp_lv_norm_color_past_norm', 'onehot_8', 'age_normalized_nevi_confidence_future_norm',
             'tbp_lv_z_past_norm', 'size_color_contrast_ratio_future_norm', 'color_variance_ratio_future_norm',
             'perimeter_to_area_ratio_past_norm', 'pred_42_patient_norm', 'shape_color_consistency', 'pred_18',
             'pred_53_future_norm', 'size_age_interaction_future_norm', 'pred_0_past_norm', 'pred_22',
             'age_normalized_nevi_confidence_2_past_norm', 'pred_25_past_norm', 'count_future', 'pred_48_future_norm',
             'pred_4', 'tbp_lv_Cext', 'color_range_patient_min_max', 'pred_37_patient_min_max_norm', 'pred_16',
             'pred_59_patient_min_max_norm', 'pred_34_future_norm', 'pred_5_future_norm', 'pred_44',
             'shape_color_consistency_future_norm', 'pred_24_patient_norm', 'pred_49', 'tbp_lv_stdLExt_patient_min_max',
             'pred_52_patient_norm', 'tbp_lv_L_patient_min_max', 'tbp_lv_L_future_norm', 'pred_58_future_norm',
             'tbp_lv_Lext_future_norm', 'pred_5_past_norm', 'lesion_shape_index_future_norm', 'pred_8_past_norm',
             'area_to_perimeter_ratio_patient_norm', 'pred_32_future_norm', 'pred_14_future_norm',
             'border_color_interaction', 'pred_53_patient_min_max_norm', 'tbp_lv_L_past_norm',
             'color_variance_ratio_patient_norm', 'pred_11_patient_min_max_norm', 'tbp_lv_Aext_patient_min_max',
             'tbp_lv_deltaLB_patient_min_max', 'lesion_visibility_score_patient_min_max',
             'mean_hue_difference_patient_min_max', 'comprehensive_lesion_index_patient_min_max',
             'pred_2_patient_min_max_norm',
             'volume_approximation_3d_patient_min_max', 'pred_39_past_norm', 'tbp_lv_minorAxisMM_past_norm', 'pred_13',
             'color_contrast_index_future_norm', 'tbp_lv_symm_2axis_angle_patient_min_max',
             'pred_50_patient_min_max_norm', 'consistency_symmetry_border', 'pred_21_future_norm',
             'pred_10_patient_min_max_norm', 'pred_1_past_norm', 'lesion_severity_index_past_norm',
             'pred_18_past_norm', 'tbp_lv_minorAxisMM_patient_norm', 'pred_9_past_norm',
             'tbp_lv_symm_2axis_angle_past_norm', 'pred_33_patient_norm', 'pred_53_patient_norm', 'std_dev_contrast',
             'tbp_lv_Bext_patient_min_max', 'tbp_lv_perimeterMM_past_norm', 'pred_38_patient_min_max_norm',
             'pred_27_patient_min_max_norm',
             'tbp_lv_Bext', 'pred_43_patient_norm', 'pred_26_past_norm', 'pred_44_patient_min_max_norm', 'pred_25', 'onehot_10', 'tbp_lv_deltaLBnorm_patient_norm', 'symmetry_border_consistency_patient_norm',
             'color_contrast_index_patient_norm', 'pred_36_patient_min_max_norm', 'pred_6',
             'border_color_interaction_patient_min_max', 'lesion_orientation_3d_patient_min_max',
             'tbp_lv_eccentricity_patient_min_max', 'tbp_lv_stdLExt_past_norm', 'pred_20_patient_norm',
             'count_per_patient', 'pred_41_patient_norm', 'hue_color_std_interaction_patient_norm',
             'comprehensive_lesion_index', 'pred_22_future_norm', 'tbp_lv_norm_color_future_norm', 'pred_38_patient_norm', 'pred_33', 'pred_40_patient_norm', 'pred_57_future_norm',
             'color_range_patient_norm', 'pred_19_patient_norm', 'lesion_orientation_3d_patient_norm',
             'pred_14_patient_min_max_norm', 'color_shape_composite_index_past_norm',
             'tbp_lv_perimeterMM_future_norm', 'pred_54_patient_norm', 'pred_48_patient_norm', 'pred_58_patient_norm',
             'border_color_interaction_2', 'pred_8_patient_min_max_norm', 'hue_contrast_past_norm',
             'comprehensive_lesion_index_future_norm', 'pred_19_future_norm', 'color_range', 'border_color_interaction_2_past_norm', 'tbp_lv_Cext_patient_norm', 'tbp_lv_Aext_past_norm',
             'pred_59', 'pred_1_patient_min_max_norm', 'tbp_lv_B_future_norm', 'pred_54_past_norm', 'index_age_size_symmetry_patient_norm',
             'tbp_lv_areaMM2_patient_min_max', 'tbp_lv_norm_color', 'onehot_32',
             'tbp_lv_perimeterMM', 'pred_32', 'volume_approximation_3d_past_norm', 'tbp_lv_C_patient_min_max',
             'log_lesion_area_future_norm', 'pred_42', 'lesion_orientation_3d', 'tbp_lv_deltaB', 'pred_46_future_norm',
             'clin_size_long_diam_mm_patient_min_max', 'pred_52_patient_min_max_norm',
             'symmetry_perimeter_interaction_patient_min_max', 'pred_12', 'tbp_lv_x_future_norm', 'pred_0',
             'tbp_lv_A_patient_min_max', 'pred_58', 'pred_45_patient_norm', 'tbp_lv_B', 'pred_42_patient_min_max_norm',
             'pred_25_patient_norm', 'consistency_color_patient_norm', 'symmetry_border_consistency_patient_min_max',
             'tbp_lv_norm_border', 'pred_40_future_norm', 'pred_29_past_norm',
             'tbp_lv_radial_color_std_max_patient_norm',
             'pred_59_past_norm', 'pred_6_patient_norm', 'size_color_contrast_ratio_patient_norm',
             'pred_51_future_norm', 'pred_40_past_norm', 'pred_57_past_norm', 'border_length_ratio_patient_min_max',
             'pred_43_patient_min_max_norm', 'border_length_ratio_future_norm', 'hue_contrast_future_norm',
             'tbp_lv_eccentricity_future_norm',
             'tbp_lv_minorAxisMM_patient_min_max', 'pred_24_past_norm', 'pred_16_future_norm', 'pred_3_patient_min_max_norm', 'pred_24',
             'pred_27', 'lesion_severity_index_future_norm', 'pred_14_patient_norm', 'overall_color_difference',
             'overall_color_difference_patient_min_max', 'onehot_15', 'normalized_lesion_size_patient_norm',
             'pred_12_patient_min_max_norm', 'tbp_lv_C', 'pred_19_patient_min_max_norm',
             'tbp_lv_eccentricity_past_norm', 'tbp_lv_deltaA_future_norm', 'pred_28_patient_norm',
             'pred_46_patient_norm', 'pred_13_past_norm', 'color_contrast_index_patient_min_max',
             'log_lesion_area', 'position_distance_3d_future_norm', 'position_distance_3d_past_norm',
             'color_asymmetry_index_patient_min_max', 'pred_55', 'pred_13_patient_norm', 'onehot_11',
             'pred_57_patient_min_max_norm', 'color_shape_composite_index_patient_min_max',
             'tbp_lv_deltaLBnorm_patient_min_max', 'hue_contrast_patient_min_max', 'pred_36_past_norm',
             'area_to_perimeter_ratio_past_norm', 'tbp_lv_x_patient_norm', 'pred_17',
             'tbp_lv_y', 'tbp_lv_C_past_norm', 'shape_color_consistency_patient_norm', 'pred_37', 'age_normalized_nevi_confidence_2_future_norm',
             'pred_15_future_norm', 'border_color_interaction_patient_norm', 'lesion_color_difference',
             'tbp_lv_norm_color_patient_norm', 'tbp_lv_perimeterMM_patient_norm', 'pred_15', 'pred_43', 'tbp_lv_Aext',
             'pred_47_patient_min_max_norm', 'pred_55_patient_min_max_norm', 'onehot_43',
             'tbp_lv_y_past_norm', 'log_lesion_area_patient_norm', 'tbp_lv_Hext_patient_norm', 'pred_12_past_norm',
             'tbp_lv_z', 'tbp_lv_radial_color_std_max_future_norm', 'color_uniformity_past_norm',
             'pred_17_patient_norm', 'area_to_perimeter_ratio_patient_min_max', 'pred_59_future_norm',
             'lesion_orientation_3d_future_norm', 'color_shape_composite_index_future_norm', 'pred_5_patient_norm',
             'tbp_lv_L', 'consistency_symmetry_border_patient_min_max', 'overall_color_difference_future_norm',
             'color_shape_composite_index', 'index_age_size_symmetry_past_norm', 'hue_contrast', 'pred_15_past_norm',
             'border_length_ratio_past_norm',
             'perimeter_to_area_ratio_patient_norm', 'pred_46_past_norm', 'pred_25_patient_min_max_norm',
             'tbp_lv_symm_2axis_patient_min_max', 'pred_1_patient_norm', 'tbp_lv_H_patient_min_max',
             'pred_58_patient_min_max_norm', 'onehot_5', 'pred_51_patient_min_max_norm', 'pred_47_patient_norm',
             'tbp_lv_deltaB_patient_norm', 'index_age_size_symmetry_future_norm', 'pred_16_patient_min_max_norm',
             'tbp_lv_stdLExt_future_norm', 'tbp_lv_C_patient_norm', 'tbp_lv_radial_color_std_max', 'tbp_lv_perimeterMM_patient_min_max', 'age_normalized_nevi_confidence', 'tbp_lv_stdLExt',
             'pred_10', 'pred_48', 'position_distance_3d_patient_min_max', 'lesion_visibility_score_past_norm', 'age_approx',
             'tbp_lv_z_patient_min_max', 'tbp_lv_areaMM2',
             'tbp_lv_B_patient_min_max', 'pred_46_patient_min_max_norm', 'overall_color_difference_patient_norm',
             'lesion_size_ratio_patient_min_max', 'pred_11_patient_norm', 'pred_29',
             'age_normalized_nevi_confidence_patient_norm', 'pred_13_future_norm',
             'age_normalized_nevi_confidence_2_patient_min_max', 'age_size_symmetry_index', 'tbp_lv_H_past_norm',
             'pred_28_patient_min_max_norm', 'pred_45', 'pred_17_future_norm', 'border_length_ratio',
             'tbp_lv_Hext_future_norm', 'position_distance_3d_patient_norm', 'tbp_lv_L_patient_norm',
             'lesion_color_difference_past_norm', 'color_variance_ratio_patient_min_max', 'pred_8_future_norm',
             'pred_21_patient_min_max_norm', 'lesion_severity_index_patient_min_max', 'pred_16_past_norm', 'onehot_46',
             'pred_12_future_norm', 'pred_15_patient_min_max_norm', 'volume_approximation_3d_future_norm',
             'tbp_lv_deltaB_future_norm', 'pred_2_future_norm', 'tbp_lv_symm_2axis', 'mean_hue_difference_patient_norm', 'area_to_perimeter_ratio', 'pred_51',
             'tbp_lv_deltaA_patient_norm', 'tbp_lv_x', 'pred_20', 'tbp_lv_nevi_confidence_patient_min_max',
'pred_51_xp_norm', 'pred_11_xp_norm', 'pred_55_xp_norm', 'tbp_lv_Aext_xp_norm', 'pred_27_xp_norm', 'consistency_symmetry_border_xp_norm', 'lesion_severity_index_basel_norm', 'pred_4_xp_norm', 'pred_20_xp_norm', 'hue_color_std_interaction_xp_norm', 'pred_2_basel_norm', 'tbp_lv_stdL_basel_norm', 'consistency_symmetry_border_basel_norm', 'area_to_perimeter_ratio_xp_norm', 'log_lesion_area_xp_norm', 'pred_10_basel_norm', 'shape_color_consistency_xp_norm', 'tbp_lv_A_xp_norm', 'tbp_lv_norm_color_basel_norm', 'pred_41_xp_norm', 'tbp_lv_symm_2axis_basel_norm', 'age_approx_basel_norm', 'size_age_interaction_basel_norm', 'pred_18_basel_norm', 'comprehensive_lesion_index_basel_norm', 'tbp_lv_area_perim_ratio_basel_norm', 'tbp_lv_color_std_mean_basel_norm', 'tbp_lv_area_perim_ratio_xp_norm', 'color_shape_composite_index_basel_norm', 'overall_color_difference_xp_norm', 'tbp_lv_B_xp_norm', 'pred_44_xp_norm', 'pred_9_basel_norm', 'tbp_lv_radial_color_std_max_xp_norm', 'pred_25_basel_norm', 'pred_42_xp_norm', 'pred_5_xp_norm', 'pred_3_future_norm', 'tbp_lv_deltaL_basel_norm', 'border_length_ratio_xp_norm', 'pred_22_xp_norm', 'pred_35_basel_norm', 'pred_39_xp_norm', 'shape_complexity_index_xp_norm', 'tbp_lv_B_basel_norm', 'age_approx_xp_norm', 'color_range_xp_norm', 'pred_53_basel_norm', 'pred_0_xp_norm', 'pred_16_xp_norm', 'pred_50_xp_norm', 'tbp_lv_Bext_basel_norm', 'pred_0_basel_norm', 'pred_8_basel_norm', 'tbp_lv_deltaLB_xp_norm', 'tbp_lv_Lext_xp_norm', 'tbp_lv_deltaL_xp_norm', 'lesion_visibility_score_basel_norm', 'consistency_color_xp_norm', 'pred_40_xp_norm', 'pred_50_basel_norm', 'tbp_lv_nevi_confidence_xp_norm', 'pred_18_xp_norm', 'pred_34_xp_norm', 'color_contrast_index_xp_norm', 'pred_31_xp_norm', 'pred_32_basel_norm', 'pred_30_basel_norm', 'luminance_contrast_xp_norm', 'pred_13_basel_norm', 'tbp_lv_symm_2axis_angle_basel_norm', 'lesion_shape_index_basel_norm', 'pred_34_basel_norm',
             'pred_5_basel_norm', 'pred_53_xp_norm', 'pred_4_basel_norm', 'color_asymmetry_index_xp_norm',
             'symmetry_border_consistency_basel_norm', 'lesion_orientation_3d_basel_norm', 'pred_43_basel_norm',
             'symmetry_perimeter_interaction', 'size_color_contrast_ratio_basel_norm', 'log_lesion_area_basel_norm',
             'pred_14_basel_norm', 'lesion_size_ratio_basel_norm', 'pred_27_basel_norm', 'pred_1_basel_norm',
             'pred_38_xp_norm', 'comprehensive_lesion_index_xp_norm', 'pred_49_xp_norm', 'tbp_lv_areaMM2_basel_norm',
             'volume_approximation_3d_basel_norm', 'tbp_lv_minorAxisMM', 'color_consistency_xp_norm', 'pred_28_xp_norm',
             'age_normalized_nevi_confidence_xp_norm', 'color_variance_ratio_basel_norm', 'pred_57_xp_norm',
             'tbp_lv_A_basel_norm', 'pred_35_xp_norm', 'pred_32_xp_norm', 'pred_24_xp_norm',
             'hue_color_std_interaction_basel_norm', 'lesion_orientation_3d_xp_norm', 'pred_37_basel_norm',
             'border_complexity_xp_norm', 'tbp_lv_norm_border_xp_norm', 'tbp_lv_norm_border_basel_norm',
             'pred_56_basel_norm', 'pred_49_basel_norm', 'pred_33_basel_norm', 'pred_45_basel_norm',
             'pred_42_basel_norm', 'pred_59_xp_norm',
             'border_color_interaction_basel_norm', 'pred_7_basel_norm', 'tbp_lv_Hext_basel_norm',
             'tbp_lv_areaMM2_xp_norm', 'pred_44_basel_norm', 'pred_48_xp_norm', 'border_color_interaction_xp_norm',
             'pred_55_basel_norm', 'symmetry_border_consistency_xp_norm', 'tbp_lv_Cext_xp_norm', 'pred_52_xp_norm',
             'tbp_lv_perimeterMM_basel_norm', 'tbp_lv_minorAxisMM_xp_norm', 'tbp_lv_minorAxisMM_basel_norm',
             'color_variance_ratio_xp_norm', 'tbp_lv_stdL_xp_norm', 'pred_37_xp_norm', 'std_dev_contrast_basel_norm',
             'lesion_severity_index_xp_norm', 'pred_54_xp_norm', 'tbp_lv_color_std_mean_xp_norm', 'pred_58_xp_norm',
             'tbp_lv_stdLExt_basel_norm', 'pred_47_basel_norm', 'pred_19_basel_norm', 'tbp_lv_deltaLBnorm_xp_norm',
             'pred_9_xp_norm', 'pred_36', 'overall_color_difference_basel_norm', 'tbp_lv_Lext_basel_norm',
             'tbp_lv_eccentricity_basel_norm', 'pred_11_basel_norm', 'pred_29_basel_norm',
             'luminance_contrast_basel_norm', 'tbp_lv_y_future_norm', 'tbp_lv_deltaLB_basel_norm', 'pred_20_basel_norm',
             'pred_36_xp_norm', 'tbp_lv_perimeterMM_xp_norm', 'tbp_lv_L_basel_norm', 'mean_hue_difference_xp_norm',
             'pred_7_xp_norm', 'tbp_lv_eccentricity_xp_norm', 'pred_23_xp_norm', 'pred_19_xp_norm',
             'border_color_interaction_2_basel_norm', 'perimeter_to_area_ratio_xp_norm', 'onehot_45',
             'pred_16_patient_norm', 'consistency_color_basel_norm', 'tbp_lv_C_basel_norm',
             'border_length_ratio_basel_norm', 'color_consistency', 'tbp_lv_symm_2axis_angle_xp_norm',
             'lesion_shape_index_xp_norm', 'pred_22_basel_norm', 'pred_3_basel_norm', 'lesion_color_difference_xp_norm',
             'shape_color_consistency_basel_norm', 'pred_58_basel_norm', 'pred_30_xp_norm',
             'pred_48_basel_norm', 'tbp_lv_nevi_confidence_basel_norm', 'pred_29_patient_min_max_norm', 'pred_45_xp_norm', 'pred_23_basel_norm', 'pred_39_basel_norm',
             'tbp_lv_deltaLBnorm_basel_norm', 'pred_59_basel_norm', 'pred_52', 'tbp_lv_y_basel_norm', 'tbp_lv_deltaA',
             'pred_3_xp_norm', 'pred_21_basel_norm', 'pred_13_xp_norm',
             'pred_14_xp_norm', 'pred_33_xp_norm', 'index_age_size_symmetry_basel_norm', 'pred_56_xp_norm',
             'lesion_size_ratio_xp_norm', 'symmetry_perimeter_interaction_basel_norm',
             'age_size_symmetry_index_basel_norm', 'color_range_basel_norm', 'pred_15_xp_norm',
             'area_to_perimeter_ratio_basel_norm', 'tbp_lv_z_basel_norm', 'color_uniformity', 'pred_43_xp_norm',
             'onehot_9', 'clin_size_long_diam_mm', 'pred_51_patient_norm', 'hue_contrast_basel_norm', 'pred_8_xp_norm',
             'color_uniformity_xp_norm', 'pred_38', 'tbp_lv_stdLExt_xp_norm', 'count_per_patient_basel_norm',
             'pred_54_basel_norm', 'lesion_visibility_score_xp_norm', 'pred_12_xp_norm', 'normalized_lesion_size',
             'std_dev_contrast_xp_norm', 'tbp_lv_Cext_basel_norm', 'tbp_lv_norm_color_xp_norm', 'pred_12_basel_norm',
             'tbp_lv_Aext_basel_norm', 'pred_38_basel_norm',
             'color_consistency_basel_norm', 'position_distance_3d_basel_norm', 'tbp_lv_Lext_basel_norm', 'tbp_lv_Aext_basel_norm',
             'color_consistency_basel_norm', 'pred_57_basel_norm',
'size_age_interaction', 'pred_17_basel_norm', 'color_uniformity_future_norm', 'pred_15_basel_norm', 'perimeter_to_area_ratio_basel_norm', 'pred_52_basel_norm', 'pred_40_basel_norm']

xgb_avoid = ['pred_27_future_norm', 'pred_22_patient_min_max_norm', 'pred_40_future_norm', 'tbp_lv_areaMM2_patient_norm', 'pred_20_patient_norm', 'normalized_lesion_size_past_norm', 'size_color_contrast_ratio_future_norm', 'tbp_lv_deltaL_past_norm', 'pred_51_patient_norm', 'pred_32_patient_norm', 'tbp_lv_deltaLB_patient_min_max', 'onehot_8',
             'pred_55_future_norm', 'tbp_lv_z_patient_norm', 'tbp_lv_deltaA_patient_norm', 'onehot_25', 'symmetry_border_consistency', 'pred_0_patient_norm', 'pred_46_future_norm', 'color_consistency_past_norm', 'pred_7_patient_norm', 'onehot_1', 'symmetry_perimeter_interaction_past_norm', 'border_color_interaction_2_future_norm', 'pred_43_future_norm',
             'luminance_contrast_future_norm', 'tbp_lv_symm_2axis', 'pred_50_patient_min_max_norm', 'std_dev_contrast_patient_min_max', 'pred_3_future_norm', 'border_complexity_future_norm', 'border_length_ratio_patient_norm', 'tbp_lv_norm_color_future_norm', 'tbp_lv_perimeterMM_past_norm', 'pred_28_past_norm', 'border_color_interaction_2_patient_min_max',
             'pred_37_patient_norm', 'onehot_33', 'onehot_2', 'onehot_3', 'onehot_31', 'std_dev_contrast_past_norm', 'onehot_34', 'onehot_35', 'onehot_36', 'shape_complexity_index_past_norm', 'onehot_37', 'onehot_39', 'onehot_41', 'onehot_42', 'onehot_44', 'border_complexity_past_norm', 'normalized_lesion_size_patient_norm', 'consistency_color_patient_norm',
             'tbp_lv_norm_border_past_norm', 'color_shape_composite_index', 'lesion_shape_index_future_norm', 'count_per_patient_future_norm', 'age_approx_patient_norm', 'consistency_symmetry_border', 'onehot_30', 'border_color_interaction_past_norm', 'onehot_19', 'onehot_6', 'onehot_7', 'onehot_12', 'onehot_13', 'onehot_14', 'onehot_15', 'onehot_16', 'onehot_18',
             'onehot_20', 'onehot_29', 'onehot_21', 'onehot_22', 'count_per_patient_past_norm', 'onehot_23', 'onehot_24', 'onehot_26', 'onehot_27', 'onehot_28', 'onehot_17',
             'pred_19_past_norm', 'pred_45_future_norm', 'tbp_lv_stdLExt_patient_norm', 'shape_complexity_index',
             'tbp_lv_x_past_norm', 'tbp_lv_area_perim_ratio_patient_min_max', 'tbp_lv_norm_border', 'tbp_lv_norm_color',
             'tbp_lv_C_past_norm', 'pred_28_patient_min_max_norm', 'pred_10_patient_norm', 'pred_8',
             'pred_29_patient_norm', 'age_size_symmetry_index_future_norm', 'pred_32', 'tbp_lv_areaMM2_past_norm',
             'border_complexity_patient_norm', 'pred_28_patient_norm', 'hue_color_std_interaction',
             'border_color_interaction_2', 'shape_complexity_index_patient_min_max', 'onehot_0',
             'lesion_color_difference_past_norm', 'age_normalized_nevi_confidence_future_norm',
             'comprehensive_lesion_index_future_norm', 'lesion_shape_index_patient_norm', 'tbp_lv_Cext_patient_norm',
             'pred_8_patient_min_max_norm', 'pred_42_patient_min_max_norm', 'tbp_lv_area_perim_ratio',
             'pred_14_future_norm', 'shape_color_consistency', 'color_variance_ratio_patient_norm',
             'symmetry_border_consistency_future_norm', 'pred_5', 'pred_27_patient_min_max_norm',
             'luminance_contrast_past_norm', 'tbp_lv_symm_2axis_past_norm', 'symmetry_border_consistency_past_norm',
             'luminance_contrast_patient_norm', 'onehot_9', 'tbp_lv_perimeterMM_patient_norm', 'onehot_4', 'onehot_38',
             'tbp_lv_norm_border_patient_min_max', 'tbp_lv_deltaB_past_norm', 'log_lesion_area_future_norm',
             'consistency_color_future_norm', 'lesion_severity_index_patient_norm', 'border_length_ratio',
             'tbp_lv_stdL_future_norm', 'log_lesion_area',
             'shape_color_consistency_patient_min_max', 'mean_hue_difference_past_norm', 'lesion_size_ratio',
             'border_length_ratio_past_norm', 'tbp_lv_symm_2axis_patient_norm', 'tbp_lv_norm_border_future_norm',
             'pred_37_future_norm', 'color_consistency_patient_norm', 'pred_14_patient_norm',
             'area_to_perimeter_ratio_past_norm', 'pred_19_future_norm', 'age_size_symmetry_index_patient_norm',
             'tbp_lv_color_std_mean_future_norm', 'log_lesion_area_patient_norm',
             'border_color_interaction_2_patient_norm', 'pred_54_future_norm', 'pred_17_future_norm',
             'pred_30_future_norm', 'onehot_43',
             'tbp_lv_Lext_future_norm', 'tbp_lv_eccentricity_patient_norm', 'luminance_contrast',
             'pred_27_patient_norm', 'pred_10_past_norm', 'shape_complexity_index_future_norm',
             'hue_color_std_interaction_future_norm', 'onehot_40', 'pred_21_past_norm', 'pred_13_future_norm',
             'std_dev_contrast_future_norm', 'clin_size_long_diam_mm_patient_min_max',
             'tbp_lv_area_perim_ratio_past_norm', 'pred_41_past_norm', 'onehot_10',
             'tbp_lv_x_future_norm', 'pred_33_future_norm', 'pred_2_past_norm', 'tbp_lv_nevi_confidence_patient_norm',
             'pred_9', 'lesion_size_ratio_patient_norm', 'consistency_symmetry_border_future_norm',
             'tbp_lv_deltaL_future_norm', 'lesion_severity_index_past_norm', 'tbp_lv_symm_2axis_angle_patient_min_max',
             'pred_34', 'shape_complexity_index_patient_norm', 'color_variance_ratio',
             'tbp_lv_B_past_norm', 'pred_23_future_norm', 'comprehensive_lesion_index_patient_min_max',
             'tbp_lv_Bext_patient_min_max', 'tbp_lv_Hext_future_norm', 'pred_57_future_norm', 'pred_23',
             'tbp_lv_areaMM2_future_norm', 'pred_12_past_norm', 'lesion_shape_index_past_norm', 'pred_14_past_norm',
             'index_age_size_symmetry_patient_norm', 'tbp_lv_Bext_patient_norm', 'tbp_lv_stdL', 'pred_4_future_norm',
             'pred_5_patient_min_max_norm', 'pred_30_patient_min_max_norm', 'consistency_symmetry_border_past_norm',
             'symmetry_perimeter_interaction_future_norm', 'border_complexity', 'pred_35',
             'tbp_lv_perimeterMM_future_norm', 'size_color_contrast_ratio_patient_norm', 'onehot_46', 'onehot_32',
             'pred_48', 'volume_approximation_3d_patient_min_max', 'tbp_lv_area_perim_ratio_future_norm',
             'pred_13_patient_min_max_norm', 'pred_31_patient_norm', 'consistency_color', 'pred_42_patient_norm',
             'pred_44_patient_norm', 'pred_54_patient_min_max_norm', 'pred_39_patient_norm',
             'lesion_visibility_score_past_norm', 'shape_color_consistency_patient_norm', 'pred_43_patient_norm',
             'pred_8_past_norm', 'perimeter_to_area_ratio_future_norm', 'tbp_lv_norm_color_past_norm',
             'pred_43_past_norm', 'tbp_lv_minorAxisMM_future_norm', 'pred_39_future_norm',
             'pred_4_patient_min_max_norm', 'pred_17_patient_min_max_norm', 'pred_20_future_norm',
             'symmetry_perimeter_interaction_patient_min_max', 'pred_50_patient_norm', 'pred_29_future_norm',
             'pred_22_future_norm', 'tbp_lv_deltaL_patient_min_max', 'tbp_lv_x_patient_norm',
             'border_complexity_patient_min_max', 'shape_color_consistency_future_norm',
             'index_age_size_symmetry_past_norm', 'age_normalized_nevi_confidence_past_norm',
             'pred_18_patient_min_max_norm', 'tbp_lv_symm_2axis_angle', 'pred_24_patient_norm', 'pred_43',
             'tbp_lv_A_future_norm', 'pred_5_past_norm', 'lesion_orientation_3d', 'tbp_lv_deltaLB', 'tbp_lv_deltaL',
             'pred_37_past_norm', 'pred_36_future_norm', 'tbp_lv_deltaLB_past_norm',
             'luminance_contrast_patient_min_max', 'tbp_lv_color_std_mean',
             'tbp_lv_Bext_past_norm', 'area_to_perimeter_ratio_future_norm', 'normalized_lesion_size_future_norm',
             'pred_15_past_norm', 'tbp_lv_deltaLB_future_norm', 'pred_35_future_norm', 'tbp_lv_deltaLB_patient_norm',
             'tbp_lv_area_perim_ratio_patient_norm', 'pred_45_patient_min_max_norm', 'pred_38_patient_norm',
             'tbp_lv_norm_border_patient_norm', 'pred_5_patient_norm', 'pred_53_past_norm',
             'pred_31_patient_min_max_norm', 'border_color_interaction_2_past_norm',
             'lesion_color_difference_future_norm',
             'pred_13_past_norm', 'pred_34_patient_norm', 'size_age_interaction_past_norm',
             'lesion_severity_index_future_norm', 'pred_2', 'pred_52_patient_min_max_norm',
             'tbp_lv_eccentricity_patient_min_max', 'color_contrast_index_patient_min_max', 'pred_36_patient_norm',
             'consistency_symmetry_border_patient_norm', 'tbp_lv_eccentricity_future_norm',
             'lesion_size_ratio_patient_min_max', 'pred_0_future_norm', 'pred_11_future_norm', 'pred_8_patient_norm',
             'pred_37_patient_min_max_norm', 'lesion_shape_index',
             'pred_56_future_norm', 'pred_30_patient_norm', 'pred_59_patient_min_max_norm',
             'pred_47_patient_min_max_norm', 'onehot_11', 'pred_33_patient_norm', 'tbp_lv_Lext_patient_norm',
             'index_age_size_symmetry_future_norm', 'color_contrast_index_patient_norm',
             'lesion_shape_index_patient_min_max', 'pred_24_future_norm', 'pred_35_patient_min_max_norm', 'pred_51',
             'size_age_interaction', 'pred_31_future_norm', 'tbp_lv_stdL_patient_norm', 'count_future',
             'color_range_past_norm', 'pred_9_patient_norm', 'tbp_lv_symm_2axis_angle_patient_norm',
             'consistency_symmetry_border_patient_min_max', 'pred_57_past_norm', 'pred_48_past_norm',
             'comprehensive_lesion_index',
             'tbp_lv_deltaLBnorm_future_norm', 'pred_33_patient_min_max_norm', 'pred_0', 'pred_54',
             'tbp_lv_Cext_future_norm', 'pred_4_patient_norm', 'hue_contrast_future_norm',
             'normalized_lesion_size_patient_min_max', 'pred_55_patient_norm', 'pred_18',
             'tbp_lv_z', 'pred_48_future_norm', 'pred_31_past_norm', 'tbp_lv_minorAxisMM_past_norm',
             'tbp_lv_deltaLBnorm_patient_norm', 'pred_48_patient_norm', 'pred_13_patient_norm',
             'pred_23_patient_min_max_norm', 'pred_24_patient_min_max_norm', 'tbp_lv_symm_2axis_patient_min_max',
             'tbp_lv_z_future_norm', 'age_size_symmetry_index_past_norm', 'pred_46_patient_min_max_norm',
             'color_contrast_index', 'pred_13', 'size_color_contrast_ratio_patient_min_max',
             'pred_32_patient_min_max_norm', 'tbp_lv_Bext_future_norm', 'pred_57_patient_norm',
             'age_size_symmetry_index_patient_min_max', 'pred_1_past_norm', 'tbp_lv_stdL_past_norm',
             'tbp_lv_Cext_patient_min_max', 'pred_1_patient_min_max_norm', 'tbp_lv_eccentricity_past_norm',
             'pred_22_patient_norm', 'border_length_ratio_patient_min_max', 'tbp_lv_deltaLBnorm_patient_min_max',
             'pred_45', 'pred_34_patient_min_max_norm', 'pred_31', 'pred_42_future_norm', 'pred_4_past_norm',
             'tbp_lv_stdL_patient_min_max', 'tbp_lv_C_patient_norm', 'pred_51_future_norm', 'tbp_lv_C_patient_min_max',
             'pred_17_patient_norm', 'tbp_lv_A_patient_min_max', 'pred_14', 'pred_54_patient_norm', 'pred_50',
             'pred_55_patient_min_max_norm', 'mean_hue_difference_future_norm', 'pred_45_patient_norm',
             'color_range_patient_min_max', 'tbp_lv_Cext_past_norm', 'age_size_symmetry_index',
             'tbp_lv_nevi_confidence_future_norm', 'color_contrast_index_future_norm', 'pred_41_future_norm',
             'pred_51_patient_min_max_norm', 'pred_4', 'perimeter_to_area_ratio_past_norm',
             'color_shape_composite_index_future_norm', 'pred_8_future_norm', 'tbp_lv_Hext_patient_norm',
             'volume_approximation_3d_future_norm', 'pred_58', 'tbp_lv_L', 'pred_40_patient_norm', 'pred_32_past_norm',
             'tbp_lv_stdLExt_patient_min_max', 'pred_1_patient_norm', 'pred_34_future_norm',
             'lesion_color_difference_patient_min_max', 'pred_14_patient_min_max_norm', 'pred_3_patient_norm',
             'pred_30', 'pred_40_past_norm', 'symmetry_border_consistency_patient_norm', 'tbp_lv_L_past_norm',
             'pred_47_patient_norm', 'log_lesion_area_past_norm', 'pred_55_past_norm',
             'size_color_contrast_ratio_past_norm', 'pred_19_patient_norm', 'lesion_orientation_3d_patient_min_max',
             'pred_22', 'pred_35_patient_norm', 'pred_42', 'pred_51_past_norm', 'pred_50_future_norm',
             'pred_53_future_norm', 'pred_7_future_norm', 'pred_52', 'pred_40',
             'symmetry_border_consistency_patient_min_max', 'pred_38_patient_min_max_norm', 'pred_32_future_norm',
             'pred_49_patient_norm', 'pred_10_future_norm', 'pred_30_past_norm', 'pred_0_patient_min_max_norm',
             'symmetry_perimeter_interaction_patient_norm', 'tbp_lv_C_future_norm', 'pred_35_past_norm',
             'tbp_lv_symm_2axis_angle_future_norm', 'pred_1_future_norm', 'pred_7_patient_min_max_norm',
             'pred_49_future_norm', 'lesion_severity_index_patient_min_max', 'log_lesion_area_patient_min_max',
             'tbp_lv_z_patient_min_max', 'pred_48_patient_min_max_norm', 'pred_23_past_norm',
             'consistency_color_past_norm', 'tbp_lv_color_std_mean_patient_min_max',
             'shape_color_consistency_past_norm', 'age_normalized_nevi_confidence_2_past_norm',
             'tbp_lv_x_patient_min_max', 'color_variance_ratio_past_norm', 'tbp_lv_stdLExt_future_norm',
             'pred_11_patient_min_max_norm', 'pred_43_patient_min_max_norm', 'size_color_contrast_ratio',
             'tbp_lv_stdLExt_past_norm', 'tbp_lv_Aext_past_norm', 'lesion_size_ratio_future_norm',
             'tbp_lv_symm_2axis_angle_past_norm', 'pred_2_patient_norm', 'tbp_lv_radial_color_std_max',
             'border_color_interaction_future_norm', 'pred_45_past_norm', 'pred_47_past_norm',
             'color_consistency_future_norm', 'pred_25', 'pred_9_future_norm', 'tbp_lv_minorAxisMM_patient_min_max',
             'pred_10', 'comprehensive_lesion_index_patient_norm', 'pred_6_patient_min_max_norm',
             'tbp_lv_Lext_patient_min_max', 'pred_58_future_norm', 'pred_7',
             'age_normalized_nevi_confidence_patient_min_max', 'lesion_severity_index', 'tbp_lv_Aext_future_norm',
             'border_color_interaction_patient_norm', 'tbp_lv_Cext', 'pred_29_past_norm', 'tbp_lv_eccentricity',
             'overall_color_difference_patient_min_max', 'lesion_orientation_3d_future_norm', 'pred_15',
             'overall_color_difference', 'pred_0_past_norm', 'pred_9_past_norm', 'tbp_lv_deltaA_future_norm',
             'hue_contrast_patient_norm', 'tbp_lv_stdLExt', 'pred_40_patient_min_max_norm',
             'color_consistency_patient_min_max', 'pred_28_future_norm', 'pred_29_patient_min_max_norm',
             'tbp_lv_deltaB', 'pred_12_future_norm', 'pred_59', 'pred_46_patient_norm', 'tbp_lv_C',
             'pred_2_patient_min_max_norm', 'lesion_color_difference', 'pred_56', 'pred_46_past_norm',
             'hue_color_std_interaction_past_norm', 'index_age_size_symmetry', 'tbp_lv_color_std_mean_past_norm',
             'pred_52_patient_norm', 'pred_21_future_norm', 'pred_11_past_norm', 'tbp_lv_deltaB_future_norm',
             'pred_57_patient_min_max_norm', 'lesion_visibility_score_patient_min_max', 'pred_12_patient_min_max_norm',
             'pred_37', 'tbp_lv_x', 'pred_44', 'pred_27_past_norm', 'tbp_lv_A_past_norm', 'tbp_lv_deltaA_past_norm',
             'pred_56_patient_min_max_norm', 'pred_2_future_norm', 'color_variance_ratio_patient_min_max',
             'color_variance_ratio_future_norm', 'pred_41_patient_norm', 'pred_3_patient_min_max_norm',
             'tbp_lv_areaMM2', 'color_uniformity_patient_norm', 'color_shape_composite_index_past_norm',
             'tbp_lv_y_past_norm', 'pred_19', 'pred_18_future_norm', 'pred_38_past_norm', 'pred_44_past_norm',
             'pred_27', 'pred_56_patient_norm', 'pred_52_future_norm', 'tbp_lv_z_past_norm',
             'pred_9_patient_min_max_norm', 'pred_11_patient_norm', 'pred_39_past_norm', 'pred_6_future_norm',
             'pred_25_patient_norm', 'tbp_lv_Lext', 'pred_17', 'color_uniformity_past_norm', 'pred_21_patient_norm',
             'pred_55', 'pred_33_past_norm', 'volume_approximation_3d_patient_norm', 'pred_39_patient_min_max_norm',
             'tbp_lv_L_patient_norm', 'color_uniformity_future_norm', 'pred_53_patient_min_max_norm', 'pred_53',
             'color_shape_composite_index_patient_min_max', 'pred_15_future_norm', 'pred_15_patient_min_max_norm',
             'tbp_lv_Bext', 'age_normalized_nevi_confidence_patient_norm', 'pred_5_future_norm',
             'tbp_lv_L_patient_min_max', 'area_to_perimeter_ratio', 'pred_44_patient_min_max_norm',
             'pred_10_patient_min_max_norm', 'tbp_lv_B_future_norm', 'tbp_lv_deltaLBnorm_past_norm',
             'pred_22_past_norm', 'color_consistency', 'color_contrast_index_past_norm',
             'position_distance_3d_past_norm', 'tbp_lv_Aext_patient_min_max', 'symmetry_perimeter_interaction',
             'border_length_ratio_future_norm', 'pred_29', 'lesion_size_ratio_past_norm', 'pred_23_patient_norm',
             'pred_3', 'color_range_future_norm', 'perimeter_to_area_ratio_patient_min_max', 'pred_6_patient_norm',
             'pred_50_past_norm', 'pred_49_patient_min_max_norm', 'pred_36_past_norm', 'pred_44_future_norm',
             'tbp_lv_symm_2axis_future_norm', 'pred_49', 'pred_25_patient_min_max_norm', 'pred_38_future_norm',
             'pred_19_patient_min_max_norm', 'pred_7_past_norm', 'pred_47_future_norm', 'pred_41_patient_min_max_norm',
             'pred_39', 'tbp_lv_deltaB_patient_norm', 'pred_54_past_norm', 'tbp_lv_Lext_past_norm',
             'area_to_perimeter_ratio_patient_norm', 'pred_25_future_norm',
             'pred_41', 'lesion_color_difference_patient_norm', 'size_age_interaction_patient_min_max', 'consistency_color_patient_min_max', 'pred_59_past_norm',
             'std_dev_contrast_patient_norm', 'lesion_orientation_3d_past_norm', 'pred_20', 'pred_58_patient_norm',
             'count_per_patient', 'age_approx_past_norm', 'hue_color_std_interaction_patient_min_max',
             'tbp_lv_L_future_norm', 'pred_16', 'tbp_lv_y', 'pred_56_past_norm', 'pred_47',
             'lesion_visibility_score_future_norm', 'tbp_lv_nevi_confidence_past_norm', 'pred_58_patient_min_max_norm',
             'pred_59_future_norm', 'pred_18_past_norm', 'tbp_lv_B', 'pred_38', 'comprehensive_lesion_index_past_norm',
             'pred_17_past_norm', 'tbp_lv_Aext', 'tbp_lv_deltaA_patient_min_max', 'overall_color_difference_past_norm',
             'color_asymmetry_index_patient_norm', 'age_normalized_nevi_confidence_2', 'pred_25_past_norm',
             'tbp_lv_y_future_norm', 'count_past', 'volume_approximation_3d', 'tbp_lv_Hext', 'pred_24_past_norm',
             'pred_21_patient_min_max_norm', 'color_shape_composite_index_patient_norm', 'pred_36_patient_min_max_norm',
             'hue_contrast_past_norm', 'pred_6', 'pred_53_patient_norm', 'tbp_lv_Hext_patient_min_max',
             'tbp_lv_A_patient_norm', 'pred_11', 'pred_20_patient_min_max_norm', 'pred_3_past_norm', 'pred_21',
             'pred_15_patient_norm', 'pred_42_past_norm', 'tbp_lv_nevi_confidence_patient_min_max', 'pred_57',
             'pred_20_past_norm', 'tbp_lv_Aext_patient_norm', 'color_range', 'pred_24', 'pred_18_patient_norm',
             'color_range_patient_norm', 'color_uniformity_patient_min_max', 'lesion_orientation_3d_patient_norm',
             'tbp_lv_deltaB_patient_min_max', 'color_asymmetry_index_past_norm', 'lesion_visibility_score',
             'tbp_lv_deltaA', 'volume_approximation_3d_past_norm', 'pred_49_past_norm',
             'tbp_lv_radial_color_std_max_patient_min_max', 'tbp_lv_A', 'pred_33',
             'tbp_lv_radial_color_std_max_past_norm', 'tbp_lv_deltaL_patient_norm',
             'perimeter_to_area_ratio_patient_norm', 'tbp_lv_areaMM2_patient_min_max',
             'tbp_lv_color_std_mean_patient_norm',
'mean_hue_difference_patient_min_max', 'pred_16_patient_norm', 'position_distance_3d', 'clin_size_long_diam_mm_past_norm', 'pred_16_patient_min_max_norm',
'pred_16_future_norm', 'pred_52_past_norm', 'clin_size_long_diam_mm_future_norm', 'tbp_lv_nevi_confidence',
'perimeter_to_area_ratio', 'color_variance_ratio_xp_norm', 'tbp_lv_nevi_confidence_xp_norm', 'symmetry_perimeter_interaction_xp_norm', 'tbp_lv_color_std_mean_basel_norm', 'pred_23_xp_norm', 'tbp_lv_stdL_basel_norm', 'shape_complexity_index_xp_norm', 'tbp_lv_color_std_mean_xp_norm', 'border_length_ratio_xp_norm',
'pred_18_basel_norm', 'pred_36_xp_norm', 'pred_32_xp_norm', 'pred_48_xp_norm', 'tbp_lv_areaMM2_basel_norm', 'pred_42_basel_norm', 'tbp_lv_eccentricity_xp_norm',
'lesion_orientation_3d_xp_norm', 'pred_1_xp_norm', 'pred_8_basel_norm', 'shape_complexity_index_basel_norm', 'tbp_lv_area_perim_ratio_basel_norm', 'tbp_lv_symm_2axis_xp_norm', 'pred_4_xp_norm', 'pred_41_xp_norm', 'tbp_lv_area_perim_ratio_xp_norm', 'pred_52_xp_norm', 'pred_27_basel_norm', 'pred_54_xp_norm', 'tbp_lv_deltaB_xp_norm', 'pred_45_xp_norm', 'pred_54_basel_norm', 'age_size_symmetry_index_basel_norm', 'lesion_shape_index_xp_norm', 'size_color_contrast_ratio_basel_norm', 'pred_42_xp_norm', 'comprehensive_lesion_index_xp_norm', 'pred_30_xp_norm', 'border_color_interaction_basel_norm', 'hue_color_std_interaction_basel_norm', 'pred_31_xp_norm', 'tbp_lv_minorAxisMM', 'pred_30_basel_norm', 'tbp_lv_norm_border_xp_norm', 'std_dev_contrast_basel_norm',
'pred_38_basel_norm', 'age_normalized_nevi_confidence_basel_norm', 'pred_7_xp_norm', 'pred_39_basel_norm', 'pred_48_basel_norm', 'consistency_color_basel_norm', 'pred_52_basel_norm', 'color_range_basel_norm', 'border_color_interaction', 'tbp_lv_radial_color_std_max_basel_norm', 'color_variance_ratio_basel_norm', 'tbp_lv_symm_2axis_basel_norm', 'tbp_lv_perimeterMM', 'tbp_lv_deltaL_xp_norm', 'color_shape_composite_index_xp_norm', 'pred_9_xp_norm', 'perimeter_to_area_ratio_basel_norm', 'pred_35_basel_norm', 'tbp_lv_deltaLB_basel_norm', 'pred_35_xp_norm', 'tbp_lv_perimeterMM_basel_norm',
'pred_22_basel_norm', 'border_complexity_xp_norm', 'border_color_interaction_2_basel_norm', 'tbp_lv_norm_color_basel_norm', 'pred_25_basel_norm',
'tbp_lv_deltaLB_xp_norm', 'pred_31_basel_norm', 'pred_22_xp_norm', 'symmetry_border_consistency_xp_norm', 'normalized_lesion_size', 'pred_34_basel_norm', 'pred_51_xp_norm', 'pred_0_basel_norm', 'pred_49_basel_norm', 'shape_color_consistency_basel_norm', 'pred_50_xp_norm', 'consistency_symmetry_border_xp_norm', 'pred_9_basel_norm', 'pred_45_basel_norm', 'size_age_interaction_patient_norm', 'age_approx', 'age_size_symmetry_index_xp_norm', 'tbp_lv_C_xp_norm',
'lesion_color_difference_xp_norm', 'pred_18_xp_norm', 'age_approx_xp_norm', 'luminance_contrast_basel_norm', 'tbp_lv_symm_2axis_angle_basel_norm', 'pred_4_basel_norm', 'pred_50_basel_norm',
'onehot_5', 'pred_36_basel_norm', 'color_asymmetry_index_future_norm', 'age_normalized_nevi_confidence', 'tbp_lv_B_patient_norm', 'border_length_ratio_basel_norm', 'position_distance_3d_future_norm', 'lesion_severity_index_basel_norm', 'pred_17_basel_norm', 'pred_2_basel_norm', 'pred_58_past_norm', 'position_distance_3d_patient_norm', 'pred_6_basel_norm', 'pred_16_past_norm', 'std_dev_contrast', 'lesion_orientation_3d_basel_norm', 'age_approx_basel_norm', 'pred_28_basel_norm', 'tbp_lv_deltaB_basel_norm', 'pred_7_basel_norm', 'tbp_lv_A_basel_norm', 'age_normalized_nevi_confidence_2_patient_min_max', 'pred_12_basel_norm', 'pred_6_past_norm', 'tbp_lv_y_basel_norm', 'tbp_lv_norm_color_patient_norm', 'tbp_lv_nevi_confidence_basel_norm', 'count_per_patient_basel_norm', 'pred_16_basel_norm', 'pred_14_basel_norm', 'tbp_lv_x_basel_norm', 'tbp_lv_Lext_basel_norm', 'pred_20_basel_norm', 'size_age_interaction_basel_norm', 'pred_3_basel_norm', 'pred_23_basel_norm', 'volume_approximation_3d_basel_norm', 'pred_21_basel_norm', 'pred_56_basel_norm', 'pred_15_basel_norm', 'hue_contrast_patient_min_max', 'symmetry_border_consistency_basel_norm', 'pred_44_basel_norm', 'pred_58_basel_norm', 'pred_29_basel_norm', 'tbp_lv_H_basel_norm', 'color_asymmetry_index', 'tbp_lv_B_basel_norm', 'tbp_lv_deltaA_basel_norm', 'pred_1_basel_norm', 'tbp_lv_C_basel_norm', 'tbp_lv_z_basel_norm', 'pred_32_basel_norm', 'pred_24_basel_norm', 'border_complexity_basel_norm', 'color_asymmetry_index_patient_min_max', 'pred_33_basel_norm', 'pred_11_basel_norm', 'tbp_lv_eccentricity_basel_norm', 'lesion_shape_index_basel_norm', 'tbp_lv_Hext_basel_norm', 'color_consistency_basel_norm', 'pred_10_basel_norm', 'pred_57_basel_norm', 'pred_59_basel_norm', 'tbp_lv_Bext_basel_norm', 'tbp_lv_Cext_basel_norm', 'pred_37_basel_norm', 'pred_43_basel_norm', 'overall_color_difference_basel_norm', 'pred_53_basel_norm', 'pred_55_basel_norm', 'pred_41_basel_norm', 'color_shape_composite_index_basel_norm', 'tbp_lv_deltaL_basel_norm', 'symmetry_perimeter_interaction_basel_norm', 'pred_46_basel_norm', 'pred_5_basel_norm', 'mean_hue_difference_basel_norm', 'color_contrast_index_basel_norm', 'tbp_lv_L_basel_norm', 'hue_contrast', 'pred_40_basel_norm', 'consistency_symmetry_border_basel_norm', 'tbp_lv_stdLExt_basel_norm', 'tbp_lv_norm_border_basel_norm', 'pred_51_basel_norm', 'position_distance_3d_basel_norm', 'tbp_lv_Aext_basel_norm', 'pred_13_basel_norm', 'lesion_size_ratio_basel_norm',
             ]

lgb_feats = [col for col in feature_cols if col not in lgb_avoid]
cat_feats = [col for col in feature_cols if col not in cat_avoid]
xgb_feats = [col for col in feature_cols if col not in xgb_avoid]


print("lgb feats:", len(lgb_feats))
print("cat feats:", len(cat_feats))
print("xgb feats:", len(xgb_feats))

lgb_params = {
    'objective':        'binary',
    'verbosity':        -1,
    'num_iterations':   200,
    'boosting_type':    'gbdt',
    'random_state':     seed,
    'lambda_l1':        0.08758718919397321,
    'lambda_l2':        0.0039689175176025465,
    'learning_rate':    0.03231007103195577,
    'max_depth':        4,
    'num_leaves':       103,
    'colsample_bytree': 0.8329551585827726,
    'colsample_bynode': 0.4025961355653304,
    'bagging_fraction': 0.7738954452473223,
    'bagging_freq':     4,
    'min_data_in_leaf': 85,
    'scale_pos_weight': 2.7984184778875543,
}


lgb_model = Pipeline([
   ('sampler_1', RandomOverSampler(sampling_strategy= 0.003 , random_state=seed)),
    ('sampler_2', RandomUnderSampler(sampling_strategy=sampling_ratio , random_state=seed)),
    ('classifier', lgb.LGBMClassifier(**lgb_params)),
])

#[I 2024-08-17 14:55:22,860] Trial 98 finished with value: 0.1710058144245913 and parameters: {'learning_rate': 0.08341853356925374, 'max_depth': 5, 'l2_leaf_reg': 6.740520715798379, 'subsample': 0.42402936337409075, 'colsample_bylevel': 0.9860546885166512, 'min_data_in_leaf': 52, 'scale_pos_weight': 2.6227279486021153}. Best is trial 98 with value: 0.1710058144245913.

cb_params = {
    'loss_function':     'Logloss',
    'iterations':        200,
    'verbose':           False,
    'random_state':      seed,
    'max_depth':         7,
    'learning_rate':     0.06936242010150652,
    'scale_pos_weight':  2.6149345838209532,
    'l2_leaf_reg':       6.216113851699493,
    'subsample':         0.6249261779711819,
    'min_data_in_leaf':  24,
    'cat_features':      [x for x in new_cat_cols if x in cat_feats],
}

cat_model = Pipeline([
   ('sampler_1', RandomOverSampler(sampling_strategy= 0.003 , random_state=seed)),
    ('sampler_2', RandomUnderSampler(sampling_strategy=sampling_ratio , random_state=seed)),
    ('classifier', cb.CatBoostClassifier(**cb_params)),
])

xgb_params = {
    'enable_categorical': True,
    'tree_method':        'hist',
    'random_state':       seed,
    'learning_rate':      0.08501257473292347,
    'lambda':             8.879624125465703,
    'alpha':              0.6779926606782505,
    'max_depth':          6,
    'subsample':          0.6012681388711075,
    'colsample_bytree':   0.8437772277074493,
    'colsample_bylevel':  0.5476090898823716,
    'colsample_bynode':   0.9928601203635129,
    'scale_pos_weight':   3.29440313334688,
}

xgb_model = Pipeline([
   ('sampler_1', RandomOverSampler(sampling_strategy=0.003, random_state=seed)),
    ('sampler_2', RandomUnderSampler(sampling_strategy=sampling_ratio , random_state=seed)),
    ('classifier', xgb.XGBClassifier(**xgb_params)),
])

estimator = VotingClassifier([
    ('lgb', lgb_model), ('xgb', xgb_model), ('cb', cat_model)
], voting='soft')


def calculate_normalizations(df, columns):
    results = {}

    # Sort the DataFrame by patient_id and age_approx
    df_fut = df.sort_values(by=['patient_id', 'age_approx'])
    df_rev = df.sort_values(by=['patient_id', 'age_approx'], ascending=[True, False])

    # Group by patient_id and calculate cumulative counts for past and future
    grouped = df_fut.groupby('patient_id')
    rev_grouped = df_rev.groupby('patient_id')

    results['isic_id'] = df['isic_id']

    # Loop over each column to normalize
    for col in columns:
        # Cumulative statistics for past rows
        cum_mean_past = grouped[col].expanding().mean().shift(0).fillna(0).reset_index(level=0, drop=True)
        cum_std_past = grouped[col].expanding().std().shift(0).fillna(0).reset_index(level=0, drop=True)

        # Reverse cumulative statistics for future rows
        cum_mean_future = rev_grouped[col].expanding().mean().shift(0).fillna(0).reset_index(level=0, drop=True)
        cum_std_future = rev_grouped[col].expanding().std().shift(0).fillna(0).reset_index(level=0, drop=True)

        cum_count_past = grouped[col].expanding().count().shift(0).fillna(0).reset_index(level=0, drop=True)
        results['count_past'] = cum_count_past

        cum_count_future = rev_grouped[col].expanding().count().shift(0).fillna(0).reset_index(level=0, drop=True)
        results['count_future'] = cum_count_future

        # Normalize columns
        results[f'{col}_future_norm'] = (df[col] - cum_mean_future) / (cum_std_future + 1e-4)
        results[f'{col}_past_norm'] = (df[col] - cum_mean_past) / (cum_std_past + 1e-4)

    return pd.DataFrame(results)


construct_preds = False

if construct_preds:
    path = "../multimodal/models/SSL_60_0.15_benign_validated_SAM_adaptive_lr_0.0002_64_vcreg_4085689_b_128_dp_0.1.bin"
    # path = "../multimodal/models/SSL_120_batch_norm_SAM_adaptive_lr_0.0002_64_vcreg_4162549_b_128_dp_0.1.bin"
    print(path)

    model = EfficientNet_pretrained_linear(60).cuda()
    model.load_state_dict(torch.load(path))
    model.eval()

    df_train_images = pd.read_csv("../isic-2024-challenge/train-metadata.csv")
    train_images_paths = sorted(glob.glob(f"{TRAIN_DIR}/*.jpg"))
    train_images_paths = [path.replace('\\', '/') for path in train_images_paths]
    df_train_images['file_path'] = df_train_images['isic_id'].apply(get_train_file_path)
    df_images = df_train_images[df_train_images["file_path"].isin(train_images_paths)].reset_index(drop=True)

    val_dataset = ISIC_multimodal_ssl_valid(df_images, transforms=data_transforms['valid'])

    val_loader = DataLoader(val_dataset, batch_size=CONFIG['valid_batch_size'],
                            shuffle=False, pin_memory=True, drop_last=False)

    test_preds = []
    isic_ids = []
    with torch.no_grad():
        bar = tqdm(enumerate(val_loader), total=len(val_loader))
        for step, data in bar:
            batch_isic_ids = list(data['isic_id'])
            images = data['images_1'].to(CONFIG["device"], dtype=torch.float)

            batch_size = images.size(0)

            model_output_1 = model(images)
            model_output = model_output_1.detach().cpu().numpy()

            test_preds.append(model_output)
            isic_ids += batch_isic_ids

    test_preds = np.concatenate(test_preds)
    print("preds shape", test_preds.shape)

    print("isic ids:", len(isic_ids))
    test_preds_df = pd.DataFrame(test_preds, columns=pred_cols)
    test_preds_df.insert(0, 'isic_id', isic_ids)
    test_preds_df.to_csv("ssl_60_0.15_validated_benign_64.csv", index=False)
    print("preds df:", test_preds_df.shape)
else:
    test_preds_df = pd.read_csv("ssl_60_0.15_validated_benign_64.csv")
    # test_preds_df = pd.read_csv("ssl_test_preds.csv.csv")


df_train = pd.merge(df_train, test_preds_df, on='isic_id', how='inner')


for pred_col in pred_cols:
    df_train[f'{pred_col}_patient_norm'] = df_train.groupby('patient_id')[pred_col].transform(lambda x: (x - x.mean()) / (x.std() + err))

for pred_col in pred_cols:
    df_train[f'{pred_col}_patient_min_max_norm'] = df_train.groupby('patient_id')[pred_col].transform(lambda x: (x - x.min()) / (x.max() - x.min() + err))


for col in columns_to_normalize:
    df_train[f'{col}_basel_norm'] = df_train.groupby(['attribution', 'anatom_site_general', 'tbp_tile_type'])[col].transform(
        lambda x: (x - x.mean()) / (x.std() + err))

# for col in columns_to_normalize:
#     new_feats.append(f'{col}_xp_norm')
#     df_train[f'{col}_xp_norm'] = df_train.groupby('onehot_9')[col].transform(
#         lambda x: (x - x.mean()) / (x.std() + err))

train_res = calculate_normalizations(df_train, columns_to_normalize)
df_train = pd.merge(df_train, train_res, on='isic_id', how='inner')

X = df_train[feature_cols]
y = df_train[target_col]

groups = df_train[group_col]

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)

sum_val = 0
# Manually perform cross-validation
for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):

    # Split data
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    lgb_X = X_train[lgb_feats]
    lgb_model.fit(lgb_X, y_train)

    cat_X = X_train[cat_feats]
    cat_model.fit(cat_X, y_train)

    xgb_X = X_train[xgb_feats]
    xgb_model.fit(xgb_X, y_train)

    X_val_lgb = X_val[lgb_feats]
    X_val_xgb = X_val[xgb_feats]
    X_val_cat = X_val[cat_feats]

    lgb_preds = lgb_model.predict_proba(X_val_lgb)[:, 1]  # Use predict_proba for soft voting
    xgb_preds = xgb_model.predict_proba(X_val_xgb)[:, 1]
    cat_preds = cat_model.predict_proba(X_val_cat)[:, 1]

    # Predict on the validation data
    lgb_score = custom_metric(lgb_preds, y_val)
    xgb_score = custom_metric(xgb_preds, y_val)
    cat_score = custom_metric(cat_preds, y_val)

    val_preds = lgb_preds * xgb_preds * cat_preds
    val_score = custom_metric(val_preds, y_val)

    sum_val += val_score
    print(f"Fold {fold + 1} - Validation Score: {val_score}", f"lgb: {lgb_score}", f"xgb: {xgb_score}", f"cat: {cat_score}")

print(f"Average Validation Score: {sum_val / 5}")
# Calculate OOF score using the custom metric

DO_TUNING = False

if DO_TUNING:
    # LightGBM
    start_time = time.time()
    study_lgb = optuna.create_study(direction='maximize', sampler=TPESampler(seed=seed))
    #study_lgb.optimize(lgb_objective, n_trials=100)
    study_lgb.optimize(lambda trial: lgb_objective(trial, sampling_ratio, df_train, feature_cols, target_col, group_col), n_trials=200)
    end_time = time.time()
    elapsed_time_lgb = end_time - start_time
    print(f"LightGBM tuning took {elapsed_time_lgb:.2f} seconds.")

    # CatBoost
    start_time = time.time()
    study_cb = optuna.create_study(direction='maximize', sampler=TPESampler(seed=seed))
    #study_cb.optimize(cb_objective, n_trials=100)
    study_cb.optimize(lambda trial: cb_objective(trial, seed, sampling_ratio, df_train, feature_cols, target_col, group_col, new_cat_cols), n_trials=200)
    end_time = time.time()
    elapsed_time_cb = end_time - start_time
    print(f"CatBoost tuning took {elapsed_time_cb:.2f} seconds.")

    # XGBoost
    start_time = time.time()
    study_xgb = optuna.create_study(direction='maximize', sampler=TPESampler(seed=seed))
    #study_xgb.optimize(xgb_objective, n_trials=100)
    study_xgb.optimize(lambda trial: xgb_objective(trial, seed, sampling_ratio, df_train, feature_cols, target_col, group_col), n_trials=200)
    end_time = time.time()
    elapsed_time_xgb = end_time - start_time
    print(f"XGBoost tuning took {elapsed_time_xgb:.2f} seconds.")

    # Print best parameters for each study
    print("Best LGBM trial:", study_lgb.best_trial)
    print("Best CatBoost trial:", study_cb.best_trial)
    print("Best XGBoost trial:", study_xgb.best_trial)


DO_FEATURE_IMPORTANCE_MODELS = True

if DO_FEATURE_IMPORTANCE_MODELS:
    lgb_X = df_train[lgb_feats]
    lgb_model.fit(lgb_X, y)

    xgb_X = df_train[xgb_feats]
    xgb_model.fit(xgb_X, y)

    cat_X = df_train[cat_feats]
    cat_model.fit(cat_X, y)

    lgb_classifier = lgb_model.named_steps['classifier']
    cat_classifier = cat_model.named_steps['classifier']
    xgb_classifier = xgb_model.named_steps['classifier']

    # Extract feature importances
    lgb_importances = lgb_classifier.feature_importances_
    feature_importances_lgb = pd.DataFrame({
        'feature': lgb_X.columns,  # Assuming X_train is a DataFrame with named columns
        'importance': lgb_importances
    }).sort_values(by='importance', ascending=False)
    print(feature_importances_lgb.to_string())

    zero_importance_lgb = feature_importances_lgb[feature_importances_lgb['importance'] <= 10]['feature'].to_list()
    print("zero importance lgb", zero_importance_lgb)
    print()

    cat_importances = cat_classifier.feature_importances_
    feature_importances_cat = pd.DataFrame({
        'feature': cat_X.columns,  # Assuming X_train is a DataFrame with named columns
        'importance': cat_importances
    }).sort_values(by='importance', ascending=False)
    print(feature_importances_cat.to_string())

    zero_importance_cat = feature_importances_cat[feature_importances_cat['importance'] <= 0.3]['feature'].to_list()
    print("zero importance cat", zero_importance_cat)
    print()

    xgb_importances = xgb_classifier.feature_importances_
    feature_importances_xgb = pd.DataFrame({
        'feature': xgb_X.columns,  # Assuming X_train is a DataFrame with named columns
        'importance': xgb_importances
    }).sort_values(by='importance', ascending=False)
    print(feature_importances_xgb.to_string())

    zero_importance_xgb = feature_importances_xgb[feature_importances_xgb['importance'] <= 0.005]['feature'].to_list()
    print("zero importance xgb", zero_importance_xgb)


