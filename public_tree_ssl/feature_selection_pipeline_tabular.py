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

columns_to_normalize = num_cols + new_num_cols + special_cols

future_norm_cols = [f'{col}_future_norm' for col in columns_to_normalize]
past_norm_cols = [f'{col}_past_norm' for col in columns_to_normalize]
future_count_cols = ['count_future']
past_count_cols = ['count_past']

feature_cols = num_cols + new_num_cols + norm_cols + special_cols + min_max_cols

df_train = read_data(train_path, err, num_cols, cat_cols, new_num_cols)
df_test = read_data(test_path, err, num_cols, cat_cols, new_num_cols)
df_subm = pd.read_csv(subm_path, index_col=id_col)

df_train, df_test, new_cat_cols = preprocess(df_train, df_test, cat_cols)

feature_cols += new_cat_cols

feature_cols += future_norm_cols
feature_cols += past_norm_cols
feature_cols += future_count_cols
feature_cols += past_count_cols

feature_cols += [f'{col}_attri_norm' for col in columns_to_normalize]
feature_cols += [f'{col}_tile_type_norm' for col in columns_to_normalize]
feature_cols += [f'{col}_anatom_norm' for col in columns_to_normalize]

feature_cols += [f'{col}_multi_min_max' for col in columns_to_normalize]

print("feature_cols", len(feature_cols))


lgb_feats =  ['clin_size_long_diam_mm', 'tbp_lv_A', 'tbp_lv_Aext', 'tbp_lv_B', 'tbp_lv_Bext', 'tbp_lv_C', 'tbp_lv_Cext', 'tbp_lv_H', 'tbp_lv_Hext', 'tbp_lv_L', 'tbp_lv_Lext', 'tbp_lv_areaMM2', 'tbp_lv_deltaA', 'tbp_lv_deltaB', 'tbp_lv_deltaL', 'tbp_lv_deltaLBnorm', 'tbp_lv_eccentricity', 'tbp_lv_minorAxisMM', 'tbp_lv_nevi_confidence', 'tbp_lv_stdLExt', 'tbp_lv_symm_2axis_angle', 'tbp_lv_x', 'tbp_lv_y', 'tbp_lv_z', 'lesion_size_ratio', 'lesion_shape_index', 'hue_contrast', 'luminance_contrast', 'lesion_color_difference', 'border_complexity', 'color_uniformity', 'position_distance_3d', 'perimeter_to_area_ratio', 'area_to_perimeter_ratio', 'lesion_visibility_score', 'color_consistency', 'consistency_color', 'size_age_interaction', 'hue_color_std_interaction', 'lesion_severity_index', 'shape_complexity_index', 'color_contrast_index', 'normalized_lesion_size', 'mean_hue_difference', 'lesion_orientation_3d', 'overall_color_difference', 'color_variance_ratio', 'border_color_interaction_2', 'size_color_contrast_ratio', 'age_normalized_nevi_confidence', 'age_normalized_nevi_confidence_2', 'color_range', 'shape_color_consistency', 'age_size_symmetry_index', 'index_age_size_symmetry', 'clin_size_long_diam_mm_patient_norm', 'tbp_lv_A_patient_norm', 'tbp_lv_Aext_patient_norm', 'tbp_lv_B_patient_norm', 'tbp_lv_Bext_patient_norm', 'tbp_lv_C_patient_norm', 'tbp_lv_Cext_patient_norm', 'tbp_lv_H_patient_norm', 'tbp_lv_Hext_patient_norm', 'tbp_lv_L_patient_norm', 'tbp_lv_Lext_patient_norm', 'tbp_lv_areaMM2_patient_norm', 'tbp_lv_color_std_mean_patient_norm', 'tbp_lv_deltaA_patient_norm', 'tbp_lv_deltaB_patient_norm', 'tbp_lv_deltaL_patient_norm', 'tbp_lv_deltaLB_patient_norm', 'tbp_lv_deltaLBnorm_patient_norm', 'tbp_lv_eccentricity_patient_norm', 'tbp_lv_minorAxisMM_patient_norm', 'tbp_lv_nevi_confidence_patient_norm', 'tbp_lv_norm_color_patient_norm', 'tbp_lv_radial_color_std_max_patient_norm', 'tbp_lv_stdL_patient_norm', 'tbp_lv_stdLExt_patient_norm', 'tbp_lv_symm_2axis_patient_norm', 'tbp_lv_symm_2axis_angle_patient_norm', 'tbp_lv_x_patient_norm', 'tbp_lv_y_patient_norm', 'tbp_lv_z_patient_norm', 'lesion_size_ratio_patient_norm', 'hue_contrast_patient_norm', 'luminance_contrast_patient_norm', 'lesion_color_difference_patient_norm', 'color_uniformity_patient_norm', 'position_distance_3d_patient_norm', 'perimeter_to_area_ratio_patient_norm', 'area_to_perimeter_ratio_patient_norm', 'lesion_visibility_score_patient_norm', 'consistency_symmetry_border_patient_norm', 'consistency_color_patient_norm', 'size_age_interaction_patient_norm', 'hue_color_std_interaction_patient_norm', 'lesion_severity_index_patient_norm', 'shape_complexity_index_patient_norm', 'color_contrast_index_patient_norm', 'log_lesion_area_patient_norm', 'normalized_lesion_size_patient_norm', 'mean_hue_difference_patient_norm', 'std_dev_contrast_patient_norm', 'color_shape_composite_index_patient_norm', 'lesion_orientation_3d_patient_norm', 'overall_color_difference_patient_norm', 'symmetry_perimeter_interaction_patient_norm', 'comprehensive_lesion_index_patient_norm', 'color_variance_ratio_patient_norm', 'border_color_interaction_patient_norm', 'border_color_interaction_2_patient_norm', 'size_color_contrast_ratio_patient_norm', 'age_normalized_nevi_confidence_patient_norm', 'age_normalized_nevi_confidence_2_patient_norm', 'color_asymmetry_index_patient_norm', 'volume_approximation_3d_patient_norm', 'color_range_patient_norm', 'shape_color_consistency_patient_norm', 'age_size_symmetry_index_patient_norm', 'index_age_size_symmetry_patient_norm', 'count_per_patient', 'age_approx_patient_min_max', 'clin_size_long_diam_mm_patient_min_max', 'tbp_lv_A_patient_min_max', 'tbp_lv_Aext_patient_min_max', 'tbp_lv_B_patient_min_max', 'tbp_lv_Bext_patient_min_max', 'tbp_lv_C_patient_min_max', 'tbp_lv_Cext_patient_min_max', 'tbp_lv_H_patient_min_max', 'tbp_lv_Hext_patient_min_max', 'tbp_lv_L_patient_min_max', 'tbp_lv_Lext_patient_min_max', 'tbp_lv_areaMM2_patient_min_max', 'tbp_lv_area_perim_ratio_patient_min_max', 'tbp_lv_color_std_mean_patient_min_max', 'tbp_lv_deltaA_patient_min_max', 'tbp_lv_deltaB_patient_min_max', 'tbp_lv_deltaL_patient_min_max', 'tbp_lv_deltaLB_patient_min_max', 'tbp_lv_deltaLBnorm_patient_min_max', 'tbp_lv_eccentricity_patient_min_max', 'tbp_lv_minorAxisMM_patient_min_max', 'tbp_lv_nevi_confidence_patient_min_max', 'tbp_lv_perimeterMM_patient_min_max', 'tbp_lv_radial_color_std_max_patient_min_max', 'tbp_lv_stdL_patient_min_max', 'tbp_lv_stdLExt_patient_min_max', 'tbp_lv_symm_2axis_patient_min_max', 'tbp_lv_symm_2axis_angle_patient_min_max', 'tbp_lv_x_patient_min_max', 'tbp_lv_y_patient_min_max', 'tbp_lv_z_patient_min_max', 'lesion_size_ratio_patient_min_max', 'lesion_shape_index_patient_min_max', 'hue_contrast_patient_min_max', 'lesion_color_difference_patient_min_max', 'border_complexity_patient_min_max', 'color_uniformity_patient_min_max', 'position_distance_3d_patient_min_max', 'perimeter_to_area_ratio_patient_min_max', 'area_to_perimeter_ratio_patient_min_max', 'lesion_visibility_score_patient_min_max', 'symmetry_border_consistency_patient_min_max', 'color_consistency_patient_min_max', 'consistency_color_patient_min_max', 'size_age_interaction_patient_min_max', 'hue_color_std_interaction_patient_min_max', 'lesion_severity_index_patient_min_max', 'color_contrast_index_patient_min_max', 'log_lesion_area_patient_min_max', 'normalized_lesion_size_patient_min_max', 'mean_hue_difference_patient_min_max', 'std_dev_contrast_patient_min_max', 'color_shape_composite_index_patient_min_max', 'lesion_orientation_3d_patient_min_max', 'overall_color_difference_patient_min_max', 'symmetry_perimeter_interaction_patient_min_max', 'comprehensive_lesion_index_patient_min_max', 'color_variance_ratio_patient_min_max', 'border_color_interaction_patient_min_max', 'border_color_interaction_2_patient_min_max', 'size_color_contrast_ratio_patient_min_max', 'age_normalized_nevi_confidence_patient_min_max', 'age_normalized_nevi_confidence_2_patient_min_max', 'color_asymmetry_index_patient_min_max', 'volume_approximation_3d_patient_min_max', 'color_range_patient_min_max', 'shape_color_consistency_patient_min_max', 'age_size_symmetry_index_patient_min_max', 'index_age_size_symmetry_patient_min_max', 'onehot_9', 'onehot_15', 'onehot_45', 'onehot_46', 'age_approx_future_norm', 'clin_size_long_diam_mm_future_norm', 'tbp_lv_A_future_norm', 'tbp_lv_Aext_future_norm', 'tbp_lv_B_future_norm', 'tbp_lv_Bext_future_norm', 'tbp_lv_C_future_norm', 'tbp_lv_Cext_future_norm', 'tbp_lv_H_future_norm', 'tbp_lv_Hext_future_norm', 'tbp_lv_L_future_norm', 'tbp_lv_Lext_future_norm', 'tbp_lv_areaMM2_future_norm', 'tbp_lv_area_perim_ratio_future_norm', 'tbp_lv_deltaA_future_norm', 'tbp_lv_deltaB_future_norm', 'tbp_lv_deltaLB_future_norm', 'tbp_lv_deltaLBnorm_future_norm', 'tbp_lv_eccentricity_future_norm', 'tbp_lv_minorAxisMM_future_norm', 'tbp_lv_nevi_confidence_future_norm', 'tbp_lv_norm_border_future_norm', 'tbp_lv_perimeterMM_future_norm', 'tbp_lv_radial_color_std_max_future_norm', 'tbp_lv_stdLExt_future_norm', 'tbp_lv_symm_2axis_future_norm', 'tbp_lv_symm_2axis_angle_future_norm', 'tbp_lv_x_future_norm', 'tbp_lv_y_future_norm', 'tbp_lv_z_future_norm', 'lesion_size_ratio_future_norm', 'lesion_shape_index_future_norm', 'hue_contrast_future_norm', 'lesion_color_difference_future_norm', 'border_complexity_future_norm', 'color_uniformity_future_norm', 'position_distance_3d_future_norm', 'perimeter_to_area_ratio_future_norm', 'area_to_perimeter_ratio_future_norm', 'lesion_visibility_score_future_norm', 'symmetry_border_consistency_future_norm', 'consistency_symmetry_border_future_norm', 'size_age_interaction_future_norm', 'lesion_severity_index_future_norm', 'shape_complexity_index_future_norm', 'color_contrast_index_future_norm', 'normalized_lesion_size_future_norm', 'mean_hue_difference_future_norm', 'lesion_orientation_3d_future_norm', 'overall_color_difference_future_norm', 'symmetry_perimeter_interaction_future_norm', 'comprehensive_lesion_index_future_norm', 'color_variance_ratio_future_norm', 'border_color_interaction_future_norm', 'size_color_contrast_ratio_future_norm', 'age_normalized_nevi_confidence_future_norm', 'age_normalized_nevi_confidence_2_future_norm', 'color_asymmetry_index_future_norm', 'volume_approximation_3d_future_norm', 'color_range_future_norm', 'border_length_ratio_future_norm', 'age_size_symmetry_index_future_norm', 'index_age_size_symmetry_future_norm', 'age_approx_past_norm', 'clin_size_long_diam_mm_past_norm', 'tbp_lv_A_past_norm', 'tbp_lv_Aext_past_norm', 'tbp_lv_B_past_norm', 'tbp_lv_C_past_norm', 'tbp_lv_Cext_past_norm', 'tbp_lv_H_past_norm', 'tbp_lv_Hext_past_norm', 'tbp_lv_L_past_norm', 'tbp_lv_Lext_past_norm', 'tbp_lv_area_perim_ratio_past_norm', 'tbp_lv_deltaA_past_norm', 'tbp_lv_deltaB_past_norm', 'tbp_lv_deltaLBnorm_past_norm', 'tbp_lv_eccentricity_past_norm', 'tbp_lv_minorAxisMM_past_norm', 'tbp_lv_nevi_confidence_past_norm', 'tbp_lv_norm_border_past_norm', 'tbp_lv_stdL_past_norm', 'tbp_lv_stdLExt_past_norm', 'tbp_lv_symm_2axis_past_norm', 'tbp_lv_symm_2axis_angle_past_norm', 'tbp_lv_x_past_norm', 'tbp_lv_y_past_norm', 'tbp_lv_z_past_norm', 'lesion_size_ratio_past_norm', 'lesion_shape_index_past_norm', 'hue_contrast_past_norm', 'luminance_contrast_past_norm', 'lesion_color_difference_past_norm', 'border_complexity_past_norm', 'color_uniformity_past_norm', 'position_distance_3d_past_norm', 'perimeter_to_area_ratio_past_norm', 'area_to_perimeter_ratio_past_norm', 'lesion_visibility_score_past_norm', 'symmetry_border_consistency_past_norm', 'consistency_symmetry_border_past_norm', 'color_consistency_past_norm', 'size_age_interaction_past_norm', 'hue_color_std_interaction_past_norm', 'lesion_severity_index_past_norm', 'shape_complexity_index_past_norm', 'color_contrast_index_past_norm', 'mean_hue_difference_past_norm', 'std_dev_contrast_past_norm', 'color_shape_composite_index_past_norm', 'lesion_orientation_3d_past_norm', 'overall_color_difference_past_norm', 'symmetry_perimeter_interaction_past_norm', 'comprehensive_lesion_index_past_norm', 'color_variance_ratio_past_norm', 'border_color_interaction_past_norm', 'size_color_contrast_ratio_past_norm', 'age_normalized_nevi_confidence_past_norm', 'age_normalized_nevi_confidence_2_past_norm', 'color_asymmetry_index_past_norm', 'volume_approximation_3d_past_norm', 'color_range_past_norm', 'shape_color_consistency_past_norm', 'age_size_symmetry_index_past_norm', 'index_age_size_symmetry_past_norm', 'count_future', 'count_past', 'age_approx_attri_norm', 'clin_size_long_diam_mm_attri_norm', 'tbp_lv_A_attri_norm', 'tbp_lv_Aext_attri_norm', 'tbp_lv_B_attri_norm', 'tbp_lv_Bext_attri_norm', 'tbp_lv_C_attri_norm', 'tbp_lv_Cext_attri_norm', 'tbp_lv_H_attri_norm', 'tbp_lv_Hext_attri_norm', 'tbp_lv_L_attri_norm', 'tbp_lv_Lext_attri_norm', 'tbp_lv_areaMM2_attri_norm', 'tbp_lv_deltaA_attri_norm', 'tbp_lv_deltaB_attri_norm', 'tbp_lv_deltaLB_attri_norm', 'tbp_lv_deltaLBnorm_attri_norm', 'tbp_lv_eccentricity_attri_norm', 'tbp_lv_minorAxisMM_attri_norm', 'tbp_lv_nevi_confidence_attri_norm', 'tbp_lv_norm_border_attri_norm', 'tbp_lv_norm_color_attri_norm', 'tbp_lv_perimeterMM_attri_norm', 'tbp_lv_stdL_attri_norm', 'tbp_lv_stdLExt_attri_norm', 'tbp_lv_symm_2axis_angle_attri_norm', 'tbp_lv_x_attri_norm', 'tbp_lv_y_attri_norm', 'tbp_lv_z_attri_norm', 'lesion_shape_index_attri_norm', 'hue_contrast_attri_norm', 'luminance_contrast_attri_norm', 'lesion_color_difference_attri_norm', 'border_complexity_attri_norm', 'color_uniformity_attri_norm', 'position_distance_3d_attri_norm', 'perimeter_to_area_ratio_attri_norm', 'area_to_perimeter_ratio_attri_norm', 'lesion_visibility_score_attri_norm', 'symmetry_border_consistency_attri_norm', 'consistency_symmetry_border_attri_norm', 'color_consistency_attri_norm', 'size_age_interaction_attri_norm', 'hue_color_std_interaction_attri_norm', 'lesion_severity_index_attri_norm', 'color_contrast_index_attri_norm', 'normalized_lesion_size_attri_norm', 'mean_hue_difference_attri_norm', 'color_shape_composite_index_attri_norm', 'lesion_orientation_3d_attri_norm', 'overall_color_difference_attri_norm', 'color_variance_ratio_attri_norm', 'border_color_interaction_attri_norm', 'size_color_contrast_ratio_attri_norm', 'age_normalized_nevi_confidence_attri_norm', 'age_normalized_nevi_confidence_2_attri_norm', 'color_asymmetry_index_attri_norm', 'volume_approximation_3d_attri_norm', 'color_range_attri_norm', 'border_length_ratio_attri_norm', 'age_size_symmetry_index_attri_norm', 'index_age_size_symmetry_attri_norm', 'count_per_patient_attri_norm', 'age_approx_tile_type_norm', 'clin_size_long_diam_mm_tile_type_norm', 'tbp_lv_A_tile_type_norm', 'tbp_lv_Aext_tile_type_norm', 'tbp_lv_B_tile_type_norm', 'tbp_lv_C_tile_type_norm', 'tbp_lv_Cext_tile_type_norm', 'tbp_lv_H_tile_type_norm', 'tbp_lv_L_tile_type_norm', 'tbp_lv_Lext_tile_type_norm', 'tbp_lv_area_perim_ratio_tile_type_norm', 'tbp_lv_deltaA_tile_type_norm', 'tbp_lv_deltaB_tile_type_norm', 'tbp_lv_deltaL_tile_type_norm', 'tbp_lv_deltaLB_tile_type_norm', 'tbp_lv_deltaLBnorm_tile_type_norm', 'tbp_lv_eccentricity_tile_type_norm', 'tbp_lv_minorAxisMM_tile_type_norm', 'tbp_lv_nevi_confidence_tile_type_norm', 'tbp_lv_norm_border_tile_type_norm', 'tbp_lv_perimeterMM_tile_type_norm', 'tbp_lv_radial_color_std_max_tile_type_norm', 'tbp_lv_stdL_tile_type_norm', 'tbp_lv_stdLExt_tile_type_norm', 'tbp_lv_symm_2axis_angle_tile_type_norm', 'tbp_lv_x_tile_type_norm', 'tbp_lv_y_tile_type_norm', 'tbp_lv_z_tile_type_norm', 'lesion_size_ratio_tile_type_norm', 'hue_contrast_tile_type_norm', 'luminance_contrast_tile_type_norm', 'border_complexity_tile_type_norm', 'color_uniformity_tile_type_norm', 'position_distance_3d_tile_type_norm', 'lesion_visibility_score_tile_type_norm', 'consistency_symmetry_border_tile_type_norm', 'color_consistency_tile_type_norm', 'consistency_color_tile_type_norm', 'size_age_interaction_tile_type_norm', 'hue_color_std_interaction_tile_type_norm', 'color_contrast_index_tile_type_norm', 'normalized_lesion_size_tile_type_norm', 'mean_hue_difference_tile_type_norm', 'std_dev_contrast_tile_type_norm', 'lesion_orientation_3d_tile_type_norm', 'overall_color_difference_tile_type_norm', 'border_color_interaction_tile_type_norm', 'size_color_contrast_ratio_tile_type_norm', 'age_normalized_nevi_confidence_tile_type_norm', 'age_normalized_nevi_confidence_2_tile_type_norm', 'color_asymmetry_index_tile_type_norm', 'volume_approximation_3d_tile_type_norm', 'color_range_tile_type_norm', 'index_age_size_symmetry_tile_type_norm', 'count_per_patient_tile_type_norm', 'age_approx_anatom_norm', 'clin_size_long_diam_mm_anatom_norm', 'tbp_lv_A_anatom_norm', 'tbp_lv_Aext_anatom_norm', 'tbp_lv_B_anatom_norm', 'tbp_lv_Bext_anatom_norm', 'tbp_lv_Cext_anatom_norm', 'tbp_lv_H_anatom_norm', 'tbp_lv_Hext_anatom_norm', 'tbp_lv_L_anatom_norm', 'tbp_lv_Lext_anatom_norm', 'tbp_lv_color_std_mean_anatom_norm', 'tbp_lv_deltaA_anatom_norm', 'tbp_lv_deltaB_anatom_norm', 'tbp_lv_deltaL_anatom_norm', 'tbp_lv_deltaLB_anatom_norm', 'tbp_lv_deltaLBnorm_anatom_norm', 'tbp_lv_eccentricity_anatom_norm', 'tbp_lv_minorAxisMM_anatom_norm', 'tbp_lv_nevi_confidence_anatom_norm', 'tbp_lv_norm_border_anatom_norm', 'tbp_lv_norm_color_anatom_norm', 'tbp_lv_radial_color_std_max_anatom_norm', 'tbp_lv_stdL_anatom_norm', 'tbp_lv_stdLExt_anatom_norm', 'tbp_lv_symm_2axis_angle_anatom_norm', 'tbp_lv_x_anatom_norm', 'tbp_lv_y_anatom_norm', 'tbp_lv_z_anatom_norm', 'lesion_size_ratio_anatom_norm', 'hue_contrast_anatom_norm', 'lesion_color_difference_anatom_norm', 'border_complexity_anatom_norm', 'color_uniformity_anatom_norm', 'position_distance_3d_anatom_norm', 'perimeter_to_area_ratio_anatom_norm', 'area_to_perimeter_ratio_anatom_norm', 'lesion_visibility_score_anatom_norm', 'symmetry_border_consistency_anatom_norm', 'consistency_symmetry_border_anatom_norm', 'color_consistency_anatom_norm', 'lesion_severity_index_anatom_norm', 'shape_complexity_index_anatom_norm', 'color_contrast_index_anatom_norm', 'log_lesion_area_anatom_norm', 'mean_hue_difference_anatom_norm', 'std_dev_contrast_anatom_norm', 'color_shape_composite_index_anatom_norm', 'lesion_orientation_3d_anatom_norm', 'overall_color_difference_anatom_norm', 'symmetry_perimeter_interaction_anatom_norm', 'comprehensive_lesion_index_anatom_norm', 'color_variance_ratio_anatom_norm', 'border_color_interaction_anatom_norm', 'size_color_contrast_ratio_anatom_norm', 'age_normalized_nevi_confidence_anatom_norm', 'age_normalized_nevi_confidence_2_anatom_norm', 'volume_approximation_3d_anatom_norm', 'color_range_anatom_norm', 'border_length_ratio_anatom_norm', 'age_size_symmetry_index_anatom_norm', 'count_per_patient_anatom_norm', 'age_approx_multi_min_max', 'clin_size_long_diam_mm_multi_min_max', 'tbp_lv_A_multi_min_max', 'tbp_lv_Aext_multi_min_max', 'tbp_lv_B_multi_min_max', 'tbp_lv_Bext_multi_min_max', 'tbp_lv_C_multi_min_max', 'tbp_lv_Cext_multi_min_max', 'tbp_lv_H_multi_min_max', 'tbp_lv_Hext_multi_min_max', 'tbp_lv_L_multi_min_max', 'tbp_lv_Lext_multi_min_max', 'tbp_lv_areaMM2_multi_min_max', 'tbp_lv_area_perim_ratio_multi_min_max', 'tbp_lv_deltaA_multi_min_max', 'tbp_lv_deltaB_multi_min_max', 'tbp_lv_deltaL_multi_min_max', 'tbp_lv_deltaLB_multi_min_max', 'tbp_lv_deltaLBnorm_multi_min_max', 'tbp_lv_eccentricity_multi_min_max', 'tbp_lv_nevi_confidence_multi_min_max', 'tbp_lv_norm_border_multi_min_max', 'tbp_lv_radial_color_std_max_multi_min_max', 'tbp_lv_stdL_multi_min_max', 'tbp_lv_stdLExt_multi_min_max', 'tbp_lv_symm_2axis_multi_min_max', 'tbp_lv_x_multi_min_max', 'tbp_lv_y_multi_min_max', 'tbp_lv_z_multi_min_max', 'lesion_size_ratio_multi_min_max', 'lesion_shape_index_multi_min_max', 'hue_contrast_multi_min_max', 'luminance_contrast_multi_min_max', 'lesion_color_difference_multi_min_max', 'color_uniformity_multi_min_max', 'position_distance_3d_multi_min_max', 'area_to_perimeter_ratio_multi_min_max', 'lesion_visibility_score_multi_min_max', 'symmetry_border_consistency_multi_min_max', 'consistency_symmetry_border_multi_min_max', 'color_consistency_multi_min_max', 'consistency_color_multi_min_max', 'size_age_interaction_multi_min_max', 'lesion_severity_index_multi_min_max', 'color_contrast_index_multi_min_max', 'log_lesion_area_multi_min_max', 'normalized_lesion_size_multi_min_max', 'mean_hue_difference_multi_min_max', 'std_dev_contrast_multi_min_max', 'color_shape_composite_index_multi_min_max', 'lesion_orientation_3d_multi_min_max', 'overall_color_difference_multi_min_max', 'symmetry_perimeter_interaction_multi_min_max', 'comprehensive_lesion_index_multi_min_max', 'color_variance_ratio_multi_min_max', 'border_color_interaction_multi_min_max', 'border_color_interaction_2_multi_min_max', 'size_color_contrast_ratio_multi_min_max', 'age_normalized_nevi_confidence_multi_min_max', 'age_normalized_nevi_confidence_2_multi_min_max', 'color_asymmetry_index_multi_min_max', 'volume_approximation_3d_multi_min_max', 'color_range_multi_min_max', 'shape_color_consistency_multi_min_max', 'border_length_ratio_multi_min_max', 'age_size_symmetry_index_multi_min_max', 'index_age_size_symmetry_multi_min_max', 'count_per_patient_multi_min_max']
cat_feats =  ['clin_size_long_diam_mm', 'tbp_lv_Aext', 'tbp_lv_B', 'tbp_lv_Bext', 'tbp_lv_C', 'tbp_lv_Cext', 'tbp_lv_H', 'tbp_lv_Hext', 'tbp_lv_L', 'tbp_lv_Lext', 'tbp_lv_color_std_mean', 'tbp_lv_deltaA', 'tbp_lv_deltaB', 'tbp_lv_deltaL', 'tbp_lv_deltaLB', 'tbp_lv_deltaLBnorm', 'tbp_lv_eccentricity', 'tbp_lv_nevi_confidence', 'tbp_lv_norm_border', 'tbp_lv_radial_color_std_max', 'tbp_lv_stdLExt', 'tbp_lv_symm_2axis', 'tbp_lv_x', 'tbp_lv_y', 'tbp_lv_z', 'lesion_size_ratio', 'lesion_shape_index', 'lesion_color_difference', 'color_uniformity', 'position_distance_3d', 'perimeter_to_area_ratio', 'color_consistency', 'size_age_interaction', 'hue_color_std_interaction', 'lesion_severity_index', 'color_contrast_index', 'log_lesion_area', 'normalized_lesion_size', 'mean_hue_difference', 'std_dev_contrast', 'symmetry_perimeter_interaction', 'comprehensive_lesion_index', 'color_variance_ratio', 'border_color_interaction', 'border_color_interaction_2', 'size_color_contrast_ratio', 'age_normalized_nevi_confidence', 'age_normalized_nevi_confidence_2', 'color_asymmetry_index', 'volume_approximation_3d', 'color_range', 'shape_color_consistency', 'border_length_ratio', 'age_size_symmetry_index', 'index_age_size_symmetry', 'age_approx_patient_norm', 'clin_size_long_diam_mm_patient_norm', 'tbp_lv_A_patient_norm', 'tbp_lv_Aext_patient_norm', 'tbp_lv_B_patient_norm', 'tbp_lv_Bext_patient_norm', 'tbp_lv_C_patient_norm', 'tbp_lv_Cext_patient_norm', 'tbp_lv_H_patient_norm', 'tbp_lv_Hext_patient_norm', 'tbp_lv_L_patient_norm', 'tbp_lv_Lext_patient_norm', 'tbp_lv_areaMM2_patient_norm', 'tbp_lv_area_perim_ratio_patient_norm', 'tbp_lv_color_std_mean_patient_norm', 'tbp_lv_deltaA_patient_norm', 'tbp_lv_deltaB_patient_norm', 'tbp_lv_deltaLBnorm_patient_norm', 'tbp_lv_eccentricity_patient_norm', 'tbp_lv_minorAxisMM_patient_norm', 'tbp_lv_nevi_confidence_patient_norm', 'tbp_lv_norm_border_patient_norm', 'tbp_lv_norm_color_patient_norm', 'tbp_lv_perimeterMM_patient_norm', 'tbp_lv_radial_color_std_max_patient_norm', 'tbp_lv_stdL_patient_norm', 'tbp_lv_stdLExt_patient_norm', 'tbp_lv_symm_2axis_patient_norm', 'tbp_lv_symm_2axis_angle_patient_norm', 'tbp_lv_x_patient_norm', 'tbp_lv_y_patient_norm', 'tbp_lv_z_patient_norm', 'lesion_size_ratio_patient_norm', 'lesion_shape_index_patient_norm', 'hue_contrast_patient_norm', 'luminance_contrast_patient_norm', 'lesion_color_difference_patient_norm', 'border_complexity_patient_norm', 'color_uniformity_patient_norm', 'position_distance_3d_patient_norm', 'perimeter_to_area_ratio_patient_norm', 'area_to_perimeter_ratio_patient_norm', 'lesion_visibility_score_patient_norm', 'symmetry_border_consistency_patient_norm', 'consistency_symmetry_border_patient_norm', 'color_consistency_patient_norm', 'consistency_color_patient_norm', 'size_age_interaction_patient_norm', 'hue_color_std_interaction_patient_norm', 'lesion_severity_index_patient_norm', 'color_contrast_index_patient_norm', 'log_lesion_area_patient_norm', 'normalized_lesion_size_patient_norm', 'mean_hue_difference_patient_norm', 'std_dev_contrast_patient_norm', 'color_shape_composite_index_patient_norm', 'lesion_orientation_3d_patient_norm', 'overall_color_difference_patient_norm', 'symmetry_perimeter_interaction_patient_norm', 'comprehensive_lesion_index_patient_norm', 'color_variance_ratio_patient_norm', 'border_color_interaction_patient_norm', 'border_color_interaction_2_patient_norm', 'size_color_contrast_ratio_patient_norm', 'age_normalized_nevi_confidence_patient_norm', 'age_normalized_nevi_confidence_2_patient_norm', 'color_asymmetry_index_patient_norm', 'volume_approximation_3d_patient_norm', 'shape_color_consistency_patient_norm', 'age_size_symmetry_index_patient_norm', 'index_age_size_symmetry_patient_norm', 'count_per_patient', 'clin_size_long_diam_mm_patient_min_max', 'tbp_lv_A_patient_min_max', 'tbp_lv_Aext_patient_min_max', 'tbp_lv_B_patient_min_max', 'tbp_lv_Bext_patient_min_max', 'tbp_lv_Cext_patient_min_max', 'tbp_lv_H_patient_min_max', 'tbp_lv_Hext_patient_min_max', 'tbp_lv_L_patient_min_max', 'tbp_lv_Lext_patient_min_max', 'tbp_lv_areaMM2_patient_min_max', 'tbp_lv_area_perim_ratio_patient_min_max', 'tbp_lv_color_std_mean_patient_min_max', 'tbp_lv_deltaA_patient_min_max', 'tbp_lv_deltaB_patient_min_max', 'tbp_lv_deltaL_patient_min_max', 'tbp_lv_deltaLB_patient_min_max', 'tbp_lv_deltaLBnorm_patient_min_max', 'tbp_lv_eccentricity_patient_min_max', 'tbp_lv_minorAxisMM_patient_min_max', 'tbp_lv_nevi_confidence_patient_min_max', 'tbp_lv_norm_border_patient_min_max', 'tbp_lv_norm_color_patient_min_max', 'tbp_lv_perimeterMM_patient_min_max', 'tbp_lv_radial_color_std_max_patient_min_max', 'tbp_lv_stdL_patient_min_max', 'tbp_lv_stdLExt_patient_min_max', 'tbp_lv_symm_2axis_patient_min_max', 'tbp_lv_x_patient_min_max', 'tbp_lv_y_patient_min_max', 'tbp_lv_z_patient_min_max', 'lesion_size_ratio_patient_min_max', 'lesion_shape_index_patient_min_max', 'hue_contrast_patient_min_max', 'luminance_contrast_patient_min_max', 'lesion_color_difference_patient_min_max', 'border_complexity_patient_min_max', 'color_uniformity_patient_min_max', 'position_distance_3d_patient_min_max', 'perimeter_to_area_ratio_patient_min_max', 'area_to_perimeter_ratio_patient_min_max', 'lesion_visibility_score_patient_min_max', 'symmetry_border_consistency_patient_min_max', 'consistency_symmetry_border_patient_min_max', 'color_consistency_patient_min_max', 'consistency_color_patient_min_max', 'size_age_interaction_patient_min_max', 'hue_color_std_interaction_patient_min_max', 'lesion_severity_index_patient_min_max', 'shape_complexity_index_patient_min_max', 'color_contrast_index_patient_min_max', 'log_lesion_area_patient_min_max', 'normalized_lesion_size_patient_min_max', 'mean_hue_difference_patient_min_max', 'std_dev_contrast_patient_min_max', 'color_shape_composite_index_patient_min_max', 'lesion_orientation_3d_patient_min_max', 'overall_color_difference_patient_min_max', 'symmetry_perimeter_interaction_patient_min_max', 'comprehensive_lesion_index_patient_min_max', 'color_variance_ratio_patient_min_max', 'border_color_interaction_patient_min_max', 'border_color_interaction_2_patient_min_max', 'size_color_contrast_ratio_patient_min_max', 'age_normalized_nevi_confidence_patient_min_max', 'age_normalized_nevi_confidence_2_patient_min_max', 'color_asymmetry_index_patient_min_max', 'volume_approximation_3d_patient_min_max', 'color_range_patient_min_max', 'shape_color_consistency_patient_min_max', 'border_length_ratio_patient_min_max', 'age_size_symmetry_index_patient_min_max', 'index_age_size_symmetry_patient_min_max', 'onehot_0', 'onehot_5', 'onehot_9', 'onehot_10', 'onehot_11', 'onehot_14', 'onehot_15', 'onehot_16', 'onehot_17', 'onehot_19', 'onehot_32', 'onehot_46', 'age_approx_future_norm', 'tbp_lv_A_future_norm', 'tbp_lv_Aext_future_norm', 'tbp_lv_B_future_norm', 'tbp_lv_C_future_norm', 'tbp_lv_H_future_norm', 'tbp_lv_Hext_future_norm', 'tbp_lv_L_future_norm', 'tbp_lv_Lext_future_norm', 'tbp_lv_areaMM2_future_norm', 'tbp_lv_area_perim_ratio_future_norm', 'tbp_lv_deltaA_future_norm', 'tbp_lv_deltaB_future_norm', 'tbp_lv_deltaL_future_norm', 'tbp_lv_deltaLBnorm_future_norm', 'tbp_lv_minorAxisMM_future_norm', 'tbp_lv_nevi_confidence_future_norm', 'tbp_lv_norm_color_future_norm', 'tbp_lv_perimeterMM_future_norm', 'tbp_lv_radial_color_std_max_future_norm', 'tbp_lv_stdL_future_norm', 'tbp_lv_stdLExt_future_norm', 'tbp_lv_symm_2axis_future_norm', 'tbp_lv_symm_2axis_angle_future_norm', 'tbp_lv_x_future_norm', 'tbp_lv_y_future_norm', 'lesion_size_ratio_future_norm', 'hue_contrast_future_norm', 'lesion_color_difference_future_norm', 'color_uniformity_future_norm', 'position_distance_3d_future_norm', 'perimeter_to_area_ratio_future_norm', 'area_to_perimeter_ratio_future_norm', 'lesion_visibility_score_future_norm', 'consistency_symmetry_border_future_norm', 'lesion_severity_index_future_norm', 'shape_complexity_index_future_norm', 'color_contrast_index_future_norm', 'log_lesion_area_future_norm', 'normalized_lesion_size_future_norm', 'mean_hue_difference_future_norm', 'std_dev_contrast_future_norm', 'color_shape_composite_index_future_norm', 'lesion_orientation_3d_future_norm', 'overall_color_difference_future_norm', 'symmetry_perimeter_interaction_future_norm', 'comprehensive_lesion_index_future_norm', 'color_variance_ratio_future_norm', 'border_color_interaction_future_norm', 'size_color_contrast_ratio_future_norm', 'age_normalized_nevi_confidence_future_norm', 'age_normalized_nevi_confidence_2_future_norm', 'color_asymmetry_index_future_norm', 'volume_approximation_3d_future_norm', 'color_range_future_norm', 'shape_color_consistency_future_norm', 'border_length_ratio_future_norm', 'age_size_symmetry_index_future_norm', 'index_age_size_symmetry_future_norm', 'age_approx_past_norm', 'clin_size_long_diam_mm_past_norm', 'tbp_lv_A_past_norm', 'tbp_lv_Aext_past_norm', 'tbp_lv_B_past_norm', 'tbp_lv_C_past_norm', 'tbp_lv_Cext_past_norm', 'tbp_lv_H_past_norm', 'tbp_lv_Hext_past_norm', 'tbp_lv_L_past_norm', 'tbp_lv_Lext_past_norm', 'tbp_lv_areaMM2_past_norm', 'tbp_lv_area_perim_ratio_past_norm', 'tbp_lv_color_std_mean_past_norm', 'tbp_lv_deltaLBnorm_past_norm', 'tbp_lv_eccentricity_past_norm', 'tbp_lv_nevi_confidence_past_norm', 'tbp_lv_norm_color_past_norm', 'tbp_lv_perimeterMM_past_norm', 'tbp_lv_stdLExt_past_norm', 'tbp_lv_x_past_norm', 'tbp_lv_z_past_norm', 'lesion_size_ratio_past_norm', 'hue_contrast_past_norm', 'luminance_contrast_past_norm', 'border_complexity_past_norm', 'color_uniformity_past_norm', 'position_distance_3d_past_norm', 'perimeter_to_area_ratio_past_norm', 'lesion_visibility_score_past_norm', 'color_consistency_past_norm', 'consistency_color_past_norm', 'color_contrast_index_past_norm', 'log_lesion_area_past_norm', 'mean_hue_difference_past_norm', 'color_shape_composite_index_past_norm', 'lesion_orientation_3d_past_norm', 'overall_color_difference_past_norm', 'comprehensive_lesion_index_past_norm', 'color_variance_ratio_past_norm', 'border_color_interaction_past_norm', 'size_color_contrast_ratio_past_norm', 'age_normalized_nevi_confidence_past_norm', 'age_normalized_nevi_confidence_2_past_norm', 'volume_approximation_3d_past_norm', 'shape_color_consistency_past_norm', 'border_length_ratio_past_norm', 'index_age_size_symmetry_past_norm', 'count_future', 'count_past', 'age_approx_attri_norm', 'tbp_lv_A_attri_norm', 'tbp_lv_Aext_attri_norm', 'tbp_lv_Bext_attri_norm', 'tbp_lv_Cext_attri_norm', 'tbp_lv_H_attri_norm', 'tbp_lv_Hext_attri_norm', 'tbp_lv_L_attri_norm', 'tbp_lv_Lext_attri_norm', 'tbp_lv_areaMM2_attri_norm', 'tbp_lv_area_perim_ratio_attri_norm', 'tbp_lv_color_std_mean_attri_norm', 'tbp_lv_deltaA_attri_norm', 'tbp_lv_deltaB_attri_norm', 'tbp_lv_deltaL_attri_norm', 'tbp_lv_deltaLB_attri_norm', 'tbp_lv_deltaLBnorm_attri_norm', 'tbp_lv_eccentricity_attri_norm', 'tbp_lv_minorAxisMM_attri_norm', 'tbp_lv_nevi_confidence_attri_norm', 'tbp_lv_norm_border_attri_norm', 'tbp_lv_perimeterMM_attri_norm', 'tbp_lv_radial_color_std_max_attri_norm', 'tbp_lv_stdLExt_attri_norm', 'tbp_lv_symm_2axis_attri_norm', 'tbp_lv_symm_2axis_angle_attri_norm', 'tbp_lv_x_attri_norm', 'tbp_lv_y_attri_norm', 'tbp_lv_z_attri_norm', 'lesion_size_ratio_attri_norm', 'hue_contrast_attri_norm', 'border_complexity_attri_norm', 'color_uniformity_attri_norm', 'position_distance_3d_attri_norm', 'perimeter_to_area_ratio_attri_norm', 'area_to_perimeter_ratio_attri_norm', 'lesion_visibility_score_attri_norm', 'symmetry_border_consistency_attri_norm', 'consistency_symmetry_border_attri_norm', 'color_consistency_attri_norm', 'size_age_interaction_attri_norm', 'lesion_severity_index_attri_norm', 'shape_complexity_index_attri_norm', 'color_contrast_index_attri_norm', 'log_lesion_area_attri_norm', 'normalized_lesion_size_attri_norm', 'mean_hue_difference_attri_norm', 'std_dev_contrast_attri_norm', 'color_shape_composite_index_attri_norm', 'lesion_orientation_3d_attri_norm', 'overall_color_difference_attri_norm', 'symmetry_perimeter_interaction_attri_norm', 'comprehensive_lesion_index_attri_norm', 'color_variance_ratio_attri_norm', 'border_color_interaction_attri_norm', 'size_color_contrast_ratio_attri_norm', 'age_normalized_nevi_confidence_attri_norm', 'age_normalized_nevi_confidence_2_attri_norm', 'color_asymmetry_index_attri_norm', 'volume_approximation_3d_attri_norm', 'color_range_attri_norm', 'shape_color_consistency_attri_norm', 'border_length_ratio_attri_norm', 'age_size_symmetry_index_attri_norm', 'index_age_size_symmetry_attri_norm', 'count_per_patient_attri_norm', 'age_approx_tile_type_norm', 'clin_size_long_diam_mm_tile_type_norm', 'tbp_lv_Aext_tile_type_norm', 'tbp_lv_B_tile_type_norm', 'tbp_lv_Bext_tile_type_norm', 'tbp_lv_C_tile_type_norm', 'tbp_lv_Cext_tile_type_norm', 'tbp_lv_H_tile_type_norm', 'tbp_lv_Hext_tile_type_norm', 'tbp_lv_L_tile_type_norm', 'tbp_lv_Lext_tile_type_norm', 'tbp_lv_area_perim_ratio_tile_type_norm', 'tbp_lv_deltaA_tile_type_norm', 'tbp_lv_deltaB_tile_type_norm', 'tbp_lv_deltaLBnorm_tile_type_norm', 'tbp_lv_eccentricity_tile_type_norm', 'tbp_lv_radial_color_std_max_tile_type_norm', 'tbp_lv_stdLExt_tile_type_norm', 'tbp_lv_symm_2axis_tile_type_norm', 'tbp_lv_z_tile_type_norm', 'hue_contrast_tile_type_norm', 'color_uniformity_tile_type_norm', 'position_distance_3d_tile_type_norm', 'area_to_perimeter_ratio_tile_type_norm', 'lesion_visibility_score_tile_type_norm', 'symmetry_border_consistency_tile_type_norm', 'consistency_symmetry_border_tile_type_norm', 'color_consistency_tile_type_norm', 'consistency_color_tile_type_norm', 'size_age_interaction_tile_type_norm', 'hue_color_std_interaction_tile_type_norm', 'lesion_severity_index_tile_type_norm', 'shape_complexity_index_tile_type_norm', 'color_contrast_index_tile_type_norm', 'log_lesion_area_tile_type_norm', 'normalized_lesion_size_tile_type_norm', 'mean_hue_difference_tile_type_norm', 'lesion_orientation_3d_tile_type_norm', 'overall_color_difference_tile_type_norm', 'symmetry_perimeter_interaction_tile_type_norm', 'size_color_contrast_ratio_tile_type_norm', 'age_normalized_nevi_confidence_2_tile_type_norm', 'volume_approximation_3d_tile_type_norm', 'color_range_tile_type_norm', 'age_size_symmetry_index_tile_type_norm', 'index_age_size_symmetry_tile_type_norm', 'count_per_patient_tile_type_norm', 'age_approx_anatom_norm', 'tbp_lv_Aext_anatom_norm', 'tbp_lv_B_anatom_norm', 'tbp_lv_Bext_anatom_norm', 'tbp_lv_Cext_anatom_norm', 'tbp_lv_H_anatom_norm', 'tbp_lv_Hext_anatom_norm', 'tbp_lv_L_anatom_norm', 'tbp_lv_Lext_anatom_norm', 'tbp_lv_areaMM2_anatom_norm', 'tbp_lv_area_perim_ratio_anatom_norm', 'tbp_lv_color_std_mean_anatom_norm', 'tbp_lv_deltaA_anatom_norm', 'tbp_lv_deltaB_anatom_norm', 'tbp_lv_deltaL_anatom_norm', 'tbp_lv_deltaLB_anatom_norm', 'tbp_lv_deltaLBnorm_anatom_norm', 'tbp_lv_eccentricity_anatom_norm', 'tbp_lv_minorAxisMM_anatom_norm', 'tbp_lv_norm_border_anatom_norm', 'tbp_lv_norm_color_anatom_norm', 'tbp_lv_perimeterMM_anatom_norm', 'tbp_lv_radial_color_std_max_anatom_norm', 'tbp_lv_stdLExt_anatom_norm', 'tbp_lv_symm_2axis_angle_anatom_norm', 'tbp_lv_x_anatom_norm', 'tbp_lv_y_anatom_norm', 'tbp_lv_z_anatom_norm', 'lesion_size_ratio_anatom_norm', 'lesion_shape_index_anatom_norm', 'border_complexity_anatom_norm', 'color_uniformity_anatom_norm', 'position_distance_3d_anatom_norm', 'perimeter_to_area_ratio_anatom_norm', 'area_to_perimeter_ratio_anatom_norm', 'color_consistency_anatom_norm', 'consistency_color_anatom_norm', 'size_age_interaction_anatom_norm', 'lesion_severity_index_anatom_norm', 'color_contrast_index_anatom_norm', 'log_lesion_area_anatom_norm', 'normalized_lesion_size_anatom_norm', 'mean_hue_difference_anatom_norm', 'std_dev_contrast_anatom_norm', 'lesion_orientation_3d_anatom_norm', 'overall_color_difference_anatom_norm', 'symmetry_perimeter_interaction_anatom_norm', 'comprehensive_lesion_index_anatom_norm', 'color_variance_ratio_anatom_norm', 'border_color_interaction_anatom_norm', 'border_color_interaction_2_anatom_norm', 'size_color_contrast_ratio_anatom_norm', 'age_normalized_nevi_confidence_anatom_norm', 'age_normalized_nevi_confidence_2_anatom_norm', 'color_asymmetry_index_anatom_norm', 'volume_approximation_3d_anatom_norm', 'color_range_anatom_norm', 'shape_color_consistency_anatom_norm', 'border_length_ratio_anatom_norm', 'age_size_symmetry_index_anatom_norm', 'index_age_size_symmetry_anatom_norm', 'count_per_patient_anatom_norm', 'age_approx_multi_min_max', 'clin_size_long_diam_mm_multi_min_max', 'tbp_lv_A_multi_min_max', 'tbp_lv_Aext_multi_min_max', 'tbp_lv_B_multi_min_max', 'tbp_lv_Bext_multi_min_max', 'tbp_lv_C_multi_min_max', 'tbp_lv_Cext_multi_min_max', 'tbp_lv_H_multi_min_max', 'tbp_lv_Hext_multi_min_max', 'tbp_lv_L_multi_min_max', 'tbp_lv_Lext_multi_min_max', 'tbp_lv_areaMM2_multi_min_max', 'tbp_lv_area_perim_ratio_multi_min_max', 'tbp_lv_color_std_mean_multi_min_max', 'tbp_lv_deltaA_multi_min_max', 'tbp_lv_deltaB_multi_min_max', 'tbp_lv_deltaL_multi_min_max', 'tbp_lv_deltaLB_multi_min_max', 'tbp_lv_deltaLBnorm_multi_min_max', 'tbp_lv_eccentricity_multi_min_max', 'tbp_lv_minorAxisMM_multi_min_max', 'tbp_lv_nevi_confidence_multi_min_max', 'tbp_lv_norm_border_multi_min_max', 'tbp_lv_norm_color_multi_min_max', 'tbp_lv_perimeterMM_multi_min_max', 'tbp_lv_radial_color_std_max_multi_min_max', 'tbp_lv_stdL_multi_min_max', 'tbp_lv_stdLExt_multi_min_max', 'tbp_lv_symm_2axis_multi_min_max', 'tbp_lv_symm_2axis_angle_multi_min_max', 'tbp_lv_x_multi_min_max', 'tbp_lv_y_multi_min_max', 'tbp_lv_z_multi_min_max', 'lesion_size_ratio_multi_min_max', 'lesion_shape_index_multi_min_max', 'hue_contrast_multi_min_max', 'luminance_contrast_multi_min_max', 'lesion_color_difference_multi_min_max', 'border_complexity_multi_min_max', 'color_uniformity_multi_min_max', 'position_distance_3d_multi_min_max', 'perimeter_to_area_ratio_multi_min_max', 'area_to_perimeter_ratio_multi_min_max', 'lesion_visibility_score_multi_min_max', 'symmetry_border_consistency_multi_min_max', 'consistency_symmetry_border_multi_min_max', 'color_consistency_multi_min_max', 'consistency_color_multi_min_max', 'size_age_interaction_multi_min_max', 'hue_color_std_interaction_multi_min_max', 'lesion_severity_index_multi_min_max', 'shape_complexity_index_multi_min_max', 'color_contrast_index_multi_min_max', 'log_lesion_area_multi_min_max', 'normalized_lesion_size_multi_min_max', 'mean_hue_difference_multi_min_max', 'std_dev_contrast_multi_min_max', 'color_shape_composite_index_multi_min_max', 'lesion_orientation_3d_multi_min_max', 'overall_color_difference_multi_min_max', 'symmetry_perimeter_interaction_multi_min_max', 'comprehensive_lesion_index_multi_min_max', 'color_variance_ratio_multi_min_max', 'border_color_interaction_multi_min_max', 'border_color_interaction_2_multi_min_max', 'size_color_contrast_ratio_multi_min_max', 'age_normalized_nevi_confidence_multi_min_max', 'age_normalized_nevi_confidence_2_multi_min_max', 'color_asymmetry_index_multi_min_max', 'volume_approximation_3d_multi_min_max', 'color_range_multi_min_max', 'shape_color_consistency_multi_min_max', 'border_length_ratio_multi_min_max', 'age_size_symmetry_index_multi_min_max', 'index_age_size_symmetry_multi_min_max', 'count_per_patient_multi_min_max']
xgb_feats =  ['clin_size_long_diam_mm', 'tbp_lv_A', 'tbp_lv_Aext', 'tbp_lv_B', 'tbp_lv_Bext', 'tbp_lv_C', 'tbp_lv_Cext', 'tbp_lv_H', 'tbp_lv_Hext', 'tbp_lv_L', 'tbp_lv_Lext', 'tbp_lv_area_perim_ratio', 'tbp_lv_color_std_mean', 'tbp_lv_deltaA', 'tbp_lv_deltaB', 'tbp_lv_deltaL', 'tbp_lv_deltaLB', 'tbp_lv_deltaLBnorm', 'tbp_lv_minorAxisMM', 'tbp_lv_nevi_confidence', 'tbp_lv_perimeterMM', 'tbp_lv_radial_color_std_max', 'tbp_lv_stdLExt', 'tbp_lv_symm_2axis_angle', 'tbp_lv_x', 'tbp_lv_y', 'tbp_lv_z', 'lesion_size_ratio', 'hue_contrast', 'luminance_contrast', 'lesion_color_difference', 'color_uniformity', 'position_distance_3d', 'perimeter_to_area_ratio', 'area_to_perimeter_ratio', 'lesion_visibility_score', 'consistency_symmetry_border', 'color_consistency', 'size_age_interaction', 'lesion_severity_index', 'color_contrast_index', 'normalized_lesion_size', 'mean_hue_difference', 'lesion_orientation_3d', 'overall_color_difference', 'color_variance_ratio', 'size_color_contrast_ratio', 'age_normalized_nevi_confidence', 'age_normalized_nevi_confidence_2', 'volume_approximation_3d', 'color_range', 'age_size_symmetry_index', 'index_age_size_symmetry', 'clin_size_long_diam_mm_patient_norm', 'tbp_lv_A_patient_norm', 'tbp_lv_Aext_patient_norm', 'tbp_lv_B_patient_norm', 'tbp_lv_Bext_patient_norm', 'tbp_lv_C_patient_norm', 'tbp_lv_Cext_patient_norm', 'tbp_lv_H_patient_norm', 'tbp_lv_Hext_patient_norm', 'tbp_lv_L_patient_norm', 'tbp_lv_Lext_patient_norm', 'tbp_lv_areaMM2_patient_norm', 'tbp_lv_color_std_mean_patient_norm', 'tbp_lv_deltaA_patient_norm', 'tbp_lv_deltaB_patient_norm', 'tbp_lv_deltaL_patient_norm', 'tbp_lv_deltaLB_patient_norm', 'tbp_lv_deltaLBnorm_patient_norm', 'tbp_lv_eccentricity_patient_norm', 'tbp_lv_minorAxisMM_patient_norm', 'tbp_lv_nevi_confidence_patient_norm', 'tbp_lv_norm_border_patient_norm', 'tbp_lv_norm_color_patient_norm', 'tbp_lv_perimeterMM_patient_norm', 'tbp_lv_radial_color_std_max_patient_norm', 'tbp_lv_stdL_patient_norm', 'tbp_lv_stdLExt_patient_norm', 'tbp_lv_symm_2axis_patient_norm', 'tbp_lv_symm_2axis_angle_patient_norm', 'tbp_lv_x_patient_norm', 'tbp_lv_y_patient_norm', 'tbp_lv_z_patient_norm', 'lesion_size_ratio_patient_norm', 'hue_contrast_patient_norm', 'lesion_color_difference_patient_norm', 'color_uniformity_patient_norm', 'position_distance_3d_patient_norm', 'perimeter_to_area_ratio_patient_norm', 'area_to_perimeter_ratio_patient_norm', 'lesion_visibility_score_patient_norm', 'symmetry_border_consistency_patient_norm', 'consistency_symmetry_border_patient_norm', 'color_consistency_patient_norm', 'size_age_interaction_patient_norm', 'hue_color_std_interaction_patient_norm', 'lesion_severity_index_patient_norm', 'color_contrast_index_patient_norm', 'log_lesion_area_patient_norm', 'normalized_lesion_size_patient_norm', 'mean_hue_difference_patient_norm', 'std_dev_contrast_patient_norm', 'lesion_orientation_3d_patient_norm', 'overall_color_difference_patient_norm', 'symmetry_perimeter_interaction_patient_norm', 'comprehensive_lesion_index_patient_norm', 'color_variance_ratio_patient_norm', 'border_color_interaction_patient_norm', 'border_color_interaction_2_patient_norm', 'size_color_contrast_ratio_patient_norm', 'age_normalized_nevi_confidence_patient_norm', 'age_normalized_nevi_confidence_2_patient_norm', 'color_asymmetry_index_patient_norm', 'volume_approximation_3d_patient_norm', 'color_range_patient_norm', 'shape_color_consistency_patient_norm', 'index_age_size_symmetry_patient_norm', 'count_per_patient', 'age_approx_patient_min_max', 'clin_size_long_diam_mm_patient_min_max', 'tbp_lv_A_patient_min_max', 'tbp_lv_Aext_patient_min_max', 'tbp_lv_B_patient_min_max', 'tbp_lv_Bext_patient_min_max', 'tbp_lv_C_patient_min_max', 'tbp_lv_Cext_patient_min_max', 'tbp_lv_H_patient_min_max', 'tbp_lv_Hext_patient_min_max', 'tbp_lv_L_patient_min_max', 'tbp_lv_Lext_patient_min_max', 'tbp_lv_areaMM2_patient_min_max', 'tbp_lv_area_perim_ratio_patient_min_max', 'tbp_lv_color_std_mean_patient_min_max', 'tbp_lv_deltaA_patient_min_max', 'tbp_lv_deltaB_patient_min_max', 'tbp_lv_deltaL_patient_min_max', 'tbp_lv_deltaLB_patient_min_max', 'tbp_lv_deltaLBnorm_patient_min_max', 'tbp_lv_eccentricity_patient_min_max', 'tbp_lv_minorAxisMM_patient_min_max', 'tbp_lv_nevi_confidence_patient_min_max', 'tbp_lv_norm_border_patient_min_max', 'tbp_lv_perimeterMM_patient_min_max', 'tbp_lv_radial_color_std_max_patient_min_max', 'tbp_lv_stdL_patient_min_max', 'tbp_lv_stdLExt_patient_min_max', 'tbp_lv_symm_2axis_patient_min_max', 'tbp_lv_symm_2axis_angle_patient_min_max', 'tbp_lv_x_patient_min_max', 'tbp_lv_y_patient_min_max', 'tbp_lv_z_patient_min_max', 'lesion_size_ratio_patient_min_max', 'lesion_shape_index_patient_min_max', 'hue_contrast_patient_min_max', 'luminance_contrast_patient_min_max', 'lesion_color_difference_patient_min_max', 'border_complexity_patient_min_max', 'color_uniformity_patient_min_max', 'position_distance_3d_patient_min_max', 'perimeter_to_area_ratio_patient_min_max', 'area_to_perimeter_ratio_patient_min_max', 'lesion_visibility_score_patient_min_max', 'symmetry_border_consistency_patient_min_max', 'consistency_symmetry_border_patient_min_max', 'color_consistency_patient_min_max', 'consistency_color_patient_min_max', 'size_age_interaction_patient_min_max', 'hue_color_std_interaction_patient_min_max', 'lesion_severity_index_patient_min_max', 'color_contrast_index_patient_min_max', 'log_lesion_area_patient_min_max', 'normalized_lesion_size_patient_min_max', 'mean_hue_difference_patient_min_max', 'std_dev_contrast_patient_min_max', 'color_shape_composite_index_patient_min_max', 'lesion_orientation_3d_patient_min_max', 'overall_color_difference_patient_min_max', 'symmetry_perimeter_interaction_patient_min_max', 'comprehensive_lesion_index_patient_min_max', 'color_variance_ratio_patient_min_max', 'border_color_interaction_patient_min_max', 'border_color_interaction_2_patient_min_max', 'size_color_contrast_ratio_patient_min_max', 'age_normalized_nevi_confidence_patient_min_max', 'age_normalized_nevi_confidence_2_patient_min_max', 'color_asymmetry_index_patient_min_max', 'volume_approximation_3d_patient_min_max', 'color_range_patient_min_max', 'shape_color_consistency_patient_min_max', 'border_length_ratio_patient_min_max', 'age_size_symmetry_index_patient_min_max', 'index_age_size_symmetry_patient_min_max', 'onehot_9', 'onehot_40', 'onehot_45', 'onehot_46', 'age_approx_future_norm', 'clin_size_long_diam_mm_future_norm', 'tbp_lv_A_future_norm', 'tbp_lv_Aext_future_norm', 'tbp_lv_B_future_norm', 'tbp_lv_Bext_future_norm', 'tbp_lv_C_future_norm', 'tbp_lv_Cext_future_norm', 'tbp_lv_H_future_norm', 'tbp_lv_Hext_future_norm', 'tbp_lv_L_future_norm', 'tbp_lv_Lext_future_norm', 'tbp_lv_area_perim_ratio_future_norm', 'tbp_lv_deltaA_future_norm', 'tbp_lv_deltaB_future_norm', 'tbp_lv_deltaLB_future_norm', 'tbp_lv_deltaLBnorm_future_norm', 'tbp_lv_eccentricity_future_norm', 'tbp_lv_minorAxisMM_future_norm', 'tbp_lv_nevi_confidence_future_norm', 'tbp_lv_norm_border_future_norm', 'tbp_lv_norm_color_future_norm', 'tbp_lv_perimeterMM_future_norm', 'tbp_lv_radial_color_std_max_future_norm', 'tbp_lv_stdL_future_norm', 'tbp_lv_stdLExt_future_norm', 'tbp_lv_symm_2axis_future_norm', 'tbp_lv_symm_2axis_angle_future_norm', 'tbp_lv_x_future_norm', 'tbp_lv_y_future_norm', 'tbp_lv_z_future_norm', 'lesion_size_ratio_future_norm', 'hue_contrast_future_norm', 'luminance_contrast_future_norm', 'lesion_color_difference_future_norm', 'border_complexity_future_norm', 'color_uniformity_future_norm', 'position_distance_3d_future_norm', 'perimeter_to_area_ratio_future_norm', 'area_to_perimeter_ratio_future_norm', 'lesion_visibility_score_future_norm', 'symmetry_border_consistency_future_norm', 'consistency_symmetry_border_future_norm', 'color_consistency_future_norm', 'size_age_interaction_future_norm', 'lesion_severity_index_future_norm', 'shape_complexity_index_future_norm', 'color_contrast_index_future_norm', 'mean_hue_difference_future_norm', 'std_dev_contrast_future_norm', 'color_shape_composite_index_future_norm', 'lesion_orientation_3d_future_norm', 'overall_color_difference_future_norm', 'symmetry_perimeter_interaction_future_norm', 'comprehensive_lesion_index_future_norm', 'color_variance_ratio_future_norm', 'border_color_interaction_future_norm', 'size_color_contrast_ratio_future_norm', 'age_normalized_nevi_confidence_future_norm', 'age_normalized_nevi_confidence_2_future_norm', 'color_asymmetry_index_future_norm', 'color_range_future_norm', 'shape_color_consistency_future_norm', 'border_length_ratio_future_norm', 'age_size_symmetry_index_future_norm', 'index_age_size_symmetry_future_norm', 'age_approx_past_norm', 'clin_size_long_diam_mm_past_norm', 'tbp_lv_A_past_norm', 'tbp_lv_B_past_norm', 'tbp_lv_Bext_past_norm', 'tbp_lv_C_past_norm', 'tbp_lv_Cext_past_norm', 'tbp_lv_H_past_norm', 'tbp_lv_Hext_past_norm', 'tbp_lv_L_past_norm', 'tbp_lv_Lext_past_norm', 'tbp_lv_area_perim_ratio_past_norm', 'tbp_lv_color_std_mean_past_norm', 'tbp_lv_deltaA_past_norm', 'tbp_lv_deltaB_past_norm', 'tbp_lv_deltaLB_past_norm', 'tbp_lv_deltaLBnorm_past_norm', 'tbp_lv_eccentricity_past_norm', 'tbp_lv_minorAxisMM_past_norm', 'tbp_lv_nevi_confidence_past_norm', 'tbp_lv_norm_color_past_norm', 'tbp_lv_perimeterMM_past_norm', 'tbp_lv_radial_color_std_max_past_norm', 'tbp_lv_stdL_past_norm', 'tbp_lv_stdLExt_past_norm', 'tbp_lv_symm_2axis_angle_past_norm', 'tbp_lv_x_past_norm', 'tbp_lv_y_past_norm', 'tbp_lv_z_past_norm', 'lesion_size_ratio_past_norm', 'lesion_shape_index_past_norm', 'hue_contrast_past_norm', 'luminance_contrast_past_norm', 'lesion_color_difference_past_norm', 'border_complexity_past_norm', 'color_uniformity_past_norm', 'position_distance_3d_past_norm', 'perimeter_to_area_ratio_past_norm', 'area_to_perimeter_ratio_past_norm', 'lesion_visibility_score_past_norm', 'symmetry_border_consistency_past_norm', 'consistency_symmetry_border_past_norm', 'color_consistency_past_norm', 'consistency_color_past_norm', 'hue_color_std_interaction_past_norm', 'lesion_severity_index_past_norm', 'shape_complexity_index_past_norm', 'color_contrast_index_past_norm', 'mean_hue_difference_past_norm', 'lesion_orientation_3d_past_norm', 'overall_color_difference_past_norm', 'symmetry_perimeter_interaction_past_norm', 'comprehensive_lesion_index_past_norm', 'color_variance_ratio_past_norm', 'border_color_interaction_past_norm', 'border_color_interaction_2_past_norm', 'size_color_contrast_ratio_past_norm', 'age_normalized_nevi_confidence_past_norm', 'age_normalized_nevi_confidence_2_past_norm', 'color_asymmetry_index_past_norm', 'volume_approximation_3d_past_norm', 'color_range_past_norm', 'shape_color_consistency_past_norm', 'border_length_ratio_past_norm', 'age_size_symmetry_index_past_norm', 'index_age_size_symmetry_past_norm', 'count_future', 'count_past', 'age_approx_attri_norm', 'clin_size_long_diam_mm_attri_norm', 'tbp_lv_A_attri_norm', 'tbp_lv_Aext_attri_norm', 'tbp_lv_B_attri_norm', 'tbp_lv_Bext_attri_norm', 'tbp_lv_C_attri_norm', 'tbp_lv_Cext_attri_norm', 'tbp_lv_H_attri_norm', 'tbp_lv_Hext_attri_norm', 'tbp_lv_L_attri_norm', 'tbp_lv_Lext_attri_norm', 'tbp_lv_areaMM2_attri_norm', 'tbp_lv_area_perim_ratio_attri_norm', 'tbp_lv_deltaA_attri_norm', 'tbp_lv_deltaB_attri_norm', 'tbp_lv_deltaL_attri_norm', 'tbp_lv_deltaLB_attri_norm', 'tbp_lv_deltaLBnorm_attri_norm', 'tbp_lv_eccentricity_attri_norm', 'tbp_lv_minorAxisMM_attri_norm', 'tbp_lv_nevi_confidence_attri_norm', 'tbp_lv_radial_color_std_max_attri_norm', 'tbp_lv_stdL_attri_norm', 'tbp_lv_stdLExt_attri_norm', 'tbp_lv_symm_2axis_angle_attri_norm', 'tbp_lv_x_attri_norm', 'tbp_lv_y_attri_norm', 'tbp_lv_z_attri_norm', 'lesion_size_ratio_attri_norm', 'hue_contrast_attri_norm', 'luminance_contrast_attri_norm', 'lesion_color_difference_attri_norm', 'color_uniformity_attri_norm', 'position_distance_3d_attri_norm', 'perimeter_to_area_ratio_attri_norm', 'area_to_perimeter_ratio_attri_norm', 'symmetry_border_consistency_attri_norm', 'consistency_symmetry_border_attri_norm', 'color_consistency_attri_norm', 'consistency_color_attri_norm', 'size_age_interaction_attri_norm', 'hue_color_std_interaction_attri_norm', 'lesion_severity_index_attri_norm', 'color_contrast_index_attri_norm', 'normalized_lesion_size_attri_norm', 'mean_hue_difference_attri_norm', 'std_dev_contrast_attri_norm', 'lesion_orientation_3d_attri_norm', 'overall_color_difference_attri_norm', 'symmetry_perimeter_interaction_attri_norm', 'comprehensive_lesion_index_attri_norm', 'size_color_contrast_ratio_attri_norm', 'age_normalized_nevi_confidence_attri_norm', 'age_normalized_nevi_confidence_2_attri_norm', 'color_asymmetry_index_attri_norm', 'volume_approximation_3d_attri_norm', 'color_range_attri_norm', 'age_size_symmetry_index_attri_norm', 'index_age_size_symmetry_attri_norm', 'count_per_patient_attri_norm', 'age_approx_tile_type_norm', 'clin_size_long_diam_mm_tile_type_norm', 'tbp_lv_A_tile_type_norm', 'tbp_lv_Aext_tile_type_norm', 'tbp_lv_B_tile_type_norm', 'tbp_lv_Bext_tile_type_norm', 'tbp_lv_C_tile_type_norm', 'tbp_lv_Cext_tile_type_norm', 'tbp_lv_H_tile_type_norm', 'tbp_lv_Hext_tile_type_norm', 'tbp_lv_L_tile_type_norm', 'tbp_lv_Lext_tile_type_norm', 'tbp_lv_areaMM2_tile_type_norm', 'tbp_lv_deltaA_tile_type_norm', 'tbp_lv_deltaB_tile_type_norm', 'tbp_lv_deltaL_tile_type_norm', 'tbp_lv_deltaLB_tile_type_norm', 'tbp_lv_deltaLBnorm_tile_type_norm', 'tbp_lv_eccentricity_tile_type_norm', 'tbp_lv_nevi_confidence_tile_type_norm', 'tbp_lv_stdL_tile_type_norm', 'tbp_lv_stdLExt_tile_type_norm', 'tbp_lv_x_tile_type_norm', 'tbp_lv_y_tile_type_norm', 'tbp_lv_z_tile_type_norm', 'lesion_size_ratio_tile_type_norm', 'hue_contrast_tile_type_norm', 'luminance_contrast_tile_type_norm', 'lesion_color_difference_tile_type_norm', 'border_complexity_tile_type_norm', 'color_uniformity_tile_type_norm', 'position_distance_3d_tile_type_norm', 'perimeter_to_area_ratio_tile_type_norm', 'area_to_perimeter_ratio_tile_type_norm', 'lesion_visibility_score_tile_type_norm', 'size_age_interaction_tile_type_norm', 'color_contrast_index_tile_type_norm', 'normalized_lesion_size_tile_type_norm', 'mean_hue_difference_tile_type_norm', 'std_dev_contrast_tile_type_norm', 'lesion_orientation_3d_tile_type_norm', 'overall_color_difference_tile_type_norm', 'border_color_interaction_tile_type_norm', 'size_color_contrast_ratio_tile_type_norm', 'age_normalized_nevi_confidence_tile_type_norm', 'age_normalized_nevi_confidence_2_tile_type_norm', 'volume_approximation_3d_tile_type_norm', 'color_range_tile_type_norm', 'shape_color_consistency_tile_type_norm', 'index_age_size_symmetry_tile_type_norm', 'count_per_patient_tile_type_norm', 'age_approx_anatom_norm', 'clin_size_long_diam_mm_anatom_norm', 'tbp_lv_A_anatom_norm', 'tbp_lv_Aext_anatom_norm', 'tbp_lv_B_anatom_norm', 'tbp_lv_Bext_anatom_norm', 'tbp_lv_C_anatom_norm', 'tbp_lv_Cext_anatom_norm', 'tbp_lv_H_anatom_norm', 'tbp_lv_Hext_anatom_norm', 'tbp_lv_L_anatom_norm', 'tbp_lv_Lext_anatom_norm', 'tbp_lv_area_perim_ratio_anatom_norm', 'tbp_lv_deltaA_anatom_norm', 'tbp_lv_deltaB_anatom_norm', 'tbp_lv_deltaL_anatom_norm', 'tbp_lv_deltaLB_anatom_norm', 'tbp_lv_deltaLBnorm_anatom_norm', 'tbp_lv_eccentricity_anatom_norm', 'tbp_lv_nevi_confidence_anatom_norm', 'tbp_lv_norm_border_anatom_norm', 'tbp_lv_norm_color_anatom_norm', 'tbp_lv_perimeterMM_anatom_norm', 'tbp_lv_radial_color_std_max_anatom_norm', 'tbp_lv_stdL_anatom_norm', 'tbp_lv_stdLExt_anatom_norm', 'tbp_lv_symm_2axis_angle_anatom_norm', 'tbp_lv_x_anatom_norm', 'tbp_lv_y_anatom_norm', 'tbp_lv_z_anatom_norm', 'lesion_size_ratio_anatom_norm', 'hue_contrast_anatom_norm', 'luminance_contrast_anatom_norm', 'lesion_color_difference_anatom_norm', 'color_uniformity_anatom_norm', 'position_distance_3d_anatom_norm', 'perimeter_to_area_ratio_anatom_norm', 'area_to_perimeter_ratio_anatom_norm', 'lesion_visibility_score_anatom_norm', 'symmetry_border_consistency_anatom_norm', 'color_consistency_anatom_norm', 'size_age_interaction_anatom_norm', 'hue_color_std_interaction_anatom_norm', 'color_contrast_index_anatom_norm', 'log_lesion_area_anatom_norm', 'normalized_lesion_size_anatom_norm', 'mean_hue_difference_anatom_norm', 'std_dev_contrast_anatom_norm', 'lesion_orientation_3d_anatom_norm', 'overall_color_difference_anatom_norm', 'color_variance_ratio_anatom_norm', 'border_color_interaction_anatom_norm', 'size_color_contrast_ratio_anatom_norm', 'age_normalized_nevi_confidence_anatom_norm', 'age_normalized_nevi_confidence_2_anatom_norm', 'color_asymmetry_index_anatom_norm', 'color_range_anatom_norm', 'index_age_size_symmetry_anatom_norm', 'count_per_patient_anatom_norm', 'age_approx_multi_min_max', 'clin_size_long_diam_mm_multi_min_max', 'tbp_lv_A_multi_min_max', 'tbp_lv_Aext_multi_min_max', 'tbp_lv_B_multi_min_max', 'tbp_lv_Bext_multi_min_max', 'tbp_lv_C_multi_min_max', 'tbp_lv_Cext_multi_min_max', 'tbp_lv_H_multi_min_max', 'tbp_lv_Hext_multi_min_max', 'tbp_lv_L_multi_min_max', 'tbp_lv_Lext_multi_min_max', 'tbp_lv_areaMM2_multi_min_max', 'tbp_lv_area_perim_ratio_multi_min_max', 'tbp_lv_color_std_mean_multi_min_max', 'tbp_lv_deltaA_multi_min_max', 'tbp_lv_deltaB_multi_min_max', 'tbp_lv_deltaL_multi_min_max', 'tbp_lv_deltaLB_multi_min_max', 'tbp_lv_deltaLBnorm_multi_min_max', 'tbp_lv_eccentricity_multi_min_max', 'tbp_lv_minorAxisMM_multi_min_max', 'tbp_lv_nevi_confidence_multi_min_max', 'tbp_lv_perimeterMM_multi_min_max', 'tbp_lv_radial_color_std_max_multi_min_max', 'tbp_lv_stdL_multi_min_max', 'tbp_lv_stdLExt_multi_min_max', 'tbp_lv_symm_2axis_multi_min_max', 'tbp_lv_x_multi_min_max', 'tbp_lv_y_multi_min_max', 'tbp_lv_z_multi_min_max', 'hue_contrast_multi_min_max', 'luminance_contrast_multi_min_max', 'lesion_color_difference_multi_min_max', 'border_complexity_multi_min_max', 'color_uniformity_multi_min_max', 'position_distance_3d_multi_min_max', 'perimeter_to_area_ratio_multi_min_max', 'area_to_perimeter_ratio_multi_min_max', 'lesion_visibility_score_multi_min_max', 'symmetry_border_consistency_multi_min_max', 'consistency_symmetry_border_multi_min_max', 'color_consistency_multi_min_max', 'consistency_color_multi_min_max', 'size_age_interaction_multi_min_max', 'hue_color_std_interaction_multi_min_max', 'lesion_severity_index_multi_min_max', 'color_contrast_index_multi_min_max', 'log_lesion_area_multi_min_max', 'normalized_lesion_size_multi_min_max', 'mean_hue_difference_multi_min_max', 'std_dev_contrast_multi_min_max', 'color_shape_composite_index_multi_min_max', 'lesion_orientation_3d_multi_min_max', 'overall_color_difference_multi_min_max', 'symmetry_perimeter_interaction_multi_min_max', 'comprehensive_lesion_index_multi_min_max', 'color_variance_ratio_multi_min_max', 'border_color_interaction_multi_min_max', 'border_color_interaction_2_multi_min_max', 'size_color_contrast_ratio_multi_min_max', 'age_normalized_nevi_confidence_multi_min_max', 'age_normalized_nevi_confidence_2_multi_min_max', 'color_asymmetry_index_multi_min_max', 'color_range_multi_min_max', 'shape_color_consistency_multi_min_max', 'border_length_ratio_multi_min_max', 'age_size_symmetry_index_multi_min_max', 'index_age_size_symmetry_multi_min_max', 'count_per_patient_multi_min_max']

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
    'max_depth':        5,
    'num_leaves':       103,
    'colsample_bytree': 0.8329551585827726,
    'colsample_bynode': 0.4025961355653304,
    'bagging_fraction': 0.7738954452473223,
    'bagging_freq':     4,
    'min_data_in_leaf': 85,
    'scale_pos_weight': 2.7984184778875543,
}


lgb_model = Pipeline([
   ('sampler_1', RandomOverSampler(sampling_strategy=0.003, random_state=seed)),
    ('sampler_2', RandomUnderSampler(sampling_strategy=sampling_ratio , random_state=seed)),
    ('classifier', lgb.LGBMClassifier(**lgb_params)),
])

#[I 2024-08-17 14:55:22,860] Trial 98 finished with value: 0.1710058144245913 and parameters: {'learning_rate': 0.08341853356925374, 'max_depth': 5, 'l2_leaf_reg': 6.740520715798379, 'subsample': 0.42402936337409075, 'colsample_bylevel': 0.9860546885166512, 'min_data_in_leaf': 52, 'scale_pos_weight': 2.6227279486021153}. Best is trial 98 with value: 0.1710058144245913.


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


for col in columns_to_normalize:
    df_train[f'{col}_attri_norm'] = df_train.groupby('attribution')[col].transform(
        lambda x: (x - x.mean()) / (x.std() + err))

    df_train[f'{col}_tile_type_norm'] = df_train.groupby('tbp_tile_type')[col].transform(
        lambda x: (x - x.mean()) / (x.std() + err))

    df_train[f'{col}_anatom_norm'] = df_train.groupby('anatom_site_general')[col].transform(
        lambda x: (x - x.mean()) / (x.std() + err))


for col in columns_to_normalize:
    df_train[f'{col}_multi_min_max'] = df_train.groupby(['attribution', 'tbp_tile_type', 'anatom_site_general'])[col].transform(
        lambda x: (x - x.min()) / (x.max() - x.min() + err))


train_res = calculate_normalizations(df_train, columns_to_normalize)
df_train = pd.merge(df_train, train_res, on='isic_id', how='inner')

X = df_train[feature_cols]
y = df_train[target_col]

groups = df_train[group_col]

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
sgkf_2 = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
sgkf_3 = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=24)
sgkf_4 = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=36)

sgkfs = [sgkf, sgkf_2, sgkf_3, sgkf_4]

all_done = False

cb_params = {
    'loss_function':     'Logloss',
    'iterations':        200,
    'verbose':           False,
    'random_state':      seed,
    'max_depth': 7,
    'learning_rate': 0.06936242010150652,
    'scale_pos_weight': 2.6149345838209532,
    'l2_leaf_reg': 6.216113851699493,
    'subsample': 0.6249261779711819,
    'min_data_in_leaf': 24,
    'cat_features':      [x for x in new_cat_cols if x in cat_feats],
}

cat_model = Pipeline([
   ('sampler_1', RandomOverSampler(sampling_strategy=0.003, random_state=seed)),
    ('sampler_2', RandomUnderSampler(sampling_strategy=sampling_ratio , random_state=seed)),
    ('classifier', cb.CatBoostClassifier(**cb_params)),
])

best_iter = (0, 0)
best_lgb = (0, 0)
best_xgb = (0, 0)
best_cat = (0, 0)
counter = 0
while not all_done:
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
    lgb_divisor = 25 if len(lgb_feats) > 1000 else 30
    lgb_feats_to_elim = len(feature_importances_lgb) // lgb_divisor
    lgb_feats_to_elim = max(3, lgb_feats_to_elim)
    least_important_lgb = feature_importances_lgb.tail(lgb_feats_to_elim)['feature'].tolist()
    lgb_feats = [col for col in lgb_feats if col not in least_important_lgb]

    cat_importances = cat_classifier.feature_importances_
    feature_importances_cat = pd.DataFrame({
        'feature': cat_X.columns,  # Assuming X_train is a DataFrame with named columns
        'importance': cat_importances
    }).sort_values(by='importance', ascending=False)
    cat_divisor = 25 if len(cat_feats) > 1000 else 30
    cat_feats_to_elim = len(feature_importances_cat) // cat_divisor
    cat_feats_to_elim = max(3, cat_feats_to_elim)
    least_important_cat = feature_importances_cat.tail(cat_feats_to_elim)['feature'].tolist()
    cat_feats = [col for col in cat_feats if col not in least_important_cat]

    xgb_importances = xgb_classifier.feature_importances_
    feature_importances_xgb = pd.DataFrame({
        'feature': xgb_X.columns,  # Assuming X_train is a DataFrame with named columns
        'importance': xgb_importances
    }).sort_values(by='importance', ascending=False)
    xgb_divisor = 25 if len(xgb_feats) > 1000 else 30
    xgb_feats_to_elim = len(feature_importances_xgb) // xgb_divisor
    xgb_feats_to_elim = max(3, xgb_feats_to_elim)
    least_important_xgb = feature_importances_xgb.tail(xgb_feats_to_elim)['feature'].tolist()
    xgb_feats = [col for col in xgb_feats if col not in least_important_xgb]

    print("iteration:", counter)
    print("best iter:", best_iter[0], "best score:", best_iter[1])
    print("best lgb:", best_lgb[0], "best score:", best_lgb[1])
    print("best xgb:", best_xgb[0], "best score:", best_xgb[1])
    print("best cat:", best_cat[0], "best score:", best_cat[1])
    print()
    print("lgb_feats = ", lgb_feats)
    print("cat_feats = ", cat_feats)
    print("xgb_feats = ", xgb_feats)
    print("lgb feats:", len(lgb_feats), "cat feats:", len(cat_feats), "xgb feats:", len(xgb_feats))
    print()

    cb_params = {
        'loss_function': 'Logloss',
        'iterations': 200,
        'verbose': False,
        'random_state': seed,
        'max_depth': 7,
        'learning_rate': 0.06936242010150652,
        'scale_pos_weight': 2.6149345838209532,
        'l2_leaf_reg': 6.216113851699493,
        'subsample': 0.6249261779711819,
        'min_data_in_leaf': 24,
        'cat_features': [x for x in new_cat_cols if x in cat_feats],
    }

    cat_model = Pipeline([
        ('sampler_1', RandomOverSampler(sampling_strategy=0.003, random_state=seed)),
        ('sampler_2', RandomUnderSampler(sampling_strategy=sampling_ratio, random_state=seed)),
        ('classifier', cb.CatBoostClassifier(**cb_params)),
    ])

    best_iter = (0, 0)
    best_lgb = (0, 0)
    best_xgb = (0, 0)
    best_cat = (0, 0)

    less_std_total = (0, 100000)
    less_std_lgb = (0, 100000)
    less_std_xgb = (0, 100000)
    less_std_cat = (0, 100000)

    sum_val = []
    sum_lgb = []
    sum_xgb = []
    sum_cat = []

    diff = []
    lgb_diff = []
    xgb_diff = []
    cat_diff = []

    for sgkf_id, sgkf in enumerate(sgkfs):
        # Manually perform cross-validation
        for fold, (train_idx, val_idx) in enumerate(sgkf.split(X.copy(), y.copy(), groups)):
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

            # Combine the predictions (soft voting by averaging the probabilities)
            val_preds = (lgb_preds + xgb_preds + cat_preds) / 3

            # Predict on the validation data
            val_score = custom_metric(val_preds, y_val)
            lgb_score = custom_metric(lgb_preds, y_val)
            xgb_score = custom_metric(xgb_preds, y_val)
            cat_score = custom_metric(cat_preds, y_val)

            # print(f"SGKF {sgkf_id}, Fold {fold + 1} - Validation Score: {val_score}", f"lgb: {lgb_score}", f"xgb: {xgb_score}", f"cat: {cat_score}")

            #####

            indexes = np.where(val_preds > 0.85)[0]
            X_val_filtered = X_val.iloc[indexes]
            y_val_filtered = y_val.iloc[indexes]
            if X_val_filtered.shape[0] == 0:
                sum_val.append(val_score)
                sum_lgb.append(lgb_score)
                sum_xgb.append(xgb_score)
                sum_cat.append(cat_score)
                continue

            df_new = pd.concat([X_train, X_val_filtered], axis=0)
            y_new = pd.concat([y_train, y_val_filtered], axis=0)

            # print("to add:", X_val_filtered.shape, "y to add:", y_val_filtered.shape)
            # print("old shape:", X_train.shape, "df new shape:", df_new.shape, "y new shape:", y_new.shape)
            # print(f"Fold {fold + 1} - Validation Score: {val_score}", f"lgb: {lgb_score}", f"xgb: {xgb_score}",
            #       f"cat: {cat_score}")

            lgb_X = df_new[lgb_feats]
            lgb_model.fit(lgb_X, y_new)

            cat_X = df_new[cat_feats]
            cat_model.fit(cat_X, y_new)

            xgb_X = df_new[xgb_feats]
            xgb_model.fit(xgb_X, y_new)

            pseudo_lgb_preds = lgb_model.predict_proba(X_val_lgb)[:, 1]  # Use predict_proba for soft voting
            pseudo_xgb_preds = xgb_model.predict_proba(X_val_xgb)[:, 1]
            pseudo_cat_preds = cat_model.predict_proba(X_val_cat)[:, 1]

            # Combine the predictions (soft voting by averaging the probabilities)
            pseudo_val_preds = (pseudo_lgb_preds + pseudo_xgb_preds + pseudo_cat_preds + lgb_preds + xgb_preds) / 5

            # Predict on the validation data
            pseudo_val_score = custom_metric(pseudo_val_preds, y_val)
            pseudo_lgb_score = custom_metric(pseudo_lgb_preds, y_val)
            pseudo_xgb_score = custom_metric(pseudo_xgb_preds, y_val)
            pseudo_cat_score = custom_metric(pseudo_cat_preds, y_val)

            diff.append(val_score - pseudo_val_score)
            lgb_diff.append(lgb_score - pseudo_lgb_score)
            xgb_diff.append(xgb_score - pseudo_xgb_score)
            cat_diff.append(cat_score - pseudo_cat_score)

            sum_val.append(pseudo_val_score)
            sum_lgb.append(pseudo_lgb_score)
            sum_xgb.append(pseudo_xgb_score)
            sum_cat.append(pseudo_cat_score)

            # print(f"Fold {fold + 1} - Validation Score: {pseudo_val_score}", f"lgb: {pseudo_lgb_score}",
            #       f"xgb: {pseudo_xgb_score}", f"cat: {pseudo_cat_score}")

    val_score = np.mean(sum_val)
    lgb_score = np.mean(sum_lgb)
    xgb_score = np.mean(sum_xgb)
    cat_score = np.mean(sum_cat)

    to_print = False

    if val_score > best_iter[1]:
        to_print = True
        best_iter = (counter, val_score)

    if lgb_score > best_lgb[1]:
        to_print = True
        best_lgb = (counter, lgb_score)

    if xgb_score > best_xgb[1]:
        to_print = True
        best_xgb = (counter, xgb_score)

    if cat_score > best_cat[1]:
        to_print = True
        best_cat = (counter, cat_score)

    val_std = np.std(sum_val, ddof=1)
    lgb_std = np.std(sum_lgb, ddof=1)
    xgb_std = np.std(sum_xgb, ddof=1)
    cat_std = np.std(sum_cat, ddof=1)

    if val_std < less_std_total[1]:
        to_print = True
        less_std_total = (counter, val_std)

    if lgb_std < less_std_lgb[1]:
        to_print = True
        less_std_lgb = (counter, lgb_std)

    if xgb_std < less_std_xgb[1]:
        to_print = True
        less_std_xgb = (counter, xgb_std)

    if cat_std < less_std_cat[1]:
        to_print = True
        less_std_cat = (counter, cat_std)

    if to_print:
        print("iteration:", counter)
        print("diff", np.mean(diff))
        print("lgb_diff", np.mean(lgb_diff))
        print("xgb_diff", np.mean(xgb_diff))
        print("cat_diff", np.mean(cat_diff))

        print("best iter:", best_iter[0], "best score:", best_iter[1], "less std:", less_std_total[0], "std:",
              less_std_total[1])
        print("best lgb:", best_lgb[0], "best score:", best_lgb[1], "less std:", less_std_lgb[0], "std:",
              less_std_lgb[1])
        print("best xgb:", best_xgb[0], "best score:", best_xgb[1], "less std:", less_std_xgb[0], "std:",
              less_std_xgb[1])
        print("best cat:", best_cat[0], "best score:", best_cat[1], "less std:", less_std_cat[0], "std:",
              less_std_cat[1])
        print()
        print("lgb_feats = ", lgb_feats)
        print("cat_feats = ", cat_feats)
        print("xgb_feats = ", xgb_feats)
        print("lgb feats:", len(lgb_feats), "cat feats:", len(cat_feats), "xgb feats:", len(xgb_feats))
        print()

        print(f"Average Validation Score: {val_score}", "average lgb:", lgb_score, "average xgb:", xgb_score,
              "average cat:", cat_score)
        print(f"Validation Standard Deviation: {val_std}", "std lgb:", lgb_std, "std xgb:", xgb_std, "std cat:",
              cat_std)
        # Calculate OOF score using the custom metric

        print()
        print("=======================================")

    lgb_done = len(lgb_feats) <= 30
    xgb_done = len(xgb_feats) <= 30
    cat_done = len(cat_feats) <= 30

    all_done = lgb_done and xgb_done and cat_done
    counter += 1




