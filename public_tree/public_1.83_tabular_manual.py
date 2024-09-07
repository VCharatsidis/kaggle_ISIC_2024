import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from optuna.samplers import TPESampler

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
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

from public_utils import custom_metric, set_seed
from tunning import lgb_objective, cb_objective, xgb_objective

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

set_seed(seed)

np.random.seed(seed)

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

norm_loc_cols = [f'{col}_patient_loc_norm' for col in num_cols + new_num_cols]
min_max_loc_cols = [f'{col}_patient_loc_min_max' for col in num_cols + new_num_cols]

# constrained_mean_cols = [f'{col}_constrained_mean' for col in num_cols + new_num_cols]
special_cols = ['count_per_patient']
feature_cols = num_cols + new_num_cols + cat_cols + special_cols + min_max_cols + norm_cols + norm_loc_cols #+ min_max_loc_cols

columns_to_normalize = num_cols + new_num_cols

future_norm_cols = [f'{col}_future_norm' for col in columns_to_normalize]
past_norm_cols = [f'{col}_past_norm' for col in columns_to_normalize]
future_count_cols = ['count_future']
past_count_cols = ['count_past']

feature_cols += future_norm_cols
feature_cols += past_norm_cols
feature_cols += future_count_cols
feature_cols += past_count_cols

# feature_cols += [f'{col}_basel_norm' for col in columns_to_normalize]
# feature_cols += [f'{col}_xp_norm' for col in columns_to_normalize]


def read_data(path):
    df = pl.read_csv(path)

    df = df.with_columns(pl.col('age_approx').cast(pl.String).replace('NA', np.nan).cast(pl.Float64))
    df = df.with_columns(
        pl.col(pl.Float64).fill_nan(pl.col(pl.Float64).median()))  # You may want to impute test data with train
    print("filled nan")

    df = df.with_columns(
        lesion_size_ratio=pl.col('tbp_lv_minorAxisMM') / pl.col('clin_size_long_diam_mm'),
        lesion_shape_index=pl.col('tbp_lv_areaMM2') / (pl.col('tbp_lv_perimeterMM') ** 2),
        hue_contrast=(pl.col('tbp_lv_H') - pl.col('tbp_lv_Hext')).abs(),
        luminance_contrast=(pl.col('tbp_lv_L') - pl.col('tbp_lv_Lext')).abs(),
        lesion_color_difference=(
                    pl.col('tbp_lv_deltaA') ** 2 + pl.col('tbp_lv_deltaB') ** 2 + pl.col('tbp_lv_deltaL') ** 2).sqrt(),
        border_complexity=pl.col('tbp_lv_norm_border') + pl.col('tbp_lv_symm_2axis'),
        color_uniformity=pl.col('tbp_lv_color_std_mean') / (pl.col('tbp_lv_radial_color_std_max') + err),
    )

    print("added lession_size_ratio")

    df = df.with_columns(
        position_distance_3d=(pl.col('tbp_lv_x') ** 2 + pl.col('tbp_lv_y') ** 2 + pl.col('tbp_lv_z') ** 2).sqrt(),
        perimeter_to_area_ratio=pl.col('tbp_lv_perimeterMM') / pl.col('tbp_lv_areaMM2'),
        area_to_perimeter_ratio=pl.col('tbp_lv_areaMM2') / pl.col('tbp_lv_perimeterMM'),
        lesion_visibility_score=pl.col('tbp_lv_deltaLBnorm') + pl.col('tbp_lv_norm_color'),
        symmetry_border_consistency=pl.col('tbp_lv_symm_2axis') * pl.col('tbp_lv_norm_border'),
        consistency_symmetry_border=pl.col('tbp_lv_symm_2axis') * pl.col('tbp_lv_norm_border') / (
                    pl.col('tbp_lv_symm_2axis') + pl.col('tbp_lv_norm_border')),
    )

    print("added position_distance_3d")

    df = df.with_columns(
        color_consistency=pl.col('tbp_lv_stdL') / pl.col('tbp_lv_Lext'),
        consistency_color=pl.col('tbp_lv_stdL') * pl.col('tbp_lv_Lext') / (
                    pl.col('tbp_lv_stdL') + pl.col('tbp_lv_Lext')),
        size_age_interaction=pl.col('clin_size_long_diam_mm') * pl.col('age_approx'),
        hue_color_std_interaction=pl.col('tbp_lv_H') * pl.col('tbp_lv_color_std_mean'),
        lesion_severity_index=(pl.col('tbp_lv_norm_border') + pl.col('tbp_lv_norm_color') + pl.col(
            'tbp_lv_eccentricity')) / 3,
        shape_complexity_index=pl.col('border_complexity') + pl.col('lesion_shape_index'),
        color_contrast_index=pl.col('tbp_lv_deltaA') + pl.col('tbp_lv_deltaB') + pl.col('tbp_lv_deltaL') + pl.col(
            'tbp_lv_deltaLBnorm'),
    )

    print("added color_consistency")

    df = df.with_columns(
        log_lesion_area=(pl.col('tbp_lv_areaMM2') + 1).log(),
        normalized_lesion_size=pl.col('clin_size_long_diam_mm') / pl.col('age_approx'),
        mean_hue_difference=(pl.col('tbp_lv_H') + pl.col('tbp_lv_Hext')) / 2,
        std_dev_contrast=((pl.col('tbp_lv_deltaA') ** 2 + pl.col('tbp_lv_deltaB') ** 2 + pl.col(
            'tbp_lv_deltaL') ** 2) / 3).sqrt(),
        color_shape_composite_index=(pl.col('tbp_lv_color_std_mean') + pl.col('tbp_lv_area_perim_ratio') + pl.col(
            'tbp_lv_symm_2axis')) / 3,
        lesion_orientation_3d=pl.arctan2(pl.col('tbp_lv_y'), pl.col('tbp_lv_x')),
        overall_color_difference=(pl.col('tbp_lv_deltaA') + pl.col('tbp_lv_deltaB') + pl.col('tbp_lv_deltaL')) / 3,
    )

    print("added log_lesion_area")

    df = df.with_columns(
        symmetry_perimeter_interaction=pl.col('tbp_lv_symm_2axis') * pl.col('tbp_lv_perimeterMM'),
        comprehensive_lesion_index=(pl.col('tbp_lv_area_perim_ratio') + pl.col('tbp_lv_eccentricity') + pl.col(
            'tbp_lv_norm_color') + pl.col('tbp_lv_symm_2axis')) / 4,
        color_variance_ratio=pl.col('tbp_lv_color_std_mean') / pl.col('tbp_lv_stdLExt'),
        border_color_interaction=pl.col('tbp_lv_norm_border') * pl.col('tbp_lv_norm_color'),
        border_color_interaction_2=pl.col('tbp_lv_norm_border') * pl.col('tbp_lv_norm_color') / (
                    pl.col('tbp_lv_norm_border') + pl.col('tbp_lv_norm_color')),
        size_color_contrast_ratio=pl.col('clin_size_long_diam_mm') / pl.col('tbp_lv_deltaLBnorm'),
        age_normalized_nevi_confidence=pl.col('tbp_lv_nevi_confidence') / pl.col('age_approx'),
        age_normalized_nevi_confidence_2=(pl.col('clin_size_long_diam_mm') ** 2 + pl.col('age_approx') ** 2).sqrt(),
        color_asymmetry_index=pl.col('tbp_lv_radial_color_std_max') * pl.col('tbp_lv_symm_2axis'),
    )

    print("added symmetry_perimeter_interaction")

    df = df.with_columns(
        volume_approximation_3d=pl.col('tbp_lv_areaMM2') * (
                    pl.col('tbp_lv_x') ** 2 + pl.col('tbp_lv_y') ** 2 + pl.col('tbp_lv_z') ** 2).sqrt(),
        color_range=(pl.col('tbp_lv_L') - pl.col('tbp_lv_Lext')).abs() + (
                    pl.col('tbp_lv_A') - pl.col('tbp_lv_Aext')).abs() + (
                                pl.col('tbp_lv_B') - pl.col('tbp_lv_Bext')).abs(),
        shape_color_consistency=pl.col('tbp_lv_eccentricity') * pl.col('tbp_lv_color_std_mean'),
        border_length_ratio=pl.col('tbp_lv_perimeterMM') / (2 * np.pi * (pl.col('tbp_lv_areaMM2') / np.pi).sqrt()),
        age_size_symmetry_index=pl.col('age_approx') * pl.col('clin_size_long_diam_mm') * pl.col('tbp_lv_symm_2axis'),
        index_age_size_symmetry=pl.col('age_approx') * pl.col('tbp_lv_areaMM2') * pl.col('tbp_lv_symm_2axis'),
    )

    print("added volume_approximation_3d")

    df = df.with_columns(
        count_per_patient=pl.col('isic_id').count().over('patient_id'),
    )

    print("added count_per_patient")

    for col in num_cols + new_num_cols:
        df = df.with_columns(
            pl.col(col).mean().over(['patient_id']).alias(f'{col}_mean'),
            pl.col(col).std().over(['patient_id']).alias(f'{col}_std')
        )

        df = df.with_columns(
            ((pl.col(col) - pl.col(f'{col}_mean')) / (pl.col(f'{col}_std') + err)).alias(f'{col}_patient_norm')
        )

    for col in num_cols + new_num_cols:
        df = df.with_columns(
            pl.col(col).max().over(['patient_id']).alias(f'{col}_max'),
            pl.col(col).min().over(['patient_id']).alias(f'{col}_min')
        )

        df = df.with_columns(
            ((pl.col(col) - pl.col(f'{col}_min')) / (pl.col(f'{col}_max') - pl.col(f"{col}_min") + err)).alias(f'{col}_patient_min_max')
        )

    # df = df.with_columns(
    #     (pl.col('clin_size_long_diam_mm') // 1).alias('group_perimeter')
    # )
    #
    for col in num_cols + new_num_cols:
        df = df.with_columns(
            pl.col(col).mean().over(['attribution', 'anatom_site_general', 'tbp_tile_type']).alias(f'{col}_mean'),
            pl.col(col).std().over(['attribution', 'anatom_site_general', 'tbp_tile_type']).alias(f'{col}_std')
        )

        df = df.with_columns(
            ((pl.col(col) - pl.col(f'{col}_mean')) / (pl.col(f'{col}_std') + err)).alias(f'{col}_patient_loc_norm')
        )

    # for col in num_cols + new_num_cols:
    #     df = df.with_columns(
    #         pl.col(col).max().over(['attribution', 'anatom_site_general']).alias(f'{col}_max'),
    #         pl.col(col).min().over(['attribution', 'anatom_site_general']).alias(f'{col}_min')
    #     )
    #
    #     df = df.with_columns(
    #         ((pl.col(col) - pl.col(f'{col}_min')) / (pl.col(f'{col}_max') - pl.col(f"{col}_min") + err)).alias(f'{col}_patient_loc_norm')
    #     )

    print("added patient_norm")

    df = df.with_columns(
        pl.col(cat_cols).cast(pl.Categorical),
    )

    print("make cat cols categorical")

    df = df.to_pandas()  # .set_index(id_col)

    return df


def preprocess_ordinal(df_train, df_test):
    global cat_cols
    category_encoder = OrdinalEncoder(
        categories='auto',
        dtype=int,
        handle_unknown='use_encoded_value',
        unknown_value=-2,
        encoded_missing_value=-1,
    )

    X_cat = category_encoder.fit_transform(df_train[cat_cols])
    for c, cat_col in enumerate(cat_cols):
        df_train[cat_col] = X_cat[:, c]

    X_cat = category_encoder.transform(df_test[cat_cols])
    for c, cat_col in enumerate(cat_cols):
        df_test[cat_col] = X_cat[:, c]

    return df_train, df_test


def preprocess(df_train, df_test):
    global cat_cols

    encoder = OneHotEncoder(sparse_output=False, dtype=np.int32, handle_unknown='ignore')
    encoder.fit(df_train[cat_cols])

    new_cat_cols = [f'onehot_{i}' for i in range(len(encoder.get_feature_names_out()))]

    df_train[new_cat_cols] = encoder.transform(df_train[cat_cols])
    df_train[new_cat_cols] = df_train[new_cat_cols].astype('category')

    df_test[new_cat_cols] = encoder.transform(df_test[cat_cols])
    df_test[new_cat_cols] = df_test[new_cat_cols].astype('category')

    for col in cat_cols:
        feature_cols.remove(col)

    feature_cols.extend(new_cat_cols)
    cat_cols = new_cat_cols

    return df_train, df_test


df_train = read_data(train_path)
df_test = read_data(test_path)
df_subm = pd.read_csv(subm_path, index_col=id_col)

df_train, df_test = preprocess(df_train, df_test)

zero_lgb = ['tbp_lv_perimeterMM_patient_norm', 'color_shape_composite_index_past_norm', 'onehot_2', 'border_color_interaction_past_norm', 'size_color_contrast_ratio_past_norm', 'color_asymmetry_index_past_norm', 'border_length_ratio_past_norm', 'consistency_symmetry_border_patient_min_max', 'onehot_4', 'onehot_25', 'normalized_lesion_size_patient_min_max', 'tbp_lv_stdL_patient_norm', 'size_age_interaction_future_norm', 'tbp_lv_radial_color_std_max_past_norm', 'lesion_visibility_score_past_norm', 'tbp_lv_stdL_past_norm', 'lesion_shape_index_future_norm', 'normalized_lesion_size_patient_norm', 'tbp_lv_deltaLBnorm_past_norm', 'border_complexity_past_norm', 'tbp_lv_perimeterMM_past_norm', 'symmetry_perimeter_interaction_future_norm', 'symmetry_border_consistency_past_norm', 'hue_color_std_interaction_future_norm', 'hue_color_std_interaction', 'age_size_symmetry_index_future_norm', 'border_length_ratio_patient_min_max', 'tbp_lv_perimeterMM_future_norm', 'tbp_lv_eccentricity_past_norm', 'onehot_30', 'onehot_17', 'shape_complexity_index_past_norm', 'border_length_ratio', 'onehot_43', 'shape_complexity_index_patient_norm', 'size_age_interaction_patient_norm', 'onehot_44', 'hue_color_std_interaction_past_norm', 'onehot_20', 'log_lesion_area_past_norm', 'border_length_ratio_future_norm', 'onehot_24', 'onehot_31', 'onehot_26', 'onehot_27', 'onehot_28', 'onehot_29', 'onehot_39', 'onehot_32', 'onehot_33', 'onehot_34', 'onehot_35', 'onehot_36', 'onehot_37', 'onehot_38', 'onehot_22', 'std_dev_contrast_future_norm', 'normalized_lesion_size_future_norm', 'onehot_42', 'std_dev_contrast_patient_min_max', 'onehot_23', 'onehot_10', 'onehot_21', 'onehot_18', 'normalized_lesion_size_past_norm', 'std_dev_contrast_past_norm', 'luminance_contrast_past_norm', 'std_dev_contrast_patient_norm', 'tbp_lv_norm_color_past_norm', 'age_normalized_nevi_confidence_patient_norm', 'border_length_ratio_patient_norm', 'size_age_interaction_patient_min_max', 'std_dev_contrast', 'onehot_1', 'log_lesion_area', 'tbp_lv_area_perim_ratio_future_norm', 'onehot_6', 'onehot_7', 'onehot_8', 'luminance_contrast_patient_norm', 'onehot_11', 'onehot_12', 'onehot_13', 'onehot_14', 'onehot_16', 'size_age_interaction_past_norm',
            'tbp_lv_deltaLB_future_norm', 'border_complexity_patient_min_max', 'tbp_lv_norm_border_future_norm',
            'shape_color_consistency_past_norm', 'tbp_lv_norm_border_past_norm', 'border_color_interaction_2_past_norm',
            'tbp_lv_norm_border_patient_norm', 'tbp_lv_symm_2axis_past_norm', 'tbp_lv_areaMM2_future_norm',
            'tbp_lv_deltaL_past_norm', 'tbp_lv_norm_color_patient_norm', 'age_normalized_nevi_confidence_future_norm',
            'tbp_lv_norm_color_future_norm', 'tbp_lv_Aext_past_norm', 'tbp_lv_symm_2axis_future_norm',
            'log_lesion_area_future_norm', 'shape_complexity_index_future_norm', 'consistency_color_past_norm',
            'tbp_lv_areaMM2_past_norm', 'tbp_lv_symm_2axis_angle_patient_min_max', 'onehot_19',

            'border_complexity_patient_norm', 'clin_size_long_diam_mm_future_norm',
            'comprehensive_lesion_index_past_norm', 'perimeter_to_area_ratio_past_norm',
            'color_contrast_index_future_norm', 'age_approx_patient_norm',

            'shape_color_consistency_future_norm', 'lesion_shape_index_patient_min_max',

            'hue_color_std_interaction_patient_min_max', 'tbp_lv_minorAxisMM_future_norm',
            'shape_color_consistency_patient_norm',
            'tbp_lv_color_std_mean_past_norm', 'tbp_lv_y_past_norm', 'area_to_perimeter_ratio_past_norm',
            'color_shape_composite_index', 'lesion_severity_index_future_norm', 'lesion_shape_index_patient_norm',
            'lesion_color_difference_past_norm',
            'consistency_color_patient_min_max', 'color_shape_composite_index_future_norm',
            'consistency_symmetry_border', 'consistency_color_future_norm', 'border_color_interaction',
            'tbp_lv_stdL_future_norm',

            'age_normalized_nevi_confidence_patient_min_max', 'tbp_lv_color_std_mean_patient_min_max',
            'tbp_lv_deltaLBnorm_patient_norm', 'shape_complexity_index',

            'onehot_41', 'color_consistency_past_norm', 'age_normalized_nevi_confidence_past_norm',
            'log_lesion_area_patient_norm', 'color_consistency_patient_norm', 'tbp_lv_norm_color',

            'tbp_lv_perimeterMM', 'tbp_lv_A_future_norm', 'shape_complexity_index_patient_min_max',
            'symmetry_perimeter_interaction_past_norm', 'tbp_lv_area_perim_ratio_patient_norm',

            'tbp_lv_color_std_mean_future_norm', 'consistency_symmetry_border_past_norm',
            'log_lesion_area_patient_min_max', 'tbp_lv_minorAxisMM_past_norm',
            'mean_hue_difference_past_norm', 'consistency_color',

            'tbp_lv_Cext_past_norm', 'border_color_interaction_2_patient_norm',
            'index_age_size_symmetry_past_norm', 'lesion_visibility_score_patient_min_max',

            'color_variance_ratio_past_norm', 'onehot_40', 'tbp_lv_radial_color_std_max',
            'tbp_lv_deltaL_patient_norm', 'tbp_lv_B_past_norm', 'onehot_3',

            'lesion_color_difference_future_norm', 'onehot_5', 'tbp_lv_L_past_norm',
            'comprehensive_lesion_index_patient_norm', 'lesion_visibility_score_future_norm',
            'index_age_size_symmetry_patient_norm', 'symmetry_border_consistency_patient_norm',
            'color_shape_composite_index_patient_norm', 'symmetry_perimeter_interaction_patient_norm',
            'border_complexity', 'comprehensive_lesion_index',

            'area_to_perimeter_ratio', 'color_variance_ratio_future_norm', 'perimeter_to_area_ratio_future_norm',
            'tbp_lv_areaMM2_patient_min_max', 'tbp_lv_deltaLB_patient_norm', 'color_range_future_norm',
            'tbp_lv_nevi_confidence_future_norm', 'luminance_contrast_patient_min_max',
            'volume_approximation_3d_past_norm', 'tbp_lv_stdLExt_past_norm',

            'tbp_lv_Lext_future_norm', 'lesion_severity_index_past_norm', 'tbp_lv_Hext_past_norm',
            'consistency_symmetry_border_future_norm', 'clin_size_long_diam_mm_past_norm',
            'age_normalized_nevi_confidence_2_past_norm', 'tbp_lv_C_past_norm',

            'tbp_lv_norm_color_patient_min_max', 'tbp_lv_deltaA_past_norm',

            'color_asymmetry_index_future_norm', 'luminance_contrast', 'tbp_lv_areaMM2_patient_norm',
            'tbp_lv_symm_2axis_angle', 'overall_color_difference_past_norm', 'onehot_15',
            'tbp_lv_symm_2axis_angle_past_norm', 'age_approx', 'size_color_contrast_ratio_future_norm',

            'lesion_size_ratio_future_norm', 'shape_color_consistency_patient_min_max', 'tbp_lv_x_past_norm',
            'color_shape_composite_index_patient_min_max', 'lesion_orientation_3d_patient_norm',
            'tbp_lv_deltaB_past_norm', 'tbp_lv_norm_border_patient_min_max',

            'mean_hue_difference_patient_norm', 'tbp_lv_eccentricity_future_norm', 'color_variance_ratio',
            'lesion_shape_index', 'onehot_9', 'border_color_interaction_2_patient_min_max',

            'tbp_lv_deltaLB_past_norm', 'tbp_lv_minorAxisMM_patient_min_max', 'tbp_lv_Hext_future_norm',
            'position_distance_3d_past_norm', 'lesion_severity_index_patient_min_max', 'border_color_interaction_2',
            'tbp_lv_deltaL_future_norm', 'tbp_lv_Bext_future_norm',

            'border_color_interaction_patient_min_max', 'shape_color_consistency', 'tbp_lv_C_future_norm',
            'index_age_size_symmetry', 'hue_color_std_interaction_patient_norm', 'perimeter_to_area_ratio_patient_norm',
            'tbp_lv_color_std_mean_patient_norm', 'onehot_46', 'age_approx_past_norm',
            'border_color_interaction_patient_norm', 'lesion_severity_index_patient_norm', 'tbp_lv_color_std_mean',
            'onehot_0', 'border_color_interaction_2_future_norm',
            'tbp_lv_symm_2axis_angle_basel_norm', 'volume_approximation_3d', 'tbp_lv_stdL_basel_norm',
            'symmetry_border_consistency', 'tbp_lv_symm_2axis_basel_norm', 'tbp_lv_L_basel_norm',
            'border_color_interaction_2_basel_norm', 'tbp_lv_B_basel_norm', 'tbp_lv_norm_border', 'tbp_lv_symm_2axis',
            'tbp_lv_minorAxisMM_basel_norm', 'tbp_lv_norm_color_basel_norm', 'tbp_lv_areaMM2_basel_norm',
            'color_shape_composite_index_basel_norm', 'tbp_lv_radial_color_std_max_basel_norm',
            'tbp_lv_Cext_patient_norm', 'log_lesion_area_basel_norm', 'lesion_severity_index_basel_norm',
            'consistency_color_basel_norm', 'color_contrast_index_basel_norm', 'lesion_shape_index_basel_norm',
            'symmetry_perimeter_interaction_basel_norm', 'consistency_symmetry_border_basel_norm',
            'symmetry_border_consistency_basel_norm', 'luminance_contrast_basel_norm', 'tbp_lv_deltaL_basel_norm',
            'hue_color_std_interaction_basel_norm', 'tbp_lv_eccentricity_basel_norm',
            'color_asymmetry_index_basel_norm', 'border_complexity_basel_norm', 'border_color_interaction_basel_norm',
            'lesion_size_ratio_basel_norm', 'border_length_ratio_basel_norm', 'onehot_45', 'lesion_color_difference',
            'tbp_lv_area_perim_ratio_basel_norm', 'std_dev_contrast_basel_norm',

            'tbp_lv_stdL_xp_norm', 'position_distance_3d', 'consistency_symmetry_border_xp_norm', 'tbp_lv_z',
            'border_color_interaction_xp_norm', 'color_shape_composite_index_xp_norm', 'tbp_lv_deltaA_basel_norm',
            'border_color_interaction_2_xp_norm', 'lesion_shape_index_xp_norm', 'tbp_lv_x_xp_norm',
            'symmetry_perimeter_interaction_xp_norm', 'tbp_lv_eccentricity_xp_norm', 'tbp_lv_perimeterMM_xp_norm',
            'tbp_lv_minorAxisMM_xp_norm', 'lesion_orientation_3d', 'tbp_lv_deltaL_xp_norm', 'tbp_lv_Hext_xp_norm',
            'color_asymmetry_index', 'lesion_orientation_3d_basel_norm', 'age_normalized_nevi_confidence_basel_norm',
            'hue_color_std_interaction_xp_norm', 'color_range_basel_norm', 'tbp_lv_A_basel_norm',
            'area_to_perimeter_ratio_basel_norm', 'position_distance_3d_basel_norm', 'age_size_symmetry_index_xp_norm',
            'comprehensive_lesion_index_xp_norm', 'tbp_lv_stdLExt_future_norm', 'tbp_lv_x_basel_norm',
            'size_color_contrast_ratio', 'shape_complexity_index_xp_norm', 'color_asymmetry_index_xp_norm',
            'shape_color_consistency_basel_norm', 'clin_size_long_diam_mm_xp_norm', 'luminance_contrast_xp_norm',
            'perimeter_to_area_ratio_xp_norm', 'tbp_lv_x', 'tbp_lv_areaMM2', 'tbp_lv_y', 'tbp_lv_deltaLB_basel_norm',
            'shape_color_consistency_xp_norm', 'tbp_lv_color_std_mean_xp_norm', 'color_variance_ratio_xp_norm',
            'tbp_lv_norm_color_xp_norm', 'tbp_lv_deltaLB_xp_norm', 'symmetry_border_consistency_xp_norm',
            'tbp_lv_norm_border_xp_norm', 'tbp_lv_Cext_xp_norm', 'age_approx_xp_norm',
            'area_to_perimeter_ratio_xp_norm', 'border_length_ratio_xp_norm', 'comprehensive_lesion_index_basel_norm',
            'border_complexity_xp_norm', 'tbp_lv_area_perim_ratio_xp_norm', 'log_lesion_area_xp_norm',
            'tbp_lv_areaMM2_xp_norm', 'std_dev_contrast_xp_norm',

            'tbp_lv_stdLExt_basel_norm', 'size_color_contrast_ratio_xp_norm', 'shape_complexity_index_basel_norm',
            'symmetry_perimeter_interaction', 'size_age_interaction_basel_norm', 'volume_approximation_3d_xp_norm',
            'age_approx_basel_norm', 'lesion_visibility_score_basel_norm', 'color_range_patient_min_max',
            'overall_color_difference_basel_norm', 'tbp_lv_H_past_norm',

            'color_variance_ratio_basel_norm', 'lesion_size_ratio_xp_norm', 'tbp_lv_symm_2axis_angle_xp_norm',
            'tbp_lv_deltaB_basel_norm', 'tbp_lv_Cext_basel_norm', 'tbp_lv_deltaLB', 'consistency_color_xp_norm',
            'tbp_lv_x_patient_norm', 'color_consistency_xp_norm', 'tbp_lv_stdL', 'normalized_lesion_size_basel_norm', 'tbp_lv_H_basel_norm',

            'tbp_lv_Bext_basel_norm', 'tbp_lv_radial_color_std_max_future_norm', 'color_consistency_future_norm', 'index_age_size_symmetry_patient_min_max', 'color_uniformity',
            'symmetry_border_consistency_patient_min_max', 'consistency_symmetry_border_patient_norm',
            'tbp_lv_deltaL_patient_loc_norm', 'tbp_lv_perimeterMM_patient_loc_norm',
            'comprehensive_lesion_index_patient_loc_norm', 'tbp_lv_deltaL_patient_min_max',
            'consistency_symmetry_border_patient_loc_norm', 'hue_color_std_interaction_patient_loc_norm',
            'consistency_color_patient_loc_norm', 'tbp_lv_color_std_mean_patient_loc_norm',
            'tbp_lv_area_perim_ratio_patient_loc_norm', 'tbp_lv_stdL_patient_loc_norm',
            'area_to_perimeter_ratio_patient_loc_norm', 'clin_size_long_diam_mm_patient_loc_norm',
            'color_shape_composite_index_patient_loc_norm', 'luminance_contrast_patient_loc_norm',
            'border_complexity_patient_loc_norm', 'log_lesion_area_patient_loc_norm',
            'border_color_interaction_patient_loc_norm', 'border_length_ratio_patient_loc_norm',
            'std_dev_contrast_patient_loc_norm',
            'border_color_interaction_2_patient_loc_norm', 'tbp_lv_perimeterMM_patient_min_max', 'volume_approximation_3d_patient_loc_norm',
            'color_contrast_index_patient_min_max', 'lesion_size_ratio_patient_loc_norm',
            'tbp_lv_area_perim_ratio_past_norm', 'color_range_past_norm', 'color_consistency_patient_loc_norm',
            'lesion_shape_index_past_norm', 'shape_complexity_index_patient_loc_norm',
            'area_to_perimeter_ratio_future_norm', 'tbp_lv_eccentricity',
            'lesion_visibility_score', 'tbp_lv_B_patient_min_max', 'symmetry_perimeter_interaction_patient_min_max',
            'tbp_lv_Bext_patient_norm', 'symmetry_perimeter_interaction_patient_loc_norm',
            'tbp_lv_norm_border_patient_loc_norm', 'tbp_lv_Aext_future_norm',
            'size_color_contrast_ratio_patient_loc_norm', 'tbp_lv_deltaB_patient_norm', 'tbp_lv_C_patient_min_max',
            'age_size_symmetry_index_past_norm', 'hue_contrast', 'symmetry_border_consistency_future_norm',
            'tbp_lv_radial_color_std_max_patient_loc_norm', 'tbp_lv_deltaA_future_norm',
            'color_variance_ratio_patient_min_max', 'luminance_contrast_future_norm', 'tbp_lv_area_perim_ratio',
            'tbp_lv_deltaB_future_norm', 'tbp_lv_minorAxisMM_patient_loc_norm', 'tbp_lv_L_patient_norm',
            'clin_size_long_diam_mm_patient_min_max', 'tbp_lv_deltaA_patient_loc_norm',
            'color_contrast_index_patient_loc_norm', 'perimeter_to_area_ratio_patient_loc_norm',
            'tbp_lv_stdLExt_patient_loc_norm', 'age_normalized_nevi_confidence_2_patient_loc_norm',
            'hue_contrast_future_norm', 'color_contrast_index_patient_norm', 'tbp_lv_B_patient_loc_norm',
            'tbp_lv_symm_2axis_patient_loc_norm', 'tbp_lv_area_perim_ratio_patient_min_max',
            'tbp_lv_stdL_patient_min_max', 'tbp_lv_deltaLBnorm_future_norm', 'tbp_lv_symm_2axis_patient_min_max',
            'tbp_lv_norm_color_patient_loc_norm', 'tbp_lv_A_patient_loc_norm', 'tbp_lv_areaMM2_patient_loc_norm',
            'color_uniformity_past_norm', 'tbp_lv_Cext_patient_loc_norm',
            'lesion_orientation_3d_past_norm', 'color_asymmetry_index_patient_norm',
            'tbp_lv_nevi_confidence_patient_min_max', 'color_range_patient_norm', 'volume_approximation_3d_future_norm',
            'volume_approximation_3d_patient_norm', 'lesion_size_ratio_patient_min_max',
            'color_asymmetry_index_patient_min_max', 'lesion_visibility_score_patient_loc_norm',
            'size_color_contrast_ratio_patient_min_max', 'volume_approximation_3d_patient_min_max',
            'color_range_patient_loc_norm', 'index_age_size_symmetry_future_norm', 'tbp_lv_Hext_patient_loc_norm',
            'tbp_lv_deltaLB_patient_min_max', 'clin_size_long_diam_mm_patient_norm',
            'area_to_perimeter_ratio_patient_norm', 'tbp_lv_eccentricity_patient_min_max', 'tbp_lv_Bext',
            'color_variance_ratio_patient_loc_norm', 'lesion_size_ratio_past_norm',
            'consistency_color_patient_norm', 'position_distance_3d_patient_loc_norm', 'tbp_lv_B',
            'tbp_lv_A_patient_norm', 'lesion_severity_index', 'position_distance_3d_future_norm',
            'tbp_lv_y_future_norm', 'shape_color_consistency_patient_loc_norm',
            'lesion_size_ratio_patient_norm', 'normalized_lesion_size_patient_loc_norm', 'tbp_lv_deltaLBnorm', 'tbp_lv_Cext', 'overall_color_difference_future_norm',
            'mean_hue_difference_patient_loc_norm', 'age_size_symmetry_index', 'tbp_lv_z_past_norm',
            'tbp_lv_nevi_confidence_patient_loc_norm', 'tbp_lv_x_patient_loc_norm', 'color_range',
            'tbp_lv_H_future_norm', 'color_contrast_index_past_norm', 'age_size_symmetry_index_patient_norm',
            'comprehensive_lesion_index_future_norm', 'tbp_lv_Cext_future_norm', 'tbp_lv_Lext_patient_min_max',
            'mean_hue_difference_future_norm',
            'hue_contrast_past_norm', 'tbp_lv_C_patient_norm', 'size_age_interaction_patient_loc_norm', 'tbp_lv_Aext_patient_min_max', 'tbp_lv_H_patient_loc_norm',
            'count_future', 'index_age_size_symmetry_patient_loc_norm', 'border_color_interaction_future_norm',
            'tbp_lv_Cext_patient_min_max', 'age_normalized_nevi_confidence', 'area_to_perimeter_ratio_patient_min_max',
            'tbp_lv_symm_2axis_angle_patient_loc_norm', 'lesion_color_difference_patient_loc_norm',
            'tbp_lv_A_past_norm',
            'tbp_lv_eccentricity_patient_loc_norm', 'tbp_lv_x_future_norm', 'color_uniformity_future_norm',
            'tbp_lv_deltaLBnorm_patient_min_max', 'tbp_lv_z_patient_norm', 'tbp_lv_B_patient_norm',
            'age_size_symmetry_index_patient_min_max', 'color_consistency_patient_min_max',
            'overall_color_difference_patient_loc_norm',
            'tbp_lv_L_future_norm', 'tbp_lv_deltaA_patient_norm', 'tbp_lv_symm_2axis_patient_norm', 'tbp_lv_Lext',
            'color_asymmetry_index_patient_loc_norm', 'tbp_lv_Aext_patient_loc_norm', 'hue_contrast_patient_min_max',
            'tbp_lv_minorAxisMM', 'tbp_lv_deltaB_patient_loc_norm', 'tbp_lv_Lext_patient_norm',
            'age_approx_patient_min_max', 'color_variance_ratio_patient_norm', 'tbp_lv_deltaA',
            'tbp_lv_x_patient_min_max', 'tbp_lv_z_patient_loc_norm', 'tbp_lv_H_patient_min_max', 'tbp_lv_A',
            'tbp_lv_symm_2axis_angle_future_norm', 'mean_hue_difference_patient_min_max', 'tbp_lv_A_patient_min_max',
            'tbp_lv_deltaL', 'age_size_symmetry_index_patient_loc_norm', 'size_color_contrast_ratio_patient_norm']


lgb_features = [x for x in feature_cols if x not in zero_lgb]

zero_cat = ['onehot_13', 'onehot_17', 'tbp_lv_deltaLB_patient_norm', 'onehot_42', 'onehot_12', 'onehot_18', 'lesion_shape_index_patient_min_max', 'onehot_44', 'onehot_14', 'onehot_16', 'onehot_37', 'onehot_41', 'onehot_33', 'onehot_29', 'onehot_7', 'onehot_31', 'onehot_28', 'onehot_26', 'onehot_25', 'onehot_34', 'log_lesion_area_future_norm', 'onehot_24', 'onehot_36', 'onehot_23', 'onehot_22', 'onehot_21', 'onehot_39', 'onehot_8', 'tbp_lv_y_future_norm', 'onehot_6', 'tbp_lv_deltaLB_future_norm', 'lesion_shape_index_past_norm', 'lesion_size_ratio_future_norm', 'tbp_lv_symm_2axis_past_norm', 'tbp_lv_C_future_norm', 'tbp_lv_Cext_future_norm', 'tbp_lv_Hext_future_norm', 'tbp_lv_Lext_future_norm', 'tbp_lv_norm_color', 'symmetry_border_consistency_past_norm', 'tbp_lv_deltaL_past_norm', 'tbp_lv_color_std_mean_past_norm', 'tbp_lv_area_perim_ratio_past_norm', 'tbp_lv_minorAxisMM_future_norm', 'hue_color_std_interaction', 'tbp_lv_perimeterMM_future_norm', 'tbp_lv_symm_2axis_future_norm', 'color_asymmetry_index_future_norm', 'consistency_symmetry_border_past_norm', 'onehot_5', 'consistency_symmetry_border_future_norm', 'onehot_4', 'onehot_3', 'onehot_2', 'onehot_1', 'onehot_0', 'tbp_lv_symm_2axis_patient_norm', 'tbp_lv_symm_2axis_angle_patient_min_max', 'lesion_shape_index_patient_norm', 'color_consistency_past_norm', 'border_complexity_patient_norm', 'symmetry_perimeter_interaction_past_norm', 'symmetry_border_consistency_patient_norm', 'border_color_interaction_2_future_norm', 'std_dev_contrast_past_norm', 'shape_complexity_index_past_norm', 'normalized_lesion_size_patient_norm', 'comprehensive_lesion_index_past_norm',
            'tbp_lv_norm_border_patient_norm', 'tbp_lv_deltaL_future_norm', 'onehot_27', 'onehot_35', 'onehot_38',
            'std_dev_contrast_future_norm', 'index_age_size_symmetry_patient_norm',
            'shape_complexity_index_patient_min_max', 'tbp_lv_stdL', 'tbp_lv_norm_border_patient_min_max',
            'border_complexity_past_norm', 'tbp_lv_Lext_past_norm', 'luminance_contrast',
            'tbp_lv_deltaLBnorm_past_norm', 'lesion_shape_index_future_norm', 'color_consistency_patient_norm',
            'color_contrast_index_future_norm', 'border_complexity_patient_min_max', 'border_complexity_future_norm',
            'perimeter_to_area_ratio_future_norm', 'tbp_lv_deltaL_patient_norm', 'luminance_contrast_patient_norm',
            'color_consistency_patient_min_max', 'tbp_lv_deltaLB_past_norm',
            'symmetry_perimeter_interaction_future_norm', 'hue_color_std_interaction_future_norm', 'border_complexity',

            'onehot_30', 'tbp_lv_area_perim_ratio_patient_norm', 'consistency_color_future_norm',
            'age_approx_patient_norm', 'normalized_lesion_size_past_norm', 'hue_color_std_interaction_patient_norm',
            'lesion_severity_index_patient_norm', 'tbp_lv_symm_2axis_angle', 'color_variance_ratio_patient_norm',
            'tbp_lv_deltaA_future_norm', 'clin_size_long_diam_mm_past_norm', 'shape_complexity_index_future_norm',
            'lesion_severity_index_future_norm', 'size_age_interaction_future_norm',

            'age_approx_patient_min_max', 'onehot_19', 'comprehensive_lesion_index_patient_min_max',
            'shape_color_consistency_past_norm', 'age_normalized_nevi_confidence_2_past_norm',
            'lesion_visibility_score_patient_min_max', 'color_variance_ratio_past_norm',
            'border_color_interaction_2_patient_min_max', 'consistency_symmetry_border',
            'color_consistency_future_norm', 'tbp_lv_area_perim_ratio_future_norm', 'luminance_contrast_future_norm',
            'hue_color_std_interaction_past_norm',

            'color_variance_ratio_future_norm', 'size_color_contrast_ratio_future_norm', 'tbp_lv_B_future_norm',
            'tbp_lv_Cext_patient_norm', 'luminance_contrast_patient_min_max',

            'symmetry_border_consistency', 'consistency_symmetry_border_patient_norm', 'onehot_40',
            'age_size_symmetry_index_patient_min_max', 'tbp_lv_areaMM2_patient_norm', 'lesion_size_ratio',
            'age_size_symmetry_index_past_norm', 'border_color_interaction_2_patient_norm', 'color_range_patient_norm',
            'mean_hue_difference_past_norm', 'tbp_lv_A_future_norm', 'tbp_lv_color_std_mean_future_norm',
            'tbp_lv_deltaLBnorm_future_norm', 'tbp_lv_C_past_norm', 'lesion_visibility_score_past_norm',
            'tbp_lv_deltaLB',

            'tbp_lv_norm_border', 'consistency_color_patient_min_max', 'color_range_patient_min_max',
            'tbp_lv_L_future_norm', 'tbp_lv_C_patient_min_max', 'tbp_lv_minorAxisMM_past_norm',

            'std_dev_contrast_patient_norm', 'tbp_lv_Bext_patient_norm', 'symmetry_perimeter_interaction_patient_norm',
            'size_color_contrast_ratio_patient_norm', 'normalized_lesion_size_future_norm', 'tbp_lv_Cext_past_norm',
            'shape_color_consistency',

            'size_age_interaction_past_norm', 'age_size_symmetry_index_patient_norm', 'color_range_future_norm',
            'tbp_lv_deltaLB_patient_min_max',

            'color_shape_composite_index_patient_norm', 'consistency_color', 'onehot_11', 'tbp_lv_Lext_patient_min_max',
            'border_length_ratio_patient_min_max', 'tbp_lv_areaMM2_past_norm',

            'overall_color_difference', 'shape_color_consistency_future_norm', 'tbp_lv_norm_border_future_norm',
            'tbp_lv_color_std_mean', 'clin_size_long_diam_mm_patient_min_max', 'tbp_lv_symm_2axis_angle_future_norm',

            'comprehensive_lesion_index_future_norm',

            'tbp_lv_areaMM2_future_norm', 'tbp_lv_stdL_future_norm', 'lesion_color_difference_past_norm', 'size_color_contrast_ratio_past_norm',

            'age_size_symmetry_index_future_norm', 'tbp_lv_Bext_past_norm', 'comprehensive_lesion_index_patient_norm',
            'lesion_color_difference_future_norm', 'tbp_lv_stdL_patient_min_max',

            'tbp_lv_norm_color_past_norm', 'onehot_20',

            'tbp_lv_norm_border_past_norm', 'tbp_lv_areaMM2', 'std_dev_contrast_patient_min_max',
            'tbp_lv_area_perim_ratio', 'border_color_interaction_patient_norm', 'border_length_ratio_patient_norm',
            'consistency_color_patient_norm',

            'tbp_lv_norm_color_patient_min_max', 'border_color_interaction', 'tbp_lv_area_perim_ratio_patient_min_max',
            'tbp_lv_eccentricity_patient_norm', 'lesion_severity_index',

            'border_color_interaction_2_past_norm', 'color_asymmetry_index_past_norm', 'color_contrast_index_patient_norm', 'tbp_lv_Bext',

            'tbp_lv_deltaL_patient_min_max', 'tbp_lv_Hext_patient_min_max', 'shape_complexity_index',

            'tbp_lv_nevi_confidence_past_norm', 'tbp_lv_Lext_patient_norm', 'lesion_color_difference_patient_norm',
            'tbp_lv_deltaL', 'tbp_lv_eccentricity_future_norm', 'border_color_interaction_basel_norm',
            'tbp_lv_Bext_basel_norm', 'tbp_lv_deltaL_basel_norm', 'tbp_lv_deltaLB_basel_norm',
            'hue_color_std_interaction_basel_norm', 'clin_size_long_diam_mm_patient_norm',
            'tbp_lv_norm_border_basel_norm',

            'tbp_lv_B_xp_norm', 'shape_color_consistency_xp_norm', 'color_asymmetry_index', 'tbp_lv_stdL_past_norm',
            'tbp_lv_radial_color_std_max_past_norm', 'tbp_lv_Cext_basel_norm', 'border_color_interaction_2', 'tbp_lv_A',
            'area_to_perimeter_ratio', 'consistency_symmetry_border_basel_norm', 'consistency_color_basel_norm',
            'log_lesion_area_xp_norm', 'lesion_shape_index_xp_norm', 'comprehensive_lesion_index_basel_norm',
            'shape_color_consistency_basel_norm', 'area_to_perimeter_ratio_past_norm',
            'color_uniformity_patient_min_max', 'tbp_lv_Cext', 'tbp_lv_radial_color_std_max',
            'tbp_lv_norm_border_xp_norm', 'tbp_lv_L', 'tbp_lv_symm_2axis_angle_xp_norm', 'border_length_ratio',

            'size_age_interaction_basel_norm', 'border_complexity_xp_norm', 'onehot_45',
            'border_length_ratio_basel_norm', 'size_color_contrast_ratio_patient_min_max', 'color_range_basel_norm',
            'tbp_lv_norm_color_xp_norm', 'tbp_lv_symm_2axis_angle_past_norm', 'lesion_visibility_score_basel_norm',
            'tbp_lv_norm_color_basel_norm', 'symmetry_border_consistency_basel_norm', 'luminance_contrast_xp_norm',
            'log_lesion_area', 'tbp_lv_norm_color_future_norm', 'hue_color_std_interaction_xp_norm',

            'color_asymmetry_index_basel_norm', 'consistency_symmetry_border_xp_norm', 'color_asymmetry_index_xp_norm',
            'tbp_lv_perimeterMM_basel_norm', 'tbp_lv_areaMM2_xp_norm', 'lesion_color_difference_xp_norm', 'tbp_lv_B',
            'tbp_lv_C_xp_norm', 'lesion_shape_index_basel_norm', 'area_to_perimeter_ratio_basel_norm',
            'perimeter_to_area_ratio',

            'tbp_lv_area_perim_ratio_basel_norm', 'consistency_color_xp_norm', 'tbp_lv_L_past_norm',
            'tbp_lv_deltaLB_xp_norm', 'color_variance_ratio', 'lesion_visibility_score', 'color_consistency_xp_norm',

            'tbp_lv_C_basel_norm', 'symmetry_perimeter_interaction', 'color_range', 'luminance_contrast_basel_norm', 'lesion_shape_index', 'tbp_lv_stdL_xp_norm', 'border_color_interaction_2_basel_norm',
            'tbp_lv_Bext_patient_loc_norm', 'tbp_lv_A_patient_min_max', 'log_lesion_area_patient_loc_norm',
            'color_range_patient_loc_norm', 'std_dev_contrast_patient_loc_norm', 'onehot_32',
            'tbp_lv_deltaLBnorm_patient_norm', 'tbp_lv_symm_2axis_angle_patient_norm',
            'border_color_interaction_patient_loc_norm', 'area_to_perimeter_ratio_future_norm',
            'volume_approximation_3d_patient_min_max', 'comprehensive_lesion_index',
            'symmetry_border_consistency_patient_min_max', 'tbp_lv_B_patient_loc_norm', 'count_future',
            'perimeter_to_area_ratio_past_norm', 'tbp_lv_L_patient_min_max', 'log_lesion_area_patient_norm',
            'shape_color_consistency_patient_loc_norm', 'tbp_lv_area_perim_ratio_patient_loc_norm',
            'symmetry_perimeter_interaction_patient_min_max', 'tbp_lv_color_std_mean_patient_loc_norm',
            'tbp_lv_Bext_future_norm', 'tbp_lv_deltaLB_patient_loc_norm', 'border_color_interaction_2_patient_loc_norm',
            'tbp_lv_x_future_norm', 'tbp_lv_x_past_norm', 'consistency_symmetry_border_patient_min_max',
            'log_lesion_area_patient_min_max', 'tbp_lv_perimeterMM_patient_min_max',
            'size_color_contrast_ratio_patient_loc_norm', 'clin_size_long_diam_mm_future_norm',
            'tbp_lv_Cext_patient_loc_norm', 'normalized_lesion_size_patient_loc_norm', 'tbp_lv_minorAxisMM',
            'tbp_lv_B_past_norm', 'symmetry_border_consistency_future_norm', 'tbp_lv_deltaLBnorm_patient_loc_norm',
            'luminance_contrast_patient_loc_norm', 'tbp_lv_deltaLBnorm_patient_min_max', 'luminance_contrast_past_norm',
            'tbp_lv_x_patient_norm', 'onehot_15', 'age_normalized_nevi_confidence_future_norm',
            'hue_color_std_interaction_patient_loc_norm', 'tbp_lv_stdLExt_future_norm',
            'lesion_size_ratio_patient_min_max', 'position_distance_3d_future_norm',
            'shape_color_consistency_patient_norm', 'shape_complexity_index_patient_loc_norm',
            'hue_color_std_interaction_patient_min_max', 'tbp_lv_areaMM2_patient_min_max',
            'tbp_lv_stdL_patient_loc_norm', 'border_length_ratio_future_norm', 'tbp_lv_C_patient_loc_norm',
            'log_lesion_area_past_norm', 'tbp_lv_deltaB_patient_norm', 'consistency_color_patient_loc_norm',
            'area_to_perimeter_ratio_patient_loc_norm', 'border_color_interaction_patient_min_max',
            'comprehensive_lesion_index_patient_loc_norm', 'tbp_lv_z_patient_norm', 'tbp_lv_deltaB_past_norm',
            'normalized_lesion_size_patient_min_max', 'tbp_lv_Hext_past_norm',
            'volume_approximation_3d_patient_loc_norm', 'color_contrast_index_patient_min_max',
            'tbp_lv_symm_2axis_patient_loc_norm', 'tbp_lv_A_patient_loc_norm', 'std_dev_contrast',
            'tbp_lv_color_std_mean_patient_min_max', 'color_asymmetry_index_patient_norm',
            'lesion_shape_index_patient_loc_norm', 'size_age_interaction', 'age_normalized_nevi_confidence_past_norm',
            'lesion_size_ratio_past_norm', 'tbp_lv_radial_color_std_max_patient_norm',
            'index_age_size_symmetry_past_norm', 'perimeter_to_area_ratio_patient_norm',
            'overall_color_difference_past_norm', 'tbp_lv_Aext_patient_loc_norm', 'index_age_size_symmetry_future_norm',
            'age_normalized_nevi_confidence_2_future_norm', 'shape_complexity_index_patient_norm',
            'tbp_lv_norm_color_patient_norm', 'tbp_lv_eccentricity', 'tbp_lv_Bext_patient_min_max',
            'mean_hue_difference_patient_norm', 'border_length_ratio_past_norm',
            'clin_size_long_diam_mm_patient_loc_norm', 'tbp_lv_Aext',
            'tbp_lv_Cext_patient_min_max', 'tbp_lv_nevi_confidence_future_norm', 'tbp_lv_Aext_patient_min_max', 'color_shape_composite_index_past_norm', 'lesion_orientation_3d',
            'tbp_lv_norm_border_patient_loc_norm', 'tbp_lv_Aext_future_norm',
            'size_color_contrast_ratio_patient_loc_norm', 'tbp_lv_deltaB_patient_norm', 'tbp_lv_C_patient_min_max',
            'age_size_symmetry_index_past_norm', 'hue_contrast', 'symmetry_border_consistency_future_norm',
            'tbp_lv_radial_color_std_max_patient_loc_norm', 'tbp_lv_deltaA_future_norm',
            'color_variance_ratio_patient_min_max', 'luminance_contrast_future_norm', 'tbp_lv_area_perim_ratio',
            'tbp_lv_deltaB_future_norm', 'tbp_lv_minorAxisMM_patient_loc_norm', 'tbp_lv_L_patient_norm',
            'clin_size_long_diam_mm_patient_min_max', 'tbp_lv_deltaA_patient_loc_norm',
            'color_contrast_index_patient_loc_norm', 'perimeter_to_area_ratio_patient_loc_norm',
            'tbp_lv_stdLExt_patient_loc_norm', 'age_normalized_nevi_confidence_2_patient_loc_norm',
            'hue_contrast_future_norm', 'color_contrast_index_patient_norm', 'tbp_lv_B_patient_loc_norm',
            'tbp_lv_symm_2axis_patient_loc_norm', 'tbp_lv_area_perim_ratio_patient_min_max',
            'tbp_lv_stdL_patient_min_max', 'tbp_lv_deltaLBnorm_future_norm', 'tbp_lv_symm_2axis_patient_min_max',
            'tbp_lv_norm_color_patient_loc_norm', 'tbp_lv_A_patient_loc_norm', 'tbp_lv_areaMM2_patient_loc_norm',
            'color_uniformity_past_norm', 'tbp_lv_Cext_patient_loc_norm',
            'tbp_lv_C', 'overall_color_difference_patient_min_max', 'tbp_lv_A_past_norm', 'age_approx_patient_loc_norm',
            'age_normalized_nevi_confidence', 'tbp_lv_norm_color_patient_loc_norm',
            'symmetry_border_consistency_patient_loc_norm', 'lesion_visibility_score_patient_loc_norm',
            'tbp_lv_deltaB_patient_loc_norm', 'tbp_lv_Aext_patient_norm', 'tbp_lv_norm_border_patient_loc_norm',
            'age_normalized_nevi_confidence_patient_norm', 'tbp_lv_minorAxisMM_patient_loc_norm',
            'border_color_interaction_past_norm', 'tbp_lv_stdL_patient_norm', 'mean_hue_difference_patient_loc_norm',
            'color_shape_composite_index', 'tbp_lv_deltaL_patient_loc_norm', 'tbp_lv_perimeterMM_patient_loc_norm',
            'tbp_lv_C_patient_norm', 'volume_approximation_3d_past_norm', 'consistency_color_past_norm',
            'tbp_lv_deltaA_patient_norm', 'tbp_lv_stdLExt_past_norm', 'tbp_lv_symm_2axis_patient_min_max',
            'tbp_lv_y_past_norm', 'tbp_lv_y', 'lesion_severity_index_past_norm',
            'shape_color_consistency_patient_min_max', 'tbp_lv_symm_2axis', 'lesion_severity_index_patient_min_max',
            'color_contrast_index_patient_loc_norm', 'overall_color_difference_patient_loc_norm',
            'size_age_interaction_patient_min_max', 'tbp_lv_perimeterMM_past_norm',
            'tbp_lv_minorAxisMM_patient_min_max', 'tbp_lv_color_std_mean_patient_norm',
            'consistency_symmetry_border_patient_loc_norm', 'tbp_lv_B_patient_min_max',
            'color_consistency_patient_loc_norm', 'overall_color_difference_future_norm', 'tbp_lv_y_patient_loc_norm',
            'tbp_lv_areaMM2_patient_loc_norm', 'tbp_lv_nevi_confidence_patient_loc_norm',
            'volume_approximation_3d_future_norm', 'lesion_orientation_3d_patient_norm',
            'tbp_lv_eccentricity_patient_min_max', 'tbp_lv_L_patient_loc_norm', 'tbp_lv_Lext_patient_loc_norm',
            'tbp_lv_radial_color_std_max_future_norm', 'lesion_orientation_3d_past_norm',
            'volume_approximation_3d_patient_norm', 'position_distance_3d_past_norm', 'tbp_lv_z_patient_loc_norm',
            'border_length_ratio_patient_loc_norm',
            'color_variance_ratio_patient_loc_norm', 'tbp_lv_B_patient_norm', 'onehot_43', 'tbp_lv_x_patient_min_max', 'lesion_size_ratio_patient_loc_norm',
            'tbp_lv_stdLExt_patient_min_max', 'color_range_past_norm', 'position_distance_3d_patient_loc_norm',
            'size_age_interaction_patient_loc_norm', 'tbp_lv_z_future_norm', 'tbp_lv_Hext_patient_loc_norm',
            'age_size_symmetry_index', 'tbp_lv_H_past_norm',
            'mean_hue_difference_patient_min_max', 'mean_hue_difference_future_norm', 'tbp_lv_Aext_past_norm',
            'tbp_lv_eccentricity_patient_loc_norm', 'area_to_perimeter_ratio_patient_min_max',
            'area_to_perimeter_ratio_patient_norm', 'age_normalized_nevi_confidence_2_patient_min_max',
            'symmetry_perimeter_interaction_patient_loc_norm', 'lesion_orientation_3d_patient_min_max', 'tbp_lv_Hext',
            'tbp_lv_nevi_confidence_patient_min_max', 'lesion_orientation_3d_patient_loc_norm',
            'index_age_size_symmetry_patient_min_max', 'size_age_interaction_patient_norm',
            'tbp_lv_deltaB', 'color_shape_composite_index_patient_min_max', 'tbp_lv_z_patient_min_max', 'tbp_lv_deltaB_patient_min_max', 'color_asymmetry_index_patient_loc_norm',
            'color_contrast_index', 'age_approx_future_norm', 'lesion_color_difference_patient_min_max', 'lesion_size_ratio_patient_norm', 'onehot_10',
            'hue_contrast_patient_min_max', 'tbp_lv_A_patient_norm', 'tbp_lv_symm_2axis_angle_patient_loc_norm',
            'border_color_interaction_future_norm', 'position_distance_3d']



cat_features = [x for x in feature_cols if x not in zero_cat]

zero_importance_xgb = ['tbp_lv_deltaA_future_norm', 'shape_complexity_index', 'shape_complexity_index_past_norm', 'tbp_lv_deltaLBnorm_past_norm', 'age_size_symmetry_index_past_norm', 'shape_complexity_index_patient_norm', 'tbp_lv_Bext_patient_norm', 'consistency_color', 'hue_color_std_interaction_past_norm', 'tbp_lv_Bext_past_norm', 'tbp_lv_eccentricity_patient_norm', 'tbp_lv_norm_border_future_norm', 'shape_color_consistency_patient_norm', 'onehot_27', 'tbp_lv_eccentricity', 'onehot_20', 'luminance_contrast_future_norm', 'tbp_lv_symm_2axis_past_norm', 'color_consistency_future_norm', 'onehot_0', 'onehot_19', 'tbp_lv_norm_border_past_norm', 'tbp_lv_deltaLB_patient_norm', 'onehot_36', 'tbp_lv_symm_2axis_angle_patient_min_max', 'onehot_1', 'consistency_symmetry_border_past_norm', 'onehot_2', 'luminance_contrast_patient_norm', 'onehot_17', 'onehot_35', 'lesion_shape_index', 'shape_complexity_index_patient_min_max', 'onehot_13', 'onehot_14', 'luminance_contrast', 'onehot_22', 'onehot_37', 'onehot_38', 'onehot_39', 'onehot_29', 'onehot_34', 'onehot_32', 'onehot_31', 'std_dev_contrast_patient_norm', 'log_lesion_area', 'std_dev_contrast_past_norm', 'onehot_28', 'onehot_26', 'luminance_contrast_past_norm', 'border_length_ratio_past_norm', 'onehot_3', 'onehot_6', 'std_dev_contrast_patient_min_max', 'onehot_10', 'onehot_11', 'onehot_12', 'normalized_lesion_size_past_norm', 'onehot_16', 'onehot_18', 'onehot_21', 'onehot_23', 'onehot_24', 'onehot_25', 'std_dev_contrast_future_norm',
                       'color_shape_composite_index_past_norm', 'symmetry_border_consistency_future_norm',
                       'age_approx_past_norm',
                       'size_color_contrast_ratio_past_norm', 'area_to_perimeter_ratio_past_norm',
                       'shape_color_consistency_past_norm', 'tbp_lv_deltaL_past_norm', 'lesion_shape_index_future_norm',
                       'area_to_perimeter_ratio', 'shape_complexity_index_future_norm', 'onehot_42', 'onehot_7',
                       'std_dev_contrast',

                        'border_complexity', 'tbp_lv_deltaLB_past_norm', 'tbp_lv_deltaA_past_norm', 'border_length_ratio_future_norm',
                       'border_length_ratio', 'age_approx_patient_norm',

                        'border_complexity_patient_norm', 'border_complexity_past_norm', 'onehot_8', 'onehot_33',

                       'consistency_symmetry_border_future_norm', 'border_color_interaction_2_past_norm',
                       'log_lesion_area_future_norm', 'tbp_lv_symm_2axis_patient_norm',
                       'border_complexity_patient_min_max', 'tbp_lv_radial_color_std_max_past_norm',
                       'normalized_lesion_size_future_norm',

                       'clin_size_long_diam_mm_future_norm',

                       'symmetry_border_consistency_patient_norm', 'tbp_lv_x_patient_min_max',
                       'volume_approximation_3d_patient_min_max', 'onehot_40',
                       'symmetry_border_consistency_patient_min_max', 'lesion_orientation_3d_future_norm',
                       'color_uniformity_patient_min_max', 'tbp_lv_Cext_patient_min_max', 'tbp_lv_Bext_patient_min_max',
                       'lesion_orientation_3d_past_norm', 'area_to_perimeter_ratio_future_norm',
                       'lesion_color_difference_past_norm', 'color_variance_ratio_patient_min_max',
                       'color_shape_composite_index_patient_min_max', 'tbp_lv_C_future_norm',
                       'tbp_lv_z_patient_min_max', 'tbp_lv_Cext', 'color_range_past_norm',
                       'hue_color_std_interaction_future_norm', 'tbp_lv_Hext_patient_min_max',
                       'tbp_lv_L_patient_min_max', 'tbp_lv_x', 'luminance_contrast_patient_min_max',
                       'age_normalized_nevi_confidence_future_norm', 'tbp_lv_nevi_confidence_patient_norm',
                       'log_lesion_area_past_norm', 'tbp_lv_z_past_norm', 'tbp_lv_symm_2axis_angle',
                       'tbp_lv_deltaLBnorm_patient_norm', 'border_length_ratio_patient_min_max',
                       'tbp_lv_area_perim_ratio_patient_min_max', 'tbp_lv_symm_2axis_angle_patient_norm',
                       'tbp_lv_A_patient_min_max', 'shape_color_consistency_future_norm',
                       'border_color_interaction_patient_min_max', 'border_color_interaction_2_patient_norm',
                       'lesion_orientation_3d_patient_min_max', 'tbp_lv_symm_2axis_patient_min_max',
                       'lesion_size_ratio_past_norm', 'overall_color_difference_patient_min_max', 'tbp_lv_stdL',
                       'tbp_lv_stdLExt_future_norm', 'tbp_lv_areaMM2_patient_norm', 'tbp_lv_C_patient_norm',
                       'tbp_lv_color_std_mean_future_norm', 'tbp_lv_area_perim_ratio',
                       'age_normalized_nevi_confidence_patient_norm', 'symmetry_border_consistency',
                       'tbp_lv_z_future_norm', 'color_consistency_patient_norm', 'tbp_lv_Aext_future_norm',
                       'border_color_interaction_patient_norm', 'tbp_lv_stdL_patient_min_max',
                       'consistency_color_past_norm', 'tbp_lv_deltaL_patient_min_max',
                       'lesion_severity_index_future_norm', 'tbp_lv_Lext_patient_norm', 'lesion_size_ratio',
                       'size_color_contrast_ratio_patient_norm', 'tbp_lv_A_past_norm', 'tbp_lv_deltaLB_patient_min_max',
                       'border_complexity_future_norm', 'tbp_lv_minorAxisMM_future_norm',
                       'color_variance_ratio_patient_norm', 'color_contrast_index_past_norm', 'tbp_lv_Hext_future_norm',
                       'lesion_visibility_score', 'tbp_lv_Aext', 'tbp_lv_A', 'tbp_lv_nevi_confidence_patient_min_max',
                       'tbp_lv_Bext', 'tbp_lv_Hext_past_norm', 'symmetry_perimeter_interaction_future_norm',
                       'hue_contrast_past_norm', 'symmetry_perimeter_interaction',
                       'age_size_symmetry_index_patient_min_max', 'symmetry_border_consistency_past_norm', 'tbp_lv_z',
                       'tbp_lv_x_patient_norm', 'tbp_lv_Aext_patient_norm', 'lesion_size_ratio_future_norm',
                       'tbp_lv_deltaL', 'border_color_interaction_2', 'count_past', 'tbp_lv_deltaLB_future_norm',
                       'color_contrast_index', 'tbp_lv_Aext_past_norm', 'tbp_lv_Aext_patient_min_max',
                       'symmetry_perimeter_interaction_past_norm', 'color_contrast_index_patient_norm',
                       'tbp_lv_Hext_patient_norm', 'tbp_lv_x_past_norm', 'tbp_lv_symm_2axis_angle_past_norm',
                       'consistency_color_future_norm', 'tbp_lv_eccentricity_future_norm',
                       'tbp_lv_Lext_patient_min_max', 'area_to_perimeter_ratio_patient_min_max',
                       'tbp_lv_Bext_future_norm', 'log_lesion_area_patient_min_max', 'consistency_symmetry_border',
                       'consistency_symmetry_border_patient_min_max', 'tbp_lv_y_past_norm',
                       'size_color_contrast_ratio_future_norm', 'tbp_lv_areaMM2_future_norm', 'tbp_lv_Cext_future_norm',
                       'border_color_interaction_2_future_norm', 'tbp_lv_Lext_future_norm',
                       'tbp_lv_color_std_mean_past_norm', 'lesion_shape_index_patient_norm',
                       'tbp_lv_area_perim_ratio_past_norm', 'color_consistency_past_norm',
                       'tbp_lv_nevi_confidence_future_norm', 'tbp_lv_area_perim_ratio_patient_norm',
                       'color_uniformity_past_norm', 'tbp_lv_deltaLB', 'tbp_lv_eccentricity_past_norm',
                       'tbp_lv_stdLExt_past_norm', 'tbp_lv_symm_2axis_angle_future_norm',
                       'border_color_interaction_future_norm', 'tbp_lv_C_patient_min_max', 'tbp_lv_deltaL_future_norm',
                       'position_distance_3d_past_norm', 'lesion_shape_index_patient_min_max',
                       'comprehensive_lesion_index_patient_min_max', 'tbp_lv_symm_2axis_future_norm',
                       'consistency_symmetry_border_patient_norm', 'tbp_lv_C', 'tbp_lv_A_future_norm',
                       'tbp_lv_Cext_past_norm', 'color_contrast_index_patient_min_max',
                       'age_size_symmetry_index_future_norm', 'color_asymmetry_index_past_norm',
                       'tbp_lv_deltaLBnorm_future_norm', 'onehot_41', 'lesion_severity_index_past_norm',
                       'lesion_size_ratio_patient_min_max', 'tbp_lv_B_past_norm',
                       'comprehensive_lesion_index_past_norm', 'lesion_size_ratio_patient_norm',
                       'tbp_lv_stdL_past_norm', 'tbp_lv_stdL_patient_norm', 'tbp_lv_norm_border_patient_min_max',

                       'lesion_shape_index_past_norm', 'onehot_44', 'border_length_ratio_patient_norm',
                       'comprehensive_lesion_index',

                       'tbp_lv_Lext_past_norm', 'lesion_orientation_3d_patient_norm', 'border_color_interaction',
                       'tbp_lv_nevi_confidence', 'count_future', 'tbp_lv_area_perim_ratio_future_norm',
                       'lesion_visibility_score_patient_min_max', 'color_uniformity_patient_norm',
                       'age_normalized_nevi_confidence_past_norm', 'symmetry_perimeter_interaction_patient_min_max',
                       'lesion_color_difference_patient_min_max', 'color_range_patient_min_max',
                       'comprehensive_lesion_index_future_norm', 'perimeter_to_area_ratio_past_norm',
                       'tbp_lv_C_past_norm', 'tbp_lv_stdL_future_norm', 'lesion_orientation_3d',
                       'tbp_lv_minorAxisMM_past_norm', 'tbp_lv_symm_2axis', 'perimeter_to_area_ratio_future_norm',
                       'tbp_lv_deltaB_patient_norm', 'tbp_lv_deltaL_patient_norm', 'tbp_lv_areaMM2_past_norm',
                       'tbp_lv_norm_border_patient_norm', 'tbp_lv_x_future_norm', 'tbp_lv_eccentricity_patient_min_max',
                       'age_size_symmetry_index_patient_norm', 'lesion_severity_index_patient_min_max',
                       'color_variance_ratio_future_norm', 'tbp_lv_Cext_patient_norm', 'color_range_future_norm',
                       'color_variance_ratio_past_norm', 'tbp_lv_norm_border', 'hue_contrast_future_norm',
                       'tbp_lv_deltaLBnorm_patient_min_max', 'onehot_30',

                       'tbp_lv_color_std_mean_patient_min_max', 'tbp_lv_norm_color_past_norm',

                       'index_age_size_symmetry_future_norm', 'onehot_43', 'color_asymmetry_index_patient_min_max',
                       'color_range', 'position_distance_3d_patient_min_max', 'tbp_lv_H_past_norm',
                       'age_normalized_nevi_confidence_2_patient_min_max', 'hue_contrast', 'hue_contrast_patient_norm',
                       'tbp_lv_norm_color_future_norm', 'lesion_visibility_score_future_norm',
                       'age_normalized_nevi_confidence', 'color_shape_composite_index', 'hue_color_std_interaction',
                       'color_shape_composite_index_future_norm', 'tbp_lv_B', 'count_per_patient',
                       'size_color_contrast_ratio_patient_min_max', 'color_shape_composite_index_patient_norm',
                       'lesion_severity_index', 'color_consistency_patient_min_max', 'mean_hue_difference_past_norm',
                       'tbp_lv_L', 'age_approx', 'color_range_patient_norm', 'tbp_lv_deltaB_patient_min_max',
                       'tbp_lv_B_patient_norm', 'tbp_lv_y', 'color_asymmetry_index_patient_norm',
                       'tbp_lv_L_future_norm', 'volume_approximation_3d_future_norm',
                       'tbp_lv_perimeterMM_patient_min_max', 'tbp_lv_Hext', 'position_distance_3d_future_norm',
                       'tbp_lv_deltaB', 'lesion_color_difference_future_norm', 'tbp_lv_z_patient_norm',
                       'index_age_size_symmetry_past_norm', 'tbp_lv_deltaA_patient_min_max',
                       'symmetry_perimeter_interaction_patient_norm', 'volume_approximation_3d_patient_norm',
                       'color_asymmetry_index', 'hue_color_std_interaction_patient_min_max',
                       'overall_color_difference_past_norm', 'volume_approximation_3d_past_norm',
                       'tbp_lv_y_future_norm', 'mean_hue_difference_future_norm', 'mean_hue_difference_patient_min_max',
                       'consistency_color_patient_norm', 'tbp_lv_deltaB_past_norm', 'color_contrast_index_future_norm',
                       'age_normalized_nevi_confidence_patient_min_max', 'tbp_lv_B_future_norm',
                       'tbp_lv_B_patient_min_max', 'color_uniformity', 'color_consistency', 'tbp_lv_Lext',
                       'hue_contrast_patient_min_max', 'color_uniformity_future_norm', 'size_age_interaction',
                       'tbp_lv_L_patient_norm', 'tbp_lv_stdLExt', 'tbp_lv_nevi_confidence_past_norm',
                       'volume_approximation_3d', 'tbp_lv_deltaA', 'tbp_lv_A_patient_norm', 'color_variance_ratio',
                       'age_normalized_nevi_confidence_2_past_norm', 'tbp_lv_stdLExt_patient_min_max',
                       'tbp_lv_color_std_mean', 'consistency_color_patient_min_max', 'overall_color_difference',
                       'onehot_46', 'tbp_lv_deltaA_patient_norm', 'comprehensive_lesion_index_patient_norm',
                       'size_color_contrast_ratio', 'tbp_lv_stdLExt_patient_norm', 'border_color_interaction_past_norm',
                       'tbp_lv_L_past_norm', 'shape_color_consistency_patient_min_max', 'onehot_15',
                       'age_normalized_nevi_confidence_basel_norm', 'mean_hue_difference_basel_norm',
                       'lesion_severity_index_patient_norm', 'tbp_lv_radial_color_std_max_future_norm',
                       'age_normalized_nevi_confidence_2_basel_norm', 'lesion_severity_index_basel_norm',
                       'age_approx_basel_norm', 'age_normalized_nevi_confidence_2_future_norm',
                       'overall_color_difference_basel_norm', 'tbp_lv_deltaB_basel_norm',
                       'mean_hue_difference_patient_norm', 'color_consistency_basel_norm',
                       'tbp_lv_areaMM2_patient_min_max', 'lesion_visibility_score_past_norm', 'tbp_lv_B_basel_norm',
                       'tbp_lv_nevi_confidence_basel_norm', 'overall_color_difference_future_norm',
                       'tbp_lv_H_basel_norm', 'border_complexity_basel_norm', 'tbp_lv_color_std_mean_patient_norm',
                       'onehot_4', 'log_lesion_area_basel_norm', 'tbp_lv_deltaB_future_norm', 'position_distance_3d',
                       'color_variance_ratio_basel_norm', 'border_color_interaction_basel_norm',
                       'tbp_lv_norm_border_basel_norm', 'tbp_lv_Lext_basel_norm', 'lesion_orientation_3d_basel_norm',
                       'hue_color_std_interaction_basel_norm', 'tbp_lv_stdLExt_basel_norm',
                       'tbp_lv_radial_color_std_max', 'tbp_lv_symm_2axis_basel_norm', 'tbp_lv_L_basel_norm',
                       'tbp_lv_area_perim_ratio_basel_norm', 'tbp_lv_z_basel_norm',
                       'age_size_symmetry_index_basel_norm', 'size_age_interaction_basel_norm', 'tbp_lv_x_basel_norm',
                       'comprehensive_lesion_index_basel_norm', 'tbp_lv_deltaA_basel_norm',
                       'color_contrast_index_basel_norm', 'tbp_lv_Hext_basel_norm',
                       'color_shape_composite_index_basel_norm', 'size_color_contrast_ratio_basel_norm',
                       'tbp_lv_color_std_mean_basel_norm', 'symmetry_border_consistency_basel_norm',
                       'tbp_lv_A_basel_norm', 'consistency_symmetry_border_basel_norm', 'tbp_lv_deltaL_basel_norm',
                       'tbp_lv_Cext_basel_norm', 'tbp_lv_deltaLB_basel_norm', 'tbp_lv_stdL_basel_norm',
                       'tbp_lv_symm_2axis_angle_basel_norm', 'shape_complexity_index_basel_norm',
                       'lesion_visibility_score_basel_norm', 'tbp_lv_C_basel_norm', 'lesion_size_ratio_basel_norm',
                       'consistency_color_basel_norm', 'tbp_lv_Bext_basel_norm', 'lesion_shape_index_basel_norm',
                       'tbp_lv_Aext_basel_norm', 'shape_color_consistency', 'shape_color_consistency_basel_norm',
                       'tbp_lv_eccentricity_basel_norm', 'symmetry_perimeter_interaction_basel_norm',
                       'index_age_size_symmetry_basel_norm', 'tbp_lv_norm_color', 'border_length_ratio_basel_norm',
                       'luminance_contrast_basel_norm', 'std_dev_contrast_basel_norm', 'onehot_45',

                       'hue_contrast_xp_norm', 'tbp_lv_perimeterMM_basel_norm', 'age_approx_xp_norm',
                       'border_complexity_xp_norm', 'lesion_color_difference_xp_norm',
                       'shape_color_consistency_xp_norm', 'tbp_lv_H_xp_norm', 'comprehensive_lesion_index_xp_norm',
                       'age_normalized_nevi_confidence_xp_norm', 'normalized_lesion_size',
                       'overall_color_difference_xp_norm', 'border_color_interaction_2_xp_norm',
                       'border_color_interaction_xp_norm', 'color_asymmetry_index_basel_norm', 'tbp_lv_deltaL_xp_norm',
                       'size_age_interaction_xp_norm', 'tbp_lv_y_xp_norm', 'lesion_severity_index_xp_norm',
                       'lesion_visibility_score_xp_norm', 'color_range_xp_norm', 'color_variance_ratio_xp_norm',
                       'clin_size_long_diam_mm_patient_min_max', 'border_color_interaction_2_basel_norm',
                       'consistency_color_xp_norm', 'log_lesion_area_xp_norm', 'tbp_lv_deltaB_xp_norm',
                       'tbp_lv_deltaLB_xp_norm', 'tbp_lv_stdLExt_xp_norm', 'size_color_contrast_ratio_xp_norm',
                       'tbp_lv_deltaA_xp_norm', 'tbp_lv_stdL_xp_norm', 'size_age_interaction_past_norm',
                       'tbp_lv_Lext_xp_norm', 'tbp_lv_z_xp_norm', 'tbp_lv_L_xp_norm', 'lesion_orientation_3d_xp_norm',
                       'color_consistency_xp_norm', 'tbp_lv_Aext_xp_norm', 'tbp_lv_x_xp_norm',
                       'tbp_lv_nevi_confidence_xp_norm', 'color_asymmetry_index_xp_norm', 'age_size_symmetry_index',
                       'tbp_lv_eccentricity_xp_norm', 'tbp_lv_Bext_xp_norm', 'color_shape_composite_index_xp_norm',
                       'lesion_size_ratio_xp_norm', 'tbp_lv_symm_2axis_xp_norm', 'color_contrast_index_xp_norm',
                       'tbp_lv_norm_border_xp_norm', 'tbp_lv_A_xp_norm', 'luminance_contrast_xp_norm',
                       'position_distance_3d_xp_norm', 'symmetry_perimeter_interaction_xp_norm', 'tbp_lv_C_xp_norm',
                       'tbp_lv_symm_2axis_angle_xp_norm', 'tbp_lv_Hext_xp_norm', 'hue_color_std_interaction_xp_norm',
                       'symmetry_border_consistency_xp_norm', 'clin_size_long_diam_mm_past_norm',
                       'lesion_shape_index_xp_norm', 'consistency_symmetry_border_xp_norm', 'tbp_lv_Cext_xp_norm',
                       'normalized_lesion_size_patient_norm', 'tbp_lv_color_std_mean_xp_norm',
                       'tbp_lv_area_perim_ratio_xp_norm', 'border_length_ratio_xp_norm',
                       'shape_complexity_index_xp_norm', 'tbp_lv_radial_color_std_max_basel_norm',
                       'std_dev_contrast_xp_norm',

'volume_approximation_3d_patient_loc_norm', 'hue_color_std_interaction_patient_loc_norm', 'tbp_lv_deltaLB_patient_loc_norm', 'tbp_lv_B_patient_loc_norm', 'comprehensive_lesion_index_patient_loc_norm', 'tbp_lv_x_patient_loc_norm', 'tbp_lv_stdLExt_patient_loc_norm', 'tbp_lv_stdL_patient_loc_norm', 'tbp_lv_deltaA_patient_loc_norm', 'tbp_lv_C_patient_loc_norm', 'border_complexity_patient_loc_norm', 'color_consistency_patient_loc_norm', 'tbp_lv_nevi_confidence_patient_loc_norm', 'color_variance_ratio_patient_loc_norm', 'size_color_contrast_ratio_patient_loc_norm', 'lesion_shape_index_patient_loc_norm', 'overall_color_difference_patient_loc_norm', 'tbp_lv_eccentricity_patient_loc_norm', 'tbp_lv_symm_2axis_patient_loc_norm', 'tbp_lv_Aext_patient_loc_norm', 'symmetry_perimeter_interaction_patient_loc_norm', 'tbp_lv_A_patient_loc_norm', 'color_contrast_index_patient_loc_norm', 'tbp_lv_y_patient_loc_norm', 'position_distance_3d_patient_loc_norm', 'tbp_lv_symm_2axis_angle_patient_loc_norm', 'tbp_lv_L_patient_loc_norm', 'tbp_lv_z_patient_loc_norm', 'shape_color_consistency_patient_loc_norm', 'tbp_lv_Lext_patient_loc_norm', 'tbp_lv_area_perim_ratio_patient_loc_norm', 'lesion_orientation_3d_patient_loc_norm', 'tbp_lv_Hext_patient_loc_norm', 'tbp_lv_Cext_patient_loc_norm', 'border_color_interaction_2_patient_loc_norm', 'tbp_lv_Bext_patient_loc_norm', 'lesion_size_ratio_patient_loc_norm', 'consistency_color_patient_loc_norm', 'tbp_lv_deltaL_patient_loc_norm', 'consistency_symmetry_border_patient_loc_norm', 'border_length_ratio_patient_loc_norm', 'luminance_contrast_patient_loc_norm', 'shape_complexity_index_patient_loc_norm', 'std_dev_contrast_patient_loc_norm'
, 'tbp_lv_areaMM2_patient_loc_norm', 'onehot_5',
'mean_hue_difference_patient_loc_norm', 'tbp_lv_radial_color_std_max_patient_norm', 'symmetry_border_consistency_patient_loc_norm', 'color_shape_composite_index_patient_loc_norm', 'tbp_lv_norm_border_patient_loc_norm', 'lesion_visibility_score_patient_loc_norm',
                       'tbp_lv_minorAxisMM_patient_loc_norm'
                       ]



xgb_features = [x for x in feature_cols if x not in zero_importance_xgb]

print(f"lgb_features: {len(lgb_features)}")
print(f"cat_features: {len(cat_features)}")
print(f"xgb_features: {len(xgb_features)}")

# [I 2024-08-17 11:47:47,716] Trial 22 finished with value: 0.1729958846162828 and parameters: {'n_iter': 242, 'lambda_l1': 0.05356915080283596, 'lambda_l2': 0.03548695950893246, 'learning_rate': 0.015844359470309228, 'max_depth': 5, 'num_leaves': 116, 'colsample_bytree': 0.7105075894232431, 'colsample_bynode': 0.6055829227054091, 'bagging_fraction': 0.5177655386536087, 'bagging_freq': 6, 'min_data_in_leaf': 68, 'scale_pos_weight': 2.1125855926322323}. Best is trial 22 with value: 0.1729958846162828.
lgb_params = {
    'objective':        'binary',
    'verbosity':        -1,
    'num_iterations':   250,
    'boosting_type':    'gbdt',
    'random_state':     seed,
    # 'lambda_l1':        0.08758718919397321,
    # 'lambda_l2':        0.0039689175176025465,
    'learning_rate':    0.05536251727552012,
    'max_depth':        5,
    'bagging_fraction': 0.8366733523272176,
'eval_metric': custom_metric,
    # 'num_leaves':       103,
    # 'colsample_bytree': 0.8329551585827726,
    # 'colsample_bynode': 0.4025961355653304,


    # 'learning_rate': 0.019313203137083224,
    # 'max_depth': 8,
    # 'bagging_fraction': 0.6065974114533222
    # 'bagging_freq':     4,
    # 'min_data_in_leaf': 85,
    # 'scale_pos_weight': 2.7984184778875543,
}

lgb_model = Pipeline([
    ('sampler_1', RandomOverSampler(sampling_strategy=0.003, random_state=seed)),
    ('sampler_2', RandomUnderSampler(sampling_strategy=sampling_ratio, random_state=seed)),
    ('classifier', lgb.LGBMClassifier(**lgb_params)),
])

#[I 2024-08-17 14:55:22,860] Trial 98 finished with value: 0.1710058144245913 and parameters: {'learning_rate': 0.08341853356925374, 'max_depth': 5, 'l2_leaf_reg': 6.740520715798379, 'subsample': 0.42402936337409075, 'colsample_bylevel': 0.9860546885166512, 'min_data_in_leaf': 52, 'scale_pos_weight': 2.6227279486021153}. Best is trial 98 with value: 0.1710058144245913.


cb_params = {
    'loss_function':     'Logloss',
    'iterations':        250,
    'verbose':           False,
    'random_state':      seed,
    'max_depth':         6,
    'learning_rate':     0.06936242010150652,
    'scale_pos_weight':  2.6149345838209532,
    'l2_leaf_reg':       6.216113851699493,
    'subsample':         0.6249261779711819,
    'min_data_in_leaf':  24,
'custom_metric': custom_metric,
    'cat_features':      [x for x in cat_cols if x in cat_features],
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
    # 'learning_rate':      0.05056902007063551,
    # # 'lambda':             8.879624125465703,
    # # 'alpha':              0.6779926606782505,
    # 'max_depth':          8,
    # 'subsample':          0.744401997795449,
    'n_estimators':       250,
    'learning_rate': 0.04235934634085518,
    'max_depth': 6,
    'subsample': 0.48365588695527867,
    'scoring': custom_metric,
    # 'colsample_bytree':   0.8437772277074493,
    # 'colsample_bylevel':  0.5476090898823716,
    # 'colsample_bynode':   0.9928601203635129,
    # 'scale_pos_weight':   3.29440313334688,

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
        results[f'{col}_future_norm'] = (df[col] - cum_mean_future) / (cum_std_future + err)
        results[f'{col}_past_norm'] = (df[col] - cum_mean_past) / (cum_std_past + err)

    return pd.DataFrame(results)


# Sort the DataFrame by patient_id and age_approx
# df_train = df_train.sort_values(by=['patient_id', 'age_approx'], ascending=[True, True])

# Define a function to count the number of rows with the same patient_id and age_approx >= current row's age_approx
# def count_older_or_equal(group):
#     return group.apply(lambda row: (group['age_approx'] >= row['age_approx']).sum(), axis=1)
#
# # Group by patient_id and apply the function
# df_train['count_future'] = df_train.groupby('patient_id', group_keys=False).apply(count_older_or_equal)
#
# def count_younger_or_equal(group):
#     return group.apply(lambda row: (group['age_approx'] <= row['age_approx']).sum(), axis=1)
#
# df_train['count_past'] = df_train.groupby('patient_id', group_keys=False).apply(count_younger_or_equal)


train_res = calculate_normalizations(df_train, columns_to_normalize)
df_train = pd.merge(df_train, train_res, on='isic_id', how='inner')

test_res = calculate_normalizations(df_test, columns_to_normalize)
df_test = pd.merge(df_test, test_res, on='isic_id', how='inner')

# for col in columns_to_normalize:
#     df_train[f'{col}_basel_norm'] = df_train.groupby('onehot_45')[col].transform(
#         lambda x: (x - x.mean()) / (x.std() + err))
#
#
# for col in columns_to_normalize:
#     df_train[f'{col}_xp_norm'] = df_train.groupby('onehot_9')[col].transform(
#         lambda x: (x - x.mean()) / (x.std() + err))

#### similar rows

# patient_ids = df_train['patient_id'].unique()
#
# for patient in patient_ids:
#     patient_data = df_train[df_train['patient_id'] == patient]


#### end similar rows


print(feature_cols)
print(len(feature_cols))

# benings = df_train[df_train['iddx_full'] == 'Benign']
# # Sort the DataFrame by column 'A' (or whichever column you're interested in)
# df_sorted = benings.sort_values(by='tbp_lv_dnn_lesion_confidence')
#
# # Calculate the number of rows that correspond to the bottom 20%
# bottom_20_percent_index = int(len(df_train) * 0.9)
#
# # Get the bottom 20% of rows
# bottom_20_percent = df_sorted.head(bottom_20_percent_index)
# top_80_percent = df_sorted.tail(len(df_sorted) - bottom_20_percent_index)
# print("mean and median:", bottom_20_percent['tbp_lv_dnn_lesion_confidence'].mean(), bottom_20_percent['tbp_lv_dnn_lesion_confidence'].median())
#
# df_train = df_train[df_train['iddx_full'] != 'Benign'].reset_index(drop=True)
# df_train = pd.concat([df_train, bottom_20_percent], ignore_index=True)
# print(df_train.shape)


X = df_train[feature_cols]
y = df_train[target_col]
groups = df_train[group_col]
# cv = StratifiedGroupKFold(5, shuffle=True, random_state=seed)

# val_score = cross_val_score(
#     estimator=estimator,
#     X=X, y=y,
#     cv=cv,
#     groups=groups,
#     scoring=custom_metric,
# )
#
# print(np.mean(val_score), val_score)

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)

sum_val = 0
# Manually perform cross-validation
for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):

    # Split data
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # X_val = pd.concat([X_val, top_80_percent[feature_cols]], ignore_index=True)
    # y_val = pd.concat([y_val, top_80_percent[target_col]], ignore_index=True)

    print(X_train.shape, X_val.shape)

    lgb_X = X_train[lgb_features]
    lgb_model.fit(lgb_X, y_train)

    cat_X = X_train[cat_features]
    cat_model.fit(cat_X, y_train)

    xgb_X = X_train[xgb_features]
    xgb_model.fit(xgb_X, y_train)

    X_val_lgb = X_val[lgb_features]
    X_val_xgb = X_val[xgb_features]
    X_val_cat = X_val[cat_features]

    lgb_preds = lgb_model.predict_proba(X_val_lgb)[:, 1]  # Use predict_proba for soft voting
    xgb_preds = xgb_model.predict_proba(X_val_xgb)[:, 1]
    cat_preds = cat_model.predict_proba(X_val_cat)[:, 1]

    # Predict on the validation data
    lgb_score = custom_metric(lgb_preds, y_val)
    xgb_score = custom_metric(xgb_preds, y_val)
    cat_score = custom_metric(cat_preds, y_val)

    val_preds = lgb_preds * xgb_preds * cat_preds
    # Combine the predictions (soft voting by averaging the probabilities)
    #val_preds = (lgb_preds + xgb_preds + cat_preds) / 3

    val_score = custom_metric(val_preds, y_val)


    sum_val += val_score
    print(f"Fold {fold + 1} - Validation Score: {val_score}", "lgb:", lgb_score, "xgb:", xgb_score, "cat:", cat_score)

print(f"Average Validation Score: {sum_val / 5}")
# Calculate OOF score using the custom metric


DO_TUNING = False

if DO_TUNING:
    # LightGBM
    start_time = time.time()
    study_lgb = optuna.create_study(direction='maximize', sampler=TPESampler(seed=seed))
    #study_lgb.optimize(lgb_objective, n_trials=100)
    study_lgb.optimize(lambda trial: lgb_objective(trial, sampling_ratio, df_train, feature_cols, target_col, group_col, seed), n_trials=50)
    end_time = time.time()
    elapsed_time_lgb = end_time - start_time
    print(f"LightGBM tuning took {elapsed_time_lgb:.2f} seconds.")

    # # CatBoost
    # start_time = time.time()
    # study_cb = optuna.create_study(direction='maximize', sampler=TPESampler(seed=seed))
    # #study_cb.optimize(cb_objective, n_trials=100)
    # study_cb.optimize(lambda trial: cb_objective(trial, seed, sampling_ratio, df_train, feature_cols, target_col, group_col, cat_cols), n_trials=50)
    # end_time = time.time()
    # elapsed_time_cb = end_time - start_time
    # print(f"CatBoost tuning took {elapsed_time_cb:.2f} seconds.")

    # XGBoost
    start_time = time.time()
    study_xgb = optuna.create_study(direction='maximize', sampler=TPESampler(seed=seed))
    #study_xgb.optimize(xgb_objective, n_trials=100)
    study_xgb.optimize(lambda trial: xgb_objective(trial, seed, sampling_ratio, df_train, feature_cols, target_col, group_col), n_trials=50)
    end_time = time.time()
    elapsed_time_xgb = end_time - start_time
    print(f"XGBoost tuning took {elapsed_time_xgb:.2f} seconds.")

    # Print best parameters for each study
    print("Best LGBM trial:", study_lgb.best_trial)
    # print("Best CatBoost trial:", study_cb.best_trial)
    print("Best XGBoost trial:", study_xgb.best_trial)


DO_FEATURE_IMPORTANCE_MODELS = True

if DO_FEATURE_IMPORTANCE_MODELS:
    lgb_X = df_train[lgb_features]
    lgb_model.fit(lgb_X, y)

    xgb_X = df_train[xgb_features]
    xgb_model.fit(xgb_X, y)

    cat_X = df_train[cat_features]
    cat_model.fit(cat_X, y)

    lgb_classifier = lgb_model.named_steps['classifier']
    xgb_classifier = xgb_model.named_steps['classifier']
    cat_classifier = cat_model.named_steps['classifier']

    # Extract feature importances
    lgb_importances = lgb_classifier.feature_importances_
    feature_importances_lgb = pd.DataFrame({
        'feature': lgb_X.columns,  # Assuming X_train is a DataFrame with named columns
        'importance': lgb_importances
    }).sort_values(by='importance', ascending=False)
    print(feature_importances_lgb.to_string())

    zero_importance_lgb = feature_importances_lgb[feature_importances_lgb['importance'] <= 45]['feature'].to_list()
    print("zero importance lgb", zero_importance_lgb)
    print()

    cat_importances = cat_classifier.feature_importances_
    feature_importances_cat = pd.DataFrame({
        'feature': cat_X.columns,  # Assuming X_train is a DataFrame with named columns
        'importance': cat_importances
    }).sort_values(by='importance', ascending=False)
    print("zero importance cat", feature_importances_cat.to_string())
    print()

    zero_importance_cat = feature_importances_cat[feature_importances_cat['importance'] <= 0.5]['feature'].to_list()
    print("zero importance cat", zero_importance_cat)

    xgb_importances = xgb_classifier.feature_importances_
    feature_importances_xgb = pd.DataFrame({
        'feature': xgb_X.columns,  # Assuming X_train is a DataFrame with named columns
        'importance': xgb_importances
    }).sort_values(by='importance', ascending=False)
    print(feature_importances_xgb.to_string())

    zero_importance_xgb = feature_importances_xgb[feature_importances_xgb['importance'] <= 0.007]['feature'].to_list()
    print("zero importance xgb", zero_importance_xgb)

