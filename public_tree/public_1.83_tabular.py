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

constrained_mean_cols = [f'{col}_constrained_mean' for col in num_cols + new_num_cols]
special_cols = ['count_per_patient']
feature_cols = num_cols + new_num_cols + cat_cols + special_cols + min_max_cols + norm_cols #+ norm_loc_cols + min_max_loc_cols


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

    df = df.with_columns(
        (pl.col('clin_size_long_diam_mm') // 1).alias('group_perimeter')
    )

    # for col in num_cols + new_num_cols:
    #     df = df.with_columns(
    #         pl.col(col).mean().over(['patient_id', 'group_perimeter']).alias(f'{col}_mean'),
    #         pl.col(col).std().over(['patient_id', 'group_perimeter']).alias(f'{col}_std')
    #     )
    #
    #     df = df.with_columns(
    #         ((pl.col(col) - pl.col(f'{col}_mean')) / (pl.col(f'{col}_std') + err)).alias(f'{col}_patient_loc_norm')
    #     )
    #
    # for col in num_cols + new_num_cols:
    #     df = df.with_columns(
    #         pl.col(col).max().over(['patient_id', 'group_perimeter']).alias(f'{col}_max'),
    #         pl.col(col).min().over(['patient_id', 'group_perimeter']).alias(f'{col}_min')
    #     )
    #
    #     df = df.with_columns(
    #         ((pl.col(col) - pl.col(f'{col}_min')) / (pl.col(f'{col}_max') - pl.col(f"{col}_min") + err)).alias(f'{col}_patient_loc_min_max')
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

# [I 2024-08-17 11:47:47,716] Trial 22 finished with value: 0.1729958846162828 and parameters: {'n_iter': 242, 'lambda_l1': 0.05356915080283596, 'lambda_l2': 0.03548695950893246, 'learning_rate': 0.015844359470309228, 'max_depth': 5, 'num_leaves': 116, 'colsample_bytree': 0.7105075894232431, 'colsample_bynode': 0.6055829227054091, 'bagging_fraction': 0.5177655386536087, 'bagging_freq': 6, 'min_data_in_leaf': 68, 'scale_pos_weight': 2.1125855926322323}. Best is trial 22 with value: 0.1729958846162828.
lgb_params = {
    'objective':        'binary',
    'verbosity':        -1,
    'num_iterations':   300,
    'boosting_type':    'gbdt',
    'random_state':     seed,
    # 'lambda_l1':        0.08758718919397321,
    # 'lambda_l2':        0.0039689175176025465,
    'learning_rate':    0.05536251727552012,
    'max_depth':        5,
    'bagging_fraction': 0.8366733523272176,
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


# cb_params = {
#     'loss_function':     'Logloss',
#     'iterations':        200,
#     'verbose':           False,
#     'random_state':      seed,
#     'max_depth':         7,
#     'learning_rate':     0.06936242010150652,
#     'scale_pos_weight':  2.6149345838209532,
#     'l2_leaf_reg':       6.216113851699493,
#     'subsample':         0.6249261779711819,
#     'min_data_in_leaf':  24,
#     'cat_features':      cat_cols,
# }
#
# cb_model = Pipeline([
#     ('sampler_1', RandomOverSampler(sampling_strategy= 0.003 , random_state=seed)),
#     ('sampler_2', RandomUnderSampler(sampling_strategy=sampling_ratio , random_state=seed)),
#     ('classifier', cb.CatBoostClassifier(**cb_params)),
# ])


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
    'max_depth': 7,
    'subsample': 0.48365588695527867
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
     ('lgb', lgb_model), ('xgb', xgb_model) #,('cb', cb_model)
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

columns_to_normalize = num_cols + new_num_cols

train_res = calculate_normalizations(df_train, columns_to_normalize)
df_train = pd.merge(df_train, train_res, on='isic_id', how='inner')

test_res = calculate_normalizations(df_test, columns_to_normalize)
df_test = pd.merge(df_test, test_res, on='isic_id', how='inner')

future_norm_cols = [f'{col}_future_norm' for col in columns_to_normalize]
past_norm_cols = [f'{col}_past_norm' for col in columns_to_normalize]
future_count_cols = ['count_future']
past_count_cols = ['count_past']

# feature_cols += future_norm_cols
# feature_cols += past_norm_cols
feature_cols += future_count_cols
feature_cols += past_count_cols

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

    # Fit the model on the training data
    estimator.fit(X_train, y_train)

    # Predict on the validation data
    val_preds = estimator.predict_proba(X_val)[:, 1]
    val_score = custom_metric(val_preds, y_val)
    sum_val += val_score
    print(f"Fold {fold + 1} - Validation Score: {val_score}")

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


DO_FEATURE_IMPORTANCE_MODELS = False

if DO_FEATURE_IMPORTANCE_MODELS:
    estimator.fit(X, y)
    # Access individual models
    lgb_model_fitted = estimator.named_estimators['lgb'].named_steps['classifier']
    xgb_model_fitted = estimator.named_estimators['xgb'].named_steps['classifier']

    # Extract feature importances
    lgb_importances = lgb_model_fitted.feature_importances_
    xgb_importances = xgb_model_fitted.feature_importances_

    # Combine the importances (e.g., by averaging them)
    combined_importances = (lgb_importances + xgb_importances) / 2

    # Create a DataFrame for better visualization (optional)
    import pandas as pd

    feature_names = df_train.columns  # Assuming X_train is a DataFrame
    feature_importances_df = pd.DataFrame({
        'feature': feature_names,
        'lgb_importance': lgb_importances,
        'xgb_importance': xgb_importances,
        'combined_importance': combined_importances
    }).sort_values(by='combined_importance', ascending=False)

    print(feature_importances_df)


    # lgb_model = estimator.named_estimators_['lgb'].named_steps['classifier']
    # lgb_feature_importance = lgb_model.booster_.feature_importance(importance_type='gain')
    # lgb_feature_importance_df = pd.DataFrame({
    #     'feature': X.columns,
    #     'importance': lgb_feature_importance
    # }).sort_values(by='importance', ascending=False)
    #
    # xgb_model = estimator.named_estimators_['xgb'].named_steps['classifier']
    # xgb_feature_importance = xgb_model.get_booster().get_score(importance_type='weight')
    # xgb_feature_importance_df = pd.DataFrame({
    #     'feature': xgb_feature_importance.keys(),
    #     'importance': xgb_feature_importance.values()
    # }).sort_values(by='importance', ascending=False)
    #
    # print(lgb_feature_importance_df)
    # print(xgb_feature_importance_df)
    #
    # # Assuming lgb_feature_importance_df is already created and contains the feature importances
    # least_important_lgb = lgb_feature_importance_df.sort_values(by='importance').head(24)
    #
    # print("Least Important Features in LightGBM:")
    # print(least_important_lgb)
    #
    # # Convert the xgb_feature_importance to a DataFrame for easier manipulation
    # least_important_xgb = xgb_feature_importance_df.sort_values(by="importance").head(6)
    #
    # print("\nLeast Important Features in XGBoost:")
    # print(least_important_xgb)
    #
    # # Extract the least important feature names from both LightGBM and XGBoost
    # least_important_lgb_features = least_important_lgb['feature'].tolist()
    # least_important_xgb_features = least_important_xgb['feature'].tolist()
    #
    # # Find the intersection of the two lists
    # common_least_important_features = list(set(least_important_lgb_features) & set(least_important_xgb_features))
    #
    # print("Common Least Important Features in Both LightGBM and XGBoost:")
    # print(common_least_important_features)

