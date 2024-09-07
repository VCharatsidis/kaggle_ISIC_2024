import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder
import polars as pl


def custom_metric(y_hat, y_true):
    min_tpr = 0.80
    max_fpr = abs(1 - min_tpr)

    v_gt = abs(y_true - 1)
    v_pred = np.array([1.0 - x for x in y_hat])

    partial_auc_scaled = roc_auc_score(v_gt, v_pred, max_fpr=max_fpr)
    partial_auc = 0.5 * max_fpr ** 2 + (max_fpr - 0.5 * max_fpr ** 2) / (1.0 - 0.5) * (partial_auc_scaled - 0.5)

    return partial_auc


def preprocess(df_train, df_test, cat_cols):
    encoder = OneHotEncoder(sparse_output=False, dtype=np.int32, handle_unknown='ignore')
    encoder.fit(df_train[cat_cols])

    new_cat_cols = [f'onehot_{i}' for i in range(len(encoder.get_feature_names_out()))]
    col_asd = [x for x in encoder.get_feature_names_out()]
    for i in range(len(new_cat_cols)):
        print(f'{new_cat_cols[i]}: {col_asd[i]}')

    df_train[new_cat_cols] = encoder.transform(df_train[cat_cols])
    df_train[new_cat_cols] = df_train[new_cat_cols].astype('category')

    df_test[new_cat_cols] = encoder.transform(df_test[cat_cols])
    df_test[new_cat_cols] = df_test[new_cat_cols].astype('category')

    return df_train, df_test, new_cat_cols


def read_data(path, err, num_cols, cat_cols, new_num_cols):
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
            pl.col(col).mean().over('patient_id').alias(f'{col}_mean'),
            pl.col(col).std().over('patient_id').alias(f'{col}_std')
        )

        df = df.with_columns(
            ((pl.col(col) - pl.col(f'{col}_mean')) / (pl.col(f'{col}_std') + err)).alias(f'{col}_patient_norm')
        )

    for col in num_cols + new_num_cols:
        df = df.with_columns(
            pl.col(col).max().over('patient_id').alias(f'{col}_max'),
            pl.col(col).min().over('patient_id').alias(f'{col}_min')
        )

        df = df.with_columns(
            ((pl.col(col) - pl.col(f'{col}_min')) / (pl.col(f'{col}_max') - pl.col(f"{col}_min") + err)).alias(f'{col}_patient_min_max')
        )

    print("added patient_norm")

    df = df.with_columns(
        pl.col(cat_cols).cast(pl.Categorical),
    )

    print("make cat cols categorical")

    df = df.to_pandas()  # .set_index(id_col)

    return df
