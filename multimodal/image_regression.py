import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import polars as pl
import torch.nn as nn
from sklearn.model_selection import GroupKFold
import gc

from sklearn.preprocessing import OneHotEncoder

from ISIC_tabular_dataset import ISIC_image_regression

from architectures.ssl_encoder import EfficientNet_regression
from p_baseline_constants import CONFIG, TRAIN_DIR, data_transforms, to_exclude
from p_baseline_utils import set_seed, get_train_file_path, VICReg, SAM

from torch.utils.data import DataLoader
import glob

import sklearn
print("sklearn version:", sklearn.__version__)


err = 1e-5

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

    df = df.with_columns(
        pl.col(cat_cols).cast(pl.Categorical),
    )

    print("make cat cols categorical")

    df = df.to_pandas()  # .set_index(id_col)

    return df


def preprocess(df_train):
    global cat_cols

    encoder = OneHotEncoder(sparse_output=False, dtype=np.int32, handle_unknown='ignore')
    encoder.fit(df_train[cat_cols])

    new_cat_cols = [f'onehot_{i}' for i in range(len(encoder.get_feature_names_out()))]

    df_train[new_cat_cols] = encoder.transform(df_train[cat_cols])
    df_train[new_cat_cols] = df_train[new_cat_cols].astype(float)

    return df_train, new_cat_cols


df_train = read_data("../isic-2024-challenge/train-metadata.csv")
df_train, new_cat_cols = preprocess(df_train)

target_cols = num_cols + new_num_cols + new_cat_cols
print("targets:", len(target_cols))

# Normalize the columns by their mean and std
for col in target_cols:
    mean = df_train[col].mean()
    std = df_train[col].std() + err
    df_train[col] = (df_train[col] - mean) / std

df_train = df_train[~df_train['isic_id'].isin(to_exclude)].reset_index(drop=True)
set_seed(CONFIG['seed'])

train_images_paths = sorted(glob.glob(f"{TRAIN_DIR}/*.jpg"))
train_images_paths = [path.replace('\\', '/') for path in train_images_paths]
df_train['file_path'] = df_train['isic_id'].apply(get_train_file_path)
df_train = df_train[df_train["file_path"].isin(train_images_paths)].reset_index(drop=True)

# benings = df[df['iddx_full'] == 'Benign']
# # Sort the DataFrame by column 'A' (or whichever column you're interested in)
# df_sorted = benings.sort_values(by='tbp_lv_dnn_lesion_confidence')
#
# # Calculate the number of rows that correspond to the bottom 20%
# bottom_20_percent_index = int(len(df) * 0.15)
#
# # Get the bottom 20% of rows
# bottom_20_percent = df_sorted.head(bottom_20_percent_index)
# print("mean and median:", bottom_20_percent['tbp_lv_dnn_lesion_confidence'].mean(), bottom_20_percent['tbp_lv_dnn_lesion_confidence'].median())
#
# df = df[df['iddx_full'] != 'Benign'].reset_index(drop=True)
# df = pd.concat([df, bottom_20_percent], ignore_index=True)
# print(df.shape)


N_SPLITS = 5
gkf = GroupKFold(n_splits=N_SPLITS)

df_train["fold"] = -1
for idx, (train_idx, val_idx) in enumerate(gkf.split(df_train, df_train["target"], groups=df_train["patient_id"])):
    df_train.loc[val_idx, "fold"] = idx


def train(df):

    fold_train_df = df[df["fold"] != 0].reset_index(drop=True)
    val_df = df[df["fold"] == 0].reset_index(drop=True)
    print("train size:", fold_train_df.shape, "val size:", val_df.shape)

    train_dataset = ISIC_image_regression(fold_train_df, target_cols, transforms=data_transforms['train'])
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['train_batch_size'],
                              shuffle=True, pin_memory=True, drop_last=True)

    val_dataset = ISIC_image_regression(val_df, target_cols, transforms=data_transforms['train'])
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['train_batch_size'],
                                shuffle=False, pin_memory=True, drop_last=False)

    pred_dim = len(target_cols)
    model = EfficientNet_regression(out_dim=pred_dim, intermediate_dim=60).cuda()
    #model.load_state_dict(torch.load("models/SSL_120_no_oversampling_SAM_adaptive_lr_0.0002_64_vcreg_4162549_b_128_dp_0.1.bin"))

    print(
        f'The efficient_net has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters')

    print("num params:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    print(f'The model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters')

    # CONFIG['T_max'] = CONFIG['epochs'] * (
    #             positive_num * CONFIG['positive_ratio_train'] // CONFIG['train_batch_size'])
    # print('T_max', CONFIG['T_max'])

    CONFIG['T_max'] = CONFIG['epochs'] * (df.shape[0] // CONFIG['train_batch_size'])
    print("T_max:", CONFIG['T_max'])

    base_optimizer = optim.AdamW
    optimizer = SAM(model.parameters(), base_optimizer, lr=CONFIG['learning_rate'], adaptive=True, rho=2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['T_max'], eta_min=CONFIG['min_lr'])

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    PATH = f"image_regression_{pred_dim}_0.15_benign_validated_SAM_adaptive_lr_{CONFIG['learning_rate']}_{CONFIG['img_size']}_vcreg_{params}_b_{CONFIG['train_batch_size']}_dp_{CONFIG['dropout']}"
    print("image size:", CONFIG['img_size'])

    best_loss = 100000
    mae_loss = nn.L1Loss()
    for epoch in range(1, CONFIG['epochs'] + 1):
        gc.collect()
        model.train()

        running_loss_vcreg = 0
        iters = 0

        for batch_idx, data in enumerate(train_loader):
            targets = data['targets'].float().cuda()
            images_1 = data['images_1'].float().cuda()
            images_2 = data['images_2'].float().cuda()

            outputs_a, _, _ = model(images_1)
            outputs_b, _, _ = model(images_2)

            mean_output = (outputs_a + outputs_b) / 2
            mae_batch = mae_loss(mean_output, targets)

            # zero the parameter gradients

            mae_batch.backward()
            optimizer.first_step(zero_grad=True)

            # second forward-backward step

            outputs_a, _, _ = model(images_1)
            outputs_b, _, _ = model(images_2)

            mean_output = (outputs_a + outputs_b) / 2
            mae_image = mae_loss(mean_output, targets)

            mae_image.backward()
            optimizer.second_step(zero_grad=True)

            optimizer.zero_grad()
            scheduler.step()

            running_loss_vcreg += mae_image.item()
            iters += 1

            if batch_idx % 100 == 0:
                val_iter = 0
                val_vcreg = 0
                with torch.no_grad():
                    for batch_idx, data in enumerate(val_loader):
                        targets = data['targets'].float().cuda()
                        images_1 = data['images_1'].float().cuda()
                        images_2 = data['images_2'].float().cuda()

                        outputs_a, _, _ = model(images_1)
                        outputs_b, _, _ = model(images_2)

                        mean_output = (outputs_a + outputs_b) / 2
                        vicreg_batch = mae_loss(mean_output, targets)

                        val_vcreg += vicreg_batch.item()
                        val_iter += 1

                val_loss = val_vcreg / val_iter

                if val_loss < best_loss:
                    best_loss = val_loss
                    torch.save(model.state_dict(), f"models/{PATH}.bin")
                    print("MODEL SAVED!", val_loss)

                print("Epoch:", epoch, f"vcreg_loss: {running_loss_vcreg / iters}", optimizer.param_groups[0]['lr'])


train(df_train)




