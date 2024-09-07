
import pandas as pd
import torch
import torch.optim as optim

from sklearn.model_selection import GroupKFold
import gc

from ISIC_tabular_dataset import ISIC_multimodal_ssl, ISIC_multimodal_ssl_no_oversample

from architectures.ssl_encoder import EfficientNet_pretrained, EfficientNet_pretrained_linear
from p_baseline_constants import CONFIG, TRAIN_DIR, data_transforms, to_exclude
from p_baseline_utils import set_seed, get_train_file_path, VICReg, SAM

from torch.utils.data import DataLoader
import glob

import sklearn
print("sklearn version:", sklearn.__version__)


df_train = pd.read_csv("../isic-2024-challenge/train-metadata.csv")

df_train = df_train[~df_train['isic_id'].isin(to_exclude)]
set_seed(CONFIG['seed'])

train_images_paths = sorted(glob.glob(f"{TRAIN_DIR}/*.jpg"))
train_images_paths = [path.replace('\\', '/') for path in train_images_paths]
df_train['file_path'] = df_train['isic_id'].apply(get_train_file_path)
df = df_train[df_train["file_path"].isin(train_images_paths)].reset_index(drop=True)

benings = df[df['iddx_full'] == 'Benign']
# Sort the DataFrame by column 'A' (or whichever column you're interested in)
df_sorted = benings.sort_values(by='tbp_lv_dnn_lesion_confidence')

# Calculate the number of rows that correspond to the bottom 20%
bottom_20_percent_index = int(len(df) * 0.15)

# Get the bottom 20% of rows
bottom_20_percent = df_sorted.head(bottom_20_percent_index)
print("mean and median:", bottom_20_percent['tbp_lv_dnn_lesion_confidence'].mean(), bottom_20_percent['tbp_lv_dnn_lesion_confidence'].median())

df = df[df['iddx_full'] != 'Benign'].reset_index(drop=True)
df = pd.concat([df, bottom_20_percent], ignore_index=True)
print(df.shape)


N_SPLITS = 5
gkf = GroupKFold(n_splits=N_SPLITS)

df["fold"] = -1
for idx, (train_idx, val_idx) in enumerate(gkf.split(df, df["target"], groups=df["patient_id"])):
    df.loc[val_idx, "fold"] = idx


def train(df):

    train_df = df[df["fold"] != 0]
    val_df = df[df["fold"] == 0]

    train_dataset = ISIC_multimodal_ssl_no_oversample(train_df, transforms=data_transforms['train'])
    # train_dataset = ISIC_multimodal_ssl(df, pos_ratio=CONFIG['positive_ratio_train'],
    #                                                   transforms=data_transforms['train'],
    #                                                   CONFIG=CONFIG)

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['train_batch_size'],
                              shuffle=True, pin_memory=True, drop_last=True)

    val_dataset = ISIC_multimodal_ssl_no_oversample(val_df, transforms=data_transforms['train'])
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['train_batch_size'],
                                shuffle=False, pin_memory=True, drop_last=False)

    pred_dim = 45
    model = EfficientNet_pretrained_linear(out_dim=pred_dim).cuda()
    #model.load_state_dict(torch.load("models/SSL_120_no_oversampling_SAM_adaptive_lr_0.0002_64_vcreg_4162549_b_128_dp_0.1.bin"))

    print(
        f'The efficient_net has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters')

    print("num params:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    print(f'The model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters')

    positive_num = train_loader.dataset.get_positive_num()
    print("positive examples:", positive_num)

    # CONFIG['T_max'] = CONFIG['epochs'] * (
    #             positive_num * CONFIG['positive_ratio_train'] // CONFIG['train_batch_size'])
    # print('T_max', CONFIG['T_max'])

    CONFIG['T_max'] = CONFIG['epochs'] * (df.shape[0] // CONFIG['train_batch_size'])
    print("T_max:", CONFIG['T_max'])

    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['T_max'], eta_min=CONFIG['min_lr'])

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    PATH = f"SSL_{pred_dim}_var_5_0.15_benign_validated_lr_{CONFIG['learning_rate']}_{CONFIG['img_size']}_vcreg_{params}_b_{CONFIG['train_batch_size']}_dp_{CONFIG['dropout']}"
    print("image size:", CONFIG['img_size'])

    best_loss = 100000

    for epoch in range(1, CONFIG['epochs'] + 1):
        gc.collect()
        model.train()

        running_loss_vcreg = 0
        iters = 0

        for batch_idx, data in enumerate(train_loader):
            images_1 = data['images_1'].float().cuda()
            images_2 = data['images_2'].float().cuda()

            outputs_a = model(images_1)
            outputs_b = model(images_2)

            vicreg_batch = VICReg(outputs_a, outputs_b)

            vicreg_batch.backward()

            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            running_loss_vcreg += vicreg_batch.item()
            iters += 1

            if batch_idx % 100 == 0:
                val_iter = 0
                val_vcreg = 0
                with torch.no_grad():
                    for batch_idx, data in enumerate(val_loader):
                        images_1 = data['images_1'].float().cuda()
                        images_2 = data['images_2'].float().cuda()

                        outputs_a = model(images_1)
                        outputs_b = model(images_2)

                        vicreg_batch = VICReg(outputs_a, outputs_b)

                        val_vcreg += vicreg_batch.item()
                        val_iter += 1

                val_loss = val_vcreg / val_iter

                if val_loss < best_loss:
                    best_loss = val_loss
                    torch.save(model.state_dict(), f"models/{PATH}.bin")
                    print("MODEL SAVED!", val_loss)

                print("Epoch:", epoch, f"vcreg_loss: {running_loss_vcreg / iters}", optimizer.param_groups[0]['lr'])


train(df)




