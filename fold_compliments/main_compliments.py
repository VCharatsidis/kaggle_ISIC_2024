import copy
import os
import glob
from torch.utils.data import DataLoader
import torch
from matplotlib import pyplot as plt

# For data manipulation
import pandas as pd

# Pytorch Imports
import torch.optim as optim

# Sklearn Imports
from sklearn.model_selection import StratifiedGroupKFold

from avg_utils import prepare_loaders_avg, run_training_avg
from base_model import ISICModel
from fold_compliments.compiments_avg import run_training_compliments
from p_baseline_utils import set_seed, get_train_file_path, count_parameters, ISICDataset, weighted_bce_loss
from pretrained_efficient_net import EfficientNetWithFeatures

# from example_utils import prepare_loaders
# For Image Models

# Albumentations for augmentations

from pytorch_baseline.p_baseline_constants import CONFIG, TRAIN_DIR, ROOT_DIR, data_transforms, b_, sr_

import warnings

from efficient_net import EfficientNet
from resnet_18 import ResNet, BasicBlock
from simple_cnn import SkinCancerDetectionModel
from train_p_baseline_utils import fetch_scheduler, valid_pAUC_one_epoch

# Get the PyTorch version
torch_version = torch.__version__
print(f"PyTorch version: {torch_version}")

# Check if CUDA is available
cuda_available = torch.cuda.is_available()
print(f"CUDA available: {cuda_available}")

print(torch.cuda.get_device_name(0))  # Should return the name of your GPU

warnings.filterwarnings("ignore")

# For descriptive error messages
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

set_seed(CONFIG['seed'])

train_images_paths = sorted(glob.glob(f"{TRAIN_DIR}/*.jpg"))
train_images_paths = [path.replace('\\', '/') for path in train_images_paths]

print("Number of train images:", len(train_images_paths))

df = pd.read_csv(f"{ROOT_DIR}/train-metadata.csv")

print("original", df.shape, "patient ids:", df["patient_id"].unique().shape)

df_positive = df[df["target"] == 1].reset_index(drop=True)
df_negative = df[df["target"] == 0].reset_index(drop=True)

print("positive", df_positive.shape, "patient ids:", df_positive["patient_id"].unique().shape)
print("negative", df_negative.shape, "patient ids:", df_negative["patient_id"].unique().shape)

# df = pd.concat([df_positive, df_negative.iloc[:df_positive.shape[0] * CONFIG['positive_ratio_valid'], :]])  # positive:negative
# print("filtered", df.shape, df.target.sum(), df["patient_id"].unique().shape)

df['file_path'] = df['isic_id'].apply(get_train_file_path)
df = df[df["file_path"].isin(train_images_paths)].reset_index(drop=True)

print("df after df[df[file_path].isin(train_images_paths)]", df.shape)

sgkf = StratifiedGroupKFold(n_splits=CONFIG['n_fold'])

for fold, (a_var, val_) in enumerate(sgkf.split(X=df, y=df.target, groups=df.patient_id)):
    print("fold:", fold, a_var.shape, val_.shape)
    df.loc[val_, "kfold"] = int(fold)

cv_auc = 0
for fold in range(0, CONFIG['n_fold']):
    CONFIG['fold'] = fold
    model = EfficientNetWithFeatures()

    print(f'The model has {count_parameters(model):,} trainable parameters')
    print("num params:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    model.to(CONFIG['device'])
    num_models = 1
    train_loaders, valid_loader = prepare_loaders_avg(df,  data_transforms, num_models, fold)

    positive_num = train_loaders[0].dataset.get_positive_num()
    CONFIG['T_max'] = CONFIG['epochs'] * (positive_num * CONFIG['positive_ratio_train'] // CONFIG['train_batch_size'])
    print('T_max', CONFIG['T_max'])

    components = []
    for i in range(num_models):
        a_model = EfficientNetWithFeatures().cuda()
        optimizer = optim.AdamW(a_model.parameters(), lr=CONFIG['learning_rate'])
        scheduler = fetch_scheduler(optimizer)
        components.append((a_model, train_loaders[i], optimizer, scheduler))

    fold_auc, models = run_training_compliments(df, fold, components, valid_loader)
    cv_auc += fold_auc

print("cv_auc:", cv_auc / CONFIG['n_fold'])

