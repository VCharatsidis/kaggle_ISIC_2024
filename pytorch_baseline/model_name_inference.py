import os
import glob
import albumentations as A
import torch
from albumentations.pytorch import ToTensorV2
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
# For data manipulation
import pandas as pd
from efficient_net import EfficientNet
# Pytorch Imports
import torch.optim as optim

# Sklearn Imports
from sklearn.model_selection import StratifiedGroupKFold

# For Image Models

# Albumentations for augmentations

from pytorch_baseline.architecture.base_model import ISICModel
from pytorch_baseline.p_baseline_constants import CONFIG, TRAIN_DIR, ROOT_DIR, data_transforms, b_, sr_
from pytorch_baseline.p_baseline_utils import set_seed, get_train_file_path, count_parameters, ISICDataset
from pytorch_baseline.train_p_baseline_utils import prepare_loaders, fetch_scheduler, run_training, valid_pAUC_one_epoch
import warnings

from simple_cnn import SkinCancerDetectionModel


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

# df = pd.concat([df_positive, df_negative.iloc[:df_positive.shape[0] * CONFIG['positive_ratio'], :]])  # positive:negative
print("filtered", df.shape, df.target.sum(), df["patient_id"].unique().shape)

df['file_path'] = df['isic_id'].apply(get_train_file_path)
df = df[df["file_path"].isin(train_images_paths)].reset_index(drop=True)

print("df after df[df[file_path].isin(train_images_paths)]", df.shape)

sgkf = StratifiedGroupKFold(n_splits=CONFIG['n_fold'])

for fold, (a_var, val_) in enumerate(sgkf.split(df, df.target, df.patient_id)):
    print("fold:", fold, a_var.shape, val_.shape)
    df.loc[val_, "kfold"] = int(fold)


import os

# Specify the directory
model_path = 'models/simple_cnn_params_7145217_b_32_dp_0.1_fold_0.bin'

for i in range(CONFIG['n_fold']):
    CONFIG['fold'] = i

    model = EfficientNet()
    model.load_state_dict(torch.load(f"{model_path}"))
    model.eval()

    print(f'The model has {count_parameters(model):,} trainable parameters')
    print("num params:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    model.to(CONFIG['device'])

    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    print("df_train:", df_train.shape, "df_valid:", df_valid.shape)

    data_transforms = {
        "valid": A.Compose([
            A.Resize(224, 224),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0
            ),
            ToTensorV2()], p=1.)
    }

    valid_dataset = ISICDataset(df_valid, transforms=data_transforms["valid"])
    valid_loader = DataLoader(valid_dataset, batch_size=CONFIG['valid_batch_size'],
                              shuffle=True, pin_memory=True)

    val_epoch_loss, val_epoch_auroc = valid_pAUC_one_epoch(model, valid_loader, device=CONFIG['device'], epoch=0)
    print(f"fold {i} val_epoch_loss: {val_epoch_loss}, val_epoch_auroc: {val_epoch_auroc}")

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    PATH = f"cv_models/simple_cnn_params_{params}_b_{CONFIG['train_batch_size']}_dp_{CONFIG['dropout']}_fold_{CONFIG['fold']}_pAUC_{val_epoch_auroc}.bin"
    torch.save(model.state_dict(), PATH)


