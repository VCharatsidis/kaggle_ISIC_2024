import os
import gc
import cv2
import math
import copy
import time
import random
import glob
from matplotlib import pyplot as plt

import h5py
from PIL import Image
from io import BytesIO

# For data manipulation
import numpy as np
import pandas as pd

# Pytorch Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torch.cuda import amp
import torchvision

# Utils
import joblib
from tqdm import tqdm
from collections import defaultdict

# Sklearn Imports
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

# For Image Models
import timm

# Albumentations for augmentations
import albumentations as A
from albumentations.pytorch import ToTensorV2

# For colored terminal text
from colorama import Fore, Back, Style
b_ = Fore.BLUE
sr_ = Style.RESET_ALL

import warnings
warnings.filterwarnings("ignore")

# For descriptive error messages
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

CONFIG = {
    "seed": 42,
    "img_size": 224,
    "model_name": "tf_efficientnet_b0_ns",
    "valid_batch_size": 32,
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
}


def set_seed(seed=42):
    '''Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.'''
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)


set_seed(CONFIG['seed'])


ROOT_DIR = "../isic-2024-challenge"
TEST_CSV = f'{ROOT_DIR}/test-metadata.csv'
TEST_HDF = f'{ROOT_DIR}/test-image.hdf5'
SAMPLE = f'{ROOT_DIR}/sample_submission.csv'

BEST_WEIGHT =["cv_models/eff_net_Weighted_positive_ratio_80_params_677489_b_64_dp_0.1_fold_0_pAUC_0.1738.bin",
              "cv_models/eff_net_Weighted_positive_ratio_80_params_677489_b_64_dp_0.1_fold_1_pAUC_0.1332.bin",
              "cv_models/eff_net_Weighted_positive_ratio_80_params_677489_b_64_dp_0.1_fold_2_pAUC_0.162.bin",
              "cv_models/eff_net_Weighted_positive_ratio_80_params_677489_b_64_dp_0.1_fold_3_pAUC_0.143.bin",
              "cv_models/eff_net_Weighted_positive_ratio_80_params_677489_b_64_dp_0.1_fold_4_pAUC_0.1325.bin"]


df = pd.read_csv(TEST_CSV)

df['target'] = 0 # dummy

df_sub = pd.read_csv(SAMPLE)


class ISICDataset(Dataset):
    def __init__(self, df, file_hdf, transforms=None):
        self.df = df
        self.fp_hdf = h5py.File(file_hdf, mode="r")
        self.isic_ids = df['isic_id'].values
        self.targets = df['target'].values
        self.transforms = transforms

    def __len__(self):
        return len(self.isic_ids)

    def __getitem__(self, index):
        isic_id = self.isic_ids[index]
        img = np.array(Image.open(BytesIO(self.fp_hdf[isic_id][()])))
        target = self.targets[index]

        if self.transforms:
            img = self.transforms(image=img)["image"]

        return {
            'image': img,
            'target': target,
        }

data_transforms = {
    "valid": A.Compose([
        A.Resize(CONFIG['img_size'], CONFIG['img_size']),
        A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0
            ),
        ToTensorV2()], p=1.)
}


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction_ratio=4):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1)
        self.fc2 = nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1)

    def forward(self, x):
        se = F.adaptive_avg_pool2d(x, 1)
        se = F.relu(self.fc1(se))
        se = torch.sigmoid(self.fc2(se))
        return x * se


# MBConv Block
class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio, stride, reduction_ratio=4, kernel_size=3):
        super(MBConvBlock, self).__init__()
        self.stride = stride
        self.expand_ratio = expand_ratio
        hidden_dim = in_channels * expand_ratio

        # Expansion phase
        self.expand_conv = nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False)
        self.bn0 = nn.BatchNorm2d(hidden_dim)

        # Depthwise convolution
        self.depthwise_conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride,
                                        padding=kernel_size // 2, groups=hidden_dim, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_dim)

        # Squeeze-and-Excitation phase
        self.se = SEBlock(hidden_dim, reduction_ratio)

        # Projection phase
        self.project_conv = nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.use_residual = (stride == 1 and in_channels == out_channels)

    def forward(self, x):
        residual = x
        x = F.relu6(self.bn0(self.expand_conv(x)))
        x = F.relu6(self.bn1(self.depthwise_conv(x)))
        x = self.se(x)
        x = self.bn2(self.project_conv(x))

        if self.use_residual:
            x = x + residual
        return x


# EfficientNet
class EfficientNet(nn.Module):
    def __init__(self, width_coefficient=1.0, depth_coefficient=1.0, dropout_rate=0.2):
        super(EfficientNet, self).__init__()

        # Configuration for EfficientNet-B0
        base_channels = 32
        base_layers = [
            # expand_ratio, channels, repeats, stride, kernel_size
            [1, 16, 1, 1, 3],
            [6, 24, 2, 2, 3],
            [6, 40, 2, 2, 5],
            [6, 80, 3, 2, 3],
#             [6, 112, 3, 1, 5],
#             [6, 192, 4, 2, 5],
#             [6, 320, 1, 1, 3],
        ]

        # Initial convolution layer
        out_channels = int(base_channels * width_coefficient)
        self.conv1 = nn.Conv2d(3, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.swish = nn.SiLU()

        # Build MBConv blocks
        layers = []
        in_channels = out_channels
        for t, c, n, s, k in base_layers:
            #c = c * 4
            out_channels = int(c * width_coefficient)
            repeats = int(n * depth_coefficient)
            for i in range(repeats):
                stride = s if i == 0 else 1
                layers.append(MBConvBlock(in_channels, out_channels, t, stride, kernel_size=k))
                in_channels = out_channels
        self.blocks = nn.Sequential(*layers)

        # Final layers
        out_channels = int(1280 * width_coefficient)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(out_channels, 1)

    def forward(self, x):
        x = self.swish(self.bn1(self.conv1(x)))
        x = self.blocks(x)
        x = self.swish(self.bn2(self.conv2(x)))
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x

models = []
for bw in BEST_WEIGHT:
    model = EfficientNet()
    # Load the model weights from the .model file
    model.load_state_dict(torch.load(bw))
    model.to(CONFIG['device'])
    model.eval()
    models.append(model)


test_dataset = ISICDataset(df, TEST_HDF, transforms=data_transforms["valid"])
test_loader = DataLoader(test_dataset, batch_size=CONFIG['valid_batch_size'], shuffle=False, pin_memory=True)

preds = []
with torch.no_grad():
    bar = tqdm(enumerate(test_loader), total=len(test_loader))
    for step, data in bar:
        images = data['image'].to(CONFIG["device"], dtype=torch.float)
        batch_size = images.size(0)
        sum_outputs = []
        for model in models:
            model_output = model(images).detach().cpu().numpy()
            sum_outputs.append(model_output)

        sum_outputs = np.array(sum_outputs)
        sum_outputs = np.mean(sum_outputs, axis=0)
        preds.append(sum_outputs)

preds = np.concatenate(preds).flatten()