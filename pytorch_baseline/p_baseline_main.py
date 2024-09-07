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

from base_model import ISICModel

# from example_utils import prepare_loaders
# For Image Models

# Albumentations for augmentations

from pytorch_baseline.p_baseline_constants import CONFIG, TRAIN_DIR, ROOT_DIR, data_transforms, b_, sr_
from pytorch_baseline.p_baseline_utils import set_seed, get_train_file_path, count_parameters, ISICDataset, weighted_bce_loss
from pytorch_baseline.train_p_baseline_utils import valid_pAUC_one_epoch, fetch_scheduler, run_training, prepare_loaders
import warnings

from efficient_net import EfficientNet
from resnet_18 import ResNet, BasicBlock
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

# set_seed(CONFIG['seed'])

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

for fold, (a_var, val_) in enumerate(sgkf.split(df, df.target, df.patient_id)):
    print("fold:", fold, a_var.shape, val_.shape)
    df.loc[val_, "kfold"] = int(fold)

cv_auc = 0
for i in range(0, CONFIG['n_fold']):
    CONFIG['fold'] = i
    model = EfficientNet()

    #model = ResNet(BasicBlock, [2, 2, 2, 2], 1)

    print(f'The model has {count_parameters(model):,} trainable parameters')
    print("num params:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    model.to(CONFIG['device'])
    train_loader, valid_loader = prepare_loaders(df, CONFIG["fold"], data_transforms, CONFIG=CONFIG)

    print("len(train_loader):", len(train_loader), "len(valid_loader):", len(valid_loader))

    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'])
    positive_num = train_loader.dataset.get_positive_num()
    print("positive_num:", positive_num)
    CONFIG['T_max'] = CONFIG['epochs'] * (positive_num * CONFIG['positive_ratio_train'] // CONFIG['train_batch_size'])
    print('T_max', CONFIG['T_max'])

    scheduler = fetch_scheduler(optimizer)

    model, history = run_training(model, optimizer, scheduler, CONFIG['epochs'], train_loader, valid_loader, CONFIG)

    df_valid = df[df.kfold == i].reset_index(drop=True)
    valid_dataset = ISICDataset(df_valid, transforms=data_transforms["valid"])
    valid_loader = DataLoader(valid_dataset, batch_size=CONFIG['valid_batch_size'], shuffle=False, pin_memory=True)
    val_epoch_loss, val_epoch_auroc = valid_pAUC_one_epoch(model, valid_loader, device=CONFIG['device'],
                                                          epoch=0, optimizer=None, criterion=weighted_bce_loss)

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    val_epoch_auroc = round(val_epoch_auroc, 4)
    cv_auc += val_epoch_auroc
    PATH = f"cv_models/eff_net_positive_ratio_{CONFIG['positive_ratio_train']}_params_{params}_b_{CONFIG['train_batch_size']}_dp_{CONFIG['dropout']}_fold_{CONFIG['fold']}_pAUC_{val_epoch_auroc}.bin"
    torch.save(model.state_dict(), PATH)
    print("fold:", CONFIG['fold'], "done!")

print("cv_auc:", cv_auc / CONFIG['n_fold'])

history = pd.DataFrame.from_dict(history)
history.to_csv("history.csv", index=False)

plt.plot(range(history.shape[0]), history["Train Loss"].values, label="Train Loss")
plt.plot(range(history.shape[0]), history["Valid Loss"].values, label="Valid Loss")
plt.xlabel("epochs")
plt.ylabel("Loss")
plt.grid()
plt.legend()
plt.show()

plt.plot(range(history.shape[0]), history["Train AUROC"].values, label="Train AUROC")
plt.plot(range(history.shape[0]), history["Valid AUROC"].values, label="Valid AUROC")
plt.xlabel("epochs")
plt.ylabel("AUROC")
plt.grid()
plt.legend()
plt.show()

plt.plot(range(history.shape[0]), history["lr"].values, label="lr")
plt.xlabel("epochs")
plt.ylabel("lr")
plt.grid()
plt.legend()
plt.show()

