import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from colorama import Fore,  Style

CONFIG = {
    "dropout": 0.1,
    "seed": 42,
    "epochs": 100,
    "img_size": 64,
    "model_name": "tf_efficientnet",
    "checkpoint_path": "models/tf_efficientnet_b0_aa-827b6e33_from_kaggle.pth",
    "train_batch_size": 128,
    "valid_batch_size": 128,
    "learning_rate": 1.4e-4,
    "val_size": 20000,
    "positive_sampling_ratio": 0.1,
    "positive_ratio_train": 1000,
    "positive_ratio_valid": 50,
    "scheduler": 'CosineAnnealingLR',
    "min_lr": 5e-6,
    "T_max": 1000,
    "weight_decay": 1e-6,
    "fold": 0,
    "n_fold": 5,
    "n_accumulate": 1,
    "device": torch.device("cuda:0"),
    "eval_every": 25
}

# For colored terminal text

b_ = Fore.BLUE
sr_ = Style.RESET_ALL

print(b_, "CONFIG:", CONFIG, sr_)

ROOT_DIR = "../isic-2024-challenge"
TRAIN_DIR = f'{ROOT_DIR}/train-image/image'

prob = 0.5
data_transforms = {
    # "train": A.Compose([
    #     A.Resize(CONFIG['img_size'], CONFIG['img_size']),
    #     A.RandomRotate90(p=0.5),
    #     A.Flip(p=0.5),
    #     A.Downscale(p=0.25),
    #     A.ShiftScaleRotate(shift_limit=0.1,
    #                        scale_limit=0.15,
    #                        rotate_limit=60,
    #                        p=0.5),
    #     A.HueSaturationValue(
    #         hue_shift_limit=0.2,
    #         sat_shift_limit=0.2,
    #         val_shift_limit=0.2,
    #         p=0.5
    #     ),
    #     A.RandomBrightnessContrast(
    #         brightness_limit=(-0.1, 0.1),
    #         contrast_limit=(-0.1, 0.1),
    #         p=0.5
    #     ),
    #     A.Normalize(
    #         mean=[0.485, 0.456, 0.406],
    #         std=[0.229, 0.224, 0.225],
    #         max_pixel_value=255.0,
    #         p=1.0
    #     ),
    #     ToTensorV2()], p=1.),

    "train": A.Compose([
        A.Transpose(p=0.5),
        A.RandomRotate90(p=0.25),
        A.Flip(p=0.5),
        A.Downscale(p=0.25),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=prob),

        A.OneOf([
            A.MotionBlur(blur_limit=5),
            A.MedianBlur(blur_limit=5),
            A.GaussianBlur(blur_limit=5),
            A.GaussNoise(var_limit=(5.0, 30.0)),
        ], p=prob),

        A.OneOf([
            A.OpticalDistortion(distort_limit=1.0),
            A.GridDistortion(num_steps=5, distort_limit=1.),
            A.ElasticTransform(alpha=3),
        ], p=prob),

        A.CLAHE(clip_limit=4.0, p=prob),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=prob),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=prob),
        A.Resize(CONFIG['img_size'], CONFIG['img_size']),
        A.CoarseDropout(max_holes=1, max_height=int(CONFIG['img_size'] * 0.375), max_width=int(CONFIG['img_size'] * 0.375), p=prob),
        A.Normalize(),
        ToTensorV2()], p=1.),

    "valid": [
        A.Compose([
        A.Resize(CONFIG['img_size'], CONFIG['img_size']),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0
        ),
        ToTensorV2()], p=1.),

        A.Compose([
        A.Transpose(p=1),
        A.Resize(CONFIG['img_size'], CONFIG['img_size']),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0
        ),
        ToTensorV2()], p=1.),

        A.Compose([
            A.RandomRotate90(p=1),
            A.Resize(CONFIG['img_size'], CONFIG['img_size']),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0
            ),
            ToTensorV2()], p=1.),

        A.Compose([
            A.Flip(p=1),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=1),
            A.Resize(CONFIG['img_size'], CONFIG['img_size']),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0
            ),
            ToTensorV2()], p=1.),


        ]
}

malignants = [
 'Malignant::Malignant epidermal proliferations::Squamous cell carcinoma in situ'
 'Malignant::Malignant adnexal epithelial proliferations - Follicular::Basal cell carcinoma::Basal cell carcinoma, Nodular'
 'Malignant::Malignant epidermal proliferations::Squamous cell carcinoma, Invasive'
 'Indeterminate::Indeterminate epidermal proliferations::Solar or actinic keratosis'
 'Malignant::Malignant adnexal epithelial proliferations - Follicular::Basal cell carcinoma::Basal cell carcinoma, Superficial'
 'Malignant::Malignant melanocytic proliferations (Melanoma)::Melanoma in situ'
 'Malignant::Malignant melanocytic proliferations (Melanoma)::Melanoma in situ::Melanoma in situ, Lentigo maligna type'
 'Malignant::Malignant melanocytic proliferations (Melanoma)::Melanoma Invasive'
 'Malignant::Malignant melanocytic proliferations (Melanoma)::Melanoma, NOS'
 'Malignant::Malignant melanocytic proliferations (Melanoma)::Melanoma in situ::Melanoma in situ, associated with a nevus'
 'Malignant::Malignant adnexal epithelial proliferations - Follicular::Basal cell carcinoma::Basal cell carcinoma, Infiltrating'
 'Malignant::Malignant adnexal epithelial proliferations - Follicular::Basal cell carcinoma'
 'Indeterminate::Indeterminate epidermal proliferations::Solar or actinic keratosis::Actinic keratosis, Bowenoid'
 'Indeterminate::Indeterminate melanocytic proliferations::Atypical intraepithelial melanocytic proliferation'
 'Malignant::Malignant melanocytic proliferations (Melanoma)::Melanoma Invasive::Melanoma Invasive, Superficial spreading'
 'Malignant::Malignant melanocytic proliferations (Melanoma)::Melanoma Invasive::Melanoma Invasive, Associated with a nevus'
 'Malignant::Malignant epidermal proliferations::Squamous cell carcinoma, Invasive::Squamous cell carcinoma, Invasive, Keratoacanthoma-type'
 'Malignant::Malignant melanocytic proliferations (Melanoma)::Melanoma metastasis'
 'Malignant::Malignant epidermal proliferations::Squamous cell carcinoma in situ::Squamous cell carcinoma in situ, Bowens disease'
 'Malignant::Malignant epidermal proliferations::Squamous cell carcinoma, NOS'
 'Malignant::Malignant melanocytic proliferations (Melanoma)::Melanoma Invasive::Melanoma Invasive, On chronically sun-exposed skin or lentigo maligna melanoma'
 'Malignant::Malignant melanocytic proliferations (Melanoma)::Melanoma in situ::Melanoma in situ, Superficial spreading'
 'Malignant::Malignant melanocytic proliferations (Melanoma)::Melanoma Invasive::Melanoma Invasive, Nodular'
]

to_exclude = ['ISIC_0573025', 'ISIC_1443812', 'ISIC_5374420']