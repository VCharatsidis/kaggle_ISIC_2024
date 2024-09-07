import random

import pandas as pd
from torch.utils.data import Dataset
import cv2

class ISIC_tabular_dataset_for_Train(Dataset):
    def __init__(self, df, pos_ratio, train_cols):
        self.df_positive = df[df["target"] == 1].reset_index()
        self.df_negative = df[df["target"] == 0].reset_index()

        self.input_positive = self.df_positive[train_cols].values
        self.input_negative = self.df_negative[train_cols].values

        self.targets_positive = self.df_positive['target'].values
        self.targets_negative = self.df_negative['target'].values

        self.pos_ratio = pos_ratio
        self.ratio = 1 / 10#pos_ratio

    def __len__(self):
        return len(self.df_positive) * self.pos_ratio

    def get_positive_num(self):
        return len(self.df_positive)

    def __getitem__(self, index):
        if random.random() <= self.ratio:
            df = self.input_positive
            targets = self.targets_positive
        else:
            df = self.input_negative
            targets = self.targets_negative

        index = index % df.shape[0]

        row = df[index]
        target = targets[index]

        return {
            'input_data': row,
            'target': target
        }


class ISIC_tabular_dataset(Dataset):
    def __init__(self, df, train_cols):
        self.df = df
        self.data = df[train_cols].values
        self.targets = df['target'].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.data[index]
        target = self.targets[index]

        return {
            'input_data': row,
            'target': target
        }


class ISIC_multimodal_dataset_for_Train(Dataset):
    def __init__(self, df, pos_ratio, train_cols, transforms, CONFIG):
        self.df_positive = df[df["target"] == 1].reset_index(drop=True)
        self.df_negative = df[df["target"] == 0].reset_index(drop=True)

        self.ambiguous = self.df_negative[self.df_negative['iddx_full'] != 'Benign']
        self.file_names_ambiguous = self.ambiguous['file_path'].values
        self.input_amibuous = self.ambiguous[train_cols].values
        self.targets_ambiguous = self.ambiguous['target'].values

        self.file_names_positive = self.df_positive['file_path'].values
        self.file_names_negative = self.df_negative['file_path'].values

        self.input_positive = self.df_positive[train_cols].values
        self.input_negative = self.df_negative[train_cols].values

        self.targets_positive = self.df_positive['target'].values
        self.targets_negative = self.df_negative['target'].values

        self.pos_ratio = pos_ratio
        self.ratio = CONFIG['positive_sampling_ratio']

        self.transforms = transforms

    def __len__(self):
        return len(self.df_positive) * self.pos_ratio

    def get_positive_num(self):
        return len(self.df_positive)

    def __getitem__(self, index):
        if random.random() <= self.ratio:
            df = self.input_positive
            targets = self.targets_positive
            file_names = self.file_names_positive
        else:
            df = self.input_negative
            targets = self.targets_negative
            file_names = self.file_names_negative

        index = index % df.shape[0]

        row = df[index]
        target = targets[index]

        img_path = file_names[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transforms:
            image_1 = self.transforms(image=img)["image"]
            image_2 = self.transforms(image=img)["image"]

        return {
            'input_data': row,
            'target': target,

            'images_1': image_1,
            'images_2': image_2,
        }


class ISIC_multimodal_ssl(Dataset):
    def __init__(self, df, pos_ratio, transforms, CONFIG):
        self.df_positive = df[df["target"] == 1].reset_index(drop=True)
        self.df_negative = df[df["target"] == 0].reset_index(drop=True)

        self.ambiguous = self.df_negative[self.df_negative['iddx_full'] != 'Benign']
        self.file_names_ambiguous = self.ambiguous['file_path'].values
        self.input_amibuous = self.ambiguous.values
        self.targets_ambiguous = self.ambiguous['target'].values

        self.file_names_positive = self.df_positive['file_path'].values
        self.file_names_negative = self.df_negative['file_path'].values

        self.input_positive = self.df_positive.values
        self.input_negative = self.df_negative.values

        self.pos_ratio = pos_ratio
        self.ratio = CONFIG['positive_sampling_ratio']

        self.transforms = transforms

    def __len__(self):
        return len(self.df_positive) * self.pos_ratio

    def get_positive_num(self):
        return len(self.df_positive)

    def __getitem__(self, index):
        if random.random() <= self.ratio:
            df = self.input_positive
            file_names = self.file_names_positive
        else:
            df = self.input_negative
            file_names = self.file_names_negative

        index = index % df.shape[0]

        img_path = file_names[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transforms:
            image_1 = self.transforms(image=img)["image"]
            image_2 = self.transforms(image=img)["image"]

        return {
            'images_1': image_1,
            'images_2': image_2,
        }


class ISIC_multimodal_ssl_no_oversample(Dataset):
    def __init__(self, df, transforms):

        self.df = df
        self.file_names_positive = df['file_path'].values
        self.input_positive = df.values
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def get_positive_num(self):
        return len(self.df)

    def __getitem__(self, index):
        file_names = self.file_names_positive

        img_path = file_names[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transforms:
            image_1 = self.transforms(image=img)["image"]
            image_2 = self.transforms(image=img)["image"]

        return {
            'images_1': image_1,
            'images_2': image_2,
        }


class ISIC_multimodal_ssl_valid(Dataset):
    def __init__(self, df, transforms):
        self.df = df
        self.file_names = df['file_path'].values
        self.transforms = transforms
        self.isic_ids = df['isic_id'].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_path = self.file_names[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        isic_id = self.isic_ids[index]

        if self.transforms:
            image_1 = self.transforms[0](image=img)["image"]

        return {
            'isic_id': isic_id,
            'images_1': image_1,
        }


class ISIC_multimodal_dataset(Dataset):
    def __init__(self, df, train_cols, transforms):
        self.df = df
        self.data = df[train_cols].values
        self.targets = df['target'].values
        self.file_names = df['file_path'].values
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.data[index]
        target = self.targets[index]

        img_path = self.file_names[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transforms:
            image_1 = self.transforms[0](image=img)["image"]
            image_2 = self.transforms[1](image=img)["image"]
            image_3 = self.transforms[2](image=img)["image"]
            image_4 = self.transforms[3](image=img)["image"]

        return {
            'images_1': image_1,
            'images_2': image_2,
            'images_3': image_3,
            'images_4': image_4,

            'input_data': row,
            'target': target
        }


class ISIC_image_regression(Dataset):
    def __init__(self, df, target_cols, transforms):
        self.df = df
        self.targets = df[target_cols].values
        self.file_names = df['file_path'].values
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        targets = self.targets[index]

        img_path = self.file_names[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        image_1 = self.transforms(image=img)["image"]
        image_2 = self.transforms(image=img)["image"]

        return {
            'images_1': image_1,
            'images_2': image_2,

            'targets': targets
        }
