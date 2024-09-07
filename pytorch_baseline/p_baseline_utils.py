import torch
import numpy as np
import os
import cv2
from torch.utils.data import Dataset
import random
from pytorch_baseline.p_baseline_constants import TRAIN_DIR, CONFIG
import torch.nn as nn
import torch.nn.functional as F


def set_seed(seed=42):
    """Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_train_file_path(image_id):
    return f"{TRAIN_DIR}/{image_id}.jpg"


class ISICDataset_for_Train(Dataset):
    def __init__(self, df, CONFIG, transforms=None):
        self.df_positive = df[df["target"] == 1].reset_index()
        self.df_negative = df[df["target"] == 0].reset_index()

        self.file_names_positive = self.df_positive['file_path'].values
        self.file_names_negative = self.df_negative['file_path'].values
        self.targets_positive = self.df_positive['target'].values
        self.targets_negative = self.df_negative['target'].values
        self.transforms = transforms
        self.CONFIG = CONFIG
        self.ratio = CONFIG['positive_sampling_ratio']

        print("df_positive train:", self.df_positive.shape)
        print("df_negative train:", self.df_negative.shape)
        print("positive_ratio_train", CONFIG['positive_ratio_train'])
        print("positive_sampling_ratio", CONFIG['positive_sampling_ratio'])

    def __len__(self):
        return len(self.df_positive) * self.CONFIG['positive_ratio_train']

    def get_positive_num(self):
        return len(self.df_positive)

    def __getitem__(self, index):
        if random.random() <= self.ratio:
            df = self.df_positive
            file_names = self.file_names_positive
            targets = self.targets_positive
        else:
            df = self.df_negative
            file_names = self.file_names_negative
            targets = self.targets_negative

        index = index % df.shape[0]

        img_path = file_names[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        target = targets[index]

        if self.transforms:
            image_1 = self.transforms(image=img)["image"]
            image_2 = self.transforms(image=img)["image"]

        return {
            'image': image_1,
            'target': target,
            'image_2': image_2,
        }


class ISICDataset(Dataset):
    def __init__(self, df, transforms=None):
        self.df = df
        self.file_names = df['file_path'].values
        self.targets = df['target'].values
        self.transforms = transforms

        self.df_positive = df[df["target"] == 1].reset_index()
        self.df_negative = df[df["target"] == 0].reset_index()
        print("df_positive valid:", self.df_positive.shape)
        print("df_negative valid:", self.df_negative.shape)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_path = self.file_names[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        target = self.targets[index]

        if self.transforms:
            image_1 = self.transforms[0](image=img)["image"]
            image_2 = self.transforms[1](image=img)["image"]
            image_3 = self.transforms[2](image=img)["image"]
            image_4 = self.transforms[3](image=img)["image"]

        return {
            'image': image_1,
            'target': target,
            'image_2': image_2,
            'image_3': image_3,
            'image_4': image_4
        }


# class ISICDataset_for_val(Dataset):
#     def __init__(self, df, val_size, transforms=None):
#         self.df = df
#         self.df_positive = df[df["target"] == 1].reset_index()
#         self.df_negative = df[df["target"] == 0].reset_index()
#         self.ratio = self.df_positive.shape[0] / df.shape[0]
#
#         self.file_names_positive = self.df_positive['file_path'].values
#         self.file_names_negative = self.df_negative['file_path'].values
#         self.targets_positive = self.df_positive['target'].values
#         self.targets_negative = self.df_negative['target'].values
#
#         self.transforms = transforms
#         self.val_size = val_size
#
#     def __len__(self):
#         return self.val_size
#
#     def __getitem__(self, index):
#         if random.random() <= self.ratio:
#             df = self.df_positive
#             file_names = self.file_names_positive
#             targets = self.targets_positive
#         else:
#             df = self.df_negative
#             file_names = self.file_names_negative
#             targets = self.targets_negative
#
#         index = index % df.shape[0]
#
#         img_path = file_names[index]
#         img = cv2.imread(img_path)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         target = targets[index]
#
#         if self.transforms:
#             img = self.transforms(image=img)["image"]
#
#         return {
#             'image': img,
#             'target': target
#         }



class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1. / p)

    def __repr__(self):
        return self.__class__.__name__ + \
            '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'


# Define the weighted BCE loss function
def weighted_bce_loss(outputs, targets):
    N = CONFIG['positive_ratio_train']  # The weight factor for the minority class
    class_weights = torch.tensor([1.0, N], dtype=torch.float32).to(torch.float32)

    bce_loss = nn.BCELoss(reduction='none')(outputs, targets)
    weight = targets * class_weights[1] + (1 - targets) * class_weights[0]
    weighted_bce_loss = weight * bce_loss
    return weighted_bce_loss.mean()


def BCEWithPolarizationPenaltyLoss(predictions, targets, penalty_strength=5.0):
    # Standard BCE loss
    bce_loss = nn.BCELoss(reduction='none')(predictions, targets)

    # Quadratic polarization penalty
    penalty = 2 - (predictions - 1) ** 2 + predictions ** 2

    # Combine the two losses
    total_loss = bce_loss * penalty

    return total_loss.mean()


def roc_star_loss( _y_true, y_pred, gamma, _epoch_true, epoch_pred):
    """
    Nearly direct loss function for AUC.
    See article,
    C. Reiss, "Roc-star : An objective function for ROC-AUC that actually works."
    https://github.com/iridiumblue/articles/blob/master/roc_star.md
        _y_true: `Tensor`. Targets (labels).  Float either 0.0 or 1.0 .
        y_pred: `Tensor` . Predictions.
        gamma  : `Float` Gamma, as derived from last epoch.
        _epoch_true: `Tensor`.  Targets (labels) from last epoch.
        epoch_pred : `Tensor`.  Predicions from last epoch.
    """
    #convert labels to boolean
    y_true = (_y_true >= 0.50)
    epoch_true = (_epoch_true >= 0.50)

    # if batch is either all true or false return small random stub value.
    if torch.sum(y_true) == 0 or torch.sum(y_true) == y_true.shape[0]:
        return torch.sum(y_pred)*1e-8

    pos = y_pred[y_true]
    neg = y_pred[~y_true]

    epoch_pos = epoch_pred[epoch_true]
    epoch_neg = epoch_pred[~epoch_true]

    # Take random subsamples of the training set, both positive and negative.
    max_pos = 1000 # Max number of positive training samples
    max_neg = 1000 # Max number of positive training samples
    cap_pos = epoch_pos.shape[0]
    cap_neg = epoch_neg.shape[0]
    epoch_pos = epoch_pos[torch.rand_like(epoch_pos) < max_pos/cap_pos]
    epoch_neg = epoch_neg[torch.rand_like(epoch_neg) < max_neg/cap_pos]

    ln_pos = pos.shape[0]
    ln_neg = neg.shape[0]

    # sum positive batch elements agaionst (subsampled) negative elements
    if ln_pos>0 :
        pos_expand = pos.view(-1,1).expand(-1,epoch_neg.shape[0]).reshape(-1)
        neg_expand = epoch_neg.repeat(ln_pos)

        diff2 = neg_expand - pos_expand + gamma
        l2 = diff2[diff2>0]
        m2 = l2 * l2
        len2 = l2.shape[0]
    else:
        m2 = torch.tensor([0], dtype=torch.float).cuda()
        len2 = 0

    # Similarly, compare negative batch elements against (subsampled) positive elements
    if ln_neg>0 :
        pos_expand = epoch_pos.view(-1,1).expand(-1, ln_neg).reshape(-1)
        neg_expand = neg.repeat(epoch_pos.shape[0])

        diff3 = neg_expand - pos_expand + gamma
        l3 = diff3[diff3>0]
        m3 = l3*l3
        len3 = l3.shape[0]
    else:
        m3 = torch.tensor([0], dtype=torch.float).cuda()
        len3=0

    if (torch.sum(m2)+torch.sum(m3))!=0 :
       res2 = torch.sum(m2)/max_pos+torch.sum(m3)/max_neg
       #code.interact(local=dict(globals(), **locals()))
    else:
       res2 = torch.sum(m2)+torch.sum(m3)

    res2 = torch.where(torch.isnan(res2), torch.zeros_like(res2), res2)

    return res2


def bce_loss(outputs, targets):
    return nn.BCELoss()(outputs, targets)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# f: encoder network, lambda, mu, nu: coefficients of the
# invariance, variance and covariance losses, N: batch size
# , D: dimension of the representations
# mse_loss: Mean square error loss function, off_diagonal:
# off-diagonal elements of a matrix, relu: ReLU activation function

def relu(x):
    return torch.maximum(x, torch.tensor(0.0))


def off_diagonal(matrix):
    n = matrix.size(0)
    assert matrix.size(1) == n, "Input must be a square matrix"

    off_diag_elements = matrix.flatten()[1:].view(n - 1, n + 1)[:, :-1].flatten()
    return off_diag_elements


def VICReg(z_a, z_b, lamda=25, mu=25, nu=1):
    # two randomly augmented versions of x: x_a, x_b

    N = z_a.shape[0]  # batch size
    D = z_a.shape[1]  # dimension of the representations

    # invariance loss
    sim_loss = (z_a - z_b) ** 2

    # variance loss
    std_z_a = torch.sqrt(z_a.var(dim=0) + 1e-04)
    std_z_b = torch.sqrt(z_b.var(dim=0) + 1e-04)

    std_loss = torch.mean(relu(5 - std_z_a)) + torch.mean(relu(5 - std_z_b))

    # covariance loss
    z_a = z_a - z_a.mean(dim=0)
    z_b = z_b - z_b.mean(dim=0)
    cov_z_a = (z_a.T @ z_a) / (N - 1)
    cov_z_b = (z_b.T @ z_b) / (N - 1)

    cov_loss = off_diagonal(cov_z_a).pow_(2).sum() / D + off_diagonal(cov_z_b).pow_(2).sum() / D

    # loss
    loss = lamda * sim_loss.mean() + mu * std_loss.mean() + nu * cov_loss.mean()

    return loss


def custom_VICReg(z_a, z_b, mu=25, nu=1):
    # two randomly augmented versions of x: x_a, x_b

    N = z_a.shape[0]  # batch size
    D = z_a.shape[1]  # dimension of the representations

    # variance loss
    std_z_a = torch.sqrt(z_a.var(dim=0) + 1e-04)
    std_z_b = torch.sqrt(z_b.var(dim=0) + 1e-04)

    std_loss = torch.mean(relu(1 - std_z_a)) + torch.mean(relu(1 - std_z_b))

    # covariance loss
    z_a = z_a - z_a.mean(dim=0)
    z_b = z_b - z_b.mean(dim=0)
    cov_z_a = (z_a.T @ z_a) / (N - 1)
    cov_z_b = (z_b.T @ z_b) / (N - 1)

    cov_loss = off_diagonal(cov_z_a).pow_(2).sum() / D + off_diagonal(cov_z_b).pow_(2).sum() / D

    # loss
    loss = mu * std_loss.mean() + nu * cov_loss.mean()

    return loss


def just_covariance(z_a, z_b, nu=1):
    # two randomly augmented versions of x: x_a, x_b

    N = z_a.shape[0]  # batch size
    D = z_a.shape[1]  # dimension of the representations

    # covariance loss
    z_a = z_a - z_a.mean(dim=0)
    z_b = z_b - z_b.mean(dim=0)
    cov_z_a = (z_a.T @ z_a) / (N - 1)
    cov_z_b = (z_b.T @ z_b) / (N - 1)

    cov_loss = off_diagonal(cov_z_a).pow_(2).sum() / D + off_diagonal(cov_z_b).pow_(2).sum() / D

    # loss
    loss = nu * cov_loss.mean()

    return loss


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups

