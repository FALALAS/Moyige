import numpy as np
from torch.utils.data import Dataset
import torch
from scipy.io import loadmat
import os


def generate_patches(noisy_img, gt, patch_size=128, stride=64):
    assert noisy_img.shape == gt.shape
    h, w, c = noisy_img.shape
    noisy_patches, gt_patches = [], []
    x_indices, y_indices = list(range(0, h-patch_size+1, stride)) + [h-patch_size, ], \
                           list(range(0, w-patch_size+1, stride)) + [w-patch_size, ]
    for i in x_indices:
        for j in y_indices:
            for k in range(0, c - 30):
                noisy_patches.append(noisy_img[i:i+patch_size, j:j+patch_size, k: k+30].copy())
                gt_patches.append(gt[i:i+patch_size, j:j+patch_size, k: k+30].copy())
    return noisy_patches, gt_patches


def data_aug(img, mode=0):#图像旋转0，90，180，270，逆时针
    # data augmentation
    if mode == 0:
        return img
    elif mode == 1:
        return np.rot90(img,k=1,axes=(1,2))
    elif mode == 2:
        return np.rot90(img,k=2,axes=(1,2))
    elif mode == 3:
        return np.rot90(img,k=3,axes=(1,2))


class PrePartitionDataset(Dataset):
    def __init__(self, data_dir, patch_size, case, length=10000):
        super().__init__()
        mat_pavia = loadmat(os.path.join(data_dir, f"Noisy_train_pavia_CASE{case}.mat"))
        mat_washington = loadmat(os.path.join(data_dir, f"Noisy_train_washington_CASE{case}.mat"))
        self.pavia, self.noisy_pavia = mat_pavia['Img'].transpose(2, 0, 1), mat_pavia['Noisy_Img'].transpose(2, 0, 1)
        self.washington, self.noisy_washington = mat_washington['Img'].transpose(2, 0, 1), mat_washington['Noisy_Img'].transpose(2, 0, 1)
        self.patch_size = patch_size
        self.length = length

    def __getitem__(self, item):
        r = np.random.rand()
        if r < 0.5:
            img, noisy_img = self.pavia, self.noisy_pavia
        else:
            img, noisy_img = self.washington, self.noisy_washington
        c, h, w = img.shape
        i, j, k = np.random.randint(0, h-self.patch_size+1), np.random.randint(0, w-self.patch_size+1),\
                  np.random.randint(0, c-30+1)
        x, y = img[k:k+30, i:i+self.patch_size, j:j+self.patch_size], noisy_img[k:k+30, i:i+self.patch_size, j:j+self.patch_size]
        aug_mode = np.random.randint(0, 4)
        x, y = data_aug(x, aug_mode), data_aug(y, aug_mode)
        x, y = torch.from_numpy(np.copy(x)), torch.from_numpy(np.copy(y))
        x, y = torch.from_numpy(np.copy(x)), torch.from_numpy(np.copy(y))
        return x, y

    def __len__(self):
        return self.length