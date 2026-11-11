from dataset.transform import random_rot_flip, random_rotate, blur, obtain_cutmix_box, normalize

from copy import deepcopy
import h5py
import math
import numpy as np
import os
from PIL import Image
import random
from scipy.ndimage.interpolation import zoom
from scipy import ndimage
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from torchvision import transforms
from dataset.transform import *

class ACDCDataset(Dataset):
    def __init__(self, name, root, mode, size=None, id_path=None, nsample=None):
        self.name = name
        self.root = root
        self.mode = mode
        self.size = size

        if mode == 'train_l' or mode == 'train_u':
            with open(id_path, 'r') as f:
                self.ids = f.read().splitlines()
            if mode == 'train_l' and nsample is not None and nsample > len(self.ids):
                self.ids *= math.ceil(nsample / len(self.ids))
                self.ids = self.ids[:nsample]
        elif mode == 'val':
            with open('splits/%s/val.txt' % name, 'r') as f:
                self.ids = f.read().splitlines()
        elif mode == 'valtest':
            with open('splits/%s/valtest.txt' % name, 'r') as f:
                self.ids = f.read().splitlines()
        elif mode == 'test':
            with open('splits/%s/test.txt' % name, 'r') as f:
                self.ids = f.read().splitlines()

    def __getitem__(self, item):
        id = self.ids[item]
        sample = h5py.File(os.path.join(self.root, id), 'r')
        img = sample['image'][:]
        mask = sample['label'][:]
        # print(img.shape)
        # print(mask.shape)
        # return

        if self.mode == 'val':
            # print('id:', id)
            file_name = id.split('/')[-1].split('.')[0]  # patient028_frame01
            return torch.from_numpy(img).float(), torch.from_numpy(mask).long(), file_name
        elif self.mode == 'valtest':
            # print('id:', id)
            file_name = id.split('/')[-1].split('.')[0]     # patient028_frame01
            return torch.from_numpy(img).float(), torch.from_numpy(mask).long(), file_name
        elif self.mode == 'test':
            # print('id:', id)
            file_name = id.split('/')[-1].split('.')[0]     # patient011_frame01
            return torch.from_numpy(img).float(), torch.from_numpy(mask).long(), file_name

        if random.random() > 0.5:
            img, mask = random_rot_flip(img, mask)
        elif random.random() > 0.5:
            img, mask = random_rotate(img, mask)
        x, y = img.shape
        img = zoom(img, (self.size / x, self.size / y), order=0)
        mask = zoom(mask, (self.size / x, self.size / y), order=0)

        if self.mode == 'train_l':
            return torch.from_numpy(img).unsqueeze(0).float(), torch.from_numpy(np.array(mask)).long()

        img = Image.fromarray((img * 255).astype(np.uint8))
        img_s1, img_s2 = deepcopy(img), deepcopy(img)
        img = torch.from_numpy(np.array(img)).unsqueeze(0).float() / 255.0

        if random.random() < 0.8:
            img_s1 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s1)
        img_s1 = blur(img_s1, p=0.5)
        cutmix_box1 = obtain_cutmix_box(self.size, p=0.5)
        img_s1 = torch.from_numpy(np.array(img_s1)).unsqueeze(0).float() / 255.0

        if random.random() < 0.8:
            img_s2 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s2)
        img_s2 = blur(img_s2, p=0.5)
        cutmix_box2 = obtain_cutmix_box(self.size, p=0.5)
        img_s2 = torch.from_numpy(np.array(img_s2)).unsqueeze(0).float() / 255.0

        # h, w = mask.shape
        # ignore_mask = Image.fromarray(np.zeros((h, w)))
        #
        # ignore_mask = torch.from_numpy(np.array(ignore_mask)).long()
        #
        # mask = torch.from_numpy(np.array(mask)).long()
        # ignore_mask[mask == 254] = 255

        # return img, img_s1, img_s2, cutmix_box1, cutmix_box2, ignore_mask
        return img, img_s1, img_s2, cutmix_box1, cutmix_box2

    def __len__(self):
        return len(self.ids)

'''
    Based on ACDCDataset above, 
    adding the ability that makes the dataloader able to process datasets such as Kvasir and TN3K
'''
class KvasirDataset(Dataset):
    def __init__(self, name, root, mode, size=None, id_path=None, nsample=None):
        self.name = name
        self.root = root
        self.mode = mode
        self.size = size

        if mode == 'train_l' or mode == 'train_u':
            with open(id_path, 'r') as f:
                self.ids = f.read().splitlines()
            if mode == 'train_l' and nsample is not None and nsample > len(self.ids):
                self.ids *= math.ceil(nsample / len(self.ids))
                self.ids = self.ids[:nsample]
        elif mode == 'val':
            with open('splits/%s/val.txt' % name, 'r') as f:
                self.ids = f.read().splitlines()
        elif mode == 'valtest':
            with open('splits/%s/valtest.txt' % name, 'r') as f:
                self.ids = f.read().splitlines()
        elif mode == 'test':
            with open('splits/%s/test.txt' % name, 'r') as f:
                self.ids = f.read().splitlines()

    def __getitem__(self, item):
        id = self.ids[item]
        sample = h5py.File(os.path.join(self.root, id), 'r')
        img = sample['image'][:]            # (h, w, 3)
        mask = sample['label'][:]           # (h, w) or (h, w, 3)
        # print(img.shape)
        # print(mask.shape)
        # return
        img = Image.fromarray(img)
        mask = Image.fromarray(mask).convert('L')

        if self.mode == 'val':
            # print('id:', id)
            file_name = id.split('/')[-1].split('.')[0]  # patient028_frame01
            img, mask = normalize(img, mask)
            return img, mask, file_name
        elif self.mode == 'valtest':
            # print('id:', id)
            file_name = id.split('/')[-1].split('.')[0]     # patient028_frame01
            img, mask = normalize(img, mask)
            return img, mask, file_name
        elif self.mode == 'test':
            # print('id:', id)
            file_name = id.split('/')[-1].split('.')[0]     # patient011_frame01
            img, mask = normalize(img, mask)
            return img, mask, file_name
        img, mask = resize(img, mask, (0.5, 2.0))
        ignore_value = 254 if self.mode == 'train_u' else 255
        img, mask = crop(img, mask, self.size, ignore_value)
        img, mask = hflip(img, mask, p=0.5)

        # arr = np.array(img)
        # print(f'id: {id} and img.shape: {arr.shape}')   # (355, 500, 3)

        if self.mode == 'train_l':
            # normalize(img, mask)
            # arr = np.array(img)
            # print(f'id: {id} and img.shape: {arr.shape}')   # (3, 252, 252)
            return normalize(img, mask)


        img_w, img_s1, img_s2 = deepcopy(img), deepcopy(img), deepcopy(img)

        if random.random() < 0.8:
            img_s1 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s1)
        # img_s1 = transforms.RandomGrayscale(p=0.2)(img_s1)
        img_s1 = blur(img_s1, p=0.5)
        cutmix_box1 = obtain_cutmix_box(self.size, p=0.5)
        # img_s1 = torch.from_numpy(np.array(img_s1)).unsqueeze(0).float() / 255.0

        if random.random() < 0.8:
            img_s2 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s2)
        img_s2 = blur(img_s2, p=0.5)
        cutmix_box2 = obtain_cutmix_box(self.size, p=0.5)

        img_s1 = normalize(img_s1)
        img_s2 = normalize(img_s2)

        mask = torch.from_numpy(np.array(mask)).long()
        # img_s2 = torch.from_numpy(np.array(img_s2)).unsqueeze(0).float() / 255.0

        # h, w = mask.shape
        # ignore_mask = Image.fromarray(np.zeros((h, w)))
        #
        # ignore_mask = torch.from_numpy(np.array(ignore_mask)).long()
        #
        # mask = torch.from_numpy(np.array(mask)).long()
        # ignore_mask[mask == 254] = 255

        # return img, img_s1, img_s2, cutmix_box1, cutmix_box2, ignore_mask
        # file_name = id.split('/')[-1].split('.')[0]  # patient028_frame01
        return normalize(img_w), img_s1, img_s2, cutmix_box1, cutmix_box2

    def __len__(self):
        return len(self.ids)