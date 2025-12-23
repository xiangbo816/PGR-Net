#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import random
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
import h5py

def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample


class ACDCdataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=RandomGenerator((224,224))):
        self.transform = transform
        self.split = split
        if self.split == "train":
            self.sample_list = open(os.path.join(list_dir, 'train_slice' +'.txt')).readlines()
        elif self.split == "valid" or self.split == "test":
            self.sample_list = open(os.path.join(list_dir, 'test'+'.txt')).readlines()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, 'ACDC_training_slices')
            filepath = data_path + "/{}".format(slice_name) + ".h5"
            h5f = h5py.File(filepath, 'r')
            image = h5f['image'][:]
            label = h5f['label'][:]
        else:
            vol_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, 'ACDC_training_volumes')
            filepath = data_path + "/{}".format(vol_name) + ".h5"
            h5f = h5py.File(filepath, 'r')
            image = h5f['image'][:]
            label = h5f['label'][:]

        sample = {'image': image, 'label': label}
        if self.transform and self.split == "train":
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample
