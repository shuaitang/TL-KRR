import os
import pandas as pd
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torch.utils.data import Dataset

import torch
import numpy as np

from PIL import Image

# def default_loader(path):
#     return Image.open(path).convert('RGB')

class KUZUSHIJI49(VisionDataset):
    resources = ['http://codh.rois.ac.jp/kmnist/dataset/k49/k49-train-imgs.npz',
                 'http://codh.rois.ac.jp/kmnist/dataset/k49/k49-train-labels.npz',
                 'http://codh.rois.ac.jp/kmnist/dataset/k49/k49-test-imgs.npz',
                 'http://codh.rois.ac.jp/kmnist/dataset/k49/k49-test-labels.npz']


    def __init__(self, root, train=True, transform=None, target_transform=None, download=True):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train

        if download:
            self.download()

        if self.train:
            data_file  = np.load(os.path.join(self.root, self.__class__.__name__, 'k49-train-imgs.npz'))['arr_0']
            label_file = np.load(os.path.join(self.root, self.__class__.__name__, 'k49-train-labels.npz'))['arr_0']
        else:
            data_file  = np.load(os.path.join(self.root, self.__class__.__name__, 'k49-test-imgs.npz'))['arr_0']
            label_file = np.load(os.path.join(self.root, self.__class__.__name__, 'k49-test-labels.npz'))['arr_0']

        # data_file = np.expand_dims(data_file, 1)
        # data_file = np.tile(data_file, [1,3,1,1])

        self.data, self.targets = data_file, torch.from_numpy(label_file).long()

    def download(self):

        for url in self.resources:
            if os.path.exists(os.path.join(self.root, self.__class__.__name__, os.path.basename(url))):
                print('Files already downloaded and verified')
            else:
                download_url(url, os.path.join(self.root, self.__class__.__name__))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        img, target = self.data[index], self.targets[index]
        # print()
        img = Image.fromarray(img, mode='L').convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        # print(img.size())
        if self.target_transform is not None:
            target = self.target_transform(target)

        # print(target)
        return img, target
