import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

from enum import Enum

class DatasetsNames(Enum):
    cifar10     = lambda x: {"train" : True if x == "train" else False}
    cifar100    = lambda x: {"train" : True if x == "train" else False}
    cub200      = lambda x: {"train" : True if x == "train" else False}
    kuzushiji49 = lambda x: {"train" : True if x == "train" else False}

    svhn     = lambda x: {"split" : x}
    stl10    = lambda x: {"split" : x}


def load_model(device, modelname, pretrained=True):

    r"""
    """

    model = getattr(models, modelname)(pretrained=pretrained).to(device)

    return model


def load_dataset(name, split, batchsize, datapath, imgsize):

    r"""
    """

    # standard preprocessing steps for imagenet models

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(imgsize),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    transform_test = transforms.Compose([
        transforms.Resize((imgsize, imgsize)),
        transforms.ToTensor(),
        normalize,
    ])

    # transform = transform_train if train else transform_test
    # for our purpose, we only consider the testing case.
    transform = transform_test

    if name == "kuzushiji49":
        from kuzushiji49 import KUZUSHIJI49
        func_ = KUZUSHIJI49
    elif name == "cub200":
        from cub200 import CUB200
        func_ = CUB200
    else:
        func_ = getattr(torchvision.datasets, name.upper())

    kws = {
            "root": datapath,
            "download": True,
            "transform": transform,
    }

    kws = {**kws, **getattr(DatasetsNames, name)(split)}

    dataset = func_(**kws)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batchsize, shuffle=False, num_workers=0)

    return loader
