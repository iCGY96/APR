import torch
from torchvision import transforms

from datasets.APR import APRecombination


normalize = transforms.Compose([
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
    ])

def train_transforms(_transforms):
    transforms_list = []
    if 'aprs' in _transforms:
        print('APRecombination', _transforms)
        transforms_list.extend([
            transforms.RandomApply([APRecombination()], p=1.0),
            transforms.RandomCrop(32, padding=4, fill=128),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    else:
        transforms_list.extend([
            transforms.RandomCrop(32, padding=4, fill=128),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

    return transforms_list


def test_transforms():
    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    return test_transform