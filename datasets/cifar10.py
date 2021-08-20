import os
import cv2
import pickle
import numpy as np
from scipy import signal
from PIL import Image

import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder
from datasets.transforms import train_transforms, test_transforms

class CIFARC(CIFAR10):
    def __init__(
            self,
            root,
            key = 'zoom_blur',
            transform = None,
            target_transform = None,
    ):

        super(CIFAR10, self).__init__(root, transform=transform,
                                      target_transform=target_transform)

        data_path = os.path.join(root, key+'.npy')
        labels_path = os.path.join(root, 'labels.npy')

        self.data = np.load(data_path)
        self.targets = np.load(labels_path)

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class CIFAR10D(object):
    def __init__(self, dataroot='', use_gpu=True, num_workers=4, batch_size=128, _transforms='', _eval=False):

        transforms_list = train_transforms(_transforms)

        train_transform = transforms.Compose(transforms_list)
        test_transform = test_transforms()
        self.train_transform = train_transform

        pin_memory = True if use_gpu else False

        data_root = os.path.join(dataroot, 'cifar10')

        trainset = CIFAR10(root=data_root, train=True, download=True, transform=train_transform)
        
        self.train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        
        testset = CIFAR10(root=data_root, train=False, download=True, transform=test_transform)
        
        self.test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        if _eval:
            self.out_loaders = dict()
            self.out_keys = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
                            'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
                            'brightness', 'contrast', 'elastic_transform', 'pixelate',
                            'jpeg_compression']

            data_root = os.path.join(dataroot, 'CIFAR-10-C')
            for key in self.out_keys:
                outset = CIFARC(root=data_root, key=key, transform=test_transform)
                out_loader = torch.utils.data.DataLoader(
                    outset, batch_size=batch_size, shuffle=False,
                    num_workers=num_workers, pin_memory=pin_memory,
                )
                self.out_loaders[key] = out_loader
        
        self.num_classes = 10


class CIFAR100D(object):
    def __init__(self, dataroot='', use_gpu=True, num_workers=4, batch_size=128, _transforms='', _eval=False):

        transforms_list = train_transforms(_transforms)

        train_transform = transforms.Compose(transforms_list)
        test_transform = test_transforms()
        self.train_transform = train_transform

        pin_memory = True if use_gpu else False

        data_root = os.path.join(dataroot, 'cifar100')

        trainset = CIFAR100(root=data_root, train=True, download=True, transform=train_transform)
        
        self.train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        
        testset = CIFAR100(root=data_root, train=False, download=True, transform=test_transform)
        
        self.test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        if _eval:
            self.out_loaders = dict()
            self.out_keys = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
                            'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
                            'brightness', 'contrast', 'elastic_transform', 'pixelate',
                            'jpeg_compression']

            data_root = os.path.join(dataroot, 'CIFAR-100-C')
            for key in self.out_keys:
                outset = CIFARC(root=data_root, key=key, transform=test_transform)
                out_loader = torch.utils.data.DataLoader(
                    outset, batch_size=batch_size, shuffle=False,
                    num_workers=num_workers, pin_memory=pin_memory,
                )
                self.out_loaders[key] = out_loader

        self.num_classes = 100
