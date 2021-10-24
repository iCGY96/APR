import os
import sys
import argparse
import datetime
import time
import csv
import os.path as osp
import numpy as np
import warnings
import importlib
import pandas as pd
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import torchvision
from datasets import CIFAR10D, CIFAR100D
from utils.utils import AverageMeter, Logger, save_networks, load_networks
from core import train, test, test_robustness

parser = argparse.ArgumentParser("Training")

# dataset
parser.add_argument('--data', type=str, default='./data')
parser.add_argument('--outf', type=str, default='./results')

parser.add_argument('-d', '--dataset', type=str, default='cifar10')
parser.add_argument('--workers', default=8, type=int, help="number of data loading workers (default: 4)")

# optimization
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--lr', type=float, default=0.1, help="learning rate for model")
parser.add_argument('--max-epoch', type=int, default=200)
parser.add_argument('--stepsize', type=int, default=30)
parser.add_argument('--aug', type=str, default='none', help='none, aprs')

# model
parser.add_argument('--model', type=str, default='wider_resnet_28_10')

# misc
parser.add_argument('--eval-freq', type=int, default=10)
parser.add_argument('--print-freq', type=int, default=100)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--use-cpu', action='store_true')
parser.add_argument('--eval', action='store_true', help="Eval", default=False)

# parameters for generating adversarial examples
parser.add_argument('--epsilon', '-e', type=float, default=0.0157,
                    help='maximum perturbation of adversaries (4/255=0.0157)')
parser.add_argument('--alpha', '-a', type=float, default=0.00784,
                    help='movement multiplier per iteration when generating adversarial examples (2/255=0.00784)')
parser.add_argument('--k', '-k', type=int, default=10,
                    help='maximum iteration when generating adversarial examples')
parser.add_argument('--perturbation_type', '-p', choices=['linf', 'l2'], default='linf',
                    help='the type of the perturbation (linf or l2)')

args = parser.parse_args()
options = vars(args)

if not os.path.exists(options['outf']):
    os.makedirs(options['outf'])

if not os.path.exists(options['data']):
    os.makedirs(options['data'])

sys.stdout = Logger(osp.join(options['outf'], 'logs.txt'))

def main():
    torch.manual_seed(options['seed'])
    os.environ['CUDA_VISIBLE_DEVICES'] = options['gpu']
    use_gpu = torch.cuda.is_available()
    if options['use_cpu']: use_gpu = False

    options.update({'use_gpu': use_gpu})

    if use_gpu:
        print("Currently using GPU: {}".format(options['gpu']))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(options['seed'])
    else:
        print("Currently using CPU")

    if 'cifar10' == options['dataset']:
        Data = CIFAR10D(dataroot=options['data'], batch_size=options['batch_size'], _transforms=options['aug'], _eval=options['eval'])
        OODData = CIFAR100D(dataroot=options['data'], batch_size=options['batch_size'], _transforms=options['aug'])
    else:
        Data = CIFAR100D(dataroot=options['data'], batch_size=options['batch_size'], _transforms=options['aug'], _eval=options['eval'])
        OODData = CIFAR10D(dataroot=options['data'], batch_size=options['batch_size'], _transforms=options['aug'])
    
    trainloader, testloader, outloader = Data.train_loader, Data.test_loader, OODData.test_loader
    num_classes = Data.num_classes

    print("Creating model: {}".format(options['model']))
    if 'wide_resnet' in options['model']:
        print('wide_resnet')
        from model.wide_resnet import WideResNet
        net = WideResNet(40, num_classes, 2, 0.0)
    elif 'allconv' in options['model']:
        print('allconv')
        from model.allconv import AllConvNet
        net = AllConvNet(num_classes)
    elif 'densenet' in options['model']:
        print('densenet')
        from model.densenet import  densenet
        net = densenet(num_classes=num_classes)
    elif 'resnext' in options['model']:
        print('resnext29')
        from model.resnext import resnext29
        net = resnext29(num_classes)
    else:
        print('resnet18')
        from model.resnet import ResNet18
        net = ResNet18(num_classes=num_classes)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    if use_gpu:
        net = nn.DataParallel(net, device_ids=[i for i in range(len(options['gpu'].split(',')))]).cuda()
        criterion = criterion.cuda()

    file_name = '{}_{}_{}'.format(options['model'], options['dataset'], options['aug'])

    if options['eval']:
        net, criterion = load_networks(net, options['outf'], file_name, criterion=criterion)
        outloaders = Data.out_loaders
        results = test(net, criterion, testloader, outloader, epoch=0, **options)
        acc = results['ACC']
        res = dict()
        res['ACC'] = dict()
        acc_res = []
        for key in Data.out_keys:
            results = test_robustness(net, criterion, outloaders[key], epoch=0, label=key, **options)
            print('{} (%): {:.3f}\t'.format(key, results['ACC']))
            res['ACC'][key] = results['ACC']
            acc_res.append(results['ACC'])
        print('Mean ACC:', np.mean(acc_res))
        print('Mean Error:', 100-np.mean(acc_res))

        return

    params_list = [{'params': net.parameters()},
                {'params': criterion.parameters()}]


    optimizer = torch.optim.SGD(params_list, lr=options['lr'], momentum=0.9, nesterov=True, weight_decay=5e-4)
    scheduler = lr_scheduler.MultiStepLR(optimizer, gamma=0.2, milestones=[60, 120, 160, 190])

    start_time = time.time()

    best_acc = 0.0
    for epoch in range(options['max_epoch']):
        print("==> Epoch {}/{}".format(epoch+1, options['max_epoch']))

        train(net, criterion, optimizer, trainloader, epoch=epoch, **options)

        if options['eval_freq'] > 0 and (epoch+1) % options['eval_freq'] == 0 or (epoch+1) == options['max_epoch'] or epoch > 160:
            print("==> Test")
            results = test(net, criterion, testloader, outloader, epoch=epoch, **options)

            if best_acc < results['ACC']:
                best_acc = results['ACC']
                print("Best Acc (%): {:.3f}\t".format(best_acc))
            
            save_networks(net, options['outf'], file_name, criterion=criterion)

        scheduler.step()

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))

if __name__ == '__main__':
    main()

