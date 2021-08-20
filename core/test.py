import os
import cv2
import os.path as osp
import numpy as np
from PIL import Image
import torch.fft
import torch
from torch.autograd import Variable
import torch.nn.functional as F

from core import evaluation
from datasets.transforms import normalize


def test(net, criterion, testloader, outloader, attack=None, epoch=None, **options):
    net.eval()
    correct, total, adv_correct = 0, 0, 0

    torch.cuda.empty_cache()

    _pred_k, _pred_u, _labels = [], [], []

    with torch.no_grad():
        for data, labels in testloader:
            if options['use_gpu']:
                data, labels = data.cuda(), labels.cuda()             
                data = normalize(data)
                logits = net(data, _eval=True)
                predictions = logits.data.max(1)[1]
                total += labels.size(0)
                correct += (predictions == labels.data).sum()
            
                _pred_k.append(logits.data.cpu().numpy())
                _labels.append(labels.data.cpu().numpy())

        for batch_idx, (data, labels) in enumerate(outloader):
            if options['use_gpu']:
                data, labels = data.cuda(), labels.cuda()
                data = normalize(data)
            with torch.set_grad_enabled(False):
                logits = net(data, _eval=True)
                _pred_u.append(logits.data.cpu().numpy())

    _pred_k = np.concatenate(_pred_k, 0)
    _pred_u = np.concatenate(_pred_u, 0)
    _labels = np.concatenate(_labels, 0)
    
    # # Out-of-Distribution detction evaluation
    x1, x2 = np.max(_pred_k, axis=1), np.max(_pred_u, axis=1)
    results = evaluation.metric_ood(x1, x2)['Bas']

    # Accuracy
    acc = float(correct) * 100. / float(total)
    results['ACC'] = acc

    print('Acc: {:.5f}'.format(acc))

    return results

def test_robustness(net, criterion, testloader, epoch=None, label='', **options):
    net.eval()
    results = dict()
    correct, total = 0, 0

    torch.cuda.empty_cache()

    with torch.no_grad():
        for data, labels in testloader:
            if options['use_gpu']:
                data, labels = data.cuda(), labels.cuda()
                data = normalize(data)
            with torch.set_grad_enabled(False):
                logits = net(data, _eval=True)
                predictions = logits.data.max(1)[1]
                total += labels.size(0)
                correct += (predictions == labels.data).sum()

    # Accuracy
    acc = float(correct) * 100. / float(total)
    results['ACC'] = acc

    return results