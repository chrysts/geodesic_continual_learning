#!/usr/bin/env python
# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
from torch.autograd import Variable
import numpy as np
import time
import os
import copy
import argparse
from PIL import Image
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
from utils_incremental.utils_pytorch import *
from algorithms.GFK_distill_cosine import GFK

gfk = GFK()

def compute_accuracy_G(tg_model, tg_feature_model, class_means, G_accum,  evalloader, scale=None, print_info=True, device=None):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tg_model.eval()
    tg_feature_model.eval()

    #evalset = torchvision.datasets.CIFAR100(root='./data', train=False,
    #                                   download=False, transform=transform_test)
    #evalset.test_data = input_data.astype('uint8')
    #evalset.test_labels = input_labels
    #evalloader = torch.utils.data.DataLoader(evalset, batch_size=128,
    #    shuffle=False, num_workers=2)

    correct = 0
    correct_icarl = 0
    correct_ncm = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(evalloader):
            inputs, targets = inputs.cuda() , targets.cuda()#to(device)
            total += targets.size(0)

            outputs = tg_model(inputs)
            outputs = F.softmax(outputs, dim=1)
            if scale is not None:
                assert(scale.shape[0] == 1)
                assert(outputs.shape[1] == scale.shape[1])
                outputs = outputs / scale.repeat(outputs.shape[0], 1).type(torch.FloatTensor).cuda()#to(device)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()

            features_nets = tg_feature_model(inputs)
            outputs_feature = np.squeeze(features_nets.cpu().numpy())
            # Compute score for iCaRL
            sqd_icarl = cdist(class_means[:,:,0].T, outputs_feature, 'sqeuclidean')
            score_icarl = torch.from_numpy((-sqd_icarl).T).cuda()#to(device)
            _, predicted_icarl = score_icarl.max(1)
            correct_icarl += predicted_icarl.eq(targets).sum().item()
            # Compute score for NCM
            #sqd_ncm = cdist(class_means[:,:,1].T, outputs_feature, 'sqeuclidean')

            score = gfk.geodesic_score(torch.from_numpy(class_means[:,:,1].T).float().cuda(),
                                       torch.from_numpy(outputs_feature).float().cuda(), G_accum)
            score_ncm = score#torch.from_numpy((score).T).cuda()#to(device) # N X C
            _, predicted_ncm = score_ncm.max(1)
            correct_ncm += predicted_ncm.eq(targets).sum().item()
            # print(sqd_icarl.shape, score_icarl.shape, predicted_icarl.shape, \
                  # sqd_ncm.shape, score_ncm.shape, predicted_ncm.shape)
    if print_info:
        print("  top 1 accuracy CNN            :\t\t{:.2f} %".format(100.*correct/total))
        print("  top 1 accuracy iCaRL          :\t\t{:.2f} %".format(100.*correct_icarl/total))
        print("  top 1 accuracy NCM            :\t\t{:.2f} %".format(100.*correct_ncm/total))

    cnn_acc = 100.*correct/total
    icarl_acc = 100.*correct_icarl/total
    ncm_acc = 100.*correct_ncm/total

    return [cnn_acc, icarl_acc, ncm_acc]


def compute_accuracy(tg_model, tg_feature_model, class_means,  evalloader, scale=None, print_info=True, device=None):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tg_model.eval()
    tg_feature_model.eval()

    #evalset = torchvision.datasets.CIFAR100(root='./data', train=False,
    #                                   download=False, transform=transform_test)
    #evalset.test_data = input_data.astype('uint8')
    #evalset.test_labels = input_labels
    #evalloader = torch.utils.data.DataLoader(evalset, batch_size=128,
    #    shuffle=False, num_workers=2)

    correct = 0
    correct_icarl = 0
    correct_ncm = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(evalloader):
            inputs, targets = inputs.cuda() , targets.cuda()#to(device)
            total += targets.size(0)

            outputs = tg_model(inputs)
            outputs = F.softmax(outputs, dim=1)
            if scale is not None:
                assert(scale.shape[0] == 1)
                assert(outputs.shape[1] == scale.shape[1])
                outputs = outputs / scale.repeat(outputs.shape[0], 1).type(torch.FloatTensor).cuda()#to(device)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()

            features_nets = tg_feature_model(inputs)
            #features_nets = F.adaptive_avg_pool2d(features_nets, 2).view(features_nets.size(0), -1)
            #features_nets, _ = tg_feature_model(inputs, is_feature=True)
            outputs_feature = np.squeeze(features_nets.cpu().numpy())
            # Compute score for iCaRL
            sqd_icarl = cdist(class_means[:,:,0].T, outputs_feature, 'sqeuclidean')
            score_icarl = torch.from_numpy((-sqd_icarl).T).cuda()#to(device)
            _, predicted_icarl = score_icarl.max(1)
            correct_icarl += predicted_icarl.eq(targets).sum().item()
            # Compute score for NCM
            sqd_ncm = cdist(class_means[:,:,1].T, outputs_feature, 'sqeuclidean')

            score_ncm = torch.from_numpy((-sqd_ncm).T).cuda()#to(device) # N X C
            _, predicted_ncm = score_ncm.max(1)
            correct_ncm += predicted_ncm.eq(targets).sum().item()
            # print(sqd_icarl.shape, score_icarl.shape, predicted_icarl.shape, \
            # sqd_ncm.shape, score_ncm.shape, predicted_ncm.shape)
    if print_info:
        print("  top 1 accuracy CNN            :\t\t{:.2f} %".format(100.*correct/total))
        print("  top 1 accuracy iCaRL          :\t\t{:.2f} %".format(100.*correct_icarl/total))
        print("  top 1 accuracy NCM            :\t\t{:.2f} %".format(100.*correct_ncm/total))

    cnn_acc = 100.*correct/total
    icarl_acc = 100.*correct_icarl/total
    ncm_acc = 100.*correct_ncm/total

    return [cnn_acc, icarl_acc, ncm_acc]