#!/usr/bin/env python
# coding=utf-8
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

def compute_features(tg_feature_model, evalloader, num_samples, num_features, device=None):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tg_feature_model.eval()

    #evalset = torchvision.datasets.CIFAR100(root='./data', train=False,
    #                                   download=False, transform=transform_test)
    #evalset.test_data = input_data.astype('uint8')
    #evalset.test_labels = np.zeros(input_data.shape[0])
    #evalloader = torch.utils.data.DataLoader(evalset, batch_size=128,
    #    shuffle=False, num_workers=2)

    features = np.zeros([num_samples, num_features])
    start_idx = 0
    with torch.no_grad():
        for inputs, targets in evalloader:
            inputs = inputs.cuda()#to(device)
            features_nets = tg_feature_model(inputs )
            #features_nets = F.adaptive_avg_pool2d(features_nets, 2).view(features_nets.size(0), -1)
            #features_nets, _ = tg_feature_model(inputs, is_feature=True )
            features[start_idx:start_idx+inputs.shape[0], :] = np.squeeze(features_nets.cpu().numpy())
            start_idx = start_idx+inputs.shape[0]
    assert(start_idx==num_samples)
    return features
