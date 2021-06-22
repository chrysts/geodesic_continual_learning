import os
import shutil
import time
import pprint

import torch
import torch.nn as nn
import torch.autograd.variable as Variable

from math import sqrt
from numpy.random import seed
from numpy.random import randn
from numpy import mean
from scipy.stats import sem
from scipy.stats import t
import numpy as np
from collections import OrderedDict
import os.path as osp

class GaussianNoise(nn.Module):

    def __init__(self, batch_size, input_shape=(3, 84, 84), std=0.05):
        super(GaussianNoise, self).__init__()
        self.shape = (batch_size,) + input_shape
        self.noise = Variable(torch.zeros(self.shape).cuda())
        self.std = std

    def forward(self, x, std=0.15):
        noise = Variable(torch.zeros(x.shape).cuda())
        noise = noise.data.normal_(0, std=std)
        return x + noise


def set_gpu(x):
    os.environ['CUDA_VISIBLE_DEVICES'] = x
    print('using gpu:', x)


def clone(tensor):
    """Detach and clone a tensor including the ``requires_grad`` attribute.
    Arguments:
        tensor (torch.Tensor): tensor to clone.
    """
    cloned = tensor.clone()#tensor.detach().clone()
    # cloned.requires_grad = tensor.requires_grad
    # if tensor.grad is not None:
    #     cloned.grad = clone(tensor.grad)
    return cloned

def clone_state_dict(state_dict):
    """Clone a state_dict. If state_dict is from a ``torch.nn.Module``, use ``keep_vars=True``.
    Arguments:
        state_dict (OrderedDict): the state_dict to clone. Assumes state_dict is not detached from model state.
    """
    return OrderedDict([(name, clone(param)) for name, param in state_dict.items()])

def ensure_path(path):
    if os.path.exists(path):
        if input('{} exists, remove? ([y]/n)'.format(path)) != 'n':
            shutil.rmtree(path)
            os.mkdir(path)
    else:
        os.mkdir(path)


class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v


def count_acc(logits, label):
    pred = torch.argmax(logits, dim=1)
    return (pred == label).type(torch.cuda.FloatTensor).mean().item()


def dot_metric(a, b):
    return torch.mm(a, b.t())


def euclidean_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    #logits = -((a - b)**2).sum(dim=2)
    logits = -((a - b)**2).sum(dim=2)
    return logits


def save_model(model, name, save_path):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    torch.save(model.state_dict(), osp.join(save_path, name + '.pth'))

class Timer():

    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / p
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)

_utils_pp = pprint.PrettyPrinter()
def pprint(x):
    _utils_pp.pprint(x)


def l2_loss(pred, label):
    return ((pred - label)**2).sum() / len(pred) / 2




def flip(x, dim):
    dim = x.dim() + dim if dim < 0 else dim
    return x[tuple(slice(None, None) if i != dim
                   else torch.arange(x.size(i)-1, -1, -1).long()
                   for i in range(x.dim()))]


def perturb(data):

    randno = np.random.randint(0, 5)
    if randno == 1:
        return torch.cat((data, data.flip(3)), dim=0)
    elif randno == 2: #180
        return torch.cat((data, data.flip(2)), dim=0)
    elif randno == 3: #90
        return torch.cat((data, data.transpose(2,3)), dim=0)
    else:
        return torch.cat((data, data.transpose(2, 3).flip(3)), dim=0)