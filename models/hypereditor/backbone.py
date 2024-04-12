import torch
from torch import nn
from torch.nn import BatchNorm2d, PReLU, Sequential, Module
from torchvision.models import resnet34

from models.hypereditor.hypernetwork import Hypernetwork


import clip
import torchvision.transforms as transforms
from collections import OrderedDict


class BackboneNet(Module):

    def __init__(self, opts):
        super(BackboneNet, self).__init__()

        self.conv1 = nn.Conv2d(opts.input_nc, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = BatchNorm2d(64)
        self.relu = PReLU(64)

        self.fc_gamma = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(512, 512)),
            ('relu1', nn.ReLU(inplace=True))
        ]))
        self.fc_beta = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(512, 512)),
            ('relu1', nn.ReLU(inplace=True))
        ]))

        resnet_basenet = resnet34(pretrained=True)
        blocks = [
            resnet_basenet.layer1,
            resnet_basenet.layer2,
            resnet_basenet.layer3,
            resnet_basenet.layer4
        ]
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(bottleneck)
        self.body = Sequential(*modules)

        if len(opts.layers_to_tune) == 0:
            self.layers_to_tune = list(range(opts.n_hypernet_outputs))
        else:
            self.layers_to_tune = [int(l) for l in opts.layers_to_tune.split(',')]

        self.hypernetworks = nn.ModuleList()
        self.n_outputs = opts.n_hypernet_outputs

        for layer_idx in range(self.n_outputs):
            if layer_idx in self.layers_to_tune:
                hypernetwork = Hypernetwork(layer_idx, opts, n_channels=512, inner_c=256)
            else:
                hypernetwork = None
            self.hypernetworks.append(hypernetwork)

    def forward(self, x, style_fix=None, text_fix=None, delta_i=None, delta_i_globle=None, weight_choose=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.body(x)
        #
        if delta_i is not None:
            weight = self.fc_gamma(delta_i)
            bias = self.fc_beta(delta_i)
            size = x.size()
            weight = weight.unsqueeze(-1).unsqueeze(-1).expand(size)
            bias = bias.unsqueeze(-1).unsqueeze(-1).expand(size)
            x = x * weight + bias

        weight_deltas = []

        if weight_choose is None:
            for j in range(self.n_outputs):
                if self.hypernetworks[j] is not None:
                    delta = self.hypernetworks[j](x)
                else:
                    delta = None
                weight_deltas.append(delta)
        else:
            for j in range(self.n_outputs):
                if j in weight_choose:
                    delta = self.hypernetworks[j](x)
                else:
                    delta = None
                weight_deltas.append(delta)
        return weight_deltas
