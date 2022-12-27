"""
From xternalz/WideResNet-pytorch

https://github.com/xternalz/WideResNet-pytorch/blob/master/wideresnet.py

Replaces BN with DAR-BN (Zada et al., 2021).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from code_from_paper import dar_bn

class BasicBlockDarBn(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlockDarBn, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None
    def forward(self, x_and_noise_mask):
        """

        :return: tuple of output and noise mask. Outputs noise mask for downstream modules in nn.Sequential.
        """
        x, noise_mask = x_and_noise_mask
        if not self.equalInOut:
            x = self.relu1(dar_bn(self.bn1, x, noise_mask))
        else:
            out = self.relu1(dar_bn(self.bn1, x, noise_mask))
        out = self.relu2(dar_bn(self.bn2, self.conv1(out if self.equalInOut else x), noise_mask))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return (torch.add(x if self.equalInOut else self.convShortcut(x), out), noise_mask)

class NetworkBlockDarBn(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlockDarBn, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)
    def forward(self, x, noise_mask):
        # Pass tuple of x and noise_mask because Sequential only accepts one input.
        out, noise_mask = self.layer((x, noise_mask))
        return out
    
class WideResNetDarBn(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0):
        super(WideResNetDarBn, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlockDarBn
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlockDarBn(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlockDarBn(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlockDarBn(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
    def forward(self, x, noise_mask):
        out = self.conv1(x)
        out = self.block1(out, noise_mask)
        out = self.block2(out, noise_mask)
        out = self.block3(out, noise_mask)
        out = self.relu(dar_bn(self.bn1, out[0], noise_mask))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)