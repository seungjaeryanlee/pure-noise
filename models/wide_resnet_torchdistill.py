"""
From torchdistill

https://github.com/yoshitomo-matsubara/torchdistill/blob/main/torchdistill/models/classification/wide_resnet.py
"""
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor
from .noise_bn_runner import NoiseBnRunner
from .noise_bn_sequential import NoiseBnSequential


class WideBasicBlock(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1, noise_bn_runner=NoiseBnRunner.STANDARD):
        super().__init__()
        self.noise_bn_runner = noise_bn_runner
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.bn1_noise = nn.BatchNorm2d(in_planes) if noise_bn_runner == NoiseBnRunner.AUXBN else None
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.bn2_noise = nn.BatchNorm2d(in_planes) if noise_bn_runner == NoiseBnRunner.AUXBN else None
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
            )

    def forward(self, x, noise_mask=None):
        out = self.noise_bn_runner.run(x, noise_mask=noise_mask, natural_bn=self.bn1, noise_bn=self.bn1_noise)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.dropout(out)
        out = self.noise_bn_runner.run(x, noise_mask=noise_mask, natural_bn=self.bn2, noise_bn=self.bn2_noise)
        out = self.relu(out)
        out = self.conv2(out)
        out += self.shortcut(x)
        return out, noise_mask


class WideResNet(nn.Module):
    def __init__(self, depth, k, dropout_p, block, num_classes, norm_layer=None, noise_bn_runner=NoiseBnRunner.STANDARD):
        super().__init__()
        self.noise_bn_runner = noise_bn_runner
        n = (depth - 4) / 6
        stage_sizes = [16, 16 * k, 32 * k, 64 * k]
        self.in_planes = 16
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv1 = nn.Conv2d(3, stage_sizes[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_wide_layer(block, stage_sizes[1], n, dropout_p, 1)
        self.layer2 = self._make_wide_layer(block, stage_sizes[2], n, dropout_p, 2)
        self.layer3 = self._make_wide_layer(block, stage_sizes[3], n, dropout_p, 2)
        self.bn1 = norm_layer(stage_sizes[3])
        self.bn1_noise = norm_layer(stage_sizes[3]) if noise_bn_runner == NoiseBnRunner.AUXBN else None
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(stage_sizes[3], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1] * (int(num_blocks) - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride, noise_bn_runner=self.noise_bn_runner))
            self.in_planes = planes
        return NoiseBnSequential(*layers)

    def _forward_impl(self, x: Tensor, noise_mask=None) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.layer1(x, noise_mask)
        x = self.layer2(x, noise_mask)
        x = self.layer3(x, noise_mask)
        x = self.noise_bn_runner.run(x, noise_mask=noise_mask, natural_bn=self.bn1, noise_bn=self.bn1_noise)
        x = self.relu(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def forward(self, x: Tensor, noise_mask=None) -> Tensor:
        return self._forward_impl(x, noise_mask=noise_mask)
