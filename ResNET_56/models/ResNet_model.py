import torch
import torch.nn as nn


class ResNetBasicBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride


        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)

        self.bn2 = nn.BatchNorm2d(out_channels)


    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)


        if self.stride != 1:
            identity = identity[:, :, ::self.stride, ::self.stride]


        if self.out_channels > self.in_channels:
            diff = self.out_channels - self.in_channels

            zeros_left = identity.new_zeros(identity.size(0), diff // 2, identity.size(2), identity.size(3))

            zeros_right = identity.new_zeros(identity.size(0), diff - diff // 2, identity.size(2), identity.size(3))

            identity = torch.cat([zeros_left, identity, zeros_right], dim=1)


        out += identity
        y = self.relu(out)

        return y


class ResNet(nn.Module):
    def __init__(self, n=3, num_classes=10):
        super().__init__()

        self.in_channels = 16

        # Init conv
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)

        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        # Residual groups
        self.layer1 = self._make_layer(16, n, stride=1)
        self.layer2 = self._make_layer(32, n, stride=2)
        self.layer3 = self._make_layer(64, n, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, out_channels, blocks, stride):

        layers = []

        layers.append(ResNetBasicBlock(self.in_channels, out_channels, stride))

        self.in_channels = out_channels

        for _ in range(1, blocks):
            layers.append(ResNetBasicBlock(self.in_channels, out_channels ))


        return nn.Sequential(*layers)


    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out
        