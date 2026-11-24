import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, n=3, num_classes=10):
        super().__init__()

        self.in_channels = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)

        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(16, 2 * n, stride=1)
        self.layer2 = self._make_layer(32, 2 * n, stride=2)
        self.layer3 = self._make_layer(64, 2 * n, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, out_channels, num_convs, stride):

        layers = []

        for i in range(num_convs):

            conv_stride = stride if i == 0 else 1

            layers.append(nn.Conv2d(self.in_channels, out_channels, kernel_size=3, stride=conv_stride, padding=1))

            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))

            self.in_channels = out_channels

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
