import torch
import torch.nn as nn


class MergeLayer(nn.Module):
    """merge input height with image"""

    def __init__(self, in_channels, out_channels):
        super(MergeLayer, self).__init__()
        self.h_layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(1e-2, inplace=True)
        )
        self.i_layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(1e-2, inplace=True)
        )

    def forward(self, x_h, x_i):
        # print(x_h.shape)
        # print(x_i.shape)
        output_h = self.h_layer(x_h)
        output_i = self.i_layer(x_i)
        output = torch.cat((output_h, output_i), 1)
        return output


class ConvBlock_(nn.Module):
    """Conv(3x3)-BN-LReLU"""

    def __init__(self, in_channels, out_channels, stride=1):
        super(ConvBlock_, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(1e-2, inplace=True)
        )

    def forward(self, x):
        output = self.conv(x)
        return output


class Discriminator(nn.Module):
    """Height Layer + Image Layer-BN"""

    def __init__(self):
        super(Discriminator, self).__init__()
        self.merge = MergeLayer(1, 32)
        self.convs = nn.Sequential(
            ConvBlock_(64, 64),
            ConvBlock_(64, 64, 2),
            ConvBlock_(64, 128),
            ConvBlock_(128, 128, 2),
            ConvBlock_(128, 256),
            ConvBlock_(256, 256, 2),
            ConvBlock_(256, 512),
            ConvBlock_(512, 512, 2)
        )
        self.fc1 = nn.Linear(512 * 32 * 32, 10)
        self.lrelu = nn.LeakyReLU(1e-2, inplace=True)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x_h, x_i):
        # print("x_h size: ", x_h.shape)
        # print("x_i size: ", x_i.shape)
        merged = self.merge(x_h, x_i)
        # print(merged.shape)
        output = self.convs(merged)
        # print(output.shape)
        output = output.view(-1, 512 * 32 * 32)
        # print(output.shape)
        vector1 = self.fc1(output)
        relu = self.lrelu(vector1)
        result = self.fc2(relu)
        return result
