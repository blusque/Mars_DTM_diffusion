""" Full assembly of the parts to form the complete network """
import torch
import torch.nn as nn


class ConvBlock_(nn.Module):
    """BN-ReLU-Conv(k=3x3)"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock_, self).__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels, out_channels
                      , kernel_size=kernel_size
                      , stride=stride
                      , padding=padding)
        )

    def forward(self, x):
        output = self.conv(x)
        return output


class DenseBlock(nn.Module):
    """Conv * 5 and dense connect"""

    def __init__(self, in_channels, out_channels, k):
        super(DenseBlock, self).__init__()
        self.conv1 = ConvBlock_(in_channels, k)
        self.conv2 = ConvBlock_(in_channels + k * 1, k)
        self.conv3 = ConvBlock_(in_channels + k * 2, k)
        self.conv4 = ConvBlock_(in_channels + k * 3, k)
        self.conv5 = ConvBlock_(in_channels + k * 4, out_channels)

    def forward(self, x):
        input1 = x
        output1 = self.conv1(input1)
        input2 = torch.cat((output1, x), 1)
        output2 = self.conv2(input2)
        input3 = torch.cat((output2, output1, x), 1)
        output3 = self.conv3(input3)
        input4 = torch.cat((output3, output2, output1, x), 1)
        output4 = self.conv4(input4)
        input5 = torch.cat((output4, output3, output2, output1, x), 1)
        output5 = self.conv5(input5)
        return output5


class Down(nn.Module):
    """Encoder arm blocks"""

    def __init__(self, in_channels, out_channels, k):
        super(Down, self).__init__()
        self.down = nn.Sequential(
            DenseBlock(in_channels, in_channels, k),
            ConvBlock_(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        output = self.down(x)
        return output


class UPB(nn.Module):
    """Unpooling-Conv-ReLU-Conv-ReLU
                |_____Conv_____|
    """

    def __init__(self, in_channels, out_channels):
        super(UPB, self).__init__()
        self.up = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(1e-2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        )
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2)

    def forward(self, x, indices):
        unpooled = self.up(x, indices)
        i = self.shortcut(unpooled)
        f = self.conv(unpooled)
        output = i + f
        return output


class Up(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.conv = nn.Sequential(
            ConvBlock_(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(1e-2, inplace=True)
        )
        self.up = UPB(out_channels, out_channels)

    def forward(self, x1, x2, indices):
        output1 = self.conv(x1)
        output2 = self.up(output1, indices)
        output = torch.cat((output2, x2), 1)
        return output


def get_indices(batch_size, input_channels, input_size, cuda):
    indices = torch.tensor([[
        [
            [(input_size * j * 2 + i) * 2 for i in range(input_size)]
            for j in range(input_size)
        ]
        for s in range(input_channels)
    ]
        for t in range(batch_size)], dtype=torch.int64)
    if cuda:
        indices = indices.cuda()
    return indices


class Generator(nn.Module):
    """encoder arm plus decoder arm"""

    def __init__(self):
        super(Generator, self).__init__()
        # down arm(encoder)
        self.extract = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(1e-2, inplace=True)
        )
        self.first_pooling = nn.MaxPool2d(3, stride=2, padding=1)
        self.down1 = Down(32, 64, 12)
        self.pooling1 = nn.MaxPool2d(2, stride=2)
        self.down2 = Down(64, 128, 12)
        self.pooling2 = nn.MaxPool2d(2, stride=2)
        self.down3 = Down(128, 256, 12)
        self.pooling3 = nn.MaxPool2d(2, stride=2)
        self.down4 = Down(256, 512, 12)
        self.pooling4 = nn.MaxPool2d(2, stride=2)

        # up arm(decoder)
        self.up1 = Up(512, 512)
        self.up2 = Up(1024, 256)
        self.up3 = Up(512, 128)
        self.up4 = Up(256, 64)
        self.up5 = Up(128, 32)
        self.reconstruct = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(1e-2, inplace=True)
        )
        self.on_cuda = False

    def forward(self, x):
        # encoder arm
        feature = self.extract(x)
        first_pooled = self.first_pooling(feature)
        output1 = self.down1(first_pooled)
        pooled1 = self.pooling1(output1)
        output2 = self.down2(pooled1)
        pooled2 = self.pooling2(output2)
        output3 = self.down3(pooled2)
        pooled3 = self.pooling3(output3)
        output4 = self.down4(pooled3)  # Nx512x16x16
        encoder_result = self.pooling4(output4)

        # decoder arm
        if output1.is_cuda:
            self.on_cuda = True
        indices1 = get_indices(output4.shape[0]
                               , output4.shape[1], 16, self.on_cuda)
        up_result1 = self.up1(encoder_result, output4, indices1)
        # print(up_result1.shape)
        indices2 = get_indices(up_result1.shape[0]
                               , up_result1.shape[1] // 4, 32, self.on_cuda)
        up_result2 = self.up2(up_result1, output3, indices2)
        # print(up_result2.shape)
        indices3 = get_indices(up_result2.shape[0]
                               , up_result2.shape[1] // 4, 64, self.on_cuda)
        up_result3 = self.up3(up_result2, output2, indices3)
        # print(up_result3.shape)
        indices4 = get_indices(up_result3.shape[0]
                               , up_result3.shape[1] // 4, 128, self.on_cuda)
        up_result4 = self.up4(up_result3, output1, indices4)
        # print(up_result4.shape)
        indices5 = get_indices(up_result4.shape[0]
                               , up_result4.shape[1] // 4, 256, self.on_cuda)
        up_result5 = self.up5(up_result4, feature, indices5)
        # print(up_result5.shape)
        result = self.reconstruct(up_result5)
        # print(result.shape)
        return result
