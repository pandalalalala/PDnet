import math
import torch
from torch import nn
import torch.nn.functional as F
from unet_model import UNet


class PDNet(nn.Module):
    def __init__(self, channels=3, branches = 2):
        super(PDNet, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        self.branches = branches
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, stride=1, padding=padding, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.block2 = UNet() #convBlock()
        self.block3 = nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False)
      
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        outputs = []
        block1 = self.block1(x)
        
        for _ in range(self.branches):
            block2 = self.block2(block1)
            noise = self.block3(block2)
            outputs.append(noise)
            block1 = block1 - block2

        return tuple(outputs)


class convBlock(nn.Module):    
    def __init__(self, features = 64):
        super(convBlock, self).__init__()
        kernel_size = 3
        padding = 1
        
        layers = []

        # layers could be varied here
        layers.append(CnnBlock(features, 5))
        layers.append(ResidualBlock(features))
        layers.append(CnnBlock(features, 5))
        layers.append(ResidualBlock(features))

        self.conv_block = nn.Sequential(*layers)
    def forward(self, x):  
        return self.conv_block(x)


class CnnBlock(nn.Module):
    def __init__(self, features = 64, num_of_layers=20):
        super(CnnBlock, self).__init__()
        kernel_size = 3
        padding = 1        
        layers = []
        for _ in range(num_of_layers):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        self.dncnn = nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.dncnn(x)
        return out

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        return x + residual



# this could be saved for U-net implementation
class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBLock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x

# reference: original DnCNN
class DnCNN(nn.Module):
    def __init__(self, channels, num_of_layers=17):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers-2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)
    def forward(self, x):
        out = self.dncnn(x)
        return out