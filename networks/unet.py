import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(convolution => [GroupNorm] => LeakyReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(1, out_channels, 1e-3),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(1, out_channels, 1e-3),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            DoubleConv(in_channels, out_channels),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        return self.maxpool_conv(x)
    
class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2, skip=True):
        x1 = self.up(x1)
        
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        if skip:
            x = torch.cat([x2, x1], dim=1)
        else:
            x = x1
        return self.conv(x)
    
class RepeatDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, repeat=1):
        super().__init__()
        self.block = [Down(in_channels, out_channels)]
        for i in range(0, repeat-1):
            self.block.append(DoubleConv(out_channels, out_channels))
        self.block = nn.ModuleList(self.block)
    
    def forward(self, x):
        for layer in self.block:
            x = layer(x)
        return x
    
class RepeatBlock(nn.Module):
    def __init__(self, in_channels, out_channels, repeat=1):
        super().__init__()
        self.block = []
        for i in range(0, repeat):
            self.block.append(DoubleConv(out_channels, out_channels))
        self.block = nn.ModuleList(self.block)
    
    def forward(self, x):
        for layer in self.block:
            x = layer(x)
        return x

class RepeatUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, repeat=1):
        super().__init__()
        self.block = [Up(in_channels, out_channels)]
        for i in range(0, repeat-1):
            self.block.append(DoubleConv(out_channels, out_channels))
        self.block = nn.ModuleList(self.block)
    
    def forward(self, x1, x2, skip=True):
        x = self.block[0](x1, x2, skip)
        for layer in self.block[1:]:
            x = layer(x)
        return x
    

class UNet(nn.Module):
    def __init__(self, in_channels, d, repeat=4, skip=True):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.d = d
        self.skip = skip
        
        # Encoder path
        self.encoder = []
        for depth in range(0, d):
            in_dim = 2**depth * self.in_channels
            self.encoder.append(RepeatDownBlock(in_dim, 2*in_dim, repeat=repeat))
        self.encoder = nn.ModuleList(self.encoder)
        
        bottleneck_dim = 2*in_dim
        self.bottleneck = RepeatBlock(bottleneck_dim, bottleneck_dim, repeat=2*repeat)

        # Decoder path
        self.decoder = []
        for depth in range(d-1, -1, -1):
            in_dim = 2**(depth+1) * self.in_channels
            in_ch_ = 2*in_dim if skip else in_dim
            self.decoder.append(RepeatUpBlock(in_ch_, in_dim//2, repeat=repeat)) #due to skip connections
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.decoder.append(DoubleConv(self.in_channels, self.in_channels))
        self.decoder = nn.ModuleList(self.decoder)
        
    def forward(self, x):
        enc_outputs = []
        for layer in self.encoder:
            x = layer(x)
            enc_outputs.append(x)
        x = self.bottleneck(x)
        for i, layer in enumerate(self.decoder[:-1]):
            x = layer(x, enc_outputs[-(i+1)], skip=self.skip)
        x = self.upsample(x)
        x = self.decoder[-1](x)
        return x