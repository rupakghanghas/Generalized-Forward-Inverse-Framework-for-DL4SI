import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import numpy as np
from math import ceil
from collections import OrderedDict
from networks.iunet_network import IUnetForwardModel, UNetForwardModel, IUnetForwardModel_Legacy, UNetForwardModel_Legacy


NORM_LAYERS = { 'bn': nn.BatchNorm2d, 'in': nn.InstanceNorm2d, 'ln': nn.LayerNorm }

forward_params = { 
        'WaveformNet':{
            'in_channels': 1, #WaveformNet 
            'encoder_channels': [32, 64, 128, 256, 512], #WaveformNet 
            'decoder_channels': [256, 128, 64, 5],#WaveformNet 
            },
         'FNO':{
             'modes1': 30, #FNO 
             'modes2': 30, #FNO 
             'width': 32, #FNO 
             'out_dim':5 #FNO
          },  
    }

forward_params_legacy = { 'in_channels': 1, 'encoder_channels': [32, 64, 128, 256, 512], 'decoder_channels': [256, 128, 64, 5],
'vel_input_channel': 1,  'vel_encoder_channel': [8, 16, 32, 64, 128], 'vel_decoder_channel': [128, 64, 32, 16, 1],
'amp_input_channel': 5, 'amp_encoder_channel': [8, 16, 32, 64, 128], 'amp_decoder_channel': [128, 64, 32, 16, 5],
'modes1': 30, 'modes2': 30, 'width': 32, 'out_dim':5 }

################################################################
# Basic Blocks
################################################################

class DeconvBlock(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=2, stride=2, padding=0, output_padding=0, norm='bn'):
        super(DeconvBlock, self).__init__()
        layers = [nn.ConvTranspose2d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding)]
        if norm in NORM_LAYERS:
            layers.append(NORM_LAYERS[norm](out_fea))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
    
class ConvBlock(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=3, stride=1, padding=1, norm='bn', relu_slop=0.2, dropout=None):
        super(ConvBlock,self).__init__()
        layers = [nn.Conv2d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size, stride=stride, padding=padding)]
        if norm in NORM_LAYERS:
            layers.append(NORM_LAYERS[norm](out_fea))
        layers.append(nn.LeakyReLU(relu_slop, inplace=True))
        if dropout:
            layers.append(nn.Dropout2d(0.8))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class ConvBlock_Tanh(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=3, stride=1, padding=1, norm='bn'):
        super(ConvBlock_Tanh, self).__init__()
        layers = [nn.Conv2d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size, stride=stride, padding=padding)]
        if norm in NORM_LAYERS:
            layers.append(NORM_LAYERS[norm](out_fea))
        layers.append(nn.Tanh())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
    
############################################################################
# WaveformNet 
############################################################################

class WaveformNet(nn.Module):
    def __init__(self, in_channels, encoder_channels, decoder_channels, **kwargs):
        super(WaveformNet, self).__init__()
        
        encoder_layers = nn.ModuleList()
        decoder_layers = nn.ModuleList()
        
        enlayer1 = nn.Sequential(nn.Conv2d(in_channels, encoder_channels[0],
                                                kernel_size=(5,5), stride=(2,2)),
                                     nn.BatchNorm2d(encoder_channels[0]),
                                     nn.GELU())
        
        encoder_layers.append(enlayer1)
        
        enlayer2 = nn.Sequential(nn.Conv2d(encoder_channels[0], encoder_channels[1],
                                                kernel_size=(5,5), stride=(2,2)),
                                     nn.BatchNorm2d(encoder_channels[1]),
                                     nn.GELU())
        
        encoder_layers.append(enlayer2)
        
        enlayer3 = nn.Sequential(nn.Conv2d(encoder_channels[1], encoder_channels[2],
                                                kernel_size=(3,3), stride=(2,2)),
                                     nn.BatchNorm2d(encoder_channels[2]),
                                     nn.GELU())
        
        encoder_layers.append(enlayer3)
        
        enlayer4 = nn.Sequential(nn.Conv2d(encoder_channels[2], encoder_channels[3],
                                                kernel_size=(3,3), stride=(2,2)),
                                     nn.BatchNorm2d(encoder_channels[3]),
                                     nn.GELU())
        
        encoder_layers.append(enlayer4)
        
        enlayer5 = nn.Sequential(nn.Conv2d(encoder_channels[3], encoder_channels[4],
                                                kernel_size=(3,3), stride=(1,1)),
                                     nn.BatchNorm2d(encoder_channels[4]),
                                     nn.GELU())
        
        encoder_layers.append(enlayer5)
        
        delayer1 = nn.Sequential(nn.ConvTranspose2d(encoder_channels[-1], decoder_channels[0],
                                                kernel_size=(10,5), stride=(4,4), padding=(0,0)),
                             nn.BatchNorm2d(decoder_channels[0]),
                             nn.Tanh())
        
        decoder_layers.append(delayer1)
        
        delayer2 = nn.Sequential(nn.ConvTranspose2d(decoder_channels[0], decoder_channels[1],
                                                kernel_size=(10,5), stride=(4,2), padding=(0,0)),
                             nn.BatchNorm2d(decoder_channels[1]),
                             nn.Tanh())
        
        decoder_layers.append(delayer2)
        
        delayer3 = nn.Sequential(nn.ConvTranspose2d(decoder_channels[1], decoder_channels[2],
                                                kernel_size=(10,5), stride=(4,2), padding=(0,0)),
                             nn.BatchNorm2d(decoder_channels[2]),
                             nn.Tanh())
        
        decoder_layers.append(delayer3)
        
        delayer4 = nn.Sequential(nn.ConvTranspose2d(decoder_channels[2], decoder_channels[3],
                                                kernel_size=(8,4), stride=(4,2), padding=(0,0)),
                             nn.BatchNorm2d(decoder_channels[3]),
                             nn.Tanh(),
                             nn.Conv2d(decoder_channels[3], decoder_channels[3], kernel_size=(10,5), stride=(2,2)))
        
        decoder_layers.append(delayer4)
        
        delayer5 = nn.Upsample((1000,70), mode='bilinear')
        
        decoder_layers.append(delayer5)
        
        self.encoder_layers = nn.Sequential(*encoder_layers)
        self.decoder_layers = nn.Sequential(*decoder_layers)
        
    def forward(self, x):
        x = self.encoder_layers(x)
        x = self.decoder_layers(x)
        return x
    
    def embedding(self, x):
        x = self.encoder_layers(x)
        return x
    
    def decoder(self, x):
        x = self.decoder_layers(x)
        return x

################################################################
# Waveform Net - v2 (performance similar to v1)
################################################################

class WaveformNet_V2(nn.Module):
    def __init__(self, dim1=32, dim2=64, dim3=128, dim4=256, dim5=512, sample_spatial=1.0, **kwargs):
        super(WaveformNet_V2, self).__init__()
        self.conv1_1 = ConvBlock(1, dim1, kernel_size=3, stride=2, padding=1) # B x 32 x 35 x35
        self.conv1_2 = ConvBlock(dim1, dim1, kernel_size=3, stride=1, padding=1) # B x 32 x 35 x35
        self.conv2_1 = ConvBlock(dim1, dim2, kernel_size=3, stride=2, padding=1) # B x 64 x 17 x 17
        self.conv2_2 = ConvBlock(dim2, dim2, kernel_size=3, stride=1, padding=1) # B x 64 x 17 x 17
        self.conv3_1 = ConvBlock(dim2, dim3, kernel_size=3, stride=2, padding=1) # B x 128 x 8 x 8
        self.conv3_2 = ConvBlock(dim3, dim3, kernel_size=3, stride=1, padding=1) # B x 128 x 8 x 8
        self.conv4_1 = ConvBlock(dim3, dim4, kernel_size=3, stride=2, padding=1) # B x 256 x 4 x 4
        self.conv4_2 = ConvBlock(dim4, dim4, kernel_size=3, stride=1, padding=1) # B x 256 x 4 x 4
        self.conv5_1 = ConvBlock(dim4, dim5, kernel_size=3, stride=2, padding=1) # B x 512 x 2 x 2
        self.conv5_2 = ConvBlock(dim5, dim5, kernel_size=3, stride=1, padding=1) # B x 512 x 2 x 2
        
        self.deconv1_1 = DeconvBlock(dim5, dim4, kernel_size=5, stride=(3,2), padding=(1, 1))
        self.deconv1_2 = ConvBlock(dim4, dim4, kernel_size=3, stride=1, padding=1)
        self.deconv2_1 = DeconvBlock(dim4, dim3, kernel_size=5, stride=(3,2), padding=(1, 1))
        self.deconv2_2 = ConvBlock(dim3, dim3, kernel_size=3, stride=1, padding=1)
        self.deconv3_1 = DeconvBlock(dim3, dim2, kernel_size=5, stride=(3,2), padding=(1, 1))
        self.deconv3_2 = ConvBlock(dim2, dim2, kernel_size=3, stride=1, padding=1)
        self.deconv4_1 = DeconvBlock(dim2, dim2, kernel_size=5, stride=(3,2), padding=(1, 1))
        self.deconv4_2 = ConvBlock(dim2, dim2, kernel_size=3, stride=1, padding=1)
        self.deconv5_1 = DeconvBlock(dim2, dim2, kernel_size=5, stride=(2,1), padding=(1, 1))
        self.deconv5_2 = ConvBlock(dim2, dim2, kernel_size=3, stride=1, padding=1)
        self.deconv6_1 = DeconvBlock(dim2, dim1, kernel_size=5, stride=(2,1), padding=(1, 1))
        self.deconv6_2 = ConvBlock(dim1, dim1, kernel_size=3, stride=1, padding=1)
        self.upsample = nn.Upsample((1000,70), mode='bilinear')
        self.deconv7_1 = ConvBlock(dim1, 5, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv5_1(x)
        x = self.conv5_2(x)
        
        x = self.deconv1_1(x)
        x = self.deconv1_2(x)
        x = self.deconv2_1(x)
        x = self.deconv2_2(x)
        x = self.deconv3_1(x)
        x = self.deconv3_2(x)
        x = self.deconv4_1(x)
        x = self.deconv4_2(x)
        x = self.deconv5_1(x)
        x = self.deconv5_2(x)
        x = self.deconv6_1(x)
        x = self.deconv6_2(x)
        x = self.upsample(x)
        x = self.deconv7_1(x)
        return x
    
################################################################
# fourier layer
################################################################

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, target_shape):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.target_shape = target_shape
        
    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=self.target_shape)
        return x

################################################################
# FNO model
################################################################

class FNO2d(nn.Module):
    def __init__(self, modes1, modes2,  width, in_dim=1, out_dim=1, setting="forward", **kwargs):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 9 # pad the domain if input is non-periodic
        self.out_dim = out_dim
        
        if setting == "forward":
            self.target_shape0 = (128 + self.padding, 70 + self.padding)
            self.target_shape1 = (256 + self.padding, 70 + self.padding)
            self.target_shape2 = (512 + self.padding, 70 + self.padding)
            self.target_shape3 = (1000 + self.padding, 70 + self.padding)
        elif setting == "inverse":
            self.target_shape0 = (512 + self.padding, 70 + self.padding)
            self.target_shape1 = (256 + self.padding, 70 + self.padding)
            self.target_shape2 = (128 + self.padding, 70 + self.padding)
            self.target_shape3 = (70 + self.padding, 70 + self.padding)
            
        
        self.fc0 = nn.Linear(in_dim+2, self.width) # +2 dimensions for grid, input channel is 3: (a(x, y), x, y)

        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2, target_shape=self.target_shape0)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2, target_shape=self.target_shape1)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2, target_shape=self.target_shape2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2, target_shape=self.target_shape3)
        
        self.upsample0 = nn.Upsample(self.target_shape0)
        self.upsample1 = nn.Upsample(self.target_shape1)
        self.upsample2 = nn.Upsample(self.target_shape2)
        self.upsample3 = nn.Upsample(self.target_shape3)
        
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, self.out_dim)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        grid = self.get_grid(x.shape, x.device)

        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        x = F.pad(x, [0,self.padding, 0,self.padding])
        
        x1 = self.conv0(x)
        x2 = self.w0(x)
        x2 = self.upsample0(x2)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x2 = self.upsample1(x2)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x2 = self.upsample2(x2)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x2 = self.upsample3(x2)
        x = x1 + x2

        x = x[..., :-self.padding, :-self.padding]
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = x.permute(0, 3, 1, 2)
        return x.contiguous()
    
    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)

################################################################
# Discriminator for Waveform GAN
################################################################
    
class Discriminator(nn.Module):
    def __init__(self, disc_in_channels=5, dim1=32, dim2=64, dim3=128, dim4=128, **kwargs):
        super(Discriminator, self).__init__()
        self.convblock1_1 = ConvBlock(disc_in_channels, dim1, stride=2)
        self.convblock1_2 = ConvBlock(dim1, dim1)
        self.convblock2_1 = ConvBlock(dim1, dim2, stride=2)
        self.convblock2_2 = ConvBlock(dim2, dim2)
        self.convblock3_1 = ConvBlock(dim2, dim3, stride=2)
        self.convblock3_2 = ConvBlock(dim3, dim3)
        self.convblock4_1 = ConvBlock(dim3, dim4, stride=2)
        self.convblock4_2 = ConvBlock(dim4, dim4)
        self.convblock5 = ConvBlock(dim4, 16, kernel_size=5, padding=0)
        
        self.fc = self.fc_layers = nn.Sequential(
                    nn.Linear(16 * 59, 128),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Linear(128, 1),
                    )
    
    def forward(self, x):
        x = self.convblock1_1(x)
        x = self.convblock1_2(x)
        x = self.convblock2_1(x)
        x = self.convblock2_2(x)
        x = self.convblock3_1(x)
        x = self.convblock3_2(x)
        x = self.convblock4_1(x)
        x = self.convblock4_2(x)
        x = self.convblock5(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x

    
model_dict = {
    'WaveformNet': WaveformNet,
    'WaveformNet_V2': WaveformNet_V2,
    'FNO': FNO2d,
    'IUnetForwardModel':IUnetForwardModel,
    'IUnetForwardModel_Legacy':IUnetForwardModel_Legacy,
    'Discriminator': Discriminator,
    'UNetForwardModel':UNetForwardModel,
    'UNetForwardModel_Legacy':UNetForwardModel_Legacy,
}