import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from math import ceil
from collections import OrderedDict

from iunets import iUNet
from networks.unet import UNet
from networks.autoencoder import AutoEncoder
from utils.config_utils import get_config_name, get_latent_dim


################################################################
# Invertible X-Net
################################################################

class IUnetModel(nn.Module):
    def __init__(self, amp_model, vel_model, iunet_model):
        super(IUnetModel, self).__init__()
        self.amp_model = amp_model
        self.vel_model = vel_model
        self.iunet_model = iunet_model
    
    def forward(self, x):
        x = self.vel_model.embedding(x)
        x = self.iunet_model(x)
        x = self.amp_model.decoder(x)
        return x 
        
    def inverse(self, x):
        x = self.amp_model.embedding(x)
        x = self.iunet_model.inverse(x)
        x = self.vel_model.decoder(x)
        return x
    
    
################################################################
# Joint Model Wrapper for Joint training baseline
################################################################

class JointModel(nn.Module):
    def __init__(self, forward_model, inverse_model):
        super(JointModel, self).__init__()
        self.forward_model = forward_model
        self.inverse_model = inverse_model
        
    def forward(self, x):
        x = self.forward_model(x) 
        return x
    
    def inverse(self, x):
        x = self.inverse_model(x)
        return x 

class IUnetForwardModel(nn.Module):
    def __init__(self, cfg_path="./configs/", latent_dim=70, **kwargs):
        super(IUnetForwardModel, self).__init__()
        
        amp_cfg_name = get_config_name(latent_dim, model_type="amplitude")
        self.amp_model = AutoEncoder(cfg_path, amp_cfg_name)
        
        vel_cfg_name = get_config_name(latent_dim, model_type="velocity")
        self.vel_model = AutoEncoder(cfg_path, vel_cfg_name)
        
        latent_channels = get_latent_dim(cfg_path, amp_cfg_name)
        self.iunet_model = iUNet(in_channels=latent_channels, dim=2, architecture=(4,4,4,4))
        
    def forward(self, x):
        x = self.vel_model.embedding(x)
        x = self.iunet_model(x)
        x = self.amp_model.decoder(x)
        return x 

class IUnetInverseModel(nn.Module):
    def __init__(self, cfg_path="./configs/", latent_dim=70, **kwargs):
        super(IUnetInverseModel, self).__init__()
        
        amp_cfg_name = get_config_name(latent_dim, model_type="amplitude")
        self.amp_model = AutoEncoder(cfg_path, amp_cfg_name)
        
        vel_cfg_name = get_config_name(latent_dim, model_type="velocity")
        self.vel_model = AutoEncoder(cfg_path, vel_cfg_name)
        
        latent_channels = get_latent_dim(cfg_path, amp_cfg_name)
        self.iunet_model = iUNet(in_channels=latent_channels, dim=2, architecture=(4,4,4,4))
    
    def forward(self, x):
        x = self.amp_model.embedding(x)
        x = self.iunet_model(x)
        x = self.vel_model.decoder(x)
        return x
        

class UNetInverseModel(nn.Module):
    def __init__(self, 
                 cfg_path="./configs/", 
                 latent_dim=70, 
                 unet_depth = 3,
                 unet_repeat_blocks = 2,
                 skip=True,
                 **kwargs):
        super(UNetInverseModel, self).__init__()
#         UNet(in_channels=128, d=2, repeat=2) # Number of params: 34M
#         UNet(in_channels=128, d=3, repeat=4) # Number of params: 278M

        amp_cfg_name = get_config_name(latent_dim, model_type="amplitude")
        self.amp_model = AutoEncoder(cfg_path, amp_cfg_name)
        
        vel_cfg_name = get_config_name(latent_dim, model_type="velocity")
        self.vel_model = AutoEncoder(cfg_path, vel_cfg_name)
        
        latent_channels = get_latent_dim(cfg_path, amp_cfg_name)
        depth = unet_depth
        self.unet_model = UNet(in_channels=latent_channels, d=depth, repeat=unet_repeat_blocks, skip=skip)

    def forward(self, x):
        x = self.amp_model.embedding(x)
        x = self.unet_model(x)
        x = self.vel_model.decoder(x)
        return x

class UNetForwardModel(nn.Module):
    def __init__(self, 
                 cfg_path="./configs/", 
                 latent_dim=70, 
                 unet_depth = 3,
                 unet_repeat_blocks = 2,
                 skip=True,
                 **kwargs):
        super(UNetForwardModel, self).__init__()
        
        amp_cfg_name = get_config_name(latent_dim, model_type="amplitude")
        self.amp_model = AutoEncoder(cfg_path, amp_cfg_name)
        
        vel_cfg_name = get_config_name(latent_dim, model_type="velocity")
        self.vel_model = AutoEncoder(cfg_path, vel_cfg_name)
        
        latent_channels = get_latent_dim(cfg_path, amp_cfg_name)
        depth = unet_depth
        self.unet_model = UNet(in_channels=latent_channels, d=depth, repeat=unet_repeat_blocks, skip=skip)
        
    def forward(self, x):
        x = self.vel_model.embedding(x)
        x = self.unet_model(x)
        x = self.amp_model.decoder(x)
        return x 
    
# Decoupled IUNet Model to understand Latent Bijectivity
class Decouple_IUnetModel(nn.Module):
    def __init__(self, amp_model, vel_model, amp_iunet_model, vel_iunet_model):
        super(Decouple_IUnetModel, self).__init__()
        self.amp_model = amp_model
        self.vel_model = vel_model
        self.amp_iunet_model = amp_iunet_model
        self.vel_iunet_model = vel_iunet_model
    
    def forward(self, x):
        x = self.vel_model.embedding(x)
        x = self.amp_iunet_model(x)
        x = self.amp_model.decoder(x)
        return x 
        
    def inverse(self, x):
        x = self.amp_model.embedding(x)
        x = self.vel_iunet_model(x)
        x = self.vel_model.decoder(x)
        return x
    
############################################################################################

################################################################
# Legacy Codes
################################################################

################################################################
# Amplitude Encoder Decoder
################################################################

class AmpAutoEncoder(nn.Module):
    def __init__(self, input_channel, encoder_channels, decoder_channels):
        super(AmpAutoEncoder, self).__init__()
        
        encoder_layers = nn.ModuleList()
        decoder_layers = nn.ModuleList()
        
        enlayer1 = nn.Sequential(nn.Conv2d(input_channel, encoder_channels[0],
                                                kernel_size=(7,1), stride=(3,1), padding=(3,0)),
                                     nn.BatchNorm2d(encoder_channels[0]),
                                     nn.LeakyReLU(0.2, inplace=True))
        encoder_layers.append(enlayer1)
        
        enlayer2 = nn.Sequential(nn.Conv2d(encoder_channels[0], encoder_channels[1],
                                                kernel_size=(7,1), stride=(2,1), padding=(1,0)),
                                     nn.BatchNorm2d(encoder_channels[1]),
                                     nn.LeakyReLU(0.2, inplace=True))
        
        encoder_layers.append(enlayer2)
        
        enlayer3 = nn.Sequential(nn.Conv2d(encoder_channels[1], encoder_channels[2],
                                                kernel_size=(5,1), stride=(2,1), padding=(1,0)),
                                     nn.BatchNorm2d(encoder_channels[2]),
                                     nn.LeakyReLU(0.2, inplace=True))
        
        encoder_layers.append(enlayer3)
        
        enlayer4 = nn.Sequential(nn.Conv2d(encoder_channels[2], encoder_channels[3],
                                                kernel_size=(5,1), stride=(1,1), padding=(0,0)),
                                     nn.BatchNorm2d(encoder_channels[3]),
                                     nn.LeakyReLU(0.2, inplace=True))
        encoder_layers.append(enlayer4)
        
        enlayer5 = nn.Sequential(nn.Conv2d(encoder_channels[3], encoder_channels[4],
                                                kernel_size=(5,1), stride=(1,1), padding=(0,0)),
                                     nn.BatchNorm2d(encoder_channels[4]),
                                     nn.LeakyReLU(0.2, inplace=True))
        encoder_layers.append(enlayer5)
        
        enlayer6 = nn.Upsample((70,70), mode='bilinear')
        
        encoder_layers.append(enlayer6)
        
        delayer1 = nn.Sequential(nn.ConvTranspose2d(encoder_channels[-1], decoder_channels[0],
                                                kernel_size=(7,1), stride=(3,1), padding=(0,0)),
                             nn.BatchNorm2d(decoder_channels[0]),
                             nn.Tanh())
        
        decoder_layers.append(delayer1)
        delayer2 = nn.Sequential(nn.ConvTranspose2d(decoder_channels[0], decoder_channels[1],
                                                kernel_size=(7,1), stride=(2,1), padding=(0,0)),
                             nn.BatchNorm2d(decoder_channels[1]),
                             nn.Tanh())
        decoder_layers.append(delayer2)
        delayer3 = nn.Sequential(nn.ConvTranspose2d(decoder_channels[1], decoder_channels[2],
                                                kernel_size=(5,1), stride=(2,1), padding=(0,0)),
                             nn.BatchNorm2d(decoder_channels[2]),
                             nn.Tanh())
        decoder_layers.append(delayer3)
        delayer4 = nn.Sequential(nn.ConvTranspose2d(decoder_channels[2], decoder_channels[3],
                                                kernel_size=(5,1), stride=(1,1), padding=(0,0)),
                                nn.BatchNorm2d(decoder_channels[3]),
                                nn.Tanh())
        
        decoder_layers.append(delayer4)
        delayer5 = nn.Sequential(nn.ConvTranspose2d(decoder_channels[3], decoder_channels[4],
                                                kernel_size=(5,1), stride=(1,1), padding=(0,0)),
                                nn.BatchNorm2d(decoder_channels[4]),
                                nn.Tanh(),
                                nn.Conv2d(decoder_channels[4], decoder_channels[4], kernel_size=(7,3), padding=(3, 1))          
                                )
        decoder_layers.append(delayer5)
        delayer6 = nn.Upsample((1000,70))
        decoder_layers.append(delayer6)
        
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
# Velocity Encoder Decoder
################################################################

class VelAutoEncoder(nn.Module):
    def __init__(self, input_channel, encoder_channels, decoder_channels):
        super(VelAutoEncoder, self).__init__()
        
        encoder_layers = nn.ModuleList()
        decoder_layers = nn.ModuleList()
        
        enlayer1 = nn.Sequential(nn.Conv2d(input_channel, encoder_channels[0],
                                                kernel_size=1, stride=1, padding=0, 
                                                padding_mode='reflect'),
                                     nn.BatchNorm2d(encoder_channels[0]),
                                     nn.LeakyReLU(0.2, inplace=True))
        encoder_layers.append(enlayer1)
        
        enlayer2 = nn.Sequential(nn.Conv2d(encoder_channels[0], encoder_channels[1],
                                                kernel_size=1, stride=1, padding=0,
                                                padding_mode='reflect'),
                                     nn.BatchNorm2d(encoder_channels[1]),
                                     nn.LeakyReLU(0.2, inplace=True))
        
        encoder_layers.append(enlayer2)
        
        enlayer3 = nn.Sequential(nn.Conv2d( encoder_channels[1], encoder_channels[2],
                                                kernel_size=1, stride=1, padding=0,
                                                padding_mode='reflect'),
                                     nn.BatchNorm2d(encoder_channels[2]),
                                     nn.LeakyReLU(0.2, inplace=True))
        
        encoder_layers.append(enlayer3)
        
        enlayer4 = nn.Sequential(nn.Conv2d(encoder_channels[2], encoder_channels[3],
                                                kernel_size=1, stride=1, padding=0,
                                                padding_mode='reflect'),
                                     nn.BatchNorm2d(encoder_channels[3]),
                                     nn.LeakyReLU(0.2, inplace=True))
        
        encoder_layers.append(enlayer4)
        
        enlayer5 = nn.Sequential(nn.Conv2d(encoder_channels[3], encoder_channels[4],
                                                kernel_size=1, stride=1, padding=0,
                                                padding_mode='reflect'),
                                     nn.BatchNorm2d(encoder_channels[4]),
                                     nn.LeakyReLU(0.2, inplace=True))
        
        encoder_layers.append(enlayer5)
        
        delayer1 = nn.Sequential(nn.ConvTranspose2d(encoder_channels[-1], decoder_channels[0],
                                                kernel_size=1, stride=1, padding=0,padding_mode='zeros'),
                             nn.BatchNorm2d(decoder_channels[0]),
                             nn.Tanh())
        
        decoder_layers.append(delayer1)
        
        delayer2 = nn.Sequential(nn.ConvTranspose2d(decoder_channels[0], decoder_channels[1],
                                                kernel_size=1, stride=1, padding=0,padding_mode='zeros'),
                             nn.BatchNorm2d(decoder_channels[1]),
                             nn.Tanh())
        
        decoder_layers.append(delayer2)
        
        delayer3 = nn.Sequential(nn.ConvTranspose2d(decoder_channels[1], decoder_channels[2],
                                                kernel_size=1, stride=1, padding=0,padding_mode='zeros'),
                             nn.BatchNorm2d(decoder_channels[2]),
                             nn.Tanh())
        
        decoder_layers.append(delayer3)
        
        delayer4 = nn.Sequential(nn.ConvTranspose2d(decoder_channels[2], decoder_channels[3],
                                                kernel_size=1, stride=1, padding=0,padding_mode='zeros'),
                             nn.BatchNorm2d(decoder_channels[3]),
                             nn.Tanh())
        
        decoder_layers.append(delayer4)
        
        delayer5 = nn.Sequential(nn.ConvTranspose2d(decoder_channels[3], decoder_channels[4],
                                                kernel_size=1, stride=1, padding=0,padding_mode='zeros'),
                             nn.BatchNorm2d(decoder_channels[4]),
                             nn.Tanh(),
                             nn.Conv2d(decoder_channels[4], decoder_channels[4], kernel_size=1, stride=1)
                             )
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

    
class IUnetForwardModel_Legacy(nn.Module):
    def __init__(self, vel_input_channel, vel_encoder_channel, vel_decoder_channel, amp_input_channel, amp_encoder_channel, amp_decoder_channel, **kwargs):
        super(IUnetForwardModel_Legacy, self).__init__()
        self.iunet_model = iUNet(in_channels=128, dim=2, architecture=(4,4,4,4))
        self.amp_model = AmpAutoEncoder(amp_input_channel, amp_encoder_channel, amp_decoder_channel)
        self.vel_model = VelAutoEncoder(vel_input_channel, vel_encoder_channel, vel_decoder_channel)
        
    def forward(self, x):
        x = self.vel_model.embedding(x)
        x = self.iunet_model(x)
        x = self.amp_model.decoder(x)
        return x 

class IUnetInverseModel_Legacy(nn.Module):
    def __init__(self, amp_input_channel, amp_encoder_channel, amp_decoder_channel, vel_input_channel, vel_encoder_channel, vel_decoder_channel, **kwargs):
        super(IUnetInverseModel_Legacy, self).__init__()
        self.iunet_model = iUNet(in_channels=128, dim=2, architecture=(4,4,4,4))
        self.amp_model = AmpAutoEncoder(amp_input_channel, amp_encoder_channel, amp_decoder_channel)
        self.vel_model = VelAutoEncoder(vel_input_channel, vel_encoder_channel, vel_decoder_channel)
        
    def forward(self, x):
        x = self.amp_model.embedding(x)
        x = self.iunet_model(x)
        x = self.vel_model.decoder(x)
        return x
    
    
class UNetInverseModel_Legacy(nn.Module):
    def __init__(self, 
                 amp_input_channel, 
                 amp_encoder_channel, 
                 amp_decoder_channel, 
                 vel_input_channel, 
                 vel_encoder_channel, 
                 vel_decoder_channel, 
                 unet_depth = 3,
                 unet_repeat_blocks = 2,
                 **kwargs):
        super(UNetInverseModel_Legacy, self).__init__()
        depth = unet_depth
        self.unet_model = UNet(in_channels=128, d=depth, repeat=unet_repeat_blocks)
        self.amp_model = AmpAutoEncoder(amp_input_channel, amp_encoder_channel, amp_decoder_channel)
        self.vel_model = VelAutoEncoder(vel_input_channel, vel_encoder_channel, vel_decoder_channel)
        
    def forward(self, x):
        x = self.amp_model.embedding(x)
        x = self.unet_model(x)
        x = self.vel_model.decoder(x)
        return x

class UNetForwardModel_Legacy(nn.Module):
    def __init__(self, 
                 vel_input_channel, 
                 vel_encoder_channel, 
                 vel_decoder_channel, 
                 amp_input_channel, 
                 amp_encoder_channel, 
                 amp_decoder_channel, 
                 unet_depth = 3,
                 unet_repeat_blocks = 2,
                 **kwargs):
        super(UNetForwardModel_Legacy, self).__init__()
        depth = unet_depth
        self.unet_model = UNet(in_channels=128, d=depth, repeat=unet_repeat_blocks)
        self.amp_model = AmpAutoEncoder(amp_input_channel, amp_encoder_channel, amp_decoder_channel)
        self.vel_model = VelAutoEncoder(vel_input_channel, vel_encoder_channel, vel_decoder_channel)
        
    def forward(self, x):
        x = self.vel_model.embedding(x)
        x = self.unet_model(x)
        x = self.amp_model.decoder(x)
        return x 

    

model_dict = {
    'IUnet': IUnetModel,
    'JoinetModel': JointModel,
    'Decouple_IUnet':Decouple_IUnetModel
}
