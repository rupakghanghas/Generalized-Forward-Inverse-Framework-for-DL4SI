import os
import json
import torch
import torch.nn as nn

class ConvBlockLegacy(nn.Module):
    """(convolution => [BatchNorm] => LeakyReLU) * 2"""
    def __init__(self, 
                 in_channels, 
                 out_channels,
                 activation_fn=nn.LeakyReLU,
                 activation_params={},
                 kernel_size=(1, 1), 
                 stride=(1, 1), 
                 padding=(0, 0),
                 padding_mode='zeros',
                 transpose_conv=False,
                ):
        super().__init__()
        conv_fn = nn.ConvTranspose2d if transpose_conv else nn.Conv2d
        self.conv_block = nn.Sequential(
            conv_fn(in_channels, out_channels, kernel_size=kernel_size, 
                    stride=stride, padding=padding, padding_mode=padding_mode),
            nn.BatchNorm2d(out_channels),
            activation_fn(**activation_params),
        )
    def forward(self, x):
        return self.conv_block(x)

#for future experiments with group norm
class ConvBlock(nn.Module):
    """(convolution => [GroupNorm] => LeakyReLU) * 2"""
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size=(1, 1), 
                 stride=(1, 1), 
                 padding=(0, 0),
                 negative_slope=0.01,
                 groups=1,
                 transpose_conv=False,
                ):
        super().__init__()
        conv_fn = nn.ConvTranspose2d if transpose_conv else nn.Conv2d
        self.conv_block = nn.Sequential(
            conv_fn(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.GroupNorm(groups, out_channels, 1e-3),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
        )
    def forward(self, x):
#         print("input shape:", x.shape)
        return self.conv_block(x)

class AutoEncoder(nn.Module):
    def __init__(self, cfg_path="./configs/", cfg_name='amplitude_config_latent_dim_70.json'):
        super(AutoEncoder, self).__init__()
        
        config_file = os.path.join(cfg_path, cfg_name)
        with open(config_file, 'r') as file:
            cfg = json.load(file)
        
        encoder_layers = nn.ModuleList()
        decoder_layers = nn.ModuleList()
        
        for i, key in enumerate(cfg["encoder_blocks"].keys()):
            enc_dict = cfg["encoder_blocks"][key]
            activation_fn = eval(enc_dict["activation"]["activation_fn"])
            activation_params = enc_dict["activation"]["activation_params"]
            in_channels = cfg["in_channels"] if i==0 else cfg["encoder_blocks"][str(i-1)]["out_channels"]
            out_channels = enc_dict["out_channels"]
            conv_block = ConvBlockLegacy(
                             in_channels=in_channels, 
                             out_channels=out_channels,
                             activation_fn=activation_fn,
                             activation_params=activation_params,
                             kernel_size=enc_dict["kernel_size"], 
                             stride=enc_dict["stride"], 
                             padding=enc_dict["padding"],
                             padding_mode=enc_dict["padding_mode"],
                             transpose_conv=False,          
                    )
            encoder_layers.append(conv_block)
        encoder_layers.append(nn.Upsample(cfg["latent_dim"], mode='bilinear'))
        
        for i, key in enumerate(cfg["decoder_blocks"].keys()):
            dec_dict = cfg["decoder_blocks"][key]
            activation_fn = eval(dec_dict["activation"]["activation_fn"])
            activation_params = dec_dict["activation"]["activation_params"]
            in_channels = out_channels if i==0 else cfg["decoder_blocks"][str(i-1)]["out_channels"]
            out_channels = dec_dict["out_channels"]
            conv_block = ConvBlockLegacy(
                             in_channels=in_channels, 
                             out_channels=out_channels,
                             activation_fn=activation_fn,
                             activation_params=activation_params,
                             kernel_size=dec_dict["kernel_size"], 
                             stride=dec_dict["stride"], 
                             padding=dec_dict["padding"],
                             padding_mode=dec_dict["padding_mode"],
                             transpose_conv=True,          
                    )
            decoder_layers.append(conv_block)
        
        #last convolutional block after decoder
        last_conv = nn.Conv2d(
                             in_channels=out_channels, 
                             out_channels=out_channels,
                             kernel_size=cfg["last_conv2d"]["kernel_size"], 
                             stride=cfg["last_conv2d"]["stride"], 
                             padding=cfg["last_conv2d"]["padding"],        
                    )
        decoder_layers.append(last_conv)
        decoder_layers.append(nn.Upsample(cfg["in_dim"], mode='bilinear'))
        
        self.encoder_layers = nn.Sequential(*encoder_layers)
        self.decoder_layers = nn.Sequential(*decoder_layers)
    
    def forward(self, x):
        x = self.encoder_layers(x)
        x = self.decoder_layers(x)
        return x
    
    def embedding(self, x):
#         print("input shape:", x.shape)
        x = self.encoder_layers(x)
        return x
    
    def decoder(self, x):
        x = self.decoder_layers(x)
        return x