import torch
import torch.nn as nn
import os
import numpy as np
from abc import abstractmethod
from VAE.residual import ResidualBlockTransposed, ResBlock, PositionalNorm

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class BaseDecoder(nn.Module):
    """Base class for decoder"""
    def __init__(self, 
                 latent_dim:int, 
                 image_size:int) -> None:
        super(BaseDecoder, self).__init__()
        self.latent_dims = latent_dim
        self.image_size = image_size

    @abstractmethod
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        pass

    def save(self, path:str) -> None:
        """Saves model to path"""
        torch.save(self.state_dict(), path)
    
    def load(self, path:str) -> None:
        """Loads model from path"""
        self.load_state_dict(torch.load(path))


class ConvDecoder1(BaseDecoder):
    def __init__(self,
                 image_size:int,
                 channels:int,
                 latent_dim:int,
                 flattened_size:int,
                 dim_before_flatten:torch.Size,
                 activation=nn.ReLU()) -> None:
        super().__init__(latent_dim=latent_dim, image_size=image_size)
        self.name = 'conv1'
        self.channels = channels
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.activation = activation
        self.flattened_size = flattened_size
        self.dim_before_flatten = dim_before_flatten

        # Fully connected layer
        self.fc = nn.Sequential(
            #nn.Linear(latent_dim, self.flattened_size), 
            #activation
            nn.Linear(latent_dim, 512),
            activation,
            nn.Linear(512, self.flattened_size),

        )
        #"""
        # Transposed convolutional block type 1
        self.t_conv_block = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            activation,
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            activation,
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            activation,
            nn.ConvTranspose2d(32, self.channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        )
        """
        # Transposed convolutional block type 2
        self.t_conv_block = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            self.activation,
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            self.activation,
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            self.activation,
            nn.ConvTranspose2d(32, self.channels, kernel_size=5, stride=1, padding=2),
        )
        #"""
        self.sigmoid = nn.Sigmoid()

    def forward(self, z:torch.Tensor) -> torch.Tensor:
        z = self.fc(z)
        z = z.view(-1, *self.dim_before_flatten[1:]) # Reshape the tensor to match the shape of the last convolutional layer in the encoder
        z_tconv = self.t_conv_block(z)
        x_hat = self.sigmoid(z_tconv)
        return x_hat
    
class ConvDecoder2(BaseDecoder):
    """Normal convolutional decoder for the resnet style encoder (slightly different params in the tconv block for correct shaping)"""
    def __init__(self,
                 image_size:int,
                 channels:int,
                 latent_dim:int,
                 flattened_size:int,
                 dim_before_flatten:torch.Size,
                 activation=nn.ReLU()) -> None:
        super().__init__(latent_dim=latent_dim, image_size=image_size)
        self.name = 'conv2'
        self.channels = channels
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.activation = activation
        self.flattened_size = flattened_size
        self.dim_before_flatten = dim_before_flatten

        # Fully connected layer
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 512),
            activation,
            nn.Linear(512, self.flattened_size),

        )

        # Transposed convolutional block 
        self.t_conv_block = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            activation,
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            activation,
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            activation,
            nn.ConvTranspose2d(32, self.channels, kernel_size=3, stride=1, padding=1, output_padding=0)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, z:torch.Tensor) -> torch.Tensor:
        z = self.fc(z)
        z = z.view(-1, *self.dim_before_flatten[1:]) # Reshape the tensor to match the shape of the last convolutional layer in the encoder
        z_tconv = self.t_conv_block(z)
        x_hat = self.sigmoid(z_tconv)
        return x_hat


class _ConvDecoder2(BaseDecoder):
    def __init__(self,
                 image_size:int,
                 channels:int,
                 latent_dim:int,
                 flattened_size:int,
                 dim_before_flatten:torch.Size,
                 activation=nn.ReLU()) -> None:
        super().__init__(latent_dim=latent_dim, image_size=image_size)
        self.name = 'conv2'
        self.channels = channels
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.activation = activation
        self.flattened_size = flattened_size
        self.dim_before_flatten = dim_before_flatten

        # The decoder architecture follows the design of a reverse ResNet
        # stacking several residual blocks into groups, operating on different
        # scales of the image. The first residual block from each group is
        # responsible for up-sizing the image and reducing the channels.
        self.net = nn.Sequential(
            # Inverse head.
            nn.Linear(latent_dim, 256 * 28 * 28),
            nn.Unflatten(dim=-1, unflattened_size=(256, 28, 28)),     

            # Body.
            ResBlock(in_chan=256, out_chan=128, scale="upscale"),   
            ResBlock(in_chan=128, out_chan=128),
            #ResBlock(in_chan=128, out_chan=128),
            #ResBlock(in_chan=128, out_chan=128),

            ResBlock(in_chan=128, out_chan=64, scale="upscale"),    
            ResBlock(in_chan=64, out_chan=64),
            #ResBlock(in_chan=64, out_chan=64),
            #ResBlock(in_chan=64, out_chan=64),

            ResBlock(in_chan=64, out_chan=32, scale="upscale"),  
            ResBlock(in_chan=32, out_chan=32),
            #ResBlock(in_chan=32, out_chan=32),
            #ResBlock(in_chan=32, out_chan=32),

            # Inverse stem.
            # Inverse stem.
            PositionalNorm(32),
            nn.ReLU(),
            nn.Conv2d(32, out_channels=channels, kernel_size=3, padding="same"),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, z:torch.Tensor) -> torch.Tensor:
        #z = z.view(-1, *self.dim_before_flatten[1:]) # Reshape the tensor to match the shape of the last convolutional layer in the encoder
        z_tconv = self.net(z)
        x_hat = self.sigmoid(z_tconv)
        return x_hat

'''
class ConvDecoder2(BaseDecoder):
    """ResNet style decoder"""
    def __init__(self,
                 image_size:int,
                 channels:int,
                 latent_dim:int,
                 flattened_size:int,
                 dim_before_flatten:torch.Size,
                 activation=nn.ReLU()) -> None:
        super().__init__(latent_dim=latent_dim, image_size=image_size)
        self.name = 'conv2'
        self.channels = channels
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.activation = activation
        self.flattened_size = flattened_size
        self.dim_before_flatten = dim_before_flatten
        self.in_channels_decoder = 64
        
        # Fully connected layer
        self.fc = nn.Sequential(
            #nn.Linear(latent_dim, self.flattened_size), 
            #activation
            nn.Linear(latent_dim, 512),
            activation,
            nn.Linear(512, self.flattened_size),
        )
        
        block = ResidualBlockTransposed 
        self.trans_layer1 = self.make_trans_layer(block,32,2)
        self.trans_layer2 = self.make_trans_layer(block,16,2,2)
        self.trans_layer3 = self.make_trans_layer(block,3,2,2)
        self.trans_layer4 = nn.ConvTranspose2d(3,1,4,2,2)
        self.sigmoid = nn.Sigmoid()
        
        print(self.trans_layer1)
        print(self.trans_layer2)
        print(self.trans_layer3)
    
 
    def make_trans_layer(self, decoder_block, out_channels, blocks, stride=1,padding=0):
        upsample = None
        if (stride != 1) or (self.in_channels_decoder != out_channels):
            upsample = nn.Sequential(
                nn.ConvTranspose2d(self.in_channels_decoder, out_channels, kernel_size=3, stride=stride,padding=padding),
                nn.BatchNorm2d(out_channels)
            )
        layers_dec = []
        layers_dec.append(decoder_block(self.in_channels_decoder, out_channels, stride,padding, upsample))
        self.in_channels_decoder = out_channels
        for i in range(1, blocks):
            layers_dec.append(decoder_block(out_channels, out_channels))
#         print(*layers_dec)
        return nn.Sequential(*layers_dec)
    
    def forward(self, z:torch.Tensor) -> torch.Tensor:
        z = self.fc(z)
        print('z-shape:', z.size())
        #x=sample.view(sample.shape[0],32,1,1)
        #z = z.view(z.shape[0],32,1,1)
        #z = z.view(-1, *self.dim_before_flatten[1:]) # Reshape the tensor to match the shape of the last convolutional layer in the encoder
        z = z.view(-1, *self.dim_before_flatten[1:])
        print('z-shape:', z.size())
        z = self.trans_layer1(z)
        print('z-shape:', z.size())
        z = self.trans_layer2(z)
        z = self.trans_layer3(z)
        z = self.trans_layer4(z)
        x_hat = self.sigmoid(z)
        return x_hat'''