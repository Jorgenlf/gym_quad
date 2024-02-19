import torch
import torch.nn as nn
import os
import numpy as np
from abc import abstractmethod

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
                 activation=nn.ReLU()) -> None:
        super().__init__(latent_dim=latent_dim, image_size=image_size)
        self.name = 'conv1'
        self.channels = channels
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.activation = activation
        self.flattened_size = flattened_size

        # Fully connected layer
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, self.flattened_size), 
            activation
        )

        # Transposed convolutional block 
        self.t_conv_block = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            activation,
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            activation,
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            activation,
            nn.ConvTranspose2d(32, channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, z:torch.Tensor) -> torch.Tensor:
        z = self.fc(z)
        z = z.view(-1, 256, 14, 14) # Reshape the tensor to match the shape of the last convolutional layer in the encoder
        z_tconv = self.t_conv_block(z)
        x_hat = self.sigmoid(z_tconv)
        return x_hat