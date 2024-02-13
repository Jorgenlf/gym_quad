import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy
from abc import abstractmethod



device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class BaseEncoder(nn.Module):
    """Base class for encoder"""
    def __init__(self, 
                 latent_dim:int, 
                 image_size:int) -> None:
        super(BaseEncoder, self).__init__()
        self.name = 'base'
        self.latent_dim = latent_dim
        self.image_size = image_size
    
    def reparameterize(self, mu, log_var, eps_weight=1):
        """ Reparameterization trick from VAE paper (Kingma and Welling). 
            Eps weight in [0,1] controls the amount of noise added to the latent space."""
        # Note: log(x²) = 2log(x) -> divide by 2 to get std.dev.
        # Thus, std = exp(log(var)/2) = exp(log(std²)/2) = exp(0.5*log(var))
        std = torch.exp(0.5*log_var)
        epsilon = torch.distributions.Normal(0, eps_weight).sample(mu.shape).to(device) # ~N(0,I)
        z = mu + (epsilon * std)
        return z
    
    @abstractmethod
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        pass
    
    def save(self, path:str) -> None:
        """Saves model to path"""
        torch.save(self.state_dict(), path)
    
    def load(self, path:str) -> None:
        """Loads model from path"""
        self.load_state_dict(torch.load(path))


class ConvEncoder1(BaseEncoder):
    def __init__(self, 
                 image_size:int, 
                 channels:int, 
                 latent_dim:int,
                 activation=nn.ReLU()) -> None:
        super().__init__()

        self.name = 'conv1'
        self.channels = channels
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.activation = activation
        
        # Convolutional block
        self.conv_block = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, stride=2, padding=1),
            activation,
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            activation,
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            activation,
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            activation,
            nn.Flatten()
        )
        

        # Calculate the size of the flattened feature maps
        # Adjust the size calculations based on the number of convolution and pooling layers
        self.flattened_size = self._get_conv_output(image_size)
        
        # Fully connected layers for mu and logvar
        self.fc_mu = nn.Sequential(
            nn.Linear(self.flattened_size, latent_dim),
            activation
        )
        
        self.fc_logvar = nn.Sequential(
            nn.Linear(self.flattened_size, latent_dim),
            activation
        )

    def _get_conv_output(self, image_size:int) -> int:
        # Helper function to calculate the size of the flattened features after conv layers
        with torch.no_grad():
            input = torch.zeros(1, self.channels, image_size, image_size)
            output = self.conv_block(input)
            return int(numpy.prod(output.size()))

    def forward(self, x:torch.Tensor) -> tuple:
        x = x.to(device)
        x = self.conv_block(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        z = super().reparameterize(mu, logvar)
        return z, mu, logvar




