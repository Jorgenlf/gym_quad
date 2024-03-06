import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
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
        super().__init__(latent_dim=latent_dim, image_size=image_size)

        self.name = 'conv1'
        self.channels = channels
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.activation = activation
        #"""
        # Convolutional block type 1
        self.conv_block = nn.Sequential(
            nn.Conv2d(self.channels, 32, kernel_size=3, stride=2, padding=1),
            self.activation,
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            self.activation,
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            self.activation,
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            self.activation,
        )
        """
        # Convolutional block type 2
        self.conv_block = nn.Sequential(
            nn.Conv2d(self.channels, 32, kernel_size=5, stride=1, padding=1),
            self.activation,
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            self.activation,
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            self.activation,
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            self.activation,
        )
        #"""

        self.flatten = nn.Flatten()
        

        # Calculate the size of the flattened feature maps
        # Adjust the size calculations based on the number of convolution and pooling layers
        self.flattened_size, self.dim_before_flatten = self._get_conv_output(image_size)
        print(f'Encoder flattened size: {self.flattened_size}; Dim before flatten: {self.dim_before_flatten}')
        
        """
        Typically, the layers that output the mean (μ) and log variance (log(σ²)) of the latent space
        distribution do not include an activation function. This is because these outputs directly 
        parameterize the latent space distribution, and constraining them with an activation function 
        (like ReLU) could limit the expressiveness of the latent representation.
        """
        
        # Fully connected layers for mu and logvar
        self.fc_mu = nn.Sequential(
            #nn.Linear(self.flattened_size, latent_dim),
            nn.Linear(self.flattened_size, 512),
            self.activation,
            nn.Linear(512, latent_dim)
            #self.activation
        )
        
        self.fc_logvar = nn.Sequential(
            #nn.Linear(self.flattened_size, latent_dim),
            nn.Linear(self.flattened_size, 512),
            self.activation,
            nn.Linear(512, latent_dim)
            #self.activation
        )

    def _get_conv_output(self, image_size:int) -> int:
        # Helper function to calculate size of the flattened feature maps as well as before the flatten layer
        # Returns the size of the flattened feature maps and the output of the conv block before the flatten layer
        with torch.no_grad():
            input = torch.zeros(1, self.channels, image_size, image_size)
            output1 = self.flatten(self.conv_block(input))
            output2 = self.conv_block(input)
            return int(numpy.prod(output1.size())), output2.size()

    def forward(self, x:torch.Tensor) -> tuple:
        x = x.to(device)
        x = self.conv_block(x)
        x = self.flatten(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        z = super().reparameterize(mu, logvar)
        return z, mu, logvar



class VGG16Encoder(BaseEncoder):
    def __init__(self, 
                 latent_dim:int, 
                 image_size:int,
                 activation=nn.ReLU()) -> None:
        super().__init__(latent_dim=latent_dim, image_size=image_size)
        self.name = 'vgg16'
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.activation = activation

        self.vgg16 = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1)
        # Freeze early layers
        for param in self.vgg16.features.parameters():
            param.requires_grad = False

        n_inputs = self.vgg16.classifier[0].in_features # 25088, num of inputs to the last fully connected layer
        
        # Change final fully connected layer to output the latent space
        self.vgg16.classifier = nn.Sequential(
            nn.Linear(n_inputs, 4096),
            self.activation,
            nn.Linear(4096, 512),
            self.activation,
            nn.Linear(512, 2*self.latent_dim),
        )

        print(self.vgg16)

        
    
    def forward(self, x:torch.Tensor) -> tuple:
        x = x.to(device)
        # Make 1 channeled input into 3 channeled input
        if x.shape[1] == 1:
            x = torch.cat((x,x,x), dim=1)
        x = self.vgg16(x)
        x = x.view(x.size(0), -1) # essentially Flatten layer the tensor
        # First self.latent_dim elements are mu, last self.latent_dim elements are logvar
        mu = x[:, :self.latent_dim]
        logvar = x[:, self.latent_dim:]
        z = super().reparameterize(mu, logvar)
        return z, mu, logvar
    

class ResNet50Encoder(BaseEncoder):
    def __init__(self, 
                 latent_dim:int, 
                 image_size:int,
                 activation=nn.ReLU()) -> None:
        super().__init__(latent_dim=latent_dim, image_size=image_size)
        self.name = 'resnet50'
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.activation = activation

        self.resnet50 = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
        # Freeze early layers
        for param in self.resnet50.parameters():
            param.requires_grad = False

        n_inputs = self.resnet50.fc.in_features # 2048, num of inputs to the last fully connected layer
        
        # Change final fully connected layer to output the latent space
        self.resnet50.fc = nn.Sequential(
            nn.Linear(n_inputs, 512),
            self.activation,
            nn.Linear(512, 2*self.latent_dim),
        )

        print(self.resnet50)

        
    
    def forward(self, x:torch.Tensor) -> tuple:
        x = x.to(device)
        # Make 1 channeled input into 3 channeled input
        if x.shape[1] == 1:
            x = torch.cat((x,x,x), dim=1)
        x = self.resnet50(x)
        x = x.view(x.size(0), -1) # essentially Flatten layer the tensor
        # First self.latent_dim elements are mu, last self.latent_dim elements are logvar
        mu = x[:, :self.latent_dim]
        logvar = x[:, self.latent_dim:]
        z = super().reparameterize(mu, logvar)
        return z, mu, logvar