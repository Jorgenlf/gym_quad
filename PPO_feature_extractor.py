import gymnasium as gym
import torch as th
import torch.nn as nn

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import matplotlib.pyplot as plt
import numpy as np


class EncoderFeatureExtractor(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space) of dimension (N_sensors x 1)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of latent dimensions
    """
    def __init__(self, 
                 observation_space: gym.spaces.Box, 
                 image_size:int, 
                 channels:int, 
                 device:str,
                 latent_dim:int,
                 activation=nn.ReLU()):
        super(EncoderFeatureExtractor, self).__init__(observation_space, features_dim=latent_dim)

        self.name = 'conv1'
        self.device = th.device(device)
        self.channels = channels
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.activation = activation

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

        self.flatten = nn.Flatten()

        # Calculate the size of the flattened feature maps
        # Adjust the size calculations based on the number of convolution and pooling layers
        self.flattened_size, self.dim_before_flatten = self._get_conv_output(image_size)
        # print(f'Encoder flattened size: {self.flattened_size}; Dim before flatten: {self.dim_before_flatten}')
        
        # Fully connected layers for mu and logvar
        self.fc_mu = nn.Sequential(
            nn.Linear(self.flattened_size, 512),
            self.activation,
            nn.Linear(512, latent_dim)
        )
        
        self.fc_logvar = nn.Sequential(
            nn.Linear(self.flattened_size, 512),
            self.activation,
            nn.Linear(512, latent_dim)
        )

    def _get_conv_output(self, image_size:int) -> int:
        # Helper function to calculate size of the flattened feature maps as well as before the flatten layer
        # Returns the size of the flattened feature maps and the output of the conv block before the flatten layer
        with th.no_grad():
            input = th.zeros(1, self.channels, image_size, image_size)
            output1 = self.flatten(self.conv_block(input))
            output2 = self.conv_block(input)
            return int(np.prod(output1.size())), output2.size()
    
    def reparameterize(self, mu, log_var, eps_weight=1):
        """ Reparameterization trick from VAE paper (Kingma and Welling). 
            Eps weight in [0,1] controls the amount of noise added to the latent space."""
        # Note: log(x²) = 2log(x) -> divide by 2 to get std.dev.
        # Thus, std = exp(log(var)/2) = exp(log(std²)/2) = exp(0.5*log(var))
        std = th.exp(0.5*log_var)
        epsilon = th.distributions.Normal(0, eps_weight).sample(mu.shape).to(self.device) # ~N(0,I)
        z = mu + (epsilon * std)
        return z

    def forward(self, x:th.Tensor) -> tuple:
        x = x.to(self.device)
        x = self.conv_block(x)
        x = self.flatten(x)
        mu = self.fc_mu(x)

        # We pass through mu during feature extraction - reparameterization is done in VAE training only

        #logvar = self.fc_logvar(x)
        #z = self.reparameterize(mu, logvar)
        #return z, mu, logvar
        # print(mu)
        return mu
    
    def get_features(self, observations:th.Tensor) -> list:
        feat = []
        out = observations
        for layer in self.conv_block:
            out = layer(out)
            if not isinstance(layer, nn.ReLU):
                feat.append(out.detach().cpu().numpy())
        
        # Usikker på om følgende to linjer trengs siden flatten ikek endrer annet enn formen
        out = self.flatten(out)
        feat.append(out.detach().cpu().numpy())
        
        for layer in self.fc_mu:
            out = layer(out)
            if not isinstance(layer, nn.ReLU):
                feat.append(out.detach().cpu().numpy())

        return feat

    def get_activations(self, observations: th.Tensor) -> list:
        feat = []
        out = observations
        for layer in self.conv_block:
            out = layer(out)
            if isinstance(layer, nn.ReLU):
                feat.append(out.detach().cpu().numpy())
        
        # Tror ikke flatten laget trengs her siden ingen relu
        out = self.flatten(out)
        feat.append(out.detach().cpu().numpy())

        for layer in self.fc_mu:
            out = layer(out)
            if isinstance(layer, nn.ReLU):
                feat.append(out.detach().cpu().numpy())

        return feat
    
    def load_params(self, path:str) -> None:
        params = th.load(path)
        self.load_state_dict(params)

    def lock_params(self) -> None:
        for param in self.parameters():
            param.requires_grad = False
        
class IMU_NN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 6):
        super(IMU_NN, self).__init__(observation_space, features_dim=features_dim)

        self.passthrough = nn.Identity()

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # shape = observations.shape #Dont need this as the imu is already a 1D tensor
        # observations = observations[:,0,:].reshape(shape[0], shape[-1])
        return self.passthrough(observations)

class domain_NN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 16):
        super(domain_NN, self).__init__(observation_space, features_dim=features_dim)

        self.passthrough = nn.Identity()

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # shape = observations.shape #dont need this as the domain is already an 1d tensor
        # observations = observations[:,0,:].reshape(shape[0], shape[-1])
        return self.passthrough(observations)

class PerceptionIMUDomainExtractor(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space) of dimension (1, 3, N_sensors)
    :param features_dim: (int) Number of features extracted.
    This corresponds to the number of unit for the last layer.
    """
    def __init__(self, observation_space: gym.spaces.Dict, 
                 img_size:int=224, 
                 features_dim:int=32,
                 device:str="cuda",
                 lock_params:bool=True,
                 pretrained_encoder_path:str=None):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super(PerceptionIMUDomainExtractor, self).__init__(observation_space, features_dim=1) # JØRGEN hvorfor er features_dim=1 her?

        extractors = {}
        total_concat_size = 0
        # We need to know size of the output of this extractor,
        # so go over all the spaces and compute output feature sizes
        for key, subspace in observation_space.spaces.items():
            if key == "perception":
                encoder = EncoderFeatureExtractor(subspace, image_size=img_size, channels=1, device=device, latent_dim=features_dim)
                # Get params from pre-trained encoder saved in file from path
                if pretrained_encoder_path is not None:
                    encoder.load_params(pretrained_encoder_path)
                if lock_params:
                    encoder.lock_params()
                extractors[key] = encoder
                total_concat_size += features_dim  # extractors[key].n_flatten
            elif key == "IMU":
                #Pass IMU features straight through to the MlpPolicy.
                extractors[key] = IMU_NN(subspace, features_dim=subspace.shape[-1]) #nn.Identity()
                total_concat_size += subspace.shape[-1]
            elif key == "domain":
                # Pass domain features straight through to the MlpPolicy.
                extractors[key] = domain_NN(subspace, features_dim=subspace.shape[-1]) #nn.Identity()
                total_concat_size += subspace.shape[-1]

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list = []

        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        return th.cat(encoded_tensor_list, dim=1)