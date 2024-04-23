import gymnasium as gym
import torch as th
import torch.nn as nn

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import matplotlib.pyplot as plt
import numpy as np

import time
import os

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


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    from stable_baselines3.common.base_class import BaseAlgorithm
    from stable_baselines3.ppo.policies import MlpPolicy
    from stable_baselines3 import PPO

    #### Test LidarCNN network circular 1D convolution:sel
    # Hyperparams
    n_sensors = 180
    kernel = 4
    padding = 4
    stride = 1

    ## Synthetic observation: (batch x channels x n_sensors)
    # Let obstacle be detected in the "edge" of the sensor array.
    # If circular padding works, should affect the outputs of the first <padding> elements
    obs = np.zeros((8, 3, n_sensors))
    obs[:, 0, :] = 150.0  # max distance
    obs[:, 0, -9:-1] = 10.0   # obstacle detected close in last 9 sensors
    obs[:, 1, :] = 0.0      # no obstacle
    obs[:, 2, :] = 0.0      # no obstacle
    obs = th.as_tensor(obs).float()

    ## Load existing convnet
    def load_net():
        from . import gym_quad
        algo = PPO
        #path = "radarCNN_example_Network.pkl"
        path = "../../radarCNN_example_Network150000.pkl"
        #path = "PPO_MlpPolicy_trained.pkl"
        #model = th.load(path)  # RunTimeError: : [enforce fail at ..\caffe2\serialize\inline_container.cc:114] . file in archive is not in a subdirectory: data
        #model = MlpPolicy.load(path)
        model = algo.load(path)


    load_net()
    print("loaded net")
    exit()
    ## Initialize convolutional layers (circular padding in all layers or just the first?)
    # First layer retains spatial structure,
    # includes circular padding to maintain the continuous radial structure of the RADAR,
    # and increased the feature-space dimensionality for extrapolation
    # (other padding types:)
    net1 = nn.Conv1d(in_channels=3, out_channels=6, kernel_size=kernel, padding=padding,
                      padding_mode='circular', stride=stride)
    # Second layer
    net2 = nn.Conv1d(in_channels=6, out_channels=3, kernel_size=kernel, padding=padding,
                      padding_mode='circular', stride=stride)
    net3 = nn.Conv1d(in_channels=3, out_channels=1, kernel_size=kernel, stride=2)
    net4 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=kernel, stride=2)

    flatten = nn.Flatten()
    act = nn.ReLU()
    #conv_weights = np.zeros(net1.weight.shape)

    #out1 = net1(obs)
    #out2 = net2(out1)
    #out3 = net3(out2)
    #out4 = net4(out3)
    out1 = act(net1(obs))
    out2 = act(net2(out1))
    out3 = act(net3(out2))
    out4 = act(net4(out3))

    feat = flatten(out4)


    ## Print shapes and characteritics of intermediate layer outputs
    obs = obs.detach().numpy()
    out1 = out1.detach().numpy()
    out2 = out2.detach().numpy()
    out3 = out3.detach().numpy()
    out4 = out4.detach().numpy()
    feat = feat.detach().numpy()

    def th2np_info(arr):
        #arr = tensor.detach().numpy()
        return "{:15.2f}{:15.2f}{:15.2f}{:15.2f}".format(arr.mean(), arr.std(), np.min(arr), np.max(arr))

    print("Observation",     obs.shape,  th2np_info(obs))
    print("First layer",     out1.shape, th2np_info(out1))
    print("Second layer",    out2.shape, th2np_info(out2))
    print("Third layer",     out3.shape, th2np_info(out3))
    print("Fourth layer",    out4.shape, th2np_info(out4))
    print("Output features", feat.shape, th2np_info(feat))

    ## PLOTTING
    plt.style.use('ggplot')
    plt.rc('font', family='serif')
    # plt.rc('font', family='serif', serif='Times')
    # plt.rc('text', usetex=True)
    plt.rc('xtick', labelsize=8)
    plt.rc('ytick', labelsize=8)
    plt.rc('axes', labelsize=8)
    plt.axis('scaled')
    def feat2radar(feat, avg=False):
        # Find length of feature vector
        n = feat.shape[-1] # number of activations differ between conv-layers
        feat = np.mean(feat, axis=0) if avg else feat[0] # average activations over batch or just select one

        # Find angles for each feature
        theta_d = 2 * np.pi / n  # Spatial spread according to the number of actications
        theta = np.array([(i + 1)*theta_d for i in range(n)]) # Angles for each activation

        # Hotfix: append first element of each list to connect the ends of the lines in the plot.
        theta = np.append(theta, theta[0])
        if len(feat.shape) > 1:
            _feat = []
            for ch, f in enumerate(feat):
                ext = np.concatenate((f, [f[0]]))
                _feat.append(ext)
        else:
            _feat = np.append(feat, feat[0])

        _feat = np.array(_feat)
        return theta, _feat  # Return angle positions & list of features.

    # sensor angles : -pi -> pi, such that the first sensor is directly behind the quadcopter, and sensors go counter-clockwise around to the back again.
    _d_sensor_angle = 2 * np.pi / n_sensors
    sensor_angles = np.array([(i + 1)*_d_sensor_angle for i in range(n_sensors)])
    sensor_distances = obs[0,0,:]

    # hotfix to connect lines on start and end
    sensor_angles = np.append(sensor_angles, sensor_angles[0])
    sensor_distances = np.append(sensor_distances, sensor_distances[0])


    fig, ax = plt.subplots(figsize=(11,11), subplot_kw={'projection': 'polar'})
    ax.set_theta_zero_location("S")
    ax.plot(sensor_angles, sensor_distances)
    ax.set_rmax(1)
    ax.set_rticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])  # Less radial ticks
    ax.set_rlabel_position(22.5)  # Move radial labels away from plotted line
    #ax.set_rscale('symlog')
    ax.grid(True)

    ax.set_title("LidarCNN: intermediate layers visualization", va='bottom')

    to_plot = [obs, out1, out2, out3, out4, feat]
    names = ["obs", "out1", "out2", "out3", "out4", "feat"]
    channels = [0,1,2,3,4,5]
    linetypes = ["solid", 'dotted', 'dashed', 'dashdot', (0, (5, 10)), (0, (3, 5, 1, 5, 1, 5))]
    CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                      '#f781bf', '#a65628', '#984ea3',
                      '#999999', '#e41a1c', '#dede00']

    layer_color = {
        'obs' : '#377eb8',
        'out1': '#ff7f00',
        'out2': '#4daf4a',
        'out3': '#f781bf',
        'out4': '#a65628',
        'feat': '#984ea3',
    }

    for arr, layer in zip(to_plot, names):
        angle, data = feat2radar(arr, avg=False)
        if len(data.shape) > 1:
            for ch, _d in enumerate(data):
                ax.plot(angle, _d, linestyle=linetypes[ch], color=layer_color[layer], label=layer+'_ch'+str(ch))
        else:
            ax.plot(angle, data, linestyle=linetypes[0], color=layer_color[layer], label=layer)

    plt.legend(loc="upper right", bbox_to_anchor=(1.2, 1.0))
    plt.tight_layout()
    plt.show()
