import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.ticker import MaxNLocator
import matplotlib
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import os
from itertools import cycle
from matplotlib.lines import Line2D
import torchvision.transforms as transforms



def plot_separated_losses(total_losses:list, BCE_losses:list, KL_losses:list, labels:list, path:str, save=False) -> None:
    """
    Plots seperated losses as [total, BCE, KL] averaged over multiple seeds as functions of number of epochs, including variance-bands
    all three loss_trajectories must be lists with entries as np.ndarray:(n_seeds, n_epochs)
    """
    print('Plotting separated loss trajectories across multiple seeds...')
    
    if isinstance(KL_losses, np.ndarray): # force list-type if only one trajectory
        KL_losses = [KL_losses]
    if isinstance(BCE_losses, np.ndarray):
        BCE_losses = [BCE_losses]
    if isinstance(total_losses, np.ndarray):
        total_losses = [total_losses]

    mapping = {0:total_losses, 1:BCE_losses, 2:KL_losses}
    name = {0:'Total loss', 1:'Reconstruction error', 2:'KL divergence'}

    plt.style.use('ggplot')
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)
    plt.rc('axes', labelsize=12)
    fig, axes = plt.subplots(1, 3, figsize=(20,5))

    x = np.arange(len(KL_losses[0][0,:])) # epochs, extracted from first trajectory
    for i in range(3):
        axes[i].set_xlabel('Epochs')
        axes[i].set_ylabel('Loss')
        axes[i].set_title(name[i])
        for j, loss_traj in enumerate(mapping[i]): # go through loss trajs (np.ndarray:(n_seeds, n_epochs))
            # Get mean and variance
            mean_error_traj = np.mean(loss_traj, axis=0)
            variance_traj = np.std(loss_traj, axis=0)
            conf_interval = 1.96 * np.std(loss_traj, axis=0) / np.sqrt(len(x)) # 95% confidence interval
            # Insert into plot
            axes[i].plot(x, mean_error_traj, label=labels[j], linewidth=1)
            axes[i].fill_between(x, mean_error_traj - conf_interval, mean_error_traj + conf_interval, alpha=0.2)
            axes[i].legend()

    if save:
        path = os.path.join(path, f'LOSS_SEPARATED.pdf')
        fig.savefig(path, bbox_inches='tight')

def reconstruct_and_plot(input_img_as_tensor, vae: nn.Module, model_name: str, experiment_id: int, path: str, i:int, cmap, save=False) -> None:
    """Plots input image and its reconstruction"""
    print('Plotting input image and its reconstruction...')

    # Get input image and its reconstruction
    input_img = input_img_as_tensor.detach().cpu().numpy().squeeze()
    x_hat, _, _ = vae(input_img_as_tensor)
    reconstruction = x_hat.detach().cpu().numpy().squeeze()

    # Plot
    plt.style.use('ggplot')
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)
    plt.rc('axes', labelsize=12)

    # Remove grid and ticks
    if save:
        if not os.path.exists(path):
            os.makedirs(path)
        
        plt.figure(figsize=(10, 10))  
        plt.imshow(input_img, cmap=cmap, vmin=0, vmax=1)
        plt.axis('off')
        save_path = os.path.join(path, f"{model_name}_experiment_{experiment_id}_{i}_input.pdf")
        plt.savefig(save_path, bbox_inches='tight')

        plt.figure(figsize=(10, 10))  
        plt.imshow(reconstruction, cmap=cmap, vmin=0, vmax=1)
        plt.axis('off')
        save_path = os.path.join(path, f"{model_name}_experiment_{experiment_id}_{i}_reconstructed.pdf")
        plt.savefig(save_path, bbox_inches='tight')


def plot_loss_ldim_sweep(total_losses:np.ndarray, BCE_losses:np.ndarray, KL_losses:np.ndarray, labels, path, save=False, include_train=True) -> None:
    """
    Plots loss trajectories from beta sweep specifically
    each loss traj is [list of shape n_ldims, list of shape n_lidims] where each list is a list of np.ndarrays of shape (n_seeds, n_epochs) (i.e. a loss trajectory of the given latent dimensionality)
    """
    
    if isinstance(KL_losses, np.ndarray): # force list-type if only one trajectory (this is the case as per 27.02)
        KL_losses = [KL_losses]
    if isinstance(BCE_losses, np.ndarray):
        BCE_losses = [BCE_losses]
    if isinstance(total_losses, np.ndarray):
        total_losses = [total_losses]
        
    mapping = {0:total_losses, 1:BCE_losses, 2:KL_losses}
    names = {0:'Total loss', 1:'Reconstruction error', 2:'KL divergence'}

    plt.style.use('ggplot')
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)
    plt.rc('axes', labelsize=12)
    fig, ax = plt.subplots(1, 3, figsize=(20,5))

    for i in range(3): # For each subplot
        # Subplot (column) settings
        ax[i].set_xlabel('Epochs')
        ax[i].set_ylabel('Loss')
        ax[i].set_title(names[i])
        loss_trajectories = mapping[i] # Total, BCE or KL divergence                  
        current_cycler = plt.rcParams['axes.prop_cycle'].by_key()['color'] # Initialize color cycler
        
        # Get all training and validation trajectories for the given loss type
        train_trajs = loss_trajectories[:len(loss_trajectories) // 2][0]
        val_trajs = loss_trajectories[len(loss_trajectories) // 2:][0]
                
        for j, val_trajs in enumerate(val_trajs): # Iterate through the trajs for all latent dims
            x = np.arange(len(val_trajs[0,:]))  # epochs
            mean_val = np.mean(val_trajs, axis=0)
            ci = 1.96 * np.std(val_trajs, axis=0) / np.sqrt(len(x)) # 95% confidence interval
            ax[i].plot(x, mean_val, label=labels[j], color=current_cycler[j], linestyle='-', linewidth=1)
            ax[i].fill_between(x, mean_val - ci, mean_val + ci, alpha=0.2)
            #if include_train:
            #    ax[i].plot(x, train_traj, color=current_cycler[j], linestyle='--', linewidth=1)
            #    ax[i].plot(x, val_traj, color=current_cycler[j], linestyle='-', linewidth=1)
            #else:
            #ax[i].plot(x, val_traj, color=current_cycler[j], linestyle='-', linewidth=1)

        ax[i].xaxis.set_major_locator(MaxNLocator(integer=True))  # Ensure integer ticks on x-axis

    # Create custom legend entries
    if include_train:
        legend_elements = [
            Line2D([0], [0], color='black', lw=1, linestyle='-', label='Validation loss'),
            Line2D([0], [0], color='black', lw=1, linestyle='--', label='Training loss')
        ] + [Line2D([0], [0], color=c, lw=1, label=label) for c, label in zip(current_cycler[:len(labels)], labels)]
    
    else:
        legend_elements = [Line2D([0], [0], color=c, lw=1, label=label) for c, label in zip(current_cycler[:len(labels)], labels)]
    
    # Adding the legend to the last subplot for clarity
    ax[-1].legend(handles=legend_elements, loc='upper right')
    if save:
        path = os.path.join(path, f'beta_sweep_separated.pdf')
        fig.savefig(path, bbox_inches='tight')



def kde(zs, xlabel:str, ylabel:str, path:str, name:str, save=True):
    """plot kde of latent space for two z vectors"""
    # create kde plot
    plt.style.use('ggplot')
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)
    plt.rc('axes', labelsize=12)
    
    sns.kdeplot(x=zs[:,0], y=zs[:,1], fill=True, levels=20, bw_adjust=0.9)
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    ax = plt.gca()
    ax.set_box_aspect(1)
    #ax.set_aspect('equal', 'box')
    
    if save:
        plt.savefig(f'{path}/{name}.pdf', bbox_inches='tight')
    plt.clf()
    plt.cla()
    
def latent_space_kde(model:nn.Module, dataloader:torch.utils.data.DataLoader, latent_dim:int, name:str, path:str, save=True, combos=[(0,1)]):
    """Run whole test set though encoder of VAE model and plot kde of latent space for combinations of the latent variables"""
    print(f'Plotting latent space kde for model "{name}"...')
    zs = np.zeros((1,latent_dim)) # 1 x l_dim to be filled vertically
    for i, x_batch in enumerate(dataloader):
        # Get latent representations for batch
        z, _, _ = model.encoder(x_batch) # z is (batch_size, latent_dim)
        z = z.detach().numpy()[0,:]
        if i == 0:
            zs = z
        else:
            zs = np.vstack((zs, z)) # Stack whole batch vertically to all zs
    
    for z_pair in combos:
        xlabel = f'z{z_pair[0]}'
        ylabel = f'z{z_pair[1]}'
        # Pick out columns from zs base on z_pair
        z1 = zs[:,z_pair[0]]
        z2 = zs[:,z_pair[1]]
        zs_pair = np.column_stack((z1,z2))
        kde(zs_pair, xlabel, ylabel,path, f'{name}_kde_{z_pair[0]}_{z_pair[1]}', save=save)

def visualize_filters(encoder:nn.Module, savepath, input_image, ending):
    # Must be changed if used on other models than conv1
    
    conv_weights = []  # List to store convolutional layer weights
    conv_layers = []  # List to store convolutional layers
    total_conv_layers = 0  # Counter for total convolutional layers
    for module in encoder.conv_block.children():
        if isinstance(module, nn.Conv2d):
            total_conv_layers += 1
            conv_weights.append(module.weight)
            conv_layers.append(module)
    print(f'Total convolutional layers: {total_conv_layers}')
    
    if input_image is not None:
        # Extract feature maps
        feature_maps = []  # List to store feature maps
        layer_names = []  # List to store layer names
        for layer in conv_layers:
            input_image = layer(input_image)
            feature_maps.append(input_image)
            layer_names.append(str(layer))
        
        print("\nFeature maps shape")
        for feature_map in feature_maps:
            print(feature_map.shape)
        
        # Process and visualize feature maps
        processed_feature_maps = []  # List to store processed feature maps
        for feature_map in feature_maps:
            feature_map = feature_map.squeeze(0)
            processed_feature_maps.append(feature_map.data.cpu().numpy())
        
    plt.style.use('ggplot')
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)
    plt.rc('axes', labelsize=12)
    
    for layer in range(1, total_conv_layers+1):
        plt.figure(figsize=(20,20))
        subplot_shapes = {1:(6, 6), 2:(8, 8), 3:(12,12), 4:(16,16)}
        for i, filt in enumerate(conv_weights[layer-1]):
            row, col = subplot_shapes[layer]
            plt.subplot(row, col, i+1)
            if input_image is not None:
                # Iterate through feature maps and plot in separate plot
                feature_map = processed_feature_maps[layer-1][i,:,:]
                plt.imshow(feature_map, cmap='gray')
                plt.axis('off')
                fname = f'conv_layer_{layer}_filtered_img_{str(ending)}.pdf'
            else:
                # Looking at the fist color channel converted to grayscale
                filtr = filt[0,:,:].detach().numpy()
                plt.imshow(filtr, cmap='gray')
                plt.axis('off')
                fname = f'conv_layer_{layer}_filter.pdf'
        plt.savefig(f"{savepath}/{fname}", bbox_inches='tight')
        plt.clf()

        
def visualize_feature_maps(encoder:nn.Module, input_image, savepath, ending):
    # Assumes input image has been transformed to fit into encoder input
    # Input may be noise or actual image
    
    img = input_image.detach().cpu().numpy().squeeze()
    
    conv_weights = []  # List to store convolutional layer weights
    conv_layers = []  # List to store convolutional layers
    total_conv_layers = 0  # Counter for total convolutional layers
    for module in encoder.conv_block.children():
        if isinstance(module, nn.Conv2d):
            total_conv_layers += 1
            conv_weights.append(module.weight)
            conv_layers.append(module)
    print(f'Total convolutional layers: {total_conv_layers}')
    
    #input_image = input_image.unsqueeze(0)  # Add a batch dimension
    
    # Extract feature maps
    feature_maps = []  # List to store feature maps
    layer_names = []  # List to store layer names
    for layer in conv_layers:
        input_image = layer(input_image)
        feature_maps.append(input_image)
        layer_names.append(str(layer))
    
    print("\nFeature maps shape")
    for feature_map in feature_maps:
        print(feature_map.shape)
    
    # Process and visualize feature maps
    processed_feature_maps = []  # List to store processed feature maps
    for feature_map in feature_maps:
        feature_map = feature_map.squeeze(0)  # Remove the batch dimension
        mean_feature_map = torch.sum(feature_map, 0) / feature_map.shape[0]  # Compute mean across channels
        processed_feature_maps.append(mean_feature_map.data.cpu().numpy())
    

    # Display processed feature maps shapes
    print("\n Processed feature maps shape")
    for fm in processed_feature_maps:
        print(fm.shape)
    
    plt.style.use('ggplot')
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)
    plt.rc('axes', labelsize=12)
    
    layer_names_new = ['Conv1', 'Conv2', 'Conv3', 'Conv4']
    # Plot the feature maps
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(1, 5, 1)
    ax.imshow(img, cmap='magma')
    ax.axis("off")
    ax.set_title("Input Image")#, fontsize=30)
    for i in range(len(processed_feature_maps)):
        ax = fig.add_subplot(1, 5, i + 2)
        ax.imshow(processed_feature_maps[i], cmap='magma')
        ax.axis("off")
        #ax.set_title(layer_names[i].split('(')[0], fontsize=30)
        ax.set_title(layer_names_new[i])#, fontsize=30)
        
    plt.savefig(f"{savepath}/feature_maps_{str(ending)}.pdf", bbox_inches='tight')


    # Find the input image that maximizes the activation for the conv layers
class ActivationMaximization:
    def __init__(self, encoder:nn.Module, epochs, cnn_layer, cnn_filter):    
        self.model = encoder
        self.model.eval()
        self.epochs = epochs
        self.cnn_layer = cnn_layer
        self.cnn_filter = cnn_filter
        self.conv_output = 0
    
    def hook_cnn_layer(self):
        def hook_fn(module, grad_in, grad_out):
            self.conv_output = grad_out[0, self.cnn_filter]
            #print('---- conv out -----')
            #print(type(self.conv_output))
            #print(self.conv_output.shape)
            #print(grad_out.shape)
            #print(grad_out[0, self.cnn_filter])
            #print('\n----- grad in ------')
            #print(grad_in[0][0][0])
            # saving the number of filters in that layer
            num_filters = grad_out.shape[1]
        self.model[self.cnn_layer].register_forward_hook(hook_fn)

    
    def visualize_activation_maximization(self, savepath):
        self.hook_cnn_layer()
        noisy_im = torch.randn(1, 1, 224, 224)
        #transform = transforms.Compose([
         #   transforms.Resize((224, 224)),
        #    transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.5, 1.5)),
        #    transforms.Lambda(lambda x: torch.clamp(x, 0, 1)),  # Clamping values to [0, 1] due to some pixels exceeding 1 (with 10^-6) when resizing and interpolating (and gaussian blur apparently)
        #])
        #processed_image = transform(noisy_im).requires_grad_() #.unsqueeze_(0).requires_grad_()
        processed_image = noisy_im.requires_grad_()
        optimizer = torch.optim.Adam([noisy_im], lr=0.1, weight_decay=1e-6)
        for e in range(1, self.epochs):
                optimizer.zero_grad() # zero out gradients
                x = processed_image
                
                # iterate through each layer of the model
                for idx, layer in enumerate(self.model):
                    # pass processed image through each layer
                    x = layer(x)
                    # activate hook when image gets to the layer in question
                    if idx == self.cnn_layer:
                        break
                # print(self.num_filters)
                # self.conv_output = x[0, self.cnn_filter]
                # loss function according to Erhan et al. (2009)
                loss = -torch.mean(self.conv_output)
                #print(loss.shape)
                print(loss.data)
                loss.backward() # calculate gradients
                optimizer.step() # update weights
                layer_img = processed_image # reconstruct image (no need img is already in 0,1 and good)
        
                if e == self.epochs-1:# or e % 100 == 0:
                    img_path = f'{savepath}/activation_maximization_filter_{self.cnn_filter}_epoch_{e}.pdf'
                    fig = plt.figure()
                    plt.imshow(layer_img.detach().cpu().numpy().squeeze(), cmap='gray')
                    plt.axis('off')
                    plt.savefig(img_path, bbox_inches='tight')
                    plt.clf()
    

# TODO: Add functionality forplotting saved .npy loss trajectories for a given number of seeds and a given number of epochs:
def plot_separated_lossed_from_file(n_epochs, seeds, path):
    pass


# TODO: Add functionality for plotting test errors as function of beta for models with different latent dimensions


# TODO: Visualizing latent space traversing..... mayyyybe