import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import os
from itertools import cycle
from matplotlib.lines import Line2D


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
        plt.imshow(input_img, cmap=cmap)
        plt.axis('off')
        save_path = os.path.join(path, f"{model_name}_experiment_{experiment_id}_{i}_input.pdf")
        plt.savefig(save_path, bbox_inches='tight')

        plt.figure(figsize=(10, 10))  
        plt.imshow(reconstruction, cmap=cmap)
        plt.axis('off')
        save_path = os.path.join(path, f"{model_name}_experiment_{experiment_id}_{i}_reconstructed.pdf")
        plt.savefig(save_path, bbox_inches='tight')


def plot_loss_ldim_sweep(total_losses:np.ndarray, BCE_losses:np.ndarray, KL_losses:np.ndarray, labels, path, save=False, include_train=True) -> None:
    # Dirty af with the indexing and stuff, but it works (for one seed at least)
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
                
        for j in range(len(train_trajs)): # Iterate through the trajs for all latent dims
            train_traj = train_trajs[j][0]
            val_traj = val_trajs[j][0]
            x = np.arange(len(train_traj))  # epochs
            print(train_traj)
            print(x)
            
            if include_train:
                ax[i].plot(x, train_traj, color=current_cycler[j], linestyle='--', linewidth=1)
                ax[i].plot(x, val_traj, color=current_cycler[j], linestyle='-', linewidth=1)
            else:
                ax[i].plot(x, val_traj, color=current_cycler[j], linestyle='-', linewidth=1)
    
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
, thresh=0.01
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
        
        

# TODO: Add functionality forplotting saved .npy loss trajectories for a given number of seeds and a given number of epochs:
def plot_separated_lossed_from_file(n_epochs, seeds, path):
    pass


# TODO: Add functionality for plotting test errors as function of beta for models with different latent dimensions


# TODO: Visualizing latent space traversing..... mayyyybe