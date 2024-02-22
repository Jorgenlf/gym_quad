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



def plot_separated_losses(total_losses:list, BCE_losses:list, KL_losses:list, labels:list, path:str, save=False) -> None:
    """
    Plots seperated losses as [total, BCE, KL] averaged over multiple seeds as functions of number of epochs, including variance-bands
    all three loss_trajectories must be lists with entries as np.ndarray:(n_seeds, n_epochs)
    """
    print('Plotting separated loss trajectories across multiple seeds...')
    
    fill_between = True
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





# TODO: Add functionality forplotting saved .npy loss trajectories for a given number of seeds and a given number of epochs:
def plot_separated_lossed_from_file(n_epochs, seeds, path):
    pass


# TODO: Add latent space kde plot functionality for experimenting w/ betas








# TODO: Visualizing latent space traversing..... mayyyybe