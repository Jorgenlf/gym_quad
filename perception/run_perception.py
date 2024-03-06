import torch
from torch.optim import Adam
import torch.nn as nn
import os
import sys
import numpy as np
import argparse
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

from utils_perception.data_reader import DataReader, SunRGBD, CustomDepthDataset, RealSenseDataset
from utils_perception import plotting

#from utils_perception.data_augmentation import DataAugmentation

from VAE.encoders import ConvEncoder1, VGG16Encoder, ResNet50Encoder
from VAE.decoders import ConvDecoder1
from VAE.vae import VAE

from train_perception import TrainerVAE


# HYPERPARAMS
#LEARNING_RATE = 0.001
#N_EPOCH = 25
#BATCH_SIZE = 64     
#LATENT_DIMS = 64

def main(args):
    # Set hyperparameters
    BATCH_SIZE    = args.batch_size     # Default: 64
    N_EPOCH       = args.epochs         # Default: 25
    LATENT_DIMS   = args.latent_dims    # Default: 12
    LEARNING_RATE = args.learning_rate  # Default: 0.001
    NUM_SEEDS     = args.num_seeds      # Default: 1
    BETA          = args.beta           # Default: 1
    #EPS_WEIGHT    = args.eps_weight     # Default: 1

    IMG_SIZE = 224
    NUM_CHANNELS = 1
    

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    
    print('Preparing data...\n') # Do this on any mode

    # Get SUN RGBD dataset
    print('Getting SUN RGBD dataset')
    sun = SunRGBD(orig_data_path="data/sunrgbd_stripped", data_path_depth="data/sunrgbd_images_depth", data_path_rgb="data/sunrgbd_images_rgb")
    

    # Define transformatons additional to the normalization and channel conversion done in the DataReader
    train_additional_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        #transforms.RandomVerticalFlip(p=0.25),
        #transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.5, 1.5)),
        #transforms.Normalize(mean=[0.291],
        #                     std=[0.147])
        #transforms.RandomRotation(degrees=(30, 70)),
        transforms.Lambda(lambda x: torch.clamp(x, 0, 1)),  # Clamping values to [0, 1] due to some pixels exceeding 1 (with 10^-6) when resizing and interpolating (and gaussian blur apparently)
    ])
    valid_additional_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        #transforms.Normalize(mean=[0.291],
        #                     std=[0.147])
        transforms.Lambda(lambda x: torch.clamp(x, 0, 1))
    ])
    
    print(f'Loading data with image size = ({NUM_CHANNELS}, {IMG_SIZE}, {IMG_SIZE}), batch size = {BATCH_SIZE}, train-test split = {0.7} and train-val split = {0.3}')
    print(f'Additional transformations to training data:\n {train_additional_transform}')
    
    dataloader_sun = DataReader(path_img_depth="data/sunrgbd_images_depth",
                            path_img_rgb="data/sunrgbd_images_rgb",
                            batch_size=BATCH_SIZE,
                            train_test_split=0.7,
                            train_val_split=0.3,
                            transforms_train=train_additional_transform,
                            transforms_validate=valid_additional_transform)
    

    # Load data and create dataloaders
    train_loader, val_loader, test_loader = dataloader_sun.load_split_data_sunrgbd(sun, seed=None, shuffle=True)
    
    print(f'Data loaded\nSize train: {len(train_loader.dataset)} | Size validation: {len(val_loader.dataset)} | Size test: {len(test_loader.dataset)}\n')


    realsense_dataset = RealSenseDataset(root_dir="data/realsense_data/depth_imgs", transform=transforms.Resize((IMG_SIZE, IMG_SIZE)))
    realsense_loader = torch.utils.data.DataLoader(realsense_dataset, batch_size=1, shuffle=True)

    print(f'Realsense data loaded\nSize: {len(realsense_loader.dataset)}\n')

    # Augment data
    #data_augmentation = DataAugmentation()
    #dataloader_train, dataloader_validate, dataloader_test = data_augmentation.augment_data(dataloader_train, dataloader_validate, dataloader_test)
    
    model_name = args.model_name
    experiment_id = args.exp_id

    if args.mode == 'train':
        if any(mode in args.plot for mode in ['losses']):
            print(f"Training...\nPlotting mode = {args.plot}")
            
            # Containers for loss trajectories
            total_train_losses = np.full((NUM_SEEDS, N_EPOCH), np.nan)
            total_val_losses = np.full((NUM_SEEDS, N_EPOCH), np.nan)
            bce_train_losses = np.full((NUM_SEEDS, N_EPOCH), np.nan)
            bce_val_losses = np.full((NUM_SEEDS, N_EPOCH), np.nan)
            kl_train_losses = np.full((NUM_SEEDS, N_EPOCH), np.nan)
            kl_val_losses = np.full((NUM_SEEDS, N_EPOCH), np.nan)

            for i in range(NUM_SEEDS):
                print(f'[Seed {i+1}/{NUM_SEEDS}]')

                # Load data with different seed
                train_loader, val_loader, test_loader = dataloader_sun.load_split_data_sunrgbd(sun, seed=None, shuffle=True)

                # Create VAE based on args.model_name
                if model_name == 'conv1':
                    encoder = ConvEncoder1(image_size=IMG_SIZE, channels=NUM_CHANNELS, latent_dim=LATENT_DIMS)
                    decoder = ConvDecoder1(image_size=IMG_SIZE, channels=NUM_CHANNELS, latent_dim=LATENT_DIMS, flattened_size=encoder.flattened_size, dim_before_flatten=encoder.dim_before_flatten)
                    vae = VAE(encoder, decoder, LATENT_DIMS, BETA).to(device)
                if model_name == 'vgg16':
                    decoder_intermediate_size = torch.Size([1,256,14,14])
                    decoder_flattened_size = 50176
                    encoder = VGG16Encoder(latent_dim=LATENT_DIMS, image_size=IMG_SIZE)
                    decoder = ConvDecoder1(image_size=IMG_SIZE, channels=NUM_CHANNELS, latent_dim=LATENT_DIMS, flattened_size=decoder_flattened_size, dim_before_flatten=decoder_intermediate_size)
                    vae = VAE(encoder, decoder, LATENT_DIMS, BETA)#.to(device)
                if model_name == 'resnet50':
                    decoder_intermediate_size = torch.Size([1,256,14,14])
                    decoder_flattened_size = 50176
                    encoder = ResNet50Encoder(latent_dim=LATENT_DIMS, image_size=IMG_SIZE)
                    decoder = ConvDecoder1(image_size=IMG_SIZE, channels=NUM_CHANNELS, latent_dim=LATENT_DIMS, flattened_size=decoder_flattened_size, dim_before_flatten=decoder_intermediate_size)
                    vae = VAE(encoder, decoder, LATENT_DIMS, BETA)


                # Train model
                optimizer = Adam(vae.parameters(), lr=LEARNING_RATE)
                trainer = TrainerVAE(model=vae, 
                                    epochs=N_EPOCH, 
                                    learning_rate=LEARNING_RATE, 
                                    batch_size=BATCH_SIZE, 
                                    dataloader_train=train_loader, 
                                    dataloader_val=val_loader, 
                                    optimizer=optimizer, 
                                    beta=BETA,
                                    reconstruction_loss="MSE")
                
                trained_epochs = trainer.train()
                    
                # Only insert to the first trained_epochs elements if early stopping has been triggered
                total_train_losses[i,:trained_epochs] = trainer.training_loss['Total loss']
                total_val_losses[i,:trained_epochs] = trainer.validation_loss['Total loss']
                bce_train_losses[i,:trained_epochs] = trainer.training_loss['Reconstruction loss']
                bce_val_losses[i,:trained_epochs] = trainer.validation_loss['Reconstruction loss']
                kl_train_losses[i,:trained_epochs] = trainer.training_loss['KL divergence loss']
                kl_val_losses[i,:trained_epochs] = trainer.validation_loss['KL divergence loss']

                # Save encoder and decoder parameters to file
                savepath_encoder = f'models/encoders'
                savepath_decoder = f'models/decoders'
                os.makedirs(savepath_encoder, exist_ok=True)
                os.makedirs(savepath_decoder, exist_ok=True)

                if args.save_model:
                    vae.encoder.save(path=os.path.join(savepath_encoder, f'encoder_{model_name}_experiment_{experiment_id}_seed{i+1}.json'))
                    vae.decoder.save(path=os.path.join(savepath_decoder, f'decoder_{model_name}_experiment_{experiment_id}_seed{i+1}.json'))

                del train_loader, val_loader, encoder, decoder, vae, optimizer, trainer

            # Save loss trajectories to file, so specific loss trajs can be plotted later on
            savepath_loss_numerical = f'results/{model_name}/numerical/losses/exp{experiment_id}'
            os.makedirs(savepath_loss_numerical, exist_ok=True)

            np.save(os.path.join(savepath_loss_numerical, f'total_train_losses.npy'), total_train_losses)
            np.save(os.path.join(savepath_loss_numerical, f'total_val_losses.npy'), total_val_losses)
            np.save(os.path.join(savepath_loss_numerical, f'bce_train_losses.npy'), bce_train_losses)
            np.save(os.path.join(savepath_loss_numerical, f'bce_val_losses.npy'), bce_val_losses)
            np.save(os.path.join(savepath_loss_numerical, f'kl_train_losses.npy'), kl_train_losses)
            np.save(os.path.join(savepath_loss_numerical, f'kl_val_losses.npy'), kl_val_losses)
            

            if "losses" in args.plot:
                # Plot the aggregated loss trajectories at end of run
                total_losses = [total_train_losses, total_val_losses]
                bce_losses = [bce_train_losses, bce_val_losses]
                kl_losses = [kl_train_losses, kl_val_losses]
                labels = ['Training loss', 'Validation loss']

                savepath_loss_plot = f'results/{model_name}/plots/losses/exp{experiment_id}'
                os.makedirs(savepath_loss_plot, exist_ok=True)
                plotting.plot_separated_losses(total_losses=total_losses,
                                            BCE_losses=bce_losses,
                                            KL_losses=kl_losses,
                                            labels=labels,
                                            path=savepath_loss_plot,
                                            save=True)
                
            # Potential other plottinfg modes for training trajectories....
            
            
            
        if 'kde' in args.plot:
            pass
            
        if "latent_dims_sweep" in args.plot:
            print('Latent dimension sweep test...')
            # Made for 1 seed only as per 27.02 due to computational resources.
            
            latent_dims = [2, 4, 8, 16, 32, 64, 128]
            
            #latent_dims = [2, 16, 64]
            
            # Containers w/ loss trajs for each latent dim
            total_val_losses_for_latent_dims = [] 
            bce_val_losses_for_latent_dims = []   
            kl_val_losses_for_latent_dims = []
            total_train_losses_for_latent_dims = []
            bce_train_losses_for_latent_dims = []
            kl_train_losses_for_latent_dims = []
            
            for l in latent_dims:
                print(f'Latent dimension: {l}')
                total_train_losses = np.full((NUM_SEEDS, N_EPOCH), np.nan)
                total_val_losses = np.full((NUM_SEEDS, N_EPOCH), np.nan)
                bce_train_losses = np.full((NUM_SEEDS, N_EPOCH), np.nan)
                bce_val_losses = np.full((NUM_SEEDS, N_EPOCH), np.nan)
                kl_train_losses = np.full((NUM_SEEDS, N_EPOCH), np.nan)
                kl_val_losses = np.full((NUM_SEEDS, N_EPOCH), np.nan)
                
                for i in range(NUM_SEEDS):
                    # Load data with different seed
                    seed = 42 # Logic regarding seeding must be changed if many seeds used
                    seed = seed + i
                    train_loader, val_loader, test_loader = dataloader_sun.load_split_data_sunrgbd(sun, seed=seed, shuffle=True)

                    # Create VAE based on args.model_name
                    if model_name == 'conv1':
                        encoder = ConvEncoder1(image_size=IMG_SIZE, channels=NUM_CHANNELS, latent_dim=l)
                        decoder = ConvDecoder1(image_size=IMG_SIZE, channels=NUM_CHANNELS, latent_dim=l, flattened_size=encoder.flattened_size, dim_before_flatten=encoder.dim_before_flatten)
                        vae = VAE(encoder, decoder, l, BETA).to(device)

                    # Train model
                    optimizer = Adam(vae.parameters(), lr=LEARNING_RATE)
                    trainer = TrainerVAE(model=vae, 
                                        epochs=N_EPOCH, 
                                        learning_rate=LEARNING_RATE, 
                                        batch_size=BATCH_SIZE, 
                                        dataloader_train=train_loader, 
                                        dataloader_val=val_loader, 
                                        optimizer=optimizer, 
                                        beta=BETA,
                                        reconstruction_loss="MSE")
                    
                    trained_epochs = trainer.train(early_stopping=False)
                    
                    # Only insert to the first trained_epochs elemts if early stopping has been triggered
                    total_train_losses[i,:trained_epochs] = trainer.training_loss['Total loss']
                    total_val_losses[i,:trained_epochs] = trainer.validation_loss['Total loss']
                    bce_train_losses[i,:trained_epochs] = trainer.training_loss['Reconstruction loss']
                    bce_val_losses[i,:trained_epochs] = trainer.validation_loss['Reconstruction loss']
                    kl_train_losses[i,:trained_epochs] = trainer.training_loss['KL divergence loss']
                    kl_val_losses[i,:trained_epochs] = trainer.validation_loss['KL divergence loss']
                    
                    print(total_train_losses)

                    # Save encoder and decoder parameters to file
                    savepath_encoder = f'models/encoders'
                    savepath_decoder = f'models/decoders'
                    os.makedirs(savepath_encoder, exist_ok=True)
                    os.makedirs(savepath_decoder, exist_ok=True)

                    if args.save_model:
                        vae.encoder.save(path=os.path.join(savepath_encoder, f'encoder_{model_name}_experiment_{experiment_id}_seed{i+1}_dim{l}.json'))
                        vae.decoder.save(path=os.path.join(savepath_decoder, f'decoder_{model_name}_experiment_{experiment_id}_seed{i+1}_dim{l}.json'))

                    del train_loader, val_loader, encoder, decoder, vae, optimizer, trainer
                
                total_val_losses_for_latent_dims.append(total_val_losses)
                bce_val_losses_for_latent_dims.append(bce_val_losses)
                kl_val_losses_for_latent_dims.append(kl_val_losses)
                total_train_losses_for_latent_dims.append(total_train_losses)
                bce_train_losses_for_latent_dims.append(bce_train_losses)
                kl_train_losses_for_latent_dims.append(kl_train_losses)
                
                # Save loss trajectories to file, so specific loss trajs can be plotted later on
                savepath_loss_numerical = f'results/{model_name}/numerical/losses/exp{experiment_id}/latent_dim_{l}'
                os.makedirs(savepath_loss_numerical, exist_ok=True)

                np.save(os.path.join(savepath_loss_numerical, f'total_train_losses.npy'), total_train_losses)
                np.save(os.path.join(savepath_loss_numerical, f'total_val_losses.npy'), total_val_losses)
                np.save(os.path.join(savepath_loss_numerical, f'bce_train_losses.npy'), bce_train_losses)
                np.save(os.path.join(savepath_loss_numerical, f'bce_val_losses.npy'), bce_val_losses)
                np.save(os.path.join(savepath_loss_numerical, f'kl_train_losses.npy'), kl_train_losses)
                np.save(os.path.join(savepath_loss_numerical, f'kl_val_losses.npy'), kl_val_losses)
                
                # Plot the aggregated loss trajectories at end of run
                total_losses = [total_train_losses, total_val_losses]
                bce_losses = [bce_train_losses, bce_val_losses]
                kl_losses = [kl_train_losses, kl_val_losses]
                labels = ['Training loss', 'Validation loss']

                savepath_loss_plot = f'results/{model_name}/plots/losses/exp{experiment_id}/latent_dim_{l}'
                os.makedirs(savepath_loss_plot, exist_ok=True)
                plotting.plot_separated_losses(total_losses=total_losses,
                                            BCE_losses=bce_losses,
                                            KL_losses=kl_losses,
                                            labels=labels,
                                            path=savepath_loss_plot,
                                            save=True)
                    
            
            # Plot the aggregated loss trajectories at end of run
            savepath_ldim_plot = f'results/{model_name}/plots/losses/exp{experiment_id}/ldim_sweep'
            os.makedirs(savepath_ldim_plot, exist_ok=True)
            labels = [f'Latent dim = {l}' for l in latent_dims]
            total_losses = [total_train_losses_for_latent_dims, total_val_losses_for_latent_dims]
            bce_losses = [bce_train_losses_for_latent_dims, bce_val_losses_for_latent_dims]
            kl_losses = [kl_train_losses_for_latent_dims, kl_val_losses_for_latent_dims]
            
            plotting.plot_loss_ldim_sweep(total_losses=total_losses,
                                        BCE_losses=bce_losses,
                                        KL_losses=kl_losses,
                                        labels=labels,
                                        path=savepath_ldim_plot,
                                        save=True,
                                        include_train=False)
            
                        
                    
                
    if args.mode == 'test':
        seed = args.seed
        full_name = f'{model_name}_experiment_{experiment_id}_seed{seed}'

        # Load model for testing
        if model_name == 'conv1':
            encoder = ConvEncoder1(image_size=IMG_SIZE, channels=NUM_CHANNELS, latent_dim=LATENT_DIMS)
            encoder.load(f"models/encoders/encoder_{full_name}.json")
            decoder = ConvDecoder1(image_size=IMG_SIZE, channels=NUM_CHANNELS, latent_dim=LATENT_DIMS, flattened_size=encoder.flattened_size, dim_before_flatten=encoder.dim_before_flatten)
            decoder.load(f"models/decoders/decoder_{full_name}.json")
            vae = VAE(encoder, decoder, LATENT_DIMS, BETA).to(device)
        if model_name == 'vgg16':
            decoder_intermediate_size = torch.Size([1,256,14,14])
            decoder_flattened_size = 50176
            encoder = VGG16Encoder(latent_dim=LATENT_DIMS, image_size=IMG_SIZE)
            encoder.load(f"models/encoders/encoder_{full_name}.json")
            decoder = ConvDecoder1(image_size=IMG_SIZE, channels=NUM_CHANNELS, latent_dim=LATENT_DIMS, flattened_size=decoder_flattened_size, dim_before_flatten=decoder_intermediate_size)
            decoder.load(f"models/decoders/decoder_{full_name}.json")
            vae = VAE(encoder, decoder, LATENT_DIMS, BETA).to(device)
        if model_name == 'resnet50':
            decoder_intermediate_size = torch.Size([1,256,14,14])
            decoder_flattened_size = 50176
            encoder = ResNet50Encoder(latent_dim=LATENT_DIMS, image_size=IMG_SIZE)
            encoder.load(f"models/encoders/encoder_{full_name}.json")
            decoder = ConvDecoder1(image_size=IMG_SIZE, channels=NUM_CHANNELS, latent_dim=LATENT_DIMS, flattened_size=decoder_flattened_size, dim_before_flatten=decoder_intermediate_size)
            decoder.load(f"models/decoders/decoder_{full_name}.json")
            vae = VAE(encoder, decoder, LATENT_DIMS, BETA).to(device)
        """encoder = ConvEncoder1(image_size=IMG_SIZE, channels=NUM_CHANNELS, latent_dim=LATENT_DIMS)
        encoder.load(f"models/encoders/encoder_{full_name}.json")
        decoder = ConvDecoder1(image_size=IMG_SIZE, channels=NUM_CHANNELS, latent_dim=LATENT_DIMS, flattened_size=encoder.flattened_size, dim_before_flatten=encoder.dim_before_flatten)
        decoder.load(f"models/decoders/decoder_{full_name}.json")
        vae = VAE(encoder, decoder, LATENT_DIMS, BETA).to(device)"""

        # Test realsense data
        savepath_realsense = f'results/{model_name}/plots/realsense/exp{experiment_id}'
        loss = 0.0
        for i, x in enumerate(realsense_loader):
            img = x.detach().cpu().numpy().squeeze()
            x_hat, _, _ = vae(x)
            valid_pixels = torch.where(x > 0, torch.ones_like(x), torch.zeros_like(x)) # Mask is defined for the whole batch (x)
            MSE_loss_ = F.mse_loss(x_hat, x, reduction='none') * valid_pixels
            recon_loss = torch.mean(torch.sum(MSE_loss_))
            loss += recon_loss.item() # just do reconstruction loss for now
            #if i < args.num_examples:
                #plotting.reconstruct_and_plot(x, vae, model_name, experiment_id, savepath_realsense, i, cmap='magma', save=True)
        
        savepath2 = f'results/{model_name}/plots/dummy/exp{experiment_id}'
        loss2 = 0.0
        for i,x in enumerate(test_loader):
            img = x.detach().cpu().numpy().squeeze()
            x_hat, _, _ = vae(x)
            valid_pixels = torch.where(x > 0, torch.ones_like(x), torch.zeros_like(x)) # Mask is defined for the whole batch (x)
            MSE_loss_ = F.mse_loss(x_hat, x, reduction='none') * valid_pixels
            recon_loss = torch.mean(torch.sum(MSE_loss_))
            loss2 += recon_loss.item() # just do reconstruction loss for now
            #if i < args.num_examples:
                #plotting.reconstruct_and_plot(x, vae, model_name, experiment_id, savepath2, i, cmap='magma', save=True)

        print(f'Average reconstruction loss for realsense data: {loss/len(realsense_loader.dataset)}')
        print(f'Average reconstruction loss for test data: {loss2/len(test_loader.dataset)}')
        
        if "reconstructions" in args.plot:
            savepath_recon = f'results/{model_name}/plots/reconstructions/exp{experiment_id}'
            os.makedirs(savepath_recon, exist_ok=True)
            for i, x in enumerate(test_loader):
                if i == args.num_examples: break
                img = x.to(device)
                plotting.reconstruct_and_plot(img, vae, model_name, experiment_id, savepath_recon, i, cmap='magma', save=True)
                
        if "kde" in args.plot:
            savepath_kde = f'results/{model_name}/plots/kde/exp{experiment_id}_realsense'
            os.makedirs(savepath_kde, exist_ok=True)
            
            combos_to_test = [(0, 1)]#, (1, 2), (0, 2), (0, 3), (1, 3), (2, 3), (0, 4), (1, 4), (2, 4), (3, 4)]
            plotting.latent_space_kde(model=vae, 
                                      dataloader=realsense_loader, 
                                      latent_dim=LATENT_DIMS, 
                                      name=model_name,
                                      path = savepath_kde,
                                      save=True, combos=combos_to_test)
            
            savepath_kde = f'results/{model_name}/plots/kde/exp{experiment_id}_test'
            os.makedirs(savepath_kde, exist_ok=True)
            plotting.latent_space_kde(model=vae,
                                        dataloader=test_loader,
                                        latent_dim=LATENT_DIMS,
                                        name=model_name,
                                        path=savepath_kde,
                                        save=True, combos=combos_to_test)
            
        









if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train β-VAE on a dataset of depth images')
    parser.add_argument('mode',
                        help='Mode of operation: train or test',
                        type=str,
                        choices=['train', 'test'],
                        default='train')
    
    parser.add_argument('--plot',
                        help='Plotting mode: losses or reconstructions',
                        type=str,
                        choices=['losses', 
                                 'reconstructions',
                                 'latent_dims_sweep',
                                 'kde'],
                        nargs="+",
                        default=['losses'])
    
    parser.add_argument('--num_examples',
                        help='Number of examples to plot reconstructions for',
                        type=int,
                        default=5)
    
    parser.add_argument('--model_name',
                        type=str,
                        choices=['conv1',
                                 'vgg16',
                                 'resnet50'],
                        help='Name of model to test',
                        default='conv1')
    
    parser.add_argument('--exp_id',
                        type=int,
                        help='Experiment id of model to train or test',
                        default=69)
    
    parser.add_argument('--num_seeds',
                        type=int,
                        help='Number of seeds to train across',
                        default=1)
    
    parser.add_argument('--seed',
                        type=int,
                        help='Only for testing: seed of model to test. Used to extract correct model from file',
                        default=1)
    
    parser.add_argument('--save_model',
                        help='Save model to file',
                        type=bool,
                        default=True)
    
    parser.add_argument('--batch_size',
                        help='Batch size',
                        type=int,
                        default=64)
    
    parser.add_argument('--epochs',
                        help='Number of epochs',
                        type=int,
                        default=25)
    
    parser.add_argument('--latent_dims',
                        help='Latent dimensions',
                        type=int,
                        default=64)
    
    parser.add_argument('--learning_rate',
                        help='Learning rate',
                        type=float,
                        default=0.001)
    
    parser.add_argument('--beta',
                        help='β value',
                        type=float,
                        default=1)

    args = parser.parse_args()
    try:
        main(args)
    except KeyboardInterrupt:
        print('\n\nKeyboard interrupt detected, exiting.')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
    finally:
        print('Done.')
