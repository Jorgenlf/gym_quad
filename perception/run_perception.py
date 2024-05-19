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
import matplotlib.pyplot as plt
from utils_perception.data_reader import CustomDepthDataset, RealSenseDataset, RealSenseDataset_v2, DataReaderRealSensev2, DataReaderSynthetic
from utils_perception import plotting
import cv2

#from utils_perception.data_augmentation import DataAugmentation

from VAE.encoders import ConvEncoder1, ConvEncoder2, VGG16Encoder, ResNet50Encoder
from VAE.decoders import ConvDecoder1, ConvDecoder2, _ConvDecoder2
from VAE.vae import VAE

from train_perception import TrainerVAE

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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
    print(f'Using device: {device}')

    # The flag below controls whether to allow TF32 on matmul. This flag defaults to False
    # in PyTorch 1.12 and later.
    torch.backends.cuda.matmul.allow_tf32 = True
    # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
    torch.backends.cudnn.allow_tf32 = True

    
    print('Preparing data...\n') # Do this on any mode

    # Get SUN RGBD dataset
    #print('Getting SUN RGBD dataset')
    #sun = SunRGBD(orig_data_path="data/sunrgbd_stripped", data_path_depth="data/sunrgbd_images_depth", data_path_rgb="data/sunrgbd_images_rgb")
    

    # Define transformatons additional to the normalization and channel conversion done in the DataReader
    train_additional_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.15),
        transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.5, 1.5)),
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


    dataloader_rs = DataReaderRealSensev2(path_img_depth="data/realsense_v2_depth",
                            batch_size=BATCH_SIZE,
                            train_test_split=0.7,
                            train_val_split=0.3,
                            transforms_train=valid_additional_transform,
                            transforms_validate=valid_additional_transform)
    
    # dataloader_synthetic = DataReaderSynthetic(path_depth="data/synthetic_depthmaps",
    #                         batch_size=BATCH_SIZE,
    #                         train_test_split=0.7,
    #                         train_val_split=0.3,
    #                         transforms_train=valid_additional_transform,
    #                         transforms_validate=valid_additional_transform)
    
    dataloader_combined = DataReaderRealSensev2(path_img_depth="data/all_imgs",
                            batch_size=BATCH_SIZE,
                            train_test_split=0.7,
                            train_val_split=0.3,
                            transforms_train=valid_additional_transform,
                            transforms_validate=valid_additional_transform)
                            
    
    train_loader_rs, val_loader_rs, test_loader_rs = dataloader_rs.load_split_data_realsense(seed=None, shuffle=True)
    #train_loader_synthetic, val_loader_synthetic, test_loader_synthetic = dataloader_synthetic.load_split_data_synthetic(seed=None, shuffle=True)
    train_loader_combined, val_loader_combined, test_loader_combined = dataloader_combined.load_split_data_realsense(seed=None, shuffle=True)




    #print(f'RealSense Data loaded\nSize train: {len(train_loader_rs.dataset)} | Size validation: {len(val_loader_rs.dataset)} | Size test: {len(test_loader_rs.dataset)}\n')

    # Augment data
    #data_augmentation = DataAugmentation()
    #dataloader_train, dataloader_validate, dataloader_test = data_augmentation.augment_data(dataloader_train, dataloader_validate, dataloader_test)
    
    model_name = args.model_name
    experiment_id = args.exp_id

    if args.mode == 'train':
        #torch.autograd.set_detect_anomaly(True)
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
                #train_loader, val_loader, test_loader = dataloader_sun.load_split_data_sunrgbd(sun, seed=None, shuffle=True)
                #train_loader, val_loader, test_loader = dataloader_rs.load_split_data_realsense(seed=None, shuffle=True)
                train_loader, val_loader, test_loader = dataloader_combined.load_split_data_realsense(seed=None, shuffle=True)
                print('Data loaded')
                print(f'Size train: {len(train_loader.dataset)} | Size validation: {len(val_loader.dataset)} | Size test: {len(test_loader.dataset)}\n')

                # Create VAE based on args.model_name
                if model_name == 'conv1':
                    encoder = ConvEncoder1(image_size=IMG_SIZE, channels=NUM_CHANNELS, latent_dim=LATENT_DIMS).to(device)
                    decoder = ConvDecoder1(image_size=IMG_SIZE, channels=NUM_CHANNELS, latent_dim=LATENT_DIMS, flattened_size=encoder.flattened_size, dim_before_flatten=encoder.dim_before_flatten).to(device)
                    vae = VAE(encoder, decoder, LATENT_DIMS, BETA).to(device)
                if model_name == 'conv2':
                    encoder = ConvEncoder2(image_size=IMG_SIZE, in_chan=NUM_CHANNELS, latent_dim=LATENT_DIMS)
                    decoder = _ConvDecoder2(image_size=IMG_SIZE, channels=NUM_CHANNELS, latent_dim=LATENT_DIMS, flattened_size=encoder.flattened_size, dim_before_flatten=encoder.dim_before_flatten)
                    vae = VAE(encoder, decoder, LATENT_DIMS, BETA).to(device)
                if model_name == 'vgg16':
                    decoder_intermediate_size = torch.Size([1,256,14,14])
                    decoder_flattened_size = 50176
                    encoder = VGG16Encoder(latent_dim=LATENT_DIMS, image_size=IMG_SIZE)
                    decoder = ConvDecoder1(image_size=IMG_SIZE, channels=NUM_CHANNELS, latent_dim=LATENT_DIMS, flattened_size=decoder_flattened_size, dim_before_flatten=decoder_intermediate_size)
                    vae = VAE(encoder, decoder, LATENT_DIMS, BETA).to(device)
                if model_name == 'resnet50':
                    decoder_intermediate_size = torch.Size([1,256,14,14])
                    decoder_flattened_size = 50176
                    encoder = ResNet50Encoder(latent_dim=LATENT_DIMS, image_size=IMG_SIZE)
                    decoder = ConvDecoder1(image_size=IMG_SIZE, channels=NUM_CHANNELS, latent_dim=LATENT_DIMS, flattened_size=decoder_flattened_size, dim_before_flatten=decoder_intermediate_size)
                    vae = VAE(encoder, decoder, LATENT_DIMS, BETA).to(device)

                print(f'Number of parameters in model: {count_parameters(vae)}')

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
                
                trained_epochs = trainer.train(early_stopping=True)
                    
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
            total_val_losses_for_betas = [] 
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
                
                
                seed = 42 # Logic regarding seeding must be changed if many seeds used
                for i in range(NUM_SEEDS):
                    # Load data with different seed
                    seed += 2*i
                    print(f'Seed {i}/{NUM_SEEDS}')
                    #train_loader, val_loader, test_loader = dataloader_sun.load_split_data_sunrgbd(sun, seed=seed, shuffle=True)
                    train_loader, val_loader, test_loader = dataloader_combined.load_split_data_realsense(seed=seed, shuffle=True)

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
                    
                    trained_epochs = trainer.train(early_stopping=True)
                    
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
                        vae.encoder.save(path=os.path.join(savepath_encoder, f'encoder_{model_name}_experiment_{experiment_id}_seed{i}_dim{l}.json'))
                        vae.decoder.save(path=os.path.join(savepath_decoder, f'decoder_{model_name}_experiment_{experiment_id}_seed{i}_dim{l}.json'))

                    del train_loader, val_loader, encoder, decoder, vae, optimizer, trainer
                
                total_val_losses_for_betas.append(total_val_losses)
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
            total_losses = [total_train_losses_for_latent_dims, total_val_losses_for_betas]
            bce_losses = [bce_train_losses_for_latent_dims, bce_val_losses_for_latent_dims]
            kl_losses = [kl_train_losses_for_latent_dims, kl_val_losses_for_latent_dims]
            
            plotting.plot_loss_ldim_sweep(total_losses=total_losses,
                                        BCE_losses=bce_losses,
                                        KL_losses=kl_losses,
                                        labels=labels,
                                        path=savepath_ldim_plot,
                                        save=True,
                                        include_train=False)
            
            
        if 'beta_and_latent_sweep' in args.plot:
            print('Beta and latent dim sweep test...')
            betas = [0.01, 0.05, 0.1, 0.5, 1, 1.5, 2, 4, 8, 16, 32, 64, 128, 256, 512]
            betas = [0.1, 0.5, 1, 1.5, 2, 4, 8, 16, 32, 64, 128, 256, 512]
            latent_dims = [16, 32, 64, 128]#[2, 4, 8, 16, 32, 64, 128]

            """
            for l in latent_dims:
                print(f'Latent dimension: {l}')
            
                # Containers w/ loss trajs for each latent dim
                total_val_losses_for_betas = [] 
                bce_val_losses_for_betas = []   
                kl_val_losses_for_betas = []
                total_train_losses_for_betas = []
                bce_train_losses_for_betas = []
                kl_train_losses_for_betas = []
                
                for b in betas:
                    print(f'Beta: {b}')
                    total_train_losses = np.full((NUM_SEEDS, N_EPOCH), np.nan)
                    total_val_losses = np.full((NUM_SEEDS, N_EPOCH), np.nan)
                    bce_train_losses = np.full((NUM_SEEDS, N_EPOCH), np.nan)
                    bce_val_losses = np.full((NUM_SEEDS, N_EPOCH), np.nan)
                    kl_train_losses = np.full((NUM_SEEDS, N_EPOCH), np.nan)
                    kl_val_losses = np.full((NUM_SEEDS, N_EPOCH), np.nan)
                    
                    
                    seed = 42 # Logic regarding seeding must be changed if many seeds used
                    for i in range(NUM_SEEDS): 
                        # Load data with different seed
                        seed += i
                        print(f'Seed {i}/{NUM_SEEDS}')
                        #train_loader, val_loader, test_loader = dataloader_sun.load_split_data_sunrgbd(sun, seed=seed, shuffle=True)
                        train_loader, val_loader, test_loader = dataloader_rs.load_split_data_realsense(seed=seed, shuffle=True)


                        # Create VAE based on args.model_name
                        if model_name == 'conv1':
                            encoder = ConvEncoder1(image_size=IMG_SIZE, channels=NUM_CHANNELS, latent_dim=l)
                            decoder = ConvDecoder1(image_size=IMG_SIZE, channels=NUM_CHANNELS, latent_dim=l, flattened_size=encoder.flattened_size, dim_before_flatten=encoder.dim_before_flatten)
                            vae = VAE(encoder, decoder, l, b).to(device)

                        # Train model
                        optimizer = Adam(vae.parameters(), lr=LEARNING_RATE)
                        trainer = TrainerVAE(model=vae, 
                                            epochs=N_EPOCH, 
                                            learning_rate=LEARNING_RATE, 
                                            batch_size=BATCH_SIZE, 
                                            dataloader_train=train_loader, 
                                            dataloader_val=val_loader, 
                                            optimizer=optimizer, 
                                            beta=b,
                                            reconstruction_loss="MSE")
                        
                        trained_epochs = trainer.train(early_stopping=False)
                        
                        # Only insert to the first trained_epochs elements if early stopping has been triggered
                        total_train_losses[i,:trained_epochs] = trainer.training_loss['Total loss']
                        total_val_losses[i,:trained_epochs] = trainer.validation_loss['Total loss']
                        bce_train_losses[i,:trained_epochs] = trainer.training_loss['Reconstruction loss']
                        bce_val_losses[i,:trained_epochs] = trainer.validation_loss['Reconstruction loss']
                        
                        kl_train_losses[i,:trained_epochs] = trainer.training_loss['KL divergence loss']
                        kl_val_losses[i,:trained_epochs] = trainer.validation_loss['KL divergence loss']
                        
                        # Normalize KL losses by dividing by beta so that losses are comparable across betas
                        kl_train_losses[i,:trained_epochs] = kl_train_losses[i,:trained_epochs] / b
                        kl_val_losses[i,:trained_epochs] = kl_val_losses[i,:trained_epochs] / b
                        
                        # Save encoder and decoder parameters to file
                        savepath_encoder = f'models/encoders'
                        savepath_decoder = f'models/decoders'
                        os.makedirs(savepath_encoder, exist_ok=True)
                        os.makedirs(savepath_decoder, exist_ok=True)

                        if args.save_model:
                            vae.encoder.save(path=os.path.join(savepath_encoder, f'encoder_{model_name}_experiment_{experiment_id}_seed{seed}_dim{l}_beta{b}.json'))
                            vae.decoder.save(path=os.path.join(savepath_decoder, f'decoder_{model_name}_experiment_{experiment_id}_seed{seed}_dim{l}_beta{b}.json'))

                        del train_loader, val_loader, encoder, decoder, vae, optimizer, trainer
                    
                    total_val_losses_for_betas.append(total_val_losses)
                    bce_val_losses_for_betas.append(bce_val_losses)
                    kl_val_losses_for_betas.append(kl_val_losses)
                    total_train_losses_for_betas.append(total_train_losses)
                    bce_train_losses_for_betas.append(bce_train_losses)
                    kl_train_losses_for_betas.append(kl_train_losses)
                    
                    # Save loss trajectories to file, so specific loss trajs can be plotted later on
                    savepath_loss_numerical = f'results/{model_name}/numerical/losses/exp{experiment_id}/latent_dim_{l}/beta_{b}'
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

                    savepath_loss_plot = f'results/{model_name}/plots/losses/exp{experiment_id}/latent_dim_{l}/beta_{b}'
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
                labels = [f'β = {b}' for b in betas]
                total_losses = [total_train_losses_for_betas, total_val_losses_for_betas]
                bce_losses = [bce_train_losses_for_betas, bce_val_losses_for_betas]
                kl_losses = [kl_train_losses_for_betas, kl_val_losses_for_betas]
                
                
                # Using same function as ldim sweep but for betas :)
                plotting.plot_loss_ldim_sweep(total_losses=total_losses,
                                            BCE_losses=bce_losses,
                                            KL_losses=kl_losses,
                                            labels=labels,
                                            path=savepath_ldim_plot,
                                            save=True,
                                            include_train=False)"""
            
            
            # Now run tests for different betas and latent dims
            test_errors = np.zeros((NUM_SEEDS, len(latent_dims), len(betas))) #rows = latent dims, cols = betas
            for i, l in enumerate(latent_dims):
                for j, b in enumerate(betas):
                    seed = 42
                    for s in range(NUM_SEEDS):
                        # Load model for l+b combo
                        seed += s
                        _,_,test_loader = dataloader_rs.load_split_data_realsense(seed=seed, shuffle=True)
                        name = f'{model_name}_experiment_{experiment_id}_seed{seed}_dim{l}_beta{b}.json'
                        encoder = ConvEncoder1(image_size=IMG_SIZE, channels=NUM_CHANNELS, latent_dim=l)
                        encoder.load(f"models/encoders/encoder_{name}")
                        decoder = ConvDecoder1(image_size=IMG_SIZE, channels=NUM_CHANNELS, latent_dim=l, flattened_size=encoder.flattened_size, dim_before_flatten=encoder.dim_before_flatten)
                        decoder.load(f"models/decoders/decoder_{name}")
                        vae = VAE(encoder, decoder, l, b).to(device)
                        
                        # test on test set
                        loss = 0.0
                        for _, x in enumerate(test_loader):
                            img = x.detach().cpu().numpy().squeeze()
                            x = x.to(device)
                            x_hat, mu, logvar = vae(x)
                            valid_pixels = torch.where(x > 0, torch.ones_like(x), torch.zeros_like(x))
                            MSE_loss_ = F.mse_loss(x_hat, x, reduction='none') * valid_pixels
                            recon_loss = torch.mean(torch.sum(MSE_loss_))
                            KLD_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                            KDL_loss_scaled = KLD_loss / b
                            loss += recon_loss.item() + KDL_loss_scaled.item()
                        
                        test_errors[s,i,j] = loss/len(test_loader.dataset) # len is 1 for realsense tets data
                        del encoder, decoder, vae
            ci_matrix = np.zeros((len(latent_dims), len(betas)))
            for i in range(len(latent_dims)):
                for j in range(len(betas)):
                    ci_matrix[i,j] = 1.96 * np.std(test_errors[:,i,j]) / np.sqrt(NUM_SEEDS)
            avg_matrix = np.mean(test_errors, axis=0)
            
            # Plot test errors with beta on x axis and loss on y axis, different latent dim is own line
            savepath_test_errors = f'results/{model_name}/plots/test_errors/exp{experiment_id}'
            os.makedirs(savepath_test_errors, exist_ok=True)
            
            plt.style.use('ggplot')
            plt.rc('font', family='serif')
            plt.rc('xtick', labelsize=12)
            plt.rc('ytick', labelsize=12)
            plt.rc('axes', labelsize=12)
            
            labels = [f'Latent dim = {l}' for l in latent_dims]
            plt.figure(figsize = (20,15))
            for l, i in enumerate(latent_dims):
                # fill between
                ln_betas = np.log(betas)
                plt.fill_between(ln_betas, avg_matrix[l,:]-ci_matrix[l,:], avg_matrix[l,:]+ci_matrix[l,:], alpha=0.2)
                plt.plot(ln_betas, avg_matrix[l,:], label=labels[l])
            plt.xlabel('ln(β)')
            plt.ylabel('Test error (β-normalized)')
            plt.legend()
            plt.savefig(f'{savepath_test_errors}/test_errors.pdf', bbox_inches='tight')
            
                          
     
                
    if args.mode == 'test':
        seed = args.seed
        #full_name = f'{model_name}_experiment_{experiment_id}_seed{seed}_dim{LATENT_DIMS}'#_beta{int(BETA)}'
        full_name = f'{model_name}_experiment_{experiment_id}_seed{seed}'

        # Load model for testing
        if model_name == 'conv1':
            encoder = ConvEncoder1(image_size=IMG_SIZE, channels=NUM_CHANNELS, latent_dim=LATENT_DIMS)
            encoder.load(f"models/encoders/encoder_{full_name}.json")
            decoder = ConvDecoder1(image_size=IMG_SIZE, channels=NUM_CHANNELS, latent_dim=LATENT_DIMS, flattened_size=encoder.flattened_size, dim_before_flatten=encoder.dim_before_flatten)
            decoder.load(f"models/decoders/decoder_{full_name}.json")
            vae = VAE(encoder, decoder, LATENT_DIMS, BETA).to(device)
        if model_name == 'conv2':
            encoder = ConvEncoder2(image_size=IMG_SIZE, in_chan=NUM_CHANNELS, latent_dim=LATENT_DIMS)
            encoder.load(f"models/encoders/encoder_{full_name}.json")
            decoder = _ConvDecoder2(image_size=IMG_SIZE, channels=NUM_CHANNELS, latent_dim=LATENT_DIMS, flattened_size=encoder.flattened_size, dim_before_flatten=encoder.dim_before_flatten)
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

        #savepath_latent_dist = f'results/{model_name}/plots/latent_distributions/exp{experiment_id}'
        #os.makedirs(savepath_latent_dist, exist_ok=True)
        #plotting.plot_latent_distributions(model=vae, dataloader=test_loader_rs, model_name='conv1', device=device, save=True, savepath=savepath_latent_dist)
        """encoder = ConvEncoder1(image_size=IMG_SIZE, channels=NUM_CHANNELS, latent_dim=LATENT_DIMS)
        encoder.load(f"models/encoders/encoder_{full_name}.json")
        decoder = ConvDecoder1(image_size=IMG_SIZE, channels=NUM_CHANNELS, latent_dim=LATENT_DIMS, flattened_size=encoder.flattened_size, dim_before_flatten=encoder.dim_before_flatten)
        decoder.load(f"models/decoders/decoder_{full_name}.json")
        vae = VAE(encoder, decoder, LATENT_DIMS, BETA).to(device)"""
        """
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
        print(f'Average reconstruction loss for test data: {loss2/len(test_loader.dataset)}')"""
        
        if "reconstructions" in args.plot:
            savepath_recon = f'results/{model_name}/plots/reconstructions/exp{experiment_id}/latent_dim_{LATENT_DIMS}/'
            os.makedirs(savepath_recon, exist_ok=True)
            for i, x in enumerate(test_loader_combined):
                if i == args.num_examples: break
                img = x.to(device)
                plotting.reconstruct_and_plot(img, vae, model_name, experiment_id, savepath_recon, i, cmap='magma', save=True, save_input=True)
            
            savepath_recon = f'results/{model_name}/plots/reconstructions/exp{experiment_id}/latent_dim_{LATENT_DIMS}_realsense/'
            os.makedirs(savepath_recon, exist_ok=True)
            for i,x in enumerate(test_loader_rs):
                if i == args.num_examples: break
                img = x.to(device)
                plotting.reconstruct_and_plot(img, vae, model_name, experiment_id, savepath_recon, i, cmap='magma', save=True, save_input=False)
                
        if "kde" in args.plot:
            savepath_kde = f'results/{model_name}/plots/kde/exp{experiment_id}'
            os.makedirs(savepath_kde, exist_ok=True)
            
            combos_to_test = [(0, 1), (1, 2), (0, 2)]#, (0, 3), (1, 3), (2, 3)]#, (0, 4), (1, 4), (2, 4), (3, 4)]
            #combos_to_test = [(0,4), (1,4), (2,4), (3,4), (5,4), (5,9), (7,6), (8,9), (8,7), (9,7), (13,2), (12,3), (11,4), (10,5), (9,6), (8,7), (7,8), (6,9), (5,10), (4,11), (3,12), (2,13), (1,14), (0,15)]
            plotting.latent_space_kde(model=vae, 
                                      dataloader=test_loader_combined, 
                                      latent_dim=LATENT_DIMS, 
                                      name=model_name,
                                      path = savepath_kde,
                                      save=True, combos=combos_to_test)
            
            #savepath_kde = f'results/{model_name}/plots/kde/exp{experiment_id}_test'
            #os.makedirs(savepath_kde, exist_ok=True)
            #plotting.latent_space_kde(model=vae,
            #                            dataloader=test_loader,
            #                            latent_dim=LATENT_DIMS,
            #                            name=model_name,
            #                            path=savepath_kde,
            #                            save=True, combos=combos_to_test)
            
        if "feature_maps" in args.plot:
            savepath_feature_maps = f'results/{model_name}/plots/feature_maps'
            os.makedirs(savepath_feature_maps, exist_ok=True)
            #input_img = torch.rand(1,1, IMG_SIZE, IMG_SIZE).to(device)
            #input_img = next(iter(test_loader_rs)).to(device)
            # get 10 random samples from test loasder and generate feature maps for them
            for i in range(25):
                input_img = next(iter(test_loader_rs)).to(device)
                plotting.visualize_feature_maps(encoder=encoder,input_image=input_img, savepath=savepath_feature_maps, ending=str(i))
                             
            #plotting.visualize_feature_maps(encoder=encoder,input_image=input_img, savepath=savepath_feature_maps)
        
        if "filters" in args.plot:
            savepath_filters = f'results/{model_name}/plots/filters'
            os.makedirs(savepath_filters, exist_ok=True)
            
            for i in range(25):
                img = next(iter(test_loader_rs)).to(device)
                img = None
                plt.style.use('ggplot')
                plt.rc('font', family='serif')
                plt.rc('xtick', labelsize=12)
                plt.rc('ytick', labelsize=12)
                plt.rc('axes', labelsize=12)
                
                #plt.figure(figsize=(20, 10))
                #img_plt = img.detach().cpu().numpy().squeeze()
                #plt.imshow(img_plt, cmap='gray')
                #plt.axis("off")
                #plt.savefig(f"{savepath_filters}/input_img_{i}.pdf", bbox_inches='tight')
                
                plotting.visualize_filters(encoder=encoder, savepath=savepath_filters, input_image=img, ending=str(i))
        
        if 'activation_maximization' in args.plot:
            savepath_act_max = f'results/{model_name}/plots/activation_maximization'
            os.makedirs(savepath_act_max, exist_ok=True)
            if model_name == 'vgg16':
                encoder = encoder.vgg16.features
            if model_name == 'conv1':
                encoder = encoder.conv_block
            
            for i in range(256):
                am = plotting.ActivationMaximization(encoder=encoder, epochs=2000, cnn_layer=3, cnn_filter=i)
                am.visualize_activation_maximization(savepath=savepath_act_max)
        
        if 'interpolate' in args.plot:
            savepath_interpolate = f'results/{model_name}/plots/interpolation/'
            os.makedirs(savepath_interpolate, exist_ok=True)
            # Get two random images from test_loader
            for i in range(10):
                savepath_i = savepath_interpolate + f'interpolation_{i}'
                i,j = np.random.randint(0, len(test_loader_rs)), np.random.randint(0, len(test_loader_rs))
                img1 = test_loader_rs.dataset[i].unsqueeze(0).to(device)
                img2 = test_loader_rs.dataset[j].unsqueeze(0).to(device)
                img1 = next(iter(test_loader_rs)).to(device)
                img2 = next(iter(test_loader_rs)).to(device)
                plotting.interpolate(autoencoder=vae, x_1=img1, x_2=img2, n=10, savepath=savepath_i)
                plt.imshow(img1.detach().cpu().numpy().squeeze(), cmap='magma')
                plt.axis('off')
                plt.savefig(f'{savepath_i}_img1.pdf', bbox_inches='tight')
                plt.imshow(img2.detach().cpu().numpy().squeeze(), cmap='magma')
                plt.axis('off')
                plt.savefig(f'{savepath_i}_img2.pdf', bbox_inches='tight')

    
            
        




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
                                 'beta_and_latent_sweep',
                                 'kde',
                                 'feature_maps',
                                 'filters',
                                 'activation_maximization',
                                 'interpolate'],
                        nargs="+",
                        default=['losses'])
    
    parser.add_argument('--num_examples',
                        help='Number of examples to plot reconstructions for',
                        type=int,
                        default=5)
    
    parser.add_argument('--model_name',
                        type=str,
                        choices=['conv1',
                                 'conv2',
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
