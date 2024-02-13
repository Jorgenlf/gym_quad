import torch
from torch.optim import Adam
import torch.nn as nn
import os
import sys
import numpy as np
import argparse

from utils_perception.data_reader import Dataloader

#from utils_perception.data_augmentation import DataAugmentation

from VAE.encoders import ConvEncoder1
from VAE.decoders import ConvDecoder1
from VAE.vae import VAE

from train_perception import TrainerVAE


# HYPERPARAMS
LEARNING_RATE = 0.001
N_EPOCH = 25
BATCH_SIZE = 64     
LATENT_DIMS = 12

def main(args):
    """# Set hyperparameters
    BATCH_SIZE    = args.batch_size     # Default: 64
    N_EPOCH       = args.epochs         # Default: 25
    LATENT_DIMS   = args.latent_dims    # Default: 12
    LEARNING_RATE = args.learning_rate  # Default: 0.001
    NUM_SEEDS     = args.num_seeds      # Default: 1
    BETA          = args.beta           # Default: 1
    EPS_WEIGHT    = args.eps_weight     # Default: 1
    """

    BATCH_SIZE    = 64     # Default: 64
    N_EPOCH       = 5       # Default: 25
    LATENT_DIMS   = 12  # Default: 12
    LEARNING_RATE = 0.001  # Default: 0.001
    NUM_SEEDS     = 1    # Default: 1
    BETA          = 1       # Default: 1
    EPS_WEIGHT    = 1
    

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Load data
    data_path = ""
    dataloader = Dataloader()
    dataloader_train, dataloader_validate, dataloader_test = dataloader.load_data(data_path = data_path,
                                                                                  batch_size = BATCH_SIZE,
                                                                                  train_test_split=0.7,
                                                                                  train_validate_split=0.3,
                                                                                  shuffle=True)
    
    # Augment data
    #data_augmentation = DataAugmentation()
    #dataloader_train, dataloader_validate, dataloader_test = data_augmentation.augment_data(dataloader_train, dataloader_validate, dataloader_test)


    # Load model
    image_size = dataloader.image_size
    channels = dataloader.channels
    
    encoder = ConvEncoder1(image_size=image_size, channels=channels, latent_dim=LATENT_DIMS)
    decoder = ConvDecoder1(image_size=image_size, channels=channels, latent_dim=LATENT_DIMS, flattened_size=encoder.flattened_size)
    vae = VAE(encoder, decoder, LATENT_DIMS, BETA).to(device)

    # Train model
    optimizer = Adam(vae.parameters(), lr=LEARNING_RATE)
    trainer = TrainerVAE(model=vae, 
                         epochs=N_EPOCH, 
                         learning_rate=LEARNING_RATE, 
                         batch_size=BATCH_SIZE, 
                         dataloader_train=dataloader_train, 
                         dataloader_val=dataloader_validate, 
                         optimizer=optimizer, 
                         beta=BETA)
    trainer.train()
    






if __name__ == '__main__':
    main()
"""

    parser = argparse.ArgumentParser(description='Train a β-VAE on a dataset of depth images')
    parser.add_argument('mode',
                        help='Mode of operation: train or test',
                        choices=['train', 'test'])

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
        print('Done.')"""