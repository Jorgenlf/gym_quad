import torch
from torch.optim import Adam
import torch.nn as nn
import os
import sys
import numpy as np
import argparse
import torchvision
import torchvision.transforms as transforms

from utils_perception.data_reader import sjef, SunRGBD

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

def main():
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
    LATENT_DIMS   = 50  # Default: 12
    LEARNING_RATE = 0.001  # Default: 0.001
    NUM_SEEDS     = 1    # Default: 1
    BETA          = 1       # Default: 1
    EPS_WEIGHT    = 1
    

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Get SUN RGBD dataset
    sun = SunRGBD(orig_data_path="data/sunrgbd_stripped", data_path_depth="data/sunrgbd_images_depth", data_path_rgb="data/sunrgbd_images_rgb")
    
    # Define transformations

    # the training transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        #transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
        #transforms.RandomRotation(degrees=(30, 70)),
        transforms.ToTensor(),
        #transforms.Normalize(
         #   mean=[0.5],
          #  std=[0.5]
        #)
    ])
    # the validation transforms
    valid_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5],
            std=[0.5]
        )
    ])
    
    
    dataloader = sjef(path_img_depth="data/sunrgbd_images_depth",
                            path_img_rgb="data/sunrgbd_images_rgb",
                            batch_size=BATCH_SIZE,
                            train_test_split=0.7,
                            train_val_split=0.3,
                            transforms_train=train_transform,
                            transforms_validate=valid_transform)
    

    # Load data and create dataloaders
    train_loader, val_loader, test_loader = dataloader.load_data_sunrgbd(sun)
    
    print(train_loader.dataset[0])
    #dislay an image
    import matplotlib.pyplot as plt
    import numpy as np
    train_features, train_labels = next(iter(train_loader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")
    img = train_features[0].squeeze()
    label = train_labels[0]
    plt.imsave('test.png', img, cmap='gray')

    # Augment data
    #data_augmentation = DataAugmentation()
    #dataloader_train, dataloader_validate, dataloader_test = data_augmentation.augment_data(dataloader_train, dataloader_validate, dataloader_test)

"""
    # Load model
    image_size = 224
    channels = 1
    
    encoder = ConvEncoder1(image_size=image_size, channels=channels, latent_dim=LATENT_DIMS)
    decoder = ConvDecoder1(image_size=image_size, channels=channels, latent_dim=LATENT_DIMS, flattened_size=encoder.flattened_size)
    vae = VAE(encoder, decoder, LATENT_DIMS, BETA).to(device)

    # Train model
    optimizer = Adam(vae.parameters(), lr=LEARNING_RATE)
    trainer = TrainerVAE(model=vae, 
                         epochs=N_EPOCH, 
                         learning_rate=LEARNING_RATE, 
                         batch_size=BATCH_SIZE, 
                         dataloader_train=train_loader, 
                         dataloader_val=val_loader, 
                         optimizer=optimizer, 
                         beta=BETA)
    trainer.train()
"""






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