import torch
from torch.optim import Adam
import torch.nn as nn
import os
import sys
import numpy as np
import argparse
import torchvision
import torchvision.transforms as transforms

from utils_perception.data_reader import DataReader, SunRGBD, CustomDepthDataset

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
        transforms.Lambda(lambda x: torch.clamp(x, 0, 1)),  # Clamping values to [0, 1] due to some pixels exceeding 1 (with 10^-6) when resizing and interpolating
        transforms.RandomHorizontalFlip(p=0.5),
        #transforms.RandomVerticalFlip(p=0.25),
        #transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.5, 1.5)),
        #transforms.RandomRotation(degrees=(30, 70)),
    ])

    valid_additional_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.Lambda(lambda x: torch.clamp(x, 0, 1))
    ])
    
    print(f'Loading data with image size = ({NUM_CHANNELS}, {IMG_SIZE}, {IMG_SIZE}), batch size = {BATCH_SIZE}, train-test split = {0.7} and train-val split = {0.3}')
    print(f'Additional transformations to training data:\n {train_additional_transform}')
    dataloader = DataReader(path_img_depth="data/sunrgbd_images_depth",
                            path_img_rgb="data/sunrgbd_images_rgb",
                            batch_size=BATCH_SIZE,
                            train_test_split=0.7,
                            train_val_split=0.3,
                            transforms_train=train_additional_transform,
                            transforms_validate=valid_additional_transform)
    

    # Load data and create dataloaders
    train_loader, val_loader, test_loader = dataloader.load_split_data_sunrgbd(sun, shuffle=True)

    print('Data loaded')
    print(f'Size train: {len(train_loader.dataset)} | Size validation: {len(val_loader.dataset)} | Size test: {len(test_loader.dataset)}\n')

    #print(train_loader.dataset[0].shape)
    #print(train_loader.dataset[0][0].shape)
    #dislay an image
    #import matplotlib.pyplot as plt
    #import numpy as np
    # Iterate through first 9 images and create a figure with a 9x9 grid, save the figure to a file
    #for i in range(9):
    #    plt.subplot(330 + 1 + i)
    #    plt.imshow(train_loader.dataset[i][0].numpy())
    #plt.show()
    

    # Augment data
    #data_augmentation = DataAugmentation()
    #dataloader_train, dataloader_validate, dataloader_test = data_augmentation.augment_data(dataloader_train, dataloader_validate, dataloader_test)
    """
    for x in train_loader:
        imgs = x.cpu().detach().numpy()
        for i, im in enumerate(imgs):
            im = np.squeeze(im)
            if np.any(im > 1):
                print(im.max())
            elif np.any(im < 0.0000001):
                print(im.min())
    """



    encoder = ConvEncoder1(image_size=IMG_SIZE, channels=NUM_CHANNELS, latent_dim=LATENT_DIMS)
    decoder = ConvDecoder1(image_size=IMG_SIZE, channels=NUM_CHANNELS, latent_dim=LATENT_DIMS, flattened_size=encoder.flattened_size)
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







if __name__ == '__main__':
    
    main()
    """
    parser = argparse.ArgumentParser(description='Train a β-VAE on a dataset of depth images')
    parser.add_argument('mode',
                        help='Mode of operation: dataprep, train or test',
                        choices=['dataprep','train', 'test'])

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
    """