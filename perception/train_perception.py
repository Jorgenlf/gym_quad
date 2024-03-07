import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class TrainerVAE():
    """Trainer class for β-VAE"""
    def __init__(self,
                 model:nn.Module,
                 epochs:int,
                 learning_rate:float,
                 batch_size:int,
                 dataloader_train:torch.utils.data.DataLoader,
                 dataloader_val:torch.utils.data.DataLoader,
                 optimizer:torch.optim.Optimizer,
                 beta:float=1,
                 reconstruction_loss:str='MSE',
                 patience:int=5,                 
                 ) -> None:

        # General parameters
        self.model = model
        self.epochs = epochs                            # Number of epochs to train the model
        self.learning_rate = learning_rate              # Learning rate for the optimizer
        self.batch_size = batch_size                    # Batch size for the dataloaders
        self.dataloader_train = dataloader_train        # Dataloader for the training set
        self.dataloader_val = dataloader_val            # Dataloader for the validation set
        self.optimizer = optimizer                      # Optimizer to use for training, Adam at default
        self.beta = beta                                # β (scaling-)parameter for the β-VAE
        self.reconstruction_loss = reconstruction_loss  # Reconstruction loss to use, BCE or MSE
        
        # Early stopping parameters
        self.patience = patience                        # Number of epochs to wait for improvement before stopping
        self.min_delta = 10.0                           # Minimum change to qualify as an improvement
        self.best_val_loss = np.inf                     # Temp to keep track of best validation loss
        self.epochs_no_improve = 0                      # Counter
        
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.training_loss = {'Total loss':[], 'Reconstruction loss':[], 'KL divergence loss':[]}
        self.validation_loss = {'Total loss':[], 'Reconstruction loss':[], 'KL divergence loss':[]}

    def loss_function(self, x_hat, x, mu, logvar, beta):
        # Create mask with ones for the valid pixels (> 0) to disregard invalid pixels in the loss
        valid_pixels = torch.where(x > 0, torch.ones_like(x), torch.zeros_like(x)) # Mask is defined for the whole batch (x)

        if self.reconstruction_loss == "BCE":
            # BCE = Binary cross entropy = reconstruction loss
            BCE_loss_ = F.binary_cross_entropy(x_hat, x, reduction='none') * valid_pixels # Masked pixel-wise loss only regarding valid pixels 
            recon_loss = torch.mean(torch.sum(BCE_loss_))#/torch.sum(valid_pixels))#, dim=(1,2,3)))#/torch.sum(valid_pixels) #torch.mean(torch.sum(BCE_loss_, dim=(1,2,3)))#/torch.sum(valid_pixels)) # Sum over all pixels. Divide by the number of valid pixels to get the average loss per valid pixel to hinder bias in training. Mean to get scalar for backprop.
                
        if self.reconstruction_loss == "MSE":
            # MSE = Mean squared error = reconstruction loss
            MSE_loss_ = F.mse_loss(x_hat, x, reduction='none') * valid_pixels
            recon_loss = torch.mean(torch.sum(MSE_loss_))#/torch.sum(valid_pixels))#, dim=(1,2,3)))#/torch.sum(valid_pixels) #torch.mean(torch.sum(BCE_loss_, dim=(1,2,3)))#/torch.sum(valid_pixels)) # Sum over all pixels. Divide by the number of valid pixels to get the average loss per valid pixel to hinder bias in training. Mean to get scalar for backprop.
            

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())#, dim=1)

        return recon_loss + beta*KLD_loss, recon_loss, KLD_loss
    
    def train_epoch(self):
        """Trains model for one epoch, returns the average loss (bce + beta*kl, bce and kl) for the epoch"""
        self.model.train()

        tot_train_loss = 0.0
        bce_train_loss = 0.0
        kl_train_loss = 0.0

        for x_batch in self.dataloader_train:
            
            x_batch = x_batch.to(self.device)
            x_hat, mu, log_var = self.model(x_batch)
            loss, bce_loss, kl_loss = self.loss_function(x_hat, x_batch, mu, log_var, self.beta)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            tot_train_loss += loss.item()
            bce_train_loss += bce_loss.item()
            kl_train_loss += kl_loss.item()
        
        avg_tot_train_loss = tot_train_loss/len(self.dataloader_train.dataset)
        avg_bce_train_loss = bce_train_loss/len(self.dataloader_train.dataset)
        avg_kl_train_loss = kl_train_loss/len(self.dataloader_train.dataset)

        return avg_tot_train_loss, avg_bce_train_loss, avg_kl_train_loss
    
    def validate_epoch(self):
        """Validates model for one epoch, returns the average validation loss (bce + beta*kl, bce and kl)"""
        self.model.eval()

        tot_val_loss = 0.0
        bce_val_loss = 0.0
        kl_val_loss = 0.0

        with torch.no_grad():
            for x_batch in self.dataloader_val:
                x_batch = x_batch.to(self.device)
                x_hat, mu, log_var = self.model(x_batch)
                loss, bce_loss, kl_loss = self.loss_function(x_hat, x_batch, mu, log_var, self.beta)
                tot_val_loss += loss.item()
                bce_val_loss += bce_loss.item()
                kl_val_loss += kl_loss.item()
        
        avg_tot_val_loss = tot_val_loss/len(self.dataloader_val.dataset)
        avg_bce_val_loss = bce_val_loss/len(self.dataloader_val.dataset)
        avg_kl_val_loss = kl_val_loss/len(self.dataloader_val.dataset)

        return avg_tot_val_loss, avg_bce_val_loss, avg_kl_val_loss
    
    def train(self, early_stopping=True):
        """Trains the model for self.epochs epochs and updates self.training_loss and self.validation_loss with the loss for each epoch"""
        print(f'Training VAE model\n Name: "{self.model.encoder.name}"\n Latent dim: {self.model.latent_dim} | β: {self.beta} | Epochs: {self.epochs} | Batch size: {self.batch_size} | Learning rate: {self.learning_rate}')
        print('------------------------------------------------------------')
        
        for epoch in range(self.epochs):
            print(f'Epoch {epoch+1}/{self.epochs}')
            # Calculate average loss for training and validation set for the epoch
            avg_tot_train_loss, avg_bce_train_loss, avg_kl_train_loss = self.train_epoch()
            avg_tot_val_loss, avg_bce_val_loss, avg_kl_val_loss = self.validate_epoch()
            
            self.training_loss['Total loss'].append(avg_tot_train_loss)
            self.training_loss['Reconstruction loss'].append(avg_bce_train_loss)
            self.training_loss['KL divergence loss'].append(avg_kl_train_loss)
            
            self.validation_loss['Total loss'].append(avg_tot_val_loss)
            self.validation_loss['Reconstruction loss'].append(avg_bce_val_loss)
            self.validation_loss['KL divergence loss'].append(avg_kl_val_loss)
            
            print(f'Training loss: {avg_tot_train_loss:.3f}           | Validation loss: {avg_tot_val_loss:.3f}')
            print(f'Reconstruction loss: {avg_bce_train_loss:.3f}     | Validation loss recon: {avg_bce_val_loss:.3f}')
            print(f'KL divergence loss: {avg_kl_train_loss:.3f}       | Validation loss KL: {avg_kl_val_loss:.3f}')
            print('------------------------------------------------------------')
            
            # Early stopping check
            if early_stopping:
                if self.validation_loss['Total loss'][-1] < self.best_val_loss - self.min_delta:
                    self.best_val_loss = self.validation_loss['Total loss'][-1]
                    self.epochs_no_improve = 0
                else:
                    self.epochs_no_improve += 1
                
                if self.epochs_no_improve >= self.patience:
                    print(f'Early stopping triggered after {epoch+1} epochs.')
                    return epoch+1
        
        if not early_stopping:
            return self.epochs
                
            