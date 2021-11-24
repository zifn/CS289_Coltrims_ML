import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class D2O_Torch_Dataset(Dataset):
  """Dataset of D2O Momenta"""

    def __init__(self, data):
        self.data = data    # Transform data before creating dataset if desired.

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.data[idx,:]

class AutoEncoderDynamic(nn.Module):
    """
    AutoEncoder with a variable number of fully connected layers, 
    specified by the user at runtime.
    
    Parameters
    ----------
    input_shape
        shape of the input data
    hidden_dims
        sizes of the hidden dimension, must be an odd length so that the middle 
        hidden dimmension is the latent space dimension that will be used for clustering.
    latent_relu
        Boolean on whether the latent dimension should have a relu non-linearity
    output_relu
        Boolean on whether the latent dimension should have a relu non-linearity;
        typically we do not want this, as our output data has negative values which
        cannot be captured with a final ReLU layer.
    dropout
        Dropout rate to apply between each layer to "corrupt" the data and force 
        the model to learn a non-trivial representation.
    """

    def __init__(self, input_shape, hidden_dims, latent_relu=True, output_relu=False, dropout=0.3):
        super().__init__()
        assert len(hidden_dims) % 2
        widths = [input_shape] + hidden_dims + [input_shape]
        L = len(widths)

        self.encoder = nn.ModuleDict() # a collection that will hold your layers
        for i in range(L//2):
            self.encoder['encoder' + str(i)] = torch.nn.Linear(widths[i], widths[i+1]).float().to(device)
        
        self.decoder = nn.ModuleDict() # a collection that will hold your layers
        for i in range(L//2, L-1):
            self.decoder['decoder' + str(i)] = torch.nn.Linear(widths[i], widths[i+1]).float().to(device)  

        self.depth = L
        self.widths = widths
        self.latent_relu = latent_relu
        self.output_relu = output_relu
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        """
        Define how your network is going to be run.

        Parameters
        ----------
        x
          Input data
        
        Returns
        -------
        x_hat 
          Reconstructed data

        """
        latent = self.run_encoder(x)
        x_hat = self.run_decoder(latent)
        return x_hat

    def run_encoder(self, x):
        # Runs encoder
        latent = x
        
        keys = list(self.encoder.keys())
        for key in keys[:-1]:
            latent = F.relu(self.encoder[key](latent))    # relu adds non linearity
            latent = self.dropout(latent)                 # After training, need to switch to "eval" mode; to continue trin
        if self.latent_relu:
            latent = F.relu(self.encoder[keys[-1]](latent))
        else:
            latent = self.encoder[keys[-1]](latent)
        latent = self.dropout(latent)     # Apply dropout to latent layer
        return latent

    def run_decoder(self, latent):
        # Runs decoder
        output = latent
        keys = list(self.decoder.keys())
        for key in keys[:-1]:
            output = F.relu(self.decoder[key](output)) # relu adds non linearity
            output = self.dropout(output)                 # After training, need to switch to "eval" mode; to continue trin
        if self.output_relu:
            output = F.relu(self.decoder[keys[-1]](output))
        else:
            output = self.decoder[keys[-1]](output)
        # No dropout on final layer
        return output
      
def training_loop_autoencoder(model_to_train, data_loader, loss_func, 
                              optimizer, last_loop_eval=True, epochs=20):
    '''
    Wrap training inside of function so that out of scope variables are automatically (?)
    removed.
    '''
    assert loss_func.reduction == 'sum', 'We expect the autoencoder loss function to be sum squared error.'

    model_to_train.train()
    epoch_losses = []
    num_epochs = epochs + 1 if last_loop_eval else epochs
    for epoch in range(num_epochs):
        if epoch == epochs and last_loop_eval:   # Don't do backprop on last epoch
          backprop = False
          model_to_train.eval()
        else:
          backprop = True
        loss = 0
        for batch_features in data_loader:
            # load it to the active device
            batch_features = batch_features.to(device).float()

            # reset the gradients back to zero
            # PyTorch accumulates gradients on subsequent backward passes
            optimizer.zero_grad()
            
            # compute reconstructions
            outputs = model_to_train(batch_features)
            # compute training reconstruction loss
            train_loss = loss_func(outputs, batch_features)   # Loss is SSE, so it is NOT averaged over the batch.
            
            if backprop:
              # compute accumulated gradients
              train_loss.backward()
              # perform parameter update based on current gradients
              optimizer.step()
            
            # add the mini-batch training loss to epoch loss
            loss += train_loss.item()
            del batch_features, outputs
            torch.cuda.empty_cache()
        
        # compute the epoch training loss
        loss = loss / len(data_loader.dataset)    # Average loss per item
        if backprop:
            # display the epoch training loss every 5 epochs
            epoch_losses.append((epoch, loss))
            print('Overall metrics of epoch {}/{} - Unscaled AE Loss per Item:  {:9.5}'.format(epoch+1, epochs, loss))

    model_to_train.eval()
    mode_string = "Evaluation" if last_loop_eval else "Training"
    print(mode_string + " Loss (AE per item) (no dropout) after {} training epochs = {:9.5f}".format(epoch, loss))
    return epoch_losses

# We want the gradient for the latent representation when we cluster.
def get_latent_representation(model_to_use, data_loader):
    assert model_to_use.training == False, "The model must be in 'eval' mode to get reconstruction."
    latent_representation = []
    for batch in data_loader:
        batch_data = batch.to(device).float()
        #run the encoder only
        latent = model_to_use.run_encoder(batch_data)
        del batch_data
        torch.cuda.empty_cache()
        latent_representation.append(latent)
    latent_representation = torch.cat(latent_representation, dim=0)
    return latent_representation

@torch.no_grad()
def get_reconstruction(model_to_use, data_loader, loss_func):
    assert model_to_use.training == False, "The model must be in 'eval' mode to get reconstruction."
    reconstruction = []
    loss = 0
    for batch in data_loader:
        batch_data = batch.to(device).float()
        output = model_to_use(batch_data)
        train_loss = loss_func(output, batch_data)   # Loss is MSE, so it is averaged over the batch.
        loss += train_loss.item()
        del batch_data
        torch.cuda.empty_cache()
        reconstruction.append(output)
    loss = loss / len(data_loader.dataset)    # Average loss per item
    print('Reconstruction loss:', loss)
    reconstruction = torch.cat(reconstruction, dim=0)
    return reconstruction, loss