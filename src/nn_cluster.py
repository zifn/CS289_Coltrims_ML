import time
import numpy as np
import sys

import autoencoder as ae

import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Can we just import Sklearn to do this clustering? How does this work for torch tensors on GPU?
@torch.no_grad()
def KMeans(x, K=5, Niter=100, clusters=None, eps=1.e-4, verbose=True):
    """Implements Lloyd's algorithm for the Euclidean metric."""
    # https://www.kernel-operations.io/keops/_auto_tutorials/kmeans/plot_kmeans_torch.html

    # TODO - Figure out how to calculate inertia.

    start = time.time()
    if clusters is None:
        # TODO: Instead do KMeans++ for initialization
        # Look at Sklearn for checking cluster consistency with different initializations.
        print('clusters argument is None.')
        clusters = x[:K, :]
    assert clusters.shape[0] == K, "We do not have {} clusters.".format(K)
    assert clusters.shape[1] == x.shape[1], "The clusters do not have the same dimensions as the data."
    N, D = x.shape  # Number of samples, dimension of the ambient space

    x_i = x.unsqueeze(1)
    c_j = clusters.unsqueeze(0)

    # K-means loop:
    # - x  is the (N, D) point cloud,
    # - cl is the (N,) vector of class labels
    # - c  is the (K, D) cloud of cluster centroids
    # TODO - Implement eps for checking change in cluster - check for convergence in Sklearn
    error = 1
    old_clusters = None
    for run in range(Niter):
        # E step: assign points to the closest cluster -------------------------
        D_ij = ((x_i - c_j) ** 2).sum(-1)  # (N, K) symbolic squared distances
        cluster_index = D_ij.argmin(dim=1).long().view(-1)  # Points -> Nearest cluster
        
        # M step: update the centroids to the normalized cluster average: ------
        # Compute the sum of points per cluster:
        clusters = torch.zeros((K, D), device=device, dtype=torch.float32)
        clusters.scatter_add_(0, cluster_index[:, None].repeat(1, D), x)

        # Divide by the number of points per cluster:
        Ncl = torch.bincount(cluster_index, minlength=K).type_as(clusters).view(K, 1)
        clusters /= Ncl  # in-place division to compute the average
        c_j = clusters.unsqueeze(0)

        if run > 10:  # Do at least 10 iterations before checking error.
            num_changes = torch.sum(torch.ne(cluster_index, old_clusters))
            error = num_changes / N
            if error < eps and verbose:
                print('Achieved unconsistency {:9.5f} with {} iterations of K-means.'.format(error, run+1))
                break

        old_clusters = cluster_index
    run += 1
    if verbose:  # Fancy display -----------------------------------------------
      if True:
          torch.cuda.synchronize()
      end = time.time()
      print(
          f"K-means for the Euclidean metric with {N:,} points in dimension {D:,}, K = {K:,}:"
      )
      print(
          "Timing for {} iterations: {:.5f}s = {} x {:.5f}s\n".format(
              run, end - start, run, (end - start) / run
          )
      )
    clusters
    return cluster_index, clusters

# Formuals are in Xie 2016 or Aljalbout 2018
def students_t_distribution(full_data, cIndex, cCentroids, nu=1):
    num_clusters = cCentroids.shape[0]
    num_points = len(cIndex)
    Q_distribution = torch.transpose(
        full_data - cCentroids.unsqueeze(1), 0, 1)   # Simplify this operations
    Q_distribution = torch.sum(torch.pow(Q_distribution, 2), dim=-1)
    Q_distribution = torch.pow(1 + Q_distribution / nu, -(nu+1)/2) # check that 1 is not identity
    return torch.div(Q_distribution, torch.unsqueeze(torch.sum(Q_distribution, axis=1), 1))

def harden_distribution(dist):
    P_distribution = torch.div(torch.pow(dist, 2), torch.unsqueeze(torch.sum(dist, axis=0), 0))
    return torch.div(P_distribution, torch.unsqueeze(torch.sum(P_distribution, axis=1), 1))

def training_loop_combined(model_to_train, 
                           data_loader, 
                           loss_func_auto, 
                           loss_func_clus, 
                           optimizer, 
                           alpha=0.75, 
                           epochs=20,
                           K=5,
                           Niter=100,
                           last_loop_eval=True,
                           verbose=False
                           ):
    '''
    Wrap training inside of function so that out of scope variables are automatically (?)
    removed.

    Do both autoencoder and clustering loss backprops using batches of O(10k).
    '''
    assert loss_func_auto.reduction == 'sum', 'We expect the autoencoder loss function to be sum squared error.'

    model_to_train.train()
    scale_factor=1.0
    clusters=None
    epoch_losses = []
    num_epochs = epochs + 1 if last_loop_eval else epochs
    for epoch in range(num_epochs):
        if epoch == epochs and last_loop_eval:   # Don't do backprop on last epoch
            backprop = False
            model_to_train.eval()
        else:
            backprop = True
        loss = 0
        a_loss = 0
        a_loss_o = 0
        c_loss = 0

        for batch_features in data_loader:
            # load it to the active device
            batch_features = batch_features.to(device).float()

            # reset the gradients back to zero
            # PyTorch accumulates gradients on subsequent backward passes
            optimizer.zero_grad()

            # compute latent representations
            batch_latents = model_to_train.run_encoder(batch_features)
            cluster_index, clusters = KMeans(batch_latents, K=K, Niter=Niter, clusters=clusters, verbose=verbose)
            q_distribution = students_t_distribution(batch_latents, cluster_index, clusters)
            p_distribution = harden_distribution(q_distribution)
            # compute training reconstruction loss
            clus_loss = loss_func_clus(torch.log(q_distribution), p_distribution)

            # compute reconstructions
            outputs = model_to_train.run_decoder(batch_latents)
            # compute training reconstruction loss
            auto_loss = loss_func_auto(scale_factor*outputs,
                                       scale_factor*batch_features)   # Loss is SSE, so it is NOT averaged over the batch.
            auto_loss_original = loss_func_auto(outputs,
                                         batch_features)
            if verbose:
                print('Scale factor:  {:9.2}, Scale AE Loss: {:9.2}, Original AE Loss: {:9.2}, KL Div Loss: {:9.2}'.format(scale_factor, auto_loss.item(), auto_loss_original.item(), clus_loss.item()))

            train_loss = alpha*auto_loss + (1-alpha)* clus_loss

            if backprop:
                # compute accumulated gradients
                train_loss.backward()
                # perform parameter update based on current gradients
                optimizer.step()
            
            # add the mini-batch training loss to epoch loss
            scale_factor = np.sqrt(clus_loss.item() / auto_loss_original.item())

            loss += train_loss.item()
            a_loss += auto_loss.item() # Sum squared error of SCALED loss
            a_loss_o += auto_loss_original.item() # Sum squared error of UNSCALED loss
            c_loss += clus_loss.item()
            del batch_features, outputs
            torch.cuda.empty_cache()
        # compute the epoch training loss
        loss = loss / len(data_loader.dataset)    # Average loss per item
        a_loss = a_loss / len(data_loader.dataset)    # Average loss per item of scaled AE
        a_loss_o = a_loss_o / len(data_loader.dataset)    # Average loss per item of unscaled AE
        c_loss = c_loss / len(data_loader.dataset)    # Average loss per item of KL Div
        #print('Scale factor:  {:9.2}, Scale AE Loss: {:9.2}, Original AE Loss: {:9.2}, KL Div Loss: {:9.2}'.format(scale_factor, auto_loss.item(), auto_loss_original.item(), clus_loss.item()))
        if backprop:
            # display the epoch training loss every 5 epochs
            epoch_losses.append((epoch, loss, a_loss, a_loss_o, c_loss))
            print('Overall metrics of epoch {}/{} - Alpha Loss per Item:  {:9.5}, Scaled AE Loss per Item:  {:9.5}, Unscaled AE Loss per Item:  {:9.5}, KLD Loss per Item:  {:9.5}'.format(epoch+1, epochs, loss, a_loss, a_loss_o, c_loss))
            #print("epoch : {}/{}, loss = {:.6f}".format(epoch, epochs, loss))

    model_to_train.eval()
    mode_string = "Evaluation" if last_loop_eval else "Training"
    print(mode_string + " Losses (alpha per item, AE per item) (no dropout) after {} training epochs = {:9.5f}, {:9.5f}".format(epoch, loss, a_loss_o))
    return epoch_losses, clusters