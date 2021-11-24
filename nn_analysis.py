import os
import yaml
from itertools import product

import numpy as np

import src.preprocess as preprocess
import src.fitting as fitting
import src.parsing as parsing
import src.cluster as clustering
import src.visualization as visualization

import matplotlib.pyplot as plt

from argparse import ArgumentParser

def visualize_clusters(directory, bins=None, plot_kwargs={}, save_kwargs={'dpi': 250, 'bbox_inches': 'tight'}):
    data, labels = parsing.read_clusters(directory, has_headers=True)

    fig = visualization.plot_electron_energy_vs_KER(data, clusters=labels, bins=bins, **plot_kwargs)
    fig.savefig(os.path.join(directory, 'electron-energy-vs-KER.png'), **save_kwargs)
    plt.close(fig)

    fig = visualization.plot_electron_energies(data, clusters=labels, bins=bins, **plot_kwargs)
    fig.savefig(os.path.join(directory, 'electron-energies.png'), **save_kwargs)
    plt.close(fig)

    fig = visualization.plot_ion_energies(data, clusters=labels, bins=bins, **plot_kwargs)
    fig.savefig(os.path.join(directory, 'ion-energies.png'), **save_kwargs)
    plt.close(fig)

    fig = visualization.plot_KER_vs_angle(data, clusters=labels, bins=bins, **plot_kwargs)
    fig.savefig(os.path.join(directory, 'KER-vs-angle.png'), **save_kwargs)
    plt.close(fig)

    fig = visualization.plot_electron_energy_vs_ion_energy_difference(data, clusters=labels, bins=bins, **plot_kwargs)
    fig.savefig(os.path.join(directory, 'electron-energy-vs-ion_energy-difference.png'), **save_kwargs)
    plt.close(fig)

def deep_k_means(data_loader, data_loader_no_shuffle, K, input_dim=15, 
                 hidden_dims=[128,32,10,32,128], dropout=0.3, lr_rate=1.e-2,  
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 alpha=0.25, epochs_ae=20, epochs_clus=20, 
                 Niter=100, verbose=False):
    # create a model from `Autoencoder` autoencoder class
    # load it to the specified device, either gpu or cpu
    model = ae.AutoEncoderDynamic(input_shape=input_dim, hidden_dims=hidden_dims, dropout=dropout).to(device)

    # Adam optimizer with learning rate lr
    optimizer = optim.Adam(model.parameters(), lr=lr_rate)

    # mean-squared error loss
    criterionSSE = nn.MSELoss(reduction='sum')
    criterionKLDiv = nn.KLDivLoss(reduction='sum')

    # Train autoencoder on all the data
    ae_losses = ae.training_loop_autoencoder(model, data_loader, criterionSSE, optimizer, epochs=epochs_ae)
    # Update autoencoder using K-means clustering loss
    combo_losses, clusters = cl.training_loop_combined(model, 
                          data_loader, 
                          criterionSSE, 
                          criterionKLDiv, 
                          optimizer, 
                          alpha=alpha, 
                          epochs=epochs_clus,
                          K=K,
                          verbose=False
                          )
    
    # Get latent space representation of all the data
    latent_representation = ae.get_latent_representation(model, data_loader_no_shuffle)
    cluster_index, clusters = cl.KMeans(latent_representation, K=K, Niter=Niter, clusters=clusters, verbose=verbose)
    assert len(cluster_index) == len(data_loader.dataset)
    return cluster_index, clusters

def optimal_DEC_hyperparameters(data, data_loader, data_loader_no_shuffle, 
                                train_data, val_data, train_indices, val_indices, 
                                max_L_to_try, bins_to_try, cluster_range, save_dir='.'):

    parameters = []
    k_labels = []
    summary_file_path = os.path.join(save_dir, "DEC-molecular-frame_results_summary.csv")
    with open(summary_file_path, "w") as f:
        f.write("index,num_clusters,cross-entropy,L_max,num_bins,N\n")

    i = 0
    for N in cluster_range:
        i += 1
        print(f"clustering with: N = {N}")
        labels, _ = deep_k_means(data_loader, data_loader_no_shuffle, N, input_dim=15, 
                                 hidden_dims=[128,32,10,32,128], dropout=0.3, alpha=0.25, 
                                 epochs_ae=20, epochs_clus=20, Niter=100, verbose=False)
        print('finished clustering')
        labels = labels.cpu().numpy()   # Need to convert to numpy for the rest of the loop.
        k_labels.append(labels)
        labels_train = labels[train_indices]
        labels_val = labels[val_indices]
        num_clusters = len(np.unique(labels))
        
        print('starting angular analysis')
        L_max, num_bins, entropy = analysis.optimal_angular_distribution_hyperparameters(train_data, val_data, labels, train_indices, val_indices, max_L_to_try, bins_to_try, "DEC", save_dir=None)
        print('ending angular analysis')       
        parameters.append((i, num_clusters, entropy, L_max, num_bins, N))
        
        temp_dir = os.path.join(save_dir, "DEC_params_" + f"{i}")
        if not os.path.isdir(temp_dir):
            os.mkdir(temp_dir)
        directory = parsing.save_clusters(labels, data, L_max, num_bins, entropy, root_dir=temp_dir, method="DEC")

        analysis.visualize_clusters(directory, 100)
        param_names = ("index", "num_clusters", "cross-entropy", "L_max", "num_bins", 'N')
        print("optimal iteration: ", [f" {w} = {n}" for w, n in zip(param_names, parameters[-1])])
        with open(summary_file_path, "w") as f:
            f.write(f"{i},{num_clusters},{entropy},{L_max},{num_bins},{N}\n")

    entropies = np.array(parameters)[:,1]
    optimal_index = np.argmin(entropies)
    optimal_parameters = parameters[optimal_index]
    print("optimal: ", [f" {w} = {n}" for w, n in zip(param_names, optimal_parameters)])

    parameters = np.array(parameters)
    print(parameters)
    file_path = os.path.join(save_dir, "DEC_k_vs_cross-entropy.png")
    plt.plot(parameters[:,1], parameters[:,2], "x")
    plt.xlabel("Number of Clusters", size=18)
    plt.ylabel("Cross Entropy", size=18)
    plt.title("DEC:\nCross Entropy vs Number of Clusters", size=20, y=1.05)
    plt.savefig(file_path)
    plt.close()
    return optimal_parameters[0], optimal_parameters[1], k_labels[optimal_index]


def analyze(filename, initial_clusters, clusters_to_try, bins_to_try, max_L_to_try, save_dir='.', viz_kwargs={}):
    
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    
    #read and preprocess data
    data = parsing.read_momentum(filename)
    data = preprocess.molecular_frame(data[:, :])
    print(f"Data read from file {filename} has shape {str(data.shape)}.")
    
    #split data
    indices = np.arange(data.shape[0])
    train_indices, test_val_indices = preprocess.data_split(indices, .70, 23)
    val_indices, test_indices = preprocess.data_split(test_val_indices, .50, 46)

    train_data = data[train_indices, :]
    val_data = data[val_indices, :]
    test_data = data[test_indices, :]
    
    #check for simple errors
    assert train_data.shape[1] == val_data.shape[1] == test_data.shape[1] == data.shape[1], \
        "Number of columns is consistent between data splits."
    assert train_data.shape[0] + val_data.shape[0] + test_data.shape[0] == data.shape[0], \
        "Number of data points is consistent between data splits."
    
    batch_size=16384
    torch_full = D2O_Torch_Dataset(data)

    full_loader_shuffle = DataLoader(torch_full, batch_size=batch_size, shuffle=True)
    full_loader_no_shuffle = DataLoader(torch_full, batch_size=batch_size, shuffle=False)
    
     num, entropy, k_labels = optimal_DEC_hyperparameters(D2O_dataset_process, full_loader_shuffle, full_loader_no_shuffle, 
                            train_data, val_data, train_indices, val_indices, 
                            max_L_to_try=max_L_to_try, bins_to_try=bins_to_try, 
                            cluster_range=clusters_to_try, save_dir='.')

if __name__ == '__main__':
    parser = ArgumentParser(
        description='Analyze a COLTRIMS dataset.',
        add_help=True
    )
    parser.add_argument('datafile', help='Path to the COLTRIMS datafile.')
    parser.add_argument('-c', '--config', help='Path to configuration file.', default='defaults.yml')

    parser.add_argument('--cinit', dest='clusters_init', help='The initial number of clusters to use.')
    parser.add_argument('--cmin', dest='clusters_min', help='The minimum cluster size.')
    parser.add_argument('--cmax', dest='clusters_max', help='The maximum cluster size.')
    parser.add_argument('--cstep', dest='clusters_step', help='The step size for the cluster grid search.')

    parser.add_argument('--bmin', dest='bins_min', help='The minimum bin size.')
    parser.add_argument('--bmax', dest='bins_max', help='The maximum bin size.')
    parser.add_argument('--bstep', dest='bins_step', help='The step size for the bin size grid search')

    parser.add_argument('-L', dest='L', help='The largest Lmax to try.')

    parser.add_argument('-s', '--save', dest='save_dir', help='Directory to save results to.')

    args = parser.parse_args()

    with open(args.config, 'r') as stream:
        cfg = yaml.load(stream, yaml.Loader)

    for key in cfg.keys():
        cfg[key] = getattr(args, key, None) or cfg[key]

    analyze(
        args.datafile,
        initial_clusters=cfg['clusters_init'],
        clusters_to_try=np.arange(
            cfg['clusters_min'], cfg['clusters_max'] + 1, cfg['clusters_step']
        ),
        bins_to_try=np.arange(
            cfg['bins_min'], cfg['bins_max'] + 1, cfg['bins_step']
        ),
        max_L_to_try=cfg['L'],
        save_dir=cfg['save_dir'],
        viz_kwargs=cfg['viz_kwargs']
    )
