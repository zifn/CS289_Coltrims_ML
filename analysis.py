import os
import yaml

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

def optimal_angular_distribution_hyperparameters(train_data, val_data, labels, train_indices, val_indices, L_max, bin_range):
    print('L_max: ', L_max)
    print('Range of bin values to consider: ', bin_range)

    parameters = []
    num_clusters = len(np.unique(labels))

    labels_train = labels[train_indices]
    labels_val = labels[val_indices]

    ion_data = []
    for i in range(num_clusters):
        ion_data.append(np.vstack((train_data[labels_train==i, 0:3], train_data[labels_train==i, 3:6])))

    val_ion_data = np.vstack((val_data[:,0:3], val_data[:,3:6]))
    val_ion_labels = np.hstack((labels_val, labels_val)).reshape(-1)
    
    # Choose best angular distribution hyperparameters
    for L in range(0, L_max+1):
        for num_bins in bin_range:
            Bs = []
            for i in range(num_clusters):
                B_lms, _ = fitting.fit_Y_lms_binning_least_squares(
                    ion_data[i], L,
                    num_bins,
                    only_even_Ls=False
                )
                Bs.append(B_lms)
            entropy = fitting.validation_cross_entropy(val_ion_data, val_ion_labels, Bs, L, only_even_Ls=False)
            parameters.append((L, num_bins, entropy))
            print(parameters[-1])

    entropies = np.array(parameters)[:,2]
    optimal_index = np.argmin(entropies)
    optimal_parameters = parameters[optimal_index]
    print("optimal_parameters = ", optimal_parameters)
    
    file_path = os.path.join("privileged", "kmeans-molecular-frame_optimal_parameters.csv")
    np.savetxt(file_path, np.array(parameters), header="L, num_bins, cross_entropy")
    return optimal_parameters

def optimal_k_means_hyperparameters(phi, data, train_data, val_data, train_indices, val_indices, cluster_range, L_max, num_bins, save=True):
    print('Range of clusters to consider: ', cluster_range)
    
    parameters = []
    k_labels = []

    for N in cluster_range:
        labels, _ = clustering.k_means_clustering(phi, num_clusters=N)
        k_labels.append(labels)
        labels_train = labels[train_indices]
        labels_val = labels[val_indices]
        
        
        ion_data = []
        for i in range(N):
            ion_data.append(np.vstack((train_data[labels_train==i, 0:3], train_data[labels_train==i, 3:6])))

        val_ion_data = np.vstack((val_data[:,0:3], val_data[:,3:6]))
        val_ion_labels = np.hstack((labels_val, labels_val)).reshape(-1)

        Bs = []
        for i in range(N):
            B_lms, lm_order = fitting.fit_Y_lms_binning_least_squares(
                ion_data[i], L_max,
                num_bins,
                only_even_Ls=False
            )
            Bs.append(B_lms)
        entropy = fitting.validation_cross_entropy(val_ion_data, val_ion_labels, Bs, L_max, only_even_Ls=False)
        parameters.append((N, entropy))
        directory = parsing.save_clusters(labels, data, L_max, num_bins, entropy, method="kmeans-molecular-frame")
        print(directory)
        visualize_clusters(directory)
        print(parameters[-1])

    entropies = np.array(parameters)[:,1]
    optimal_index = np.argmin(entropies)
    optimal_parameters = parameters[optimal_index]
    print("optimal_parameters = ", optimal_parameters)
    
    file_path = os.path.join("privileged", "kmeans-molecular-frame_k_vs_cross_entropy.csv")
    np.savetxt(file_path, np.array(parameters), header="k, cross_entropy")
    return optimal_parameters[0], optimal_parameters[1], k_labels[optimal_index]


def analyze(filename, initial_clusters, clusters_to_try, bins_to_try, max_L_to_try, viz_kwargs={}):
    """
    # PREPROCESSING
    # 1. Split data into training, validation, and testing splits of 70%, 15%, 15%.
    #    Training will be used to learn the B_LM distributions and through
    #    cross-validation to choose the binning hyperparameters. Validation will be
    #    used to choose the best clustering method. Testing will be used to evaluate
    #    match between angular momentum distributions and clustering.
    #    NOTE: Data IS shuffled in this process.
    # 2. (Optional) Convert to molecular frame. Rotate data set so that outgoing
    #    momenta of hydrogens (ions, columns 0 and 1 of data) define the coordinate
    #    axes.
    # 3. (Optional) Standarize (demean, unit variance) or whiten (demean, identity
    #    covariance matrix) data.
    # 4. Featurize data to add polynomial features. We expect 2nd order polynomial
    #    features to be physically relavent as these are used to calculate energies
    #    of the involved scattering constituents.
    """
    data = parsing.read_momentum(filename)
    data = preprocess.molecular_frame(data[:10000, :])
    print(f"Data read from file {filename} has shape {str(data.shape)}.")
    phi = preprocess.generate_feature_matrix(data)
    
    indices = np.arange(data.shape[0])
    train_indices, test_val_indices = preprocess.data_split(indices, .70, 23)
    val_indices, test_indices = preprocess.data_split(test_val_indices, .50, 46)

    train_data = data[train_indices, :]
    val_data = data[val_indices, :]
    test_data = data[test_indices, :]

    assert train_data.shape[1] == val_data.shape[1] == test_data.shape[1] == data.shape[1], \
        "Number of columns is consistent between data splits."
    assert train_data.shape[0] + val_data.shape[0] + test_data.shape[0] == data.shape[0], \
        "Number of data points is consistent between data splits."

    k5_labels, _ = clustering.k_means_clustering(phi, num_clusters=initial_clusters)

    L_max, num_bins, _ = optimal_angular_distribution_hyperparameters(train_data, val_data, k5_labels, train_indices, val_indices, max_L_to_try, bins_to_try)

    num, entropy, k_labels = optimal_k_means_hyperparameters(phi, data, train_data, test_data, train_indices, test_indices, clusters_to_try, L_max, num_bins)
    
    directory = parsing.save_clusters(k_labels, data, L_max, num_bins, entropy, method="kmeans-molecular-frame")
    visualize_clusters(directory, **viz_kwargs)


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

    args = parser.parse_args()

    with open(args.config, 'r') as stream:
        cfg = yaml.load(stream, yaml.Loader)

    for key in cfg.keys():
        cfg[key] = getattr(args, key, None) or cfg[key]
    
    print(type(cfg['viz_kwargs']))
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
        viz_kwargs=cfg['viz_kwargs']
    )
