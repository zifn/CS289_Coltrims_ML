import os
import numpy as np

import src.preprocess as preprocess
import src.fitting as fitting
import src.parsing as parsing
import src.cluster as clustering
import src.visualization as visualization

import matplotlib.pyplot as plt

from argparse import ArgumentParser

def save_clusters(cluster_labels, data, L_max, bins, entropy, clustering_method="kmeans-molecular-frame"):
    k = len(np.unique(cluster_labels))
    root_dir = "privileged"
    if not os.path.isdir(root_dir):
        os.mkdir(root_dir)

    dir_name = f"{clustering_method}_with_{k}_clusters_{L_max}_{bins}_{int(entropy)}"
    dir_name = os.path.join(root_dir, dir_name)

    #while os.path.isdir(dir_name):
    #    dir_name += "_"
    os.mkdir(dir_name)
    
    for cluster_label in np.unique(cluster_labels):
        cluster_data = data[cluster_labels == cluster_label]
        file_name = f"cluster_{cluster_label}_of_{k}.dat"
        file_path = os.path.join(dir_name, file_name)
        parsing.write_momentum(file_path, cluster_data, write_headers=True)

    return dir_name

def read_clusters(directory, has_headers=False):
    # Cluster directories are named as follows
    # f'{clustering-method}_with_{k}_clusters_{L_max}_{bins}_{entropy}'
    cluster_metadata = os.path.basename(directory.strip('/\\')).split('_')
    method, _, k, _, L_max, bins, entropy = cluster_metadata
    
    cluster_files = os.listdir(directory)

    data = []
    labels = []
    for filename in cluster_files:
        if not filename.endswith('.dat'):
            continue
        idx = int(filename.split('_')[1])
        cluster_data = parsing.read_momentum(os.path.join(directory, filename), has_headers=has_headers)

        N = cluster_data.shape[0]

        data.append(cluster_data)
        labels.append(idx*np.ones(N))

    return np.vstack(data), np.concatenate(labels)

def visualize_clusters(directory, save_kwargs={'dpi': 250}):
    data, labels = read_clusters(directory, has_headers=True)

    fig = visualization.plot_electron_energy_vs_KER(data, clusters=labels)
    fig.savefig(os.path.join(directory, 'electron-energy-vs-KER.png'), **save_kwargs)

    fig = visualization.plot_electron_energies(data, clusters=labels)
    fig.savefig(os.path.join(directory, 'electron-energies.png'), **save_kwargs)

    fig = visualization.plot_ion_energies(data, clusters=labels)
    fig.savefig(os.path.join(directory, 'ion-energies.png'), **save_kwargs)

    fig = visualization.plot_KER_vs_angle(data, clusters=labels)
    fig.savefig(os.path.join(directory, 'KER-vs-angle.png'), **save_kwargs)

    fig = visualization.plot_electron_energy_vs_ion_energy_difference(data, clusters=labels)
    fig.savefig(os.path.join(directory, 'electron-energy-vs-ion_energy-difference.png'), **save_kwargs)
    

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
    return optimal_parameters

def optimal_k_means_hyperparameters(phi, train_data, val_data, train_indices, val_indices, cluster_range, L_max, num_bins):
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
        print(parameters[-1])

    entropies = np.array(parameters)[:,1]
    optimal_index = np.argmin(entropies)
    optimal_parameters = parameters[optimal_index]
    print("optimal_parameters = ", optimal_parameters)
    return optimal_parameters[0], optimal_parameters[1], k_labels[optimal_index]


def analyze(fileName):
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
    data = parsing.read_momentum(fileName)
    data = preprocess.molecular_frame(data[:10000, :])
    print('Data read from file ' + fileName + ' has shape ' + str(data.shape) + '.')
    phi = preprocess.generate_feature_matrix(data)
    
    indices = np.arange(data.shape[0])
    train_indices, test_val_indices = preprocess.data_split(indices, .70, 23)
    val_indices, test_indices = preprocess.data_split(test_val_indices, .50, 46)

    train_data = data[train_indices, :]
    val_data = data[val_indices, :]
    test_data = data[test_indices, :]

    assert train_data.shape[1] == val_data.shape[1] == test_data.shape[1] == data.shape[1], \
        'Number of columns is consistent between data splits.'
    assert train_data.shape[0] + val_data.shape[0] + test_data.shape[0] == data.shape[0], \
        'Number of data points is consistent between data splits.'

    # Cluster one via k-means with 5 clusters
    num_clusters = 5
    k5_labels, _ = clustering.k_means_clustering(phi, num_clusters=num_clusters)

    max_L_to_try = 6
    bin_range = np.arange(50,160,10)
    L_max, num_bins, _ = optimal_angular_distribution_hyperparameters(train_data, val_data, k5_labels, train_indices, val_indices, max_L_to_try, bin_range)

    # With best ang. dist. parameters, choose number of clusters in k-means with lowest cross_entropy.
    cluster_range = np.arange(2,8)
    num, entropy, k_labels = optimal_k_means_hyperparameters(phi, train_data, test_data, train_indices, test_indices, cluster_range, L_max, num_bins)    
    #save_clusters(found_labels[optimal_index], data, L, num_bins, entropies[optimal_index], clustering_method="kmeans-molecular_frame")
    
    directory = save_clusters(k_labels, data, L_max, num_bins, entropy, clustering_method="kmeans-molecular-frame")
    visualize_clusters(directory)


if __name__ == '__main__':
    parser = ArgumentParser(
        description='Analyze a COLTRIMS dataset.',
        add_help=True
    )
    parser.add_argument('file', help='Path to the COLTRIMS datafile.')
    parser.add_argument('-c', '--config', help='Path to configuration file.')

    args = parser.parse_args()

    analyze(args.file)
