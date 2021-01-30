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

def optimal_k_means_hyperparameters(phi, data, train_data, val_data, train_indices, val_indices, cluster_range, L_max, num_bins, save_dir='.'):
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
        
        electron_data = []
        for i in range(N):
            electron_data.append(np.vstack((train_data[labels_train==i, 9:12], train_data[labels_train==i, 12:15])))
        val_elec_data = np.vstack((val_data[:,9:12], val_data[:,12:15]))
        val_elec_labels = np.hstack((labels_val, labels_val)).reshape(-1)

        Bs = []
        for i in range(N):
            B_lms, lm_order = fitting.fit_Y_lms_binning_least_squares(
                electron_data[i], L_max,
                num_bins,
                only_even_Ls=False
            )
            Bs.append(B_lms)
        entropy = fitting.validation_cross_entropy(val_elec_data, val_elec_labels, Bs, L_max, only_even_Ls=False)
        parameters.append((N, entropy))

        directory = parsing.save_clusters(labels, data, L_max, num_bins, entropy, root_dir=save_dir, method="kmeans-molecular-frame")

        visualize_clusters(directory)
        print(parameters[-1])

    entropies = np.array(parameters)[:,1]
    optimal_index = np.argmin(entropies)
    optimal_parameters = parameters[optimal_index]
    print("optimal_parameters = ", optimal_parameters)
    
    file_path = os.path.join(save_dir, "kmeans-molecular-frame_k_vs_cross_entropy.csv")
    np.savetxt(file_path, np.array(parameters), delimiter=',', header="k,cross_entropy", comments='')
    return optimal_parameters[0], optimal_parameters[1], k_labels[optimal_index]

def optimal_angular_distribution_hyperparameters(train_data, val_data, labels, train_indices, val_indices, L_max, bin_range, save_dir=None):
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
    
    electron_data = []
    for i in range(num_clusters):
        electron_data.append(np.vstack((train_data[labels_train==i, 9:12], train_data[labels_train==i, 12:15])))

    val_elec_data = np.vstack((val_data[:,9:12], val_data[:,12:15]))
    val_elec_labels = np.hstack((labels_val, labels_val)).reshape(-1)
    
    # Choose best angular distribution hyperparameters
    for L in range(0, L_max+1):
        for num_bins in bin_range:
            Bs = []
            for i in range(num_clusters):
                B_lms, _ = fitting.fit_Y_lms_binning_least_squares(
                    electron_data[i], L,
                    num_bins,
                    only_even_Ls=False
                )
                if not any(np.isnan(B_lms)):
                    Bs.append(B_lms)
            entropy = fitting.validation_cross_entropy(val_elec_data, val_elec_labels, Bs, L, only_even_Ls=False)
            parameters.append((L, num_bins, entropy))

    entropies = np.array(parameters)[:,2]
    optimal_index = np.argmin(entropies)
    optimal_parameters = parameters[optimal_index]
    print("optimal_parameters = ", optimal_parameters)
    
    if save_dir != None:
        file_path = os.path.join(save_dir, "kmeans-molecular-frame_optimal_parameters.csv")
        np.savetxt(file_path, np.array(parameters), delimiter=',', header="L,num_bins,cross_entropy", comments='')
    return optimal_parameters



def optimal_optics_hyperparameters(phi, data, train_data, val_data, train_indices, val_indices, max_L_to_try, bins_to_try, save_dir='.'):

    min_samp_list = 10**np.arange(1, 7, 1)
    max_eps_array = 10.0**np.arange(5, 100, 10)
    parameters = []
    k_labels = []
    param_names = ("index", "num_clusters", "cross-entropy", "L_max", "num_bins", "min_samp", "max_eps")
    summary_file_path = os.path.join(save_dir, "optics-molecular-frame_results_summary.csv")
    with open(summary_file_path, "w") as f:
        f.write("index,num_clusters,cross-entropy,L_max,num_bins,min_samp,max_eps\n")
    
    i = 0
    for min_samp, max_eps in product(min_samp_list, max_eps_array):
        i += 1
        print(f"clustering with: min_samp = {min_samp} max_eps = {max_eps}")
        labels, _ = clustering.optics_clustering(phi, min_samp, max_eps)
        k_labels.append(labels)
        labels_train = labels[train_indices]
        labels_val = labels[val_indices]
        num_clusters = len(np.unique(labels))
        
        
        L_max, num_bins, entropy = optimal_angular_distribution_hyperparameters(train_data, val_data, labels, train_indices, val_indices, max_L_to_try, bins_to_try, save_dir=None)
        parameters.append((i, num_clusters, entropy, L_max, num_bins, min_samp, max_eps))
        
        temp_dir = os.path.join(save_dir, f"optics_params_{i}")
        if not os.path.isdir(temp_dir):
            os.mkdir(temp_dir)
        directory = parsing.save_clusters(labels, data, L_max, num_bins, entropy, root_dir=temp_dir, method="optics")

        visualize_clusters(directory, 100)
        print(sum([f" {w} = {n}" for w, n in zip(param_names, param_names[-1])]))
        with open(summary_file_path, "w") as f:
            f.write(f"{i},{num_clusters},{entropy},{L_max},{num_bins},{min_samp},{max_eps}\n")

    entropies = np.array(parameters)[:,1]
    optimal_index = np.argmin(entropies)
    optimal_parameters = parameters[optimal_index]
    print("optimal: " + sum([f" {w} = {n}" for w, n in zip(param_names, optimal_parameters)]))

    file_path = os.path.join(save_dir, "optics_k_vs_cross-entropy.png")
    plt.plot(parameters[1], parameters[2], "x")
    plt.xlabel("Number of Clusters", size=18)
    plt.ylabel("Cross Entropy", size=18)
    plt.title("OPTICS:\nCross Entropy vs Number of Clusters", size=20, y=1.05)
    plt.savefig(file_path)
    plt.close()
    return optimal_parameters[0], optimal_parameters[1], k_labels[optimal_index]

def long_optimal_kmeans_hyperparameters(phi, data, train_data, val_data, train_indices, val_indices, max_L_to_try, bins_to_try, cluster_range, save_dir='.'):

    min_samp_list = 10**np.arange(1, 7, 1)
    max_eps_array = 10.0**np.arange(5, 100, 10)
    parameters = []
    k_labels = []
    summary_file_path = os.path.join(save_dir, "kmeans-molecular-frame_results_summary.csv")
    with open(summary_file_path, "w") as f:
        f.write("index,num_clusters,cross-entropy,L_max,num_bins,N\n")

    i = 0
    for N in cluster_range:
        i += 1
        print(f"clustering with: N = {N}")
        labels, _ = clustering.k_means_clustering(phi, N)
        k_labels.append(labels)
        labels_train = labels[train_indices]
        labels_val = labels[val_indices]
        num_clusters = len(np.unique(labels))
        
        
        L_max, num_bins, entropy = optimal_angular_distribution_hyperparameters(train_data, val_data, labels, train_indices, val_indices, max_L_to_try, bins_to_try, save_dir=None)
        parameters.append((i, num_clusters, entropy, L_max, num_bins, N))
        
        temp_dir = os.path.join(save_dir, "kmeans_params_" + f"{i}")
        if not os.path.isdir(temp_dir):
            os.mkdir(temp_dir)
        directory = parsing.save_clusters(labels, data, L_max, num_bins, entropy, root_dir=temp_dir, method="kmeans")

        visualize_clusters(directory, 100)
        param_names = ("index", "num_clusters", "cross-entropy", "L_max", "num_bins", 'N')
        print("optimal iteration: ", [f" {w} = {n}" for w, n in zip(param_names, parameters[-1])])
        with open(summary_file_path, "w") as f:
            f.write(f"{i},{num_clusters},{entropy},{L_max},{num_bins},{N}\n")

    entropies = np.array(parameters)[:,1]
    optimal_index = np.argmin(entropies)
    optimal_parameters = parameters[optimal_index]
    print("optimal: ", [f" {w} = {n}" for w, n in zip(param_names, optimal_parameters)])

    parameters = np.array(parameters)
    
    file_path = os.path.join(save_dir, "kmeans_k_vs_cross-entropy.png")
    plt.plot(parameters[1], parameters[2], "x")
    plt.xlabel("Number of Clusters", size=18)
    plt.ylabel("Cross Entropy", size=18)
    plt.title("kmeans:\nCross Entropy vs Number of Clusters", size=20, y=1.05)
    plt.savefig(file_path)
    plt.close()
    return optimal_parameters[0], optimal_parameters[1], k_labels[optimal_index]

def analyze(filename, initial_clusters, clusters_to_try, bins_to_try, max_L_to_try, save_dir='.', viz_kwargs={}):
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
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    data = parsing.read_momentum(filename)
    data = preprocess.molecular_frame(data[:, :])
    print(f"Data read from file {filename} has shape {str(data.shape)}.")
    phi = preprocess.generate_feature_matrix(data)
    print("Generated Features")
    phi = preprocess.perform_PCA(phi, components=30)
    print("PCA-ed features")
    
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

    if False:
        k5_labels, _ = clustering.k_means_clustering(phi, num_clusters=initial_clusters)

        L_max, num_bins, _ = optimal_angular_distribution_hyperparameters(train_data, val_data, k5_labels, train_indices, val_indices, max_L_to_try, bins_to_try, save_dir)

        num, entropy, k_labels = optimal_k_means_hyperparameters(phi, data, train_data, test_data, train_indices, test_indices, clusters_to_try, L_max, num_bins, save_dir)

        directory = parsing.save_clusters(k_labels, data, L_max, num_bins, entropy, root_dir=save_dir, method="kmeans-molecular-frame")
        visualize_clusters(directory, **viz_kwargs)
    elif False:
         num, entropy, k_labels = optimal_optics_hyperparameters(phi, data, train_data, test_data, train_indices, test_indices, max_L_to_try, bins_to_try, save_dir)
    elif True:
         num, entropy, k_labels = long_optimal_kmeans_hyperparameters(phi, data, train_data, test_data, train_indices, test_indices, max_L_to_try, bins_to_try, clusters_to_try, save_dir)

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
