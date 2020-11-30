import numpy as np

import src.preprocess as preprocess
import src.fitting as fitting
import src.parsing as parsing
import src.cluster as clustering
#import src.visualization   # Doesn't exist till Larry pushes branch.

from argparse import ArgumentParser

def analyze(fileName):
    data = parsing.read_momentum(fileName)
    data = data[:10000, :]
    print('Data read from file ' + fileName + ' has shape ' + str(data.shape) + '.')
    phi = preprocess.generate_feature_matrix(data)
    ################################################################################
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
    ################################################################################
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
    print('Got to clustering.')
    # Cluster one via k-means with 5 clusters
    num_clusters = 5
    k5_labels, k5_centers = clustering.k_means_clustering(phi, num_clusters=num_clusters)

    # Choose best angular distribution hyperparameters
    ion_data = np.vstack((data[:,0:3], data[:,3:6]))
    assert np.all(ion_data.shape == (data.shape[0]*2, 3))
    entropies = []
    parameters = []
    L_max = 2

    k5_labels_train = k5_labels[train_indices]

    ion_data = []
    for i in range(num_clusters):
        ion_data.append(np.vstack((train_data[k5_labels_train==i, 0:3], train_data[k5_labels_train==i, 3:6])))

    val_ion_data = np.vstack((val_data[:,0:3], val_data[:,3:6]))
    print(val_ion_data.shape)
    val_ion_labels = np.vstack((k5_labels[val_indices,None], k5_labels[val_indices, None]))
    print(val_ion_labels.shape)
    print(val_ion_labels[0:10])
    
    for L in range(0, L_max+1):
        for num_bins in range(50, 110, 10):
            print(L, num_bins)
            Bs = []
            for i in range(num_clusters):
                B_lms, lm_order = fitting.fit_Y_lms_binning_least_squares(
                    ion_data[i], L,
                    num_bins,
                    only_even_Ls=False
                )
                Bs.append(B_lms)
            entropies.append(fitting.validation_cross_entropy(val_ion_data, val_ion_labels, Bs, L, only_even_Ls=False))
            parameters.append((L, num_bins))

    entropies = np.array(entropies)
    optimal_index = np.argmin(entropies)
    optimal_parameters = parameters[optimal_index]
    print(optimal_parameters)

    # With best ang. dist. parameters, choose best clustering method and hyperparameters; repeat process.


if __name__ == '__main__':
    parser = ArgumentParser(
        description='Analyze a COLTRIMS dataset.',
        add_help=True
    )
    parser.add_argument('file', help='Path to the COLTRIMS datafile.')
    parser.add_argument('-c', '--config', help='Path to configuration file.')
    
    args = parser.parse_args()

    analyze(args.file)
