import src.preprocess as preprocess
import src.fitting as fitting
import src.parsing as parsing
import src.cluster as clustering
#import src.visualization   # Doesn't exist till Larry pushes branch.

def analyze(fileName):
    data = src.parsing.read_momentum(fileName)
    print('Data read from file ' + fileName + ' has shape ' + data.shape + '.')

    ################################################################################
    # PREPROCESSING
    # 1. Split data into training, validation, and testing splits of 70%, 15%, 15%.
    #    Training will be used to learn the BLM distributions and through
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

    assert train_data.shape[1] == val_data.shape[1] == test_data.shape[1] == data.shape[1], 'Number of columns is consistent between data splits.'
    assert train_data.shape[0] + val_data.shape[0] + test_data.shape[0] == data.shape[0], 'Number of data points is consistent between data splits.'

    train_phi = preprocess.generate_feature_matrix(train_data, 2)
    val_phi = preprocess.generate_feature_matrix(val_data, 2)
    test_phi = preprocess.generate_feature_matrix(test_data, 2)


def main():
    analyze('D2O_momenta.dat')
