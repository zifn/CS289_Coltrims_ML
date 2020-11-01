import numpy as np
import scipy as sp
import sklearn
import sklearn.preprocessing
import copy

def generate_feature_matrix(data, degree=2):

    """
    Function to generate a featurized data matrix from a data matrix where each
    row is a scattering event. We use polynomial feature with arbitrary degree
    as well as dot product of different ion/electron momenta.

    Parameters
    ------------
    data : np.array [todo] How does this work if I pass in a pandas array
        Array containing a row for each scattering event. We have 15 columns
        (px, py, pz) for each of 5 scattering products: ion 1, ion 2, neutral,
        electron 1, electron 2.
    degree : int
        Degree of the featurization desired. For $n$ features, a degree $d$
        featurization will have $(d+n)C(d) = (d+n)C(n)$ features. With $n$
        fixed, this scales as $d^n$. We will include a constant feature of one.

    Returns
    --------
    array or dataframe
        Returns featurized matrix
    """

    featurizer = sklearn.preprocessing.PolynomialFeatures(degree=degree,
                                                            interaction_only=False,
                                                            include_bias=True)
    # interaction_only - use distinct input features only
    # include_bias - include a bias column of all ones.

    feature_data = featurizer.fit_transform(data)

    assert(feature_data.shape[0] == data.shape[0])
    assert(feature_data.shape[1] == sp.special.comb(data.shape[1] + degree, degree))

    return feature_data

def feature_index_recursion(n, d):
    """
    Function to generate the indices of the columns to combine to generate the
    polynomial featurization desired for checking. Does not include the
    constant feature of 1s.

    Parameters
    ------------
    n : int
        Number of features (columns) in our data.
    d : int
        Degree of the featurization desired.

    Returns
    --------
    List of Lists
        Returns indices that must be combined to generate our desired
        featurization. For n = 4 and d = 2, example output will be [[0], [1],
        [2], [3], [0,0], [0,1], [0,2], [0,3], [1,1], [1,2], [1,3], [2,2], [2,3]
        [3,3]].
    """
    if d == 1:      # Base case of recursion
        return [[i] for i in range(n)], [0,1,2,3]
    else:           # Inductive step
        previous_indices, previous_nums = feature_index_recursion(n, d-1)
        new_previous_nums = copy.deepcopy(previous_nums)
        indices = copy.deepcopy(previous_indices)
        starting_index = int(sp.special.comb(n + d-2, d-2) - 1)

        for i in range(n):
            candidates = copy.deepcopy(previous_indices)
            candidates = candidates[starting_index + previous_nums[i]:]

            for a in candidates:
                a.insert(0,i)
            indices = indices + candidates

            if i < n-1:
                new_previous_nums[i+1] = new_previous_nums[i]+len(candidates)

        return indices, new_previous_nums

def generate_feature_matrix_by_hand(data, degree=2):

    """
    Function performs the same task as 'generate_feature_matrix' but by hand
    rather than with sklearn for testing purposes only. Use the above function
    for actual learning.

    Parameters
    ------------
    data : np.array [todo] How does this work if I pass in a pandas array
        Array containing a row for each scattering event. We have 15 columns
        (px, py, pz) for each of 5 scattering products: ion 1, ion 2, neutral,
        electron 1, electron 2.
    degree : int
        Degree of the featurization desired. Degree must be larger than 0 and
        less than 3. We do this because it's hard to generate the features
        otherwise by hand.

    Returns
    --------
    array or dataframe
        Returns featurized matrix
    """
    n = data.shape[1]
    d = degree

    feature_indices, _ = feature_index_recursion(n, d)
    feature_data = np.ones(data.shape[0]).reshape(data.shape[0],1)
    for i in feature_indices:
        column = np.ones(data.shape[0])

        for j in i:
            column *= data[:,j]
        feature_data = np.column_stack([feature_data, column])

    assert(feature_data.shape[0] == data.shape[0])
    assert(feature_data.shape[1] == sp.special.comb(data.shape[1] + degree, degree))

    return feature_data

