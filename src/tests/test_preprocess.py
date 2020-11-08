import copy
import scipy as sp
import numpy as np
from .. import preprocess

def feature_index_recursion(n, d):
    """
    Function to generate the indices of the columns to combine to generate the
    polynomial featurization desired for checking. Does not include the
    constant feature of 1s. This function is not optimally efficient as we have
    to recompute the columns from scratch each time; e.g. for the third order
    polynomial feature of [xyz], we would multiply columns x, y, and z rather
    than x * yz.

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
        # previous indices is the indices required to create the degree d-1 features.
        # previous_nums are the indices in the previous degree features array where we
            # start index to append on the current column to get the desired degree
            # features; i.e. for degree 2, previous indices are [[0],[1],[2],[3]] and
            # previous nums are [0,1,2,3]. Thus when prepending [0] to each of these, we
            # start at index 0. When prepending [1], we start at index 1 to prevent redundancy
            # (don't need [1,0] since this is equivalent to [0.1]).
        previous_indices, previous_nums = feature_index_recursion(n, d-1)
        new_previous_nums = copy.deepcopy(previous_nums)
        indices = copy.deepcopy(previous_indices)

        # We want the index of where the degree d-1 features begin in the
            # previous_indices array. We do this by calculating the number of degree d-2
            # features (remember to not include the constant feature).
        starting_index = int(sp.special.comb(n + d-2, d-2) - 1)

        for i in range(n):
            candidates = copy.deepcopy(previous_indices)

            # Candidates are the indices to which we append [i].
            candidates = candidates[starting_index + previous_nums[i]:]

            for a in candidates:
                a.insert(0,i)   # insert i to the beginning of each element of the candidate list.
            indices = indices + candidates  # concatenate the indices and candidates list.

            if i < n-1:
                # Update the previous_nums array by adding the number of features we preprend [i] to.
                new_previous_nums[i+1] = new_previous_nums[i]+len(candidates)

        return indices, new_previous_nums

def generate_feature_matrix_by_hand(data, degree=2):

    """
    Function performs the same task as 'featurize.generate_feature_matrix' but
    by hand rather than with sklearn for testing purposes only. Use the
    function in the featurize module for actual learning.

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

    assert feature_data.shape[0] == data.shape[0]
    assert feature_data.shape[1] == sp.special.comb(data.shape[1] + degree, degree)

    return feature_data


def test_featurize():
    data = np.random.rand(200).reshape(50,4)

    for i in range(10):
        by_hand = generate_feature_matrix_by_hand(data, degree=i+1)
        by_sklearn = preprocess.generate_feature_matrix(data, degree=i+1)
        assert np.all(np.isclose(by_hand, by_sklearn))
