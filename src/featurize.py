import numpy as np
import scipy as sp
import sklearn
import sklearn.preprocessing

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
