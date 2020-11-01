import numpy as np
from .. import featurize

def test_placeholder():
    data = np.random.rand(200).reshape(50,4)

    for i in range(10):
        by_hand = featurize.generate_feature_matrix_by_hand(data, degree=i+1)
        by_sklearn = generate_feature_matrix(data, degree=i+1)
        assert(np.all(np.isclose(by_hand, by_sklearn)))
    pass
