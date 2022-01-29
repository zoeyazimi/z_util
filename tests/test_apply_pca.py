# test the PCA function
from z_util.PCA_ import apply_pca
import numpy as np

def test_apply_pca():
    rng = np.random.RandomState(5)
    X = np.dot(rng.rand(100, 2), rng.randn(2, 2))
    Xp = apply_pca(X, 2)
    assert X.shape == Xp.shape
