import pytest
import numpy as np

from BIBgen.preprocessing import *

def test_whitening_matrices():
    data = np.random.rand(50, 5)

    W, W_inv, mu = whitening_matrices(data)
    transformed = (data - mu) @ W.T

    assert np.mean(transformed, axis=0) == pytest.approx(np.zeros(5), abs=1e-4)
    assert np.cov(transformed, rowvar=False) == pytest.approx(np.identity(5), abs=1e-4)