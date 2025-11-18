import pytest
import numpy as np
import torch

from BIBgen.models import *

def test_FourierEncoding():
    frequencies = torch.rand(8)
    fourier_encoding = FourierEncoding(16, frequencies)
    
    x = torch.rand(64)
    result = fourier_encoding(x)

    thetas = torch.outer(x, 2 * np.pi * frequencies)
    sin_elems = torch.sin(thetas)
    cos_elems = torch.cos(thetas)
    expected_result = torch.cat((sin_elems, cos_elems), dim=1)

    assert result.size() == torch.Size([64, 16])
    assert result.detach().numpy() == pytest.approx(expected_result, abs=1e-6)

def test_VarianceTower():
    initial_variances = torch.rand(1000)
    variance_tower = VarianceTower(1000, initial_variances)
    
    taus = torch.randint(1000, size=(100,))
    result = variance_tower(taus)
    expected_result = initial_variances[taus]
    
    assert result.size() == torch.Size([100])
    assert result.detach().numpy() == pytest.approx(expected_result)