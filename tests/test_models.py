import pytest
import numpy as np
import torch

from BIBgen.models import *

def test_FourierEncoding():
    frequencies = torch.rand(8)
    fourier_encoding = FourierEncoding(16, frequencies)
    
    x = torch.rand(64)
    result = fourier_encoding(x)

    thetas = torch.outer(x, frequencies)
    sin_elems = torch.sin(thetas)
    cos_elems = torch.cos(thetas)
    expected_result = torch.cat((sin_elems, cos_elems), dim=1)

    assert result.size() == torch.Size([64, 16])
    assert result.detach().numpy() == pytest.approx(expected_result, abs=1e-6)

def test_EquivariantLayer():
    layer = EquivariantLayer(4, 8)

    input_set = torch.rand((128, 4))
    output_set = layer(input_set)
    assert output_set.size() == torch.Size([128, 8])

    transpose_idx = torch.randperm(128)
    assert output_set[transpose_idx].detach().numpy() == pytest.approx(layer(input_set[transpose_idx]).detach().numpy(), abs=1e-4)