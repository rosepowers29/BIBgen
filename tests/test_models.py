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

def test_FourierEncoding_batched():
    frequencies = torch.rand(8)
    fourier_encoding = FourierEncoding(16, frequencies)

    x = torch.rand(5, 64)
    result = fourier_encoding(x)
    assert result.size() == torch.Size([5, 64, 16])

    expected_result = torch.stack([fourier_encoding(event) for event in x])
    assert result.detach().numpy() == pytest.approx(expected_result.detach().numpy(), abs=1e-6)

def test_EquivariantLayer():
    layer = EquivariantLayer(4, 8)

    input_set = torch.rand((128, 4))
    output_set = layer(input_set)
    assert output_set.size() == torch.Size([128, 8])

    transpose_idx = torch.randperm(128)
    assert output_set[transpose_idx].detach().numpy() == pytest.approx(layer(input_set[transpose_idx]).detach().numpy(), abs=1e-4)

def test_EquivariantLayer_batched():
    layer = EquivariantLayer(4, 8)

    input_set = torch.rand((5, 128, 4))
    output_set = layer(input_set).detach()
    assert output_set.size() == torch.Size([5, 128, 8])

    comp_output_set = torch.stack([layer(event).detach() for event in input_set])
    assert comp_output_set.numpy() == pytest.approx(output_set.numpy(), abs=1e-6)

def test_EquivariantDenoiser():
    model = EquivariantDenoiser(
        n_timesteps = 25,
        tau_encoding_dimension = 8,
        position_encoding_dimension = 8,
        hidden_layer_size = 32,
        n_hidden_layers = 1
    )

    tau = torch.tensor(12)
    input_set = torch.rand((24, 4))
    output_set = model(input_set, tau=tau).detach()
    assert output_set.size() == torch.Size((24, 4))

    transpose_idx = torch.randperm(24)
    transposed_output_set = model(input_set[transpose_idx], tau).detach()
    assert output_set[transpose_idx].numpy() == pytest.approx(transposed_output_set.numpy(), abs=1e-4)

def test_EquivariantDenoiser_batched():
    model = EquivariantDenoiser(
        n_timesteps = 25,
        tau_encoding_dimension = 8,
        position_encoding_dimension = 8,
        hidden_layer_size = 32,
        n_hidden_layers = 1
    )

    tau = torch.tensor([12, 13, 14, 15, 16])
    input_set = torch.rand((5, 24, 4))
    output_set = model(input_set, tau=tau).detach()
    assert output_set.size() == torch.Size((5, 24, 4))

    expected_output = torch.stack([model(x, tau=t) for x, t in zip(input_set, tau)]).detach()
    assert output_set.numpy() == pytest.approx(expected_output.numpy(), abs=1e-6)