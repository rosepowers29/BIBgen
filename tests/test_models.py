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

def test_PositionalEncoding():
    scale = 8.0
    dimension = 16
    half_dimension = dimension // 2
    encoding = PositionalEncoding(dimension, scale=scale)

    assert not isinstance(encoding.frequency_table, torch.nn.Parameter)
    assert list(encoding.parameters()) == []

    powers = torch.arange(half_dimension, dtype=torch.float32) / half_dimension
    expected_frequencies = 2 * torch.pi * scale ** powers
    assert encoding.frequency_table.squeeze(0).numpy() == pytest.approx(expected_frequencies.numpy(), abs=1e-6)

    x = torch.rand(64)
    result = encoding(x)

    thetas = torch.outer(x, expected_frequencies)
    expected_result = torch.cat((torch.sin(thetas), torch.cos(thetas)), dim=1)

    assert result.size() == torch.Size([64, 16])
    assert result.numpy() == pytest.approx(expected_result.numpy(), abs=1e-6)

def test_PositionalEncoding_batched():
    encoding = PositionalEncoding(16, scale=4.0)

    x = torch.rand(5, 64)
    result = encoding(x)
    assert result.size() == torch.Size([5, 64, 16])

    expected_result = torch.stack([encoding(event) for event in x])
    assert result.numpy() == pytest.approx(expected_result.numpy(), abs=1e-6)

def test_PositionalEncoding_requires_even_dimension():
    with pytest.raises(ValueError):
        PositionalEncoding(15, scale=4.0)

def test_GaussianFourierFeatures():
    dimension = 32
    half_dimension = dimension // 2
    scales = (2.0, 4.0, 8.0)
    encoding = GaussianFourierFeatures(dimension, input_dim=3, scales=scales, seed=0)

    assert not isinstance(encoding.frequency_table, torch.nn.Parameter)
    assert list(encoding.parameters()) == []
    assert encoding.frequency_table.size() == torch.Size([half_dimension, 3])

    x = torch.rand(64, 3)
    result = encoding(x)
    assert result.size() == torch.Size([64, dimension])

    thetas = x @ encoding.frequency_table.T
    expected_result = torch.cat((torch.sin(thetas), torch.cos(thetas)), dim=-1)
    assert result.numpy() == pytest.approx(expected_result.numpy(), abs=1e-6)

    empirical_std = encoding.frequency_table.std(dim=0) / (2 * torch.pi)
    assert empirical_std.numpy() == pytest.approx(np.array(scales), rel=0.5)

def test_GaussianFourierFeatures_scalar_scale_broadcasts():
    encoding = GaussianFourierFeatures(16, input_dim=3, scales=4.0, seed=0)
    assert encoding.frequency_table.size() == torch.Size([8, 3])

def test_GaussianFourierFeatures_reproducible_with_seed():
    a = GaussianFourierFeatures(16, input_dim=3, scales=4.0, seed=42)
    b = GaussianFourierFeatures(16, input_dim=3, scales=4.0, seed=42)
    assert a.frequency_table.numpy() == pytest.approx(b.frequency_table.numpy())

def test_GaussianFourierFeatures_batched():
    encoding = GaussianFourierFeatures(16, input_dim=3, scales=4.0, seed=0)

    x = torch.rand(5, 64, 3)
    result = encoding(x)
    assert result.size() == torch.Size([5, 64, 16])

    expected_result = torch.stack([encoding(event) for event in x])
    assert result.numpy() == pytest.approx(expected_result.numpy(), abs=1e-6)

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

def test_EquivariantDenoiser_predict_variances():
    model = EquivariantDenoiser(
        n_timesteps = 25,
        tau_encoding_dimension = 8,
        position_encoding_dimension = 8,
        hidden_layer_size = 32,
        n_hidden_layers = 1,
        predict_variances = True,
    )

    tau = torch.tensor(12)
    input_set = torch.rand((24, 4))
    mu, var = model(input_set, tau=tau)
    assert mu.size() == torch.Size((24, 4))
    assert var.size() == torch.Size((24, 4))
    assert (var.detach() >= 0).all()

@pytest.mark.parametrize("kind", ["learned", "positional", "gaussian"])
def test_EquivariantDenoiser_position_encoding(kind):
    model = EquivariantDenoiser(
        n_timesteps = 25,
        tau_encoding_dimension = 8,
        position_encoding_dimension = 8,
        hidden_layer_size = 32,
        n_hidden_layers = 1,
        use_position_encoding = True,
        position_encoding_kind = kind,
        position_encoding_scale = (1.0, 2.0, 4.0) if kind != "learned" else None,
    )

    tau = torch.tensor(12)
    input_set = torch.rand((24, 4))
    output_set = model(input_set, tau=tau)
    assert output_set.size() == torch.Size((24, 4))

    transpose_idx = torch.randperm(24)
    transposed_output_set = model(input_set[transpose_idx], tau)
    assert output_set[transpose_idx].detach().numpy() == pytest.approx(transposed_output_set.detach().numpy(), abs=1e-4)

    output_set.sum().backward()
    if kind == "learned":
        assert model.pos1_encoding.frequency_table.grad is not None
    elif kind == "positional":
        assert not model.pos1_encoding.frequency_table.requires_grad
        assert model.pos1_encoding.frequency_table.grad is None
    elif kind == "gaussian":
        assert not model.pos_encoding.frequency_table.requires_grad
        assert model.pos_encoding.frequency_table.grad is None

def _make_gaussian_model(seed):
    return EquivariantDenoiser(
        n_timesteps = 25,
        tau_encoding_dimension = 8,
        position_encoding_dimension = 8,
        hidden_layer_size = 32,
        n_hidden_layers = 1,
        use_position_encoding = True,
        position_encoding_kind = "gaussian",
        position_encoding_scale = (1.0, 2.0, 4.0),
        position_encoding_seed = seed,
    )

def test_EquivariantDenoiser_gaussian_encoding_seed_reproducible():
    model_a = _make_gaussian_model(seed=7)
    model_b = _make_gaussian_model(seed=7)
    assert model_a.pos_encoding.frequency_table.numpy() == pytest.approx(model_b.pos_encoding.frequency_table.numpy())

def test_EquivariantDenoiser_gaussian_encoding_no_seed_varies():
    model_a = _make_gaussian_model(seed=None)
    model_b = _make_gaussian_model(seed=None)
    assert model_a.pos_encoding.frequency_table.numpy() != pytest.approx(model_b.pos_encoding.frequency_table.numpy())

def test_EquivariantDenoiser_gaussian_encoding_requires_scale():
    with pytest.raises(ValueError):
        EquivariantDenoiser(
            n_timesteps = 25,
            tau_encoding_dimension = 8,
            position_encoding_dimension = 8,
            hidden_layer_size = 32,
            n_hidden_layers = 1,
            use_position_encoding = True,
            position_encoding_kind = "gaussian",
        )

def test_EquivariantDenoiser_predict_variances_batched():
    model = EquivariantDenoiser(
        n_timesteps = 25,
        tau_encoding_dimension = 8,
        position_encoding_dimension = 8,
        hidden_layer_size = 32,
        n_hidden_layers = 1,
        predict_variances = True,
    )

    tau = torch.tensor([12, 13, 14, 15, 16])
    input_set = torch.rand((5, 24, 4))
    mu, var = model(input_set, tau=tau)
    assert mu.size() == torch.Size((5, 24, 4))
    assert var.size() == torch.Size((5, 24, 4))
    assert (var.detach() >= 0).all()