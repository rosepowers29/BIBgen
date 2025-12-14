import os

import pytest
import h5py
import numpy as np

from BIBgen.training import *
from BIBgen.loss_function import *
from BIBgen.models import *

def create_dummy_dataset(nevents : int, ntau : int, random : bool = True):
    dummy_fname = "_dummy_file.hdf5"

    with h5py.File(dummy_fname, "w") as fout:
        train_group = fout.create_group("training")
        for event_no in range(nevents):
            pseudodata = np.random.rand(ntau, 10, 4) if random else np.array([np.ones((10, 4)) * t for t in range(ntau)])
            train_group.create_dataset("evt_{}".format(event_no), data=pseudodata)

    return dummy_fname

def test_DataLoader():
    dummy_fname = create_dummy_dataset(2, 11, random=False)

    with h5py.File(dummy_fname, "r") as fin:
        dataloader = DataLoader(fin, "training")
        
        for epoch in range(5):
            instances = np.zeros(10)
            for istep in range(dataloader.nsteps):
                x, y, tau = next(dataloader)
                assert x == pytest.approx(y + 1)
                instances[tau] += 1
            assert np.all(instances == 2)

    os.remove(dummy_fname)

def test_train():
    device = torch.device("cpu")

    model = EquivariantDenoiser(
        n_timesteps = 10,
        tau_encoding_dimension = 4,
        position_encoding_dimension = 4,
        hidden_layer_size = 8,
        n_hidden_layers = 1
    ).to(device)
    gaussian_nll = GaussianNLLLoss()
    loss_fn = lambda pred, y: gaussian_nll(*pred, y)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-7)

    dummy_fname = create_dummy_dataset(2, 11)

    with h5py.File(dummy_fname, "r") as fin:
        dataloader = DataLoader(fin, "training")

        train(dataloader, model, loss_fn, optimizer, device)

    os.remove(dummy_fname)

def test_evaluate():
    device = torch.device("cpu")

    model = EquivariantDenoiser(
        n_timesteps = 10,
        tau_encoding_dimension = 4,
        position_encoding_dimension = 4,
        hidden_layer_size = 8,
        n_hidden_layers = 1
    ).to(device)
    gaussian_nll = GaussianNLLLoss()
    loss_fn = lambda pred, y: gaussian_nll(*pred, y)

    dummy_fname = create_dummy_dataset(2, 11)

    with h5py.File(dummy_fname, "r") as fin:
        dataloader = DataLoader(fin, "training")

        evaluate(dataloader, model, loss_fn, device)

    os.remove(dummy_fname)