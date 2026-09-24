import os

import pytest
import h5py
import numpy as np

from BIBgen.training import *
from BIBgen.losses import *
from BIBgen.models import *

def create_dummy_dataset(nevents : int, ntau : int, random : bool = True):
    dummy_fname = "_dummy_file.hdf5"

    with h5py.File(dummy_fname, "w") as fout:
        train_group = fout.create_group("training")
        for event_no in range(nevents):
            pseudodata = np.random.rand(ntau + 1, 10, 4) if random else np.array([np.ones((10, 4)) * t for t in range(ntau + 1)])
            train_group.create_dataset("evt_{}".format(event_no), data=pseudodata)

    schedule = torch.arange(1, ntau + 1) * 1e-3
    return dummy_fname, schedule

def test_UnbatchedDataLoader():
    dummy_fname, _ = create_dummy_dataset(2, 10, random=False)

    with h5py.File(dummy_fname, "r") as fin:
        dataloader = UnbatchedDataLoader(fin, "training")

        for epoch in range(5):
            instances = np.zeros(10)
            for istep in range(dataloader.nsteps):
                x, y, tau, x_0 = next(dataloader)
                assert x == pytest.approx(y + 1)
                assert x_0 == pytest.approx(np.zeros((10, 4)))
                instances[tau] += 1
            assert np.all(instances == 2)

    os.remove(dummy_fname)

def test_BatchedDataLoader():
    dummy_fname, _ = create_dummy_dataset(4, 10, random=False)

    with h5py.File(dummy_fname, "r") as fin:
        dataloader = BatchedDataLoader(fin, "training", batch_size=2)

        for epoch in range(5):
            instances = np.zeros(10)
            for istep in range(dataloader.nsteps):
                x, y, tau, x_0 = next(dataloader)
                assert x == pytest.approx(y + 1)
                assert x_0 == pytest.approx(np.zeros((10, 4)))
                instances[tau] += 1
            assert np.all(instances == 4)

    os.remove(dummy_fname)

class _CountingFile:
    """Wraps an h5py.File and counts top-level __getitem__ calls, so tests can
    check how many times raw event data was actually read off disk."""
    def __init__(self, real):
        self._real = real
        self.event_read_calls = {}

    def __getitem__(self, key):
        self.event_read_calls[key] = self.event_read_calls.get(key, 0) + 1
        return self._real[key]

def test_BatchedDataLoader_caches_event_reads():
    nevents, ntau, batch_size = 3, 10, 2
    dummy_fname, _ = create_dummy_dataset(nevents, ntau, random=False)

    with h5py.File(dummy_fname, "r") as fin:
        counting_file = _CountingFile(fin)
        dataloader = BatchedDataLoader(counting_file, "training", batch_size=batch_size)

        event_paths = [f"training/evt_{i}" for i in range(nevents)]
        # __init__ already reads each event once or twice (nhits lookup, plus an
        # extra ntau lookup for event 0) -- capture that baseline rather than
        # assuming a fixed count.
        counts_after_init = {path: counting_file.event_read_calls.get(path, 0) for path in event_paths}
        assert all(count >= 1 for count in counts_after_init.values())

        for epoch in range(3):
            for istep in range(dataloader.nsteps):
                next(dataloader)

        # Each event is visited nbatches(event_id) times per epoch (5 times/epoch
        # here, x3 epochs = 15 visits total), but the cache is empty at __init__
        # time, so __next__ should trigger exactly one more read per event (to
        # populate the cache on first touch) regardless of epoch count -- every
        # visit after that first one is served from the in-memory cache, not a
        # fresh HDF5 read.
        for path in event_paths:
            assert counting_file.event_read_calls.get(path, 0) == counts_after_init[path] + 1

    os.remove(dummy_fname)

def test_train():
    dummy_fname, schedule = create_dummy_dataset(2, 10)
    device = torch.device("cpu")

    for dl_class, kwargs in ((UnbatchedDataLoader, {}), (BatchedDataLoader, {"batch_size" : 5})):
        model = EquivariantDenoiser(
            n_timesteps = 10,
            tau_encoding_dimension = 4,
            position_encoding_dimension = 4,
            hidden_layer_size = 8,
            n_hidden_layers = 1
        ).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-7)

        gaussian_nll = GaussianNLLLoss()
        loss_fn = lambda pred, X, y, tau, x_0: gaussian_nll(pred, schedule[tau], y)
        with h5py.File(dummy_fname, "r") as fin:
            dataloader = dl_class(fin, "training", **kwargs)

            train(dataloader, model, loss_fn, optimizer, device)

    os.remove(dummy_fname)

def test_evaluate():
    dummy_fname, schedule = create_dummy_dataset(2, 10)
    device = torch.device("cpu")

    for dl_class, kwargs in ((UnbatchedDataLoader, {}), (BatchedDataLoader, {"batch_size" : 5})):
        model = EquivariantDenoiser(
            n_timesteps = 10,
            tau_encoding_dimension = 4,
            position_encoding_dimension = 4,
            hidden_layer_size = 8,
            n_hidden_layers = 1
        ).to(device)

        gaussian_nll = GaussianNLLLoss()
        loss_fn = lambda pred, X, y, tau, x_0: gaussian_nll(pred, schedule[tau], y)
        with h5py.File(dummy_fname, "r") as fin:
            dataloader = dl_class(fin, "training", **kwargs)

            test_loss = evaluate(dataloader, model, loss_fn, device)
            assert np.isfinite(test_loss), test_loss
    
    os.remove(dummy_fname)