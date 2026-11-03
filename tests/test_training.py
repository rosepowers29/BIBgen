import os

import pytest
import h5py
import numpy as np

from BIBgen.training import *
from BIBgen.losses import *
from BIBgen.models import *
from BIBgen.features import N_SCALAR_FEATURES

def create_dummy_dataset(nevents : int, ntau : int, random : bool = True):
    dummy_fname = "_dummy_file.hdf5"

    with h5py.File(dummy_fname, "w") as fout:
        train_group = fout.create_group("training")
        for event_no in range(nevents):
            pseudodata = np.random.rand(ntau + 1, 10, N_SCALAR_FEATURES) if random else np.array([np.ones((10, N_SCALAR_FEATURES)) * t for t in range(ntau + 1)])
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
                x, y, tau = next(dataloader)
                assert x == pytest.approx(y + 1)
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
                x, y, tau = next(dataloader)
                assert x == pytest.approx(y + 1)
                instances[tau] += 1
            assert np.all(instances == 4)

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
        loss_fn = lambda pred, y, tau: gaussian_nll(pred, schedule[tau], y)
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
        loss_fn = lambda pred, y, tau: gaussian_nll(pred, schedule[tau], y)
        with h5py.File(dummy_fname, "r") as fin:
            dataloader = dl_class(fin, "training", **kwargs)

            test_loss = evaluate(dataloader, model, loss_fn, device)
            assert np.isfinite(test_loss), test_loss
    
    os.remove(dummy_fname)
TINY_TEMPLATE = {
    "name" : "EquivariantDenoiser",
    "hyperparameters" : {
        "tau_encoding_dimension" : 4,
        "position_encoding_dimension" : 4,
        "hidden_layer_size" : {"type" : "int", "exponentiate" : True, "range" : [2, 3]},
        "n_hidden_layers" : {"type" : "int", "range" : [1, 2]},
    },
}

def run_tiny_hyperoptimize(storage, ntrials, timeout=None, loss_fn=None):
    dummy_fname, schedule = create_dummy_dataset(2, 10)
    gaussian_nll = GaussianNLLLoss()
    if loss_fn is None:
        loss_fn = lambda pred, y, tau: gaussian_nll(pred, schedule[tau], y)
    with h5py.File(dummy_fname, "r") as fin:
        loader = BatchedDataLoader(fin, "training", batch_size=5)
        result = hyperoptimize(
            TINY_TEMPLATE, loader, loader, loss_fn, torch.device("cpu"), n_timesteps=10,
            nepochs=1, ntrials=ntrials, storage=storage, study_name="test", timeout=timeout
        )
    os.remove(dummy_fname)
    return result

def test_hyperoptimize_resumes_to_fixed_total(tmp_path):
    import time
    import optuna

    storage = "sqlite:///" + str(tmp_path / "study.db")
    schedule = torch.arange(1, 11) * 1e-3
    gaussian_nll = GaussianNLLLoss()
    def slow_loss_fn(pred, y, tau):
        # 8 calls per trial (4 training + 4 validation batches), so a trial takes ~0.8 s
        time.sleep(0.1)
        return gaussian_nll(pred, schedule[tau], y)

    # A timeout already passed starts no trial at all
    assert run_tiny_hyperoptimize(storage, ntrials=3, timeout=0) == (None, None, 0)

    # Each run starts one trial before its 0.3 s timeout and lets it finish, so the study
    # accumulates across runs and stops at the total rather than restarting
    for ndone in (1, 2, 3, 3):
        best_val_loss, best_config, ntrials_done = run_tiny_hyperoptimize(storage, ntrials=3, timeout=0.3, loss_fn=slow_loss_fn)
        assert ntrials_done == ndone
        assert np.isfinite(best_val_loss)

    study = optuna.load_study(study_name="test", storage=storage)
    assert len(study.trials) == 3
    assert best_config["name"] == "EquivariantDenoiser"
    assert best_config["hyperparameters"]["hidden_layer_size"] == 2**study.best_params["hidden_layer_size"]
    assert best_config["hyperparameters"]["position_encoding_dimension"] == 4

def test_hyperoptimize_retries_interrupted_trial(tmp_path):
    import optuna
    from optuna.trial import TrialState

    storage = "sqlite:///" + str(tmp_path / "study.db")
    # A trial left running, as if the job had been killed partway through it
    study = optuna.create_study(storage=storage, study_name="test", direction="minimize")
    trial = study.ask()
    params = {"hidden_layer_size" : trial.suggest_int("hidden_layer_size", 2, 3), "n_hidden_layers" : trial.suggest_int("n_hidden_layers", 1, 2)}

    _, _, ntrials_done = run_tiny_hyperoptimize(storage, ntrials=2)
    assert ntrials_done == 2

    trials = optuna.load_study(study_name="test", storage=storage).trials
    assert len(trials) == 3  # the interrupted trial does not count towards the total
    assert trials[0].state == TrialState.FAIL and "error" not in trials[0].user_attrs
    retry = [t for t in trials if t.user_attrs.get("retry_of") == trials[0].number]
    assert len(retry) == 1 and retry[0].state == TrialState.COMPLETE and retry[0].params == params

def test_hyperoptimize_counts_errored_trials(tmp_path):
    import optuna
    from optuna.trial import TrialState

    def failing_loss_fn(pred, y, tau):
        raise RuntimeError("Nonfinite test loss")

    storage = "sqlite:///" + str(tmp_path / "study.db")
    assert run_tiny_hyperoptimize(storage, ntrials=2, loss_fn=failing_loss_fn) == (None, None, 2)
    trials = optuna.load_study(study_name="test", storage=storage).trials
    assert len(trials) == 2 and all(t.state == TrialState.FAIL and "error" in t.user_attrs for t in trials)
