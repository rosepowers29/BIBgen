from __future__ import annotations

import h5py
import numpy as np
import pytest
import torch

from BIBgen.generation import Generator, infer_log_energy
from BIBgen.preprocessing import Sphering


def _physical_events(rng, n_events):
    """Events in physical units (E [GeV], phi, s, z) with BIB-like magnitudes."""
    events = []
    for _ in range(n_events):
        n = int(rng.integers(20, 60))
        events.append(np.stack([
            rng.lognormal(np.log(4.3e-5), 1.0, n),
            rng.uniform(-np.pi / 4, np.pi / 4, n),
            rng.uniform(1860, 2180, n),
            rng.uniform(-2300, 2300, n),
        ], axis=-1))
    return events


def _write_training_file(path, log_energy, stamp_attr=True):
    """Write a small file with the raw_cyl_* layout and return its sphering."""
    rng = np.random.default_rng(0)
    groups = {g : _physical_events(rng, 10) for g in ("training", "validation")}
    if log_energy:
        for evts in groups.values():
            for e in evts:
                e[:, 0] = np.log(e[:, 0])
    sphering = Sphering.from_data(np.concatenate(groups["training"]))
    with h5py.File(path, "w") as f:
        for g, evts in groups.items():
            for i, e in enumerate(evts):
                f.create_dataset(f"{g}/evt_{i}/tau0", data=sphering.transform(e))
        f["transformation/mu"] = sphering.mu
        f["transformation/std"] = sphering.std
        if stamp_attr:
            f["transformation"].attrs["log_energy"] = log_energy
    return sphering, [e.shape[0] for evts in groups.values() for e in evts]


@pytest.mark.parametrize("log_energy", [True, False])
def test_cache_roundtrip(tmp_path, log_energy):
    _, sizes = _write_training_file(tmp_path / "train.hdf5", log_energy)
    gen = Generator()
    gen.calculate_cache(str(tmp_path / "train.hdf5"))
    gen.save_cache(str(tmp_path / "cache.npz"))

    loaded = Generator()
    loaded.load_cache(str(tmp_path / "cache.npz"))
    assert loaded.log_energy is log_energy
    assert np.array_equal(loaded.sorted_nhits, np.sort(sizes))
    assert np.allclose(loaded.sphering.mu, gen.sphering.mu)
    assert np.allclose(loaded.sphering.std, gen.sphering.std)


@pytest.mark.parametrize("log_energy", [True, False])
def test_missing_attribute_is_inferred(tmp_path, log_energy):
    _write_training_file(tmp_path / "train.hdf5", log_energy, stamp_attr=False)
    gen = Generator()
    with pytest.warns(UserWarning, match="no log_energy"):
        gen.calculate_cache(str(tmp_path / "train.hdf5"))
    assert gen.log_energy is log_energy


def test_old_cache_without_log_energy(tmp_path):
    np.savez(tmp_path / "old.npz", sorted_nhits=np.array([1, 2]), mu=np.array([-10.0, 0, 2000, 0]),
             std=np.ones(4))
    gen = Generator()
    with pytest.warns(UserWarning, match="no log_energy"):
        gen.load_cache(str(tmp_path / "old.npz"))
    assert gen.log_energy is True


def test_infer_log_energy():
    assert infer_log_energy(np.array([-10.1, 0.0, 2000.0, 0.0]))
    assert not infer_log_energy(np.array([1.6e-4, 0.0, 2000.0, 0.0]))


@pytest.mark.parametrize("log_energy", [True, False])
def test_unsphere_recovers_physical_units(tmp_path, log_energy):
    sphering, _ = _write_training_file(tmp_path / "train.hdf5", log_energy)
    gen = Generator()
    gen.calculate_cache(str(tmp_path / "train.hdf5"))

    physical = _physical_events(np.random.default_rng(1), 1)[0]
    stored = physical.copy()
    if log_energy:
        stored[:, 0] = np.log(stored[:, 0])
    sphered = torch.from_numpy(sphering.transform(stored)).float()

    recovered = gen.unsphere(sphered)
    assert recovered.dtype == np.float32
    assert recovered == pytest.approx(physical, rel=1e-5)
    assert (recovered[:, 0] > 0).all()
