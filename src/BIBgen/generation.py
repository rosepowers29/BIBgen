from __future__ import annotations

import warnings
from collections.abc import Iterator

import h5py
import numpy as np
import torch
from torch import nn

from BIBgen.preprocessing import Sphering


def generate_sphered(
    model : nn.Module,
    n_members : int,
    device : torch.device,
    schedule : torch.Tensor | None = None,
    demo : bool = False,
    verbosity : int = 0
):
    """
    Run the denoising procedure on white noise.
    Does not reverse the preprocessing on the output in any way.
    Requires a fixed number of hits

    Parameters
    ----------
    model : nn.Module
        Denoising model
    n_members : int
        Number of hits to appear in this event
    device : torch.Device
        device on which to perform, usually torch.device("cuda") or torch.device("cpu")
    schedule : torch.Tensor | None
        If model does not predict variances, then the noise schedule msut be provided in lieu of it
    demo : bool
        While denoising, whether to sample from N(mu, std) or just choose mu. Note setting the latter often risks collapse of the distribution.
    verbosity : bool
        Print diagnostics for white noise and step wise results

    Returns
    -------
    event : torch.Tensor
        The generated event with shape (n_members, 4)

    Raises
    ------
    ValueError
        If model does not predict variances and schedule is not provided
    RuntimeError
        If variances predicted by model become non-finite
    """
    model = model.to(device)
    current_event = torch.normal(mean=0, std=1, size=(n_members, 4), device=device)
    if verbosity >= 1:
        print("White noise:", current_event[:10])

    with torch.inference_mode():
        for tau in torch.arange(model.n_timesteps-1, -1, -1, device=device):
            result = model(current_event, tau)

            if schedule is None:
                if len(result) != 2:
                    raise ValueError("If schedule is not provided, result should be mu and var")
                mu, var = result
                if not torch.isfinite(var).all():
                    print("Non-finite var at tau", tau, flush=True)
                    raise RuntimeError("var has NaN/Inf")
                if (var <= 0).any():
                    print("Non-positive var at tau", tau, "min", var.min().item(), flush=True)
            else:
                mu = result
                var = schedule[tau]

            current_event = torch.normal(mean=mu, std=torch.sqrt(var)) if not demo else mu

            if verbosity >= 2:
                print("tau", tau,
                "mu mean", mu.abs().mean().item(), "mu max", mu.abs().max().item(),
                "var mean", var.mean().item(), "var max", var.max().item(),
                "x max", current_event.abs().max().item(),
                "x mean", current_event.abs().mean().item(),
                flush=True)

    return current_event

def infer_log_energy(mu: np.ndarray) -> bool:
    """
    Infer whether the energy feature was stored as ln(E), for files that predate the
    `log_energy` attribute.

    BIB hit energies are far below 1 GeV, so the mean of ln(E) is negative while the
    mean of raw E is positive.

    Parameters
    ----------
    mu : numpy.ndarray
        Sphering mean with the energy feature first

    Returns
    -------
    log_energy : bool
        True if the energy feature looks like ln(E)

    Examples
    --------
    >>> import numpy as np
    >>> infer_log_energy(np.array([-10.1, 0.0, 2000.0, 0.0]))
    True
    >>> infer_log_energy(np.array([1.6e-4, 0.0, 2000.0, 0.0]))
    False
    """
    return bool(mu[0] < 0)


class Generator:
    """
    Generate an arbitrary number of events in physical units, with sizes drawn from
    the empirical hit-count distribution of a training file.

    The hit-count distribution, the sphering transformation and the `log_energy` flag
    are computed once with `calculate_cache` and stored with `save_cache`, so
    generation jobs only need the small `.npz` cache, not the (large) training HDF5.
    """

    def __init__(self) -> None:
        self.model: nn.Module | None = None
        self.schedule: torch.Tensor | None = None
        self.device: torch.device | None = None

        self.sphering: Sphering | None = None
        self.log_energy: bool | None = None
        self.sorted_nhits: np.ndarray | None = None

    def load_model(self, model: nn.Module, device: torch.device, schedule: torch.Tensor | None = None) -> None:
        """
        Attach a trained denoiser.

        Parameters
        ----------
        model : nn.Module
            Denoising model
        device : torch.device
            Device on which to generate
        schedule : torch.Tensor | None
            Noise schedule; required if the model does not predict variances
        """
        self.model = model
        self.schedule = schedule
        self.device = device

    def calculate_cache(self, training_file: str) -> None:
        """
        Read the hit-count distribution, sphering and `log_energy` flag from a training file.

        Only dataset shapes are read, so no hit data is loaded into memory. If the file
        predates the `log_energy` attribute, the flag is inferred with `infer_log_energy`
        and a warning is issued.

        Parameters
        ----------
        training_file : str
            Path to a `raw_*` or `diffused_*` HDF5 file with `training`, `validation`
            and `transformation` groups
        """
        assert training_file.endswith(".hdf5")
        nhits: list[int] = []
        with h5py.File(training_file, "r") as f:
            for group in ("training", "validation"):
                nhits.extend(f[group][evt]["tau0"].shape[0] for evt in f[group])

            mu = np.array(f["transformation/mu"])
            std = np.array(f["transformation/std"])
            stored_log_energy = f["transformation"].attrs.get("log_energy")

        self.sphering = Sphering(mu, std)
        if stored_log_energy is None:
            self.log_energy = infer_log_energy(mu)
            warnings.warn(
                f"{training_file} has no log_energy attribute; inferred log_energy={self.log_energy} "
                f"from energy mean {mu[0]:.3g}",
                stacklevel=2,
            )
        else:
            self.log_energy = bool(stored_log_energy)
        self.sorted_nhits = np.sort(np.array(nhits))

    def save_cache(self, outpath: str) -> None:
        """
        Save the hit-count distribution, sphering and `log_energy` flag to an `.npz` file.

        Raises
        ------
        ValueError
            If `calculate_cache` or `load_cache` has not been run
        """
        if self.sorted_nhits is None or self.sphering is None or self.log_energy is None:
            msg = "No cache to be saved; run calculate_cache or load_cache first"
            raise ValueError(msg)
        assert outpath.endswith(".npz")
        np.savez(
            outpath,
            sorted_nhits=self.sorted_nhits,
            mu=self.sphering.mu,
            std=self.sphering.std,
            log_energy=self.log_energy,
        )

    def load_cache(self, inpath: str) -> None:
        """
        Load a cache written by `save_cache`.

        Caches written before `log_energy` was stored have it inferred with
        `infer_log_energy`, with a warning.
        """
        assert inpath.endswith(".npz")
        with np.load(inpath) as fin:
            self.sorted_nhits = fin["sorted_nhits"]
            mu = fin["mu"]
            self.sphering = Sphering(mu, fin["std"])
            if "log_energy" in fin.files:
                self.log_energy = bool(fin["log_energy"])
            else:
                self.log_energy = infer_log_energy(mu)
                warnings.warn(
                    f"{inpath} has no log_energy entry; inferred log_energy={self.log_energy}",
                    stacklevel=2,
                )

    def unsphere(self, sphered: torch.Tensor) -> np.ndarray:
        """
        Map a sphered event back to physical units `(E [GeV], phi, s [mm], z [mm])`.

        Undoes the sphering, then exponentiates the energy feature if it was stored as ln(E).

        Parameters
        ----------
        sphered : torch.Tensor
            Sphered event of shape `(n_hits, 4)`

        Returns
        -------
        event : numpy.ndarray
            Event of shape `(n_hits, 4)` in physical units, float32
        """
        if self.sphering is None or self.log_energy is None:
            msg = "Cache must be calculated or loaded before unsphering"
            raise ValueError(msg)
        mu = np.asarray(self.sphering.mu, dtype=np.float64)
        std = np.asarray(self.sphering.std, dtype=np.float64)
        event = std * sphered.cpu().numpy().astype(np.float64) + mu  # Sphering.untransform
        if self.log_energy:
            event[:, 0] = np.exp(event[:, 0])
        return event.astype(np.float32)

    def generate(self, nevents: int, rng: np.random.Generator | None = None) -> Iterator[np.ndarray]:
        """
        Generate events lazily, one at a time, so that memory does not grow with `nevents`.

        Event sizes are sampled from a piecewise-linear interpolation of the empirical
        quantile function of the cached hit counts. Each event is unsphered with `unsphere`,
        so the output is directly comparable to the `test` group with `exp` applied to
        its energy if `log_energy` is set.

        Parameters
        ----------
        nevents : int
            Number of events to generate
        rng : numpy.random.Generator | None
            Source of randomness for the event sizes (fresh entropy if None)

        Yields
        ------
        event : numpy.ndarray
            Event of shape `(n_hits, 4)` in physical units `(E [GeV], phi, s [mm], z [mm])`
        """
        if self.sorted_nhits is None or self.sphering is None or self.log_energy is None:
            msg = "Cache must be calculated or loaded before generating"
            raise ValueError(msg)
        if self.model is None or self.device is None:
            msg = "Model must be loaded before generating"
            raise ValueError(msg)
        rng = rng if rng is not None else np.random.default_rng()

        n_training_events = len(self.sorted_nhits)
        cum_prob = (np.arange(n_training_events) + 0.5) / n_training_events
        gen_nhits = np.interp(rng.uniform(size=nevents), cum_prob, self.sorted_nhits).round().astype(int)

        for n in gen_nhits:
            yield self.unsphere(generate_sphered(self.model, int(n), self.device, self.schedule))
