from typing import Callable

import h5py
import torch
import numpy as np

class BaseDataLoader:
    def __init__(self, infile : h5py.File, key : str, shuffle : bool = True):
        self.infile = infile
        self.key = key
        self.shuffle = shuffle

        self.event_ids = list(infile[key].keys())
        self.nevents = len(self.event_ids)
        self.ntau = len(infile[key + "/" + self.event_ids[0]]) - 1

    def __next__(self):
        raise NotImplementedError

class UnbatchedDataLoader(BaseDataLoader):
    def __init__(self, infile : h5py.File, key : str, shuffle : bool = True):
        """
        Class that is an iterator on shuffled diffused pairs for training a denoising network.

        Parameters
        ----------
        infile : h5py.File
            File object from which to read
        key : str
            Overall dataset key, e.g. "training", "validation", with which to index infile
        shuffle : bool = True
            Whether to shuffle the pairs when iterating or return in order

        Returns
        -------
        self : DataLoader
            Iterator that returns pairs of tau+1 and tau examples from the file
        """
        super().__init__(infile, key, shuffle)

        self.idx = self.nevents * self.ntau

    @property
    def nsteps(self):
        """
        Total number of training steps, number of events * number of diffusion steps
        """
        return self.nevents * self.ntau

    def __next__(self):
        """
        Goes through every possible pair of diffusion timesteps.
        After going through every one, reshuffles if shuffl=True in the constructor.

        Returns
        -------
        tau_plus1 : torch.Tensor
            Event at diffusion timestep tau+1
        tau : torch.Tensor
            Event at diffusion timestep tau
        """
        if self.idx >= self.nsteps:
            self.shuffle_order = torch.randperm(self.nsteps) if self.shuffle else torch.arange(self.nsteps)
            self.idx = 0

        event_no = self.shuffle_order[self.idx] // self.ntau
        tau = self.shuffle_order[self.idx] % self.ntau
        self.idx += 1

        event = self.infile[self.key + "/" + self.event_ids[event_no]]
        return torch.from_numpy(event[tau+1]).to(dtype=torch.float32), torch.from_numpy(event[tau]).to(dtype=torch.float32), tau

class BatchedDataLoader(BaseDataLoader):
    def __init__(self,
        infile : h5py.File,
        key : str,
        batch_size : int | None = None,
        hit_budget : int | None = None,
        shuffle : bool = True
    ):
        if batch_size is None and hit_budget is None:
            raise ValueError("One of batch_size or hit_budget must be provided")
        super().__init__(infile, key, shuffle)

        if batch_size is None:
            possible_factors = np.arange(1, round(np.sqrt(self.ntau)))
            self.ntau_factors = possible_factors[self.ntau % possible_factors == 0]
        else:
            if self.ntau % batch_size != 0:
                raise ValueError("Number of diffusion steps must be divisible by batch_size")

        self.batch_size_per_event = {}
        self.nhits_per_event = []
        for event_id in self.event_ids:
            nhits = len(infile[key + "/" + event_id][0])
            self.nhits_per_event.append(nhits)
            
            if batch_size is not None:
                self.batch_size_per_event[event_id] = batch_size
            else:
                possible_batch_sizes = self.ntau_factors * infile[self.key + "/" + event_id].shape[1]
                valid_batch_sizes = possible_batch_sizes[possible_batch_sizes <= hit_budget]
                self.batch_size_per_event[event_id] = 1 if valid_batch_sizes.shape[0] == 0 else valid_batch_sizes[-1]

        self.nsteps = 0
        for event_id in self.batch_size_per_event:
            self.nsteps += self.nbatches(event_id)

        self.idx = self.nsteps

    def nbatches(self, event_id):
        return self.ntau // self.batch_size_per_event[event_id]
        
    def __next__(self):
        if self.idx >= self.nsteps:
            self.shuffle_order = torch.randperm(self.nsteps) if self.shuffle else torch.arange(self.nsteps)
            self.batch_indices = {
                event_id : torch.reshape(
                    torch.randperm(self.ntau), (self.nbatches(event_id), self.batch_size_per_event[event_id])
                ) for event_id in self.batch_size_per_event
            }

            self.paths = []
            for event_id in self.event_ids:
                for i in range(self.nbatches(event_id)):
                    self.paths.append((event_id, i))

            assert len(self.paths) == self.nsteps
            self.idx = 0

        event_id, batch_no = self.paths[self.idx]
        batch_taus = self.batch_indices[event_id][batch_no]
        self.idx += 1

        event = self.infile[self.key + "/" + event_id][:]
        return torch.from_numpy(event[batch_taus + 1]).to(dtype=torch.float32), torch.from_numpy(event[batch_taus]).to(dtype=torch.float32), batch_taus
        # return event[batch_taus + 1].to(dtype=torch.float32), event[batch_taus].to(dtype=torch.float32), batch_taus

def train(
    dataloader : BaseDataLoader,
    model : torch.nn.Module,
    loss_fn : Callable,
    optimizer : torch.optim.Optimizer,
    device : torch.device,
    scaler : torch.amp.GradScaler | None = None
):
    """
    Trains the model for one epoch.

    Parameters
    ----------
    dataloader : DataLoader
        dataloader for training data
    model : torch.nn.Module
        Denoising model
    loss_fn : Callable
        loss function to be called directly on model output, i.e. loss_fn(pred, y, tau) instead of loss_fn(mu, std, y)
    optimizer : torch.optim.Optimizer
        Optimizer for gradient descent
    device : torch.device
        device on which to perform, usually torch.device("cuda") or torch.device("cpu")

    Returns
    -------
    loss : float
        Training loss on the final iteration
    """
    # assert scaler is not None or device.type != "cuda"
    model.train()
    for istep in range(dataloader.nsteps):
        X, y, tau = next(dataloader)
        X, y, tau = X.to(device), y.to(device), tau.to(device)

        # Compute prediction error
        if scaler is not None:
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                pred = model(X, tau)
                loss = loss_fn(pred, y, tau)
        else:
            pred = model(X, tau)
            loss = loss_fn(pred, y, tau)

        # Backpropagation
        try:
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        except torch.cuda.OutOfMemoryError:
            print("Failed step {}, cuda out of memory".format(istep))
            print("X.size() : {}, y.size() : {}".format(X.size(), y.size()))
            print("pred.size() : {}".format(pred.size()))
            raise RuntimeError("Intentional exit")

    optimizer.zero_grad()
    return loss.item()

def evaluate(dataloader : BaseDataLoader, model : torch.nn.Module, loss_fn : Callable, device : torch.device):
    """
    Parameters
    ----------
    dataloader : DataLoader
        dataloader for training data
    model : torch.nn.Module
        Denoising model
    loss_fn : Callable
        loss function to be called directly on model output, i.e. loss_fn(pred, y, tau) instead of loss_fn(mu, std, y)
    device : torch.device
        device on which to perform, usually torch.device("cuda") or torch.device("cpu")

    Returns
    -------
    test_loss : float
        Average loss over all validation events
    """

    model.eval()
    test_loss = 0
    with torch.no_grad():
        for istep in range(dataloader.nsteps):
            X, y, tau = next(dataloader)
            X, y, tau = X.to(device), y.to(device), tau.to(device)
            pred = model(X, tau)
            test_loss += loss_fn(pred, y, tau).item()

            if not np.isfinite(test_loss):
                print("test_loss:", test_loss)
                print("X:", X)
                print("y:", y)
                print("pred:", pred)
                raise RuntimeError("Nonfinite test loss")

    return test_loss / dataloader.nsteps