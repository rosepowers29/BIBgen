from typing import Callable

import h5py
import torch

class DataLoader:
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
        self.infile = infile
        self.key = key
        self.shuffle = shuffle

        self.event_ids = list(infile[key].keys())
        self.nevents = len(self.event_ids)
        self.ntau = len(infile[key + "/" + self.event_ids[0]]) - 1

        self.idx = self.nevents * self.ntau

    @property
    def nsteps(self):
        """
        Total number of trainig steps, number of events * number of diffusion steps
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
            self.shuffle_order = torch.randperm(self.nevents * self.ntau) if self.shuffle else torch.arange(self.nevents * self.ntau)
            self.idx = 0

        event_no = self.shuffle_order[self.idx] // self.ntau
        tau = self.shuffle_order[self.idx] % self.ntau
        self.idx += 1

        event = self.infile[self.key + "/" + self.event_ids[event_no]]
        return torch.from_numpy(event[tau+1]).to(dtype=torch.float32), torch.from_numpy(event[tau]).to(dtype=torch.float32), tau

def train(dataloader : DataLoader, model : torch.nn.Module, loss_fn : Callable, optimizer : torch.optim.Optimizer, device : torch.device):
    """
    Trains the model for one epoch.

    Parameters
    ----------
    dataloader : DataLoader
        dataloader for training data
    model : torch.nn.Module
        Denoising model
    loss_fn : Callable
        loss function to be called directly on model output, i.e. loss_fn(pred, y) instead of loss_fn(mu, std, y)
    optimizer : torch.optim.Optimizer
        Optimizer for gradient descent
    device : torch.device
        device on which to perform, usually torch.device("cuda") or torch.device("cpu")

    Returns
    -------
    loss : float
        Training loss on the final iteration
    """
    model.train()
    for istep in range(dataloader.nsteps):
        X, y, tau = next(dataloader)
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X, tau)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return loss.item()

def evaluate(dataloader : DataLoader, model : torch.nn.Module, loss_fn : Callable, device : torch.device):
    """
    Parameters
    ----------
    dataloader : DataLoader
        dataloader for training data
    model : torch.nn.Module
        Denoising model
    loss_fn : Callable
        loss function to be called directly on model output, i.e. loss_fn(pred, y) instead of loss_fn(mu, std, y)
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
            X, y = X.to(device), y.to(device)
            pred = model(X, tau)
            test_loss += loss_fn(pred, y).item()

    return test_loss / dataloader.nsteps