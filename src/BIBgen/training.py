import h5py
import torch

class DataLoader:
    def __init__(infile : h5py.File, key : str, shuffle : bool = True):
        self.infile = infile
        self.key = key
        self.shuffle = shuffle

        self.event_ids = infile[key].keys()
        self.nevents = len(event_ids)
        self.ntau = len(infile[key + "/" + event_ids[0]]) - 1

        self.idx = self.nevents * self.ntau

    @property
    def nsteps(self):
        return self.nevents * self.ntau

    def __iter__(self):
        if self.idx >= len(self):
            self.shuffle_order = torch.randperm(self.nevents * self.ntau) if self.shuffle else torch.arange(self.nevents * self.ntau)
            self.idx = 0

        event_no = self.shuffle_order[self.idx] // self.ntau
        tau = self.shuffle_order[self.idx] % self.ntau
        self.idx += 1

        event = self.infile[self.key + "/" + self.event_id[event_no]]
        return event[tau+1], event[tau]

def train(dataloader : DataLoader, model : torch.nn, loss_fn, optimizer, device):
    model.train()
    for istep in range(dataloader.nsteps):
        X, y = next(dataloader)
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return loss.item()

def evaluate(dataloader : DataLoader, model : torch.nn, loss_fn, device):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for istep in range(dataloader.nsteps):
            X, y = next(dataloader)
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()

    test_loss /= dataloader

    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")