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
                possible_workloads = self.ntau_factors * infile[self.key + "/" + event_id].shape[1]
                valid_factors = self.ntau_factors[possible_workloads <= hit_budget]
                self.batch_size_per_event[event_id] = 1 if valid_factors.shape[0] == 0 else valid_factors[-1]

        self.nsteps = 0
        for event_id in self.batch_size_per_event:
            self.nsteps += self.nbatches(event_id)

        self.idx = self.nsteps

        # The most recently read event. All batches of an event are served consecutively, so each
        # event is read from the file once per epoch rather than once per batch.
        self.cached_event_id = None
        self.cached_event = None

    def nbatches(self, event_id):
        return self.ntau // self.batch_size_per_event[event_id]

    def __next__(self):
        if self.idx >= self.nsteps:
            # Shuffle the order of events each epoch, and the diffusion timesteps within each event.
            # Batches of one event stay together so that the event only has to be read once.
            self.shuffle_order = torch.randperm(self.nevents) if self.shuffle else torch.arange(self.nevents)
            self.batch_indices = {
                event_id : torch.reshape(
                    torch.randperm(self.ntau) if self.shuffle else torch.arange(self.ntau),
                    (self.nbatches(event_id), self.batch_size_per_event[event_id])
                ) for event_id in self.batch_size_per_event
            }

            self.paths = []
            for event_no in self.shuffle_order:
                event_id = self.event_ids[event_no]
                for i in range(self.nbatches(event_id)):
                    self.paths.append((event_id, i))

            assert len(self.paths) == self.nsteps
            self.idx = 0

        event_id, batch_no = self.paths[self.idx]
        batch_taus = self.batch_indices[event_id][batch_no]
        self.idx += 1

        if event_id != self.cached_event_id:
            self.cached_event = None  # release the previous event before reading the next
            self.cached_event = self.infile[self.key + "/" + event_id][:]
            self.cached_event_id = event_id
        event = self.cached_event
        return torch.from_numpy(event[batch_taus + 1]).to(dtype=torch.float32), torch.from_numpy(event[batch_taus]).to(dtype=torch.float32), batch_taus