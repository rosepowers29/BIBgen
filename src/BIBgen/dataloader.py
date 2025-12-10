import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np


#-----------------Define dataset class---------------------#

class DiffusedPairDataset(Dataset):
    def __init__(self, diffused_data_path, group, evt, transform=None):
        """
        Initializes the diffused dataset, read in from HDF5 file.

        Argsuments:
            diffused_data_path (str): Path to the HDF5 file.
            group (str): Name of the group within the HDF5 file (training or validation)
            evt (str): Label of the event (name of the dataset) within the group to load.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        super().__init__()
        self.diffused_data_path = diffused_data_path
        self.group = group
        self.evt = evt
        self.transform = transform
        self.data_len = self._get_dataset_length()

    def _get_dataset_length(self):
        """
        Helper to extract number of timestep pairs from the event dataset.
        
        Returns: N_t (int): The number of timestep pairs in an the event dataset (should be 100 for our use case)
        
        """
        with h5py.File(self.diffused_data_path, 'r') as file:
            group = file[self.group]
            evt = group[self.evt]
            N_t = evt.shape[0]-1
            return N_t

    def __len__(self):
        """
        Get the length of the pairs available in the dataset. In this case, this should get us 100 since we have 101 timesteps.
        
        Defined in _get_dataset_length.
        """
        return self.data_len

    def __getitem__(self, idx):
        """
        Retrieves a single timestep and the timestep before it from the event dataset. Datasets have shape (101, N, 4), with 101 timesteps.
        __getitem__ retrieves the 4 components of each of N hits in self.evt at time step idx and idx-1.

        Arguments:
            idx(int): The timestep we want, integer between 1 and 100

        Returns:
            current_tensor, previous_tensor: (pytorch tensors in form of pair of 2d arrays of shape (N, 4)): The 4 components for each of the N hits at time step idx, idx-1.     
        """
        with h5py.File(self.diffused_data_path, 'r') as file:
            group = file[self.group]
            evt = group[self.evt]
            at_idx = evt[idx, :, :]
            before_idx = evt[idx-1, :, :]

        # Convert to PyTorch tensors
        current_tensor = torch.from_numpy(at_idx)
        previous_tensor = torch.from_numpy(before_idx)


        return current_tensor, previous_tensor
    

    #------------DataLoader-----------------#
    # implement dataloading as function within the class
    def load_data(self):
        """
        Creates the dataloader which spits out consecutive pairs.

        Returns:
            loaded_pairs (DataLoader instance): A DataLoader object with the 100 consecutive pairs in the event.
        
        """
        # set batch_size=1 to return one pair per iteration
        loaded_pairs = DataLoader(self, batch_size=1, shuffle=False)
        return loaded_pairs
