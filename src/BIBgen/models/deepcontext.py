from collections import OrderedDict

import torch
from torch import nn

class ContextTower(nn.Module):
    def __init__(self,
        input_size : int, 
        output_size : int, 
        context_size : int, 
        hidden_layer_size : int, 
        n_accum_hidden : int, 
        n_update_hidden : int
    ):
        """
        Parameters
        ----------

        Returns
        -------
        self : ContextTower
            torch module useable in neural networks
        """
        super().__init__()

        self.accumulater = common.make_mlp(input_size, context_size, hidden_layer_size, n_accum_hidden)
        self.updater = common.make_mlp(input_size + context_size, output_size, hidden_layer_size, n_update_hidden)

    def forward(self, input_set : torch.Tensor):
        r"""
        Parameters
        ----------
        input_set : torch.Tensor
            Input set with shape (n_batch, n_members, input_size) or (n_members, input_size)

        Returns
        -------
        output_set : torch.Tensor
            Output set with shape (n_batch, n_members, output_size) or (n_members, output_size)
        """
        nhits = input_set.shape[-2]
        nhits_feature = input_set.new_full((*input_set.shape[:-1], 1), torch.log(nhits)) # (n_batch, n_members, 1)

        context_contributions = self.accumulator(input_set) # (n_batch, n_members, context_size)
        context = context_contributions.mean(dim=-2) # (n_batch, context_size)
        input_wcontext = torch.cat([
            input_set,
            context.unsqueeze(-2).expand(-1, nhits, -1), # (n_batch, n_members, context_size)
            nhits_feature
        ], dim=-1) # (n_batch, n_members, input_size + context_size + 1)

        delta = self.updater(input_wcontext)
        return input_set + delta