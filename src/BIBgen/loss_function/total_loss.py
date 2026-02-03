'''
Combine loss from gaussian_nll and bc_loss with optional weighting
'''
import torch
from gaussian_nll import GaussianNLLLoss
from bc_loss import BoundaryConditionLoss

class TotalLoss(torch.nn.Module):
    """
    Docstring for TotalLoss
    """
    
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred, variance, actual, subdet="ecal", barrel=True, bc_weight = 1.):
        """
        Docstring for forward
        
        :param self: Description
        :param pred: Description
        :param variance: Description
        :param actual: Description
        :param subdet: Description
        :param barrel: Description
        :param bc_weight: Description
        """
        gnll_loss = GaussianNLLLoss()
        gaus = gnll_loss.forward(pred, variance, actual)
        bcond_loss = BoundaryConditionLoss()
        bc = bcond_loss.forward(pred, subdet, barrel)

        loss_tot = gaus + bc * bc_weight

        return loss_tot
