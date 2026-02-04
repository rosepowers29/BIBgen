'''
Combine loss from gaussian_nll and bc_loss with optional weighting
'''
import torch
from gaussian_nll import GaussianNLLLoss
from bc_loss import BoundaryConditionLoss

class TotalLoss(torch.nn.Module):
    """
    Adds loss from gaussian NLL model and boundary condition penalty.
    Weights can be introduced as needed.
    """
    
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred, variance, actual, subdet="ecal", barrel=True, bc_weight = 1.):
        """
        Calls gaussian and BC penalty loss methods and adds the two values together, with optional weighting.
        
        Arguments:
        pred (torch tensor): the predicted mean of each hit (N, 4)
        variance (torch tensor): predicted variances (N, 4) 
        actual (torch tensor): ground truth (N, 4)
        subdet (string, default="ecal"): the subdetector we are modeling
        barrel (boolean, default = True): toggle between barrel and endcap
        bc_weight (float, default = 1.0): optional weight parameter for the BC penalty loss

        Returns:
        loss_tot (double): total loss
        """
        gnll_loss = GaussianNLLLoss()
        gaus = gnll_loss(pred, variance, actual)
        bcond_loss = BoundaryConditionLoss()
        bc = bcond_loss(pred, subdet, barrel)

        loss_tot = gaus + bc * bc_weight

        return loss_tot
