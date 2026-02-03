'''
Loss function to penalize model output outside the detector boundaries.
'''

import numpy as np
import torch


class BoundaryConditionLoss(torch.nn.Module):
    """
    Docstring for BoundaryConditionLoss
    """


    
    def __init__(self):
        super().__init__()

    

    def forward(self, pred, subdet="ecal", barrel=True):
        """
        Calculates the loss based on how far the hit is outside of the detector bounds.
        
        Arguments:
        pred: the predicted mean of each hit (N, 4)
        subdet (string, default="ecal"): the subdetector we are modeling
        barrel (bool, default=True): toggle between barrel and endcap

        Returns:
        loss: distance in [cm] outside of the detector volume
        """
        # the dictionaries are of the form bounds = {"subdetector": [rmin, rmax, zmin, zmax]}

        barrel_bounds = {"vertex": [3.0, 10.4, 0., 65.],
                        "innertracker": [12.7, 55.4, 48.2, 69.2], #FIXME -- unique geometry
                        "outertracker": [81.9, 148.6, 0., 124.9],
                        "ecal": [185.7, 212.5, 0., 230.7],
                        "hcal": [212.5, 411.3, 0., 257.5]}
    
        endcap_bounds = {"vertex": [2.5, 11.2, 8.0, 28.2],
                        "innertracker": [40.5, 55.5, 52.4, 219.0],
                        "outertracker": [61.8, 143.0, 131.0, 219.0],
                        "ecal": [31.0, 212.5, 230.7, 257.5],
                        "hcal": [30.7, 411.3, 257.5, 456.2]}
        if barrel:
            bounds = barrel_bounds[subdet]
        else:
            bounds = endcap_bounds[subdet]
        rmin = bounds[0]
        rmax = bounds[1]
        zmin = bounds[2]
        zmax = bounds[3]

        x = pred[:,1]
        y = pred[:,2]
        z = pred[:,3]

        r = np.sqrt(x**2 + y**2)
        
        # calculate "loss" arraywise before taking mean

        r_less = r < rmin # boolean mask for bad r's (too small)
        r_greater = r > rmax # boolean mask for bad r's (too big)
        z_out = abs(z) > zmax # boolean mask for bad z's
        r_bad = r_less | r_greater | z_out # includsive boolean mask for bad hits
        r_bad = r_bad.double() # convert to 1.0 and 0.0 bc we will use this for the losses

        r_bad[z_out] = abs(z[z_out]) - zmax
        r_bad[r_less] = rmin - r[r_less]
        r_bad[r_greater] = r[r_greater]-rmax
        r_bad[r_less & z_out] = np.sqrt((rmin - r[r_less & z_out])**2 + (abs(z[r_less & z_out]) - zmax)**2)
        r_bad[r_greater & z_out] = np.sqrt((r[r_greater & z_out] - rmax)**2 + (abs(z[r_greater & z_out]) - zmax)**2)

        # return loss as mean distance outside of detector for batch
        loss = torch.mean(r_bad)
        return loss
        
    