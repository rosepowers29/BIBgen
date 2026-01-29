#!/bin/bash

# specify pythonpath in container (can find correct path using check_modules.sh in utils)
export PYTHONPATH=$PYTHONPATH:/opt/conda/lib/python3.10/site-packages

# install a user installation instance of h5py since it does not come with the container
pip install h5py

# REPLACE with your executable
python train.py diffused_cyl_phipi4_medium.hdf5 noise_schedule.csv -c -e 101

# END
