#!/bin/bash

# specify pythonpath in container (can find correct path using check_modules.sh in utils)
export PYTHONPATH=$PYTHONPATH:/opt/conda/lib/python3.10/site-packages

# install a user installation instance of h5py since it does not come with the container
pip install h5py

ls
export PYTHONPATH=$PWD/src:$PYTHONPATH
python train.py diffused_cyl_phipi4_large.hdf5 config/noise_schedule.csv config/equivariant_denoiser.json -e 151 -b 5

# END
