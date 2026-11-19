#!/bin/bash

# specify pythonpath in container (can find correct path using check_modules.sh in utils)
export PYTHONPATH=$PYTHONPATH:/opt/conda/lib/python3.10/site-packages

# install a user installation instance of h5py since it does not come with the container
pip install h5py

# REPLACE with your executable
# Arguments: model.pth config tag schedule cache nevents
export PYTHONPATH=$PWD/src:$PYTHONPATH
python from_cache.py $1 config/models/$2 config/noise_schedules/$4 $5 -o $3_gen.hdf5 -n $6
# END
