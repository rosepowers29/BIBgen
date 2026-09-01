#!/bin/bash

# specify pythonpath in container (can find correct path using check_modules.sh in utils)
export PYTHONPATH=$PYTHONPATH:/opt/conda/lib/python3.10/site-packages

# install a user installation instance of h5py since it does not come with the container
pip install h5py

# REPLACE with your executable
export PYTHONPATH=$PWD/src:$PYTHONPATH
python generate_like.py $1 config/$2 config/$4 test_sizes_large.csv -t $3
# END
