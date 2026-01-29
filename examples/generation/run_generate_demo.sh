#!/bin/bash

# specify pythonpath in container (can find correct path using check_modules.sh in utils)
export PYTHONPATH=$PYTHONPATH:/opt/conda/lib/python3.10/site-packages

# install a user installation instance of h5py since it does not come with the container
pip install h5py

# REPLACE with your executable
python generate_demo.py denoiser_v4.pth noise_schedule.csv -c -n 10 -s 5077 # mean number of barrel hits in pi/2 slice

# END
