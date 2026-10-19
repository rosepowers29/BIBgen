#!/bin/bash

export PYTHONPATH=$PYTHONPATH:/opt/conda/lib/python3.10/site-packages
pip install h5py

export PYTHONPATH=$PWD/src:$PYTHONPATH
python train.py $1 config/noise_schedules/$4 config/models/$2 -e 151 -b ${5:-5} -t $3
