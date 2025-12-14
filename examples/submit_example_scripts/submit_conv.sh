#!/bin/bash

source setup.sh
python slcio_to_hdf5.py -i BIB_sim_* -o jobsub_sample.hdf5 -s $1 -e $2
