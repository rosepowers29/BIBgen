#!/usr/bin/env python

# some imports we need for other scripts, should load with no errors
import sys
import h5py
import numpy as np
import torch

# ensure input file has transferred properly and is readable
f = h5py.File("sim_mp_test.hdf5", 'r')

# read some info from input file and write to output file
with open("output.txt", 'w') as file:
    for item in list(f.keys()):
        file.write(str(item)+'\n')
    file.close()
