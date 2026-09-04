#!/bin/bash

export PYTHONPATH=$PYTHONPATH:/opt/conda/lib/python3.10/site-packages
pip install matplotlib
export PYTHONPATH=$PWD/src:$PYTHONPATH
python inspect_predicted_variance.py denoiser_$1.pth config/$2 config/$3 -o predicted_variance_$1.png
