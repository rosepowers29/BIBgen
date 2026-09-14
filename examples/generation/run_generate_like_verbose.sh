#!/bin/bash

export PYTHONPATH=$PYTHONPATH:/opt/conda/lib/python3.10/site-packages
pip install h5py
export PYTHONPATH=$PWD/src:$PYTHONPATH
python generate_like.py denoiser_cosine_predvar.pth config/equivariant_denoiser_learned_variance.json config/noise_schedule_cosine.csv test_sizes_debug.csv -t cosine_predvar_chaindebug -v 2
