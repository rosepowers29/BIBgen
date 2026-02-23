# BIBgen

BIBgen is a machine-learning project that aims to provide an alternative to resource-intensive simulations of beam-induced background (BIB) in a future muon collider, specifically the MAIA detector. For background information on the MAIA detector, see [the most recent MAIA preprint](https://arxiv.org/abs/2502.00181); for background information on the equivariant layer used in the model, see [this paper on Deep Sets](https://arxiv.org/pdf/1703.06114), and for more information on our diffusion model, see [this publication](https://proceedings.neurips.cc/paper_files/paper/2020/file/4c5bcfec8584af0d967f1ab10179ca4b-Paper.pdf).

## Getting Started

To get started with the BIBgen framework, fork or clone this repository. In the root directory, install with
```bash
python -m pip install .
```
or
```bash
uv pip install .
```

To do more than look around, the package should be run on an access point of the OSPool. To get access to an OSG machine, fill out the application at [this link](https://www.osgconnect.net/signup). 
## Testing

To run tests, use:
```bash
pytest tests/
```

Note about testing: unless you are in an environment equipped with the ``pyLCIO`` software, exclude ``test_conversion.py`` from your tests. ``pyLCIO`` is not distributed with PyPI, so you cannot install the package as a dependency. It is only needed for the file conversion step (see ``scripts/convert_slcio_files.py``).

## Preprocessing

Setup a directory to store preprocessed data.
```bash
mkdir data
cd examples
```

Produce raw training data. Currently this script splits events into training, validation, and testing groups, grabs only the ECAL barrel hits, converts to cylindrical coordinates, takes a $\phi$ slice, and normalizes every variable. Variables are normalized so that the training data has $\mu = 0$ and $\sigma = 1$. The mean and standard deviation are stored in the output file so that the transformation can be reversed during generation.
```bash
uv run make_training_data.py /scratch/rosep8/BIBgen/src/BIBgen/sim_mm_0_1000.hdf5 /scratch/rosep8/BIBgen/src/BIBgen/sim_mp_0_1000.hdf5 -o ../data/raw_cyl_phipi4_large.hdf5 -s 700,200,100 -c -p 0.785398
```

Run the forward diffusion process on the training and validation datasets. The output has events in the same structure, but each event now contains the noisy timesteps according to `noise_schedule.csv`. `noise_schedule.csv` uses a 100 step quadratic schedule such that $\overline \alpha_t \approx 10^{-5}$.
```bash
uv run diffuse.py ../data/raw_cyl_phipi4_large.hdf5 noise_schedule.csv -o ../data/diffused_cyl_phipi4_large.hdf5
```

## Training

Run the training, which will produce a file `denoiser.pth` with trained model weights. Note that the noise schedule must be provided since these are a proxy for predicted variances. The script will automatically use a cuda device if one is detected.
```bash
uv run training/train.py ../data/diffused_cyl_phipi4_large.hdf5 noise_schedule.csv -e 151 -b 5
```

Condor scripts to submit the above script on OSPool is provided.
```bash
condor_submit training/submit_train.sub
```

## Generation

First extract the sizes of all the test events.
```bash
uv run generation/write_test_sizes.py ../data/raw_cyl_phipi4_large.hdf5 -o generation/test_sizes_large.csv
```

Generate a new dataset with the trained model. The model is completely agnostic to event size, so we purposely generate events of the same sizes as the test. This produces a file `like.hdf`
```bash
uv run generation/generate_like.py denoiser.pth noise_schedule.csv test_sizes_large.csv
```

The above script can be submitted to OSPool.
```bash
condor_submit generation/submit_generate_like.sub
```

## Analysis

Plot kinematic distributions of the test data and generated data.
```bash
uv run plot_comparison.py ../data/raw_cyl_phipi4_large.hdf5 generation/like.hdf5
```

Plot the wavelengths that the model's learned Fourier encoding converged to.
```bash
uv run plot_encoding.py training/denoiser.pth ../data/raw_cyl_phipi4_large.hdf5
```
