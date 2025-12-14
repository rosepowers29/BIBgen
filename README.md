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
