# Temporally-Coherent Surface Reconstruction via Metric-Consistent Atlases

The implementation of the ICCV'21 paper **Temporally-Coherent Surface 
Reconstruction via Metric-Consistent Atlases** and its extension 
**Temporally-Consistent Surface Reconstruction using Metrically-Consistent 
Atlases** (currently under TPAMI review).

<div float="left">
    <img src="doc/img/teaser/cat_walk.gif" width="500" style="margin: 0; padding: 0" />
    <img src="doc/img/teaser/handstand.gif" width="500" style="margin: 0; padding: 0" />
</div>
<div float="left">
    <img src="doc/img/teaser/horse_gallop.gif" width="500" style="margin: 0; padding: 0" />
    <img src="doc/img/teaser/camel_collapse.gif" width="500" style="margin: 0; padding: 0" />
</div>

## Install
The framework was tested with Python 3.8, PyTorch 1.7.0. and CUDA 11.0. The 
easiest way to work with the code is to create a new virtual Python environment 
and install the required packages.

1. Install the [virtualenvwrapper](https://virtualenvwrapper.readthedocs.io/en/latest/).
2. Create a new environment and install the required packages.

```shell
mkvirtualenv --python=python3.8 tcsr
pip install -r requirements.txt
```

3. Get the code and prepare the environment as follows:

```shell
git clone git@github.com:bednarikjan/temporally_coherent_surface_reconstruction.git
git submodule update --init --recursive
export PYTHONPATH="{PYTHONPATH}:path/to/dir/temporally_coherent_surface_reconstruction"
```

## Get the Data

## Train

## Evaluate

## Visualize

## Citing this Work

## Acknowledgements

## TODO
