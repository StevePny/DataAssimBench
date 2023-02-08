# DataAssimBench

![Tests](https://github.com/StevePny/DataAssimBench/actions/workflows/python-ci-conda.yml/badge.svg)

This work follows the effort initiated by Rasp et al. in the WeatherBench <https://github.com/pangeo-data/WeatherBench>. Here, we create training sets and a process required to develop data assimilation methods and transition them from conception to full scale Earth system models.  

The field of data assimilation (DA) studies the integration of theory with observations. Models alone cannot make predictions. Data assimilation originated out of the need for operational weather forecast models to ingest observations in realtime so that computer models of the atmosphere could be initialized from a "best guess" state of the current conditions.  

Today, applied DA has matured in operational weather forecasting to include the entire online cycled process of continually ingesting numerous disparate observational data sources and integrating them with numerical prediction models to make regular forecasts, while estimating errors and uncertainties in this process and accounting for them along the way. The process can also include correcting inaccuracies in the model formulations or applying post-processing to forecasts to improve agreement with observations.  

The user may have to manually install:
conda install -c conda-forge jax
conda install -c conda-forge pyqg

## Installation

#### Clone Repo:

```bash
git clone git@github.com:StevePny/DataAssimBench.git
```

#### Set Up Conda Environment

```bash
cd DataAssimBench
conda env create -f environment.yml
conda activate dab
```

#### Install dabench
```bash
pip install .
```

## Quick Start

For more detailed examples, go to the [DataAssimBench-Examples](https://github.com/StevePny/DataAssimBench-Examples) repo.

#### Importing data generators

```python
from dabench import data
help(data) # View data classes, e.g. data.Lorenz96
help(data.Lorenz96) # Get more info about Lorenz96 class
```

#### Generating data

All the data generators are set up with reasonable defaults. Generating data is as easy as:

```python
l96_obj = data.Lorenz96() # Create data generator object
l96_obj.generate(n_steps=1000) # Generate Lorenz96 simulation data
l96_obj.values # View the output values
```

This example is for a Lorenz96 model, but all of the models and data downloaders work in a similar way. 

#### Customizing generation options

All the data generators are customizable.

For models (e.g. Lorenz63, Lorenz96, SQGTurb), this means you can change initial conditions, model parameters, timestep size, number of timesteps, etc.

For data downloaders (e.g. ENSOIDX, AWS, GCP), this means changing which variables you download, the lat/lon bounding box, the time period, etc.

The recommended way of specifying options is to pass a keyword argument (kwargs) dictionary. The exact options vary between the different types of data generators, so be sure to check the specific documentation for your chosen model/downloader more info.

- For example, for Lorenz96 we can changing the forcing term, system_dim, and timestep delta_t like this:

```python
l96_options = {'forcing_term': 7.5,
               'system_dim': 5,
               'delta_t': 0.05}
l96_obj = data.Lorenz96(**l96_options) # Create data generator object
l96_obj.generate(n_steps=1000) # Generate Lorenz96 simulation data
l96_obj.values # View the output values
```

- For example, for the Amazon Web Services (AWS) ERA5 data downloader, we can select our variables and time period like this:

```python
aws_options = {'variables': ['air_pressure_at_mean_sea_level', 'sea_surface_temperature'],
               'years': [1984, 1985]}
aws_obj = data.AWS(**aws_options) # Create data generator object
aws_obj.load() # Loads data. Can also use aws_obj.generate()
aws_obj.values # View the output values
```

