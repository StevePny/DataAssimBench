# DataAssimBench

![Tests](https://github.com/StevePny/DataAssimBench/actions/workflows/python-ci-conda.yml/badge.svg)

This work follows the effort initiated by Rasp et al. in the WeatherBench <https://github.com/pangeo-data/WeatherBench>. Here, we create training sets and a process required to develop data assimilation methods and transition them from conception to full scale Earth system models.  

The field of data assimilation (DA) studies the integration of theory with observations. Models alone cannot make predictions. Data assimilation originated out of the need for operational weather forecast models to ingest observations in realtime so that computer models of the atmosphere could be initialized from a "best guess" state of the current conditions.  

Today, applied DA has matured in operational weather forecasting to include the entire online cycled process of continually ingesting numerous disparate observational data sources and integrating them with numerical prediction models to make regular forecasts, while estimating errors and uncertainties in this process and accounting for them along the way. The process can also include correcting inaccuracies in the model formulations or applying post-processing to forecasts to improve agreement with observations.  

Most of the software here was adapted from code developed for the following publications. Please cite these works when using this software package: 

- Penny, S. G., Smith, T. A., Chen, T.-C., Platt, J. A., Lin, H.-Y., Goodliff, M., & Abarbanel, H. D. I. (2022). Integrating recurrent neural networks with data assimilation for scalable data-driven state estimation. Journal of Advances in Modeling Earth Systems, 14, e2021MS002843. https://doi.org/10.1029/2021MS002843  

- Smith, T. A., Penny, S. G., Platt, J. A., & Chen, T.-C. (2023). Temporal subsampling diminishes small spatial scales in recurrent neural network emulators of geophysical turbulence. Journal of Advances in Modeling Earth Systems, 15, e2023MS003792. https://doi.org/10.1029/2023MS003792  

- Platt, J.A., S.G. Penny, T.A. Smith, T.-C. Chen, H.D.I. Abarbanel, (2022). A systematic exploration of reservoir computing for forecasting complex spatiotemporal dynamics,
Neural Networks,Volume 153, 530-552, ISSN 0893-6080, https://doi.org/10.1016/j.neunet.2022.06.025.


## Installation

We recommend setting up a virtual environment using either [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) or [virtualenv](https://virtualenv.pypa.io/en/latest/user_guide.html).

#### Clone Repo:

```bash
git clone git@github.com:StevePny/DataAssimBench.git
cd ./DataAssimBench
```

#### Setup virtual environment
1. In your cloned project directory, create a virtual environment

    ```bash
    python3 -m venv ./.venv
    ```
2. Activate the virtual environment

    ```bash
    source ./.venv/bin/activate
    ```

#### Install dabench
```bash
pip install -e ".[full]"
```

This will create a full installation including the ability to access cloud data or interface with other packages such as qgs. Alternatively, for a minimal installation, run:

```bash
pip install -e .
```


## Quick Start

For more detailed examples, go to the [DataAssimBench-Examples](https://github.com/StevePny/DataAssimBench-Examples) repo.

#### Importing data generators

```python
import dabench as dab
help(dab.data) # View data classes, e.g. data.Lorenz96
help(dab.data.Lorenz96) # Get more info about Lorenz96 class
```

#### Generating data

All of the data objects are set up with reasonable defaults. Generating data is as easy as:

```python
l96_obj = dab.data.Lorenz96() # Create data generator object
ds_l96 = l96_obj.generate(n_steps=1000) # Generate Lorenz96 simulation data as Xarray Dataset
ds_l96.dab.flatten().values # View output values flattened along time dimension
```
This example is for a Lorenz96 model, but all of the data objects work in a similar way.  

#### Sampling observations

Now that we have a generated dataset, we can easily generate noisy observations from it like so:

```python
obs = dab.observer.Observer(
        ds_l96, # Our generated Dataset object
        random_time_density=0.4, # Randomly sampling at ~40% of times
        random_location_density=0.3, # Randomly sample ~30% of variables
        # random_location_count = 10, # Alternatively, can specify number of locations to sample
        error_sd=1.2 # Add Gaussian Noise with SD = 1.2
)
obs_vec = obs.observe() # Run observe() method to generate observations
obs_vec
```

The Observer class is very flexible, allowing users to provide specific times and locations or randomly generate them. You can also choose to use "stationary" or "nonstationary" observers, indicating whether to sample the same locations at each observation time step or to sample different ones (default is "stationary").

#### Customizing generation options

All data objects are customizable.

For data-generators (e.g. numerical models such as Lorenz63, Lorenz96, SQGTurb), this means you can change initial conditions, model parameters, timestep size, number of timesteps, etc.

For data-downloaders (e.g. ENSOIDX, GCP), this means changing which variables you download, the lat/lon bounding box, the time period, etc.

The recommended way of specifying options is to pass a keyword argument (kwargs) dictionary. The exact options vary between the different types of data objects, so be sure to check the specific documentation for your chosen generator/downloader more info.

- For example, for the Lorenz96 data-generator we can change the forcing term, system_dim, and integration timestep delta_t like this:

```python
l96_options = {'forcing_term': 7.5,
               'system_dim': 5,
               'delta_t': 0.05}
l96_obj = dab.data.Lorenz96(**l96_options) # Create data generator object
ds_l96 = l96_obj.generate(n_steps=1000) # Generate Lorenz96 simulation data
ds_l96 # View the output values
```

- For example, for the Google Cloud (GCP) ERA5 data-downloader, we can select our variables and time period like this:

```python
gcp_options = {'variables': ['2m_temperature', 'sea_surface_temperature'],
               'date_start': '2020-06-01'
               'date_end': '2020-06-07'}
gcp_obj = dab.data.GCP(**gcp_options) # Create data generator object
ds_gcp = gcp_obj.load() # Loads data. Can also use gcp_obj.generate()
ds_gcp # View the output values
```
