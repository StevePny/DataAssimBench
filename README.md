# DataAssimBench

This work follows the effort initiated by Rasp et al. in the WeatherBench <https://github.com/pangeo-data/WeatherBench>. Here, we create training sets and a process required to develop data assimilation methods and transition them from conception to full scale Earth system models.  

The field of data assimilation (DA) studies the integration of theory with observations. Models alone cannot make predictions. Data assimilation originated out of the need for operational weather forecast models to ingest observations in realtime so that computer models of the atmosphere.  

Today, applied DA has matured in operational weather forecasting to include the entire online cycled process of continually ingesting numerous disparate observational data sources and integrating them with numerical prediction models to make regular forecasts. The process can also include correcting the models or applying post-processing to forecasts.  


## Installation

#### Clone Repo:

```bash
git clone git@github.com:StevePny/DataAssimBench.git
```

#### Set Up Conda Environment

```bash
cd DataAssimBench
conda env create -f environment.yml
conda activate ddc
```

#### Install dabench
```bash
pip install .
```

## Quick Start

```python
import dabench
help(dabench.data)
```


