#!/usr/bin/env python

# Author: 
# Stephen G. Penny
# 7/30/24 - 8/9/24
# Adapted from:
# https://neuralgcm.readthedocs.io/en/latest/inference_demo.html

# ==============================================
# dabench interface:
from dabench import vector, model

# ==============================================
# Required for neuralGCM:
import jax
import numpy as np
import pickle
import xarray as xr

# For timing the run
import time

# For managing time stamps
from datetime import datetime, timedelta

# Dynamical core tools
from dinosaur import horizontal_interpolation
from dinosaur import spherical_harmonic
from dinosaur import xarray_utils

# Full model with NN and dycore
import neuralgcm

# Interface to Google Cloud Services
import gcsfs

# For reading input yaml file
import yaml

# For plotting
import matplotlib.pyplot as plt
# ==============================================

# For type checking
from typing import Any, Callable, Mapping, MutableMapping, Sequence, TypeVar
DatasetOrDataArray = TypeVar(
    'DatasetOrDataArray', xr.Dataset, xr.DataArray
)

class NeuralGCM(model.Model):
     
    def __init__(self,
                 system_dim=None,
                 time_dim=None,
                 delta_t=None,
                 model_obj=None,
                 params=None,
                 infile=None):
        super().__init__(system_dim=None,
                         time_dim=None,
                         delta_t=None,
                         model_obj=None)

        # Infile override, if provided
        if self.infile is not None:
            params = self.load_config(infile)

        # -----------------------
        # Set up input defaults
        # -----------------------
  
        # Initialize gcs token
        self.gcs = gcsfs.GCSFileSystem(token='anon')

        # Load the model
        self.forcing_type = params.get("forcing_type", "deterministic")
        self.atm_res = params.get("atm_res", "1_4")
        self.model_name = f'neural_gcm_dynamic_forcing_{self.forcing_type}_{self.atm_res}_deg.pkl'
        self.gcs_key = params.get("gcs_key", f'gs://gresearch/neuralgcm/04_30_2024/{self.model_name}')

        # Load ics
        self.era5_path  = params.get("era5_path", 'gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3')
        self.ics_path   = params.get("ics_path", self.era5_path)
        self.bcs_path   = params.get("bcs_path", self.era5_path)
        self.forecast_hours = params.get("forecast_hours",4*24)
        self.forecast_delta = timedelta(hours=self.forecast_hours)
        self.data_stride = params.get("data_stride",24)

        self.start_time = params.get("start_time",'2020-02-14')
        self.datetime_starttime = datetime.strptime(self.start_time, "%Y-%m-%d") # %H:%M:%S') 
        end_time = self.datetime_starttime + self.forecast_delta
        self.end_time   = end_time.strftime("%Y-%m-%d") #, %H:%M:%S")

        # Regrid the ics and bcs
        self.interpolation_method = params.get("interpolation_method","conservative")

        # Timing options
        self.inner_steps = params.get("inner_steps",24) # save model outputs once every 24 hours
        self.outer_steps = params.get("outer_steps",4)  # 4*24 // inner_steps,  # total of 4 days 
        self.timedelta = np.timedelta64(1, 'h') * self.inner_steps
        self.times = (np.arange(self.outer_steps) * self.inner_steps)

        # Pre-proceessing support
        input_variables = {'u': 'u_component_of_wind',
                           'v': 'v_component_of_wind',
                           't': 'temperature',
                           'q': 'specific_humidity',
                           'zh': 'geopotential',
                           'clwc': 'specific_cloud_liquid_water_content',
                           'ciwc': 'specific_cloud_ice_water_content',
                           'sst': 'sea_surface_temperature',
                           'ciconc': 'sea_ice_cover',
                           } 

        self.input_variables = params.get("input_variables", input_variables)

        # May be useful to pull directly on these levels: 
        # https://www.ecmwf.int/en/forecasts/datasets/set-i#I-i-b
        input_levels = np.array([   1,    2,    3,    5,    7,   10,   20,   30,   50,   70,  100,  125, \
        150,  175,  200,  225,  250,  300,  350,  400,  450,  500,  550,  600, \
        650,  700,  750,  775,  800,  825,  850,  875,  900,  925,  950,  975, \
        1000])

        self.input_levels = params.get("input_levels", input_levels)

        # Run forecast
        self.use_sfc_forecast = params.get("use_sfc_forecast",False)
        self.random_seed = params.get("random_seed",42)

        # Plotting
        self.plot_hpa_level=params.get("plot_hpa_level",850)
        self.plot_x=params.get("plot_x",'longitude')
        self.plot_y=params.get("plot_y",'latitude')
        self.plot_row=params.get("plot_row",'time')
        self.plot_col=params.get("plot_col",'model')
        self.plot_robust=params.get("plot_robust",True)
        self.plot_aspect=params.get("plot_aspect",2)
        self.plot_size=params.get("plot_size",2)
        self.plot_show=params.get("plot_show",True)

        attrs = vars(self)
        print(', '.join("%s: %s" % item for item in attrs.items()))


    def load_config(self, infile):
        with open(infile, 'r') as f:
            params = yaml.load(f, Loader=yaml.SafeLoader)

        params['model_name'] = f"neural_gcm_dynamic_forcing_{params['forcing_type']}_{params['atm_res']}_deg.pkl"
        params['gcs_key'] = f"gs://gresearch/neuralgcm/04_30_2024/{params['model_name']}"
        params['outer_steps'] = int(params['forecast_hours']) // int(params['inner_steps'])  # Total of n days
        print(type(self).__name__, params)
        return params
        

    def load_model(self):

        forcing_type = self.forcing_type
        atm_res = self.atm_res
        gcs_key = self.gcs_key

        # -----------------------------
        # Load the model
        # -----------------------------
        with self.gcs.open(gcs_key, 'rb') as f:
            ckpt = pickle.load(f)

        model = neuralgcm.PressureLevelModel.from_checkpoint(ckpt)

        return model


    def set_grid_info(self, full_data):
        self.latitude_nodes=full_data.sizes['latitude']
        self.longitude_nodes=full_data.sizes['longitude']
        self.latitude_spacing=xarray_utils.infer_latitude_spacing(full_data.latitude)
        self.longitude_offset=xarray_utils.infer_longitude_offset(full_data.longitude)


    def load_ics(self, model, override=''):

        start_time = self.start_time
        end_time = self.end_time
        data_stride = self.data_stride

        era5_path = self.era5_path
        full_data = xr.open_zarr(self.gcs.get_mapper(era5_path), chunks=None)
        self.set_grid_info(full_data)

        print(f'Variables = {[model.input_variables]}')

        # TODO: make this only a single time step and add another method
        #       to load a 'reference' dataset for evaluation.

        # NOTE: The data slice below collects data for the evaluation
        # only a single datetime is needed to initialize the forecast model.
        # ALSO: Regridding requires the data to be first loaded into memory.
        # Because this full dataset is gigantic (100s of TB) we’ll only
        # regrid a few time slices:
        sliced_data = (
            full_data
            [model.input_variables]
            .sel(time=slice(start_time, end_time, data_stride))   # Select data range from full reanalysis dataset
            .compute()       # Pull it into memory
        )

        return sliced_data


    def load_bcs(self, model, override=''):

        start_time = self.start_time
        end_time = self.end_time
        data_stride = self.data_stride

        era5_path = self.era5_path
        full_data = xr.open_zarr(self.gcs.get_mapper(era5_path), chunks=None)

        print(f'Variables = {[model.forcing_variables]}')

        # Regridding requires the data to be first loaded into memory.
        # Because this full dataset is gigantic (100s of TB) we’ll only
        # regrid a single time point:
        sliced_data = (
            full_data
            [model.forcing_variables]
#               [model.input_variables + model.forcing_variables]
## See:
## https://neuralgcm.readthedocs.io/en/latest/datasets.html#time-shifting
            .pipe(
                xarray_utils.selective_temporal_shift,
                variables=model.forcing_variables,
                time_shift='24 hours',
            )
            .sel(time=slice(start_time, end_time, data_stride))   # Select data range from full reanalysis dataset
            .compute()       # Pull it into memory
        )

        return sliced_data


    def get_regridder(self,model,skipna=True):

        method = self.interpolation_method
        
        latitude_nodes=self.latitude_nodes
        longitude_nodes=self.longitude_nodes
        latitude_spacing=self.latitude_spacing
        longitude_offset=self.longitude_offset

        print('get_regridder::')
        print(f'latitude_nodes = {latitude_nodes}')
        print(f'longitude_nodes = {longitude_nodes}')
        print(f'latitude_spacing = {latitude_spacing}')
        print(f'longitude_offset = {longitude_offset}')

        # Get grid for source dataset
        source_grid = spherical_harmonic.Grid(
                                            latitude_nodes=latitude_nodes,
                                            longitude_nodes=longitude_nodes,
                                            latitude_spacing=latitude_spacing,
                                            longitude_offset=longitude_offset,
                                            )

        # Get grid for neural GCM
        target_grid = model.data_coords.horizontal

        # ------------------------------------
        # build a Regridder object:
        # ------------------------------------
        # Note: Other available regridders include BilinearRegridder and NearestRegridder.
        # Note: skipna=True in ConservativeRegridder means grid cells with a mix of NaN/non-NaN 
        #       values should be filled skipping NaN values. This ensures sea surface 
        #       temperature and sea ice cover remains defined in coarse grid cells that 
        #       overlap coastlines.
        if method=='conservative':
            regridder = horizontal_interpolation.ConservativeRegridder(
                                                                       source_grid, 
                                                                       target_grid, 
                                                                       skipna=skipna
                                                                       )
        elif method=='bilinear':
            regridder = horizontal_interpolation.BilinearRegridder(
                                                                       source_grid, 
                                                                       target_grid, 
                                                                       skipna=skipna
                                                                       )
        elif method=='nearest':
            regridder = horizontal_interpolation.NearestRegridder(
                                                                       source_grid, 
                                                                       target_grid, 
                                                                       skipna=skipna
                                                                       )
        else:
            raise Exception(f'No valid interpolation method provided. method = {method}')

        return regridder


    def regrid_input(self, model, data, fill_nans=False):
        # Regridding data
        # See: https://neuralgcm.readthedocs.io/en/latest/datasets.html
        # Preparing a dataset stored on a different horizontal grid for NeuralGCM requires two steps:

        # 1) Horizontal regridding to a Gaussian grid. For processing fine-resolution data conservative 
        #    regridding is most appropriate (and is what we used to train NeuralGCM).
        #
        # 2) Filling in all missing values (NaN), to ensure all inputs are valid. Forcing fields like 
        #    sea_surface_temperature are only defined over ocean in ERA5, and NeuralGCM’s surface model 
        #    also includes a mask that ignores values over land, but we still need to fill all NaN values 
        #    to them leaking into our model outputs.
        #
        # Utilities for both of these operations are packaged as part of Dinosaur.

        # build a Regridder object:
        print ('regrid_input:: self.get_regridder...')
        regridder = self.get_regridder(model)

        # Perform regridding operation
        print ('regrid_input:: xarray_utils.regrid...')
        eval_data = xarray_utils.regrid(data, regridder)

        # Fill in fields like SST that may be NaN over land
        if (fill_nans):
            eval_data = xarray_utils.fill_nan_with_nearest(eval_data)

        return eval_data
        

    def run_forecast(self, model, ics_data, bcs_data):
        # NOTE: the ICs and BCs are extracted from the 'eval_data'.
        #       It is preferable to input these separately, so that
        #       the eval dataset can change, and since the IC and BC
        #       may have different time dimensions.

        use_sfc_forecast = self.use_sfc_forecast
        
        # Get the initial conditions
        inputs = model.inputs_from_xarray(ics_data.isel(time=0))

        # Get initial surface boundary conditions
        input_forcings_t0 = model.forcings_from_xarray(bcs_data.isel(time=0))
        rng_key = jax.random.key(self.random_seed)  # optional for deterministic models

        # Set up combined ICs and BCs
        initial_state = model.encode(inputs, input_forcings_t0, rng_key)

        # Get forecast surface boundary conditions. Either:
        # (a) use persistence for forcing variables (SST and sea ice cover), or
        # (b) use a forecast of SBCs (sst and sea ice)
        if not use_sfc_forecast:
            # Use a persistence forecast instead
            sfc_forcing_forecast = model.forcings_from_xarray(bcs_data.head(time=1))
            #NOTE: ".head(time=1)" gets the first time step and keeps the time dimension, 
            #       unlike ".isel(time=0)" which collapses the time dimension
        else:
            sfc_forcing_forecast = model.forcings_from_xarray(bcs_data)

        # make forecast
        # see: https://neuralgcm.readthedocs.io/en/latest/trained_models.html#advancing-in-time
        print('run_forecast:: steps = {self.outer_steps}')
        print('run_forecast:: timedelta = {self.timedelta}')
        final_state, predictions = model.unroll(
            initial_state,
            sfc_forcing_forecast,
            steps=self.outer_steps,
            timedelta=self.timedelta,
            start_with_input=True,
        )
        predictions_ds = model.data_to_xarray(predictions, times=self.times)

        return predictions_ds


    def plot_results(self, model, eval_data, predictions_ds):

        inner_steps = self.inner_steps
        data_stride = self.data_stride
        outer_steps = self.outer_steps

        plot_x = self.plot_x
        plot_y = self.plot_y
        plot_row = self.plot_row
        plot_col = self.plot_col
        plot_robust = self.plot_robust
        plot_aspect = self.plot_aspect
        plot_size = self.plot_size
        plot_hpa_level = self.plot_hpa_level
        plot_show = self.plot_show

        #STEVE: get times from predictions_ds
        times = predictions_ds['time'].values

        # Selecting ERA5 targets from exactly the same time slice
        target_trajectory = model.inputs_from_xarray(
            eval_data
            .thin(time=(inner_steps // data_stride))
            .isel(time=slice(outer_steps))
        )
        target_data_ds = model.data_to_xarray(target_trajectory, times=times)

        combined_ds = xr.concat([target_data_ds, predictions_ds], 'model')
        combined_ds.coords['model'] = ['ERA5', 'NeuralGCM']

        # Visualize ERA5 vs NeuralGCM trajectories
        combined_ds.specific_humidity.sel(level=plot_hpa_level).plot(
            x=plot_x, y=plot_y, row=plot_row, col=plot_col, robust=plot_robust, aspect=plot_aspect, size=plot_size
        );
        filename = 'plot_specific_humidity'
        plt.savefig(filename)
        if (plot_show):
            plt.show()

        combined_ds.u_component_of_wind.sel(level=plot_hpa_level).plot(
            x=plot_x, y=plot_y, row=plot_row, col=plot_col, robust=plot_robust, aspect=plot_aspect, size=plot_size
        );
        filename = 'plot_u_component_of_wind'
        plt.savefig(filename)
        if (plot_show):
            plt.show()

        combined_ds.temperature.sel(level=plot_hpa_level).plot(
            x=plot_x, y=plot_y, row=plot_row, col=plot_col, robust=plot_robust, aspect=plot_aspect, size=plot_size
        );
        filename = 'plot_temperature'
        plt.savefig(filename)
        if (plot_show):
            plt.show()

        combined_ds.geopotential.sel(level=plot_hpa_level).plot(
            x=plot_x, y=plot_y, row=plot_row, col=plot_col, robust=plot_robust, aspect=plot_aspect, size=plot_size
        );
        filename = 'plot_geopotential'
        plt.savefig(filename)
        if (plot_show):
            plt.show()


    def report_timing(self, timing_label=''):
    
        if not hasattr(self, "timing_start_time"):
            self.timing_start_process_time = time.process_time()
            self.timing_start_time = time.time()
            return

        # get execution time
        timing_end_process_time = time.process_time()
        seconds = timing_end_process_time - self.timing_start_process_time
        minutes = seconds / 60.0 
        print(f'< === {timing_label} ===')
        print(f'CPU Execution time so far: {minutes} minutes.') 

        # get wall clock time
        timing_end_time = time.time()
        seconds = timing_end_time - self.timing_start_time
        minutes = seconds / 60.0 
        print(f'Wall Clock Execution time so far: {minutes} minutes.') 
        print(f'  === {timing_label} === >')


    def full_sequence(self):

        # initialize the start time
        self.report_timing()

        # Load the model and model weights from a file
        print('Loading model...')
        model = self.load_model()
        self.report_timing("load model")

        # --------------------------
        # Load ICs and BCs
        # --------------------------

        # Get the ECMWF initial conditions and boundary conditions
        print('Getting ECMWF ICs (this may take about 5-7 minutes)...')
        ics_sliced = self.load_ics(model)
        self.report_timing("get ics")

        # Get the ECMWF initial conditions and boundary conditions
        print('Getting ECMWF BCs (should be < 1 min)...')
        bcs_sliced = self.load_bcs(model)
        self.report_timing("get bcs")

        # --------------------------
        # Regrid ICs and BCs
        # --------------------------

        # Regrid the ECMWF ICs to the model input grid
        print('Regridding ECMWF ICs (should be < 1 min)...')
        ics_eval = self.regrid_input(model, data=ics_sliced, fill_nans=False)
        self.report_timing("regrid ics")
        era5_eval = ics_eval


        # Regrid the ECMWF BCs to the model input grid
        print('Regridding ECMWF BCs (should be < 1 min)...')
        bcs_eval = self.regrid_input(model, data=bcs_sliced, fill_nans=True)
        self.report_timing("regrid bcs")

        # --------------------------
        # Run the forecast and store intermediate steps
        # --------------------------

        print('Running forecast (this may take a while, e.g. about 15-minutes per simulation day)...')
        predictions_ds = self.run_forecast(model,ics_eval,bcs_eval) #eval_data)
        print('Finished forecast!')
        self.report_timing("run forecast")

        # Final timing
        self.report_timing("Final")

        # --------------------------
        # Plot the results
        # --------------------------

        print('Plotting results...')
        self.plot_results(model,era5_eval,predictions_ds)


if __name__ == "__main__":

    # Get input parameters:
    params = None
    infile = '_neuralgcm.yaml'

    # Create model class
    model = NeuralGCM(params=params, infile=infile)

    # Print key input params:
    print(f'demo_start_time = {model.start_time}')
    print(f'demo_end_time = {model.end_time}')
    print(f'data_inner_steps = {model.data_stride}')
    print(f'inner_steps = {model.inner_steps}')
    print(f'outer_steps = {model.outer_steps}')
    print(f'timedelta = {model.timedelta}')
    print(f'datetime_starttime = {model.datetime_starttime}')
    print(f'forecast_delta = {model.forecast_delta}')
    print(f'times = {model.times}')
#   input("Press Enter to continue...")

    # Run all
    model.full_sequence()
