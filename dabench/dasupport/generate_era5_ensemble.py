# Sample a series of initial conditions from era5 in order to generate a test initial ensemble

import argparse

# For converting strings into datetime objects
from datetime import datetime, timedelta

# Interface to Google Cloud Services
import gcsfs
import xarray as xr
from dateutil.relativedelta import relativedelta

from helpers.constants import ERA5_CONTROL_VARIABLES
from helpers.timing import report_timing


#%% Parse arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description="Process command line inputs.")

    # Define the arguments
    parser.add_argument(
        "--atmosphere_ensemble_s3_key",
        type=str,
        required=True,
        default=None,
        help="The s3 path for the ensemble zarr store.",
    )
    parser.add_argument(
        "--date_format",
        type=str,
        required=False,
        default="%Y%m%dZ%H",
        help="Date format. Default: %Y%m%dZ%H",
    )
    parser.add_argument(
        "--target_date",
        type=str,
        required=True,
        default=None, #datetime.strptime(f"{YEAR}{MONTH}{DAY}Z{HOUR}",'%Y%m%dZ%H'),
        help="Initialization date. Default format: %Y%m%dZ%H",
    )
    parser.add_argument(
        "--ensemble_size",
        type=int,
        required=True,
        default=None,
        help="Number of ensemble members",
    )
    parser.add_argument(
        "--sample_strategy",
        type=str,
        required=False,
        default="consecutive_day",
        help="{'multi_year'|'multi_month'|'consecutive_day'}",
    )
    parser.add_argument(
        "--start_date",
        type=str,
        required=True,
        default=None, #datetime.strptime(f"{YEAR-1}{MONTH}{DAY}Z{HOUR}",'%Y%m%dZ%H'),
        help="Date to start backwards count for sample strategy",
    )
    parser.add_argument(
        "--era5_path",
        type=str,
        required=False,
        default="gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3",
        help="Cloud-based source of the ERA5 dataset to access as ensemble members.",
    )
    # Parse the arguments
    args = parser.parse_args()
    return args


#%% Define the initial ensemble


def define_init_ensemble(
    ensemble_size, init_ensemble_start_date, init_ensemble_sample_strategy="multi_year"
):

    if init_ensemble_sample_strategy == "multi_year":
        increment = relativedelta(years=1)
    elif init_ensemble_sample_strategy == "multi_month":
        increment = timedelta(months=1)
    elif init_ensemble_sample_strategy == "consecutive_day":
        increment = timedelta(days=1)
    else:
        raise Exception(
            f"Not a valid init_ensemble_sampling_strategy = {init_ensemble_sample_strategy}"
        )

    init_ensemble_member_dates = []
    for i in range(ensemble_size):
        init_ensemble_member_dates.append(init_ensemble_start_date - i * increment)

    print(f"ensemble member init date list = {init_ensemble_member_dates}")

    return init_ensemble_member_dates


def main(
    date_format:str="%Y%m%dZ%H",
    atmosphere_ensemble_s3_key:str=None,
    target_date:datetime=datetime.strptime("19990101Z00","%Y%m%dZ%H"),
    sample_strategy:str="consecutive_day",
    start_date:datetime=datetime.strptime("19981231Z00","%Y%m%dZ%H"),
    ensemble_size:int=4,
    era5_path:str="gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3",
    ):

    #%% Set up the gcp access to era5
    if era5_path[0:2] == "gs":
        gcs = gcsfs.GCSFileSystem(token="anon")
        ds_era5 = xr.open_zarr(gcs.get_mapper(era5_path), chunks=None)
    else:
        raise Exception("Non-GCP source for ERA5 not yet supported. EXITING...")
    report_timing(timing_label="build_test_ensemble_era5:: access remote zarr store")

    #%% Reorder the latitudes
    # Following:
    # https://stackoverflow.com/questions/54677161/xarray-reverse-an-array-along-one-coordinate
    # (ECMWF latitudes are often stored N to S instead of - to +)
    ds_era5 = ds_era5.isel(latitude=slice(None, None, -1))
    print(ds_era5.latitude)
    assert ds_era5.latitude[0] < ds_era5.latitude[-1]

    #%% Determine dates for initial ensemble sampling
    init_ensemble_member_dates = define_init_ensemble(
        ensemble_size=ensemble_size,
        init_ensemble_start_date=start_date,
        init_ensemble_sample_strategy=sample_strategy,
    )

    #%% Now sample the selection from era5 and put into new zarr store on s3

    #%% Sample from era5
    ds_init_ens = ds_era5[ERA5_CONTROL_VARIABLES].sel(time=init_ensemble_member_dates)
    report_timing(
        timing_label="build_test_ensemble_era5:: select time steps as ensemble members"
    )
    print(ds_init_ens)

    #%% Update time to target and add ensemble dimension
    ds_init_ens = ds_init_ens.rename_dims(dims_dict={"time": "member"})
    ds_init_ens["member"] = range(ensemble_size)
    ds_init_ens = ds_init_ens.drop_vars("time")
    report_timing(
        timing_label="build_test_ensemble_era5:: add member dimension to replace time"
    )
    print(ds_init_ens)

    #%% Select target date from era5 for recentering the ensemble
    ds_target = ds_era5[ERA5_CONTROL_VARIABLES].sel(time=target_date)

    #%% Compute the 10m diagnostic wind speed and 10m neutral wind speed
    if ('10m_u_component_of_neutral_wind' in  ERA5_CONTROL_VARIABLES and
        '10m_v_component_of_neutral_wind' in ERA5_CONTROL_VARIABLES):
        ds_init_ens['ws10n'] = (ds_init_ens['10m_u_component_of_neutral_wind']**2 + ds_init_ens['10m_v_component_of_neutral_wind']**2)**(0.5)
        ds_target['ws10n'] = (ds_target['10m_u_component_of_neutral_wind']**2 + ds_target['10m_v_component_of_neutral_wind']**2)**(0.5)
        report_timing(
            timing_label="build_test_ensemble_era5:: computing neutral wind speeds at 10m (ws10n)"
        )
    if ('10m_u_component_of_wind' in  ERA5_CONTROL_VARIABLES and
        '10m_v_component_of_wind' in ERA5_CONTROL_VARIABLES):
        ds_init_ens['ws10m'] = (ds_init_ens['10m_u_component_of_wind']**2 + ds_init_ens['10m_v_component_of_wind']**2)**(0.5)
        ds_target['ws10m'] = (ds_target['10m_u_component_of_wind']**2 + ds_target['10m_v_component_of_wind']**2)**(0.5)
        report_timing(
            timing_label="build_test_ensemble_era5:: computing diagnostic wind speeds at 10m (ws10m)"
        )

    #%% Recenter ensemble to target date
    print(f'build_test_ensemble_era5:: re-centering ensemble with ensemble_size = {ensemble_size} to target_date = {target_date}...')
    ds_mean = ds_init_ens.mean(dim="member")
    ds_diff = ds_target - ds_mean
    ds_init_ens = ds_init_ens + ds_diff
    report_timing(
        timing_label="build_test_ensemble_era5:: recenter ensemble to target date"
    )
    print(ds_init_ens)

    #%% Now add time back on as a singleton dimension
    ds_init_ens = ds_init_ens.expand_dims(dim={"time": [target_date]}, axis=0)
    report_timing(
        timing_label="build_test_ensemble_era5:: add time dimension back on to dataset structure"
    )
    print(ds_init_ens)

    #%% Add some checks to make sure dimensions haven't changed
    assert ds_era5.sizes['latitude'] == ds_init_ens.sizes['latitude']
    assert ds_era5.sizes['longitude'] == ds_init_ens.sizes['longitude']
    assert ds_era5.sizes['level'] == ds_init_ens.sizes['level']

    #%% Upload to s3 as zarr
    print('Uploading to s3 zarr...')
    ds_init_ens.to_zarr(atmosphere_ensemble_s3_key, mode="w")
    report_timing(
        timing_label="build_test_ensemble_era5:: upload to s3 as a new zarr store"
    )


#%% Main access
if __name__ == "__main__":
    args = parse_arguments()

    # %% Process input arguments
    report_timing(timing_label="build_test_ensemble_era5:: initializing...")

    main(
        date_format=args.date_format,
        atmosphere_ensemble_s3_key=args.atmosphere_ensemble_s3_key,
        target_date=args.target_date,
        sample_strategy=args.sample_strategy,
        start_date=args.start_date,
        ensemble_size=args.ensemble_size,
        era5_path=args.e
