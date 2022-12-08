"""Load data from AWS Registry of Open Data

For now just ERA5 ECMWF data:
https://registry.opendata.aws/ecmwf-era5/

For list of variables, see:
    https://github.com/planet-os/notebooks/blob/master/aws/era5-pds.md
    (Note: Do not include the .nc extension in variable names)
"""

import warnings
import xarray as xr
from dabench.data import _data


class AWS(_data.Data):
    """Class for loading ERA5 data from AWS Open Data

    Notes:
        Source: https://registry.opendata.aws/ecmwf-era5/
        Data is HRES sub-daily.

    Attributes:
        system_dim (int): System dimension
        time_dim (int): Total time steps
        variables (list of strings): Names of ERA5 variables to load.
            For list of variables, see:
            https://github.com/planet-os/notebooks/blob/master/aws/era5-pds.md
            NOTE: Do NOT include .nc extension in variable name.
            Default is ['air_temperature_at_2_metres']
        months (list of strings): List of months to include in '01', '02', etc.
            format.
        years (list of integers): List of years to include. Data is available
            from 1979 to present.
        min_lat (float): Minimum latitude for bounding box. If None, loads
            global data (which can be VERY large). Bounding box default covers
            Cuba.
        max_lat (float): Max latitude for bounding box (see min_lat for info).
        min_lon (float): Min latitude for bounding box (see min_lat for info).
        max_lon (float): Max latitude for bounding box (see min_lat for info).
        store_as_jax (bool): Store values as jax array instead of numpy array.
            Default is False (store as numpy).
    """
    def __init__(
            self,
            variables=['air_temperature_at_2_metres'],
            months=['01', '02', '03', '04', '05', '06',
                    '07', '08', '09', '10', '11', '12'],
            years=[2020],
            # Defaults are Cuba bounding box
            min_lat=19.8554808619,
            max_lat=23.1886107447,
            min_lon=-84.9749110583,
            max_lon=-74.1780248685,
            system_dim=None,
            time_dim=None,
            store_as_jax=False,
            **kwargs
            ):
        self.variables = variables
        self.months = months
        self.years = years
        self.min_lat = min_lat
        self.max_lat = max_lat
        self.min_lon = min_lon
        self.max_lon = max_lon

        super().__init__(system_dim=system_dim, time_dim=time_dim,
                         values=None, delta_t=None, store_as_jax=store_as_jax,
                         **kwargs)

    def _build_urls(self):
        file_pattern = 'http://era5-pds.s3.amazonaws.com/zarr/{year}/{month}/data/{variable}.zarr'
        urls_mapper = [file_pattern.format(year=y, month=m, variable=v)
                       for y in self.years
                       for m in self.months
                       for v in self.variables
                       ]

        return urls_mapper

    def _load_aws_era5(self):
        """Load data from AWS OpenDataStore"""

        urls_mapper = self._build_urls()

        ds = xr.open_mfdataset(urls_mapper, engine='zarr',
                               coords='minimal', compat='override',
                               parallel=True)

        if self.min_lat is not None and self.max_lat is not None:
            # Subset by lat boundaries
            ds = ds.sel(lat=slice(self.max_lat, self.min_lat))
        if self.min_lon is not None and self.max_lon is not None:
            # Convert west longs to degrees east
            subset_min_lon = self.min_lon
            subset_max_lon = self.max_lon
            if subset_min_lon < 0:
                subset_min_lon += 360
            if subset_max_lon < 0:
                subset_max_lon += 360
            # Subset by lon boundaries
            ds = ds.sel(lon=slice(subset_min_lon, subset_max_lon))

        self._import_xarray_ds(ds)

    def generate(self):
        """Alias for _load_aws_era5"""
        warnings.warn('AWS.generate() is an alias for the load() method. '
                      'Proceeding with downloading ERA5 data from AWS...')
        self._load_aws_era5()

    def load(self):
        """Alias for _load_aws_era5"""
        self._load_aws_era5()
