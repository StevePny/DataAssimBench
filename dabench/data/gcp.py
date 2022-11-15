"""Load data from Google Cloud Platform Public Datasets

For now just ERA5 ECMWF data:
https://cloud.google.com/storage/docs/public-datasets/era5

For list of variables and more info, see:
    https://github.com/google-research/ARCO-ERA5#data-description

Note:
    Does not support data stored as spherical harmonic coefficients,
    and so cannot import 'model-level-wind' or 'single-level-surface'.
"""


import warnings
import xarray as xr
from dabench.data import data


class DataGCP(data.Data):
    """Class for loading ERA5 data from Google Cloud Platform

    Notes:
        Source: https://cloud.google.com/storage/docs/public-datasets/era5
        Data is hourly

    Attributes:
        system_dim (int): System dimension
        time_dim (int): Total time steps
        data_type (string): Which data type to load, must be one of:
            'single-level-forecast', 'single-level-reanalysis', or
            'model-level-moisture'. Default is 'single-level-reanalysis'.
        variables (list of strings): Abbreviated names of ERA5 variables to
            load. For description of variables and which data_type they belong
            to, see:
            https://github.com/google-research/ARCO-ERA5#data-description
            Default is ['sst'] (sea surface temperature)
        levels (int): For multi-level data, specifies which levels to download.
            Default is [1] (surface). Only applies to model-level-moisture.
        date_start (string): Start of time range to download, in 'yyyy-mm-dd'
            format. Can also just specify year ('yyyy') or year and month
            ('yyyy-mm'). Default is '2020-06-01'.
        date_end (string): End of time range to download, in 'yyyy-mm-dd'
            format. Can also just specify year ('yyyy') or year and month
            ('yyyy-mm'). Default is '2020-9-30'.
        min_lat (float): Minimum latitude for bounding box. If None, loads
            global data (which can be VERY large). Bounding box default covers
            Cuba.
        max_lat (float): Max latitude for bounding box (see min_lat for info).
        min_lon (float): Min latitude for bounding box (see min_lat for info).
        max_lon (float): Max latitude for bounding box (see min_lat for info).
    """
    def __init__(
            self,
            data_type='single-level-reanalysis',
            variables=['sst'],
            date_start='2020-06-01',
            date_end='2020-9-30',
            levels=[1],
            min_lat=19.8554808619,
            max_lat=23.1886107447,
            min_lon=-84.9749110583,
            max_lon=-74.1780248685,
            system_dim=None,
            time_dim=None,
            **kwargs
            ):

        # Check that data type is valid
        if data_type not in ['model-level-moisture', 'single-level-forecast',
                             'single-level-reanalysis']:
            raise ValueError(
                '{} is not an valid data_type. Select one of:\n '
                '"model-level-moisture", "single-level-forecast", '
                '"single-level-reanalysis"'.format(data_type)
                )

        self.data_type = data_type
        self.variables = variables
        self.date_start = date_start
        self.date_end = date_end
        self.levels = levels
        self.min_lat = min_lat
        self.max_lat = max_lat
        self.min_lon = min_lon
        self.max_lon = max_lon

        super().__init__(system_dim=system_dim, time_dim=time_dim,
                         values=None, delta_t=None, **kwargs)

    def _build_url(self):
        file_pattern = 'http://storage.googleapis.com/gcp-public-data-arco-era5/co/{data_type}.zarr'
        url = file_pattern.format(data_type=self.data_type)

        return url

    def _load_gcp_era5(self):
        """Load ERA5 data from Google Cloud Platform"""

        url = self._build_url()

        ds = xr.open_zarr(url, chunks={'time': 48}, consolidated=True)

        # Check that variables are in zarr
        missing_vars = [v for v in self.variables if v not in ds.data_vars]
        if len(missing_vars) > 0:
            raise ValueError(
                '{vnames} are not valid variables for data_type = {dtype}.\n'
                'Valid variables are: {ds_vars}'.format(
                    vnames=missing_vars, dtype=self.data_type,
                    ds_vars=list(ds.data_vars))
                )

        # Subset by selected variables
        ds = ds[self.variables]

        # Slice by time
        ds = ds.sel(time=slice(self.date_start, self.date_end))

        # Select levels (only for model-level-moisture)
        if self.data_type == 'model-level-moisture':
            ds = ds.where(ds.hybrid.isin(self.levels, drop=True))

        if self.min_lat is not None and self.max_lat is not None:
            # Subset by lat boundaries
            ds = ds.where(
                (ds.latitude < self.max_lat) &
                (ds.latitude > self.min_lat),
                drop=True)
        if self.min_lon is not None and self.max_lon is not None:
            # Convert west longs to degrees east
            subset_min_lon = self.min_lon
            subset_max_lon = self.max_lon
            if subset_min_lon < 0:
                subset_min_lon += 360
            if subset_max_lon < 0:
                subset_max_lon += 360
            # Subset by lon boundaries
            ds = ds.where(
                (ds.longitude < subset_max_lon) &
                (ds.longitude > subset_min_lon),
                drop=True)

        self._import_xarray_ds(ds)

    def generate(self):
        """Alias for _load_gcp_era5"""
        warnings.warn('DataGCP.generate() is an alias for the load() method. '
                      'Proceeding with downloading ERA5 data from GCP...')
        self._load_gcp_era5()

    def load(self):
        """Alias for _load_gcp_era5"""
        self._load_gcp_era5()
