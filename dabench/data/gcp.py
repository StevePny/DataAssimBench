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
from dabench.data import _data


class GCP(_data.Data):
    """Class for loading ERA5 data from Google Cloud Platform

    Notes:
        Source: https://cloud.google.com/storage/docs/public-datasets/era5
        Data is hourly

    Attributes:
        variables: Names of ERA5 variables to
            load. For description of variables, see:
            https://github.com/google-research/arco-era5?tab=readme-ov-file#full_37-1h-0p25deg-chunk-1zarr-v3
            Default is ['2m_temperature'] (Air temperature at 2 metres)
        date_start: Start of time range to download, in 'yyyy-mm-dd'
            format. Can also just specify year ('yyyy') or year and month
            ('yyyy-mm'). Default is '2020-06-01'.
        date_end: End of time range to download, in 'yyyy-mm-dd'
            format. Can also just specify year ('yyyy') or year and month
            ('yyyy-mm'). Default is '2020-9-30'.
        min_lat: Minimum latitude for bounding box. If None, loads
            global data (which can be VERY large). Bounding box default covers
            Cuba.
        max_lat: Max latitude for bounding box (see min_lat for info).
        min_lon: Min latitude for bounding box (see min_lat for info).
        max_lon: Max latitude for bounding box (see min_lat for info).
        store_as_jax: Store values as jax array instead of numpy array.
            Default is False (store as numpy).
    """
    def __init__(
            self,
            variables: list[str] = ['2m_temperature'],
            date_start: str = '2020-01-01',
            date_end: str = '2020-12-31',
            min_lat: float = 19.8554808619,
            max_lat: float =  23.1886107447,
            min_lon: float = -84.9749110583,
            max_lon: float = -74.1780248685,
            store_as_jax: bool = False,
            **kwargs
            ):

        self.variables = variables
        self.date_start = date_start
        self.date_end = date_end
        self.min_lat = min_lat
        self.max_lat = max_lat
        self.min_lon = min_lon
        self.max_lon = max_lon

        super().__init__(values=None, delta_t=None,
                         store_as_jax=store_as_jax, x0=None,
                         **kwargs)


    def _load_gcp_era5(self) -> xr.Dataset:
        """Load ERA5 data from Google Cloud Platform"""

        url = 'http://storage.googleapis.com/gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3'

        ds = xr.open_zarr(url, chunks={'time': 48}, consolidated=True, decode_coords='all')

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

        if self.min_lat is not None and self.max_lat is not None:
            # Subset by lat boundaries
            ds = ds.sel(latitude=slice(self.max_lat, self.min_lat))
        if self.min_lon is not None and self.max_lon is not None:
            # Convert west longs to degrees east
            subset_min_lon = self.min_lon
            subset_max_lon = self.max_lon
            if subset_min_lon < 0:
                subset_min_lon += 360
            if subset_max_lon < 0:
                subset_max_lon += 360
            # Subset by lon boundaries
            ds = ds.sel(longitude=slice(subset_min_lon, subset_max_lon))

        # Assign system dimension
        ds = ds.assign_attrs(system_dim=ds.to_stacked_array('system',['time']).sizes['system'])

        return ds

    def generate(self) -> xr.Dataset:
        """Alias for _load_gcp_era5"""
        warnings.warn('GCP.generate() is an alias for the load() method. '
                      'Proceeding with downloading ERA5 data from GCP...')
        return self._load_gcp_era5()

    def load(self) -> xr.Dataset:
        """Alias for _load_gcp_era5"""
        return self._load_gcp_era5()
