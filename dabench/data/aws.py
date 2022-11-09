"""Load data from AWS Registry of Open Data

For now just ERA5 ECMWF data:
https://registry.opendata.aws/ecmwf-era5/
"""


import xarray as xr
from dabench.data import data
import fsspec


class DataAWS(data.Data):
    """Class for loading data from AWS Open Data"""

    def __init__(
            self,
            variables=['air_temperature_at_2_metres'],
            months=['01', '02', '03', '04', '05', '06',
                    '07', '08', '09', '10', '11', '12'],
            years=[2020],
            system_dim=None,
            time_dim=None,
            **kwargs
            ):
        self.variables = variables
        self.months = months
        self.years = years

        super().__init__(system_dim=system_dim, time_dim=time_dim,
                         values=None, delta_t=None, **kwargs)

    def _build_urls(self):
        
        file_pattern = 'http://era5-pds.s3.amazonaws.com/zarr/{year}/{month}/data/{variable}.zarr'
        urls_mapper = [file_pattern.format(year=y, month=m, variable=v)
                       for y in self.years
                       for m in self.months
                       for v in self.variables
                       ]

        return urls_mapper

    def load_aws_era5(self):

        urls_mapper = self._build_urls()

        ds = xr.open_mfdataset(urls_mapper, engine='zarr',
                               concat_dim='time0', combine='nested',
                               coords='minimal', compat='override',
                               parallel=True)

        self._import_xarray_ds(ds)


