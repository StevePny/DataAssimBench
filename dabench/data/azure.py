"""Load data from Azure Planetary Computer

For now just ERA5 ECMWF data:
https://planetarycomputer.microsoft.com/dataset/era5-pds

For list of variables, see:
    https://github.com/planet-os/notebooks/blob/master/aws/era5-pds.md
    (Note 1: AWS and Azure have the same variables and variable names)
    (Note 2: Do not include the .nc extension in variable names)
"""

import warnings
import xarray as xr
from dabench.data import data
import pystac_client
import planetary_computer


class DataAzure(data.Data):
    """Class for loading ERA5 data from Azure Planetary Computer

    Notes:
        Source: https://planetarycomputer.microsoft.com/dataset/era5-pds
        Data is HRES sub-daily.

    Attributes:
        system_dim (int): System dimension
        time_dim (int): Total time steps
        variables (list of strings): Names of ERA5 variables to load.
            For list of variables, see:
            https://github.com/planet-os/notebooks/blob/master/aws/era5-pds.md
            NOTE: Do NOT include .nc extension in variable name.
            Default is ['air_temperature_at_2_metres']
        date_start (string): Start of time range to download, in 'yyyy-mm-dd'
            format. Can also just specify year ('yyyy') or year and month
            ('yyyy-mm'). Default is '2020-01'.
        date_end (string): End of time range to download, in 'yyyy-mm-dd'
            format. Can also just specify year ('yyyy') or year and month
            ('yyyy-mm'). Default is '2020-12'.
        min_lat (float): Minimum latitude for bounding box. If None, loads
            global data (which can be VERY large). Bounding box default covers
            Cuba.
        max_lat (float): Max latitude for bounding box (see min_lat for info).
        min_lon (float): Min latitude for bounding box (see min_lat for info).
        max_lon (float): Max latitude for bounding box (see min_lat for info).
    """
    def __init__(
            self,
            variables=['air_temperature_at_2_metres'],
            date_start='2020-01',
            date_end='2020-12',
            # Defaults are Cuba bounding box
            min_lat=19.8554808619,
            max_lat=23.1886107447,
            min_lon=-84.9749110583,
            max_lon=-74.1780248685,
            system_dim=None,
            time_dim=None,
            **kwargs
            ):
        self.variables = variables
        self.date_start = date_start
        self.date_end = date_end
        self.min_lat = min_lat
        self.max_lat = max_lat
        self.min_lon = min_lon
        self.max_lon = max_lon

        # Valid variable types
        self._fc_variables = [
            'precipitation_amount_1hour_Accumulation',
            'air_temperature_at_2_metres_1hour_Maximum',
            'air_temperature_at_2_metres_1hour_Minimum',
            'integral_wrt_time_of_surface_direct_downwelling_shortwave_flux_in_air_1hour_Accumulation'
            ]
        self._an_variables = [
            'surface_air_pressure', 'sea_surface_temperature',
            'eastward_wind_at_10_metres', 'air_temperature_at_2_metres',
            'eastward_wind_at_100_metres', 'northward_wind_at_10_metres',
            'northward_wind_at_100_metres', 'air_pressure_at_mean_sea_level',
            'dew_point_temperature_at_2_metres'
            ]

        super().__init__(system_dim=system_dim, time_dim=time_dim,
                         values=None, delta_t=None, **kwargs)

    def _fc_or_an_from_variable(self, var_name):
        if var_name in self._fc_variables:
            return 'fc'
        elif var_name in self._an_variables:
            return 'an'
        else:
            return None

    def _build_urls(self):
        data_types = [self._fc_or_an_from_variable(v) for v in self.variables]
        if None in data_types:
            missing_vars = self.variables[data_types == 'None']
            raise ValueError(
                '{vnames} are not valid variables.\n'
                'Valid variables are: {ds_vars}'.format(
                    vnames=missing_vars,
                    ds_vars=(self._fc_variables + self._an_variables))
                )
        elif 'fc' in data_types and 'an' in data_types:
            raise ValueError(
                'Your selected variables ({vnames}) contain both "analysis" '
                'and "forecast" variable types, which are not compatible.\n'
                'Select only variables of one type or the other.\n'
                '"analysis" variables: {an_vars}.\n'
                '"forecast" variables {fc_vars}.'.format(
                    vnames=self.variables, an_vars=self._an_variables,
                    fc_vars=self._fc_variables)
                )
        # Find assets
        catalog = pystac_client.Client.open(
            'https://planetarycomputer.microsoft.com/api/stac/v1/',
            modifier=planetary_computer.sign_inplace,
            )
        time_range = '{}/{}'.format(self.date_start, self.date_end)
        search = catalog.search(
            collections=['era5-pds'],
            datetime=time_range,
            query={'era5:kind': {'eq': data_types[0]}}
            )
        items = search.get_all_items()
        xarray_kwargs = items[0].assets[
            self.variables[0]].extra_fields['xarray:open_kwargs']

        links = [it.assets[v].href for it in items for v in self.variables]

        return links, xarray_kwargs

    def _load_azure_era5(self):
        """Load data from AWS OpenDataStore"""

        urls_mapper, xarray_kwargs = self._build_urls()

        ds = xr.open_mfdataset(urls_mapper, parallel=True, **xarray_kwargs)

        # Slice by time
        ds = ds.sel(time=slice(self.date_start, self.date_end))

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
        """Alias for _load_azure_era5"""
        warnings.warn('DataAWS.generate() is an alias for the load() method. '
                      'Proceeding with downloading ERA5 data from AWS...')
        self._load_azure_era5()

    def load(self):
        """Alias for _load_azure_era5"""
        self._load_azure_era5()
