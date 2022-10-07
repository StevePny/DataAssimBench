"""Load data from CPC ENSO index into data object"""

from urllib import request
import logging
import warnings
import jax.numpy as jnp
import numpy as np

from dabench.data import data


class DataENSOIDX(data.Data):
    """Class to get ENSO index from CPC website

    Notes:
        Source: https://www.cpc.ncep.noaa.gov/data/indices/

    Attributes:
        system_dim (int): system dimension
        time_dim (int): total time steps
        file_dict (dict): Lists of files to get. Dict keys are type of data:
                'wnd': Wind
                'slp': Sea level pressure
                'soi': Southern Oscillation Index
                'soi3m' Southerm Oscillation Index 3-month running mean
                'sst': SST indices
                'desst': Detrended Nino3.4 index
                'rsst': Regional SST index (North and South Atlantic, and
                    Tropics)
                'olr': Outgoing long-wave radiance
                'cpolr': Central Pacific OLR
            Dict values are individual files from the website, see full list at
                https://www.cpc.ncep.noaa.gov/data/indices/
            Default is {'wnd': ['zwnd200'], 'slp': ['darwin']}
        var_types (dict): List of variables within file to get. Dict keys are
            type of data (see list in file_dict description). Dict values are
            type of variable:
                'ori' = Original
                'ano' = Anomaly
                'std' = Standardized Anomaly
            sst (sea surface temp) has nino3, nino34, and nino12 options, see
                for more info: https://climatedataguide.ucar.edu/climate-data/nino-sst-indices-nino-12-3-34-4-oni-and-tni
            rsst (regional sea surface temp) has prefixes: 'sa_' is
                South Atlantic, 'na_' is North Atlantic, 'tr_' is Tropics.
            Default depends on data type (first string in each value list)
                'wnd'='ori', 'lp'='ori', 'soi'='ano', 'soi3m'='ori',
                'eqsoi'='std', 'sst'='nino12', 'desst'='ori', 'rsst'='na',
                'olr'='ori', 'cpolr'='ano'
    """

    def __init__(self, file_dict=None, var_types=None, system_dim=None,
                 time_dim=None, **kwargs):

        """Initialize DataENSOIDX object, subclass of Data"""

        # Full list of file names at bottom of this page:
        # https://www.cpc.ncep.noaa.gov/data/indices/Readme.index.shtml
        # This dict is used as an example. Can copy and edit down as needed
        file_dict_full = {'wnd': ['zwnd200', 'wpac850', 'cpac850', 'epac850',
                                  'qbo.u30.index', 'qbo.u50.index'],
                          'slp': ['darwin', 'tahiti'],
                          'soi': ['soi'],
                          'soi3m': ['soi.3m.txt'],
                          'eqsoi': ['rindo_slpa.for', 'repac_slpa.for',
                                    'reqsoi.for', 'reqsoi.3m.for'],
                          'sst': ['sstoi.indices',
                                  'ersst5.nino.mth.91-20.ascii'],
                          'desst': ['detrend.nino34.ascii.txt'],
                          'rsst': ['sstoi.atl.indices'],
                          'olr': ['olr'],
                          'cpolr': ['cpolr.mth.91-20.ascii']
                          }

        # ori: Original, ano: Anomaly, std: Standardized Anomaly
        # Some data types have unique values, e.g. nino12.
        # See full descriptions at https://www.cpc.ncep.noaa.gov/data/indices/
        var_types_full = {'wnd': ['ori', 'ano', 'std'],
                          'slp': ['ori', 'ano', 'std'],
                          'soi': ['ano', 'std'],
                          'soi3m': ['ori'],
                          'eqsoi': ['std'],
                          'sst': ['nino12', 'nino12_ano', 'nino3', 'nino3_ano',
                                  'nino4', 'nino4_ano', 'nino34',
                                  'nino34_ano'],
                          'desst': ['ori', 'adj', 'ano'],
                          'rsst': ['na', 'na_ano', 'sa', 'sa_ano', 'tr',
                                   'tr_ano'],
                          'olr': ['ori', 'ano', 'std'],
                          'cpolr': ['ano']}

        # Default if file_dict is None
        if file_dict is None:
            file_dict = {'wnd': ['zwnd200'],
                         'slp': ['darwin']}

        # Defaults if var_types in None
        if var_types is None:
            var_types = {'wnd': ['ori'],
                         'slp': ['ori'],
                         'soi': ['ano'],
                         'soi3m': ['ori'],
                         'eqsoi': ['std'],
                         'sst': ['nino12'],
                         'desst': ['ori'],
                         'rsst': ['na'],
                         'olr': ['ori'],
                         'cpolr': ['ano']}

        all_years = {}
        all_vals = {}
        # Loop over variable types
        for var in file_dict:
            # Loop over file names within variable types
            for file_name in file_dict[var]:
                all_vals, all_years = self._download_cpc_vals(
                        file_name, var, var_types, var_types_full,
                        all_vals, all_years)

        # Combine all variable values and years
        common_vals, common_years, names = self._combine_vals_years(
            all_vals, all_years)
        # Transpose vals to fit (time_dim, system_dim) convention of dabench
        values = common_vals.T
        times = common_years
        self.names = names
        logging.debug('ENSOIDXData.__init__: system dim x time dim: %s x %s',
                      len(names), len(times))

        # Set system_dim
        if system_dim is None:
            system_dim = len(names)
        elif system_dim != len(names):
            warnings.warn('ENSOIDXData.__init__: provided system_dim is '
                          '{}, but setting to len(names) = {}.'.format(
                              system_dim, len(names))
                          )
            system_dim = len(names)

        super().__init__(system_dim=system_dim, time_dim=time_dim,
                         values=values, delta_t=None, **kwargs)

    def _download_cpc_vals(self, file_name, var, var_types, var_types_full,
                           all_vals, all_years):
        """Downloads data for one file_name and variable pair

        Returns:
            Tuple containing values (ndarray), years (ndarray), and dict key
                name (string).
        """

        # The downloaded text files are all different, these dicts help parse
        # Number of blocks (each representing a variable) in file
        n_block = {'wnd': 3,
                   'slp': 3,
                   'soi': 2,
                   'soi3m': 1,
                   'eqsoi': 1,
                   'sst': 1,
                   'desst': 1,
                   'rsst': 1,
                   'olr': 3,
                   'cpolr': 1}

        # Size of header at top of file and between blocks
        n_header = {'wnd': 4,
                    'slp': 4,
                    'soi': 4,
                    'soi3m': 1,
                    'eqsoi': 0,
                    'sst': 1,
                    'desst': 1,
                    'rsst': 1,
                    'olr': 4,
                    'cpolr': 1}
        # List to store text lines from file
        tmp = []
        for line in request.urlopen(
                'https://www.cpc.ncep.noaa.gov/data/indices/' +
                file_name):
            tmp.append(line)
        n_lines = len(tmp)
        # These variables share common file format
        # Use _get_vals()
        if var in ['wnd', 'slp', 'soi', 'soi3m', 'olr']:
            block_size = int(n_lines/n_block[var])
            for ni, i in enumerate(jnp.arange(n_block[var])[np.in1d(
                    var_types_full[var], var_types[var])]):
                vals, years = self._get_vals(tmp[i * block_size:
                                                 (i+1) * block_size],
                                             n_header[var])
                name = file_name + '_' + var_types[var][ni]
                logging.debug('ENSOIDXData.__init__: Opening %s', name)
                all_vals[name] = vals
                all_years[name] = years
        # eqsoi uses _get_eqsoi()
        elif var == 'eqsoi':
            vals, years = self._get_eqsoi(tmp,)
            name = file_name+'_'+var_types[var][0]
            logging.debug('ENSOIDXData.__init__: Opening %s', name)
            all_vals[name] = vals
            all_years[name] = years
        # These vars use _get_sst()
        elif var in ['sst', 'cpolr', 'desst', 'rsst']:
            vals, years = self._get_sst(tmp, jnp.in1d(
                    var_types_full[var], var_types[var])
                )
            for i in range(len(var_types[var])):
                name = file_name+'_'+var_types[var][i]
                logging.debug('ENSOIDXData.__init__: Opening %s', name)
                all_vals[name] = vals[i]
                all_years[name] = years
        else:
            raise ValueError('Variable name {} not recognized'.format(
                var))

        return all_vals, all_years

    def _combine_vals_years(self, all_vals, all_years):
        # Find common years between variables
        common_years = list(all_years.values())[0]
        for v in all_vals:
            # get rid of missing values
            valid_years = all_years[v][(all_vals[v] != -999.9) &
                                       (all_vals[v] != 999.9)]
            common_years = jnp.intersect1d(common_years, valid_years)

        # Concatenate values across variables
        common_vals = []
        for v in all_vals:
            common_vals.append(all_vals[v][jnp.in1d(all_years[v],
                                                    common_years)])
        common_vals = jnp.array(common_vals)
        names = list(all_vals.keys())

        return common_vals, common_years, names

    def _get_vals(self, tmp, n_header):
        """Parses text lines from files of most data types

        Args:
            tmp (list): List of lines from text file
            n_header (int): Number of lines in header at top of file and
                between blocks.

        Returns:
            Tuple of data values (ndarray) and years (ndarray).
        """

        n_years = len(tmp)-n_header
        vals = []
        years = []
        for y in range(n_years):
            years.append(int(tmp[n_header:][y][:4]))
            try:
                vals.append([float(tmp[n_header:][y][4:][m*6:m*6+6])
                             for m in range(12)])
            # TODO: Fix bare except
            except:
                vals.append([float(tmp[n_header:][y][4:][m*7:m*7+7])
                             for m in range(12)])

        vals = jnp.concatenate(jnp.array(vals))
        years = jnp.concatenate(jnp.reshape(jnp.array(years*12),
                                            (12, -1)).T +
                                jnp.linspace(0, 1, 12, endpoint=False))

        return vals, years

    def _get_eqsoi(self, tmp):
        """Parses text lines from eqsoi file.

        Args:
            tmp (list): List of lines from text file

        Returns:
            Tuple of data values (ndarray) and years (ndarray).
        """
        vals = []
        years = []
        for line in tmp:
            l_split = [float(e) for e in line.split()]
            years.append(l_split[0])
            vals.append(l_split[1:])
        vals = jnp.concatenate(jnp.array(vals))
        years = jnp.concatenate(jnp.reshape(jnp.array(years*12), (12, -1)).T +
                                jnp.linspace(0, 1, 12, endpoint=False))

        return vals, years

    def _get_sst(self, tmp, var_types_indices):
        """Parses text lines from sst file.

        Args:
            tmp (list): List of lines from text file
            var_types_indices (list): List of variable type indices. Variable
                types  are: ['nino12', 'nino12_ano', 'nino3', 'nino3_ano',
                             'nino4', 'nino4_ano', 'nino34', 'nino34_ano']
                [0] is 'nino12' only, [1] is 'nino12_ano', [0, 2] is 'nino12'
                and 'nino3', etc.

        Returns:
            Tuple of data values (ndarray) and years (ndarray).
        """
        n_header = 1
        n_row = len(tmp)
        n_v = len(var_types_indices)
        vals = []
        years = []
        for r in range(n_header, n_row):
            years.append(float(tmp[r][:4]) + (float(tmp[r][4:8])-1)/12)
            vals.append([float(tmp[r][8:][v*8:v*8+8])
                         for v in jnp.arange(n_v)[var_types_indices]])
        vals = jnp.array(vals).T
        years = jnp.array(years)

        return vals, years
