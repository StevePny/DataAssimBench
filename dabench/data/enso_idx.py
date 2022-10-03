"""Load data from CPC ENSO index into data object"""

from urllib import request
import logging
import warnings
import jax.numpy as jnp

from dabench.data import data


class DataENSOIDX(data.Data):
    """Class to get ENSO index from CPC website

    Notes:
        Source: https://www.cpc.ncep.noaa.gov/data/indices/

    Attributes:
        system_dim (int): system dimension
        time_dim (int): total time steps
        file_list (dict): list of files to get
        vtype (dict): list of variables in file
    """

    def __init__(self, file_list, vtype, system_dim=None, input_dim=None,
                 output_dim=None, time_dim=None, values=None, times=None,
                 **kwargs):

        """Initialize DataENSOIDX object, subclass of Data"""

        # Full list (https://www.cpc.ncep.noaa.gov/data/indices/)
        file_list_full = {'wnd': ['zwnd200', 'wpac850', 'cpac850', 'epac850', 
                                  'qbo.u30.index', 'qbo.u50.index'],  # Wind
                          'slp': ['darwin', 'tahiti'],  # Sea level pressure
                          'soi': ['soi'],  # Southern Oscillation index
                          'soi3m': ['soi.3m.txt'],  # Southern Oscillation index 3-month running mean
                          'eqsoi': ['rindo_slpa.for', 'repac_slpa.for', 
                                    'reqsoi.for', 'reqsoi.3m.for'],  # equatorial SOI
                          'sst': ['sstoi.indices', 
                                  'ersst5.nino.mth.91-20.ascii'],  # SST indices (see vtype)
                          'desst': ['detrend.nino34.ascii.txt'],  # detrended Nino3.4 index
                          'rsst': ['sstoi.atl.indices'],  # Regional SST index (north and south atlantic and tropics)
                          'olr': ['olr'],  # Outgoing long-wave radiance
                          'cpolr': ['cpolr.mth.91-20.ascii']  # Central pacific OLR
                          } 

        # ori: Original, ano: Anomaly, std: Standardized Anomaly
        vtype_full = {'wnd': ['ori', 'ano', 'std'],
                      'slp': ['ori', 'ano', 'std'],
                      'soi': ['ano', 'std'],
                      'soi3m': ['ori'],
                      'eqsoi': ['std'],
                      'sst': ['nino12', 'nino12_ano', 'nino3', 'nino3_ano',
                              'nino4', 'nino4_ano', 'nino34', 'nino34_ano'],
                      'desst': ['ori', 'adj', 'ano'],
                      'rsst': ['na', 'na_ano', 'sa', 'sa_ano', 'tr', 'tr_ano'],
                      'olr': ['ori', 'ano', 'std'],
                      'cpolr': ['ano']}

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

        if file_list is None:
            file_list = file_list_full

        all_years = {}
        all_vals = {}
        for var in file_list:
            for file in file_list[var]:
                tmp = []
                for line in request.urlopen("https://www.cpc.ncep.noaa.gov/data/indices/"+file):
                    tmp.append(line)
                n_lines = len(tmp)
                if var in ['wnd', 'slp', 'soi', 'soi3m', 'olr']:
                    block_size = int(n_lines/n_block[var])
                    for ni, i in enumerate(jnp.arange(n_block[var])[jnp.in1d(
                            vtype_full[var], vtype[var])]):
                        vals, years = self._get_vals(tmp[i * block_size:(i+1) *
                                                         block_size],
                                                     n_header[var])
                        name = file+"_"+vtype[var][ni]
                        logging.debug(f"ENSOIDXData.__init__: Opening {name}")
                        all_vals[name] = vals
                        all_years[name] = years
                elif var == 'eqsoi':
                    vals, years = self._get_eqsoi(tmp,)
                    name = file+"_"+vtype[var][0]
                    logging.debug(f"ENSOIDXData.__init__: Opening {name}")
                    all_vals[name] = vals
                    all_years[name] = years
                elif var in ['sst', 'cpolr', 'desst', 'rsst']:
                    vals, years = self._get_sst(tmp, jnp.in1d(vtype_full[var],
                                                vtype[var]))
                    for i in range(len(vtype[var])):
                        name = file+"_"+vtype[var][i]
                        logging.debug(f"ENSOIDXData.__init__: Opening {name}")
                        all_vals[name] = vals[i]
                        all_years[name] = years
        
        common_years = list(all_years.values())[0]
        for v in all_vals:
            # get rid of missing values
            valid_years = all_years[v][(all_vals[v] != -999.9) &
                                       (all_vals[v] != 999.9)]
            common_years = jnp.intersect1d(common_years, valid_years)

        common_vals = []
        for v in all_vals:
            common_vals.append(all_vals[v][jnp.in1d(all_years[v],
                                                    common_years)])
        common_vals = jnp.array(common_vals)
        names = list(all_vals.keys())
        
        values = common_vals
        times = common_years
        self.names = names
        logging.debug(f"ENSOIDXData.__init__: system dim x time dim: {len(names)} x {len(times)}")

        if (system_dim is None):
            system_dim = len(names)
        elif (system_dim != len(names)):
            warnings.warn(f"ENSOIDXData.__init__: provided system_dim is {system_dim}, but setting to len(names) = {len(names)}.")
            system_dim = len(names)

        super().__init__(system_dim=system_dim, input_dim=input_dim,
                         output_dim=output_dim, time_dim=time_dim,
                         values=values, delta_t=None, **kwargs)

    def _get_vals(self, tmp, n_header):
        n_years = len(tmp)-n_header
        vals = []
        years = []
        for y in range(n_years):
            years.append(int(tmp[n_header:][y][:4]))
            try:
                vals.append([float(tmp[n_header:][y][4:][m*6:m*6+6])
                             for m in range(12)])
            except:
                vals.append([float(tmp[n_header:][y][4:][m*7:m*7+7])
                             for m in range(12)])

        vals = jnp.concatenate(vals)
        years = jnp.concatenate(jnp.reshape(jnp.array(years*12),
                                            (12, -1)).T +
                                jnp.linspace(0, 1, 12, endpoint=False))
        return vals, years

    def _get_eqsoi(self, tmp):
        vals = []
        years = []
        for line in tmp:
            l = [float(e) for e in line.split()]
            years.append(l[0])
            vals.append(l[1:])
        vals = jnp.concatenate(vals)
        years = jnp.concatenate(jnp.reshape(jnp.array(years*12), (12, -1)).T +
                                jnp.linspace(0, 1, 12, endpoint=False))
        return vals, years    

    def _get_sst(self, tmp, vtype):
        n_header = 1
        n_row = len(tmp)
        n_v = len(vtype)
        vals = []
        years = []
        for r in range(n_header, n_row):
            years.append(float(tmp[r][:4]) + (float(tmp[r][4:8])-1)/12)
            vals.append([float(tmp[r][8:][v*8:v*8+8])
                         for v in jnp.arange(n_v)[vtype]])
        vals = jnp.array(vals).T
        years = jnp.array(years)
        return vals, years
