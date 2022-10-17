"""Interface with pyqg to run quasigeostrpohic models

Requires pyqg: https://pyqg.readthedocs.io/

"""
import logging
import numpy as np
from copy import deepcopy

from dabench.data import data

logging.basicConfig(filename='logfile.log', level=logging.DEBUG)

try:
    import pyqg
except ImportError:
    pyqg = None
    raise ImportError(
        'Package: pyqg not found!\n'
        'To install via conda: conda install -c conda-forge pyqg\n'
        'For more information: https://pyqg.readthedocs.io/en/latest/installation.html'
        )


class DataPYQG(data.Data):
    """ Class to set up quasi-geotropic model

    The data class is simply a wrapper of a "optional" pyqg package.
    See https://pyqg.readthedocs.io

    Attributes:
        system_dim (int): system dimension
        beta (float): Gradient of coriolis parameter. Units: meters^-1 *
            seconds^-1
        rek (float): Linear drag in lower layer. Units: seconds^-1
        rd (float): Deformation radius. Units: meters.
        delta (float): Layer thickness ratio (H1/H2)
        U1 (float): Upper layer flow. Units: m/s
        U2 (float): Lower layer flow. Units: m/s
        H1 (float): Upper layer thickness.
        H2 (float): Lower layer thickness.
        nx (int): Number of grid points in the x direction.
        ny (int): Number of grid points in the y direction (default: nx).
        L (float): Domain length in x direction. Units: meters.
        W (float): Domain width in y direction. Units: meters (default: L).
        filterfac (float): amplitdue of the spectral spherical filter
            (originally 18.4, later changed to 23.6).
        delta_t          (float): Numerical timestep. Units: seconds.
        twrite (int): Interval for cfl writeout. Units: number of timesteps.
        tmax (float): Total time of integration (overwritten by t_final).
            Units: seconds.
        tavestart (float): Start time for averaging. Units: seconds.
        tsnapstart (float): Start time for snapshot writeout. Units: seconds.
        taveint (float): Time interval for accumulation of diagnostic averages.
            Units: seconds. (For performance purposes, averaging does not have
            to occur every timestep)
        tsnapint (float): Time interval for snapshots. Units: seconds.
        ntd (int): Number of threads to use. Should not exceed the number of
            cores on your machine.
    """
    def __init__(self,
                 beta=1.5e-11,
                 rd=15000.0,
                 delta=0.25,
                 H1=500,
                 U1=0.025,
                 U2=0.0,
                 x0=None,
                 delta_t=7200,
                 time_dim=None,
                 values=None,
                 times=None,
                 **kwargs):
        """ Initialize DataPYQG object, subclass of Data

        See https://pyqg.readthedocs.io/en/latest/api.html for more details.

        Args:

        """

        self.m = pyqg.QGModel(beta=beta, rd=rd, delta=delta, H1=H1, U1=U1,
                              U2=U2, twrite=10000, ntd=4, **kwargs)

        system_dim = self.m.q.size
        super().__init__(system_dim=system_dim, time_dim=time_dim,
                         values=values, times=times, delta_t=delta_t,
                         **kwargs)

        self.x0 = x0

    def generate(self, n_steps=None, t_final=None, x0=None):
        """Generates values and times, saves them to the data object

        Notes:
            Either provide n_steps or t_final in order to indicate the length
            of the forecast. These are used to set the values, times, and
            time_dim attributes.

        Args:
            n_steps (int): Number of timesteps. One of n_steps OR
                t_final must be specified.
            t_final (float): Final time of trajectory. One of n_steps OR
                t_final must be specified.
            x0 (ndarray, optional): the initial conditions. Can also be
                provided when initializing model object. If provided by
                both, the generate() arg takes precedence.
        """

        # Checks
        # Check that n_steps or t_final is supplied
        if n_steps is not None:
            t_final = n_steps * self.delta_t
        elif t_final is not None:
            n_steps = int(t_final/self.delta_t)
        else:
            raise TypeError('Either n_steps or t_final must be supplied as an '
                            'input argument.')

        # Check that x0 initial conditions is supplied
        if x0 is None:
            if self.x0 is not None:
                x0 = self.x0
                if (len(x0.shape) != 3) and (x0.shape[0] != 2):
                    raise ValueError(
                        'Initial condition x0 must be 3D array and the last '
                        'dimension must to 2 for this 2-layer QG model')
                self.m.set_q1q2(x0[0], x0[1])
            else:
                raise TypeError('Initial condition is None, x0 = {}. it must '
                                'either be provided as an argument or set as '
                                'an attribute in the model object.'.format(x0))

        # Integrate and store values and times
        self.m.dt = self.delta_t
        self.m.tmax = t_final
        self.times = np.arange(0, t_final, self.delta_t)

        # Run simulation
        qs = self.__advance__()

        # Save values
        self.values = qs
        self.time_dimension = self.values.shape[0]

    def forecast(self, n_steps=None, t_final=None, x0=None):
        """Alias for self.generate(), except returns values as output"""
        self.generate(n_steps, t_final, x0)

        return self.values

    def __advance__(self,):
        """Advances the QG model according to set attributes

        Returns:
            qs (array_like): absolute potential vorticity (relative potential
                vorticity + background vorticity).
        """
        qs = []
        for _ in self.m.run_with_snapshots(tsnapstart=0, tsnapint=self.m.dt):
            qs.append(deepcopy(self.m.q))

        # q was in (nz,ny,nx), qs is now in (nt,nz,nx,ny)
        qs = np.moveaxis(np.array(qs), -2, -1) 
        qs[:, 0] += self.m.Qy[0]*self.m.y  
        qs[:, 1] += self.m.Qy[1]*self.m.y

        return qs
