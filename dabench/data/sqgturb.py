"""
sqgturb: Surface Quasi-Geostrophic Turbulance
(a.k.a. constant PV f-plane QG turbulence)

Original source software developed by Jeff Whitaker
https://github.com/jswhit/sqgturb

copyright: 2016 by Jeffrey Whitaker.

Permission to use, copy, modify, and distribute this software and
its documentation for any purpose and without fee is hereby granted,
provided that the above copyright notice appear in all copies and that
both the copyright notice and this permission notice appear in
supporting documentation.

THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE,
INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS. IN NO
EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, INDIRECT OR
CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF
USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR
OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
PERFORMANCE OF THIS SOFTWARE.

References:
http://journals.ametsoc.org/doi/pdf/10.1175/2008JAS2921.1 (section 3)
http://journals.ametsoc.org/doi/pdf/10.1175/1520-0469%281978%29035%3C0774%3AUPVFPI%3E2.0.CO%3B2

Dynamics include Ekman damping, linear thermal relaxation back to equilibrium
    jet, and hyperdiffusion. The pv has units of meters per second. scale by
    f*theta0/g to convert to temperature.
Uses the FFT spectral collocation method with 4th order Runge Kutta time
    stepping (dealiasing with 2/3 rule, hyperdiffusion treated implicitly).
"""

import jax.numpy as jnp
from jax.numpy.fft import rfft2, irfft2
from dabench.data import data


class DataSQGturb(data.Data):
    """Class to set up SQGturb model and manage data.

    Attributes:
        system_dim (int): The dimension of the system state
        time_dim (int): The dimension of the timeseries (not used)
        delta_t (float): model time step (seconds)
        x0 (ndarray, float): Initial state, array of floats of size
            (system_dim).
        f (float): coriolis
        nqr (float): Brunt-Vaisalla (buoyancy) freq squared
        L (float): size of square domain
        H (float): height of upper boundary
        U (float): basic state velocity at z = H
        r (float): Ekman damping (at z=0)
        tdiab (float): thermal relaxation damping
        diff_order (int): hyperdiffusion order
        diff_efold (float): hyperdiff time scale
        symmetric (bool): symmetric jet, or jet with U=0 at sf
        dealias (bool): if True, dealiasing applied using 2/3 rule
        precision (char): 'single' or 'double'. Default is 'single'
        tstart (float): initialize time counter
    """

    def __init__(self,
                 pv,
                 f=1.0e-4,
                 nsq=1.0e-4,
                 L=20.0e6,
                 H=10.0e3,
                 U=30.0,
                 r=0.0,
                 tdiab=10.0 * 86400,
                 diff_order=8,
                 diff_efold=None, # diff_efold = 86400./3.
                 symmetric=True,
                 dealias=True,
                 precision='single',
                 tstart=0,
                 system_dim=None,
                 input_dim=None,
                 output_dim=None,
                 time_dim=None,
                 values=None,
                 times=None,
                 delta_t=None,
                 **kwargs,
                 ):

        super().__init__(system_dim=system_dim, input_dim=input_dim,
                         output_dim=output_dim, time_dim=time_dim,
                         values=values, times=times, delta_t=delta_t,
                         **kwargs)
         
        # Set the initial state and dimensions
        pvspec = rfft2(pv)
        self.x0 = pvspec.ravel()
        self.Nv, self.Nx, self.Ny = pvspec.shape
        system_dim = self.Nv * self.Nx * self.Ny

        # initialize SQG model.
        if pv.shape[0] != 2:
            raise ValueError("1st dim of pv should be 2")
        # N is number of grid points in each direction in physical space
        N = pv.shape[1]
        # N should be even
        if N % 2:
            raise ValueError("2nd dim of pv (N) must be even"
                             "(powers of 2 are fastest)")
        self.N = N

        # Set data type based on precision attribute
        if precision == "single":
            # ffts in single precision (faster, default)
            dtype = jnp.float32
        elif precision == "double":
            # ffts in double precision
            dtype = jnp.float64
        else:
            raise ValueError("Precision must be 'single' or 'double'")

        # Time step and diff_efold must both be specified
        if delta_t is None:
            raise ValueError("must specify time step delta_t = {}".format(
                delta_t))
        if diff_efold is None:
            raise ValueError("must specify efolding time scale for diffusion")

        # Force arrays to be float32 for precision='single' (for faster ffts)
        self.nsq = jnp.array(nsq, dtype)
        self.f = jnp.array(f, dtype)
        self.H = jnp.array(H, dtype)
        self.U = jnp.array(U, dtype)
        self.L = jnp.array(L, dtype)
        self.delta_t = jnp.array(delta_t, dtype)
        self.dealias = dealias
        if r < 1.0e-10:
            self.ekman = False
        else:
            self.ekman = True
        self.r = jnp.array(r, dtype)          # Ekman damping (at z=0)
        self.tdiab = jnp.array(tdiab, dtype)  # thermal relaxation damping.

        # Initialize time counter
        self.t = tstart

        # Setup basic state pv (for thermal relaxation)
        self.symmetric = symmetric
        y = jnp.arange(0, self.L, self.L / self.N, dtype=dtype)
        pi = jnp.array(jnp.pi, dtype)
        l = 2.0 * pi / self.L
        mu = l * jnp.sqrt(nsq) * self.H / self.f
        if symmetric:
            # symmetric version, no difference between upper and lower boundary
            # l = 2.*pi/L and mu = l*N*H/f
            # u = -0.5*U*np.sin(l*y)*np.sinh(mu*(z-0.5*H)/H)*np.sin(l*y)/np.sinh(0.5*mu)
            # theta = (f*theta0/g)*(0.5*U*mu/(l*H))*np.cosh(mu*(z-0.5*H)/H)*
            # np.cos(l*y)/np.sinh(0.5*mu)
            # + theta0 + (theta0*nsq*z/g)
            pvbar = (
                -(mu * 0.5 * self.U / (l * self.H))
                * jnp.cosh(0.5 * mu)
                * jnp.cos(l * y)
                / jnp.sinh(0.5 * mu)
                )
        else:
            # asymmetric version, equilibrium state has no flow at surface and
            # temp gradient slightly weaker at sfc.
            # u = U*np.sin(l*y)*np.sinh(mu*z/H)*np.sin(l*y)/np.sinh(mu)
            # theta = (f*theta0/g)*(U*mu/(l*H))*np.cosh(mu*z/H)*
            # np.cos(l*y)/np.sinh(mu)
            # + theta0 + (theta0*nsq*z/g)
            pvbar = (-(mu * self.U / (l * self.H)) *
                     jnp.cos(l * y) / jnp.sinh(mu))
            pvbar = pvbar.at[1, :].set(pvbar[0, :] * jnp.cosh(mu))
        pvbar = pvbar.astype(dtype)

        # Add extra dimension to support multiplication
        pvbar = jnp.expand_dims(pvbar, 2)
        pvbar = pvbar * jnp.ones((2, N, N), dtype)
        self.pvbar = pvbar
        # state to relax to with timescale tdiab
        self.pvspec_eq = rfft2(pvbar)
        # initial pv field (spectral)
        self.pvspec = rfft2(pv)

        # Spectral variables
        k = (N * jnp.fft.fftfreq(N))[0: (N // 2) + 1]
        l = N * jnp.fft.fftfreq(N)
        k, l = jnp.meshgrid(k, l)
        k = k.astype(dtype)
        l = l.astype(dtype)

        # Dimensionalize wavenumbers.
        k = 2.0 * pi * k / self.L
        l = 2.0 * pi * l / self.L
        self.ksqlsq = k ** 2 + l ** 2
        self.ik = (1.0j * k).astype(jnp.complex64)
        self.il = (1.0j * l).astype(jnp.complex64)

        # Arrays needed for dealiasing nonlinear Jacobian
        if dealias:
            k_pad = ((3 * N // 2) * jnp.fft.fftfreq(3 * N // 2))[
                    0: (3 * N // 4) + 1]
            l_pad = (3 * N // 2) * jnp.fft.fftfreq(3 * N // 2)
            k_pad, l_pad = jnp.meshgrid(k_pad, l_pad)
            k_pad = k_pad.astype(dtype)
            l_pad = l_pad.astype(dtype)
            k_pad = 2.0 * pi * k_pad / self.L
            l_pad = 2.0 * pi * l_pad / self.L
            self.ik_pad = (1.0j * k_pad).astype(jnp.complex64)
            self.il_pad = (1.0j * l_pad).astype(jnp.complex64)

        mu = jnp.sqrt(self.ksqlsq) * jnp.sqrt(self.nsq) * self.H / self.f
        mu = mu.clip(jnp.finfo(mu.dtype).eps)  # clip to avoid NaN
        self.Hovermu = self.H / mu
        mu = mu.astype(jnp.float64)  # cast to avoid overflow in sinh
        self.tanhmu = jnp.tanh(mu).astype(dtype)  # cast back to original type
        self.sinhmu = jnp.sinh(mu).astype(dtype)
        self.diff_order = jnp.array(diff_order, dtype)  # hyperdiffusion order
        self.diff_efold = jnp.array(diff_efold, dtype)  # hyperdiff time scale
        ktot = jnp.sqrt(self.ksqlsq)
        ktotcutoff = jnp.array(pi * N / self.L, dtype)

        # Integrating factor for hyperdiffusion
        # with efolding time scale for diffusion of shortest wave (N/2)
        self.hyperdiff = jnp.exp((-self.dt / self.diff_efold) *
                                 (ktot / ktotcutoff) ** self.diff_order)
