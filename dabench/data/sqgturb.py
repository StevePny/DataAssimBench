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
        pv (ndarray): Potential vorticity array
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
                 diff_efold=None,
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
            raise ValueError('1st dim of pv should be 2')
        # N is number of grid points in each direction in physical space
        N = pv.shape[1]
        # N should be even
        if N % 2:
            raise ValueError('2nd dim of pv (N) must be even'
                             '(powers of 2 are fastest)')
        self.N = N

        # Set data type based on precision attribute
        if precision == 'single':
            # ffts in single precision (faster, default)
            dtype = jnp.float32
        elif precision == 'double':
            # ffts in double precision
            dtype = jnp.float64
        else:
            raise ValueError('Precision must be "single" or "double"')

        # Time step and diff_efold must both be specified
        if delta_t is None:
            raise ValueError('must specify time step delta_t = {}'.format(
                delta_t))
        if diff_efold is None:
            raise ValueError('must specify efolding time scale for diffusion')

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

    # Private support methods
    def _invert(self, pvspec=None):
        """Inverts pvspec"""

        if pvspec is None:
            pvspec = self.pvspec

        # invert boundary pv to get streamfunction
        psispec = jnp.empty((2, self.N, self.N // 2 + 1), dtype=pvspec.dtype)
        psispec = jnp.array([self.Hovermu * ((pvspec[1] / self.sinhmu) -
                                             (pvspec[0] / self.tanhmu)),
                             self.Hovermu * ((pvspec[1] / self.tanhmu) -
                                             (pvspec[0] / self.sinhmu))
                             ], dtype=pvspec.dtype)

        return psispec

    def _invert_inverse(self, psispec=None):
        """ Performs inverse operation of '_invert'
        (Not used here.)
        """

        if psispec is None:
            psispec = self._invert(self.pvspec)

        # given streamfunction, return PV
        alpha = self.Hovermu
        th = self.tanhmu
        sh = self.sinhmu
        tmp1 = 1.0 / sh ** 2 - 1.0 / th ** 2
        tmp1 = tmp1.at[0, 0].set(1.0)
        pvspec = jnp.array([((psispec[0] / th) - (psispec[1] / sh)) /
                            (alpha * tmp1),
                            ((psispec[0] / sh) - (psispec[1] / th)) /
                            (alpha * tmp1)
                            ], dtype=psispec.dtype)
        # area mean PV not determined by streamfunction
        pvspec = pvspec.at[:, 0, 0].set(0.0)

        return pvspec

    def _specpad(self, specarr):
        """Pads spectral arrays with zeros to interpolate to 3/2 larger grid
            using inverse fft."""
        # Take care of normalization factor for inverse transform.
        specarr_pad = jnp.zeros((2, 3 * self.N // 2, 3 * self.N // 4 + 1),
                                dtype=specarr.dtype)
        specarr_pad = specarr_pad.at[:, :(self.N // 2), :(self.N // 2)].set(
            (2.25 * specarr[:, :(self.N // 2), :(self.N // 2)]))
        specarr_pad = specarr_pad.at[:, (-self.N // 2):, : (self.N // 2)].set(
            (2.25 * specarr[:, (-self.N // 2):, :(self.N // 2)]))

        # Include negative Nyquist frequency.
        specarr_pad = specarr_pad.at[:, :(self.N // 2), (self.N // 2)].set(
            jnp.conjugate(2.25 * specarr[:, :(self.N // 2), -1]))
        specarr_pad = specarr_pad.at[:, (-self.N // 2):, (self.N // 2)].set(
            jnp.conjugate(2.25 * specarr[:, (-self.N // 2):, -1]))
        return specarr_pad

    def _spectrunc(self, specarr):
        """Truncates spectral array to 2/3 size"""
        specarr_trunc = jnp.zeros((2, self.N, self.N // 2 + 1),
                                  dtype=specarr.dtype)
        specarr_trunc = specarr_trunc.at[
            :, :(self.N // 2), :(self.N // 2)].set(
            specarr[:, :(self.N // 2), :(self.N // 2)])
        specarr_trunc = specarr_trunc.at[
            :, (-self.N // 2):, :(self.N // 2)].set(
            specarr[:, (-self.N // 2):, :(self.N // 2)])
        return specarr_trunc

    def _xyderiv(self, specarr):
        """Calculates x and y derivatives"""
        if not self.dealias:
            xderiv = self.ifft2(self.ik * specarr)
            yderiv = self.ifft2(self.il * specarr)
        else:
            # pad spectral coeffs with zeros for dealiased jacobian
            specarr_pad = self._specpad(specarr)
            xderiv = self.ifft2(self.ik_pad * specarr_pad)
            yderiv = self.ifft2(self.il_pad * specarr_pad)
        return xderiv, yderiv

    # Public support methods
    def fft2(self, pv):
        """Alias method for FFT of PV"""
        return rfft2(pv)

    def ifft2(self, pvspec):
        """Alias method for inverse FFT of PV Spectral"""
        return irfft2(pvspec)

    def map2dto1d(self, pv):
        """Maps 2D PV to 1D system state"""
        return pv.ravel()

    def map1dto2d(self, x):
        """Maps 1D state vector to 2D PV"""
        return jnp.reshape(x, (self.Nv, self.Nx, self.Ny))

    def fft2_2dto1d(self, pv):
        """Runs FFT then maps from 2D to 1D"""
        pvspec = self.fft2(pv)
        return self.map2dto1d(pvspec)

    def ifft2_2dto1d(self, pvspec):
        """Runs inverse FFT then maps from 2D to 1D"""
        pv = self.ifft2(pvspec)
        return self.map2dto1d(pv)

    def map1dto2d_fft2(self, x):
        """Maps for 1D to 2D then runs FFT"""
        pv = self.map1dto2d(x)
        return self.fft2(pv)

    def map1dto2d_ifft2(self,  x):
        """Maps for 1D to 2D then runs inverse FFT"""
        pvspec = self.map1dto2d(x)
        return self.ifft2(pvspec)

    # Integration methods
    def integrate(self, f, x0, t_final, delta_t=None, include_x0=True,
                  t=None, **kwargs):
        """Advances pv forward number of timesteps given by 'n_steps' instance var.

        Note:
            If pv not specified, use pvspec instance variable.

        Args:
            f (function): right hand side (rhs) of the ODE
            x0 (ndarray): potential vorticity (pvspec) initial condition in
                spectral space
        """

        # Convert input state vector to a 2D spectral array
        pvspec = self.map1dto2d(x0)

        # Get number of time steps
        n_steps = int(t_final/self.delta_t)

        # Checks
        # Make sure that there is no remainder
        if not n_steps * delta_t == t_final:
            raise ValueError('Cannot have remainder in nsteps = {}, '
                             'delta_t = {}, t_final = {}, and n_steps * '
                             'delta_t = {}'.format(n_steps, delta_t, t_final,
                                                   n_steps*delta_t))

        # If delta_t not specified as arg for method, use delta_t from object
        if delta_t is None:
            delta_t = self.delta_t

        # If t not specified as arg for method, use t from object
        if t is None:
            t = self.t

        # If including initial state, add 1 to n_steps
        if include_x0:
            n_steps = n_steps + 1

        self.time_dimension = n_steps
        times = t + jnp.arange(n_steps)*delta_t
        values = jnp.empty((n_steps, self.system_dim), dtype=x0.dtype)

        # Integreate in spectral space
        for i in range(n_steps):
            values = values.at[i, :].set(pvspec.ravel())
            pvspec = self._rk4(f, pvspec)
            pvspec = self.hyperdiff * pvspec

            if jnp.isnan(pvspec).any():
                raise Exception('Model values contain NaNs. '
                                'EXITING at i={}...'.format(i))

        # Update internal states
        self.pvspec = pvspec
        self.t = times[-1]

        return values, times

    def rhs(self, pvspec=None):
        """computes tendencies of pv on z=0, H. Inverts pv to get 
            streamfunction.
        """

        if pvspec is None:
            pvspec = self.pvspec
        psispec = self._invert(pvspec)

        # Nonlinear jacobian and thermal relaxation
        psix, psiy = self._xyderiv(psispec)
        pvx, pvy = self._xyderiv(pvspec)
        jacobian = psix * pvy - psiy * pvx
        jacobianspec = self.fft2(jacobian)
        if self.dealias:
            # 2/3 rule: truncate spectral coefficients of jacobian
            jacobianspec = self._spectrunc(jacobianspec)
        dpvspecdt = ((1.0 / self.tdiab) * (self.pvspec_eq - pvspec) -
                     jacobianspec)

        # Ekman damping at boundaries.
        if self.ekman:
            dpvspecdt = dpvspecdt.at[0].set(dpvspecdt[0] + self.r *
                                            self.ksqlsq * psispec[0])
            # for asymmetric jet (U=0 at sfc), no Ekman layer at lid
            if self.symmetric:
                dpvspecdt = dpvspecdt.at[1].set(dpvspecdt[1] - self.r *
                                                self.ksqlsq * psispec[1])

        # save wind field
        self.u = -psiy
        self.v = psix
        return dpvspecdt

    def _rk4(self, f, x):
        """Updates pv using 4th order runge-kutta time step with implicit
            "integrating factor" treatment of hyperdiffusion.
        """

        self.rkstep = 0
        k1 = self.dt * f(x)
        self.rkstep = 1
        k2 = self.dt * f(x + 0.5 * k1)
        self.rkstep = 2
        k3 = self.dt * f(x + 0.5 * k2)
        self.rkstep = 3
        k4 = self.dt * f(x + k3)
        y = x + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        return y

