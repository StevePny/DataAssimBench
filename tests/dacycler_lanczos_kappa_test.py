"""The CG-implied Lanczos Ritz values (DABENCH_LANCZOS_KAPPA) recover a
known spectrum: lambda_max exactly, kappa as a lower bound."""
import numpy as np
import jax
import jax.numpy as jnp

from dabench.dacycler._var4d_operator_utils import pcg_lanczos_solve
from dabench.dacycler._var4d_operator import _print_lanczos_kappa


def test_lanczos_kappa_recovers_known_spectrum(capsys):
    jax.config.update("jax_enable_x64", True)
    rng = np.random.default_rng(0)
    n = 60
    Q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    ev = np.logspace(0, 4, n)
    A = jnp.asarray(Q @ np.diag(ev) @ Q.T)
    _, info = pcg_lanczos_solve(lambda p: A @ p,
                                jnp.asarray(rng.normal(size=n)),
                                max_iter=60, tol=1e-14)
    _print_lanczos_kappa(info["alphas"], info["betas"], info["n_iter"], 0)
    out = capsys.readouterr().out
    rmax = float(out.split("ritz_max=")[1].split()[0])
    kap = float(out.split("kappa_est=")[1].split()[0])
    assert abs(rmax - ev[-1]) / ev[-1] < 1e-6
    assert 0.5 * ev[-1] / ev[0] < kap <= 1.0001 * ev[-1] / ev[0]
