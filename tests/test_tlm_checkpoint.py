"""Checkpointed TLM scan (OceanCurrentsML, 17 Nov 2026): the blocked, checkpointed scan gives the same increments and the same gradient as the plain scan."""
import jax
import jax.numpy as jnp
import numpy as np

from dabench.dacycler import _var4d_operator_utils as ou
from dabench.dacycler._var4d_weak_utils import window_tlm_rollout_forced

jax.config.update("jax_enable_x64", True)


def _setup(T=12, D=5):
    rng = np.random.default_rng(0)
    A = jnp.asarray(rng.normal(size=(D, D)) * 0.3)
    step = lambda x: jnp.tanh(A @ x) + 0.1 * x  # noqa: E731
    xs = [jnp.asarray(rng.normal(size=D))]
    for _ in range(T):
        xs.append(step(xs[-1]))
    traj = jnp.stack(xs)
    tlm = lambda x, dx: jax.jvp(step, (x,), (dx,))[1]  # noqa: E731
    return traj, tlm, jnp.asarray(rng.normal(size=D)), jnp.asarray(rng.normal(size=(T, D)))


def _cost(traj, tlm, dx0, deta):
    return jnp.sum(window_tlm_rollout_forced(tlm, traj, dx0, deta) ** 2) + jnp.sum(ou.window_tlm_rollout(tlm, traj, dx0) ** 3)


def test_blocked_scan_matches_plain():
    traj, tlm, dx0, deta = _setup()
    try:
        ou.set_tlm_checkpoint_block(None)
        v0, g0 = jax.value_and_grad(_cost, argnums=(2, 3))(traj, tlm, dx0, deta)
        for b in (4, 5, 12, 50):  # 5: largest divisor of 12 below it is 4; 50: plain
            ou.set_tlm_checkpoint_block(b)
            v1, g1 = jax.value_and_grad(_cost, argnums=(2, 3))(traj, tlm, dx0, deta)
            np.testing.assert_allclose(v1, v0, rtol=1e-12)
            for a, c in zip(g1, g0):
                np.testing.assert_allclose(a, c, rtol=1e-10, atol=1e-12)
    finally:
        ou.set_tlm_checkpoint_block(None)
