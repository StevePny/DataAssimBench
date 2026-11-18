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


def test_obs_in_scan_selects_due_observations_and_refuses_overflow():
    """OceanCurrentsML, 18 Nov 2026: the in-scan observation evaluates only the observations due at each step (a table of OBS_PER_STEP_MAX); it equals the gather from the stacked
    trajectory, and a step with more observation times than the table gives NaN (never silently drops one)."""
    from dabench.dacycler import _var4d_weak_utils as wu
    traj, tlm, dx0, deta = _setup()
    T, D = deta.shape
    Hs = jnp.asarray(np.random.default_rng(1).normal(size=(5, 3, D)))
    owi = jnp.asarray([0, 3, 3, 7, 12])
    ref = jnp.stack([Hs[i] @ window_tlm_rollout_forced(tlm, traj, dx0, deta)[int(owi[i])] for i in range(5)])
    try:
        wu.set_obs_per_step_max(2)
        np.testing.assert_allclose(wu.window_tlm_obs_forced(tlm, traj, dx0, deta, Hs, owi), ref, rtol=1e-12, atol=1e-12)
        wu.set_obs_per_step_max(1)
        assert np.isnan(np.asarray(wu.window_tlm_obs_forced(tlm, traj, dx0, deta, Hs, owi))).all()
    finally:
        wu.set_obs_per_step_max(1)
