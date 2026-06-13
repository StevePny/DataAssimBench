"""Unit tests for ``scripts/bench_4dvar_v9_picks.py`` helpers.

The pure-Python helpers (pick discovery, ckpt discovery, fingerprint
extraction, summary writers) are tested without loading V9; the V9-
dependent probe is gated on the Run-16 checkpoint being present.
"""
from __future__ import annotations

import json
import os
import sys

import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "scripts"))

CKPT_DIR = os.path.join(REPO, "runs", "v9-run16")
SLICE = os.path.join(
        REPO, "data", "slices",
        "talagrand_bve_t21_cache_jet_attractor_LOW_K50_fullstep_tail1000.zarr")

_skip_no_ckpt = pytest.mark.skipif(
        not os.path.isdir(CKPT_DIR),
        reason=f"V9 checkpoint missing: {CKPT_DIR}")
_skip_no_slice = pytest.mark.skipif(
        not os.path.isdir(SLICE),
        reason=f"BVE-LOW slice missing: {SLICE}")


@pytest.fixture
def bench_module():
    """Import the bench script as a module; skip if dabench missing."""
    try:
        import bench_4dvar_v9_picks as bm
    except ImportError as exc:
        pytest.skip(f"bench module unavailable: {exc}")
    return bm


# ----- pick / checkpoint discovery -------------------------------

def test_discover_pick_dirs_unique_match(tmp_path, bench_module):
    """Each label maps to its single matching subdir."""
    (tmp_path / "A_run123").mkdir()
    (tmp_path / "B_run456").mkdir()
    out = bench_module._discover_pick_dirs(str(tmp_path), ["A", "B"])
    assert out["A"].endswith("A_run123")
    assert out["B"].endswith("B_run456")


def test_discover_pick_dirs_no_match_raises(tmp_path, bench_module):
    """A pick label with no matching subdir raises FileNotFoundError."""
    (tmp_path / "A_run123").mkdir()
    with pytest.raises(FileNotFoundError):
        bench_module._discover_pick_dirs(str(tmp_path), ["Z"])


def test_discover_pick_dirs_ambiguous_raises(tmp_path, bench_module):
    """Two subdirs sharing a label prefix raise RuntimeError."""
    (tmp_path / "A_run1").mkdir()
    (tmp_path / "A_run2").mkdir()
    with pytest.raises(RuntimeError, match="Ambiguous"):
        bench_module._discover_pick_dirs(str(tmp_path), ["A"])


def test_discover_ckpt_file_unique(tmp_path, bench_module):
    """One ``v9_ckpt_*.eqx`` ⇒ returns its basename."""
    (tmp_path / "v9_ckpt_P4_epoch42.eqx").write_bytes(b"\x00")
    assert (bench_module._discover_ckpt_file(str(tmp_path))
            == "v9_ckpt_P4_epoch42.eqx")


def test_discover_ckpt_file_zero_raises(tmp_path, bench_module):
    """No ``.eqx`` in the pick dir ⇒ RuntimeError."""
    with pytest.raises(RuntimeError, match="Expected exactly one"):
        bench_module._discover_ckpt_file(str(tmp_path))


def test_discover_ckpt_file_multiple_raises(tmp_path, bench_module):
    """Multiple ``.eqx`` in the pick dir ⇒ RuntimeError."""
    (tmp_path / "v9_ckpt_P4_epoch20.eqx").write_bytes(b"\x00")
    (tmp_path / "v9_ckpt_P4_epoch60.eqx").write_bytes(b"\x00")
    with pytest.raises(RuntimeError, match="Expected exactly one"):
        bench_module._discover_ckpt_file(str(tmp_path))


# ----- fingerprint -----------------------------------------------

def test_pick_fingerprint_extracts_sobolev_keys(bench_module):
    """The fingerprint is exactly the six distinguishing keys."""
    hp = {
        "sobolev_balance_mode": "pcgrad",
        "lambda_sobolev_max": 0.005,
        "sobolev_anneal_floor": 0.0,
        "sobolev_balance_c": 0.0,
        "phase": "P4",
        "last_saved_epoch": 60,
        "unrelated_key": 17,
    }
    fp = bench_module._pick_fingerprint(hp)
    assert set(fp) == {
        "sobolev_balance_mode", "lambda_sobolev_max",
        "sobolev_anneal_floor", "sobolev_balance_c",
        "phase", "last_saved_epoch"}
    assert fp["sobolev_balance_mode"] == "pcgrad"
    assert fp["last_saved_epoch"] == 60


# ----- summary writer --------------------------------------------

def _synth_record(pick: str, mode_skill: dict) -> dict:
    """Minimal record for ``_write_summary`` to consume."""
    osse = {
        m: {"sigma_clim": 1.0, "sigma_bg": 0.1, "sigma_obs": 0.01,
            "rmse_bg": 0.5, "rmse_freerun": 0.1,
            "rmse_ana": 0.1 * (1.0 - s),
            "cycle_seconds": 1.0, "n_match": 4,
            "skill_vs_freerun": s}
        for m, s in mode_skill.items()
    }
    return {
        "pick": pick, "pick_dir": f"/tmp/{pick}",
        "ckpt_file": f"v9_ckpt_P4_epoch{42 if pick == 'A' else 20}.eqx",
        "aug_norm_path": "/tmp/aug.npy",
        "fingerprint": {"sobolev_balance_mode": "none",
                        "lambda_sobolev_max": 0.25,
                        "sobolev_anneal_floor": 0.005,
                        "sobolev_balance_c": 0.0,
                        "phase": "P4", "last_saved_epoch": 60},
        "gate": {"passed": True, "a1_superposition": 1e-15,
                 "a3_zero_bias": 0.0, "a5_adjoint_relerr": 1e-15},
        "tlm_probe": {"taylor_ratio": 2.5, "stream2_vs_jvp_cos": 0.3,
                      "valid_tlm": False,
                      "taylor_nl_norm": 1e-3, "taylor_lin_norm": 4e-4,
                      "stream2_norm": 0.4, "jvp_stream1_norm": 1.0,
                      "stream2_vs_jvp_relerr": 0.95},
        "osse": osse,
        "protocol": {"n_cycles": 1, "steps_per_window": 2,
                     "n_outer": 1, "n_inner": 5, "n_obs_loc": 16,
                     "sigma_bg_frac": 0.1, "sigma_obs_frac": 0.01,
                     "lm_lambda": 0.0, "B_rank": 10, "B_probes": 24,
                     "tlm_modes": list(mode_skill), "identity_B": False,
                     "seed": 271},
    }



def test_write_summary_single_mode(tmp_path, bench_module):
    """Single TLM mode ⇒ one CSV row per pick, MD has one skill column."""
    records = [_synth_record("A", {"stream2": 0.3})]
    bench_module._write_summary(records, str(tmp_path))
    csv_text = (tmp_path / "summary.csv").read_text()
    assert csv_text.count("\n") == 2  # header + 1 row
    assert ",stream2," in csv_text
    assert ",autodiff," not in csv_text
    md_text = (tmp_path / "summary.md").read_text()
    assert "+30.0%" in md_text
    assert "skill_S2" in md_text and "skill_jvp" in md_text


def test_write_summary_both_modes(tmp_path, bench_module):
    """Both modes ⇒ two CSV rows per pick, both skill columns rendered."""
    records = [
        _synth_record("A", {"stream2": 0.3, "autodiff": -0.5}),
        _synth_record("B", {"stream2": 0.1, "autodiff": 0.2}),
    ]
    bench_module._write_summary(records, str(tmp_path))
    csv_text = (tmp_path / "summary.csv").read_text()
    # header + 2 modes * 2 picks
    assert csv_text.count("\n") == 5
    assert csv_text.count(",stream2,") == 2
    assert csv_text.count(",autodiff,") == 2
    md_text = (tmp_path / "summary.md").read_text()
    # Per-pick row in MD has both skill columns populated.
    assert "+30.0%" in md_text and "-50.0%" in md_text
    assert "+10.0%" in md_text and "+20.0%" in md_text


def test_write_summary_emits_provenance_protocol(tmp_path, bench_module):
    """The MD header echoes the protocol knobs from the first record."""
    records = [_synth_record("A", {"stream2": 0.0})]
    bench_module._write_summary(records, str(tmp_path))
    md_text = (tmp_path / "summary.md").read_text()
    assert "n_cycles=1" in md_text
    assert "sigma_bg_frac=0.1" in md_text
    assert "B_rank=10" in md_text


# ----- TLM-consistency probe (V9-dependent) ----------------------

@_skip_no_ckpt
@_skip_no_slice
def test_probe_tlm_consistency_returns_expected_keys(bench_module):
    """``_probe_tlm_consistency`` returns all advertised keys, all finite.

    Loads the Run-16 checkpoint (the only one guaranteed to ship the
    fitted ``v9_aug_norm.npy``) and runs the probe at the BVE-LOW
    truth IC, then asserts the dict shape + that every numeric field
    is a finite Python float.
    """
    import jax.numpy as jnp
    import numpy as np
    import zarr
    from mltlm.dinosaur._v9_load import load_v9_from_run_dir
    v9 = load_v9_from_run_dir(CKPT_DIR)
    x = jnp.asarray(np.asarray(zarr.open(SLICE, mode="r")["x"][0]).ravel())
    out = bench_module._probe_tlm_consistency(v9, x, seed=0)
    expected = {"taylor_ratio", "taylor_nl_norm", "taylor_lin_norm",
                "stream2_norm", "jvp_stream1_norm",
                "stream2_vs_jvp_relerr", "stream2_vs_jvp_cos",
                "valid_tlm"}
    assert set(out) == expected
    for k in expected - {"valid_tlm"}:
        v = out[k]
        assert isinstance(v, float)
        assert np.isfinite(v), f"{k} not finite: {v}"
    assert isinstance(out["valid_tlm"], bool)
    # Magnitudes must be strictly positive (zero would mean the
    # probe collapsed to a degenerate δx or driver = identity).
    assert out["stream2_norm"] > 0.0
    assert out["jvp_stream1_norm"] > 0.0
    assert out["taylor_lin_norm"] > 0.0


def test_pick_fingerprint_round_trips_via_json(tmp_path, bench_module):
    """Fingerprint json-serialises and round-trips losslessly.

    Pins the per-pick JSON contract the bench driver writes -- the
    fingerprint is the canonical identifier on the verdict table.
    """
    hp = {
        "sobolev_balance_mode": "static_anchor",
        "lambda_sobolev_max": 0.25,
        "sobolev_anneal_floor": 0.0,
        "sobolev_balance_c": 0.1,
        "phase": "P4",
        "last_saved_epoch": 60,
    }
    fp = bench_module._pick_fingerprint(hp)
    p = tmp_path / "fp.json"
    p.write_text(json.dumps(fp))
    rt = json.loads(p.read_text())
    assert rt == fp


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])