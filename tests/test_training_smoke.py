"""
Training harness smoke tests.

SCOPE — plumbing only, not learning quality:
  1. check_env: SB3's gym-compliance checker passes on DeliveryEnv with zero
     errors.  This is the formal proof that the env is API-compliant.
  2. tiny_ppo_run: a minimal PPO run (few hundred timesteps) completes without
     error and produces a model file + a non-empty learning-curve CSV.
     This proves the plumbing (env → Monitor → DummyVecEnv → PPO → callback
     → CSV writer) works end-to-end.

These tests deliberately do NOT assert:
  - Any reward threshold or learning curve slope.
  - Convergence or policy quality (too few steps; that is out of scope here).
  - Byte-identical CSV across runs (torch nondeterminism without the full
    determinism stack; harness correctness does not require it).

Skip behaviour:
  The entire module is skipped cleanly when stable_baselines3 is not installed
  (via pytest.importorskip at module level).  The 308 core tests are unaffected
  and gain no torch dependency.
"""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import pytest

# Skip entire module if SB3 is not installed.  This is the only guard needed:
# all subsequent imports of SB3 symbols in this file are safe once this passes.
pytest.importorskip("stable_baselines3")

from stable_baselines3.common.env_checker import check_env  # noqa: E402

import delivery_sim  # noqa: F401, E402 — triggers all @register decorators
from delivery_sim.config.schema import (  # noqa: E402
    CourierConfig,
    DemandConfig,
    RewardConfig,
    ScenarioConfig,
    StoreConfig,
    WorldConfig,
)
from delivery_sim.envs.single_agent import DeliveryEnv  # noqa: E402

_PROJECT_ROOT = Path(__file__).parent.parent
_SCRIPTS_DIR = _PROJECT_ROOT / "scripts"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_smoke_config(seed: int = 0) -> ScenarioConfig:
    """Minimal two-store config with cost_aware reward for smoke tests."""
    return ScenarioConfig(
        name="smoke",
        seed=seed,
        dt=1.0,
        max_steps=1000,
        world=WorldConfig(width=1000.0, height=1000.0),
        stores=[
            StoreConfig(name="warehouse_a", x=200.0, y=200.0, capacity=20),
            StoreConfig(name="warehouse_b", x=800.0, y=600.0, capacity=15),
        ],
        couriers=[
            CourierConfig(courier_type="BikeCourier", count=3, speed=1.5, cost_per_unit=0.01),
        ],
        demand=DemandConfig(generator_type="PoissonDemandGenerator", rate=0.5),
        reward=RewardConfig(function_type="cost_aware"),
        decision_interval=100.0,
        max_coverage_radius=1000.0,
        observation_preset="operational",
    )


# ---------------------------------------------------------------------------
# Test 1: gym API compliance
# ---------------------------------------------------------------------------

def test_check_env_passes() -> None:
    """SB3's check_env must pass on DeliveryEnv with zero errors.

    check_env raises AssertionError (or warns) for any gym-compliance gap.
    This is the formal proof that standard wrappers plug in with zero custom
    glue.  If this test fails, that is a FLAG about the env's gym-compliance,
    not a test to patch around.
    """
    env = DeliveryEnv(_make_smoke_config())
    # warn=True surfaces deprecation warnings as pytest output; errors raise.
    check_env(env, warn=True)
    env.close()


# ---------------------------------------------------------------------------
# Test 2: tiny PPO run — plumbing proof, NOT a learning test
# ---------------------------------------------------------------------------

def test_tiny_ppo_run(tmp_path: Path) -> None:
    """A 256-step PPO run completes and writes model + non-empty CSV.

    Hyperparameters chosen to be fast, not optimal:
      --total-timesteps 256  (≈25 episodes at 10 steps/episode)
      --n-steps 64           (4 rollouts → 4 CSV rows)
      --batch-size 32
      --n-epochs 1

    Assertions (all plumbing, none learning):
      1. train.py exits with code 0.
      2. ppo_delivery.zip exists and is non-empty (model was saved).
      3. learning_curve.csv exists and has ≥ 1 data row (callback wrote).
      4. CSV header matches expected columns (no silent schema drift).

    This test does NOT check reward values, learning curves, or KPI thresholds.
    """
    result = subprocess.run(
        [
            sys.executable,
            str(_SCRIPTS_DIR / "train.py"),
            "--total-timesteps", "256",
            "--n-steps", "64",
            "--batch-size", "32",
            "--n-epochs", "1",
            "--seed", "0",
            "--output-dir", str(tmp_path),
        ],
        capture_output=True,
        text=True,
        cwd=str(_PROJECT_ROOT),
    )
    assert result.returncode == 0, (
        f"train.py exited with code {result.returncode}\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )

    model_file = tmp_path / "ppo_delivery.zip"
    assert model_file.exists(), f"Model file not found: {model_file}"
    assert model_file.stat().st_size > 0, "Model file is empty"

    csv_file = tmp_path / "learning_curve.csv"
    assert csv_file.exists(), f"Learning curve CSV not found: {csv_file}"

    with csv_file.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Header schema check — catches silent renames / column drops.
    expected_cols = {
        "rollout", "timestep", "n_episodes", "mean_episode_return",
        "delivery_rate_of_total", "failed_rate_resolved", "pending_at_horizon",
        "mean_total_delivery_cost", "courier_utilization",
    }
    assert reader.fieldnames is not None, "CSV has no header"
    assert expected_cols.issubset(set(reader.fieldnames)), (
        f"Missing CSV columns: {expected_cols - set(reader.fieldnames)}"
    )

    # At least one data row — confirms the callback fired.
    assert len(rows) >= 1, (
        "learning_curve.csv has no data rows; KPICallback._on_rollout_end never fired"
    )
