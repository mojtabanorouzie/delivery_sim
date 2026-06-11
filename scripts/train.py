#!/usr/bin/env python
"""
PPO training harness for DeliveryEnv coverage-radius control (cost_aware reward).

USAGE:
    python scripts/train.py [--scenario PATH] [--total-timesteps N] [--seed S] ...

BALANCED SCENARIO (recommended — scenarios/balanced.yaml):
    Two stores (warehouse_a/b), 20 BikeCouriers, Poisson demand rate=0.025,
    decision_interval=100 sim-s, horizon=4000 sim-s → 40 env steps per episode.
    Reward overridden to CostAwareReward (cost_aware) regardless of what the YAML
    specifies.  Obs preset: operational (richer per-store coverage signal).

KNOWN OPTIMUM (balanced scenario, cost_aware reward):
    Fixed r≈700 is the empirical peak: delivery_rate ~82 %, cost reasonable.
    At r=1000 warehouse_a (alphabetical-first) absorbs all orders, effective
    courier cycle rises, throughput falls to ~80 %.  So a converged agent should
    land near r≈700-ish, NOT at max, NOT at 0.

COMPUTE NOTE (balanced scenario, 1e6 timesteps):
    40 steps/episode × 25 episodes/rollout (n_steps=2048/~82 steps) = ~1 rollout/episode.
    1 000 000 steps ÷ 40 steps/episode ≈ 25 000 episodes ≈ 488 rollouts → 488 CSV rows.
    Wall-clock estimate: ~8-20 min on a modern CPU (no GPU needed for MLP policy).

PPO HYPERPARAMETER DEFAULTS (researcher starting points — NOT tuned, NOT optimal):
    n_steps=2048   rollout buffer size per env per update
    batch_size=64  minibatch size
    n_epochs=10    gradient passes per rollout
    gamma=0.99     discount factor
    gae_lambda=0.95 GAE lambda
    clip_range=0.2 PPO clipping coefficient
    learning_rate=3e-4 Adam learning rate
    ent_coef=0.0   entropy bonus coefficient
    vf_coef=0.5    value-function loss coefficient
    max_grad_norm=0.5 gradient norm clip

LOGGING GRANULARITY (important — read before interpreting the CSV):
    The CSV is per-EPISODE, one row per rollout (averaged over completed episodes
    in that rollout).  "mean_episode_return" = mean of (sum of per-step cost_aware
    rewards over the episode).  KPI columns — delivery_rate_of_total,
    failed_rate_resolved, etc. — are episode-cumulative values from info["kpi"],
    resolved at truncation.  Both columns are at the same granularity (per
    episode), but they measure OUTCOME, not the per-step gradient signal.
    The cost_aware reward is computed per decision-interval; the episode-end cost
    aggregate only loosely explains why the return moved.  Do not read this CSV as
    step-level alignment between reward and KPIs.

FAILED_RATE DEFINITION:
    failed_rate_resolved = failed_orders / (delivered_orders + failed_orders)
    Denominator = RESOLVED orders only.  total_orders from KPICollector includes
    in-flight orders at horizon (never dispatched or mid-route at truncation).
    Those are reported separately as pending_at_horizon.

NOTE ON ACTION NORMALIZATION (CRITICAL — DO NOT REMOVE):
    SB3 PPO's MlpPolicy initialises the action_net with ortho gain=0.01, giving
    initial mean actions ~0 in native coordinates.  With std=1.0 in the native
    action space [0, 1000], ~99.9% of sampled actions land in [0, 3], which is
    effectively r=0 — the collapse observed in runs without normalisation.
    Fix: wrap the env with RescaleAction(-1, 1) → policy outputs in [-1, 1] →
    initial mean maps to radius=500, std=1 → uniform exploration of [0, 1000].
    This is a harness wrapper, NOT an env change (env is unchanged).
    evaluate.py MUST mirror this wrapper so the trained model's actions are
    interpreted correctly.

NOTE ON TRAINING SEED:
    A single seed produces a deterministic scenario; every episode is the same
    Poisson sequence.  The agent can still learn coverage dynamics within that
    scenario.  For curriculum / multi-scenario training, modify build_training_config
    to accept a seed range and wrap with an env that increments seeds on reset.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import IO, Any

import numpy as np
from gymnasium.wrappers import RescaleAction
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

import delivery_sim  # noqa: F401 — triggers all @register decorators
from delivery_sim.config.loader import load_scenario
from delivery_sim.config.schema import (
    CourierConfig,
    DemandConfig,
    RewardConfig,
    ScenarioConfig,
    StoreConfig,
    WorldConfig,
)
from delivery_sim.envs.single_agent import DeliveryEnv

# ---------------------------------------------------------------------------
# CSV column layout
# ---------------------------------------------------------------------------

_CSV_FIELDS = [
    "rollout",
    "timestep",
    "n_episodes",
    "mean_episode_return",
    "delivery_rate_of_total",   # delivered / total_created (includes in-flight)
    "failed_rate_resolved",     # failed / (delivered + failed), resolved only
    "pending_at_horizon",       # total_created - delivered - failed
    "mean_total_delivery_cost",
    "courier_utilization",
]


# ---------------------------------------------------------------------------
# Scenario config
# ---------------------------------------------------------------------------

def load_scenario_for_training(
    scenario_path: Path, seed: int, obs_preset: str = "operational"
) -> ScenarioConfig:
    """Load a YAML scenario and patch reward=cost_aware + obs_preset for training.

    The balanced.yaml ships with SparseDeliveryReward; we override to cost_aware
    here so the agent optimises the coverage–cost tradeoff rather than raw
    delivery count.  The YAML's other values (couriers, demand, horizon, world)
    are preserved unchanged.
    """
    cfg = load_scenario(scenario_path)
    return cfg.model_copy(update={
        "seed": seed,
        "reward": RewardConfig(function_type="cost_aware"),
        "observation_preset": obs_preset,
    })


def build_training_config(seed: int, obs_preset: str = "operational") -> ScenarioConfig:
    """Two-store delivery scenario with cost_aware reward.

    Identical layout to scenarios/example.yaml except:
      - reward set to cost_aware
      - observation_preset set to obs_preset (default "operational")
      - decision_interval=100, max_coverage_radius=1000 (explicit)
    """
    return ScenarioConfig(
        name="training",
        seed=seed,
        dt=1.0,
        max_steps=1000,
        world=WorldConfig(width=1000.0, height=1000.0),
        stores=[
            StoreConfig(name="warehouse_a", x=200.0, y=200.0, capacity=20),
            StoreConfig(name="warehouse_b", x=800.0, y=600.0, capacity=15),
        ],
        couriers=[
            CourierConfig(
                courier_type="BikeCourier",
                count=3,
                speed=1.5,
                cost_per_unit=0.01,
            ),
        ],
        demand=DemandConfig(generator_type="PoissonDemandGenerator", rate=0.5),
        reward=RewardConfig(function_type="cost_aware"),
        decision_interval=100.0,
        max_coverage_radius=1000.0,
        observation_preset=obs_preset,
    )


# ---------------------------------------------------------------------------
# KPI aggregation helper
# ---------------------------------------------------------------------------

def _agg_kpis(kpis: list[dict[str, Any]]) -> dict[str, float]:
    """Return mean floats over a list of episode KPI dicts."""
    if not kpis:
        nan = float("nan")
        return {
            "delivery_rate_of_total": nan,
            "failed_rate_resolved": nan,
            "pending_at_horizon": nan,
            "mean_total_delivery_cost": nan,
            "courier_utilization": nan,
        }

    delivered_arr = np.array([int(k.get("delivered_orders", 0)) for k in kpis], dtype=float)
    failed_arr = np.array([int(k.get("failed_orders", 0)) for k in kpis], dtype=float)
    total_arr = np.array([int(k.get("total_orders", 0)) for k in kpis], dtype=float)
    cost_arr = np.array([float(k.get("total_delivery_cost", float("nan"))) for k in kpis])
    util_arr = np.array([float(k.get("courier_utilization", float("nan"))) for k in kpis])

    resolved_arr = delivered_arr + failed_arr
    # Per-episode failed_rate_resolved; NaN when no orders resolved in that episode.
    fail_rates = np.where(resolved_arr > 0, failed_arr / resolved_arr, float("nan"))
    dr_of_total = np.where(total_arr > 0, delivered_arr / total_arr, float("nan"))

    return {
        "delivery_rate_of_total": float(np.nanmean(dr_of_total)),
        "failed_rate_resolved": float(np.nanmean(fail_rates)),
        "pending_at_horizon": float(np.nanmean(total_arr - delivered_arr - failed_arr)),
        "mean_total_delivery_cost": float(np.nanmean(cost_arr)),
        "courier_utilization": float(np.nanmean(util_arr)),
    }


# ---------------------------------------------------------------------------
# SB3 callback: per-rollout CSV logging
# ---------------------------------------------------------------------------

class KPICallback(BaseCallback):
    """Log per-episode return and KPIs to CSV; one row per rollout.

    Collects episode data from:
      info["episode"]  — added by Monitor wrapper at episode end
      info["kpi"]      — added by DeliveryEnv when truncated=True

    Both keys appear in the same info dict at episode end.  KPIs are
    episode-cumulative (see module docstring on logging granularity).

    This callback asserts NOTHING about learning quality — it is a
    recording instrument only.
    """

    def __init__(self, csv_path: Path, verbose: int = 0) -> None:
        super().__init__(verbose)
        self._csv_path = csv_path
        self._episode_returns: list[float] = []
        self._episode_kpis: list[dict[str, Any]] = []
        self._rollout_count = 0
        self._fh: IO[str] | None = None
        self._writer: Any = None  # csv.DictWriter

    def _on_training_start(self) -> None:
        self._fh = self._csv_path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._fh, fieldnames=_CSV_FIELDS)
        self._writer.writeheader()

    def _on_step(self) -> bool:
        infos: list[dict[str, Any]] = self.locals.get("infos") or []
        for info in infos:
            ep = info.get("episode")
            if ep is not None:
                self._episode_returns.append(float(ep["r"]))
            kpi = info.get("kpi")
            if kpi is not None:
                self._episode_kpis.append(dict(kpi))
        return True

    def _on_rollout_end(self) -> None:
        if not self._episode_returns:
            return
        self._rollout_count += 1
        kpi_agg = _agg_kpis(self._episode_kpis)
        row: dict[str, Any] = {
            "rollout": self._rollout_count,
            "timestep": int(self.num_timesteps),
            "n_episodes": len(self._episode_returns),
            "mean_episode_return": round(float(np.mean(self._episode_returns)), 4),
            **{k: round(v, 4) if not np.isnan(v) else "nan" for k, v in kpi_agg.items()},
        }
        assert self._writer is not None
        self._writer.writerow(row)
        if self._fh is not None:
            self._fh.flush()
        self._episode_returns.clear()
        self._episode_kpis.clear()

    def _on_training_end(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


# ---------------------------------------------------------------------------
# Learning-curve plot
# ---------------------------------------------------------------------------

def _plot_learning_curve(csv_path: Path, plot_path: Path) -> None:
    """Write a PNG learning-curve plot from the CSV artifact."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot.", file=sys.stderr)
        return

    timesteps: list[int] = []
    returns: list[float] = []
    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                timesteps.append(int(row["timestep"]))
                val = row["mean_episode_return"]
                returns.append(float(val) if val != "nan" else float("nan"))
            except (ValueError, KeyError):
                continue

    if not timesteps:
        print("No data in learning curve CSV; skipping plot.", file=sys.stderr)
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(timesteps, returns, linewidth=1.5, color="steelblue")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Mean Episode Return")
    ax.set_title("PPO Learning Curve — DeliveryEnv (cost_aware reward)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(plot_path), dpi=100)
    plt.close(fig)
    print(f"Plot saved -> {plot_path}")


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(
    *,
    total_timesteps: int = 10_000,
    seed: int = 0,
    obs_preset: str = "operational",
    output_dir: Path = Path("outputs"),
    scenario_path: Path | None = None,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    learning_rate: float = 3e-4,
    ent_coef: float = 0.0,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
) -> Path:
    """Train PPO; return path to saved model (without .zip extension)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "learning_curve.csv"
    plot_path = output_dir / "learning_curve.png"
    model_path = output_dir / "ppo_delivery"

    if scenario_path is not None:
        config = load_scenario_for_training(scenario_path, seed=seed, obs_preset=obs_preset)
    else:
        config = build_training_config(seed=seed, obs_preset=obs_preset)

    def _make_env() -> Any:
        # RescaleAction maps policy output [-1,1] → env action [0, max_r].
        # Without this, PPO's initial mean~0 + std=1 gives radius~0.4 → collapse.
        return Monitor(RescaleAction(
            DeliveryEnv(config), min_action=np.float32(-1.0), max_action=np.float32(1.0)
        ))

    vec_env = DummyVecEnv([_make_env])

    # PPO defaults below are documented starting points, not optimal values.
    model = PPO(
        "MlpPolicy",
        vec_env,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        learning_rate=learning_rate,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        seed=seed,
        verbose=1,
    )

    callback = KPICallback(csv_path=csv_path)
    model.learn(total_timesteps=total_timesteps, callback=callback)
    model.save(str(model_path))

    print(f"\nArtifacts written to {output_dir}/")
    print(f"  model  : {model_path}.zip")
    print(f"  csv    : {csv_path}")

    _plot_learning_curve(csv_path, plot_path)
    return model_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--scenario", type=Path, default=None, dest="scenario_path",
        help="Path to YAML scenario file (e.g. scenarios/balanced.yaml). "
             "When given, reward is overridden to cost_aware and obs_preset is applied "
             "on top of the YAML. Omit to use the built-in legacy config.",
    )
    p.add_argument("--total-timesteps", type=int, default=10_000, dest="total_timesteps",
                   help="Total env steps (default 10 000; use 1 000 000+ for balanced scenario)")
    p.add_argument("--seed", type=int, default=0, help="RNG seed for env + PPO")
    p.add_argument("--obs-preset", type=str, default="operational", dest="obs_preset",
                   help="ObservationSpec preset name (default 'operational')")
    p.add_argument("--output-dir", type=Path, default=Path("outputs"), dest="output_dir",
                   help="Directory for model, CSV, and plot")
    # PPO hyperparameters (all defaults are SB3 standard / documented above)
    p.add_argument("--n-steps", type=int, default=2048, dest="n_steps")
    p.add_argument("--batch-size", type=int, default=64, dest="batch_size")
    p.add_argument("--n-epochs", type=int, default=10, dest="n_epochs")
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95, dest="gae_lambda")
    p.add_argument("--clip-range", type=float, default=0.2, dest="clip_range")
    p.add_argument("--learning-rate", type=float, default=3e-4, dest="learning_rate")
    p.add_argument("--ent-coef", type=float, default=0.01, dest="ent_coef",
                   help="Entropy bonus coefficient (default 0.01; enables exploration of "
                        "continuous coverage action; prior collapse at 0.0)")
    p.add_argument("--vf-coef", type=float, default=0.5, dest="vf_coef")
    p.add_argument("--max-grad-norm", type=float, default=0.5, dest="max_grad_norm")
    return p.parse_args()


if __name__ == "__main__":
    _args = _parse_args()
    train(
        total_timesteps=_args.total_timesteps,
        seed=_args.seed,
        obs_preset=_args.obs_preset,
        output_dir=_args.output_dir,
        scenario_path=_args.scenario_path,
        n_steps=_args.n_steps,
        batch_size=_args.batch_size,
        n_epochs=_args.n_epochs,
        gamma=_args.gamma,
        gae_lambda=_args.gae_lambda,
        clip_range=_args.clip_range,
        learning_rate=_args.learning_rate,
        ent_coef=_args.ent_coef,
        vf_coef=_args.vf_coef,
        max_grad_norm=_args.max_grad_norm,
    )
