#!/usr/bin/env python
"""
Evaluation harness: trained PPO policy vs fixed-coverage baselines.

USAGE:
    # Baselines only (no trained model needed):
    python scripts/evaluate.py

    # Trained policy + baselines:
    python scripts/evaluate.py --model-path outputs/ppo_delivery

    # More eval seeds:
    python scripts/evaluate.py --model-path outputs/ppo_delivery --n-eval-seeds 10

BASELINES:
    Four fixed-coverage policies that set the same coverage_radius for every
    store on every step, regardless of observation:
      baseline_min  : radius = 0         (no coverage → maximum fail rate)
      baseline_mid  : radius = max_r / 2 (moderate coverage)
      baseline_opt  : radius = 700       (diagnosed optimum on balanced scenario;
                                          delivery_rate peaks ~82 % before warehouse_a
                                          cycle-time penalty kicks in above r≈700)
      baseline_max  : radius = max_r     (full coverage → maximum delivery cost /
                                          warehouse_a absorbs all orders → lower DR)

VERDICT FRAMING:
    SUCCESS  — trained policy's mean_output_radius lands near the optimum band
               AND its KPIs beat fixed r=1000 (proved interior optimum).
    PARTIAL  — beats r=0/r=500 but ties r=700/r=1000 (learned "high coverage"
               but not the interior optimum — likely w_cost too weak or undertrained).
    COLLAPSE — mean_output_radius ≈ 0 OR delivery_rate ≈ baseline_min
               (exploration still failed or scenario is wrong — check ent_coef and
               the learning-curve CSV shape).

EVAL SEEDS:
    Held-out seeds [1000, 1001, ..., 1000 + n_eval_seeds - 1], disjoint from
    the default training seeds [0, ...).  Each policy runs on the same N seeds.

FAILED_RATE DEFINITION (matches train.py exactly):
    failed_rate_resolved = failed / (delivered + failed), resolved orders only.
    total_orders includes in-flight at horizon; pending_at_horizon reports those.
    Both train.py's callback and this script use this identical formula.

METRICS TABLE:
    Each policy row shows mean ± std over eval seeds for:
      episode_return        sum of per-step cost_aware rewards over the episode
      delivery_rate_total   delivered / total_created (includes in-flight)
      failed_rate_resolved  failed / (delivered + failed), resolved only
      pending_at_horizon    in-flight orders at truncation (per episode)
      total_delivery_cost   sum of courier-distance costs for delivered orders
      courier_utilization   fraction of total courier-time spent delivering
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from gymnasium.wrappers import RescaleAction

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
# Type alias
# ---------------------------------------------------------------------------

PolicyFn = Callable[[np.ndarray], np.ndarray]


# ---------------------------------------------------------------------------
# Scenario config (must match train.py's build_training_config)
# ---------------------------------------------------------------------------

def load_scenario_for_eval(
    scenario_path: Path, seed: int, obs_preset: str = "operational"
) -> ScenarioConfig:
    """Load a YAML scenario and patch reward=cost_aware + obs_preset for evaluation.

    Mirrors load_scenario_for_training in train.py so train/eval configs match.
    """
    cfg = load_scenario(scenario_path)
    return cfg.model_copy(update={
        "seed": seed,
        "reward": RewardConfig(function_type="cost_aware"),
        "observation_preset": obs_preset,
    })


def build_eval_config(seed: int, obs_preset: str = "operational") -> ScenarioConfig:
    """Same two-store scenario as train.py; seed is replaced per eval episode."""
    return ScenarioConfig(
        name="eval",
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
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(config: ScenarioConfig, eval_seed: int, policy_fn: PolicyFn) -> dict[str, float]:
    """Run one episode with policy_fn; return metrics dict.

    The env is wrapped with RescaleAction(-1, 1) to match the training harness.
    policy_fn must output actions in [-1, 1] (the wrapped action space).
    """
    raw_env: Any = DeliveryEnv(config)
    # Must mirror the RescaleAction wrapper used in train.py's _make_env().
    env: Any = RescaleAction(raw_env, min_action=np.float32(-1.0), max_action=np.float32(1.0))
    obs, _ = env.reset(seed=eval_seed)
    episode_return = 0.0
    final_kpi: dict[str, Any] = {}
    terminated = False
    truncated = False
    # Track mean-over-stores action at each decision step so we can report
    # WHERE the policy actually set coverage_radius (crucial for diagnosing
    # whether the trained agent found the interior optimum near r≈700).
    # max_r from config for de-normalising [-1,1] actions back to radius units.
    max_r = config.max_coverage_radius
    step_mean_radii: list[float] = []
    while not (terminated or truncated):
        action = policy_fn(obs)
        # Convert normalised [-1,1] action to radius in [0, max_r] for reporting.
        step_mean_radii.append(float(np.mean((action + 1.0) / 2.0 * max_r)))
        obs, reward, terminated, truncated, info = env.step(action)
        episode_return += float(reward)
        if terminated or truncated:
            final_kpi = dict(info.get("kpi") or {})
    env.close()

    delivered = int(final_kpi.get("delivered_orders", 0))
    failed = int(final_kpi.get("failed_orders", 0))
    total = int(final_kpi.get("total_orders", 0))
    resolved = delivered + failed
    failed_rate = failed / resolved if resolved > 0 else float("nan")
    mean_radius = float(np.mean(step_mean_radii)) if step_mean_radii else float("nan")

    return {
        "episode_return": episode_return,
        "delivery_rate_total": delivered / total if total > 0 else float("nan"),
        "failed_rate_resolved": failed_rate,
        "pending_at_horizon": float(total - delivered - failed),
        "total_delivery_cost": float(final_kpi.get("total_delivery_cost", float("nan"))),
        "courier_utilization": float(final_kpi.get("courier_utilization", float("nan"))),
        "mean_output_radius": mean_radius,
    }


# ---------------------------------------------------------------------------
# Policy factories
# ---------------------------------------------------------------------------

def make_constant_policy(radius: float, n_stores: int, max_r: float = 1000.0) -> PolicyFn:
    """Fixed-radius baseline: always returns *radius* in the wrapped [-1, 1] action space.

    The env is wrapped with RescaleAction(-1, 1) to match the training harness, so
    actions must be in [-1, 1] here.  *radius* is given in native [0, max_r] units;
    this function normalises it to [-1, 1] before building the constant action.
    """
    normalized = float(2.0 * radius / max_r - 1.0)
    action = np.full((n_stores,), fill_value=normalized, dtype=np.float32)

    def policy(obs: np.ndarray) -> np.ndarray:  # noqa: ARG001 — obs intentionally unused
        return action

    return policy


def make_ppo_policy(model: Any) -> PolicyFn:
    """Deterministic PPO policy wrapper."""

    def policy(obs: np.ndarray) -> np.ndarray:
        action, _ = model.predict(obs, deterministic=True)
        return np.asarray(action, dtype=np.float32)

    return policy


# ---------------------------------------------------------------------------
# Multi-seed evaluation
# ---------------------------------------------------------------------------

_METRIC_KEYS = [
    "episode_return",
    "delivery_rate_total",
    "failed_rate_resolved",
    "pending_at_horizon",
    "total_delivery_cost",
    "courier_utilization",
    "mean_output_radius",
]


def evaluate_policy(
    name: str,
    policy_fn: PolicyFn,
    config_template: ScenarioConfig,
    eval_seeds: list[int],
) -> dict[str, Any]:
    """Run policy on each seed; return name + per-metric mean/std arrays."""
    results: list[dict[str, float]] = []
    for seed in eval_seeds:
        results.append(run_episode(config_template, seed, policy_fn))

    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    for key in _METRIC_KEYS:
        vals = np.array([r[key] for r in results], dtype=float)
        means[key] = float(np.nanmean(vals))
        stds[key] = float(np.nanstd(vals))

    return {"name": name, "means": means, "stds": stds, "n": len(results)}


# ---------------------------------------------------------------------------
# Table printer
# ---------------------------------------------------------------------------

def _fmt(mean: float, std: float, width: int = 14) -> str:
    if np.isnan(mean):
        return f"{'nan':>{width}}"
    return f"{mean:7.3f} +/-{std:5.3f}"


def print_results_table(policy_results: list[dict[str, Any]]) -> None:
    """Print a formatted comparison table to stdout."""
    col_w = 16
    headers = [
        "Policy", "Return", "DR/total", "FR/resolved", "Pending", "Cost", "Util", "MeanRadius",
    ]
    widths = [22, col_w, col_w, col_w, col_w, col_w, col_w, col_w]

    header_row = "  ".join(h.ljust(w) for h, w in zip(headers, widths))
    sep = "  ".join("-" * w for w in widths)
    print()
    print("=" * len(sep))
    print("Policy vs Baselines -- mean +/- std over held-out seeds")
    print("Metrics: Return=episode return | DR/total=delivery_rate_of_total |")
    print("  FR/resolved=failed_rate_resolved (failed/(delivered+failed)) |")
    print("  Pending=in-flight orders at horizon | Cost=total_delivery_cost |")
    print("  Util=courier_utilization | MeanRadius=mean coverage_radius output")
    print("  KNOWN OPTIMUM: fixed r~700 (delivery_rate peaks ~82% on balanced scenario)")
    print("=" * len(sep))
    print(header_row)
    print(sep)

    for pr in policy_results:
        m = pr["means"]
        s = pr["stds"]
        name_str = pr["name"][:22].ljust(22)
        cols = [
            name_str,
            _fmt(m["episode_return"], s["episode_return"]),
            _fmt(m["delivery_rate_total"], s["delivery_rate_total"]),
            _fmt(m["failed_rate_resolved"], s["failed_rate_resolved"]),
            _fmt(m["pending_at_horizon"], s["pending_at_horizon"]),
            _fmt(m["total_delivery_cost"], s["total_delivery_cost"]),
            _fmt(m["courier_utilization"], s["courier_utilization"]),
            _fmt(m["mean_output_radius"], s["mean_output_radius"]),
        ]
        print("  ".join(c.ljust(w) for c, w in zip(cols, widths)))

    print(sep)
    _print_verdict(policy_results)
    print()


def _print_verdict(policy_results: list[dict[str, Any]]) -> None:
    """Classify training outcome as SUCCESS / PARTIAL / COLLAPSE and print details."""
    trained = next((p for p in policy_results if p["name"] == "ppo_trained"), None)
    if trained is None:
        return

    def _by_name(name: str) -> dict[str, Any] | None:
        return next((p for p in policy_results if p["name"] == name), None)

    b_min = _by_name("baseline_min_0")
    b_opt = next(
        (p for p in policy_results if p["name"].startswith("baseline_opt_")), None
    )
    b_max = next(
        (p for p in policy_results if p["name"].startswith("baseline_max_")), None
    )

    t_dr = trained["means"]["delivery_rate_total"]
    t_radius = trained["means"]["mean_output_radius"]
    min_dr = b_min["means"]["delivery_rate_total"] if b_min else float("nan")
    opt_dr = b_opt["means"]["delivery_rate_total"] if b_opt else float("nan")
    max_dr = b_max["means"]["delivery_rate_total"] if b_max else float("nan")
    t_ret = trained["means"]["episode_return"]
    max_ret = b_max["means"]["episode_return"] if b_max else float("nan")

    print()
    print("=" * 60)
    # Classify: COLLAPSE → radius ≈ 0 or DR ≈ baseline_min
    is_collapse = (
        (not np.isnan(t_radius) and t_radius < 50.0)
        or (not np.isnan(t_dr) and not np.isnan(min_dr) and abs(t_dr - min_dr) < 0.05)
    )
    # SUCCESS: beats r=1000 on return AND mean radius in optimum band [500,900]
    in_opt_band = not np.isnan(t_radius) and 500.0 <= t_radius <= 900.0
    beats_max_return = (
        not np.isnan(t_ret) and not np.isnan(max_ret) and t_ret > max_ret
    )
    is_success = in_opt_band and beats_max_return

    if is_collapse:
        verdict = "COLLAPSE"
        detail = (
            f"mean_output_radius={t_radius:.1f} ~ 0  OR  delivery_rate~baseline_min.\n"
            "  Agent learned nothing.  Likely causes: ent_coef=0 (zero exploration) or\n"
            "  scenario still starved.  Inspect learning_curve.csv: if return is flat\n"
            "  from rollout 1, exploration is the issue; if it rises then crashes, check\n"
            "  reward scale and n_steps."
        )
    elif is_success:
        verdict = "SUCCESS"
        detail = (
            f"mean_output_radius={t_radius:.1f} in [500,900] (optimum band)  AND\n"
            f"  episode_return ({t_ret:.3f}) > baseline_max ({max_ret:.3f}).\n"
            "  Agent found the interior optimum: high delivery rate at lower cost\n"
            "  than naive full-coverage.  If radius ~700, it nailed the diagnosed peak."
        )
    else:
        verdict = "PARTIAL"
        detail = (
            f"mean_output_radius={t_radius:.1f}  DR={t_dr:.3f}  "
            f"(opt_baseline DR={opt_dr:.3f}  max_baseline DR={max_dr:.3f}).\n"
            "  Agent learned some coverage (beats baseline_min) but did not beat\n"
            "  baseline_max on return OR did not land near the optimum band.\n"
            "  Next steps: more timesteps (try 5M); or reduce w_cost if radius collapses\n"
            "  to max; or increase ent_coef if radius stays mid-range."
        )

    print(f"VERDICT: {verdict}")
    print(detail)
    print()

    # Per-metric delta table (trained vs each baseline)
    baselines = [p for p in policy_results if p["name"] != "ppo_trained"]
    print("Per-metric delta (trained vs baselines, ^=trained higher, note direction of 'good'):")
    for metric in ("episode_return", "delivery_rate_total", "failed_rate_resolved",
                   "pending_at_horizon", "total_delivery_cost", "courier_utilization"):
        t_mean = trained["means"][metric]
        if np.isnan(t_mean):
            continue
        parts = []
        for b in baselines:
            b_mean = b["means"][metric]
            if np.isnan(b_mean):
                continue
            diff = t_mean - b_mean
            sign = "^" if diff > 0 else "v"
            parts.append(f"vs {b['name']}: {sign}{abs(diff):.3f}")
        if parts:
            print(f"  {metric:30s}  {' | '.join(parts)}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

_OPT_RADIUS = 700.0  # diagnosed delivery-rate peak on balanced scenario


def main(
    model_path: Path | None,
    n_eval_seeds: int,
    obs_preset: str,
    scenario_path: Path | None = None,
) -> None:
    eval_seeds = list(range(1000, 1000 + n_eval_seeds))
    if scenario_path is not None:
        config_template = load_scenario_for_eval(scenario_path, seed=1000, obs_preset=obs_preset)
    else:
        config_template = build_eval_config(seed=1000, obs_preset=obs_preset)
    max_r = config_template.max_coverage_radius
    n_stores = len(config_template.stores)

    baselines: list[tuple[str, PolicyFn]] = [
        ("baseline_min_0", make_constant_policy(0.0, n_stores, max_r)),
        (f"baseline_mid_{max_r / 2:.0f}", make_constant_policy(max_r / 2.0, n_stores, max_r)),
        (f"baseline_opt_{_OPT_RADIUS:.0f}", make_constant_policy(_OPT_RADIUS, n_stores, max_r)),
        (f"baseline_max_{max_r:.0f}", make_constant_policy(max_r, n_stores, max_r)),
    ]

    policies: list[tuple[str, PolicyFn]] = []

    if model_path is not None:
        model_file = (
            Path(str(model_path) + ".zip") if not str(model_path).endswith(".zip") else model_path
        )
        if not model_file.exists():
            print(f"Model file not found: {model_file}", file=sys.stderr)
            sys.exit(1)
        try:
            from stable_baselines3 import PPO as _PPO
        except ImportError:
            print("stable-baselines3 not installed; cannot load trained model.", file=sys.stderr)
            sys.exit(1)
        model = _PPO.load(str(model_path))
        policies.append(("ppo_trained", make_ppo_policy(model)))
        print(f"Loaded model from {model_file}")

    policies.extend(baselines)

    print(f"\nEvaluating {len(policies)} policies on seeds {eval_seeds[0]}-{eval_seeds[-1]} ...")
    all_results: list[dict[str, Any]] = []
    for name, policy_fn in policies:
        print(f"  {name} ...", end=" ", flush=True)
        result = evaluate_policy(name, policy_fn, config_template, eval_seeds)
        all_results.append(result)
        print(f"done  (mean return {result['means']['episode_return']:.2f})")

    print_results_table(all_results)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--scenario", type=Path, default=None, dest="scenario_path",
        help="Path to YAML scenario file (e.g. scenarios/balanced.yaml). "
             "Must match the scenario used for training.",
    )
    p.add_argument(
        "--model-path", type=Path, default=None, dest="model_path",
        help="Path to trained model (without .zip); omit to run baselines only",
    )
    p.add_argument(
        "--n-eval-seeds", type=int, default=5, dest="n_eval_seeds",
        help="Number of held-out seeds to evaluate on (default 5)",
    )
    p.add_argument(
        "--obs-preset", type=str, default="operational", dest="obs_preset",
        help="ObservationSpec preset (must match training preset)",
    )
    return p.parse_args()


if __name__ == "__main__":
    _args = _parse_args()
    main(
        model_path=_args.model_path,
        n_eval_seeds=_args.n_eval_seeds,
        obs_preset=_args.obs_preset,
        scenario_path=_args.scenario_path,
    )
