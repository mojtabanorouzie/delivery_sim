#!/usr/bin/env python
"""
Scenario leverage diagnostic: sweep fixed coverage_radius values, aggregate KPIs
over N seeds, and report the capacity/leverage curve.

Torch-free; reuses DeliveryEnv + KPICollector from the delivery_sim package.

USAGE
-----
    # Confirm example.yaml is starved:
    python scripts/diagnose_scenario.py

    # Verify balanced scenario leverage curve:
    python scripts/diagnose_scenario.py --config scenarios/balanced.yaml

    # Custom sweep:
    python scripts/diagnose_scenario.py --config scenarios/balanced.yaml \\
        --radii 0 100 300 700 1000 --seeds 42 43 44

    # Save a leverage-curve plot (requires matplotlib):
    python scripts/diagnose_scenario.py --config scenarios/balanced.yaml --plot

OUTPUT
------
One table row per radius showing mean ± std over seeds for:
  DR/total     delivery_rate_total  = delivered / total_created
  FR/resolvd   failed_rate_resolved = failed / (delivered + failed)
  Pending      in-flight orders at horizon (not delivered, not failed)
  Util         courier_utilization  = busy_courier_time / (n_couriers × horizon)
  Cost         total_delivery_cost  (sum of courier-distance costs for deliveries)
  Orders       total orders created per episode
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

import delivery_sim  # noqa: F401 — triggers all @register decorators
from delivery_sim.config.loader import load_scenario
from delivery_sim.config.schema import ScenarioConfig
from delivery_sim.envs.single_agent import DeliveryEnv

# ── defaults ─────────────────────────────────────────────────────────────────

_DEFAULT_RADII: list[float] = [
    0.0, 50.0, 100.0, 150.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 1000.0,
]
_DEFAULT_SEEDS: list[int] = [42, 43, 44, 45, 46]

_SWEEP_KEYS: tuple[str, ...] = (
    "delivery_rate_total",
    "failed_rate_resolved",
    "pending_at_horizon",
    "courier_utilization",
    "total_delivery_cost",
    "total_orders",
)

# ── per-episode runner ────────────────────────────────────────────────────────


def _run_one(config: ScenarioConfig, radius: float, seed: int) -> dict[str, float]:
    """Run one episode with *radius* set on every store; return KPI dict."""
    cfg = config.model_copy(update={"seed": seed})
    env = DeliveryEnv(cfg)
    env.reset(seed=seed)

    n_stores = len(cfg.stores)
    action = np.full((n_stores,), fill_value=radius, dtype=np.float32)

    final_kpi: dict[str, Any] = {}
    terminated = False
    truncated = False
    while not (terminated or truncated):
        _, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            final_kpi = dict(info.get("kpi") or {})
    env.close()

    delivered = int(final_kpi.get("delivered_orders", 0))
    failed = int(final_kpi.get("failed_orders", 0))
    total = int(final_kpi.get("total_orders", 0))
    resolved = delivered + failed

    return {
        "delivery_rate_total": delivered / total if total > 0 else float("nan"),
        "failed_rate_resolved": failed / resolved if resolved > 0 else float("nan"),
        "pending_at_horizon": float(total - delivered - failed),
        "courier_utilization": float(
            final_kpi.get("courier_utilization", float("nan"))
        ),
        "total_delivery_cost": float(
            final_kpi.get("total_delivery_cost", float("nan"))
        ),
        "total_orders": float(total),
    }


# ── sweep ─────────────────────────────────────────────────────────────────────


def sweep(
    config: ScenarioConfig,
    radii: list[float],
    seeds: list[int],
) -> list[dict[str, Any]]:
    """Sweep *radii*; return per-radius dicts with mean/std over *seeds*.

    Each row contains:
      ``radius`` plus ``{key}_mean`` and ``{key}_std`` for every key in
      ``_SWEEP_KEYS``.  NaN seeds are excluded from the aggregation via
      ``np.nanmean`` / ``np.nanstd``.
    """
    rows: list[dict[str, Any]] = []
    for radius in radii:
        per_seed = [_run_one(config, radius, s) for s in seeds]
        row: dict[str, Any] = {"radius": radius}
        for key in _SWEEP_KEYS:
            vals = np.asarray([d[key] for d in per_seed], dtype=float)
            row[f"{key}_mean"] = float(np.nanmean(vals))
            row[f"{key}_std"] = float(np.nanstd(vals))
        rows.append(row)
    return rows


# ── table printer ─────────────────────────────────────────────────────────────


def print_table(rows: list[dict[str, Any]], config: ScenarioConfig) -> None:
    """Print the leverage-curve table to stdout."""
    horizon = config.max_steps * config.dt
    n_couriers = sum(c.count for c in config.couriers)
    speed = config.couriers[0].speed if config.couriers else float("nan")
    demand = config.demand.rate

    # back-of-envelope capacity line
    # mean leg ≈ 530 units (stores near corners, 1000×1000 world); prep_time=30s fixed
    mean_leg = 530.0
    cycle_est = 2.0 * mean_leg / speed + 30.0
    throughput_est = n_couriers / cycle_est if cycle_est > 0 else float("nan")
    ratio_est = demand / throughput_est if throughput_est > 0 else float("nan")

    hdr_line = (
        f"\nScenario : {config.name}"
        f"  |  horizon={horizon:.0f}s  |  demand={demand}/s"
        f"  |  couriers={n_couriers}  |  speed={speed}"
        f"\nCapacity : cycle~{cycle_est:.0f}s  throughput~{throughput_est:.4f} del/s"
        f"  demand/throughput~{ratio_est:.1f}x"
        f"  (demand/throughput>1 = over-capacity; prep_time=30s hardcoded FLAG)"
    )
    col_hdr = (
        f"  {'radius':>7}  {'DR/total':>12}  {'FR/resolvd':>14}"
        f"  {'Pending':>9}  {'Util':>8}  {'Cost':>8}  {'Orders':>7}"
    )
    sep = "-" * len(col_hdr)
    print(hdr_line)
    print(col_hdr)
    print(sep)
    for row in rows:
        r = float(row["radius"])
        dr_m = float(row["delivery_rate_total_mean"])
        dr_s = float(row["delivery_rate_total_std"])
        fr_m = float(row["failed_rate_resolved_mean"])
        fr_s = float(row["failed_rate_resolved_std"])
        pend = float(row["pending_at_horizon_mean"])
        util = float(row["courier_utilization_mean"])
        cost = float(row["total_delivery_cost_mean"])
        ordr = float(row["total_orders_mean"])
        print(
            f"  {r:>7.0f}  {dr_m:>6.3f} ±{dr_s:.3f}  "
            f"{fr_m:>8.3f} ±{fr_s:.3f}  "
            f"{pend:>9.1f}  {util:>7.3f}  "
            f"{cost:>7.1f}  {ordr:>6.1f}"
        )
    print(sep)
    # balance verdict
    hi = rows[-1]
    lo = rows[0]
    dr_hi = float(hi["delivery_rate_total_mean"])
    dr_lo = float(lo["delivery_rate_total_mean"])
    ut_hi = float(hi["courier_utilization_mean"])
    pend_hi = float(hi["pending_at_horizon_mean"])
    fr_lo = float(lo["failed_rate_resolved_mean"])
    ok_dr_hi = "OK" if dr_hi > 0.75 else "FAIL"
    ok_util = "OK" if 0.65 < ut_hi < 0.97 else "FAIL"
    ok_pend = "OK" if pend_hi < 15 else "FAIL"
    ok_fr_lo = "OK" if fr_lo > 0.85 or np.isnan(fr_lo) else "FAIL"
    ok_dr_lo = "OK" if dr_lo < 0.10 else "FAIL"
    print(
        f"  Balance (r=max): DR={dr_hi:.3f}>{0.75}[{ok_dr_hi}]"
        f"  Util={ut_hi:.3f} in (0.65,0.97)[{ok_util}]"
        f"  Pending={pend_hi:.1f}<15[{ok_pend}]"
    )
    print(
        f"  Balance (r=0):   FR={fr_lo:.3f}>{0.85}[{ok_fr_lo}]"
        f"  DR={dr_lo:.3f}<0.10[{ok_dr_lo}]"
    )
    print()


# ── optional plot ─────────────────────────────────────────────────────────────


def _maybe_plot(rows: list[dict[str, Any]], config: ScenarioConfig) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[diagnose] matplotlib not installed; skipping --plot", file=sys.stderr)
        return

    radii_vals = [float(r["radius"]) for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    panels = [
        (axes[0], "delivery_rate_total", "Delivery Rate (total)"),
        (axes[1], "failed_rate_resolved", "Fail Rate (resolved)"),
        (axes[2], "courier_utilization", "Courier Utilization"),
    ]
    for ax, key, label in panels:
        means = [float(r[f"{key}_mean"]) for r in rows]
        stds = [float(r[f"{key}_std"]) for r in rows]
        ax.errorbar(radii_vals, means, yerr=stds, marker="o", capsize=4, lw=1.5)
        ax.set_xlabel("coverage_radius")
        ax.set_ylabel(label)
        ax.set_title(config.name)
        ax.set_xlim(-20, max(radii_vals) + 20)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.35)

    n_couriers = sum(c.count for c in config.couriers)
    speed = config.couriers[0].speed if config.couriers else float("nan")
    fig.suptitle(
        f"{config.name}  demand={config.demand.rate}/s  "
        f"couriers={n_couriers}  speed={speed}  horizon={config.max_steps*config.dt:.0f}s",
        fontsize=10,
    )
    fig.tight_layout()
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"leverage_{config.name}.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"[diagnose] plot → {out}")


# ── main ──────────────────────────────────────────────────────────────────────


def main(
    config_path: Path,
    radii: list[float],
    seeds: list[int],
    plot: bool,
) -> None:
    """Load *config_path*, sweep *radii* × *seeds*, print table, optionally plot."""
    config = load_scenario(config_path)
    n_eps = len(radii) * len(seeds)
    print(
        f"[diagnose] {config_path.name}: "
        f"{len(radii)} radii × {len(seeds)} seeds = {n_eps} episodes ..."
    )
    rows = sweep(config, radii, seeds)
    print_table(rows, config)
    if plot:
        _maybe_plot(rows, config)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--config",
        type=Path,
        default=Path("scenarios/example.yaml"),
        help="Scenario YAML to sweep (default: scenarios/example.yaml)",
    )
    p.add_argument(
        "--radii",
        type=float,
        nargs="+",
        default=_DEFAULT_RADII,
        metavar="R",
        help="Coverage radii to sweep (default: 0 50 100 … 1000)",
    )
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=_DEFAULT_SEEDS,
        metavar="S",
        help="Episode seeds (default: 42 43 44 45 46)",
    )
    p.add_argument(
        "--plot",
        action="store_true",
        help="Save leverage-curve PNG to outputs/ (requires matplotlib)",
    )
    return p.parse_args()


if __name__ == "__main__":
    _args = _parse_args()
    main(
        config_path=Path(_args.config),
        radii=[float(r) for r in _args.radii],
        seeds=[int(s) for s in _args.seeds],
        plot=bool(_args.plot),
    )
