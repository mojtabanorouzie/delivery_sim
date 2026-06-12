"""B0 fingerprint regression guard.

Runs balanced.yaml and example.yaml with the post-B10 engine and asserts
EXACT equality against KPI values captured from the pre-B1 commit
(git stash / restore on 2026-06-12).

What this proves:
  1. spawn(2)[0] entropy == spawn(1)[0] entropy for both scenario seeds
     (balanced seed=42, example seed=42) — gen_rng stream unchanged.
  2. prep_time defaulting to 30.0 via StoreConfig did not alter BuiltinStore
     behavior (old hardcoded value was also 30.0).
  3. New PoissonDemandGenerator kwargs (intensity, profile, burst_*, **_ignored)
     did not perturb the demand draw sequence.
  4. All B-realistic additions are truly additive — no side-effect on the
     B0-era scenarios that the trained agent's results depend on.

These values were captured by running the pre-B1 simulator (HEAD commit) on
the same hardware/numpy version, then verified by running post-B10 and
confirming dict equality.
"""

from __future__ import annotations

from pathlib import Path

from delivery_sim.config.loader import load_scenario
from delivery_sim.engine.simulator import Simulator
from delivery_sim.metrics.collector import KPICollector

_SCENARIOS = Path(__file__).parent.parent / "scenarios"

# Pre-B1 KPI values captured from HEAD commit (before any B-realistic changes).
# Keys are limited to the B0 KPI surface (no B7 return_* keys).
_B0_BALANCED = {
    "courier_utilization": 0.6417087594408648,
    "delivered_orders": 85,
    "delivery_rate": 0.7798165137614679,
    "failed_orders": 16,
    "mean_delivery_time": 580.7407672425021,
    "mean_pickup_latency": 381.5029635292752,
    "p50_delivery_time": 537.5347580054929,
    "p95_delivery_time": 884.3098124595388,
    "sla_violations": 0,
    "total_delivery_cost": 702.1944782341906,
    "total_orders": 109,
}

_B0_EXAMPLE = {
    "courier_utilization": 0.9962621466037089,
    "delivered_orders": 2,
    "delivery_rate": 0.00392156862745098,
    "failed_orders": 70,
    "mean_delivery_time": 697.3856194842908,
    "mean_pickup_latency": 457.61423749153965,
    "p50_delivery_time": 697.3856194842908,
    "p95_delivery_time": 875.5561322249847,
    "sla_violations": 0,
    "total_delivery_cost": 20.021568584528723,
    "total_orders": 510,
}


def _run(scenario_name: str) -> dict:
    cfg = load_scenario(_SCENARIOS / f"{scenario_name}.yaml")
    sim = Simulator(cfg)
    col = KPICollector()
    sim.attach_collector(col)
    sim.run()
    return col.summary()


class TestB0Fingerprint:
    def test_balanced_yaml_matches_b0_fingerprint(self) -> None:
        """Post-B10 balanced.yaml KPIs must exactly match pre-B1 values."""
        current = _run("balanced")
        for key, expected in _B0_BALANCED.items():
            assert current[key] == expected, (
                f"balanced.yaml B0 fingerprint drift on '{key}': "
                f"expected={expected!r}, got={current[key]!r}"
            )

    def test_example_yaml_matches_b0_fingerprint(self) -> None:
        """Post-B10 example.yaml KPIs must exactly match pre-B1 values."""
        current = _run("example")
        for key, expected in _B0_EXAMPLE.items():
            assert current[key] == expected, (
                f"example.yaml B0 fingerprint drift on '{key}': "
                f"expected={expected!r}, got={current[key]!r}"
            )

    def test_balanced_yaml_has_zero_return_rate(self) -> None:
        """balanced.yaml must not have return_rate set (uses default 0.0)."""
        cfg = load_scenario(_SCENARIOS / "balanced.yaml")
        assert cfg.return_rate == 0.0

    def test_example_yaml_has_zero_return_rate(self) -> None:
        """example.yaml must not have return_rate set (uses default 0.0)."""
        cfg = load_scenario(_SCENARIOS / "example.yaml")
        assert cfg.return_rate == 0.0
