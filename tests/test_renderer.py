"""
Tests for the render layer: PygameRenderer protocol compliance and
observer-invariance.

All tests in this module are skipped when pygame is not installed, so the
core 310-test suite stays pygame-free.  When pygame IS installed the
observer-invariance test runs via the SDL dummy video driver
(SDL_VIDEODRIVER=dummy) — no physical display required, works in CI.
"""

from __future__ import annotations

import os

import pytest

# Set SDL to dummy *before* any pygame import so there's no display required.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

# Skip entire module when pygame is absent.
pygame = pytest.importorskip("pygame", reason="pygame not installed — skipping renderer tests")

import delivery_sim  # noqa: E402, F401
from delivery_sim.config.schema import (  # noqa: E402
    CourierConfig,
    DemandConfig,
    RewardConfig,
    ScenarioConfig,
    StoreConfig,
    WorldConfig,
)
from delivery_sim.engine.simulator import Simulator  # noqa: E402
from delivery_sim.render.protocol import (  # noqa: E402
    SnapshotConsumer,
    StoreSnapshot,
    WorldSnapshot,
)
from delivery_sim.render.pygame_renderer import PygameRenderer  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _small_config(seed: int = 99) -> ScenarioConfig:
    """Tiny, fast scenario for renderer tests (completes in < 1 s)."""
    return ScenarioConfig(
        name="renderer_test",
        seed=seed,
        dt=1.0,
        max_steps=60,
        world=WorldConfig(width=500.0, height=500.0),
        stores=[
            StoreConfig(name="s0", x=100.0, y=100.0, capacity=5, coverage_radius=200.0),
        ],
        couriers=[
            CourierConfig(courier_type="BikeCourier", count=2, speed=2.0, cost_per_unit=0.01),
        ],
        demand=DemandConfig(generator_type="PoissonDemandGenerator", rate=0.2),
        reward=RewardConfig(function_type="SparseDeliveryReward"),
        decision_interval=10.0,
        max_coverage_radius=500.0,
    )


def _make_renderer(config: ScenarioConfig | None = None) -> PygameRenderer:
    cfg = config or _small_config()
    return PygameRenderer(
        world_width=cfg.world.width,
        world_height=cfg.world.height,
        fps=0,  # unlimited — no sleep in CI
    )


def _order_fingerprints(sim: Simulator) -> dict[str, str]:
    """(order_id → status.name) after a completed run."""
    assert sim.world is not None
    return {oid: o.status.name for oid, o in sim.world.active_orders.items()}


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------

def test_pygame_renderer_satisfies_snapshot_consumer_protocol() -> None:
    r = _make_renderer()
    assert isinstance(r, SnapshotConsumer)
    r.close()


def test_store_snapshot_exposes_coverage_radius() -> None:
    snap = StoreSnapshot(store_id="s0", x=10.0, y=20.0, coverage_radius=350.0)
    assert snap.coverage_radius == 350.0


def test_renderer_consume_does_not_raise_on_empty_snapshot() -> None:
    r = _make_renderer()
    empty = WorldSnapshot(tick=0, elapsed=0.0, stores=(), couriers=(), orders=())
    r.consume(empty)
    r.close()


def test_renderer_close_is_idempotent() -> None:
    r = _make_renderer()
    r.close()
    r.close()  # second call must not raise


# ---------------------------------------------------------------------------
# Observer-invariance with PygameRenderer
# ---------------------------------------------------------------------------

def test_observer_invariance_with_pygame_renderer() -> None:
    """Headless run and PygameRenderer run produce identical order outcomes.

    This is the hard correctness requirement: the renderer is a pure observer
    that must NOT influence the simulation.  SDL_VIDEODRIVER=dummy set above
    means no window appears but all drawing code executes normally.
    """
    config = _small_config(seed=77)

    # Reference: headless (no consumer)
    sim_h = Simulator(config)
    sim_h.run()
    headless_fp = _order_fingerprints(sim_h)

    # Under test: renderer attached as consumer
    renderer = _make_renderer(config)
    sim_r = Simulator(config)
    sim_r.run(consumer=renderer)
    renderer_fp = _order_fingerprints(sim_r)

    assert headless_fp == renderer_fp, (
        "Observer-invariance violated: headless and renderer runs diverged.\n"
        f"  headless : {headless_fp}\n"
        f"  renderer : {renderer_fp}"
    )


def test_renderer_receives_coverage_radius_from_snapshot() -> None:
    """Snapshots passed to the renderer carry coverage_radius from StoreSnapshot."""
    captured: list[float] = []

    class _Probe:
        def consume(self, snapshot: WorldSnapshot) -> None:
            for s in snapshot.stores:
                captured.append(s.coverage_radius)

        def close(self) -> None:
            pass

    config = _small_config()
    sim = Simulator(config)
    sim.run(consumer=_Probe())

    assert len(captured) > 0, "no snapshots received"
    assert all(r >= 0.0 for r in captured), "coverage_radius must be non-negative"
    # The config sets coverage_radius=200.0; all snapshots must reflect that
    expected = config.stores[0].coverage_radius
    assert all(abs(r - expected) < 1e-9 for r in captured), (
        f"unexpected coverage_radius values: {set(captured)}"
    )
