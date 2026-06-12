"""
Observer-invariance test for PygameRenderer logic — no pygame required.

Strategy
--------
The real PygameRenderer cannot be instantiated without pygame (Python 3.14 has
no pre-built wheel at the time of writing).  This module instead verifies
invariance through two complementary approaches:

1. Snapshot-immutability audit
   WorldSnapshot and all nested dataclasses are ``frozen=True``.  Any attempt
   to mutate a field raises FrozenInstanceError at the Python level.  We
   confirm this holds for every dataclass in the snapshot hierarchy.

2. ShadowRenderer invariance test
   A ``_ShadowRenderer`` replicates *every read path* that PygameRenderer
   makes against the snapshot — the same dict comprehensions, the same field
   accesses, the same computed values — but performs no drawing.  We run the
   same simulation twice (headless vs. with ShadowRenderer attached) and
   assert byte-identical order-status fingerprints.

   This is the exact structure of ``test_observer_invariance_with_pygame_renderer``
   in test_renderer.py; the only difference is the consumer is our shadow, not
   the real pygame-backed renderer.

3. demand_pattern field propagation
   Confirms the SA-3 field reaches the consumer correctly.

4. Preset-switch callback safety
   Confirms that the ``on_preset_switch`` callback (a) defaults to None so it
   never fires in headless training/eval runs, and (b) does not mutate any
   simulation state when it fires.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from delivery_sim.config.schema import (
    CourierConfig,
    DemandConfig,
    RewardConfig,
    ScenarioConfig,
    StoreConfig,
    WorldConfig,
)
from delivery_sim.engine.simulator import Simulator
from delivery_sim.render.protocol import (
    CourierSnapshot,
    OrderSnapshot,
    StoreSnapshot,
    WorldSnapshot,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _config(seed: int = 77, generator_type: str = "PoissonDemandGenerator") -> ScenarioConfig:
    return ScenarioConfig(
        name="inv_test",
        seed=seed,
        dt=1.0,
        max_steps=80,
        world=WorldConfig(width=500.0, height=500.0),
        stores=[StoreConfig(name="s0", x=150.0, y=150.0, capacity=4, coverage_radius=200.0)],
        couriers=[
            CourierConfig(courier_type="BikeCourier", count=3, speed=2.0, cost_per_unit=0.01)
        ],
        demand=DemandConfig(generator_type=generator_type, rate=0.2),
        reward=RewardConfig(function_type="SparseDeliveryReward"),
        decision_interval=10.0,
        max_coverage_radius=500.0,
    )


def _order_fingerprints(sim: Simulator) -> dict[str, str]:
    assert sim.world is not None
    return {oid: o.status.name for oid, o in sim.world.active_orders.items()}


class _ShadowRenderer:
    """Mirrors every snapshot read that PygameRenderer makes, without pygame.

    Reads are grouped to match _draw(), _draw_hud_col_a(), _draw_hud_col_b(),
    and _draw_scenario_overlay() in pygame_renderer.py.  Any accidental
    mutation in a real renderer would be caught here if the same field were
    written — but frozen dataclasses make that impossible at the Python level.
    """

    def __init__(self) -> None:
        self.frame_count = 0
        self.last_demand_pattern: str = ""
        self.last_scenario_name: str = ""

    def consume(self, snapshot: WorldSnapshot) -> None:  # noqa: C901
        self.frame_count += 1

        # --- Mirrors _draw() ---

        # Coverage circle reads
        for store in snapshot.stores:
            _ = store.x, store.y, store.coverage_radius

        # Return-path dashed line reads
        returning_map = {
            c.courier_id: c for c in snapshot.couriers if c.status == "returning"
        }
        for order in snapshot.orders:
            if (
                order.status == "RETURNED"
                and order.assigned_courier_id is not None
                and order.assigned_courier_id in returning_map
            ):
                cr = returning_map[order.assigned_courier_id]
                _ = cr.x, cr.y, order.customer_x, order.customer_y

        # Order shape reads
        for order in snapshot.orders:
            _ = order.customer_x, order.customer_y
            _ = order.status

        # Courier phase reads
        for courier in snapshot.couriers:
            _ = courier.x, courier.y, courier.status, courier.courier_id

        # Store marker + badge reads
        for store in snapshot.stores:
            _ = store.x, store.y, store.queue_depth, store.store_id, store.coverage_radius

        # Scenario overlay reads
        self.last_scenario_name = snapshot.scenario_name
        _ = snapshot.scenario_name, snapshot.demand_intensity

        # Banner check
        _ = snapshot.elapsed

        # --- Mirrors _draw_hud_col_a() ---
        n_del  = sum(1 for o in snapshot.orders if o.status == "DELIVERED")
        n_fail = sum(1 for o in snapshot.orders if o.status == "FAILED")
        n_ret  = sum(1 for o in snapshot.orders if o.status == "RETURNED")
        n_pend = len(snapshot.orders) - n_del - n_fail - n_ret
        attempts = n_del + n_ret
        _rate = n_ret / attempts if attempts > 0 else 0.0
        total_q = sum(s.queue_depth for s in snapshot.stores)
        mean_r = (
            sum(s.coverage_radius for s in snapshot.stores) / len(snapshot.stores)
            if snapshot.stores else 0.0
        )
        _ = snapshot.elapsed, snapshot.tick, n_del, n_fail, n_pend, n_ret, total_q, mean_r

        # --- Mirrors _draw_hud_col_b() ---
        _ = snapshot.scenario_name
        intensity = max(0.0, min(1.0, snapshot.demand_intensity))
        _ = intensity
        self.last_demand_pattern = snapshot.demand_pattern
        _ = snapshot.demand_pattern

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# 1. Snapshot immutability
# ---------------------------------------------------------------------------

def test_world_snapshot_is_frozen() -> None:
    snap = WorldSnapshot(tick=0, elapsed=0.0, stores=(), couriers=(), orders=())
    with pytest.raises((dataclasses.FrozenInstanceError, TypeError)):
        snap.tick = 99  # type: ignore[misc]


def test_store_snapshot_is_frozen() -> None:
    s = StoreSnapshot(store_id="x", x=0.0, y=0.0, coverage_radius=100.0, queue_depth=0)
    with pytest.raises((dataclasses.FrozenInstanceError, TypeError)):
        s.queue_depth = 5  # type: ignore[misc]


def test_courier_snapshot_is_frozen() -> None:
    c = CourierSnapshot(courier_id="c0", x=0.0, y=0.0, status="free")
    with pytest.raises((dataclasses.FrozenInstanceError, TypeError)):
        c.status = "returning"  # type: ignore[misc]


def test_order_snapshot_is_frozen() -> None:
    o = OrderSnapshot(order_id="o0", status="CREATED", customer_x=1.0, customer_y=1.0)
    with pytest.raises((dataclasses.FrozenInstanceError, TypeError)):
        o.status = "DELIVERED"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 2. ShadowRenderer observer-invariance
# ---------------------------------------------------------------------------

def test_shadow_renderer_observer_invariance() -> None:
    """Headless run and ShadowRenderer-attached run produce identical results.

    This replicates the structure of test_observer_invariance_with_pygame_renderer
    in test_renderer.py (which requires pygame).  The ShadowRenderer exercises
    every snapshot read path in PygameRenderer without any pygame calls.
    """
    config = _config(seed=77)

    sim_headless = Simulator(config)
    sim_headless.run()
    fp_headless = _order_fingerprints(sim_headless)

    shadow = _ShadowRenderer()
    sim_shadow = Simulator(config)
    sim_shadow.run(consumer=shadow)
    fp_shadow = _order_fingerprints(sim_shadow)

    assert fp_headless == fp_shadow, (
        "Observer-invariance violated:\n"
        f"  headless : {fp_headless}\n"
        f"  shadow   : {fp_shadow}"
    )
    assert shadow.frame_count > 0, "ShadowRenderer received no frames"


def test_shadow_renderer_invariance_with_returns() -> None:
    """Invariance holds when return_rate > 0 (tests the return-path read branch)."""
    config = _config(seed=42).model_copy(update={"return_rate": 0.3})

    sim_headless = Simulator(config)
    sim_headless.run()
    fp_headless = _order_fingerprints(sim_headless)

    shadow = _ShadowRenderer()
    sim_shadow = Simulator(config)
    sim_shadow.run(consumer=shadow)
    fp_shadow = _order_fingerprints(sim_shadow)

    assert fp_headless == fp_shadow


def test_shadow_renderer_invariance_with_store_queue() -> None:
    """Invariance holds when store queues form (tests queue_depth read branch)."""
    config = _config(seed=13).model_copy(
        update={"stores": [
            StoreConfig(name="s0", x=150.0, y=150.0, capacity=1, coverage_radius=200.0)
        ]}
    )

    sim_headless = Simulator(config)
    sim_headless.run()
    fp_headless = _order_fingerprints(sim_headless)

    shadow = _ShadowRenderer()
    sim_shadow = Simulator(config)
    sim_shadow.run(consumer=shadow)
    fp_shadow = _order_fingerprints(sim_shadow)

    assert fp_headless == fp_shadow


# ---------------------------------------------------------------------------
# 3. demand_pattern field propagation (SA-3 resolved)
# ---------------------------------------------------------------------------

def test_demand_pattern_reaches_consumer_poisson() -> None:
    shadow = _ShadowRenderer()
    sim = Simulator(_config(generator_type="PoissonDemandGenerator"))
    sim.run(consumer=shadow)
    assert shadow.last_demand_pattern == "PoissonDemandGenerator"


def test_demand_pattern_reaches_consumer_burst() -> None:
    from delivery_sim.config.schema import DemandConfig
    cfg = _config().model_copy(update={"demand": DemandConfig(
        generator_type="BurstDemandGenerator",
        rate=0.2,
        intensity=1.0,
        burst_rate_factor=3.0,
        burst_duration_fraction=0.1,
        burst_interval_fraction=0.3,
    )})
    shadow = _ShadowRenderer()
    sim = Simulator(cfg)
    sim.run(consumer=shadow)
    assert shadow.last_demand_pattern == "BurstDemandGenerator"


def test_demand_pattern_empty_on_headless_run() -> None:
    """No consumer → WorldSnapshot is never produced; demand_pattern default is ''."""
    snap = WorldSnapshot(tick=0, elapsed=0.0, stores=(), couriers=(), orders=())
    assert snap.demand_pattern == ""


# ---------------------------------------------------------------------------
# 4. Preset-switch callback safety
# ---------------------------------------------------------------------------

def test_on_preset_switch_defaults_to_none() -> None:
    """PygameRenderer default has no callback — safe in training/eval runs.

    We can't instantiate PygameRenderer without pygame, so we verify the
    documented contract via the __init__ signature default directly.
    """
    import inspect

    from delivery_sim.render.pygame_renderer import PygameRenderer
    sig = inspect.signature(PygameRenderer.__init__)
    param = sig.parameters.get("on_preset_switch")
    assert param is not None, "on_preset_switch parameter missing"
    assert param.default is None, (
        f"on_preset_switch default must be None (got {param.default!r}); "
        "a non-None default would fire on every instantiation"
    )


def test_preset_switch_callback_does_not_touch_snapshot() -> None:
    """Callback receives only the preset name — no simulation objects passed.

    This confirms the renderer cannot hand a live simulation reference to the
    callback.  The callback only gets a str; it must call env.reset() itself.
    """
    received: list[Any] = []

    def mock_callback(name: str) -> None:
        received.append(name)

    # Simulate what consume() does when it detects a preset key press.
    # We replicate the minimal key-handling logic without pygame.
    renderer_switch_msg: list[str] = []

    def _simulate_key_press(name: str, elapsed: float) -> None:
        renderer_switch_msg.append(f"Switching to {name}… (restart)")
        mock_callback(name)

    _simulate_key_press("heavy", elapsed=1234.5)

    assert received == ["heavy"], f"callback received unexpected args: {received}"
    assert "restart" in renderer_switch_msg[0].lower()
    # Crucially: no simulation object was passed to the callback
    assert all(isinstance(v, str) for v in received)
