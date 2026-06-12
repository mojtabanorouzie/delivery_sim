"""B9 — Snapshot additions.

SA-1: StoreSnapshot.queue_depth reflects live store queue depth.
SA-2: StoreSnapshot backward compat — existing constructions without queue_depth still work.
SA-3: WorldSnapshot.scenario_name and demand_intensity populated by Simulator.
SA-4: demand_intensity matches generator's current_intensity at snapshot time.
SA-5: WorldState.snapshot() accepts scenario_name/demand_intensity kwargs.
SA-6: DailyProfile intensity visible in snapshots during high/low phases.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest

from delivery_sim.config.schema import (
    CourierConfig,
    DemandConfig,
    ProfileBreakpoint,
    ScenarioConfig,
    StoreConfig,
)
from delivery_sim.engine.simulator import Simulator
from delivery_sim.engine.world_state import WorldState
from delivery_sim.entities.store import BuiltinStore
from delivery_sim.metrics.collector import KPICollector
from delivery_sim.render.protocol import (
    StoreSnapshot,
    WorldSnapshot,
)

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# SA-2: backward compat — existing snapshot constructions without queue_depth
# ---------------------------------------------------------------------------

class TestSnapshotBackwardCompat:
    def test_store_snapshot_default_queue_depth(self) -> None:
        snap = StoreSnapshot(store_id="s0", x=10.0, y=20.0, coverage_radius=350.0)
        assert snap.queue_depth == 0

    def test_world_snapshot_default_extras(self) -> None:
        snap = WorldSnapshot(tick=0, elapsed=0.0, stores=(), couriers=(), orders=())
        assert snap.scenario_name == ""
        assert snap.demand_intensity == 0.0

    def test_store_snapshot_with_queue_depth(self) -> None:
        snap = StoreSnapshot(store_id="s0", x=10.0, y=20.0, coverage_radius=350.0, queue_depth=3)
        assert snap.queue_depth == 3


# ---------------------------------------------------------------------------
# SA-5: WorldState.snapshot() kwargs
# ---------------------------------------------------------------------------

class TestWorldStateSnapshotKwargs:
    def _make_world(self) -> WorldState:
        store = BuiltinStore(store_id="s0", x=500.0, y=500.0, capacity=5, prep_time=30.0)
        return WorldState(
            width=1000.0, height=1000.0,
            stores=[store], couriers=[], active_orders={}, courier_phase={},
        )

    def test_snapshot_no_kwargs_defaults(self) -> None:
        world = self._make_world()
        snap = world.snapshot(0, 0.0)
        assert snap.scenario_name == ""
        assert snap.demand_intensity == 0.0

    def test_snapshot_with_kwargs(self) -> None:
        world = self._make_world()
        snap = world.snapshot(5, 100.0, scenario_name="my_test", demand_intensity=2.5)
        assert snap.scenario_name == "my_test"
        assert snap.demand_intensity == pytest.approx(2.5)

    def test_snapshot_store_queue_depth_zero(self) -> None:
        world = self._make_world()
        snap = world.snapshot(0, 0.0)
        assert snap.stores[0].queue_depth == 0

    def test_snapshot_store_queue_depth_nonzero(self) -> None:
        store = BuiltinStore(store_id="s0", x=500.0, y=500.0, capacity=0, prep_time=30.0)
        store.enqueue_waiter("c0", "o0", 0.0)
        store.enqueue_waiter("c1", "o1", 0.0)
        world = WorldState(
            width=1000.0, height=1000.0,
            stores=[store], couriers=[], active_orders={}, courier_phase={},
        )
        snap = world.snapshot(0, 0.0)
        assert snap.stores[0].queue_depth == 2


# ---------------------------------------------------------------------------
# SA-1, SA-3: queue_depth and scenario_name in full sim run
# ---------------------------------------------------------------------------

def _capacity1_cfg(
    generator_type: str = "PoissonDemandGenerator", seed: int = 42
) -> ScenarioConfig:
    return ScenarioConfig(
        name="test_scenario",
        seed=seed,
        dt=1.0,
        max_steps=3000,
        stores=[StoreConfig(
            name="s", x=500.0, y=500.0,
            capacity=1, prep_time=60.0, coverage_radius=1000.0,
        )],
        couriers=[CourierConfig(courier_type="BikeCourier", count=5, speed=2.0)],
        demand=DemandConfig(generator_type=generator_type, rate=0.03),
    )


@dataclass
class SnapshotRecorder:
    """Records all snapshots produced by Simulator.run(consumer=...)."""

    snapshots: list[WorldSnapshot] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.snapshots = []

    def consume(self, snapshot: WorldSnapshot) -> None:
        self.snapshots.append(snapshot)

    def close(self) -> None:
        pass


class TestSimulatorSnapshotContent:
    def test_scenario_name_in_snapshot(self) -> None:
        cfg = _capacity1_cfg()
        sim = Simulator(cfg)
        recorder = SnapshotRecorder()
        sim.run(consumer=recorder)
        assert len(recorder.snapshots) > 0
        for snap in recorder.snapshots:
            assert snap.scenario_name == "test_scenario"

    def test_demand_intensity_poisson_is_one(self) -> None:
        cfg = _capacity1_cfg()
        sim = Simulator(cfg)
        recorder = SnapshotRecorder()
        sim.run(consumer=recorder)
        for snap in recorder.snapshots:
            assert snap.demand_intensity == pytest.approx(1.0)

    def test_queue_depth_nonzero_during_cap1_episode(self) -> None:
        cfg = _capacity1_cfg()
        sim = Simulator(cfg)
        recorder = SnapshotRecorder()
        col = KPICollector()
        sim.attach_collector(col)
        sim.run(consumer=recorder)
        # With capacity=1 and 5 couriers + high demand, the queue must form
        # at some snapshot.  Check that max observed queue_depth > 0.
        max_depth = max(
            snap.stores[0].queue_depth
            for snap in recorder.snapshots
        )
        assert max_depth > 0

    def test_store_coverage_radius_in_snapshot(self) -> None:
        cfg = _capacity1_cfg()
        sim = Simulator(cfg)
        recorder = SnapshotRecorder()
        sim.run(consumer=recorder)
        for snap in recorder.snapshots:
            assert snap.stores[0].coverage_radius == 1000.0


# ---------------------------------------------------------------------------
# SA-4: demand_intensity varies for DailyProfileDemandGenerator
# ---------------------------------------------------------------------------

class TestDemandIntensityInSnapshot:
    def test_daily_profile_intensity_varies_across_snapshots(self) -> None:
        profile = [
            ProfileBreakpoint(time_fraction=0.0, rate_factor=0.1),
            ProfileBreakpoint(time_fraction=0.5, rate_factor=5.0),
            ProfileBreakpoint(time_fraction=1.0, rate_factor=0.1),
        ]
        cfg = ScenarioConfig(
            name="daily_profile_test",
            seed=42,
            dt=1.0,
            max_steps=3000,
            stores=[StoreConfig(
                name="s", x=500.0, y=500.0,
                capacity=20, prep_time=30.0, coverage_radius=1000.0,
            )],
            couriers=[CourierConfig(courier_type="BikeCourier", count=5, speed=2.0)],
            demand=DemandConfig(
                generator_type="DailyProfileDemandGenerator",
                rate=0.03,
                profile=profile,
            ),
        )
        sim = Simulator(cfg)
        recorder = SnapshotRecorder()
        sim.run(consumer=recorder)
        intensities = [s.demand_intensity for s in recorder.snapshots]
        # With a profile from 0.1→5.0→0.1, intensity must vary
        assert max(intensities) > min(intensities)
        # Peak is around t=1500 (midpoint of 3000), trough at start/end
        early = [s.demand_intensity for s in recorder.snapshots if s.elapsed < 300]
        mid = [s.demand_intensity for s in recorder.snapshots
               if 1300 < s.elapsed < 1700]
        if early and mid:
            assert sum(mid) / len(mid) > sum(early) / len(early)
