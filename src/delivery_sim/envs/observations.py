"""
ObservationSpec ABC and default preset library.

Layer: Control / RL Interface.

An ObservationSpec defines WHAT the RL agent observes: it maps the current
world state + KPICollector + interval statistics into a fixed-length float32
vector, and declares the gym Box bounds so observation_space is derived from
the spec rather than hard-coded in the env.

Registration
------------
All concrete specs are registered under the "observation" category::

    @register("observation", name="my_preset")
    class MyObservation(ObservationSpec): ...

Selecting a preset by name in config::

    observation_preset: my_preset

The env calls ``create("observation", config.observation_preset)`` at reset.

Default presets
---------------
"minimal" (n + 1 features):
    coverage_radius[i] / max_r  for i in 0..n-1
    interval_delivery_rate  (sentinel 0.5 when no orders completed this step)

"standard" (n + 5 features) — backward-compatible default:
    coverage_radius[i] / max_r  for i in 0..n-1
    interval_delivery_rate      (sentinel 0.5)
    interval_failed_rate        (sentinel 0.0)
    busy_courier_fraction       (live, from world.courier_phase)
    mean_delivery_time / horizon (episode mean, clipped [0, 1])
    pending_count / max_pending  (clipped [0, 1])

    The sentinel pair (0.5, 0.0) sums to 0.5, which is unreachable when real
    orders exist (delivery_rate + failed_rate == 1.0 then), so the two cases
    are unambiguously distinguishable.

"operational" (n + 3 features):
    coverage_radius[i] / max_r  for i in 0..n-1
    interval_failed_rate        (sentinel 0.0)
    busy_courier_fraction       (live)
    pending_count / max_pending  (clipped [0, 1])

    Focuses on the three signals most relevant to coverage decisions: how often
    coverage is missing (failed_rate), how loaded the fleet is (busy), and
    how much work is queued (pending).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from delivery_sim.registry import register

if TYPE_CHECKING:
    from delivery_sim.engine.world_state import WorldState
    from delivery_sim.metrics.collector import KPICollector


class ObservationSpec(ABC):
    """Maps world state + KPIs to a fixed-length float32 observation vector.

    Concrete subclasses are registered via ``@register("observation")`` so the
    scenario config can select them by name without importing the class.

    The ``observe`` / ``bounds`` contract
    -------------------------------------
    - ``observe`` must return a vector whose shape matches ``bounds(n_stores)``
      element-count, i.e. ``len(low) == len(high) == len(observe(...))``.
    - All returned values must satisfy ``low[i] <= v[i] <= high[i]``.
    - ``bounds`` must not depend on any mutable state (call it any time).
    """

    @abstractmethod
    def observe(
        self,
        world: WorldState,
        collector: KPICollector,
        interval_delivered: int,
        interval_failed: int,
        interval_total: int,
        max_r: float,
        max_pending: float,
        horizon: float,
    ) -> np.ndarray:
        """Return a float32 observation vector for the current env step.

        Args:
            world:              Current world state.
            collector:          Episode KPICollector (may be queried before finalize).
            interval_delivered: Orders delivered this decision interval.
            interval_failed:    Orders failed this decision interval.
            interval_total:     Terminal orders this interval (delivered + failed).
            max_r:              Action-space upper bound; used to normalise radii.
            max_pending:        Normaliser for pending-order count.
            horizon:            Episode duration in sim-seconds.

        Returns:
            1-D float32 array of length ``n_features(n_stores)``.
        """
        raise NotImplementedError

    @abstractmethod
    def bounds(self, n_stores: int) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(low, high)`` arrays defining the gym Box bounds.

        Both arrays have shape ``(n_features,)`` and dtype float32.  The env
        derives ``observation_space`` directly from these arrays.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# "minimal" preset — n + 1 features
# ---------------------------------------------------------------------------

@register("observation", name="minimal")
class MinimalObservation(ObservationSpec):
    """Smallest useful vector: coverage_radius per store + interval delivery rate.

    Vector layout (length n + 1, all in [0, 1]):
      obs[0..n-1]  coverage_radius[i] / max_r
      obs[n]       interval delivery rate (sentinel 0.5 when no orders)
    """

    def observe(
        self,
        world: WorldState,
        collector: KPICollector,  # noqa: ARG002
        interval_delivered: int,
        interval_failed: int,  # noqa: ARG002
        interval_total: int,
        max_r: float,
        max_pending: float,  # noqa: ARG002
        horizon: float,  # noqa: ARG002
    ) -> np.ndarray:
        """Return n + 1 float32 vector: normalised coverage + interval delivery rate."""
        coverage = np.clip(
            np.array(
                [s.coverage_radius / max_r for s in world.stores],                  dtype=np.float32,
            ),
            0.0,
            1.0,
        )
        delivery_rate = (
            float(interval_delivered) / interval_total
            if interval_total > 0
            else 0.5
        )
        return np.append(coverage, np.float32(delivery_rate)).astype(np.float32)

    def bounds(self, n_stores: int) -> tuple[np.ndarray, np.ndarray]:
        """Return ([0]*n+1, [1]*n+1) Box bounds."""
        n = n_stores + 1
        return np.zeros(n, dtype=np.float32), np.ones(n, dtype=np.float32)


# ---------------------------------------------------------------------------
# "standard" preset — n + 5 features (backward-compatible default)
# ---------------------------------------------------------------------------

@register("observation", name="standard")
class StandardObservation(ObservationSpec):
    """Current shipped vector: coverage + delivery/failed rates + busy + mean_dt + pending.

    Vector layout (length n + 5, all in [0, 1]):
      obs[0..n-1]  coverage_radius[i] / max_r
      obs[n]       interval delivery rate (sentinel 0.5 when no orders)
      obs[n+1]     interval failed rate   (sentinel 0.0 when no orders)
      obs[n+2]     busy-courier fraction  (live, from world.courier_phase)
      obs[n+3]     episode mean_delivery_time / horizon, clipped [0, 1]
      obs[n+4]     pending order count / max_pending, clipped [0, 1]

    This is the default and produces a byte-identical vector to the pre-refactor
    hardcoded ``_build_obs`` implementation.
    """

    def observe(
        self,
        world: WorldState,
        collector: KPICollector,
        interval_delivered: int,
        interval_failed: int,
        interval_total: int,
        max_r: float,
        max_pending: float,
        horizon: float,
    ) -> np.ndarray:
        """Return n + 5 float32 vector: coverage + rates + busy + mean_dt + pending."""
        coverage = np.clip(
            np.array(
                [s.coverage_radius / max_r for s in world.stores],                  dtype=np.float32,
            ),
            0.0,
            1.0,
        )

        if interval_total > 0:
            delivery_rate: float = float(interval_delivered) / interval_total
            failed_rate: float = float(interval_failed) / interval_total
        else:
            delivery_rate = 0.5
            failed_rate = 0.0

        busy = sum(1 for phase in world.courier_phase.values() if phase != "free")
        busy_fraction = float(busy) / max(1, len(world.couriers))

        mean_dt = float(collector.summary()["mean_delivery_time"])
        mean_dt_norm = (
            float(np.clip(mean_dt / horizon, 0.0, 1.0)) if horizon > 0.0 else 0.0
        )

        pending = sum(1 for o in world.active_orders.values() if not o.is_terminal)
        pending_norm = float(np.clip(pending / max_pending, 0.0, 1.0))

        scalars = np.array(
            [delivery_rate, failed_rate, busy_fraction, mean_dt_norm, pending_norm],
            dtype=np.float32,
        )
        return np.concatenate([coverage, scalars]).astype(np.float32)

    def bounds(self, n_stores: int) -> tuple[np.ndarray, np.ndarray]:
        """Return ([0]*n+5, [1]*n+5) Box bounds."""
        n = n_stores + 5
        return np.zeros(n, dtype=np.float32), np.ones(n, dtype=np.float32)


# ---------------------------------------------------------------------------
# "operational" preset — n + 3 features
# ---------------------------------------------------------------------------

@register("observation", name="operational")
class OperationalObservation(ObservationSpec):
    """Coverage-agent-tuned vector: coverage + failed rate + busy + pending.

    Focuses on the three signals most directly relevant to coverage decisions:
    how many orders are missing coverage (failed_rate), how loaded the fleet
    is (busy_fraction), and how much unassigned demand is queued (pending).
    Mean delivery time is excluded — it lags reality by the full episode and
    adds noise when coverage changes frequently.

    Vector layout (length n + 3, all in [0, 1]):
      obs[0..n-1]  coverage_radius[i] / max_r
      obs[n]       interval failed rate (sentinel 0.0 when no orders)
      obs[n+1]     busy-courier fraction (live)
      obs[n+2]     pending order count / max_pending, clipped [0, 1]
    """

    def observe(
        self,
        world: WorldState,
        collector: KPICollector,  # noqa: ARG002
        interval_delivered: int,  # noqa: ARG002
        interval_failed: int,
        interval_total: int,
        max_r: float,
        max_pending: float,
        horizon: float,  # noqa: ARG002
    ) -> np.ndarray:
        """Return n + 3 float32 vector: coverage + failed rate + busy + pending."""
        coverage = np.clip(
            np.array(
                [s.coverage_radius / max_r for s in world.stores],                  dtype=np.float32,
            ),
            0.0,
            1.0,
        )

        failed_rate = (
            float(interval_failed) / interval_total if interval_total > 0 else 0.0
        )

        busy = sum(1 for phase in world.courier_phase.values() if phase != "free")
        busy_fraction = float(busy) / max(1, len(world.couriers))

        pending = sum(1 for o in world.active_orders.values() if not o.is_terminal)
        pending_norm = float(np.clip(pending / max_pending, 0.0, 1.0))

        scalars = np.array(
            [failed_rate, busy_fraction, pending_norm],
            dtype=np.float32,
        )
        return np.concatenate([coverage, scalars]).astype(np.float32)

    def bounds(self, n_stores: int) -> tuple[np.ndarray, np.ndarray]:
        """Return ([0]*n+3, [1]*n+3) Box bounds."""
        n = n_stores + 3
        return np.zeros(n, dtype=np.float32), np.ones(n, dtype=np.float32)
