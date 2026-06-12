"""
DemandGenerator ABC and built-in implementations.

Layer: Domain Entities.

Three concrete generators are provided:
  PoissonDemandGenerator  — stationary homogeneous Poisson (memoryless)
  DailyProfileDemandGenerator — piecewise-linear time-varying rate scaled by a
                                 breakpoint profile over the episode horizon
  BurstDemandGenerator    — periodic on/off bursts with a configurable multiplier

Time-varying generators (DailyProfile, Burst) require ``horizon > 0`` at
``reset()`` time (V-3 reproducibility verification).  Callers must pass
``horizon=<episode_seconds>``; raising ValueError on ``horizon == 0`` prevents
the silent "flat profile" bug that occurs when horizon is forgotten.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np

from delivery_sim.registry import register

if TYPE_CHECKING:
    from delivery_sim.config.schema import ProfileBreakpoint


class DemandGenerator(ABC):
    """Generates customer order arrivals each simulation tick.

    Implementations must be stateless (or reset-able) to guarantee
    reproducibility when re-seeded with the same RNG state.
    """

    @abstractmethod
    def generate(
        self, sim_time: float, rng: np.random.Generator
    ) -> list[dict[str, Any]]:
        """Return a list of new order specs (dicts) for the current tick.

        Each dict must contain at least: ``store_id``, ``customer_x``,
        ``customer_y``.  The engine stamps the ``order_id`` and ``created_at``
        fields before constructing ``Order`` objects.
        """
        raise NotImplementedError

    @abstractmethod
    def next_event(
        self, sim_time: float, rng: np.random.Generator
    ) -> tuple[float, dict[str, Any]] | None:
        """Return ``(absolute_arrival_time, order_attrs)`` for the next demand event.

        ``order_attrs`` contains ``customer_x`` and ``customer_y`` only — no
        ``store_id``.  Store assignment is deliberately omitted: the Simulator's
        dispatch logic must call ``Store.covers()`` to find a covering store,
        keeping ``coverage_radius`` as a live control variable for RL agents.
        An uncovered customer results in a FAILED order, not a random fallback.

        Draw order (FIXED — reordering breaks seed-compatibility of saved results):
          1. inter-arrival delay
          2. ``customer_x``
          3. ``customer_y``

        This method is the **sole consumer** of the *rng* argument for demand
        scheduling and placement.  The Simulator must pass a dedicated child RNG
        stream (one of the ``SeedSequence.spawn(N)`` children) and must not share
        it with any other consumer.

        Horizon scheduling convention (caller's responsibility): schedule the
        returned event only when ``arrival_time < horizon`` (strict less-than).
        This is consistent with the run-loop termination condition
        ``sim_time >= horizon``; an event at exactly ``horizon`` must not be
        scheduled, because the loop would exit before processing it, silently
        dropping the order.

        Returns ``None`` when the process produces no events (e.g. rate == 0).
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self, rng: np.random.Generator, *, horizon: float = 0.0) -> None:
        """Reset any internal state so the generator can be re-run reproducibly.

        Args:
            rng:     Seeded RNG provided by the engine (child of episode seed).
            horizon: Episode duration in sim-seconds.  Time-varying generators
                     MUST validate ``horizon > 0`` and raise ``ValueError``
                     when it is not, because their intensity profile is
                     undefined without a finite horizon reference (V-3).
                     Stationary generators (e.g. Poisson) may ignore this.
        """
        raise NotImplementedError

    @abstractmethod
    def current_intensity(self, sim_time: float) -> float:
        """Return the instantaneous demand intensity multiplier at *sim_time*.

        The multiplier is relative to the base rate: 1.0 = base rate, 2.0 =
        double rate, etc.  Used by WorldSnapshot (B9) to expose the current
        demand level to RL agents.
        """
        raise NotImplementedError


@register("demand_generator")
class PoissonDemandGenerator(DemandGenerator):
    """Homogeneous Poisson arrival process with uniformly distributed locations."""

    def __init__(
        self,
        rate: float,
        dt: float,
        world_width: float = 1000.0,
        world_height: float = 1000.0,
        store_ids: list[str] | None = None,
        intensity: float = 1.0,
        **_ignored: Any,
    ) -> None:
        self.rate = rate
        self.intensity = intensity
        self.dt = dt
        self.world_width = world_width
        self.world_height = world_height
        self.store_ids: list[str] = store_ids or []

    def generate(
        self, sim_time: float, rng: np.random.Generator
    ) -> list[dict[str, Any]]:
        """Sample n ~ Poisson(rate * dt) orders for the current tick.

        Customer locations are drawn uniformly from [0, world_width) ×
        [0, world_height).  A store is chosen uniformly at random from
        ``store_ids``; if ``store_ids`` is empty, ``store_id`` is set to ``""``.

        Reproducibility guarantee: all draws come exclusively from the *rng*
        argument supplied by the caller.  This class holds no RNG state — the
        same (config, rng-seed) pair always produces an identical demand stream.
        """
        if self.rate == 0.0:
            return []
        n: int = int(rng.poisson(self.rate * self.dt))
        orders: list[dict[str, Any]] = []
        for _ in range(n):
            cx = float(rng.uniform(0.0, self.world_width))
            cy = float(rng.uniform(0.0, self.world_height))
            if self.store_ids:
                store_id = str(rng.choice(self.store_ids))
            else:
                store_id = ""
            orders.append({"store_id": store_id, "customer_x": cx, "customer_y": cy})
        return orders

    def next_event(
        self, sim_time: float, rng: np.random.Generator
    ) -> tuple[float, dict[str, Any]] | None:
        """Return ``(arrival_time, order_attrs)`` for the next Poisson arrival.

        Returns ``None`` when ``rate == 0`` without consuming any RNG state.

        Draw order (fixed — do not reorder):
          1. delay ~ Exponential(1 / rate)
          2. customer_x ~ Uniform(0, world_width)
          3. customer_y ~ Uniform(0, world_height)
        """
        if self.rate == 0.0:
            return None
        delay = float(rng.exponential(1.0 / self.rate))
        cx = float(rng.uniform(0.0, self.world_width))
        cy = float(rng.uniform(0.0, self.world_height))
        return (sim_time + delay, {"customer_x": cx, "customer_y": cy})

    def reset(self, rng: np.random.Generator, *, horizon: float = 0.0) -> None:  # noqa: ARG002
        """No-op: the memoryless Poisson process has no accumulated state."""

    def current_intensity(self, sim_time: float) -> float:  # noqa: ARG002
        """Constant intensity — always 1.0."""
        return 1.0


@register("demand_generator")
class DailyProfileDemandGenerator(DemandGenerator):
    """Time-varying Poisson process with a piecewise-linear intensity profile.

    The profile is a list of (time_fraction, rate_factor) breakpoints that
    define how the effective arrival rate changes over the episode.  Between
    breakpoints the rate_factor is linearly interpolated.

    Effective rate at time t = base_rate * intensity * profile_factor(t/horizon)

    Requires ``horizon > 0`` at reset() time (V-3).
    """

    def __init__(
        self,
        rate: float,
        dt: float,
        world_width: float = 1000.0,
        world_height: float = 1000.0,
        store_ids: list[str] | None = None,
        profile: list[ProfileBreakpoint] | None = None,
        intensity: float = 1.0,
        **_ignored: Any,
    ) -> None:
        self.rate = rate
        self.intensity = intensity
        self.dt = dt
        self.world_width = world_width
        self.world_height = world_height
        self.store_ids: list[str] = store_ids or []
        # Sort breakpoints by time_fraction once at construction
        raw = profile or []
        self._profile: list[tuple[float, float]] = sorted(
            ((bp.time_fraction, bp.rate_factor) for bp in raw),
            key=lambda x: x[0],
        )
        self._horizon: float = 0.0

    def reset(self, rng: np.random.Generator, *, horizon: float = 0.0) -> None:  # noqa: ARG002
        if horizon <= 0.0:
            raise ValueError(
                "DailyProfileDemandGenerator requires horizon > 0 — "
                f"got {horizon!r}.  Pass horizon=<episode_seconds> to reset()."
            )
        self._horizon = horizon

    def current_intensity(self, sim_time: float) -> float:
        """Piecewise-linearly interpolated profile factor × intensity."""
        if not self._profile or self._horizon <= 0.0:
            return self.intensity
        tf = max(0.0, min(1.0, sim_time / self._horizon))
        if tf <= self._profile[0][0]:
            return self.intensity * self._profile[0][1]
        if tf >= self._profile[-1][0]:
            return self.intensity * self._profile[-1][1]
        for i in range(len(self._profile) - 1):
            t0, f0 = self._profile[i]
            t1, f1 = self._profile[i + 1]
            if t0 <= tf <= t1:
                alpha = (tf - t0) / (t1 - t0)
                return self.intensity * (f0 + alpha * (f1 - f0))
        return self.intensity  # unreachable; guards above are exhaustive

    def next_event(
        self, sim_time: float, rng: np.random.Generator
    ) -> tuple[float, dict[str, Any]] | None:
        effective_rate = self.rate * self.current_intensity(sim_time)
        if effective_rate == 0.0:
            return None
        delay = float(rng.exponential(1.0 / effective_rate))
        cx = float(rng.uniform(0.0, self.world_width))
        cy = float(rng.uniform(0.0, self.world_height))
        return (sim_time + delay, {"customer_x": cx, "customer_y": cy})

    def generate(
        self, sim_time: float, rng: np.random.Generator
    ) -> list[dict[str, Any]]:
        effective_rate = self.rate * self.current_intensity(sim_time)
        if effective_rate == 0.0:
            return []
        n = int(rng.poisson(effective_rate * self.dt))
        orders: list[dict[str, Any]] = []
        for _ in range(n):
            cx = float(rng.uniform(0.0, self.world_width))
            cy = float(rng.uniform(0.0, self.world_height))
            orders.append({"customer_x": cx, "customer_y": cy})
        return orders


@register("demand_generator")
class BurstDemandGenerator(DemandGenerator):
    """Periodic burst demand: alternating high-rate (burst) and normal-rate (quiet) phases.

    Cycle structure (repeated from t=0):
      [0, burst_len)              → effective_rate = base_rate * intensity * burst_rate_factor
      [burst_len, cycle_length)   → effective_rate = base_rate * intensity

    where:
      burst_len    = burst_duration_fraction * horizon
      cycle_length = (burst_duration_fraction + burst_interval_fraction) * horizon

    Requires ``horizon > 0`` at reset() time (V-3).
    """

    def __init__(
        self,
        rate: float,
        dt: float,
        world_width: float = 1000.0,
        world_height: float = 1000.0,
        store_ids: list[str] | None = None,
        burst_rate_factor: float = 5.0,
        burst_duration_fraction: float = 0.1,
        burst_interval_fraction: float = 0.3,
        intensity: float = 1.0,
        **_ignored: Any,
    ) -> None:
        self.rate = rate
        self.intensity = intensity
        self.dt = dt
        self.world_width = world_width
        self.world_height = world_height
        self.store_ids: list[str] = store_ids or []
        self.burst_rate_factor = burst_rate_factor
        self.burst_duration_fraction = burst_duration_fraction
        self.burst_interval_fraction = burst_interval_fraction
        self._horizon: float = 0.0

    def reset(self, rng: np.random.Generator, *, horizon: float = 0.0) -> None:  # noqa: ARG002
        if horizon <= 0.0:
            raise ValueError(
                "BurstDemandGenerator requires horizon > 0 — "
                f"got {horizon!r}.  Pass horizon=<episode_seconds> to reset()."
            )
        self._horizon = horizon

    def current_intensity(self, sim_time: float) -> float:
        """Return burst_rate_factor during burst phase, 1.0 during quiet phase."""
        if self._horizon <= 0.0:
            return self.intensity
        cycle_len = (self.burst_duration_fraction + self.burst_interval_fraction) * self._horizon
        if cycle_len <= 0.0:
            return self.intensity
        burst_len = self.burst_duration_fraction * self._horizon
        pos = sim_time % cycle_len
        if pos < burst_len:
            return self.intensity * self.burst_rate_factor
        return self.intensity

    def next_event(
        self, sim_time: float, rng: np.random.Generator
    ) -> tuple[float, dict[str, Any]] | None:
        effective_rate = self.rate * self.current_intensity(sim_time)
        if effective_rate == 0.0:
            return None
        delay = float(rng.exponential(1.0 / effective_rate))
        cx = float(rng.uniform(0.0, self.world_width))
        cy = float(rng.uniform(0.0, self.world_height))
        return (sim_time + delay, {"customer_x": cx, "customer_y": cy})

    def generate(
        self, sim_time: float, rng: np.random.Generator
    ) -> list[dict[str, Any]]:
        effective_rate = self.rate * self.current_intensity(sim_time)
        if effective_rate == 0.0:
            return []
        n = int(rng.poisson(effective_rate * self.dt))
        orders: list[dict[str, Any]] = []
        for _ in range(n):
            cx = float(rng.uniform(0.0, self.world_width))
            cy = float(rng.uniform(0.0, self.world_height))
            orders.append({"customer_x": cx, "customer_y": cy})
        return orders
