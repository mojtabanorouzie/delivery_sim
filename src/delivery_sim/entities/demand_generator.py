"""
DemandGenerator ABC and built-in Poisson implementation.

Layer: Domain Entities.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from delivery_sim.registry import register


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
    def reset(self, rng: np.random.Generator) -> None:
        """Reset any internal state so the generator can be re-run reproducibly."""
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
    ) -> None:
        self.rate = rate
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

    def reset(self, rng: np.random.Generator) -> None:
        """No-op: the memoryless Poisson process has no accumulated state."""
