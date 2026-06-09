"""
KPICollector — episode-level metrics aggregator.

Layer: Control / RL Interface.

Metrics are collected by observing discrete events/transitions emitted by the
Simulator, NOT by sampling WorldSnapshots.  This guarantees dt-independence:
a KPI value is identical regardless of the observer cadence (dt), because the
collector only advances its state at the exact event timestamps where state
transitions occur.

Usage::

    collector = KPICollector(sla_seconds=1800.0)
    sim.attach_collector(collector)
    sim.run()
    kpis = collector.summary()   # dict of all episode KPIs
    collector.reset()            # clear for next episode
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from delivery_sim.entities.order import Order


class KPICollector:
    """Collects and aggregates delivery KPIs over a single episode.

    Attach to a Simulator via ``Simulator.attach_collector(collector)`` before
    calling ``sim.run()``.  The Simulator calls the ``on_*`` notification
    methods at the exact event timestamps; ``finalize()`` is called once at
    episode end (inside ``run()``).

    After ``finalize()``, ``summary()`` returns a flat dict of all KPIs.
    Call ``reset()`` before reusing for a new episode.

    Empty-episode conventions (zero deliveries / zero couriers / zero horizon):
    - mean_delivery_time, p50_delivery_time, p95_delivery_time = 0.0
    - mean_pickup_latency = 0.0
    - delivery_rate = 0.0
    - courier_utilization = 0.0
    - total_delivery_cost = 0.0
    All denominators are guarded; no ZeroDivisionError is possible.
    """

    def __init__(self, sla_seconds: float = 3600.0) -> None:
        """
        Args:
            sla_seconds: Delivery time threshold; orders exceeding this are
                counted as SLA violations.  Pass ``float('inf')`` to disable.
        """
        self.sla_seconds = sla_seconds
        self._total_orders: int = 0
        self._delivered_orders: int = 0
        self._failed_orders: int = 0
        self._delivery_times: list[float] = []
        self._pickup_latencies: list[float] = []
        self._total_delivery_cost: float = 0.0
        self._sla_violations: int = 0
        # Courier utilisation tracking
        self._courier_busy_since: dict[str, float] = {}
        self._courier_total_busy: float = 0.0
        # Set by finalize()
        self._courier_utilization: float = 0.0

    # ------------------------------------------------------------------
    # Notification methods — called by Simulator at event timestamps
    # ------------------------------------------------------------------

    def on_order_created(self, order: Order, sim_time: float) -> None:  # noqa: ARG002
        """Record a newly created order."""
        self._total_orders += 1

    def on_order_delivered(
        self, order: Order, sim_time: float, cost: float
    ) -> None:
        """Record a successfully delivered order.

        Args:
            order:    The order that just transitioned to DELIVERED.
            sim_time: The exact event timestamp (used as fallback only;
                      timestamps are read from order.timestamps for precision).
            cost:     Total delivery cost (both legs) computed by the Simulator.
        """
        from delivery_sim.entities.order import OrderStatus

        self._delivered_orders += 1
        self._total_delivery_cost += cost

        created_t = order.timestamps.get(OrderStatus.CREATED, 0.0)
        delivered_t = order.timestamps.get(OrderStatus.DELIVERED, sim_time)
        delivery_time = delivered_t - created_t
        self._delivery_times.append(delivery_time)
        if delivery_time > self.sla_seconds:
            self._sla_violations += 1

        picked_up_t = order.timestamps.get(OrderStatus.PICKED_UP)
        if picked_up_t is not None:
            self._pickup_latencies.append(picked_up_t - created_t)

    def on_order_failed(self, order: Order, sim_time: float) -> None:  # noqa: ARG002
        """Record an order that reached the FAILED terminal state."""
        self._failed_orders += 1

    def on_courier_busy(self, courier_id: str, sim_time: float) -> None:
        """Record that *courier_id* transitioned from free → non-free at *sim_time*."""
        self._courier_busy_since[courier_id] = sim_time

    def on_courier_free(self, courier_id: str, sim_time: float) -> None:
        """Record that *courier_id* became free at *sim_time*, closing its interval."""
        busy_start = self._courier_busy_since.pop(courier_id, None)
        if busy_start is not None:
            self._courier_total_busy += sim_time - busy_start

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    def finalize(self, num_couriers: int, horizon: float) -> None:
        """Close any still-open courier intervals and compute utilisation.

        Called by ``Simulator.run()`` once, after the event loop drains.
        Couriers still in a busy phase at *horizon* have their interval
        closed at *horizon*.

        Args:
            num_couriers: Total number of couriers in the episode fleet.
            horizon:      Episode duration (``max_steps * dt``).
        """
        for busy_start in self._courier_busy_since.values():
            self._courier_total_busy += max(0.0, horizon - busy_start)
        self._courier_busy_since.clear()

        denom = num_couriers * horizon
        self._courier_utilization = (
            self._courier_total_busy / denom if denom > 0.0 else 0.0
        )

    def summary(self) -> dict[str, float | int]:
        """Return a flat dict of all episode KPIs.

        May be called before ``finalize()`` (courier_utilization will be 0.0).
        Percentiles over an empty delivery list return 0.0 per the
        empty-episode convention documented on this class.
        """
        delivery_times = self._delivery_times
        if delivery_times:
            arr = np.asarray(delivery_times, dtype=float)
            mean_dt = float(np.mean(arr))
            p50_dt = float(np.percentile(arr, 50))
            p95_dt = float(np.percentile(arr, 95))
        else:
            mean_dt = p50_dt = p95_dt = 0.0

        pickup_lats = self._pickup_latencies
        mean_pl = (
            float(np.mean(np.asarray(pickup_lats, dtype=float)))
            if pickup_lats
            else 0.0
        )

        return {
            "total_orders": self._total_orders,
            "delivered_orders": self._delivered_orders,
            "failed_orders": self._failed_orders,
            "delivery_rate": (
                self._delivered_orders / self._total_orders
                if self._total_orders > 0
                else 0.0
            ),
            "mean_delivery_time": mean_dt,
            "p50_delivery_time": p50_dt,
            "p95_delivery_time": p95_dt,
            "mean_pickup_latency": mean_pl,
            "courier_utilization": self._courier_utilization,
            "total_delivery_cost": self._total_delivery_cost,
            "sla_violations": self._sla_violations,
        }

    def as_dict(self) -> dict[str, float | int]:
        """Alias for ``summary()``."""
        return self.summary()

    def reset(self) -> None:
        """Clear all accumulated state for a new episode."""
        self._total_orders = 0
        self._delivered_orders = 0
        self._failed_orders = 0
        self._delivery_times = []
        self._pickup_latencies = []
        self._total_delivery_cost = 0.0
        self._sla_violations = 0
        self._courier_busy_since = {}
        self._courier_total_busy = 0.0
        self._courier_utilization = 0.0


@dataclass
class EpisodeMetrics:
    """Aggregated KPIs for one complete episode (legacy flat dataclass).

    Prefer ``KPICollector.summary()`` for the full KPI surface.
    """

    total_orders: int = 0
    delivered_orders: int = 0
    failed_orders: int = 0
    total_delivery_time: float = 0.0
    sla_violations: int = 0

    @property
    def delivery_rate(self) -> float:
        """Fraction of orders successfully delivered (0.0 if no orders)."""
        return self.delivered_orders / max(self.total_orders, 1)

    @property
    def mean_delivery_time(self) -> float:
        """Mean delivery time in seconds (0.0 if no deliveries yet)."""
        return self.total_delivery_time / max(self.delivered_orders, 1)
