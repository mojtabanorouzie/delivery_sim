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
        # Store queue tracking (B5)
        self._store_queue_entries: dict[str, float] = {}  # order_id → queued_at
        self._store_wait_times: list[float] = []
        self._max_store_queue_depth: int = 0
        # Return tracking (B7)
        self._returned_orders: int = 0
        self._return_total_cost: float = 0.0
        # order_id → sim_time when the courier arrived customer (start of return leg)
        self._return_leg_starts: dict[str, float] = {}
        self._return_leg_times: list[float] = []

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

    def on_order_returned(
        self,
        order: Order,
        sim_time: float,
        cost: float,
    ) -> None:
        """Record an order refused at the customer's door (RETURNED terminal state).

        Records the return event and stamps *sim_time* as the start of the
        return leg so on_courier_returned_to_store can compute leg duration.
        """
        self._returned_orders += 1
        self._return_total_cost += cost
        # Stamp return-leg start time keyed by order_id for later completion.
        self._return_leg_starts[order.order_id] = sim_time

    def on_courier_returned_to_store(
        self,
        courier_id: str,  # noqa: ARG002
        order_id: str,
        sim_time: float,
    ) -> None:
        """Record the moment a returning courier arrives back at the store.

        Computes the return-leg travel time from the stamp written in
        on_order_returned and appends it to _return_leg_times.
        """
        start = self._return_leg_starts.pop(order_id, None)
        if start is not None:
            self._return_leg_times.append(sim_time - start)

    # ------------------------------------------------------------------
    # Store queue notifications (B5)
    # ------------------------------------------------------------------

    def on_order_queued_at_store(
        self,
        order_id: str,
        store_id: str,  # noqa: ARG002
        sim_time: float,
        queued: bool = False,
    ) -> None:
        """Called when a courier arrives at a store.

        queued=True  → courier entered the overflow queue (slot was full).
        queued=False → courier went straight to a prep slot (no wait).
        Only queued=True entries contribute to wait-time and depth metrics.
        """
        if not queued:
            return
        self._store_queue_entries[order_id] = sim_time
        depth = len(self._store_queue_entries)
        if depth > self._max_store_queue_depth:
            self._max_store_queue_depth = depth

    def on_order_dequeued_from_store(
        self,
        order_id: str,
        store_id: str,  # noqa: ARG002
        wait_time: float,
    ) -> None:
        """Called when a queued courier is dequeued and prep starts."""
        self._store_queue_entries.pop(order_id, None)
        self._store_wait_times.append(wait_time)

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

        terminal = self._delivered_orders + self._failed_orders + self._returned_orders
        return {
            "total_orders": self._total_orders,
            "delivered_orders": self._delivered_orders,
            "failed_orders": self._failed_orders,
            "returned_orders": self._returned_orders,
            "delivery_rate": (
                self._delivered_orders / self._total_orders
                if self._total_orders > 0
                else 0.0
            ),
            "return_rate": (
                self._returned_orders / terminal
                if terminal > 0
                else 0.0
            ),
            "mean_delivery_time": mean_dt,
            "p50_delivery_time": p50_dt,
            "p95_delivery_time": p95_dt,
            "mean_pickup_latency": mean_pl,
            "courier_utilization": self._courier_utilization,
            "total_delivery_cost": self._total_delivery_cost,
            "sla_violations": self._sla_violations,
            # Store queue metrics (B5)
            "mean_store_wait_time": (
                float(np.mean(np.asarray(self._store_wait_times, dtype=float)))
                if self._store_wait_times else 0.0
            ),
            "max_store_queue_depth": self._max_store_queue_depth,
            # Return metrics (B7)
            "mean_return_leg_time": (
                float(np.mean(np.asarray(self._return_leg_times, dtype=float)))
                if self._return_leg_times else 0.0
            ),
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
        # Store queue (B5)
        self._store_queue_entries = {}
        self._store_wait_times = []
        self._max_store_queue_depth = 0
        # Returns (B7)
        self._returned_orders = 0
        self._return_total_cost = 0.0
        self._return_leg_starts = {}
        self._return_leg_times = []


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
