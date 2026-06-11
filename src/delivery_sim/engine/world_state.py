"""
WorldState — mutable live simulation state and snapshot factory.

Layer: Simulation Engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from delivery_sim.entities.courier import Courier
    from delivery_sim.entities.order import Order
    from delivery_sim.entities.store import Store
    from delivery_sim.render.protocol import WorldSnapshot


@dataclass
class WorldState:
    """Mutable container for all live entity state, owned by ``Simulator``.

    Access is always single-threaded (same thread as the simulation loop).
    Use ``snapshot()`` to get an immutable copy for rendering or metrics.

    ``active_orders`` holds every order created during the episode, including
    terminal (DELIVERED / FAILED) orders, so that full transition histories
    remain available for metrics and state-machine audits after the run.

    ``courier_phase`` is the Simulator-owned phase map; valid values are
    ``"free"``, ``"en-route-store"``, ``"at-store"``, ``"en-route-customer"``.
    It is distinct from ``courier.status`` (which the Simulator never reads for
    phase decisions).

    ``store_index`` and ``courier_index`` are O(1) lookup caches kept in sync
    with the ``stores`` / ``couriers`` lists by the Simulator.
    """

    width: float
    height: float
    stores: list[Store] = field(default_factory=list)
    couriers: list[Courier] = field(default_factory=list)
    active_orders: dict[str, Order] = field(default_factory=dict)
    courier_phase: dict[str, str] = field(default_factory=dict)
    store_index: dict[str, Store] = field(default_factory=dict, repr=False)
    courier_index: dict[str, Courier] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if not self.store_index:
            self.store_index = {s.store_id: s for s in self.stores}
        if not self.courier_index:
            self.courier_index = {c.courier_id: c for c in self.couriers}

    def snapshot(self, tick: int, elapsed: float) -> WorldSnapshot:
        """Produce an immutable ``WorldSnapshot`` at simulation time *elapsed*.

        Courier positions come from ``position_at(elapsed)`` (read-only analytic
        query); courier status strings come from ``courier_phase`` (not from
        ``courier.status``).  All tuples are sorted by entity id for stable
        ordering.  No mutable state is leaked or modified.
        """
        from delivery_sim.render.protocol import (
            CourierSnapshot,
            OrderSnapshot,
            StoreSnapshot,
            WorldSnapshot,
        )

        store_snaps = tuple(
            StoreSnapshot(store_id=s.store_id, x=s.x, y=s.y, coverage_radius=s.coverage_radius)
            for s in sorted(self.stores, key=lambda s: s.store_id)
        )
        courier_snaps = tuple(
            CourierSnapshot(
                courier_id=c.courier_id,
                x=c.position_at(elapsed)[0],
                y=c.position_at(elapsed)[1],
                status=self.courier_phase.get(c.courier_id, "free"),
            )
            for c in sorted(self.couriers, key=lambda c: c.courier_id)
        )
        order_snaps = tuple(
            OrderSnapshot(
                order_id=o.order_id,
                status=o.status.name,
                customer_x=o.customer_x,
                customer_y=o.customer_y,
                assigned_courier_id=o.assigned_courier_id,
            )
            for o in sorted(self.active_orders.values(), key=lambda o: o.order_id)
        )
        return WorldSnapshot(
            tick=tick,
            elapsed=elapsed,
            stores=store_snaps,
            couriers=courier_snaps,
            orders=order_snaps,
        )
