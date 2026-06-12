"""
Snapshot dataclasses and SnapshotConsumer protocol.

Layer: Visualization (pure consumer; nothing here imports from the engine).

``WorldSnapshot`` is the only data contract between the engine and any
renderer.  Renderers depend on this module; the engine depends on it only
through TYPE_CHECKING to produce snapshots.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class StoreSnapshot:
    """Immutable view of a single store at one tick."""

    store_id: str
    x: float
    y: float
    coverage_radius: float
    queue_depth: int = 0


@dataclass(frozen=True)
class CourierSnapshot:
    """Immutable view of a single courier at one tick."""

    courier_id: str
    x: float
    y: float
    status: str


@dataclass(frozen=True)
class OrderSnapshot:
    """Immutable view of a single order at one tick."""

    order_id: str
    status: str
    customer_x: float
    customer_y: float
    assigned_courier_id: str | None = None


@dataclass(frozen=True)
class WorldSnapshot:
    """Complete immutable view of world state at one tick.

    Produced by ``WorldState.snapshot()`` and consumed by any renderer or
    metrics collector that wants a point-in-time picture.
    """

    tick: int
    elapsed: float
    stores: tuple[StoreSnapshot, ...]
    couriers: tuple[CourierSnapshot, ...]
    orders: tuple[OrderSnapshot, ...]
    scenario_name: str = ""
    demand_intensity: float = 0.0
    demand_pattern: str = ""   # generator_type string, e.g. "PoissonDemandGenerator"


@runtime_checkable
class SnapshotConsumer(Protocol):
    """Interface that all renderers and passive observers must implement."""

    def consume(self, snapshot: WorldSnapshot) -> None:
        """Process (render / record) a single world snapshot."""
        ...

    def close(self) -> None:
        """Release any held resources (windows, file handles, sockets)."""
        ...
