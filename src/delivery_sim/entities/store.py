"""
Store ABC and built-in implementation.

Layer: Domain Entities.

Design note — RoutingModel injection asymmetry:
    ``Store.covers()`` receives the RoutingModel as a *per-call* argument.
    Stores are stateless with respect to geometry; the world's routing model is
    owned by the engine and shared across all entities.  There is no reason for
    a store to hold a permanent reference to it.  Contrast with ``BikeCourier``,
    which stores the RoutingModel at construction because movement is a
    continuous per-tick operation and the model is stable for the courier's
    lifetime.  The asymmetry is intentional — do not "fix" one to match the
    other.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque

from delivery_sim.registry import register
from delivery_sim.routing.base import RoutingModel


class Store(ABC):
    """Abstract base for all store implementations.

    A store accepts orders, prepares them, and makes them available for
    courier pick-up. Concrete subclasses must be registered via
    ``@register("store")`` so the engine can instantiate them by name.
    """

    @property
    @abstractmethod
    def store_id(self) -> str:
        """Unique identifier for this store."""
        raise NotImplementedError

    @property
    @abstractmethod
    def x(self) -> float:
        """X coordinate in the world."""
        raise NotImplementedError

    @property
    @abstractmethod
    def y(self) -> float:
        """Y coordinate in the world."""
        raise NotImplementedError

    @property
    @abstractmethod
    def coverage_radius(self) -> float:
        """Current service radius in world-units. Non-negative by convention."""
        raise NotImplementedError

    @coverage_radius.setter
    @abstractmethod
    def coverage_radius(self, value: float) -> None:
        raise NotImplementedError

    @abstractmethod
    def covers(self, point_x: float, point_y: float, routing: RoutingModel) -> bool:
        """Return True iff (point_x, point_y) lies within this store's coverage area.

        The boundary is inclusive: a point at exactly ``coverage_radius``
        routing-distance from the store is considered covered.

        *routing* is passed per-call; see module-level design note.
        """
        raise NotImplementedError

    @abstractmethod
    def can_prepare(self, order_id: str) -> bool:
        """Return True if this store can currently accept and prepare *order_id*."""
        raise NotImplementedError

    @abstractmethod
    def start_preparation(self, order_id: str, sim_time: float) -> float:
        """Begin preparing *order_id*; return estimated pick-up time."""
        raise NotImplementedError

    @abstractmethod
    def complete_preparation(self, order_id: str) -> None:
        """Free the prep slot held by *order_id*.

        Called by the Simulator in _handle_order_ready immediately before
        dequeue_next_waiter(), so the freed slot is available to the next waiter.
        Idempotent: a missing order_id is silently ignored.
        """
        raise NotImplementedError

    @abstractmethod
    def enqueue_waiter(
        self, courier_id: str, order_id: str, arrived_at: float
    ) -> None:
        """Add a courier to the overflow queue because all prep slots were full.

        Each entry is stamped with a monotonic enqueue_seq counter so that
        same-sim_time arrivals always dequeue in a fully deterministic order
        (FIFO by counter, not by Python dict/set iteration order).
        """
        raise NotImplementedError

    @abstractmethod
    def dequeue_next_waiter(self) -> tuple[str, str, float] | None:
        """Pop the next waiter and return (courier_id, order_id, arrived_at).

        Returns None when the queue is empty.  Called only after
        complete_preparation() has freed a slot, so can_prepare() is True
        immediately after this call returns a non-None value.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def queue_depth(self) -> int:
        """Number of couriers currently waiting for a prep slot."""
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        """Reset all preparation state for a new episode."""
        raise NotImplementedError


@register("store")
class BuiltinStore(Store):
    """Simple fixed-position store with configurable order capacity and coverage.

    ``coverage_radius`` is a public, mutable float attribute.  An RL agent can
    write to it at any point during an episode to tune the store's service area
    (e.g. expanding radius to capture more demand, shrinking to reduce cost).

    Convention: ``coverage_radius`` should be non-negative.  Setting it to a
    negative value is defined — ``covers()`` will return ``False`` for every
    query, including the store's own location, because
    ``routing.distance(...) >= 0 > negative_radius`` — but it is not a
    meaningful operational state.  The config schema enforces ``>= 0`` at
    parse time; runtime mutations are the caller's responsibility.

    Queue determinism: _waiting holds 4-tuples (enqueue_seq, courier_id,
    order_id, arrived_at).  enqueue_seq is a per-store monotonic counter so
    same-sim_time arrivals always dequeue in insertion order regardless of
    courier_id sort order or event-queue internals.
    """

    def __init__(
        self,
        store_id: str,
        x: float,
        y: float,
        capacity: int = 10,
        prep_time: float = 30.0,
        coverage_radius: float = 500.0,
    ) -> None:
        self._store_id = store_id
        self._x = x
        self._y = y
        self.capacity = capacity
        self.prep_time = prep_time
        self._coverage_radius: float = coverage_radius
        self._active_orders: dict[str, float] = {}
        # Each entry: (enqueue_seq, courier_id, order_id, arrived_at)
        self._waiting: deque[tuple[int, str, str, float]] = deque()
        self._enqueue_seq: int = 0

    @property
    def store_id(self) -> str:
        return self._store_id

    @property
    def x(self) -> float:
        return self._x

    @property
    def y(self) -> float:
        return self._y

    @property
    def coverage_radius(self) -> float:
        return self._coverage_radius

    @coverage_radius.setter
    def coverage_radius(self, value: float) -> None:
        self._coverage_radius = value

    def covers(self, point_x: float, point_y: float, routing: RoutingModel) -> bool:
        """Return True iff routing distance from store to (point_x, point_y) is
        <= coverage_radius (inclusive boundary).

        Returns False for all inputs when coverage_radius is negative.
        """
        return routing.distance(self._x, self._y, point_x, point_y) <= self._coverage_radius

    def can_prepare(self, order_id: str) -> bool:
        """Return True iff active-order count is strictly below capacity."""
        return len(self._active_orders) < self.capacity

    def start_preparation(self, order_id: str, sim_time: float) -> float:
        """Record *order_id* as active and return its estimated pick-up time.

        Returns ``sim_time + prep_time``.  Callers must check ``can_prepare``
        first; this method does not enforce the capacity limit.
        """
        pickup_time = sim_time + self.prep_time
        self._active_orders[order_id] = pickup_time
        return pickup_time

    def complete_preparation(self, order_id: str) -> None:
        """Free the prep slot for *order_id*.  Idempotent."""
        self._active_orders.pop(order_id, None)

    def enqueue_waiter(
        self, courier_id: str, order_id: str, arrived_at: float
    ) -> None:
        """Append to the FIFO overflow queue with a deterministic sequence stamp."""
        self._waiting.append((self._enqueue_seq, courier_id, order_id, arrived_at))
        self._enqueue_seq += 1

    def dequeue_next_waiter(self) -> tuple[str, str, float] | None:
        """Pop the earliest waiter; returns (courier_id, order_id, arrived_at) or None."""
        if not self._waiting:
            return None
        _seq, courier_id, order_id, arrived_at = self._waiting.popleft()
        return courier_id, order_id, arrived_at

    @property
    def queue_depth(self) -> int:
        return len(self._waiting)

    def reset(self) -> None:
        """Clear all in-progress and waiting state for a fresh episode."""
        self._active_orders.clear()
        self._waiting.clear()
        self._enqueue_seq = 0
