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
        #: Service radius in world-units. Non-negative by convention; see class docstring.
        self.coverage_radius = coverage_radius
        self._active_orders: dict[str, float] = {}

    @property
    def store_id(self) -> str:
        return self._store_id

    @property
    def x(self) -> float:
        return self._x

    @property
    def y(self) -> float:
        return self._y

    def covers(self, point_x: float, point_y: float, routing: RoutingModel) -> bool:
        """Return True iff routing distance from store to (point_x, point_y) is
        <= coverage_radius (inclusive boundary).

        Returns False for all inputs when coverage_radius is negative.
        """
        return routing.distance(self._x, self._y, point_x, point_y) <= self.coverage_radius

    def can_prepare(self, order_id: str) -> bool:
        """Return True iff active-order count is strictly below capacity.

        The check is inclusive: exactly ``capacity`` concurrent orders are
        allowed; the (capacity+1)-th call returns False.
        """
        return len(self._active_orders) < self.capacity

    def start_preparation(self, order_id: str, sim_time: float) -> float:
        """Record *order_id* as active and return its estimated pick-up time.

        Returns ``sim_time + prep_time``.  Does not guard against exceeding
        capacity — callers should check ``can_prepare`` first.
        """
        pickup_time = sim_time + self.prep_time
        self._active_orders[order_id] = pickup_time
        return pickup_time

    def reset(self) -> None:
        """Clear all in-progress orders for a fresh episode."""
        self._active_orders.clear()
