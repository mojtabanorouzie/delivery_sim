"""
Courier / DeliveryProvider ABC and built-in implementations.

Layer: Domain Entities.

Design note — RoutingModel injection asymmetry:
    ``BikeCourier`` stores the RoutingModel at construction because trajectory
    computation happens at assignment time and the routing model is stable for
    the courier's lifetime.  Contrast with ``Store.covers()``, which receives
    the RoutingModel as a per-call argument because coverage queries are
    sporadic and the store itself is geometry-stateless.  The asymmetry is
    intentional — do not "fix" one to match the other.

ADR-002: couriers are time-parameterised trajectories.  Each leg is established
by ``assign(…, from_x, from_y)``; the Simulator supplies both endpoints
explicitly.  ``_origin_x/_origin_y`` is the idle/spawn position after episode
``reset()`` only; it is never read during trajectory math.  Multi-leg journeys
(origin→store→customer) are orchestrated by the Simulator: one ``assign()``
call per leg, with the leg-start passed as ``from_x/from_y``.  ``reset()`` is
reserved for episode boundaries; it is never called between legs mid-episode.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum, auto

from delivery_sim.registry import register
from delivery_sim.routing.base import RoutingModel


class CourierStatus(Enum):
    """Operational states of a courier."""

    IDLE = auto()
    DISPATCHED = auto()
    AT_STORE = auto()
    DELIVERING = auto()
    UNAVAILABLE = auto()


class Courier(ABC):
    """Abstract base for all courier (DeliveryProvider) implementations.

    Concrete subclasses must be registered via ``@register("courier")`` so
    the engine can build fleets from YAML config.

    Movement contract (ADR-002):
    - ``assign()`` establishes a fully-determined leg from an explicit start.
    - ``position_at(t)`` queries position at any absolute sim time *t*.
    - ``arrival_time()`` returns the pre-computed arrival timestamp.
    - There is no ``step()`` method; couriers do not mutate per tick.
    - ``reset()`` is the episode-boundary operation only; leg sequencing is
      handled by the Simulator supplying ``from_x/from_y`` to each ``assign()``.
    """

    @property
    @abstractmethod
    def courier_id(self) -> str:
        """Unique identifier for this courier."""
        raise NotImplementedError

    @property
    @abstractmethod
    def status(self) -> CourierStatus:
        """Current operational status."""
        raise NotImplementedError

    @property
    @abstractmethod
    def speed(self) -> float:
        """Maximum travel speed in world-units per second."""
        raise NotImplementedError

    @property
    @abstractmethod
    def capacity(self) -> int:
        """Maximum number of packages this courier can carry simultaneously."""
        raise NotImplementedError

    @abstractmethod
    def cost(self, distance: float) -> float:
        """Return the operational cost of travelling *distance* world-units."""
        raise NotImplementedError

    @abstractmethod
    def position_at(self, t: float) -> tuple[float, float]:
        """Return (x, y) position at absolute simulation time *t*.

        - No active trajectory (idle after reset): returns spawn position.
        - t <= assign_time: returns the leg-start (from_x, from_y).
        - t >= arrival_time: returns the target, clamped (no overshoot).
        - otherwise: linear interpolation at constant speed.
        """
        raise NotImplementedError

    @abstractmethod
    def arrival_time(self) -> float | None:
        """Absolute simulation time of arrival at the current target.

        Returns ``None`` when the courier has no active trajectory (idle after
        episode ``reset()``).  The Simulator schedules a COURIER_ARRIVED event
        at this time; it is the authoritative arrival signal.
        """
        raise NotImplementedError

    @abstractmethod
    def assign(
        self,
        order_id: str,
        store_id: str,
        sim_time: float,
        target_x: float,
        target_y: float,
        from_x: float,
        from_y: float,
    ) -> None:
        """Establish a trajectory leg from *(from_x, from_y)* to *(target_x, target_y)*.

        Precondition — SETTLED: the courier must not be in motion.  Either
        ``arrival_time()`` is ``None`` (no prior leg) or
        ``sim_time >= arrival_time()`` (prior leg has completed).

        Raises ``ValueError`` if ``sim_time < arrival_time()`` (mid-motion
        reassignment); this prevents silent teleportation of a moving courier.

        The Simulator supplies ``from_x/from_y`` explicitly; the courier does
        not read spawn state to determine the leg start.  After this call,
        ``position_at`` and ``arrival_time`` are fully determined.
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self, origin_x: float, origin_y: float) -> None:
        """Episode boundary.  Return courier to *(origin_x, origin_y)* in IDLE state.

        Clears all trajectory state: ``arrival_time()`` returns ``None``,
        ``has_target`` returns ``False``.

        Called at episode start only.  NOT for leg transitions mid-episode —
        those use ``assign()`` with explicit ``from_x/from_y``.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def has_target(self) -> bool:
        """True while this courier has an active trajectory (DISPATCHED).

        This is a **status predicate** for snapshots and render — it signals
        busy vs. idle.  It is NOT the arrival detection path.  Arrival is
        detected by the Simulator scheduling a COURIER_ARRIVED event at
        ``arrival_time()``.  Do not poll ``has_target`` to infer arrival.
        """
        raise NotImplementedError


@register("courier")
class BikeCourier(Courier):
    """Ground courier with a straight-line, constant-speed trajectory per leg.

    Each leg is established by ``assign()``; the Simulator supplies the leg
    start as ``from_x/from_y``.  Swapping to a road-graph routing model at a
    later step requires no changes here.

    See module-level design note on RoutingModel injection asymmetry.
    """

    def __init__(
        self,
        courier_id: str,
        x: float,
        y: float,
        routing: RoutingModel,
        speed: float = 1.5,
        capacity: int = 1,
        cost_per_unit: float = 0.01,
    ) -> None:
        self._courier_id = courier_id
        # _origin_x/_origin_y: idle/spawn position ONLY.  Set at construction
        # and by reset() (episode boundary).  NOT the trajectory start — that
        # is supplied explicitly to assign() as from_x/from_y.  The two
        # concepts are now distinct; do not conflate them.
        self._origin_x = x
        self._origin_y = y
        self._routing = routing
        self._speed = speed
        self._capacity = capacity
        self._cost_per_unit = cost_per_unit
        self._status = CourierStatus.IDLE
        self._current_order_id: str | None = None
        self._target_store_id: str | None = None
        self._assign_time: float | None = None
        self._start_x: float | None = None  # leg start, from from_x at assign()
        self._start_y: float | None = None  # leg start, from from_y at assign()
        self._target_x: float | None = None
        self._target_y: float | None = None
        self._arrival_time_val: float | None = None

    @property
    def courier_id(self) -> str:
        return self._courier_id

    @property
    def status(self) -> CourierStatus:
        return self._status

    @property
    def speed(self) -> float:
        return self._speed

    @property
    def capacity(self) -> int:
        return self._capacity

    def cost(self, distance: float) -> float:
        """Return ``cost_per_unit * distance``."""
        return self._cost_per_unit * distance

    def assign(
        self,
        order_id: str,
        store_id: str,
        sim_time: float,
        target_x: float,
        target_y: float,
        from_x: float,
        from_y: float,
    ) -> None:
        """Establish leg from *(from_x, from_y)* to *(target_x, target_y)*.

        Raises ``ValueError`` if the courier is still in motion
        (``sim_time < arrival_time()``).  The two-leg journey is legal because
        leg 2 is only assigned after the leg-1 COURIER_ARRIVED event fires,
        guaranteeing ``sim_time >= arrival_time()`` at that point.
        """
        if (
            self._arrival_time_val is not None
            and sim_time < self._arrival_time_val
        ):
            raise ValueError(
                f"assign() on courier {self._courier_id!r} at "
                f"sim_time={sim_time!r} while still in motion "
                f"(arrival_time={self._arrival_time_val!r}); "
                f"only settled couriers may be assigned a new leg"
            )
        self._current_order_id = order_id
        self._target_store_id = store_id
        self._assign_time = sim_time
        self._start_x = from_x
        self._start_y = from_y
        self._target_x = target_x
        self._target_y = target_y
        travel = self._routing.travel_time(from_x, from_y, target_x, target_y, self._speed)
        self._arrival_time_val = sim_time + travel
        self._status = CourierStatus.DISPATCHED

    def position_at(self, t: float) -> tuple[float, float]:
        """Return (x, y) at absolute sim time *t* along the active leg.

        Regions:
        - No active leg (idle after episode reset): spawn origin for all *t*.
        - t <= assign_time: leg-start (from_x, from_y).
        - t >= arrival_time: target, clamped (no overshoot).
        - otherwise: linear interpolation at constant speed.
        """
        if self._target_x is None:
            return (self._origin_x, self._origin_y)

        assert self._arrival_time_val is not None
        assert self._assign_time is not None
        assert self._target_y is not None
        assert self._start_x is not None
        assert self._start_y is not None

        if t >= self._arrival_time_val:
            return (self._target_x, self._target_y)
        if t <= self._assign_time:
            return (self._start_x, self._start_y)

        duration = self._arrival_time_val - self._assign_time
        fraction = (t - self._assign_time) / duration
        return (
            self._start_x + fraction * (self._target_x - self._start_x),
            self._start_y + fraction * (self._target_y - self._start_y),
        )

    def arrival_time(self) -> float | None:
        """Return pre-computed arrival timestamp, or None when idle."""
        return self._arrival_time_val

    def reset(self, origin_x: float, origin_y: float) -> None:
        """Episode boundary.  Return to *(origin_x, origin_y)* in IDLE state.

        Clears all trajectory state including leg-start fields.  The routing
        model is retained — it is a world-level dependency, not episode state.
        """
        self._origin_x = origin_x
        self._origin_y = origin_y
        self._status = CourierStatus.IDLE
        self._current_order_id = None
        self._target_store_id = None
        self._assign_time = None
        self._start_x = None
        self._start_y = None
        self._target_x = None
        self._target_y = None
        self._arrival_time_val = None

    @property
    def has_target(self) -> bool:
        """True iff an active trajectory exists; see ``Courier.has_target``."""
        return self._target_x is not None
