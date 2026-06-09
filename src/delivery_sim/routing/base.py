"""
RoutingModel ABC — spatial distance and path interface.

Layer: Domain Entities.

Swap implementations (e.g. Euclidean → road-graph) by registering a new
subclass and changing ``routing.model_type`` in the scenario YAML.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class RoutingModel(ABC):
    """Computes distances, travel times, and waypoint routes between positions.

    All spatial queries in the simulation go through this interface so that
    the underlying geometry can be swapped without touching the engine.
    """

    @abstractmethod
    def distance(self, x1: float, y1: float, x2: float, y2: float) -> float:
        """Return the travel distance (in world units) between two points."""
        raise NotImplementedError

    @abstractmethod
    def travel_time(
        self, x1: float, y1: float, x2: float, y2: float, speed: float
    ) -> float:
        """Return expected travel time in seconds given *speed* world-units/s."""
        raise NotImplementedError

    @abstractmethod
    def route(
        self, x1: float, y1: float, x2: float, y2: float
    ) -> list[tuple[float, float]]:
        """Return an ordered list of (x, y) waypoints from (x1,y1) to (x2,y2).

        The first element is always (x1, y1) and the last is always (x2, y2).
        """
        raise NotImplementedError
