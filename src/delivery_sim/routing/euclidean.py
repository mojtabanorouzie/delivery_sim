"""
Straight-line (Euclidean) routing model.

Layer: Domain Entities.

Assumes a continuous 2D plane with no obstacles; distances are L2 norms.
All coordinates are in world units (pixels / metres — the unit is opaque to
this model).
"""

from __future__ import annotations

import numpy as np

from delivery_sim.registry import register
from delivery_sim.routing.base import RoutingModel


@register("routing", name="euclidean")
class EuclideanRouting(RoutingModel):
    """Routing model using straight-line Euclidean distance.

    Assumptions:
    - Continuous 2D plane; no walls, roads, or obstacles.
    - All directions are equally traversable at the given speed.
    - Route is the straight segment from start to end.

    Swap to a road-graph model by registering a new RoutingModel subclass
    and setting ``routing.model_type`` to its registry name in the scenario YAML.
    """

    def distance(self, x1: float, y1: float, x2: float, y2: float) -> float:
        """Return L2 distance between (x1, y1) and (x2, y2).

        Returns 0.0 for identical points.
        """
        return float(np.hypot(x2 - x1, y2 - y1))

    def travel_time(
        self, x1: float, y1: float, x2: float, y2: float, speed: float
    ) -> float:
        """Return distance / speed in seconds.

        Raises ``ValueError`` if *speed* is <= 0.
        """
        if speed <= 0.0:
            raise ValueError(f"speed must be positive, got {speed!r}")
        return self.distance(x1, y1, x2, y2) / speed

    def route(
        self, x1: float, y1: float, x2: float, y2: float
    ) -> list[tuple[float, float]]:
        """Return the two-point straight-line segment [(x1,y1), (x2,y2)].

        For identical start and end the segment is degenerate but the
        contract (first == (x1,y1), last == (x2,y2)) is still satisfied.
        """
        return [(x1, y1), (x2, y2)]
