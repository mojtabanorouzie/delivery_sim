"""
EventQueue — min-heap priority queue for simulation events.

Layer: Simulation Engine.

Events are ordered by (time, priority, seq); lower priority integer = higher
urgency at the same timestamp.  *seq* is a monotonically increasing counter
assigned at construction time, guaranteeing a total order — two events that
share (time, priority) are popped in FIFO (insertion) order.  This prevents
heapq from ever reaching event_type or payload in its comparison, which would
be non-deterministic for dict payloads.
"""

from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass, field
from typing import Any

_event_counter = itertools.count()


@dataclass(order=True)
class Event:
    """A scheduled simulation event.

    Comparison key: (time, priority, seq).  *event_type* and *payload* are
    excluded from comparison and interpreted solely by the Simulator.
    """

    time: float
    priority: int
    seq: int = field(default_factory=lambda: next(_event_counter))
    event_type: str = field(compare=False, default="")
    payload: Any = field(compare=False, default=None)


class EventQueue:
    """Min-heap priority queue for ``Event`` objects.

    The Simulator pushes future events here and pops them in time order
    during each tick.  Thread-safety is not guaranteed; the engine is
    single-threaded.
    """

    def __init__(self) -> None:
        self._heap: list[Event] = []

    def push(self, event: Event) -> None:
        """Schedule *event* into the queue."""
        heapq.heappush(self._heap, event)

    def pop(self) -> Event:
        """Remove and return the earliest event.

        Raises ``IndexError`` if the queue is empty.
        """
        return heapq.heappop(self._heap)

    def peek(self) -> Event | None:
        """Return the earliest event without removing it, or ``None`` if empty."""
        return self._heap[0] if self._heap else None

    def is_empty(self) -> bool:
        """Return True if no events are scheduled."""
        return len(self._heap) == 0

    def clear(self) -> None:
        """Discard all scheduled events."""
        self._heap.clear()

    def __len__(self) -> int:
        return len(self._heap)
