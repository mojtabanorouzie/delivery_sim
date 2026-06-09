"""
SimClock — event-driven simulation clock (ADR-001).

Layer: Simulation Engine.

The clock advances to arbitrary timestamps via ``advance_to(t)``, driven by
the next scheduled event's time.  ``dt`` is retained as an OPTIONAL observer /
sampling cadence for renderers and metrics collectors — it is NOT the
simulation resolution and is never used to compute the advance step size.
"""

from __future__ import annotations


class SimClock:
    """Tracks simulation time by jumping to event timestamps.

    The canonical advance path is ``advance_to(t)``, where *t* is the time
    of the next event popped from the ``EventQueue``.  Ticks are observers,
    never the source of truth (ADR-001).
    """

    def __init__(self, dt: float = 1.0) -> None:
        """*dt* is the OPTIONAL observer/sampling cadence, not sim resolution."""
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")
        self.dt = dt
        self._time: float = 0.0
        self._tick: int = 0

    @property
    def tick(self) -> int:
        """Number of ``advance_to`` calls since last reset (observer counter)."""
        return self._tick

    @property
    def elapsed(self) -> float:
        """Current simulation time in seconds."""
        return self._time

    def advance_to(self, t: float) -> None:
        """Jump the clock to absolute simulation time *t*.

        *t* must be >= the current ``elapsed`` time; going backwards is a
        caller error (event queue guarantees events are popped in order).
        """
        if t < self._time:
            raise ValueError(
                f"advance_to({t!r}) would move clock backwards from {self._time!r}"
            )
        self._time = t
        self._tick += 1

    def reset(self) -> None:
        """Reset the clock to time zero for a new episode."""
        self._time = 0.0
        self._tick = 0
