"""Tests for SimClock — event-driven advance (ADR-001)."""

from __future__ import annotations

import pytest

from delivery_sim.engine.clock import SimClock


class TestSimClockAdvanceTo:
    def test_initial_elapsed_is_zero(self) -> None:
        clock = SimClock()
        assert clock.elapsed == pytest.approx(0.0)

    def test_advance_to_sets_elapsed(self) -> None:
        clock = SimClock()
        clock.advance_to(42.5)
        assert clock.elapsed == pytest.approx(42.5)

    def test_advance_to_arbitrary_jump(self) -> None:
        """Clock can jump by any amount — no fixed dt required."""
        clock = SimClock()
        clock.advance_to(1.0)
        clock.advance_to(1000.0)
        assert clock.elapsed == pytest.approx(1000.0)

    def test_advance_to_same_time_is_allowed(self) -> None:
        """Advancing to the current time is a no-op (not backwards)."""
        clock = SimClock()
        clock.advance_to(5.0)
        clock.advance_to(5.0)  # should not raise
        assert clock.elapsed == pytest.approx(5.0)

    def test_advance_to_backwards_raises(self) -> None:
        clock = SimClock()
        clock.advance_to(10.0)
        with pytest.raises(ValueError, match="backwards"):
            clock.advance_to(9.99)

    def test_tick_counts_advance_to_calls(self) -> None:
        """tick is an observer counter — it counts advance_to calls only."""
        clock = SimClock()
        assert clock.tick == 0
        clock.advance_to(1.0)
        clock.advance_to(2.0)
        clock.advance_to(3.0)
        assert clock.tick == 3

    def test_reset_clears_time_and_tick(self) -> None:
        clock = SimClock()
        clock.advance_to(100.0)
        clock.advance_to(200.0)
        clock.reset()
        assert clock.elapsed == pytest.approx(0.0)
        assert clock.tick == 0

    def test_advance_after_reset(self) -> None:
        clock = SimClock()
        clock.advance_to(50.0)
        clock.reset()
        clock.advance_to(10.0)
        assert clock.elapsed == pytest.approx(10.0)


class TestSimClockDt:
    def test_dt_default_is_one(self) -> None:
        clock = SimClock()
        assert clock.dt == pytest.approx(1.0)

    def test_dt_is_stored_but_does_not_govern_advance(self) -> None:
        """dt is retained as an optional observer cadence; advance_to ignores it."""
        clock = SimClock(dt=0.5)
        clock.advance_to(7.3)  # not a multiple of dt — still valid
        assert clock.elapsed == pytest.approx(7.3)

    def test_dt_zero_raises(self) -> None:
        with pytest.raises(ValueError):
            SimClock(dt=0.0)

    def test_dt_negative_raises(self) -> None:
        with pytest.raises(ValueError):
            SimClock(dt=-1.0)
