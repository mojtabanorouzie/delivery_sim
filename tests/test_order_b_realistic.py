"""B2 — Order state machine tests for the RETURNED terminal state."""

from __future__ import annotations

import pytest

from delivery_sim.entities.order import ALLOWED_TRANSITIONS, Order, OrderStatus


class TestReturnedStateExists:
    def test_returned_in_enum(self) -> None:
        assert hasattr(OrderStatus, "RETURNED")

    def test_returned_in_allowed_transitions(self) -> None:
        assert OrderStatus.RETURNED in ALLOWED_TRANSITIONS

    def test_returned_is_terminal_no_outgoing(self) -> None:
        assert ALLOWED_TRANSITIONS[OrderStatus.RETURNED] == frozenset()


class TestReturnedTransition:
    def _order_in_transit(self) -> Order:
        o = Order(
            order_id="o1", store_id="s1",
            customer_x=10.0, customer_y=20.0, created_at=0.0,
        )
        o.transition(OrderStatus.ASSIGNED, 1.0)
        o.transition(OrderStatus.PREPARING, 2.0)
        o.transition(OrderStatus.PICKED_UP, 3.0)
        o.transition(OrderStatus.IN_TRANSIT, 3.0)
        return o

    def test_in_transit_to_returned_allowed(self) -> None:
        o = self._order_in_transit()
        o.transition(OrderStatus.RETURNED, 10.0)
        assert o.status == OrderStatus.RETURNED

    def test_returned_timestamp_recorded(self) -> None:
        o = self._order_in_transit()
        o.transition(OrderStatus.RETURNED, 10.0)
        assert o.timestamps[OrderStatus.RETURNED] == 10.0

    def test_returned_is_terminal(self) -> None:
        o = self._order_in_transit()
        o.transition(OrderStatus.RETURNED, 10.0)
        assert o.is_terminal is True

    def test_returned_to_anything_raises(self) -> None:
        o = self._order_in_transit()
        o.transition(OrderStatus.RETURNED, 10.0)
        with pytest.raises(ValueError):
            o.transition(OrderStatus.FAILED, 11.0)

    def test_delivered_to_returned_raises(self) -> None:
        o = self._order_in_transit()
        o.transition(OrderStatus.DELIVERED, 10.0)
        with pytest.raises(ValueError):
            o.transition(OrderStatus.RETURNED, 11.0)

    def test_failed_to_returned_raises(self) -> None:
        o = Order(
            order_id="o2", store_id="s1",
            customer_x=0.0, customer_y=0.0, created_at=0.0,
        )
        o.transition(OrderStatus.FAILED, 1.0)
        with pytest.raises(ValueError):
            o.transition(OrderStatus.RETURNED, 2.0)

    def test_created_to_returned_raises(self) -> None:
        o = Order(
            order_id="o3", store_id="s1",
            customer_x=0.0, customer_y=0.0, created_at=0.0,
        )
        with pytest.raises(ValueError):
            o.transition(OrderStatus.RETURNED, 1.0)


class TestIsTerminal:
    def test_delivered_is_terminal(self) -> None:
        o = Order(order_id="o", store_id="s", customer_x=0.0, customer_y=0.0, created_at=0.0)
        for status in [
            OrderStatus.ASSIGNED, OrderStatus.PREPARING,
            OrderStatus.PICKED_UP, OrderStatus.IN_TRANSIT, OrderStatus.DELIVERED,
        ]:
            o.transition(status, 1.0)
        assert o.is_terminal is True

    def test_failed_is_terminal(self) -> None:
        o = Order(order_id="o", store_id="s", customer_x=0.0, customer_y=0.0, created_at=0.0)
        o.transition(OrderStatus.FAILED, 1.0)
        assert o.is_terminal is True

    def test_returned_is_terminal(self) -> None:
        o = Order(order_id="o", store_id="s", customer_x=0.0, customer_y=0.0, created_at=0.0)
        for status in [
            OrderStatus.ASSIGNED, OrderStatus.PREPARING,
            OrderStatus.PICKED_UP, OrderStatus.IN_TRANSIT, OrderStatus.RETURNED,
        ]:
            o.transition(status, 1.0)
        assert o.is_terminal is True

    def test_in_transit_not_terminal(self) -> None:
        o = Order(order_id="o", store_id="s", customer_x=0.0, customer_y=0.0, created_at=0.0)
        for status in [
            OrderStatus.ASSIGNED, OrderStatus.PREPARING,
            OrderStatus.PICKED_UP, OrderStatus.IN_TRANSIT,
        ]:
            o.transition(status, 1.0)
        assert o.is_terminal is False
