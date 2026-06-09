"""Smoke tests for the Order state machine (DoD item 3)."""

from __future__ import annotations

import pytest

from delivery_sim.entities.order import ALLOWED_TRANSITIONS, Order, OrderStatus


@pytest.fixture
def fresh_order() -> Order:
    return Order(
        order_id="ord-001",
        store_id="store-a",
        customer_x=100.0,
        customer_y=200.0,
        created_at=0.0,
    )


def test_initial_status(fresh_order: Order) -> None:
    assert fresh_order.status == OrderStatus.CREATED


def test_created_at_stamped_in_timestamps(fresh_order: Order) -> None:
    assert fresh_order.timestamps[OrderStatus.CREATED] == 0.0


def test_legal_transition_created_to_assigned(fresh_order: Order) -> None:
    fresh_order.transition(OrderStatus.ASSIGNED, sim_time=5.0)
    assert fresh_order.status == OrderStatus.ASSIGNED
    assert fresh_order.timestamps[OrderStatus.ASSIGNED] == 5.0


def test_full_happy_path(fresh_order: Order) -> None:
    """CREATED → ASSIGNED → PREPARING → PICKED_UP → IN_TRANSIT → DELIVERED."""
    steps = [
        (OrderStatus.ASSIGNED, 1.0),
        (OrderStatus.PREPARING, 2.0),
        (OrderStatus.PICKED_UP, 3.0),
        (OrderStatus.IN_TRANSIT, 4.0),
        (OrderStatus.DELIVERED, 5.0),
    ]
    for status, t in steps:
        fresh_order.transition(status, sim_time=t)

    assert fresh_order.status == OrderStatus.DELIVERED
    assert fresh_order.is_terminal


def test_illegal_transition_raises_value_error(fresh_order: Order) -> None:
    """Direct CREATED → DELIVERED transition must be rejected."""
    with pytest.raises(ValueError, match="Illegal transition"):
        fresh_order.transition(OrderStatus.DELIVERED, sim_time=1.0)


def test_terminal_delivered_cannot_transition(fresh_order: Order) -> None:
    fresh_order.transition(OrderStatus.ASSIGNED, sim_time=1.0)
    fresh_order.transition(OrderStatus.PREPARING, sim_time=2.0)
    fresh_order.transition(OrderStatus.PICKED_UP, sim_time=3.0)
    fresh_order.transition(OrderStatus.IN_TRANSIT, sim_time=4.0)
    fresh_order.transition(OrderStatus.DELIVERED, sim_time=5.0)

    with pytest.raises(ValueError):
        fresh_order.transition(OrderStatus.FAILED, sim_time=6.0)


def test_failure_from_any_non_terminal_state() -> None:
    """FAILED must be reachable from every non-terminal state."""
    non_terminal = [
        s for s, targets in ALLOWED_TRANSITIONS.items() if targets
    ]
    for status in non_terminal:
        if status == OrderStatus.DELIVERED:
            continue
        order = Order(
            order_id=f"ord-{status.name}",
            store_id="s",
            customer_x=0.0,
            customer_y=0.0,
            created_at=0.0,
            status=status,
        )
        order.transition(OrderStatus.FAILED, sim_time=99.0)
        assert order.status == OrderStatus.FAILED


def test_is_terminal_false_for_in_progress(fresh_order: Order) -> None:
    assert not fresh_order.is_terminal
    fresh_order.transition(OrderStatus.ASSIGNED, sim_time=1.0)
    assert not fresh_order.is_terminal


def test_is_terminal_true_for_failed(fresh_order: Order) -> None:
    fresh_order.transition(OrderStatus.FAILED, sim_time=1.0)
    assert fresh_order.is_terminal
