"""
Order entity and its central state machine.

Layer: Domain Entities.

The state machine is fully defined here (enum + transition map + guard).
The *logic* that drives transitions (couriers calling transition()) is stubbed
in step-2/step-3 and lives in the Simulator / Courier implementations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto


class OrderStatus(Enum):
    """All possible states of an Order, in lifecycle order."""

    CREATED = auto()
    ASSIGNED = auto()
    PREPARING = auto()
    PICKED_UP = auto()
    IN_TRANSIT = auto()
    DELIVERED = auto()
    FAILED = auto()
    RETURNED = auto()  # courier reached customer but delivery was refused at the door


# Explicit allowed-transition map. Any transition not listed here is illegal.
ALLOWED_TRANSITIONS: dict[OrderStatus, frozenset[OrderStatus]] = {
    OrderStatus.CREATED: frozenset({OrderStatus.ASSIGNED, OrderStatus.FAILED}),
    OrderStatus.ASSIGNED: frozenset({OrderStatus.PREPARING, OrderStatus.FAILED}),
    OrderStatus.PREPARING: frozenset({OrderStatus.PICKED_UP, OrderStatus.FAILED}),
    OrderStatus.PICKED_UP: frozenset({OrderStatus.IN_TRANSIT, OrderStatus.FAILED}),
    OrderStatus.IN_TRANSIT: frozenset({OrderStatus.DELIVERED, OrderStatus.FAILED,
                                        OrderStatus.RETURNED}),
    OrderStatus.DELIVERED: frozenset(),
    OrderStatus.FAILED: frozenset(),
    OrderStatus.RETURNED: frozenset(),  # terminal — courier carries order back to store
}


@dataclass
class Order:
    """A customer delivery order with a fully tracked state machine.

    All mutable state changes go through ``transition()`` so the timestamp
    log is always consistent.
    """

    order_id: str
    store_id: str
    customer_x: float
    customer_y: float
    created_at: float
    status: OrderStatus = field(default=OrderStatus.CREATED)
    timestamps: dict[OrderStatus, float] = field(default_factory=dict)
    assigned_courier_id: str | None = None
    delivery_cost: float = 0.0  # real courier-distance cost; set at terminal transition

    def __post_init__(self) -> None:
        if OrderStatus.CREATED not in self.timestamps:
            self.timestamps[OrderStatus.CREATED] = self.created_at

    def transition(self, new_status: OrderStatus, sim_time: float) -> None:
        """Move the order to *new_status*, recording *sim_time* as the timestamp.

        Raises ``ValueError`` for any transition not in ``ALLOWED_TRANSITIONS``.
        """
        allowed = ALLOWED_TRANSITIONS[self.status]
        if new_status not in allowed:
            raise ValueError(
                f"Illegal transition: {self.status.name} -> {new_status.name}"
            )
        self.status = new_status
        self.timestamps[new_status] = sim_time

    @property
    def is_terminal(self) -> bool:
        """True if the order has reached a terminal state (DELIVERED, FAILED, or RETURNED)."""
        return self.status in (OrderStatus.DELIVERED, OrderStatus.FAILED, OrderStatus.RETURNED)
