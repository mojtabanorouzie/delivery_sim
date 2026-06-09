"""Domain entities: Store, Courier, Order, DemandGenerator.

Importing this package triggers registration of all built-in types.
"""

from __future__ import annotations

from delivery_sim.entities.courier import BikeCourier, Courier, CourierStatus
from delivery_sim.entities.demand_generator import DemandGenerator, PoissonDemandGenerator
from delivery_sim.entities.order import ALLOWED_TRANSITIONS, Order, OrderStatus
from delivery_sim.entities.store import BuiltinStore, Store

__all__ = [
    "Store",
    "BuiltinStore",
    "Courier",
    "BikeCourier",
    "CourierStatus",
    "Order",
    "OrderStatus",
    "ALLOWED_TRANSITIONS",
    "DemandGenerator",
    "PoissonDemandGenerator",
]
