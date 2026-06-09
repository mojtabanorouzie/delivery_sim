"""
delivery_sim — extensible last-mile delivery simulation for RL research.

Public API re-exported here so users need only ``import delivery_sim``.

Importing this package triggers registration of all built-in entity types
(BuiltinStore, BikeCourier, PoissonDemandGenerator, EuclideanRouting,
SparseDeliveryReward) via their ``@register`` decorators.
"""

from __future__ import annotations

# Trigger built-in registrations.
import delivery_sim.entities  # noqa: F401
import delivery_sim.rewards  # noqa: F401
import delivery_sim.routing  # noqa: F401
from delivery_sim.config.loader import load_scenario
from delivery_sim.config.schema import ScenarioConfig
from delivery_sim.engine.simulator import Simulator
from delivery_sim.entities.courier import BikeCourier, Courier, CourierStatus
from delivery_sim.entities.demand_generator import DemandGenerator, PoissonDemandGenerator
from delivery_sim.entities.order import ALLOWED_TRANSITIONS, Order, OrderStatus
from delivery_sim.entities.store import BuiltinStore, Store
from delivery_sim.envs.multi_agent import DeliveryParallelEnv
from delivery_sim.envs.single_agent import DeliveryEnv
from delivery_sim.registry import clear, create, list_registered, register
from delivery_sim.render.protocol import SnapshotConsumer, WorldSnapshot
from delivery_sim.routing.base import RoutingModel
from delivery_sim.routing.euclidean import EuclideanRouting

__version__ = "0.1.0"

__all__ = [
    # Registry
    "register",
    "create",
    "list_registered",
    "clear",
    # Config
    "load_scenario",
    "ScenarioConfig",
    # Entities
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
    # Routing
    "RoutingModel",
    "EuclideanRouting",
    # Engine
    "Simulator",
    # Envs
    "DeliveryEnv",
    "DeliveryParallelEnv",
    # Render
    "SnapshotConsumer",
    "WorldSnapshot",
]
