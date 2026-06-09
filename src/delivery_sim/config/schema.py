"""
Pydantic v2 schemas for scenario configuration.

Layer: cross-cutting (consumed by all layers; no layer imports back into config).

One (ScenarioConfig, seed) pair must produce a fully reproducible episode.
All pluggable type names (courier_type, generator_type, etc.) are resolved
against the registry at engine initialisation time, not here.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class WorldConfig(BaseModel):
    """Spatial extent of the simulation world."""

    width: float = 1000.0
    height: float = 1000.0


class StoreConfig(BaseModel):
    """Configuration for a single store instance."""

    name: str
    x: float
    y: float
    capacity: int = 10
    coverage_radius: float = Field(default=500.0, ge=0.0)


class CourierConfig(BaseModel):
    """Configuration for a fleet of identical couriers."""

    courier_type: str
    count: int = 1
    speed: float = 1.5
    capacity: int = 1
    cost_per_unit: float = 0.01


class DemandConfig(BaseModel):
    """Configuration for the demand generator."""

    generator_type: str
    rate: float = 1.0


class RoutingConfig(BaseModel):
    """Configuration for the spatial routing model."""

    model_type: str = "euclidean"


class RewardConfig(BaseModel):
    """Configuration for the reward function."""

    function_type: str = "SparseDeliveryReward"


class ScenarioConfig(BaseModel):
    """Top-level scenario configuration.

    A single (ScenarioConfig, seed) pair is the unit of reproducibility:
    every random draw in a run must come from the seeded RNG derived here.
    """

    name: str
    seed: int = 42
    dt: float = Field(default=1.0, gt=0, description="World tick duration in seconds")
    max_steps: int = Field(default=1000, gt=0)
    world: WorldConfig = Field(default_factory=WorldConfig)
    stores: list[StoreConfig] = Field(default_factory=list)
    couriers: list[CourierConfig] = Field(default_factory=list)
    demand: DemandConfig = Field(
        default_factory=lambda: DemandConfig(generator_type="PoissonDemandGenerator")
    )
    routing: RoutingConfig = Field(default_factory=RoutingConfig)
    reward: RewardConfig = Field(default_factory=RewardConfig)
    decision_interval: float = Field(
        default=100.0, gt=0,
        description="Env step size in sim-seconds (how often the agent acts)",
    )
    max_coverage_radius: float = Field(
        default=1000.0, gt=0,
        description="Action-space upper bound for per-store coverage_radius",
    )
    observation_preset: str = Field(
        default="standard",
        description="Named ObservationSpec preset (see delivery_sim.envs.observations)",
    )
