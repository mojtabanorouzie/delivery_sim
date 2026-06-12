"""
Pydantic v2 schemas for scenario configuration.

Layer: cross-cutting (consumed by all layers; no layer imports back into config).

One (ScenarioConfig, seed) pair must produce a fully reproducible episode.
All pluggable type names (courier_type, generator_type, etc.) are resolved
against the registry at engine initialisation time, not here.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


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
    prep_time: float = Field(default=30.0, gt=0.0)
    coverage_radius: float = Field(default=500.0, ge=0.0)


class CourierConfig(BaseModel):
    """Configuration for a fleet of identical couriers."""

    courier_type: str
    count: int = 1
    speed: float = 1.5
    capacity: int = 1
    cost_per_unit: float = 0.01


class ProfileBreakpoint(BaseModel):
    """One (time_fraction, rate_factor) point in a DailyProfileDemandGenerator profile.

    time_fraction is a fraction of the episode duration in [0.0, 1.0].
    rate_factor is a non-negative multiplier on the effective_rate at that time.
    """

    time_fraction: float = Field(ge=0.0, le=1.0)
    rate_factor: float = Field(ge=0.0)


class DemandConfig(BaseModel):
    """Configuration for the demand generator."""

    generator_type: str
    rate: float = 1.0
    # B-realistic additions (all defaulted; existing YAMLs load unchanged):
    intensity: float = Field(default=1.0, gt=0.0)
    profile: list[ProfileBreakpoint] = Field(default_factory=list)
    burst_rate_factor: float = Field(default=5.0, gt=1.0)
    burst_duration_fraction: float = Field(default=0.1, gt=0.0, lt=1.0)
    burst_interval_fraction: float = Field(default=0.3, gt=0.0, lt=1.0)

    @model_validator(mode="after")
    def _burst_windows_dont_overlap(self) -> DemandConfig:
        if self.burst_duration_fraction + self.burst_interval_fraction > 1.0:
            raise ValueError(
                "burst_duration_fraction + burst_interval_fraction must be <= 1.0 "
                f"(got {self.burst_duration_fraction} + {self.burst_interval_fraction} "
                f"= {self.burst_duration_fraction + self.burst_interval_fraction:.3f})"
            )
        return self


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
    return_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Probability that a courier's delivery attempt is refused at the door. "
            "0.0 = no returns (default, preserves pre-B-realistic behavior). "
            "Values above ~0.05 are unrealistic; 1.0 is test-only."
        ),
    )
