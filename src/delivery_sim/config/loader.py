"""
YAML scenario loader.

Layer: cross-cutting (consumed by engine and RL wrappers).

The only function here that is fully implemented in the scaffold.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from delivery_sim.config.schema import ScenarioConfig


def load_scenario(path: str | Path) -> ScenarioConfig:
    """Load and validate a scenario config from a YAML file.

    Raises ``FileNotFoundError`` if *path* does not exist.
    Raises ``pydantic.ValidationError`` if the YAML does not match the schema.
    """
    resolved = Path(path)
    with resolved.open(encoding="utf-8") as fh:
        raw: Any = yaml.safe_load(fh)
    return ScenarioConfig.model_validate(raw)
