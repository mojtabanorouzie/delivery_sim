"""Render module: snapshot protocol and renderer implementations."""

from __future__ import annotations

from delivery_sim.render.headless import HeadlessRenderer
from delivery_sim.render.protocol import (
    CourierSnapshot,
    OrderSnapshot,
    SnapshotConsumer,
    StoreSnapshot,
    WorldSnapshot,
)

__all__ = [
    "SnapshotConsumer",
    "WorldSnapshot",
    "StoreSnapshot",
    "CourierSnapshot",
    "OrderSnapshot",
    "HeadlessRenderer",
    "PygameRenderer",
]


def __getattr__(name: str) -> object:
    if name == "PygameRenderer":
        from delivery_sim.render.pygame_renderer import PygameRenderer  # noqa: PLC0415
        return PygameRenderer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
