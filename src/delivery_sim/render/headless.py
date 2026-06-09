"""
Headless no-op renderer — safe to use during training (zero overhead).

Layer: Visualization.
"""

from __future__ import annotations

from delivery_sim.render.protocol import SnapshotConsumer, WorldSnapshot


class HeadlessRenderer:
    """Discards every snapshot silently.

    This is the default renderer so that training runs never touch a display.
    It satisfies the ``SnapshotConsumer`` protocol without inheriting from it.
    """

    def consume(self, snapshot: WorldSnapshot) -> None:
        """Discard *snapshot* — no I/O performed."""

    def close(self) -> None:
        """No-op — no resources to release."""


# Verify the protocol is satisfied at import time (caught by tests + mypy).
_: SnapshotConsumer = HeadlessRenderer()
