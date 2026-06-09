"""
String-keyed class registry shared across all pluggable domain types.

Layer: cross-cutting (no layer dependency).

This is the only intentional module-level mutable state in delivery_sim.
Users register custom types here so YAML scenario configs can reference them
by name without importing Python classes directly.

Usage::

    @register("courier", name="my_bike")
    class MyBikeCourier(Courier): ...

    instance = create("courier", "my_bike", courier_id="c1", x=0.0, y=0.0)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

F = TypeVar("F", bound=type[Any])

# Module-level mutable state (intentional; documented in ARCHITECTURE.md).
_REGISTRY: dict[str, dict[str, type[Any]]] = {}


def register(category: str, name: str | None = None) -> Callable[[F], F]:
    """Decorator that registers *cls* under *category* / *name*.

    If *name* is omitted, ``cls.__name__`` is used as the registry key.
    """

    def decorator(cls: F) -> F:
        key = name if name is not None else cls.__name__
        _REGISTRY.setdefault(category, {})[key] = cls
        return cls

    return decorator


def create(category: str, name: str, **kwargs: Any) -> Any:
    """Instantiate a registered class by *category* and *name*, forwarding *kwargs*."""
    if category not in _REGISTRY:
        raise KeyError(f"Category {category!r} not found in registry")
    if name not in _REGISTRY[category]:
        raise KeyError(f"Name {name!r} not found in category {category!r}")
    return _REGISTRY[category][name](**kwargs)


def list_registered(category: str) -> list[str]:
    """Return all registered names for *category*, or an empty list if unknown."""
    return list(_REGISTRY.get(category, {}).keys())


def clear(category: str | None = None) -> None:
    """Clear one *category* or the entire registry.

    Primarily intended for test isolation.
    """
    if category is None:
        _REGISTRY.clear()
    else:
        _REGISTRY.pop(category, None)
