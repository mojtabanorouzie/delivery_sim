"""Smoke tests for the registry (DoD item 1)."""

from __future__ import annotations

from collections.abc import Generator

import pytest

from delivery_sim.registry import clear, create, list_registered, register

_CAT = "_test_smoke"  # private category; won't clash with built-ins


@pytest.fixture(autouse=True)
def _isolate() -> Generator[None, None, None]:
    yield
    clear(_CAT)


def test_register_and_retrieve() -> None:
    """Registering a dummy provider then retrieving it by name works."""

    @register(_CAT, name="dummy")
    class DummyProvider:
        def __init__(self, value: int = 0) -> None:
            self.value = value

    instance = create(_CAT, "dummy", value=42)
    assert isinstance(instance, DummyProvider)
    assert instance.value == 42


def test_default_name_is_class_name() -> None:
    @register(_CAT)
    class AutoNamed:
        pass

    assert "AutoNamed" in list_registered(_CAT)


def test_list_registered_returns_all_names() -> None:
    @register(_CAT, name="alpha")
    class Alpha:
        pass

    @register(_CAT, name="beta")
    class Beta:
        pass

    names = list_registered(_CAT)
    assert "alpha" in names
    assert "beta" in names


def test_unknown_category_raises() -> None:
    with pytest.raises(KeyError, match="Category"):
        create("__no_such_category__", "anything")


def test_unknown_name_raises() -> None:
    @register(_CAT, name="seed")
    class Seed:
        pass

    with pytest.raises(KeyError, match="Name"):
        create(_CAT, "__no_such_name__")


def test_clear_single_category() -> None:
    @register(_CAT, name="temp")
    class Temp:
        pass

    clear(_CAT)
    assert list_registered(_CAT) == []
