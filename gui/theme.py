"""Skin selection helpers for the Cortex64 v2 board themes."""

from __future__ import annotations

from gui.constants import ALL_SKINS

_active_skin_index = 0


def get_skin() -> dict:
    """Return the active board skin."""
    return ALL_SKINS[_active_skin_index]


def set_skin(index: int) -> None:
    """Set the active board skin by index."""
    global _active_skin_index
    _active_skin_index = max(0, min(len(ALL_SKINS) - 1, int(index)))


def skin_name() -> str:
    """Return the active board skin name."""
    return get_skin()["name"]
