"""Color palette definitions for baseline bots."""

from __future__ import annotations

from typing import Mapping

DEFAULT_BOT_COLOR = "#9AA0A6"

HEURISTIC_STYLE_COLORS: dict[str, str] = {
    "cautious": "#A0CBE8",
    "balanced": "#4E79A7",
    "aggressive": "#2F4B7C",
}

BOT_COLORS: dict[str, str] = {
    "random": "#F28E2B",
    "static": "#59A14F",
    "linear_rl": "#E15759",
}


def resolve_bot_color(
    bot_type: str,
    params: Mapping[str, object] | None = None,
    *,
    default: str | None = DEFAULT_BOT_COLOR,
) -> str | None:
    """Return a hex color for a bot type and optional params."""
    if bot_type == "heuristic":
        style = None
        if params is not None:
            style = params.get("style")
        if isinstance(style, str) and style in HEURISTIC_STYLE_COLORS:
            return HEURISTIC_STYLE_COLORS[style]
        return HEURISTIC_STYLE_COLORS["balanced"]
    color = BOT_COLORS.get(bot_type)
    if color is not None:
        return color
    return default
