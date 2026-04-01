#!/usr/bin/env python3
"""Command planning utilities for ver04."""

from __future__ import annotations

from typing import Any


ACTION_ORDER = {
    "zoom_in": 10,
    "dolly_in": 10,
    "zoom_out": 11,
    "arc_shot": 12,
    "low_angle": 15,
    "high_angle": 16,
    "replace_background": 20,
    "remove_instance": 30,
    "increase_quantity": 31,
    "replace_instance": 32,
    "insert_instance": 33,
    "change_color": 40,
    "change_human_motion": 41,
    "apply_style": 50,
    "keep_unchanged": 80,
}


def plan_commands(commands: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def rank(cmd: dict[str, Any]) -> tuple[int, int, str]:
        return (
            ACTION_ORDER.get(str(cmd.get("action", "")), 999),
            int(cmd.get("priority", 999)),
            str(cmd.get("command_id", "")),
        )

    return sorted(commands, key=rank)
