"""Small animation manager used by the board and overlays."""

from __future__ import annotations

import math

import pygame

from gui.constants import ANIM_PIECE_MS, ANIM_PULSE_MS, ANIM_SHAKE_MS


class AnimationManager:
    """Tracks active move, shake, and pulse animations."""

    def __init__(self) -> None:
        self.animations: list[dict] = []

    def add_piece_move(
        self,
        piece_surf: pygame.Surface,
        start_px: tuple[int, int],
        end_px: tuple[int, int],
        duration_ms: int | None = None,
    ) -> None:
        """Animate a piece sliding from start_px to end_px."""
        self.animations.append(
            {
                "type": "move",
                "surf": piece_surf.copy(),
                "start": start_px,
                "end": end_px,
                "elapsed": 0.0,
                "duration": float(duration_ms or ANIM_PIECE_MS),
            }
        )

    def add_shake(self, duration_ms: int | None = None) -> None:
        """Trigger a board shake animation for illegal moves."""
        self.animations = [a for a in self.animations if a["type"] != "shake"]
        self.animations.append(
            {"type": "shake", "elapsed": 0.0, "duration": float(duration_ms or ANIM_SHAKE_MS)}
        )

    def add_check_pulse(self, square_px: tuple[int, int], duration_ms: int | None = None) -> None:
        """Pulse red on a king square when in check."""
        self.animations.append(
            {
                "type": "pulse",
                "pos": square_px,
                "elapsed": 0.0,
                "duration": float(duration_ms or ANIM_PULSE_MS),
            }
        )

    def update(self, dt_ms: float) -> None:
        """Advance active animations and discard finished ones."""
        for animation in self.animations:
            animation["elapsed"] += dt_ms
        self.animations = [a for a in self.animations if a["elapsed"] < a["duration"]]

    def get_shake_offset_x(self) -> int:
        """Return the current board x-offset for the illegal move shake."""
        for animation in self.animations:
            if animation["type"] == "shake":
                t = animation["elapsed"] / max(animation["duration"], 1.0)
                return int(math.sin(t * math.pi * 6) * 3)
        return 0

    def is_animating_move(self) -> bool:
        """Return True if any piece slide animation is active."""
        return any(a["type"] == "move" for a in self.animations)

    def draw_moving_pieces(self, surface: pygame.Surface) -> None:
        """Draw animated pieces after the board and before static pieces."""
        for animation in self.animations:
            if animation["type"] != "move":
                continue
            t = min(1.0, animation["elapsed"] / max(animation["duration"], 1.0))
            eased = 1.0 - (1.0 - t) ** 2
            x = animation["start"][0] + (animation["end"][0] - animation["start"][0]) * eased
            y = animation["start"][1] + (animation["end"][1] - animation["start"][1]) * eased
            surface.blit(animation["surf"], (int(x), int(y)))

    def draw_check_pulse(self, surface: pygame.Surface, square_size: int) -> None:
        """Draw pulsing red overlays on checked king squares."""
        for animation in self.animations:
            if animation["type"] != "pulse":
                continue
            t = animation["elapsed"] / max(animation["duration"], 1.0)
            alpha = int(abs(math.sin(t * math.pi * 3)) * 160)
            overlay = pygame.Surface((square_size, square_size), pygame.SRCALPHA)
            overlay.fill((255, 76, 106, alpha))
            surface.blit(overlay, animation["pos"])
