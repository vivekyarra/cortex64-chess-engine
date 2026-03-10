"""Optional sound loading and playback for GUI events."""

from __future__ import annotations

import os

import pygame


class SoundManager:
    """Loads and plays a small set of GUI sound effects."""

    def __init__(self, assets_dir: str = "gui/assets/sounds") -> None:
        self.enabled = True
        self.sounds: dict[str, pygame.mixer.Sound] = {}
        self._load(assets_dir)

    def _load(self, base: str) -> None:
        if not pygame.mixer.get_init():
            try:
                pygame.mixer.init()
            except Exception:
                self.enabled = False
                return
        for name in ["move", "capture", "check", "illegal", "game_end"]:
            path = os.path.join(base, f"{name}.wav")
            if not os.path.exists(path):
                continue
            try:
                self.sounds[name] = pygame.mixer.Sound(path)
            except Exception:
                continue

    def play(self, name: str) -> None:
        """Play a named sound if available and enabled."""
        if self.enabled and name in self.sounds:
            self.sounds[name].play()

    def set_enabled(self, value: bool) -> None:
        """Enable or disable sound playback."""
        self.enabled = bool(value)
