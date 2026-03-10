"""Application entry point and screen router for Cortex64 v2."""

from __future__ import annotations

import argparse
import sys

import pygame

from gui import theme
from gui.constants import BG_DARK, FPS, WINDOW_H, WINDOW_W
from gui.game import GameScreen
from gui.screens.main_menu import MainMenuScreen
from gui.sound import SoundManager
from gui.state import AppState, GameConfig, ensure_data_dirs, load_profile, load_settings


def parse_args() -> argparse.Namespace:
    """Parse CLI flags for direct-launch backward compatibility."""
    parser = argparse.ArgumentParser(description="Cortex64 Chess Engine v2")
    parser.add_argument("--depth", type=int, default=None)
    parser.add_argument("--move-time", type=float, default=None, dest="move_time")
    parser.add_argument("--human", type=str, default=None, choices=["white", "black"])
    parser.add_argument("--model", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    """Initialize pygame, load settings, and run the active screen loop."""
    args = parse_args()
    ensure_data_dirs()
    settings = load_settings()
    profile = load_profile()
    theme.set_skin(settings.get("skin_index", 0))

    pygame.init()
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption("Cortex64 — AI Chess Engine v2.0")
    clock = pygame.time.Clock()

    app_state = AppState(
        settings=settings,
        profile=profile,
        model_path=args.model or "ai/models/chess_cnn.pt",
        sound_manager=SoundManager(),
    )
    if app_state.sound_manager is not None:
        app_state.sound_manager.set_enabled(bool(settings.get("sound", True)))

    if args.depth is not None or args.human is not None:
        config = GameConfig(
            mode="ai",
            depth=args.depth or 6,
            move_time=args.move_time or 1.5,
            human=args.human or str(profile.get("preferred_side", "white")).lower(),
            model=args.model or app_state.model_path,
            time_control=None,
            username=str(profile.get("username", "Soprano")),
            settings=dict(settings),
        )
        current_screen = GameScreen(screen, app_state, config)
    else:
        current_screen = MainMenuScreen(screen, app_state)

    while True:
        dt = clock.tick(FPS)
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        next_screen = current_screen.update(events, dt)
        screen.fill(BG_DARK)
        current_screen.draw(screen)
        pygame.display.flip()
        if next_screen is not None and next_screen is not current_screen:
            current_screen = next_screen


if __name__ == "__main__":
    main()
