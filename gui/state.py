"""Shared GUI state, persistence helpers, and screen data models."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from engine.board import Move

DATA_DIR = Path(__file__).resolve().parent / "data"
PROFILE_PATH = DATA_DIR / "profile.json"
SETTINGS_PATH = DATA_DIR / "settings.json"
EXPORTS_DIR = Path(__file__).resolve().parent / "exports"

DEFAULT_PROFILE = {
    "username": "Soprano",
    "games_played": 0,
    "wins": 0,
    "losses": 0,
    "draws": 0,
    "preferred_side": "white",
}

DEFAULT_SETTINGS = {
    "skin_index": 0,
    "sound": True,
    "animations": "normal",
    "show_legal_moves": True,
    "show_eval_bar": True,
    "font_size": "normal",
}

DIFFICULTY_PRESETS = [
    {"label": "Beginner", "depth": 2, "move_time": 0.5},
    {"label": "Casual", "depth": 4, "move_time": 1.0},
    {"label": "Intermediate", "depth": 6, "move_time": 1.5},
    {"label": "Advanced", "depth": 8, "move_time": 2.0},
    {"label": "Master", "depth": 12, "move_time": 3.0},
]


@dataclass
class TimeControl:
    """Simple minutes plus increment time control."""

    minutes: int = 0
    increment: int = 0

    @property
    def enabled(self) -> bool:
        return self.minutes > 0

    @property
    def initial_ms(self) -> int:
        return int(self.minutes) * 60 * 1000

    @property
    def increment_ms(self) -> int:
        return int(self.increment) * 1000

    @property
    def label(self) -> str:
        if not self.enabled:
            return "Unlimited"
        return f"{self.minutes}+{self.increment}"


TIME_CONTROL_PRESETS = [
    ("Unlimited", None),
    ("10+0", TimeControl(minutes=10, increment=0)),
    ("5+3", TimeControl(minutes=5, increment=3)),
    ("3+2", TimeControl(minutes=3, increment=2)),
    ("Custom", "custom"),
]


@dataclass
class GameConfig:
    """Configuration for a game or analysis session."""

    mode: str = "ai"
    depth: int = 6
    move_time: float = 1.5
    human: str = "white"
    model: Optional[str] = None
    time_control: Optional[TimeControl] = None
    username: str = "Soprano"
    settings: dict[str, Any] = field(default_factory=dict)

    @property
    def human_color(self) -> str:
        return self.human

    @property
    def has_ai(self) -> bool:
        return self.mode == "ai"

    @classmethod
    def from_preset(
        cls,
        preset_index: int,
        mode: str = "ai",
        human: str = "white",
        username: str = "Soprano",
        settings: Optional[dict[str, Any]] = None,
        time_control: Optional[TimeControl] = None,
        model: Optional[str] = None,
    ) -> "GameConfig":
        preset = DIFFICULTY_PRESETS[max(0, min(len(DIFFICULTY_PRESETS) - 1, preset_index))]
        return cls(
            mode=mode,
            depth=int(preset["depth"]),
            move_time=float(preset["move_time"]),
            human=human,
            model=model,
            time_control=time_control,
            username=username or "Soprano",
            settings=dict(settings or {}),
        )


@dataclass
class MoveRecord:
    """Presentation-friendly move metadata used by the game and analysis views."""

    move: Move
    san: str
    color: str
    move_num: int
    quality: Optional[str] = None
    eval_cp: float = 0.0
    delta_cp: float = 0.0
    best_move_uci: str = ""
    explanation: str = ""


@dataclass
class GameSummary:
    """Snapshot of a completed game used by results and analysis screens."""

    config: GameConfig
    result: str
    reason: str
    winner: Optional[str]
    move_records: list[MoveRecord] = field(default_factory=list)
    evals: list[float] = field(default_factory=list)
    total_time_ms: int = 0
    background: Any = None
    export_path: Optional[str] = None
    completed_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))


@dataclass
class AppState:
    """Cross-screen application state."""

    settings: dict[str, Any]
    profile: dict[str, Any]
    model_path: str = "ai/models/chess_cnn.pt"
    sound_manager: Any = None
    last_summary: Optional[GameSummary] = None
    previous_screen: str = "main_menu"


def ensure_data_dirs() -> None:
    """Ensure GUI data and export directories exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)


def load_profile() -> dict[str, Any]:
    """Load the player profile while keeping the schema backward compatible."""
    ensure_data_dirs()
    data = dict(DEFAULT_PROFILE)
    if PROFILE_PATH.exists():
        try:
            with PROFILE_PATH.open("r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            if isinstance(loaded, dict):
                data.update(loaded)
        except Exception:
            pass
    return data


def save_profile(profile: dict[str, Any]) -> None:
    """Persist the player profile without changing the schema."""
    ensure_data_dirs()
    payload = dict(DEFAULT_PROFILE)
    payload.update({key: profile.get(key, value) for key, value in DEFAULT_PROFILE.items()})
    with PROFILE_PATH.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def load_settings() -> dict[str, Any]:
    """Load UI settings with defaults for missing keys."""
    ensure_data_dirs()
    settings = dict(DEFAULT_SETTINGS)
    if SETTINGS_PATH.exists():
        try:
            with SETTINGS_PATH.open("r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            if isinstance(loaded, dict):
                settings.update(loaded)
        except Exception:
            pass
    return settings


def save_settings(settings: dict[str, Any]) -> None:
    """Persist UI settings immediately."""
    ensure_data_dirs()
    payload = dict(DEFAULT_SETTINGS)
    payload.update(settings)
    with SETTINGS_PATH.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def choose_side(selection: str) -> str:
    """Resolve Random side selections into White or Black."""
    if selection == "random":
        return random.choice(["white", "black"])
    return selection


def result_token(summary: GameSummary) -> str:
    """Convert a game summary into a PGN result token."""
    if summary.result == "draw" or summary.winner is None:
        return "1/2-1/2"
    if summary.config.mode == "ai":
        if summary.result == "win":
            return "1-0" if summary.config.human == "white" else "0-1"
        return "0-1" if summary.config.human == "white" else "1-0"
    return "1-0" if summary.winner == "white" else "0-1"


def export_summary_pgn(summary: GameSummary) -> str:
    """Write a completed game to gui/exports and return the file path."""
    ensure_data_dirs()
    white_name = summary.config.username if summary.config.human == "white" or summary.config.mode == "human" else "Cortex64"
    black_name = "Cortex64" if summary.config.mode == "ai" and summary.config.human == "white" else summary.config.username
    if summary.config.mode == "human":
        white_name = f"{summary.config.username} (White)"
        black_name = f"{summary.config.username} (Black)"
    result = result_token(summary)
    move_tokens: list[str] = []
    for index, record in enumerate(summary.move_records):
        if record.color == "white":
            move_tokens.append(f"{record.move_num}.")
        move_tokens.append(record.san)
    move_tokens.append(result)

    pgn = "\n".join(
        [
            '[Event "Cortex64 Game"]',
            '[Site "Local"]',
            f'[Date "{datetime.now().strftime("%Y.%m.%d")}"]',
            f'[White "{white_name}"]',
            f'[Black "{black_name}"]',
            f'[Result "{result}"]',
            "",
            " ".join(move_tokens),
            "",
        ]
    )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = EXPORTS_DIR / f"cortex64_{stamp}.pgn"
    path.write_text(pgn, encoding="utf-8")
    return str(path)
