from __future__ import annotations

import argparse

from engine.board import BLACK, WHITE
from gui.game import Game


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Chess engine with PyTorch CNN evaluation and Pygame GUI.")
    p.add_argument("--depth", type=int, default=12, help="AI search depth ceiling (plies).")
    p.add_argument("--move-time", type=float, default=1.8, help="AI time budget per move in seconds.")
    p.add_argument(
        "--human",
        type=str,
        default="white",
        choices=("white", "black"),
        help="Human side to play.",
    )
    p.add_argument("--model", type=str, default="ai/models/chess_cnn.pt", help="Path to saved PyTorch model.")
    p.add_argument("--window-size", type=int, default=720, help="Board window size in pixels (square).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    human_color = WHITE if args.human == "white" else BLACK
    Game(
        ai_depth=args.depth,
        ai_move_time_limit=args.move_time,
        human_color=human_color,
        model_path=args.model,
        window_size=args.window_size,
    ).run()


if __name__ == "__main__":
    main()
