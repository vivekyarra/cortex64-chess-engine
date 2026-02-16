from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from ai.model import ChessCNN, DEFAULT_IN_CHANNELS


class NeuralEvaluator:
    """Loads a saved CNN and evaluates board positions.

    The model outputs a scalar in [-1, 1], positive = advantage for White.
    """

    def __init__(self, model_path: str = "ai/models/chess_cnn.pt", device: Optional[str] = None) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = ChessCNN(in_channels=DEFAULT_IN_CHANNELS).to(self.device)
        self.model.eval()
        self.model_path = model_path
        self.loaded = self.load(model_path)

    def load(self, model_path: str) -> bool:
        path = Path(model_path)
        self.model_path = str(path)

        if not path.exists():
            # Still usable: model remains randomly initialized.
            self.model.eval()
            return False

        data = torch.load(path, map_location=self.device)
        if isinstance(data, dict) and "state_dict" in data:
            in_channels = int(data.get("in_channels", DEFAULT_IN_CHANNELS))
            if in_channels != self.model.in_channels:
                self.model = ChessCNN(in_channels=in_channels).to(self.device)
            self.model.load_state_dict(data["state_dict"])
        else:
            # Back-compat: allow loading raw state_dict.
            self.model.load_state_dict(data)

        self.model.eval()
        return True

    @torch.no_grad()
    def evaluate(self, board) -> float:
        planes = board.legal_tensor()  # (C, 8, 8) float32
        x = torch.from_numpy(planes).unsqueeze(0).to(self.device)  # (1, C, 8, 8)
        y = self.model(x)
        return float(y.item())


def main() -> None:
    # Quick manual smoke test: evaluate the starting position.
    from engine.board import Board

    board = Board()
    evaluator = NeuralEvaluator()
    score = evaluator.evaluate(board)
    status = "loaded" if evaluator.loaded else "random-weights (run ai/train.py to train)"
    print(f"Model: {status}")
    print(f"Eval(start position): {score:+.4f}  (positive = White)")


if __name__ == "__main__":
    main()

