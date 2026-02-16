from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ai.model import ChessCNN, DEFAULT_IN_CHANNELS
from engine.board import KING, PIECE_VALUES, piece_type
from engine.board import Board
from engine.move_generator import generate_legal_moves


def material_target(board: Board) -> float:
    """Deterministic target in [-1, 1] from material balance (White - Black)."""

    score = 0
    for _, piece in board.iter_pieces(color=1):
        pt = piece_type(piece)
        if pt != KING:
            score += PIECE_VALUES[pt]
    for _, piece in board.iter_pieces(color=-1):
        pt = piece_type(piece)
        if pt != KING:
            score -= PIECE_VALUES[pt]

    # Max non-king material per side is ~4000; clamp to keep targets stable.
    return float(max(-1.0, min(1.0, score / 4000.0)))


def generate_random_position(rng: np.random.Generator, min_plies: int, max_plies: int) -> Board:
    board = Board()
    plies = int(rng.integers(min_plies, max_plies + 1))
    for _ in range(plies):
        moves = generate_legal_moves(board)
        if not moves:
            break
        move = moves[int(rng.integers(0, len(moves)))]
        board.push(move)
    return board


def build_dataset(
    samples: int, min_plies: int, max_plies: int, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = np.zeros((samples, DEFAULT_IN_CHANNELS, 8, 8), dtype=np.float32)
    y = np.zeros((samples,), dtype=np.float32)

    for i in range(samples):
        board = generate_random_position(rng, min_plies=min_plies, max_plies=max_plies)
        x[i] = board.legal_tensor()
        y[i] = material_target(board)

    return x, y


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    print("Generating training data...")
    x_np, y_np = build_dataset(args.samples, args.min_plies, args.max_plies, args.seed)
    dataset = TensorDataset(torch.from_numpy(x_np), torch.from_numpy(y_np))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = ChessCNN(in_channels=DEFAULT_IN_CHANNELS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn: nn.Module = nn.MSELoss()

    model.train()
    for epoch in range(1, args.epochs + 1):
        running = 0.0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += float(loss.item()) * xb.size(0)

        avg = running / len(dataset)
        print(f"Epoch {epoch}/{args.epochs} - MSE: {avg:.6f}")

    out_path = Path(args.model_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "in_channels": model.in_channels}, out_path)
    print(f"Saved model to: {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the chess CNN on a simple material-based target.")
    p.add_argument("--samples", type=int, default=4000, help="Number of random positions to generate.")
    p.add_argument("--min-plies", type=int, default=4, help="Minimum random plies from start position.")
    p.add_argument("--max-plies", type=int, default=40, help="Maximum random plies from start position.")
    p.add_argument("--epochs", type=int, default=5, help="Training epochs.")
    p.add_argument("--batch-size", type=int, default=128, help="Batch size.")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    p.add_argument("--seed", type=int, default=0, help="Random seed.")
    p.add_argument("--device", type=str, default=None, help="torch device (e.g. cpu, cuda).")
    p.add_argument("--model-out", type=str, default="ai/models/chess_cnn.pt", help="Output model path.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()

