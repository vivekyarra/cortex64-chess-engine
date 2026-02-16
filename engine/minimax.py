from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from engine.board import (
    EMPTY,
    Move,
    PIECE_VALUES,
    piece_type,
)
from engine.move_generator import generate_legal_moves, is_in_check


INF = 10**9
MATE_SCORE = 100_000


@dataclass
class SearchResult:
    move: Optional[Move]
    score: float
    nodes: int


def _captured_piece(board, move: Move) -> int:
    if move.is_en_passant:
        # Captured pawn sits behind destination square.
        direction = -8 if board.side_to_move < 0 else 8
        return board.piece_at(move.to_sq + direction)
    return board.piece_at(move.to_sq)


def _move_order_score(board, move: Move) -> int:
    """Heuristic for move ordering (higher searched first)."""

    score = 0
    moved = board.piece_at(move.from_sq)
    captured = _captured_piece(board, move)

    if move.promotion is not None:
        score += 20_000 + PIECE_VALUES.get(piece_type(move.promotion), 0)

    if captured != EMPTY:
        # MVV-LVA-ish.
        score += 10_000 + PIECE_VALUES.get(piece_type(captured), 0) - PIECE_VALUES.get(piece_type(moved), 0) // 10

    if move.is_castling:
        score += 200

    return score


def _ordered_moves(board) -> list[Move]:
    moves = generate_legal_moves(board)
    moves.sort(key=lambda m: _move_order_score(board, m), reverse=True)
    return moves


def negamax(board, depth: int, alpha: float, beta: float, evaluator, ply: int, node_counter: list[int]) -> float:
    """Negamax with alpha-beta pruning.

    Returns score from the perspective of the side to move.
    """

    node_counter[0] += 1

    moves = _ordered_moves(board)
    if depth == 0 or not moves:
        if not moves:
            if is_in_check(board, board.side_to_move):
                return -MATE_SCORE + ply
            return 0.0  # stalemate

        # Leaf evaluation: evaluator returns perspective of White; flip for side-to-move.
        return float(board.side_to_move) * float(evaluator.evaluate(board))

    best = -INF
    for move in moves:
        board.push(move)
        score = -negamax(board, depth - 1, -beta, -alpha, evaluator, ply + 1, node_counter)
        board.pop()

        if score > best:
            best = score
        if best > alpha:
            alpha = best
        if alpha >= beta:
            break
    return best


def find_best_move(board, depth: int, evaluator) -> SearchResult:
    """Find the best move for the current side using alpha-beta minimax."""

    node_counter = [0]
    best_move: Optional[Move] = None
    best_score = -INF

    moves = _ordered_moves(board)
    if not moves:
        if is_in_check(board, board.side_to_move):
            return SearchResult(move=None, score=-MATE_SCORE, nodes=node_counter[0])
        return SearchResult(move=None, score=0.0, nodes=node_counter[0])

    alpha = -INF
    beta = INF

    for move in moves:
        board.push(move)
        score = -negamax(board, depth - 1, -beta, -alpha, evaluator, ply=1, node_counter=node_counter)
        board.pop()

        if score > best_score:
            best_score = score
            best_move = move
        if score > alpha:
            alpha = score

    return SearchResult(move=best_move, score=float(best_score), nodes=node_counter[0])
