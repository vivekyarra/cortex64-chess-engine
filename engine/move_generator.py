from __future__ import annotations

from typing import Iterable, List, Tuple

from engine.board import (
    BISHOP,
    BLACK,
    EMPTY,
    KING,
    KNIGHT,
    Move,
    PAWN,
    QUEEN,
    ROOK,
    WHITE,
    B_KINGSIDE,
    B_QUEENSIDE,
    W_KINGSIDE,
    W_QUEENSIDE,
    coord_to_square,
    piece_color,
    piece_type,
    square_to_coord,
)


def _in_bounds(row: int, col: int) -> bool:
    return 0 <= row < 8 and 0 <= col < 8


KNIGHT_OFFSETS: Tuple[Tuple[int, int], ...] = (
    (-2, -1),
    (-2, 1),
    (-1, -2),
    (-1, 2),
    (1, -2),
    (1, 2),
    (2, -1),
    (2, 1),
)

DIAGONALS: Tuple[Tuple[int, int], ...] = ((-1, -1), (-1, 1), (1, -1), (1, 1))
ORTHOGONALS: Tuple[Tuple[int, int], ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))


def is_square_attacked(board, square: int, by_color: int) -> bool:
    """Return True if `square` is attacked by `by_color`."""

    row, col = square_to_coord(square)

    # Pawns: reverse-lookup pawn origins that would attack (row, col).
    if by_color == WHITE:
        pr = row + 1
        if pr < 8:
            for dc in (-1, 1):
                pc = col + dc
                if _in_bounds(pr, pc) and int(board.squares[pr, pc]) == WHITE * PAWN:
                    return True
    else:
        pr = row - 1
        if pr >= 0:
            for dc in (-1, 1):
                pc = col + dc
                if _in_bounds(pr, pc) and int(board.squares[pr, pc]) == BLACK * PAWN:
                    return True

    # Knights.
    for dr, dc in KNIGHT_OFFSETS:
        rr, cc = row + dr, col + dc
        if not _in_bounds(rr, cc):
            continue
        p = int(board.squares[rr, cc])
        if p != EMPTY and piece_color(p) == by_color and piece_type(p) == KNIGHT:
            return True

    # Bishops / Queens (diagonals).
    for dr, dc in DIAGONALS:
        rr, cc = row + dr, col + dc
        while _in_bounds(rr, cc):
            p = int(board.squares[rr, cc])
            if p != EMPTY:
                if piece_color(p) == by_color and piece_type(p) in (BISHOP, QUEEN):
                    return True
                break
            rr += dr
            cc += dc

    # Rooks / Queens (orthogonals).
    for dr, dc in ORTHOGONALS:
        rr, cc = row + dr, col + dc
        while _in_bounds(rr, cc):
            p = int(board.squares[rr, cc])
            if p != EMPTY:
                if piece_color(p) == by_color and piece_type(p) in (ROOK, QUEEN):
                    return True
                break
            rr += dr
            cc += dc

    # King adjacency.
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr, cc = row + dr, col + dc
            if not _in_bounds(rr, cc):
                continue
            p = int(board.squares[rr, cc])
            if p != EMPTY and piece_color(p) == by_color and piece_type(p) == KING:
                return True

    return False


def is_in_check(board, color: int) -> bool:
    king_sq = board.white_king_sq if color == WHITE else board.black_king_sq
    return is_square_attacked(board, king_sq, -color)


def _gen_pawn_moves(board, from_sq: int, color: int) -> Iterable[Move]:
    row, col = square_to_coord(from_sq)
    direction = -1 if color == WHITE else 1
    start_row = 6 if color == WHITE else 1
    promotion_row = 0 if color == WHITE else 7

    # One step forward.
    r1 = row + direction
    if _in_bounds(r1, col) and int(board.squares[r1, col]) == EMPTY:
        to_sq = coord_to_square(r1, col)
        if r1 == promotion_row:
            for promo in (QUEEN, ROOK, BISHOP, KNIGHT):
                yield Move(from_sq, to_sq, promotion=color * promo)
        else:
            yield Move(from_sq, to_sq)

        # Two steps from starting row.
        r2 = row + 2 * direction
        if row == start_row and _in_bounds(r2, col) and int(board.squares[r2, col]) == EMPTY:
            yield Move(from_sq, coord_to_square(r2, col))

    # Captures (including en passant).
    for dc in (-1, 1):
        cc = col + dc
        rr = row + direction
        if not _in_bounds(rr, cc):
            continue
        to_sq = coord_to_square(rr, cc)
        target = int(board.squares[rr, cc])
        if target != EMPTY and piece_color(target) == -color:
            if rr == promotion_row:
                for promo in (QUEEN, ROOK, BISHOP, KNIGHT):
                    yield Move(from_sq, to_sq, promotion=color * promo)
            else:
                yield Move(from_sq, to_sq)
        elif board.en_passant == to_sq:
            yield Move(from_sq, to_sq, is_en_passant=True)


def _gen_knight_moves(board, from_sq: int, color: int) -> Iterable[Move]:
    row, col = square_to_coord(from_sq)
    for dr, dc in KNIGHT_OFFSETS:
        rr, cc = row + dr, col + dc
        if not _in_bounds(rr, cc):
            continue
        target = int(board.squares[rr, cc])
        if target == EMPTY or piece_color(target) == -color:
            yield Move(from_sq, coord_to_square(rr, cc))


def _gen_sliding_moves(board, from_sq: int, color: int, directions: Tuple[Tuple[int, int], ...]) -> Iterable[Move]:
    row, col = square_to_coord(from_sq)
    for dr, dc in directions:
        rr, cc = row + dr, col + dc
        while _in_bounds(rr, cc):
            target = int(board.squares[rr, cc])
            if target == EMPTY:
                yield Move(from_sq, coord_to_square(rr, cc))
            else:
                if piece_color(target) == -color:
                    yield Move(from_sq, coord_to_square(rr, cc))
                break
            rr += dr
            cc += dc


def _gen_king_moves(board, from_sq: int, color: int) -> Iterable[Move]:
    row, col = square_to_coord(from_sq)

    # Normal king moves (legality checked later).
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr, cc = row + dr, col + dc
            if not _in_bounds(rr, cc):
                continue
            target = int(board.squares[rr, cc])
            if target == EMPTY or piece_color(target) == -color:
                yield Move(from_sq, coord_to_square(rr, cc))

    # Castling (must ensure squares are empty and not attacked).
    if color == WHITE:
        king_start = coord_to_square(7, 4)
        if from_sq != king_start:
            return
        if is_square_attacked(board, from_sq, BLACK):
            return

        # Kingside.
        if board.castling_rights & W_KINGSIDE:
            f1 = coord_to_square(7, 5)
            g1 = coord_to_square(7, 6)
            rook_sq = coord_to_square(7, 7)
            if (
                board.piece_at(rook_sq) == WHITE * ROOK
                and board.piece_at(f1) == EMPTY
                and board.piece_at(g1) == EMPTY
                and not is_square_attacked(board, f1, BLACK)
                and not is_square_attacked(board, g1, BLACK)
            ):
                yield Move(from_sq, g1, is_castling=True)

        # Queenside.
        if board.castling_rights & W_QUEENSIDE:
            d1 = coord_to_square(7, 3)
            c1 = coord_to_square(7, 2)
            b1 = coord_to_square(7, 1)
            rook_sq = coord_to_square(7, 0)
            if (
                board.piece_at(rook_sq) == WHITE * ROOK
                and board.piece_at(d1) == EMPTY
                and board.piece_at(c1) == EMPTY
                and board.piece_at(b1) == EMPTY
                and not is_square_attacked(board, d1, BLACK)
                and not is_square_attacked(board, c1, BLACK)
            ):
                yield Move(from_sq, c1, is_castling=True)
    else:
        king_start = coord_to_square(0, 4)
        if from_sq != king_start:
            return
        if is_square_attacked(board, from_sq, WHITE):
            return

        # Kingside.
        if board.castling_rights & B_KINGSIDE:
            f8 = coord_to_square(0, 5)
            g8 = coord_to_square(0, 6)
            rook_sq = coord_to_square(0, 7)
            if (
                board.piece_at(rook_sq) == BLACK * ROOK
                and board.piece_at(f8) == EMPTY
                and board.piece_at(g8) == EMPTY
                and not is_square_attacked(board, f8, WHITE)
                and not is_square_attacked(board, g8, WHITE)
            ):
                yield Move(from_sq, g8, is_castling=True)

        # Queenside.
        if board.castling_rights & B_QUEENSIDE:
            d8 = coord_to_square(0, 3)
            c8 = coord_to_square(0, 2)
            b8 = coord_to_square(0, 1)
            rook_sq = coord_to_square(0, 0)
            if (
                board.piece_at(rook_sq) == BLACK * ROOK
                and board.piece_at(d8) == EMPTY
                and board.piece_at(c8) == EMPTY
                and board.piece_at(b8) == EMPTY
                and not is_square_attacked(board, d8, WHITE)
                and not is_square_attacked(board, c8, WHITE)
            ):
                yield Move(from_sq, c8, is_castling=True)


def generate_pseudo_legal_moves(board) -> List[Move]:
    color = board.side_to_move
    moves: List[Move] = []
    for from_sq, piece in board.iter_pieces(color):
        pt = piece_type(piece)
        if pt == PAWN:
            moves.extend(_gen_pawn_moves(board, from_sq, color))
        elif pt == KNIGHT:
            moves.extend(_gen_knight_moves(board, from_sq, color))
        elif pt == BISHOP:
            moves.extend(_gen_sliding_moves(board, from_sq, color, DIAGONALS))
        elif pt == ROOK:
            moves.extend(_gen_sliding_moves(board, from_sq, color, ORTHOGONALS))
        elif pt == QUEEN:
            moves.extend(_gen_sliding_moves(board, from_sq, color, DIAGONALS))
            moves.extend(_gen_sliding_moves(board, from_sq, color, ORTHOGONALS))
        elif pt == KING:
            moves.extend(_gen_king_moves(board, from_sq, color))
    return moves


def generate_legal_moves(board) -> List[Move]:
    """Generate all legal moves for the current side to move."""

    color = board.side_to_move
    legal: List[Move] = []
    for move in generate_pseudo_legal_moves(board):
        board.push(move)
        illegal = is_in_check(board, color)
        board.pop()
        if not illegal:
            legal.append(move)
    return legal


def legal_moves_from(board, from_sq: int) -> List[Move]:
    return [m for m in generate_legal_moves(board) if m.from_sq == from_sq]


def is_checkmate(board) -> bool:
    return is_in_check(board, board.side_to_move) and len(generate_legal_moves(board)) == 0


def is_stalemate(board) -> bool:
    return (not is_in_check(board, board.side_to_move)) and len(generate_legal_moves(board)) == 0

