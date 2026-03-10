"""UI helper functions shared across Cortex64 GUI screens."""

from __future__ import annotations

import math
from pathlib import Path

import pygame

from engine.board import BISHOP, KING, KNIGHT, PAWN, QUEEN, ROOK
from gui.constants import ACCENT, WHITE_COL

PIECE_IMAGE_FILES = {
    PAWN: "wp.png",
    KNIGHT: "wn.png",
    BISHOP: "wb.png",
    ROOK: "wr.png",
    QUEEN: "wq.png",
    KING: "wk.png",
    -PAWN: "bp.png",
    -KNIGHT: "bn.png",
    -BISHOP: "bb.png",
    -ROOK: "br.png",
    -QUEEN: "bq.png",
    -KING: "bk.png",
}

FALLBACK_PIECE_LABELS = {
    PAWN: "P",
    KNIGHT: "N",
    BISHOP: "B",
    ROOK: "R",
    QUEEN: "Q",
    KING: "K",
    -PAWN: "p",
    -KNIGHT: "n",
    -BISHOP: "b",
    -ROOK: "r",
    -QUEEN: "q",
    -KING: "k",
}


def asset_path(relative: str) -> Path | None:
    """Resolve an asset path, supporting both assets/ and assests/ directories."""
    base = Path(__file__).resolve().parent
    for root in ("assets", "assests"):
        candidate = base / root / relative
        if candidate.exists():
            return candidate
    return None


def blur_surface(surface: pygame.Surface, scale: float = 0.12) -> pygame.Surface:
    """Create a soft blur by downscaling and upscaling the surface."""
    width = max(1, int(surface.get_width() * scale))
    height = max(1, int(surface.get_height() * scale))
    small = pygame.transform.smoothscale(surface, (width, height))
    return pygame.transform.smoothscale(small, surface.get_size())


def fit_text(font: pygame.font.Font, text: str, max_width: int) -> str:
    """Trim text with an ellipsis until it fits."""
    trimmed = text
    while trimmed and font.size(trimmed)[0] > max_width:
        trimmed = trimmed[:-4] + "..." if len(trimmed) > 4 else trimmed[:-1]
    return trimmed or text[:1]


def make_avatar_surface(size: int, label: str, fill_color: tuple[int, int, int]) -> pygame.Surface:
    """Build a circular avatar placeholder from an initial."""
    surface = pygame.Surface((size, size), pygame.SRCALPHA)
    center = (size // 2, size // 2)
    pygame.draw.circle(surface, (12, 35, 47, 230), center, size // 2)
    pygame.draw.circle(surface, fill_color, center, size // 2 - 2)
    font = pygame.font.SysFont("segoeui", max(14, size // 2), bold=True)
    text = font.render((label[:1] or "?").upper(), True, WHITE_COL)
    surface.blit(text, text.get_rect(center=center))
    return surface


def load_piece_images(square_size: int) -> dict[int, pygame.Surface]:
    """Load board piece images with a vector fallback."""
    images: dict[int, pygame.Surface] = {}
    piece_px = square_size - 8
    font = pygame.font.SysFont("segoeui", max(24, square_size // 2), bold=True)
    for piece, filename in PIECE_IMAGE_FILES.items():
        path = asset_path(f"pieces/{filename}")
        if path is None:
            images[piece] = create_piece_fallback(piece, piece_px, font)
            continue
        try:
            image = pygame.image.load(str(path)).convert_alpha()
            images[piece] = pygame.transform.smoothscale(image, (piece_px, piece_px))
        except Exception:
            images[piece] = create_piece_fallback(piece, piece_px, font)
    return images


def create_piece_fallback(piece: int, size: int, font: pygame.font.Font | None = None) -> pygame.Surface:
    """Create a simple fallback piece glyph if PNG loading fails."""
    font = font or pygame.font.SysFont("segoeui", max(24, size // 2), bold=True)
    surf = pygame.Surface((size, size), pygame.SRCALPHA)
    center = (surf.get_width() // 2, surf.get_height() // 2)
    radius = min(center) - 2
    fill = (244, 246, 248, 245) if piece > 0 else (22, 31, 41, 245)
    stroke = (35, 57, 80) if piece > 0 else (220, 227, 234)
    pygame.draw.circle(surf, fill, center, radius)
    pygame.draw.circle(surf, stroke, center, radius, width=2)
    label = FALLBACK_PIECE_LABELS.get(piece, "?")
    text = font.render(label, True, stroke if piece < 0 else (22, 35, 48))
    surf.blit(text, text.get_rect(center=center))
    return surf


def draw_arrow(
    surface: pygame.Surface,
    start: tuple[float, float],
    end: tuple[float, float],
    color: tuple[int, int, int],
    width: int = 12,
    alpha: int = 180,
) -> None:
    """Draw a thick arrow between two points."""
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    distance = math.hypot(dx, dy)
    if distance < 1:
        return

    ux = dx / distance
    uy = dy / distance
    head_len = 24
    head_w = 18
    line_end = (end[0] - ux * head_len * 0.6, end[1] - uy * head_len * 0.6)

    overlay = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
    pygame.draw.line(
        overlay,
        (*color, alpha),
        (int(start[0]), int(start[1])),
        (int(line_end[0]), int(line_end[1])),
        width,
    )

    left = (
        end[0] - ux * head_len + uy * head_w * 0.5,
        end[1] - uy * head_len - ux * head_w * 0.5,
    )
    right = (
        end[0] - ux * head_len - uy * head_w * 0.5,
        end[1] - uy * head_len + ux * head_w * 0.5,
    )
    pygame.draw.polygon(
        overlay,
        (*color, alpha),
        [(int(end[0]), int(end[1])), (int(left[0]), int(left[1])), (int(right[0]), int(right[1]))],
    )
    surface.blit(overlay, (0, 0))


def glow_rect(
    surface: pygame.Surface,
    rect: pygame.Rect | tuple[int, int, int, int],
    color: tuple[int, int, int] = ACCENT,
    radius: int = 16,
    alpha: int = 40,
) -> None:
    """Draw a soft glow around a rectangle."""
    rect = pygame.Rect(rect)
    glow = pygame.Surface((rect.w + radius * 2, rect.h + radius * 2), pygame.SRCALPHA)
    pygame.draw.rect(glow, (*color, alpha), glow.get_rect(), border_radius=radius)
    surface.blit(glow, (rect.x - radius, rect.y - radius))
