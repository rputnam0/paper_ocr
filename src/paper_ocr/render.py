from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Optional

import fitz  # PyMuPDF
from PIL import Image, ImageEnhance, ImageOps


@dataclass
class RenderResult:
    image_bytes: bytes
    mime_type: str
    width: int
    height: int
    format: str


def _render_page_to_image(page: fitz.Page, dpi: int, rotation: int = 0) -> Image.Image:
    matrix = fitz.Matrix(dpi / 72, dpi / 72)
    if rotation:
        matrix = matrix.pre_rotate(rotation)
    pix = page.get_pixmap(matrix=matrix, alpha=False)
    mode = "RGB" if pix.n < 4 else "RGBA"
    img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
    if mode == "RGBA":
        img = img.convert("RGB")
    return img


def _resize_longest(img: Image.Image, max_dim: int) -> Image.Image:
    w, h = img.size
    if max(w, h) <= max_dim:
        return img
    if w >= h:
        new_w = max_dim
        new_h = int(h * (max_dim / w))
    else:
        new_h = max_dim
        new_w = int(w * (max_dim / h))
    return img.resize((new_w, new_h), Image.LANCZOS)


def _apply_scan_preprocess(img: Image.Image) -> Image.Image:
    img = ImageOps.autocontrast(img)
    img = ImageEnhance.Sharpness(img).enhance(1.2)
    return img


def render_page(
    page: fitz.Page,
    route: str,
    max_dim: int = 1288,
    scan_preprocess: bool = False,
    rotation: int = 0,
) -> RenderResult:
    dpi = 200 if route == "anchored" else 300
    img = _render_page_to_image(page, dpi=dpi, rotation=rotation)
    if scan_preprocess and route == "unanchored":
        img = _apply_scan_preprocess(img)
    img = _resize_longest(img, max_dim)

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    png_bytes = buf.getvalue()
    if len(png_bytes) < 20 * 1024 * 1024:
        return RenderResult(
            image_bytes=png_bytes,
            mime_type="image/png",
            width=img.width,
            height=img.height,
            format="PNG",
        )

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85, optimize=True)
    jpg_bytes = buf.getvalue()
    return RenderResult(
        image_bytes=jpg_bytes,
        mime_type="image/jpeg",
        width=img.width,
        height=img.height,
        format="JPEG",
    )
