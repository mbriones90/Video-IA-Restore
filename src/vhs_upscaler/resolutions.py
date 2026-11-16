"""Resoluciones de salida: proporcion nativa vs lienzo 16:9.

Para cada altura (720, 1080, 1440, 2160) se ofrecen dos variantes:
- Nativa: el ancho se calcula con la proporcion real del video. No agrega
  barras ni modifica nada (para VHS 4:3: 960x720, 1440x1080, 2880x2160...).
- 16:9: lienzo panoramico estandar. Si el video no es 16:9, MODIFICA EL LIENZO
  agregando barras negras (pillarbox/letterbox). NUNCA estira la imagen.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd

STANDARD_HEIGHTS = (720, 1080, 1440, 2160)
MAX_WIDTH = 3840  # techo 4K


@dataclass(frozen=True)
class OutputSize:
    key: str  # "1080_native" | "1080_169"
    width: int
    height: int
    native: bool
    adds_bars: bool
    label: str


def aspect_text(width: int, height: int) -> str:
    g = gcd(width, height) or 1
    return f"{width // g}:{height // g}"


def native_dimensions(src_w: int, src_h: int, height: int, sar: float = 1.0) -> tuple[int, int]:
    """Ancho/alto pares que conservan la proporcion VISIBLE del original.

    `sar` es la proporcion del pixel: una captura NTSC marcada 8:9 tiene
    pixeles mas estrechos que altos, y hay que ensanchar la imagen al pasarla
    a pixeles cuadrados o sale comprimida de lado.
    """
    width = max(2, round(height * (src_w * sar) / src_h / 2) * 2)
    if width > MAX_WIDTH:  # proporciones muy anchas: se limita el ancho a 4K
        width = MAX_WIDTH
        height = max(2, round(width * src_h / (src_w * sar) / 2) * 2)
    return width, height


def build_output_sizes(src_w: int, src_h: int, sar: float = 1.0) -> list[OutputSize]:
    sizes: list[OutputSize] = []
    # El texto de la proporcion describe lo que se VE, no la rejilla de pixeles.
    disp_w = max(2, round(src_w * sar))
    ar = aspect_text(disp_w, src_h)
    is_169 = abs((disp_w / src_h) - (16 / 9)) < 0.01
    for h in STANDARD_HEIGHTS:
        nw, nh = native_dimensions(src_w, src_h, h, sar)
        sizes.append(OutputSize(
            key=f"{h}_native", width=nw, height=nh, native=True, adds_bars=False,
            label=f"{nw}x{nh} - Proporcion nativa {ar} (SIN barras, no modifica la imagen)",
        ))
        w169 = round(h * 16 / 9 / 2) * 2
        if is_169:
            detail = "tu video ya es 16:9: SIN barras"
        else:
            detail = "MODIFICA el lienzo: agrega barras negras, NO estira la imagen"
        sizes.append(OutputSize(
            key=f"{h}_169", width=w169, height=h, native=False, adds_bars=not is_169,
            label=f"{w169}x{h} - Lienzo 16:9 ({detail})",
        ))
    return sizes
