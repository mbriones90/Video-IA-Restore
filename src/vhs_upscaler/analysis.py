"""Analisis automatico de la cinta: paridad de campos, overscan y rango.

Tres cosas que hasta ahora habia que adivinar a mano y que, si se fallan,
arruinan la restauracion en silencio:

- **Paridad de campos** (TFF/BFF): si se invierte, el movimiento sale a saltos
  o con un temblor de un campo. Se mide con el filtro `idet` de FFmpeg.
- **Overscan / franja de cabezales**: las barras negras y el ruido de los
  bordes. Se mide con `cropdetect`.
- **Rango de color** (limitado 16-235 vs completo 0-255): si se asume el
  equivocado, los negros se aplastan o se lavan. Se lee de la cabecera.

Todo se muestrea en varios puntos del video, nunca en uno solo: el principio
de una cinta suele ser negro o barras de color y daria resultados absurdos.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass

from .ffmpeg_utils import VideoInfo, get_ffmpeg_exe
from .logging_setup import get_logger

log = get_logger(__name__)

# Puntos de muestreo como fraccion de la duracion: se evitan los extremos
# (cabecera negra, barras de color, cola sin senal).
SAMPLE_POINTS = (0.15, 0.35, 0.55, 0.75)


def _flags() -> int:
    return getattr(subprocess, "CREATE_NO_WINDOW", 0)


def _run_filter(path: str, second: float, vf: str, frames: int) -> str:
    """Corre un filtro de analisis y devuelve lo que FFmpeg escribio en stderr."""
    cmd = [
        get_ffmpeg_exe(), "-hide_banner", "-nostats",
        "-ss", f"{second:.3f}", "-i", str(path),
        "-map", "0:v:0", "-an", "-sn", "-dn",
        "-vf", vf, "-frames:v", str(frames), "-f", "null", "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, creationflags=_flags())
    return proc.stderr.decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Paridad de campos
# ---------------------------------------------------------------------------

@dataclass
class ParityResult:
    parity: str          # "tff" | "bff" | "progressive" | "auto" (sin conclusion)
    tff: int
    bff: int
    progressive: int
    undetermined: int
    confidence: float    # 0.0 - 1.0

    @property
    def interlaced(self) -> bool:
        return self.parity in ("tff", "bff")

    def summary(self) -> str:
        if self.parity == "progressive":
            return (f"Material PROGRESIVO (no entrelazado) con {self.confidence:.0%} "
                    f"de confianza: puedes desactivar el desentrelazado.")
        if self.parity == "auto":
            return ("No se pudo determinar la paridad con seguridad "
                    f"(TFF={self.tff}, BFF={self.bff}, progresivo={self.progressive}). "
                    "Se deja en 'Auto'.")
        nombre = "TFF - campo superior primero" if self.parity == "tff" \
            else "BFF - campo inferior primero"
        return (f"Paridad detectada: {nombre} ({self.confidence:.0%} de confianza, "
                f"TFF={self.tff} / BFF={self.bff} / progresivo={self.progressive}).")


_IDET_RE = re.compile(
    r"Multi frame detection:\s*TFF:\s*(\d+)\s*BFF:\s*(\d+)\s*"
    r"Progressive:\s*(\d+)\s*Undetermined:\s*(\d+)"
)


def detect_field_parity(path: str, info: VideoInfo, frames_per_sample: int = 120) -> ParityResult:
    """Decide TFF / BFF / progresivo con el filtro `idet` en varios puntos.

    `idet` necesita movimiento para decidir: por eso se analizan varios tramos
    y se suman los votos en vez de fiarse de una sola muestra.
    """
    tff = bff = prog = undet = 0
    for frac in SAMPLE_POINTS:
        second = max(0.0, info.duration * frac)
        if second >= max(info.duration - 1.0, 0.0):
            continue
        stderr = _run_filter(path, second, "idet", frames_per_sample)
        # idet imprime un resumen por muestra; se toma el de "Multi frame",
        # que es el mas fiable porque mira la secuencia, no el frame suelto.
        matches = _IDET_RE.findall(stderr)
        if not matches:
            continue
        a, b, c, d = matches[-1]
        tff += int(a); bff += int(b); prog += int(c); undet += int(d)

    total = tff + bff + prog + undet
    if total == 0:
        log.warning("idet no devolvio datos para %s", path)
        return ParityResult("auto", 0, 0, 0, 0, 0.0)

    interlaced = tff + bff
    # Margen deliberadamente alto: equivocarse aqui cuesta una cinta entera.
    if prog > interlaced * 2 and prog / total > 0.7:
        return ParityResult("progressive", tff, bff, prog, undet, prog / total)
    if interlaced == 0:
        return ParityResult("auto", tff, bff, prog, undet, 0.0)
    if tff >= bff * 2:
        return ParityResult("tff", tff, bff, prog, undet, tff / max(interlaced, 1))
    if bff >= tff * 2:
        return ParityResult("bff", tff, bff, prog, undet, bff / max(interlaced, 1))
    return ParityResult("auto", tff, bff, prog, undet, 0.0)


# ---------------------------------------------------------------------------
# Overscan / franja de cabezales
# ---------------------------------------------------------------------------

@dataclass
class CropResult:
    left: int
    top: int
    right: int
    bottom: int

    @property
    def any(self) -> bool:
        return any((self.left, self.top, self.right, self.bottom))

    def summary(self, width: int, height: int) -> str:
        if not self.any:
            return "No se detectaron bordes negros ni franja de cabezales."
        return (f"Bordes detectados: izq={self.left}, arr={self.top}, "
                f"der={self.right}, aba={self.bottom} px "
                f"(imagen util {width - self.left - self.right}x"
                f"{height - self.top - self.bottom}).")


_CROP_RE = re.compile(r"crop=(\d+):(\d+):(\d+):(\d+)")


def detect_crop(path: str, info: VideoInfo, frames_per_sample: int = 90,
                limit: int = 24) -> CropResult:
    """Propone el recorte de overscan con `cropdetect`.

    Se queda con el recuadro MAS CONSERVADOR de todas las muestras (el que
    menos recorta): asi nunca se corta imagen real porque una escena oscura
    hiciera creer que hay borde negro.
    """
    best: CropResult | None = None
    for frac in SAMPLE_POINTS:
        second = max(0.0, info.duration * frac)
        if second >= max(info.duration - 1.0, 0.0):
            continue
        # reset=0 => acumula sobre toda la muestra en vez de reiniciar.
        stderr = _run_filter(
            path, second, f"cropdetect=limit={limit}:round=2:reset=0", frames_per_sample
        )
        matches = _CROP_RE.findall(stderr)
        if not matches:
            continue
        w, h, x, y = (int(v) for v in matches[-1])
        if w <= 0 or h <= 0 or w > info.width or h > info.height:
            continue
        cand = CropResult(
            left=x, top=y,
            right=max(0, info.width - w - x), bottom=max(0, info.height - h - y),
        )
        if best is None:
            best = cand
        else:
            best = CropResult(
                left=min(best.left, cand.left), top=min(best.top, cand.top),
                right=min(best.right, cand.right), bottom=min(best.bottom, cand.bottom),
            )

    if best is None:
        return CropResult(0, 0, 0, 0)
    # El pipeline exige multiplos de 2 (croma 4:2:0); se redondea hacia abajo
    # para no recortar de mas.
    return CropResult(*(v - (v % 2) for v in (best.left, best.top, best.right, best.bottom)))


# ---------------------------------------------------------------------------
# Rango y espacio de color
# ---------------------------------------------------------------------------

@dataclass
class ColorInfo:
    range: str = "limited"        # limited | full
    matrix: str = "bt601"         # bt601 | bt709
    detected_range: bool = False  # True si venia etiquetado en el archivo
    detected_matrix: bool = False

    def summary(self) -> str:
        r = "completo (0-255, PC)" if self.range == "full" else "limitado (16-235, TV)"
        m = "BT.709 (HD)" if self.matrix == "bt709" else "BT.601 / SMPTE 170M (SD)"
        origen_r = "etiquetado en el archivo" if self.detected_range else "supuesto"
        origen_m = "etiquetado en el archivo" if self.detected_matrix else "supuesto por la resolucion"
        return f"Rango {r} ({origen_r}) | Matriz {m} ({origen_m})."


_STREAM_RE = re.compile(r"Stream #\d+:\d+.*?:\s*Video:\s*(.*)")


def detect_color(path: str, info: VideoInfo) -> ColorInfo:
    """Lee rango y matriz de la cabecera; si no estan, los deduce del tamano.

    FFmpeg los imprime entre parentesis tras el pix_fmt, p. ej.
    `yuv420p(tv, bt470bg/bt470bg/smpte170m)` o `yuv420p(pc, bt709)`.
    """
    cmd = [get_ffmpeg_exe(), "-hide_banner", "-i", str(path)]
    proc = subprocess.run(cmd, capture_output=True, creationflags=_flags())
    text = proc.stderr.decode("utf-8", errors="replace")

    out = ColorInfo()
    # SD (<= 576 lineas) es BT.601; por encima, BT.709. Es el valor por defecto
    # correcto para una captura de VHS y solo se usa si el archivo no lo dice.
    out.matrix = "bt601" if info.height <= 576 else "bt709"

    m = _STREAM_RE.search(text)
    if m:
        # La descripcion trae VARIOS parentesis: "h264 (High), yuv420p(tv,
        # bt470bg/bt470bg/smpte170m), 720x480". Mirar solo el primero caia
        # siempre en "(High)" y no detectaba nada nunca, asi que se juntan
        # todos y se busca sobre el conjunto.
        flat = " ".join(re.findall(r"\(([^)]*)\)", m.group(1))).lower()
        flat = flat.replace(",", " ").replace("/", " ")
        if re.search(r"\bpc\b|\bfull\b", flat):
            out.range, out.detected_range = "full", True
        elif re.search(r"\btv\b|\blimited\b", flat):
            out.range, out.detected_range = "limited", True
        if "bt709" in flat:
            out.matrix, out.detected_matrix = "bt709", True
        elif "smpte170m" in flat or "bt470bg" in flat or "bt601" in flat:
            out.matrix, out.detected_matrix = "bt601", True
    return out


# ---------------------------------------------------------------------------
# Analisis completo
# ---------------------------------------------------------------------------

@dataclass
class Analysis:
    parity: ParityResult
    crop: CropResult
    color: ColorInfo

    def summary(self, info: VideoInfo) -> str:
        return "\n".join((
            self.parity.summary(),
            self.crop.summary(info.width, info.height),
            self.color.summary(),
        ))


def analyze(path: str, info: VideoInfo, progress=None) -> Analysis:
    """Corre los tres analisis. `progress(texto)` se llama entre pasos."""
    def step(msg: str) -> None:
        log.info(msg)
        if progress:
            progress(msg)

    step("Analizando paridad de campos (idet)...")
    parity = detect_field_parity(path, info)
    step("Analizando bordes negros y overscan (cropdetect)...")
    crop = detect_crop(path, info)
    step("Leyendo rango y espacio de color...")
    color = detect_color(path, info)
    return Analysis(parity, crop, color)
