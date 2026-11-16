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
    capture: "CaptureCheck"

    def summary(self, info: VideoInfo) -> str:
        lineas = [self.parity.summary()]
        # El aviso del reescalado va JUNTO al veredicto de paridad, no al final:
        # es justo lo que lo hace no concluyente, y separarlos invita a leer
        # "progresivo" y dar el tema por cerrado.
        aviso = self.parity_caveat()
        if aviso:
            lineas.append(aviso)
        lineas.append(self.crop.summary(info.width, info.height))
        lineas.append(self.color.summary())
        lineas.append(self.capture.summary())
        return "\n".join(lineas)

    def parity_caveat(self) -> str:
        """Matiza un 'progresivo' cuando la captura venia reescalada.

        `idet` detecta PEINADO, no entrelazado. Un reescalado vertical mezcla
        lineas vecinas, que en una senal entrelazada pertenecen a campos (y
        por tanto a instantes) distintos: el dentado desaparece y a partir de
        ahi idet dira "progresivo" para siempre, aunque el original tuviera
        campos y el movimiento haya quedado emborronado.
        """
        if self.parity.parity != "progressive" or not self.capture.rescaled:
            return ""
        return (
            "  OJO: este veredicto NO es concluyente. La imagen viene "
            "reescalada verticalmente, y eso borra el peinado que idet busca. "
            "Lo seguro es que en ESTE archivo ya no queda nada que "
            "desentrelazar; si la cinta llego entrelazada a la capturadora, no "
            "se puede saber desde aqui. Para salir de dudas, graba 30 s a la "
            "resolucion nativa del dispositivo, sin escalar, y analiza eso."
        )


# ---------------------------------------------------------------------------
# Resolucion de captura
# ---------------------------------------------------------------------------

# Resoluciones a las que entrega una capturadora de verdad. Si el archivo no
# es ninguna de estas, alguien escalo por el camino (el lienzo de OBS, casi
# siempre).
CAPTURE_STANDARDS = {
    (720, 480): "NTSC D1", (720, 486): "NTSC D1 de 486 lineas",
    (704, 480): "NTSC 4SIF", (640, 480): "NTSC de pixel cuadrado",
    (352, 480): "Half-D1 NTSC",
    (720, 576): "PAL D1", (704, 576): "PAL 4CIF",
    (768, 576): "PAL de pixel cuadrado", (352, 576): "Half-D1 PAL",
}
# Lineas utiles que entrega cada sistema. Es el listón contra el que se mide
# cuanta resolucion se tiro antes de grabar.
SYSTEM_LINES = {"ntsc": 480, "pal": 576}


@dataclass
class CaptureCheck:
    width: int             # tamano del fotograma en el archivo
    height: int
    content_width: int     # imagen util, ya sin barras y con el pixel corregido
    content_height: int
    system: str            # ntsc | pal | desconocido
    expected_lines: int
    standard_name: str     # "" si no coincide con ninguna resolucion de captura
    rescaled: bool         # se perdio resolucion vertical antes de grabar
    lost_pct: float        # % de pixeles perdidos frente a la captura nativa

    @property
    def severity(self) -> str:
        if self.rescaled:
            return "grave"
        return "ok" if self.standard_name else "aviso"

    def summary(self) -> str:
        if self.severity == "ok":
            return (f"Resolucion de captura: {self.width}x{self.height} "
                    f"({self.standard_name}). Nativa, nada que objetar.")
        if self.severity == "aviso":
            return (f"Resolucion: {self.width}x{self.height}, con "
                    f"{self.content_width}x{self.content_height} de imagen util. "
                    f"No es una resolucion de captura estandar, pero conserva "
                    f"las {self.expected_lines} lineas del sistema: no se "
                    f"perdio detalle.")
        objetivo = SYSTEM_LINES.get(self.system, self.expected_lines)
        ancho = 720 if objetivo in (480, 576) else self.content_width
        return (
            f"AVISO - captura reescalada: el archivo es {self.width}x{self.height} "
            f"y la imagen util son {self.content_width}x{self.content_height}, "
            f"cuando una captura {self.system.upper()} nativa da "
            f"{ancho}x{objetivo}.\n"
            f"  Se perdio el {self.lost_pct:.0f}% de los pixeles ANTES de grabar, "
            f"y eso no lo recupera ningun escalado con IA. Revisa en OBS el "
            f"tamano del lienzo (Base), la resolucion de salida y el tamano de "
            f"la fuente dentro de la escena: los tres deberian dejar la imagen "
            f"a {ancho}x{objetivo} sin encoger."
        )


def check_capture(info: VideoInfo, crop: CropResult | None = None) -> CaptureCheck:
    """Decide si la captura llega a la resolucion nativa de su sistema.

    El sistema se deduce de los FPS, no del alto: el alto es justo lo que
    puede venir falseado por un reescalado.
    """
    fps = info.fps or 0.0
    # PAL primero y con rangos estrechos: un rango NTSC generoso se tragaba
    # los 25 fps de PAL y lo clasificaba al reves.
    if 24.5 <= fps <= 25.5 or 49.0 <= fps <= 51.0:
        system = "pal"
    elif 29.0 <= fps <= 30.5 or 58.0 <= fps <= 61.0:
        system = "ntsc"
    else:
        system = "desconocido"
    expected = SYSTEM_LINES.get(system, 480)

    crop = crop or CropResult(0, 0, 0, 0)
    content_h = max(1, info.height - crop.top - crop.bottom)
    content_w = max(1, round((info.width - crop.left - crop.right) * (info.sar or 1.0)))

    standard = CAPTURE_STANDARDS.get((info.width, info.height), "")
    # Margen del 5 %: 486 contra 480, o un recorte de un par de lineas, no son
    # un reescalado.
    rescaled = system != "desconocido" and content_h < expected * 0.95
    nativo = (720 if expected in (480, 576) else content_w) * expected
    lost = max(0.0, 100.0 * (1.0 - (content_w * content_h) / nativo))
    return CaptureCheck(
        width=info.width, height=info.height,
        content_width=content_w, content_height=content_h,
        system=system, expected_lines=expected, standard_name=standard,
        rescaled=rescaled, lost_pct=lost,
    )


def analyze(path: str, info: VideoInfo, progress=None) -> Analysis:
    """Corre los analisis. `progress(texto)` se llama entre pasos."""
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
    step("Comprobando la resolucion de captura...")
    capture = check_capture(info, crop)
    return Analysis(parity, crop, color, capture)
