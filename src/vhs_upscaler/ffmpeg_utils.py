"""Localizacion de FFmpeg y lectura de metadatos de video.

FFmpeg se obtiene del paquete imageio-ffmpeg (binario incluido), asi no hace
falta instalar nada a mano en Windows. Si el usuario tiene ffmpeg en el PATH,
se prefiere ese.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


def get_ffmpeg_exe() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as exc:
        raise RuntimeError(
            "No se encontro FFmpeg. Ejecuta iniciar.bat para instalar las dependencias."
        ) from exc


@dataclass
class VideoInfo:
    path: Path
    width: int
    height: int
    fps: float
    frame_count: int
    duration: float
    # Proporcion del PIXEL (sample aspect ratio). Una captura NTSC a 720x480
    # suele venir marcada 8:9 (=0.889): sus pixeles no son cuadrados y tratarla
    # como si lo fueran estira la imagen un 12 %.
    sar: float = 1.0

    @property
    def display_width(self) -> float:
        """Ancho en pixeles CUADRADOS (lo que de verdad se ve)."""
        return self.width * self.sar

    def summary(self) -> str:
        mins, secs = divmod(int(self.duration), 60)
        extra = ""
        if abs(self.sar - 1.0) > 0.01:
            extra = f"  |  pixel {self.sar:.3f} (se ve como {self.display_width:.0f}x{self.height})"
        return (
            f"{self.width}x{self.height}  |  {self.fps:.2f} fps  |  "
            f"{mins:02d}:{secs:02d} min  |  ~{self.frame_count} frames{extra}"
        )


_SAR_RE = re.compile(r"\[SAR (\d+):(\d+)")


def probe_sar(path: str | Path) -> float:
    """Lee la proporcion de pixel de la cabecera. 1.0 si no esta o no se puede."""
    try:
        exe = get_ffmpeg_exe()
    except RuntimeError:
        return 1.0
    try:
        proc = subprocess.run(
            [exe, "-hide_banner", "-i", str(path)], capture_output=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired):
        return 1.0
    m = _SAR_RE.search(proc.stderr.decode("utf-8", errors="replace"))
    if not m:
        return 1.0
    num, den = int(m.group(1)), int(m.group(2))
    if num <= 0 or den <= 0:
        return 1.0
    return num / den


def probe_video(path: str | Path) -> VideoInfo:
    import cv2

    path = Path(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {path}")
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = count / fps if fps > 0 else 0.0
        return VideoInfo(path, width, height, fps, count, duration, probe_sar(path))
    finally:
        cap.release()


def read_frame(path: str | Path, second: float = 0.0):
    """Lee un frame BGR en el segundo indicado (para vistas previas)."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(second * fps)))
        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
        if not ok:
            raise RuntimeError("No se pudo leer ningun frame del video.")
        return frame
    finally:
        cap.release()


def run_quiet(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        args,
        capture_output=True,
        text=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
