"""Utilidades comunes de las pruebas.

Las pruebas que necesitan FFmpeg, VapourSynth o GPU llevan la marca `slow`:

    uv run --group dev pytest                 # todo
    uv run --group dev pytest -m "not slow"   # solo lo que corre en cualquier PC
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _flags() -> int:
    return getattr(subprocess, "CREATE_NO_WINDOW", 0)


@pytest.fixture(scope="session")
def ffmpeg() -> str:
    from vhs_upscaler.ffmpeg_utils import get_ffmpeg_exe

    try:
        return get_ffmpeg_exe()
    except RuntimeError:
        pytest.skip("FFmpeg no disponible")


@pytest.fixture(scope="session")
def sample_video(tmp_path_factory, ffmpeg) -> Path:
    """Clip sintetico de 2 s: barato, deterministico y sin depender de VIDEOS/.

    Se genera a 640x480 (4:3, como una captura de VHS) con una pista de audio,
    para que las pruebas del pipeline ejerciten tambien el mux final.
    """
    out = tmp_path_factory.mktemp("media") / "sample.mp4"
    cmd = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "testsrc=size=640x480:rate=30:duration=2",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
        "-c:a", "aac", "-shortest", str(out),
    ]
    proc = subprocess.run(cmd, capture_output=True, creationflags=_flags())
    if proc.returncode != 0 or not out.exists():
        pytest.skip("No se pudo generar el clip de prueba con FFmpeg")
    return out


@pytest.fixture
def jobs_dir(tmp_path, monkeypatch):
    """Aisla output/.jobs para que las pruebas no toquen trabajos reales."""
    from vhs_upscaler import pipeline

    work = tmp_path / "jobs"
    work.mkdir()
    monkeypatch.setattr(pipeline, "JOBS_DIR", work)
    return work
