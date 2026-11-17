"""Gestion de modelos: manifiesto, descarga selectiva y verificacion.

El mismo archivo ONNX sirve para CPU, CUDA y DirectML; lo que cambia entre
PCs es el runtime que instala UV. El campo "backends" del manifiesto permite
en el futuro tener variantes exclusivas (por ejemplo fp16 solo para CUDA).
"""

from __future__ import annotations

import hashlib
import json
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .paths import MANIFEST_PATH, MODELS_DIR, ensure_dirs

# Modelo sin IA que siempre esta disponible, incluso sin descargar nada.
BICUBIC_ID = "bicubic"
BICUBIC_NAME = "Bicubico (sin IA, prueba rapida)"


@dataclass
class ModelEntry:
    id: str
    name: str
    file: str
    url: str
    sha256: str
    scale: int
    size_mb: float
    backends: list[str]
    description: str
    # Pesos convertidos a ncnn (backend Vulkan). Viajan con el proyecto;
    # se regeneran con tools/make_ncnn.py.
    ncnn_param: str = ""
    ncnn_bin: str = ""

    @property
    def path(self) -> Path:
        return MODELS_DIR / self.file


def load_manifest() -> list[ModelEntry]:
    data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return [ModelEntry(**m) for m in data["models"]]


def get_model(model_id: str) -> ModelEntry:
    for m in load_manifest():
        if m.id == model_id:
            return m
    raise KeyError(f"Modelo desconocido: {model_id}")


def is_downloaded(entry: ModelEntry) -> bool:
    return entry.path.exists() and entry.path.stat().st_size > 0


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(entry: ModelEntry) -> bool:
    if not is_downloaded(entry):
        return False
    if not entry.sha256:
        return True
    return _sha256(entry.path) == entry.sha256.lower()


def download(entry: ModelEntry, log=print) -> Path:
    """Descarga un modelo con verificacion de integridad SHA-256."""
    ensure_dirs()
    if verify(entry):
        log(f"Ya descargado y verificado: {entry.name}")
        return entry.path
    if is_downloaded(entry):
        log(f"Archivo corrupto o incompleto, se descarga de nuevo: {entry.file}")
        entry.path.unlink()
    tmp = entry.path.with_suffix(entry.path.suffix + ".part")
    log(f"Descargando {entry.name} ({entry.size_mb:.1f} MB)...")
    req = urllib.request.Request(entry.url, headers={"User-Agent": "video-ia-restore"})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp, tmp.open("wb") as f:
            shutil.copyfileobj(resp, f, length=1024 * 512)
    except Exception:
        tmp.unlink(missing_ok=True)  # no dejar un .part huerfano
        raise
    if entry.sha256:
        digest = _sha256(tmp)
        if digest != entry.sha256.lower():
            tmp.unlink(missing_ok=True)
            raise ValueError(
                f"Checksum incorrecto para {entry.file}: {digest}. Descarga descartada."
            )
    tmp.replace(entry.path)
    log(f"Listo: {entry.path}")
    return entry.path


def download_for_backend(backend: str, log=print) -> list[Path]:
    """Descarga solo los modelos que aplican al backend elegido en esta PC."""
    paths = []
    for entry in load_manifest():
        if backend in entry.backends:
            paths.append(download(entry, log=log))
        else:
            log(f"Omitido (no aplica a '{backend}'): {entry.name}")
    return paths
