"""Rutas del proyecto, resueltas desde la raiz (donde esta iniciar.bat)."""

from pathlib import Path

# paths.py esta en <raiz>/src/vhs_upscaler/paths.py
ROOT_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT_DIR / "models"
MANIFEST_PATH = MODELS_DIR / "manifest.json"
CONFIG_DIR = ROOT_DIR / "config"
CONFIG_PATH = CONFIG_DIR / "config.local.json"
OUTPUT_DIR = ROOT_DIR / "output"


def ensure_dirs() -> None:
    for d in (MODELS_DIR, CONFIG_DIR, OUTPUT_DIR):
        d.mkdir(parents=True, exist_ok=True)
