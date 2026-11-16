"""Descarga selectiva de modelos segun el backend de esta PC.

Uso:
  uv run python -m vhs_upscaler.tools.download_models --backend directml
  uv run python -m vhs_upscaler.tools.download_models --backend cuda --model realesrgan-x4plus
"""

from __future__ import annotations

import argparse

from ..models import download, download_for_backend, get_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Descarga de modelos de IA para VHS Upscaler")
    parser.add_argument("--backend", default="cpu", choices=["cpu", "cuda", "directml"],
                        help="Backend elegido para esta PC")
    parser.add_argument("--model", default=None, help="Descargar solo un modelo por su id")
    args = parser.parse_args()

    if args.model:
        download(get_model(args.model))
    else:
        download_for_backend(args.backend)


if __name__ == "__main__":
    main()
