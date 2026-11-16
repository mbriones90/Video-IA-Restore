"""Regenera los pesos ncnn (backend Vulkan) a partir de los ONNX del manifiesto.

Los archivos .ncnn.param/.ncnn.bin ya vienen con el proyecto; solo hace falta
ejecutar esto si actualizas los modelos ONNX o agregas uno nuevo.

Uso:
  uv run python tools/make_ncnn.py               Convierte todos los modelos
  uv run python tools/make_ncnn.py --model realesr-general-x4v3

Requisito: los .onnx ya descargados en models/ (iniciar.bat lo hace).
La conversion usa pnnx en un entorno efimero (no contamina el proyecto).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vhs_upscaler.models import load_manifest  # noqa: E402


def convert(onnx_path: Path) -> tuple[Path, Path]:
    print(f"Convirtiendo {onnx_path.name} ...")
    result = subprocess.run(
        ["uv", "run", "--no-project", "--with", "pnnx", "pnnx", str(onnx_path),
         "inputshape=[1,3,64,64]"],
        cwd=ROOT, capture_output=True, text=True,
    )
    # pnnx termina con exit code 1 en pasos opcionales aunque la conversion ncnn OK
    stem = onnx_path.stem.replace("-", "_")
    param = onnx_path.with_name(f"{stem}.ncnn.param")
    binf = onnx_path.with_name(f"{stem}.ncnn.bin")
    if not param.exists() or not binf.exists():
        print(result.stdout[-2000:] if result.stdout else "", result.stderr[-2000:] if result.stderr else "")
        raise RuntimeError(f"pnnx no genero {param.name}")
    # Limpieza de archivos intermedios
    for extra in onnx_path.parent.glob(f"{onnx_path.stem.replace('-', '_')}*.pnnx.*"):
        extra.unlink(missing_ok=True)
    for extra in (onnx_path.with_name(onnx_path.stem + "_ncnn.py"),
                  onnx_path.with_name(onnx_path.stem + "_pnnx.py"),
                  onnx_path.with_name(onnx_path.stem + ".pnnxsim.onnx")):
        extra.unlink(missing_ok=True)
    return param, binf


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="id del modelo (por defecto: todos)")
    args = parser.parse_args()

    for entry in load_manifest():
        if args.model and entry.id != args.model:
            continue
        if not entry.path.exists():
            print(f"Omitido (ONNX no descargado): {entry.name}")
            continue
        param, binf = convert(entry.path)
        print(f"  OK: {param.name} + {binf.name}")
        print(f"  Recuerda registrarlos en models/manifest.json de '{entry.id}' "
              f"si aun no lo estan.")


if __name__ == "__main__":
    main()
