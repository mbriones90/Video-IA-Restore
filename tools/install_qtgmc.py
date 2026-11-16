"""Instala QTGMC (VapourSynth) para el desentrelazado profesional.

Hace todo lo necesario en esta PC:
1. uv sync --group qtgmc  (VapourSynth + plugins que estan en pip)
2. Descarga e instala RemoveGrainVS.dll y TTempSmooth.dll en el directorio
   de plugins de VapourSynth (los dos que no estan en pip).

Uso:
  uv run python tools/install_qtgmc.py
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "runtime" / "cache"


def find_uv() -> list[str]:
    """UV del PATH o el portable que incluye el proyecto."""
    uv = shutil.which("uv")
    if uv:
        return [uv]
    portable = ROOT / "runtime" / "bin" / "uv.exe"
    if portable.is_file():
        return [str(portable)]
    return ["uv"]


def sync_groups() -> list[str]:
    """Grupos para uv sync: qtgmc + el backend de esta PC (config/backend.txt).

    Incluir el backend es imprescindible: un sync que solo pida el grupo qtgmc
    desinstalaria onnxruntime-gpu / onnxruntime-directml si estaban presentes.
    """
    groups = ["--group", "qtgmc"]
    backend_file = ROOT / "config" / "backend.txt"
    if backend_file.is_file():
        backend = backend_file.read_text(encoding="utf-8").strip().lower()
        if backend in ("cpu", "cuda", "directml"):
            groups += ["--group", backend]
    return groups


PLUGIN_ZIPS = {
    "RemoveGrainVS.dll": (
        "https://github.com/vapoursynth/vs-removegrain/releases/download/R1/removegrain-r1.7z",
        "win64",  # subcarpeta dentro del 7z
    ),
    "TTempSmooth.dll": (
        "https://github.com/HomeOfVapourSynthEvolution/VapourSynth-TTempSmooth/releases/download/r4.1/TTempSmooth-r4.1-win64.7z",
        "",
    ),
}
SEVEN_ZIP = "https://www.7-zip.org/a/7zr.exe"


def main() -> None:
    ap = argparse.ArgumentParser(description="Instala QTGMC (VapourSynth)")
    ap.add_argument("--skip-sync", action="store_true",
                    help="No ejecutar uv sync (el lanzador ya lo hizo)")
    args = ap.parse_args()

    if not args.skip_sync:
        print("[1/3] Sincronizando dependencias del grupo qtgmc...")
        subprocess.run(find_uv() + ["sync", *sync_groups()], cwd=ROOT, check=True)
    else:
        print("[1/3] Sync omitido (--skip-sync)")

    print("[2/3] Descargando extractor 7zr y plugins RemoveGrain/TTempSmooth...")
    CACHE.mkdir(parents=True, exist_ok=True)
    sevenzr = CACHE / "7zr.exe"
    if not sevenzr.exists():
        urllib.request.urlretrieve(SEVEN_ZIP, sevenzr)

    sys.path.insert(0, str(ROOT / "src"))
    import vapoursynth

    plugdir = Path(vapoursynth.get_plugin_dir())
    for dll_name, (url, subdir) in PLUGIN_ZIPS.items():
        if (plugdir / dll_name).exists():
            print(f"  Ya instalado: {dll_name}")
            continue
        archive = CACHE / Path(url).name
        if not archive.exists():
            print(f"  Descargando {Path(url).name}...")
            req = urllib.request.Request(url, headers={"User-Agent": "vhs-upscaler"})
            with urllib.request.urlopen(req, timeout=120) as resp, open(archive, "wb") as f:
                f.write(resp.read())
        dest = CACHE / archive.stem
        subprocess.run([str(sevenzr), "x", str(archive), f"-o{dest}", "-y"],
                       check=True, capture_output=True)
        src = dest / subdir / dll_name if subdir else dest / dll_name
        if not src.exists():
            candidates = list(dest.rglob(dll_name))
            if not candidates:
                raise RuntimeError(f"No se encontro {dll_name} en {archive.name}")
            src = candidates[0]
        dest_file = plugdir / dll_name
        dest_file.write_bytes(src.read_bytes())
        print(f"  Instalado: {dll_name} -> {dest_file}")

    print("[3/3] Verificando instalacion...")
    probe = (
        "import vapoursynth as vs; from vapoursynth import core; "
        "print('Plugins OK:', hasattr(core, 'ffms2'), hasattr(core, 'mv'), "
        "hasattr(core, 'mvu'), hasattr(core, 'rgvs'), hasattr(core, 'fmtc'), "
        "hasattr(core, 'znedi3'), hasattr(core, 'resize2'))"
    )
    subprocess.run([sys.executable, "-c", probe], check=True)
    print("\nQTGMC listo. En la app, paso 2, elige 'QTGMC' como metodo de desentrelazado.")


if __name__ == "__main__":
    main()
