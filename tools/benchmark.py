"""Mide la velocidad de escalado de CADA dispositivo disponible en ESTA PC.

Asi puedes comparar en tu propia maquina (p. ej. CUDA vs Vulkan en la RTX 4070)
sin adivinar numeros.

Uso:
  uv run python tools/benchmark.py
  uv run python tools/benchmark.py --model realesrgan-x4plus --frames 6
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from vhs_upscaler.hardware import detect_hardware  # noqa: E402
from vhs_upscaler.inference import create_upscaler  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark de dispositivos de escalado")
    ap.add_argument("--frames", type=int, default=12, help="Iteraciones medidas (tras 2 de calentamiento)")
    ap.add_argument("--size", default="720x480", help="Tamano del frame de prueba, ej. 720x480")
    ap.add_argument("--model", default="realesr-general-x4v3")
    args = ap.parse_args()

    w, h = (int(x) for x in args.size.split("x"))
    hw = detect_hardware()
    print(f"PC: NVIDIA={hw.nvidia_gpu or 'no'} | GPUs Vulkan={hw.vulkan_gpus} | "
          f"Providers ORT={', '.join(hw.available_providers) or 'ninguno'}")
    candidates = [d for d in ("cuda", "vulkan", "directml", "cpu") if hw.device_available(d)]
    print(f"Frame {w}x{h} -> x4 | modelo {args.model} | dispositivos: {', '.join(candidates)}\n")

    rng = np.random.default_rng(1)
    frame = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    for device in candidates:
        try:
            up = create_upscaler(args.model, device, log=lambda *a: None)
            up.upscale(frame)
            up.upscale(frame)  # calentamiento (carga de pesos/compilacion)
            t0 = time.perf_counter()
            for _ in range(args.frames):
                up.upscale(frame)
            dt = (time.perf_counter() - t0) / args.frames
            horas_cinta = dt * 29.97 * 3600 / 3600
            print(f"{device:10s}: {dt * 1000:7.1f} ms/frame  ({1 / dt:5.1f} fps)  "
                  f"-> 1 h de cinta a 29.97 fps: ~{horas_cinta:.1f} h")
        except Exception as exc:
            print(f"{device:10s}: ERROR - {exc}")


if __name__ == "__main__":
    main()
