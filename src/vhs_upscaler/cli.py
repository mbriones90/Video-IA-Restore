"""Procesamiento por linea de comandos (sin interfaz grafica).

Ejemplo:
  uv run python -m vhs_upscaler.cli "video.mp4" --device cpu --model bicubic
"""

from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path

from .config import AppConfig
from .ffmpeg_utils import probe_video
from .hardware import detect_hardware, resolve_device
from .models import BICUBIC_ID, load_manifest
from .paths import OUTPUT_DIR
from .logging_setup import log_to, setup_logging
from .pipeline import (
    EXPORT_FORMATS, ProcessOptions, ProcessingCancelled, VideoProcessor,
    find_resumable_jobs,
)
from .resolutions import build_output_sizes


def apply_analysis(opts: ProcessOptions, result) -> None:
    """Vuelca lo detectado sobre las opciones, sin pisar lo que el usuario fijo."""
    if result.parity.parity == "progressive":
        opts.deinterlace = False
    elif result.parity.interlaced and opts.field_parity == "auto":
        opts.field_parity = result.parity.parity
    if result.crop.any:
        opts.crop_left, opts.crop_top = result.crop.left, result.crop.top
        opts.crop_right, opts.crop_bottom = result.crop.right, result.crop.bottom
    if opts.color_range == "auto":
        opts.color_range = result.color.range
    if opts.color_matrix == "auto":
        opts.color_matrix = result.color.matrix


def describe_applied(opts: ProcessOptions, result) -> str:
    partes = [
        "desentrelazado=" + ("no" if not opts.deinterlace else opts.deinterlace_method),
        f"paridad={opts.field_parity}",
    ]
    if result.crop.any:
        partes.append(f"recorte={opts.crop_left},{opts.crop_top},"
                      f"{opts.crop_right},{opts.crop_bottom}")
    partes.append(f"rango={opts.color_range}")
    partes.append(f"matriz={opts.color_matrix}")
    return ", ".join(partes)


def main() -> None:
    parser = argparse.ArgumentParser(description="Escala un video VHS a 1080p")
    parser.add_argument("input", help="Video de entrada (grabacion de OBS)")
    parser.add_argument("--output", default=None, help="Ruta de salida (por defecto output/)")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "directml", "vulkan"])
    parser.add_argument("--model", default=None,
                        help="id del modelo del manifiesto, o 'bicubic' para prueba sin IA")
    parser.add_argument("--format", default="mp4_h264",
                        choices=["mp4_h264", "mp4_h265", "mkv_h264", "mkv_h265",
                                 "webm_vp9", "mov_prores", "mkv_ffv1"],
                        help="Formato de export (ver EXPORT_FORMATS en pipeline.py)")
    parser.add_argument("--quality", default="alta", choices=["alta", "media", "rapida"])
    parser.add_argument("--height", type=int, default=1080, choices=[720, 1080, 1440, 2160],
                        help="Altura de salida (2160 = 4K)")
    parser.add_argument("--canvas", default="169", choices=["169", "native"],
                        help="169: lienzo 16:9 con barras si aplica; native: proporcion original sin barras")
    parser.add_argument("--bitrate-mbps", type=float, default=0.0,
                        help="Si es mayor que 0, codifica con bitrate objetivo en Mbps en vez de CRF")
    parser.add_argument("--no-deinterlace", action="store_true")
    parser.add_argument("--no-denoise", action="store_true")
    parser.add_argument("--interpolate", type=int, default=0, choices=[0, 50, 60])
    parser.add_argument("--deint", default="bwdif", choices=["bwdif", "yadif", "qtgmc"],
                        help="Metodo de desentrelazado (qtgmc requiere VapourSynth)")
    parser.add_argument("--parity", default="auto", choices=["auto", "bff", "tff"],
                        help="Paridad de campos: bff=NTSC tipico, tff=PAL tipico")
    parser.add_argument("--crop", default="0,0,0,0",
                        help="Recorte overscan 'izquierda,arriba,derecha,abajo' en px (multiplos de 2)")
    parser.add_argument("--no-chroma-denoise", action="store_true",
                        help="Desactiva la limpieza extra de crominancia")
    parser.add_argument("--grain", action="store_true",
                        help="Agrega grano de pelicula ligero (anti 'oleoso')")
    parser.add_argument("--grain-strength", type=int, default=6,
                        help="Intensidad del grano 1-20 (recomendado 4-8)")
    parser.add_argument("--mono-audio", action="store_true",
                        help="Unifica el audio estereo a mono (canal central)")
    parser.add_argument("--deflicker", action="store_true",
                        help="Estabiliza el brillo entre frames (bombeo del AGC del reproductor)")
    parser.add_argument("--color-range", default="auto", choices=["auto", "limited", "full"],
                        help="Rango del ORIGEN. 'auto' lo lee de la cabecera del archivo")
    parser.add_argument("--color-matrix", default="auto", choices=["auto", "bt601", "bt709"],
                        help="Matriz del ORIGEN. 'auto' la deduce de la resolucion")
    parser.add_argument("--analyze", action="store_true",
                        help="Analiza paridad, overscan y color, aplica lo detectado y continua")
    parser.add_argument("--analyze-only", action="store_true",
                        help="Solo analiza y muestra el resultado, sin procesar")
    parser.add_argument("--segment-seconds", type=int, default=300,
                        help="Tamano de cada tramo reanudable en segundos (0 = sin segmentar)")
    parser.add_argument("--no-resume", action="store_true",
                        help="Ignora los segmentos ya calculados y rehace el trabajo")
    parser.add_argument("--no-fp16", action="store_true",
                        help="Desactiva la variante fp16 del modelo en GPU")
    parser.add_argument("--jobs", action="store_true",
                        help="Lista los trabajos a medias que se pueden reanudar y termina")
    parser.add_argument("--test", action="store_true",
                        help="Modo prueba: solo un segmento corto (ver --test-start/--test-secs)")
    parser.add_argument("--test-start", type=float, default=15.0, help="Segundo inicial del modo prueba")
    parser.add_argument("--test-secs", type=float, default=10.0, help="Duracion del modo prueba")
    args = parser.parse_args()

    try:
        crop_l, crop_t, crop_r, crop_b = (int(v) for v in args.crop.split(","))
    except ValueError:
        print("Error: --crop debe ser 'izquierda,arriba,derecha,abajo' en numeros enteros")
        sys.exit(2)

    ruta_log = setup_logging(console=False)
    if ruta_log:
        print(f"Registro detallado: {ruta_log}")

    if args.jobs:
        pendientes = find_resumable_jobs()
        if not pendientes:
            print("No hay trabajos a medias.")
        for job in pendientes:
            hechos = sum(1 for seg in job.segments if seg.done)
            print(f"{job.job_id}  {Path(job.input_path).name}  "
                  f"{hechos}/{len(job.segments)} segmentos listos")
        return

    cfg = AppConfig.load()
    model_id = args.model or cfg.model_id or BICUBIC_ID
    hw = detect_hardware()
    device, aviso = resolve_device(args.device, hw)
    if aviso:
        print(aviso)
    print(f"Dispositivo: {device} | Providers ORT: {', '.join(hw.available_providers) or 'ninguno'}")
    print("Modelos conocidos: " + ", ".join(m.id for m in load_manifest()) + f", {BICUBIC_ID}")

    info = probe_video(args.input)
    opts = ProcessOptions(
        input_path=args.input, output_path="", model_id=model_id, device=device,
        export_format=args.format, quality=args.quality,
        deinterlace=not args.no_deinterlace, denoise=not args.no_denoise,
        interpolate_fps=args.interpolate,
        deinterlace_method=args.deint, field_parity=args.parity,
        crop_left=crop_l, crop_top=crop_t, crop_right=crop_r, crop_bottom=crop_b,
        chroma_denoise=not args.no_chroma_denoise,
        grain=args.grain, grain_strength=args.grain_strength,
        audio_mono=args.mono_audio,
        deflicker=args.deflicker,
        color_range=args.color_range, color_matrix=args.color_matrix,
        source_sar=info.sar,
        fp16=not args.no_fp16,
        segment_seconds=max(0, args.segment_seconds),
        resume=not args.no_resume,
        start_seconds=args.test_start if args.test else 0.0,
        max_seconds=args.test_secs if args.test else 0.0,
    )

    if args.analyze or args.analyze_only:
        from .analysis import analyze as run_analysis

        print("\n--- Analisis automatico de la cinta ---")
        resultado = run_analysis(args.input, info, progress=lambda m: print("  " + m))
        print(resultado.summary(info))
        apply_analysis(opts, resultado)
        print("Ajustes aplicados: " + describe_applied(opts, resultado))
        if args.analyze_only:
            return

    key = f"{args.height}_{args.canvas}"
    size = next(s for s in build_output_sizes(info.width, info.height, info.sar)
                if s.key == key)
    print(f"Resolucion de salida: {size.label}")
    opts.target_width, opts.target_height = size.width, size.height
    opts.quality_mode = "bitrate" if args.bitrate_mbps > 0 else "crf"
    opts.bitrate_mbps = args.bitrate_mbps if args.bitrate_mbps > 0 else 16.0

    suffix = "_prueba" if args.test else ""
    ext = EXPORT_FORMATS[args.format]["ext"]
    output = args.output or str(OUTPUT_DIR / f"{Path(args.input).stem}{suffix}_{size.width}x{size.height}.{ext}")
    opts.output_path = output

    def progress(done: int, total: int) -> None:
        pct = done * 100 // max(total, 1)
        sys.stdout.write(f"\rProgreso: {done}/{total} frames ({pct}%)   ")
        sys.stdout.flush()

    try:
        VideoProcessor().run(opts, progress=progress, log=log_to(print, "cli"),
                             cancel_event=threading.Event())
        print()
    except ProcessingCancelled:
        print("\nProcesamiento cancelado.")
    except Exception as exc:
        print(f"\nError: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
