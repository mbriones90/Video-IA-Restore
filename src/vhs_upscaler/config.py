"""Configuracion local de la maquina (config/config.local.json).

Guarda las preferencias del usuario: dispositivo, modelo y ajustes de salida.
Es un archivo por PC; no se comparte entre maquinas.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields

from .paths import CONFIG_PATH, OUTPUT_DIR, ensure_dirs


@dataclass
class AppConfig:
    # "auto" resuelve el mejor dispositivo disponible al iniciar.
    device: str = "auto"  # auto | cpu | cuda | directml | vulkan
    model_id: str = "realesr-general-x4v3"
    output_dir: str = field(default_factory=lambda: str(OUTPUT_DIR))
    codec: str = "h264"  # obsoleto (compatibilidad); usar export_format
    export_format: str = "mp4_h264"  # clave de pipeline.EXPORT_FORMATS
    quality: str = "alta"  # alta | media | rapida (modo CRF)
    output_size: str = "1080_169"  # "<altura>_native" | "<altura>_169"
    quality_mode: str = "crf"  # crf | bitrate
    bitrate_mbps: float = 16.0  # bitrate objetivo cuando quality_mode = "bitrate"
    deinterlace: bool = True
    deinterlace_method: str = "bwdif"  # bwdif | yadif | qtgmc (requiere VapourSynth)
    field_parity: str = "auto"  # auto | bff (NTSC tipico) | tff (PAL tipico)
    crop_left: int = 0
    crop_right: int = 0
    crop_top: int = 0
    crop_bottom: int = 0
    chroma_denoise: bool = True  # limpieza de crominancia (sangrado de color VHS)
    qtgmc_double_fps: bool = True  # 29.97i -> 59.94p (doble tasa temporal)
    denoise: bool = True
    denoise_strength: int = 3  # 0-10, fuerza de la reduccion de ruido
    denoise_method: str = "hqdn3d"  # hqdn3d (FFmpeg) | mvtools (VapourSynth)
    color_correct: bool = False
    brightness: int = 0  # -100 a 100
    contrast: int = 100  # 0 a 200 (100 = sin cambio)
    saturation: int = 110  # 0 a 200 (100 = sin cambio)
    deband: bool = False
    grain: bool = False  # grano de pelicula ligero (anti "oleoso") en la salida
    grain_strength: int = 6  # 1-20
    audio_mono: bool = False  # unificar audio estereo a mono (canal central)
    interpolate_fps: int = 0  # 0 = fps original; 50 o 60 = interpolar
    test_start: int = 15  # segundo inicial de la prueba rapida
    test_secs: int = 10   # duracion de la prueba rapida
    # --- Color del ORIGEN ("auto" = leerlo del archivo / deducirlo) ---
    color_range: str = "auto"   # auto | limited | full
    color_matrix: str = "auto"  # auto | bt601 | bt709
    # --- Estabilizacion ---
    deflicker: bool = False     # corrige el bombeo de brillo del AGC
    deflicker_window: int = 5   # frames de la media movil
    # --- Exposicion automatica por escena (dia / noche) ---
    # Activada de serie: en una cinta familiar casi siempre hay escenas de
    # interior o de noche, y la correccion es conservadora (ver AutoExposure).
    auto_exposure: bool = True
    auto_exposure_target: int = 56     # mediana de luma objetivo (0-255)
    auto_exposure_strength: int = 70   # 0-100
    auto_exposure_smoothing: int = 30  # frames de la media movil del exponente
    # --- Rendimiento y reanudacion ---
    fp16: bool = True              # variante de media precision del modelo en GPU
    segment_minutes: int = 5       # tamano de cada tramo reanudable (0 = sin trocear)
    resume: bool = True            # reaprovechar tramos ya calculados

    @classmethod
    def load(cls) -> "AppConfig":
        ensure_dirs()
        if not CONFIG_PATH.exists():
            return cls()
        try:
            data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return cls()
        valid = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in valid})

    def save(self) -> None:
        ensure_dirs()
        CONFIG_PATH.write_text(
            json.dumps(asdict(self), indent=2, ensure_ascii=False), encoding="utf-8"
        )
