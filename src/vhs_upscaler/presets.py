"""Configuraciones predefinidas (modo facil).

Cada preset describe, en palabras simples, para que cinta sirve y que le hace.
Aplicar un preset ajusta todos los controles del paso 2; si despues cambias
cualquier opcion a mano, el combo pasa a "Personalizado".
"""

from __future__ import annotations

PRESETS: dict[str, dict] = {
    "facil": {
        "label": "Facil - VHS estandar (recomendado para empezar)",
        "desc": (
            "Para una cinta VHS normal: quita las rayas de entrelazado, limpia el "
            "grano de cinta y el sangrado de color, y escala a 1080p con el modelo "
            "rapido. Si no sabes que elegir, usa este y mira la vista previa."
        ),
        "values": {
            "model_id": "realesr-general-x4v3", "deinterlace": True,
            "deinterlace_method": "bwdif", "field_parity": "auto",
            "denoise": True, "denoise_strength": 3, "chroma_denoise": True,
            "color_correct": False, "deband": False, "grain": False, "grain_strength": 6, "audio_mono": False,
            "interpolate_fps": 0,
            "output_size": "1080_169", "export_format": "mp4_h264",
            "quality_mode": "crf", "quality": "alta", "bitrate_mbps": 16.0,
            "qtgmc_double_fps": True,
        },
    },
    "ruidosa": {
        "label": "Cinta muy ruidosa o danada",
        "desc": (
            "Para cintas con mucho grano, estatica o manchas de color: limpieza de "
            "ruido FUERTE, correccion de crominancia y suavizado de bandas de color. "
            "La imagen queda mas lisa pero mucho mas limpia."
        ),
        "values": {
            "model_id": "realesr-general-x4v3", "deinterlace": True,
            "deinterlace_method": "bwdif", "field_parity": "auto",
            "denoise": True, "denoise_strength": 6, "chroma_denoise": True,
            "color_correct": False, "deband": True, "grain": False, "grain_strength": 6, "audio_mono": False,
            "output_size": "1080_169", "export_format": "mp4_h264",
            "quality_mode": "crf", "quality": "alta", "bitrate_mbps": 16.0,
            "qtgmc_double_fps": True,
        },
    },
    "buena": {
        "label": "Cinta en buen estado",
        "desc": (
            "Para cintas bien conservadas con poco ruido: limpieza SUAVE para no "
            "borrar detalle fino. Deja casi intacta la textura original."
        ),
        "values": {
            "model_id": "realesr-general-x4v3", "deinterlace": True,
            "deinterlace_method": "bwdif", "field_parity": "auto",
            "denoise": True, "denoise_strength": 2, "chroma_denoise": False,
            "color_correct": False, "deband": False, "grain": False, "grain_strength": 6, "audio_mono": False,
            "interpolate_fps": 0,
            "output_size": "1080_169", "export_format": "mp4_h264",
            "quality_mode": "crf", "quality": "alta", "bitrate_mbps": 16.0,
            "qtgmc_double_fps": True,
        },
    },
    "maxima": {
        "label": "Maxima calidad (lento, mejor con GPU)",
        "desc": (
            "Usa el modelo grande x4plus, que recupera mas detalle que el rapido. "
            "Puede tardar MUCHO mas por video (recomendado solo con NVIDIA o "
            "DirectML, y probando primero con el boton de prueba rapida)."
        ),
        "values": {
            "model_id": "realesrgan-x4plus", "deinterlace": True,
            "deinterlace_method": "bwdif", "field_parity": "auto",
            "denoise": True, "denoise_strength": 3, "chroma_denoise": True,
            "color_correct": False, "deband": False, "grain": False, "grain_strength": 6, "audio_mono": False,
            "interpolate_fps": 0,
            "output_size": "1080_169", "export_format": "mp4_h264",
            "quality_mode": "crf", "quality": "alta", "bitrate_mbps": 16.0,
            "qtgmc_double_fps": True,
        },
    },
}

CUSTOM_KEY = "custom"
