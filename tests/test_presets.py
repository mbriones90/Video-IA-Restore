"""Presets del modo facil.

Un preset fija TODOS los controles del paso 2 de una vez. La trampa clasica es
que alguien agregue una opcion nueva y se olvide de meterla en los presets:
entonces, al cambiar de preset, esa opcion se queda con el valor del preset
anterior y el usuario ve un ajuste que el no puso.
"""

from __future__ import annotations

import pytest

from vhs_upscaler.config import AppConfig
from vhs_upscaler.pipeline import ProcessOptions
from vhs_upscaler.presets import CUSTOM_KEY, PRESETS, TARGET_LUMA


def test_hay_presets_y_custom_no_es_uno_de_ellos():
    assert PRESETS
    assert CUSTOM_KEY not in PRESETS


@pytest.mark.parametrize("clave", list(PRESETS))
def test_cada_preset_tiene_etiqueta_y_explicacion(clave):
    preset = PRESETS[clave]
    assert preset["label"].strip()
    assert len(preset["desc"].strip()) > 40, "la explicacion tiene que servir de algo"


@pytest.mark.parametrize("clave", list(PRESETS))
def test_todas_las_claves_de_un_preset_existen_en_la_configuracion(clave):
    """Una clave mal escrita se ignoraria en silencio para siempre."""
    validas = set(AppConfig().__dataclass_fields__)
    # output_size y model_id viven en AppConfig; el resto tambien debe estar.
    sobran = set(PRESETS[clave]["values"]) - validas
    assert not sobran, f"claves que no existen en AppConfig: {sobran}"


@pytest.mark.parametrize("clave", list(PRESETS))
def test_todos_los_presets_nivelan_dia_y_noche(clave):
    valores = PRESETS[clave]["values"]
    assert valores["auto_exposure"] is True
    assert valores["auto_exposure_target"] == TARGET_LUMA


@pytest.mark.parametrize("clave", list(PRESETS))
def test_la_intensidad_esta_en_un_rango_sensato(clave):
    fuerza = PRESETS[clave]["values"]["auto_exposure_strength"]
    assert 40 <= fuerza <= 90, "ni inutil ni agresiva"


def test_la_cinta_ruidosa_corrige_mas_suave_que_la_normal():
    """Aclarar sube el grano: a mas ruido, mano mas ligera."""
    assert (PRESETS["ruidosa"]["values"]["auto_exposure_strength"]
            < PRESETS["facil"]["values"]["auto_exposure_strength"])


def test_los_presets_fijan_las_opciones_que_podrian_arrastrarse():
    """Si un preset no nombra una opcion, cambiar de preset la dejaria pegada."""
    comunes = set.intersection(*(set(p["values"]) for p in PRESETS.values()))
    for necesaria in ("auto_exposure", "auto_exposure_strength", "deflicker",
                      "denoise", "denoise_strength", "deband", "grain"):
        assert necesaria in comunes, f"{necesaria} falta en algun preset"


def test_la_configuracion_por_defecto_trae_la_exposicion_activada():
    assert AppConfig().auto_exposure is True


def test_los_valores_por_defecto_de_config_y_pipeline_son_coherentes():
    cfg = AppConfig()
    opts = ProcessOptions(input_path="x", output_path="y")
    assert cfg.auto_exposure_target == opts.auto_exposure_target
    assert cfg.auto_exposure_smoothing == opts.auto_exposure_smoothing
