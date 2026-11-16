"""Construccion de filtros, dimensiones y resoluciones de salida."""

from __future__ import annotations

import pytest

from vhs_upscaler.pipeline import (
    EXPORT_FORMATS, ProcessOptions, build_decode_filters, build_output_filter,
    cropped_dimensions,
)
from vhs_upscaler.resolutions import build_output_sizes, native_dimensions


def opts(**kw) -> ProcessOptions:
    base = dict(input_path="x.mp4", output_path="y.mp4")
    base.update(kw)
    return ProcessOptions(**base)


# --------------------------------------------------------------- recorte ----

def test_el_recorte_impar_se_rechaza():
    with pytest.raises(ValueError, match="multiplo de 2"):
        cropped_dimensions(720, 480, opts(crop_left=3))


def test_un_recorte_que_deja_la_imagen_minuscula_se_rechaza():
    with pytest.raises(ValueError, match="demasiado pequena"):
        cropped_dimensions(720, 480, opts(crop_left=340, crop_right=340))


def test_el_recorte_resta_de_ambos_lados():
    assert cropped_dimensions(720, 480, opts(crop_left=8, crop_right=8,
                                             crop_top=4, crop_bottom=6)) == (704, 470)


# --------------------------------------------------------------- filtros ----

def test_qtgmc_cae_a_bwdif_en_la_ruta_ffmpeg():
    """Nunca se procesa sin desentrelazar por accidente."""
    filtros = build_decode_filters(opts(deinterlace=True, deinterlace_method="qtgmc"))
    assert any(f.startswith("bwdif") for f in filtros)


def test_sin_desentrelazado_no_hay_filtro_de_campos():
    filtros = build_decode_filters(opts(deinterlace=False))
    assert not any(f.startswith(("bwdif", "yadif")) for f in filtros)


def test_la_paridad_llega_al_filtro():
    filtros = build_decode_filters(opts(deinterlace_method="yadif", field_parity="bff"))
    assert "yadif=mode=send_frame:parity=bff:deint=all" in filtros


def test_el_rango_de_color_explicito_entra_antes_que_nada():
    filtros = build_decode_filters(opts(color_range="full", deinterlace=False, denoise=False))
    assert filtros[0] == "scale=in_range=pc:out_range=pc"


def test_el_rango_automatico_no_agrega_filtro():
    filtros = build_decode_filters(opts(color_range="auto", deinterlace=False, denoise=False))
    assert not any("in_range" in f for f in filtros)


def test_el_orden_de_los_filtros_es_el_correcto():
    """Desentrelazar -> recortar -> limpiar: al reves se arrastra basura."""
    filtros = build_decode_filters(opts(deinterlace=True, crop_left=8, crop_right=8,
                                        denoise=True, denoise_strength=3))
    nombres = [f.split("=")[0] for f in filtros]
    assert nombres.index("bwdif") < nombres.index("crop") < nombres.index("hqdn3d")


# ---------------------------------------------------------------- salida ----

def test_la_salida_nunca_estira_la_imagen():
    vf = build_output_filter(opts(target_width=1920, target_height=1080))
    assert "force_original_aspect_ratio=decrease" in vf
    assert "pad=1920:1080" in vf
    assert vf.endswith("setsar=1")


def test_la_salida_fija_la_matriz_y_el_rango():
    vf = build_output_filter(opts())
    assert "out_color_matrix=bt709" in vf
    assert "out_range=tv" in vf


def test_el_grano_se_aplica_antes_del_pad():
    """Si no, las barras negras salen con ruido."""
    vf = build_output_filter(opts(grain=True, grain_strength=8))
    assert vf.index("noise=") < vf.index("pad=")


def test_sin_grano_no_hay_filtro_de_ruido():
    assert "noise=" not in build_output_filter(opts(grain=False))


# ----------------------------------------------------------- resoluciones ---

def test_la_proporcion_nativa_con_pixel_cuadrado():
    """720x480 con pixeles cuadrados se ve 3:2, no 4:3."""
    assert native_dimensions(720, 480, 1080) == (1620, 1080)


def test_la_proporcion_nativa_respeta_el_pixel_no_cuadrado():
    """NTSC 720x480 marcado SAR 8:9 se ve 4:3 -> 1440x1080, no 1620."""
    assert native_dimensions(720, 480, 1080, sar=8 / 9) == (1440, 1080)


def test_el_filtro_de_salida_corrige_el_pixel_no_cuadrado():
    vf = build_output_filter(opts(source_sar=8 / 9))
    assert vf.startswith("scale=round(iw*0.888889/2)*2:ih,setsar=1")


def test_sin_pixel_raro_no_se_agrega_correccion():
    assert build_output_filter(opts(source_sar=1.0)).startswith("scale=1920:1080")


def test_la_etiqueta_de_proporcion_usa_el_aspecto_visible():
    sizes = {s.key: s for s in build_output_sizes(720, 480, sar=8 / 9)}
    assert "4:3" in sizes["1080_native"].label


def test_todas_las_resoluciones_son_pares():
    for size in build_output_sizes(720, 486):
        assert size.width % 2 == 0 and size.height % 2 == 0


def test_un_video_16_9_no_lleva_barras():
    sizes = {s.key: s for s in build_output_sizes(1920, 1080)}
    assert sizes["1080_169"].adds_bars is False


def test_un_video_4_3_si_lleva_barras_en_lienzo_16_9():
    sizes = {s.key: s for s in build_output_sizes(720, 480)}
    assert sizes["1080_169"].adds_bars is True
    assert sizes["1080_native"].adds_bars is False


# -------------------------------------------------------------- formatos ----

def test_todos_los_formatos_estan_completos():
    for key, fmt in EXPORT_FORMATS.items():
        for campo in ("label", "video", "audio", "ext", "rate", "pix_fmt"):
            assert campo in fmt, f"{key} no define {campo}"


def test_ya_no_se_ofrecen_los_formatos_intermedios_de_10_bits():
    assert "mov_prores" not in EXPORT_FORMATS
    assert "mkv_ffv1" not in EXPORT_FORMATS
