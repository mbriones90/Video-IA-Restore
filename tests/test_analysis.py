"""Analisis de la cinta: parseo de idet/cropdetect y deteccion real."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from vhs_upscaler import analysis
from vhs_upscaler.analysis import (
    Analysis, ColorInfo, CropResult, check_capture, detect_field_parity,
)
from vhs_upscaler.ffmpeg_utils import VideoInfo, probe_video

IDET_TFF = (
    "[Parsed_idet_0 @ 0x1] Repeated Fields: Neither: 100 Top: 0 Bottom: 0\n"
    "[Parsed_idet_0 @ 0x1] Single frame detection: TFF: 80 BFF: 2 Progressive: 10 Undetermined: 8\n"
    "[Parsed_idet_0 @ 0x1] Multi frame detection: TFF: 95 BFF: 1 Progressive: 2 Undetermined: 2\n"
)
IDET_BFF = IDET_TFF.replace("TFF: 95 BFF: 1", "TFF: 1 BFF: 95")
IDET_PROG = IDET_TFF.replace("TFF: 95 BFF: 1 Progressive: 2", "TFF: 1 BFF: 1 Progressive: 98")


def info(duration=100.0) -> VideoInfo:
    return VideoInfo(Path("x.mp4"), 720, 480, 29.97, 2997, duration)


def fake_idet(monkeypatch, salida: str):
    monkeypatch.setattr(analysis, "_run_filter", lambda *a, **k: salida)


# ------------------------------------------------------------- paridad -----

def test_detecta_tff(monkeypatch):
    fake_idet(monkeypatch, IDET_TFF)
    r = detect_field_parity("x.mp4", info())
    assert r.parity == "tff"
    assert r.interlaced
    assert r.confidence > 0.9


def test_detecta_bff(monkeypatch):
    fake_idet(monkeypatch, IDET_BFF)
    assert detect_field_parity("x.mp4", info()).parity == "bff"


def test_detecta_progresivo(monkeypatch):
    fake_idet(monkeypatch, IDET_PROG)
    r = detect_field_parity("x.mp4", info())
    assert r.parity == "progressive"
    assert not r.interlaced


def test_si_esta_empatado_se_queda_en_auto(monkeypatch):
    """Equivocar la paridad arruina la cinta: ante la duda, no decidir."""
    fake_idet(monkeypatch, IDET_TFF.replace("TFF: 95 BFF: 1", "TFF: 50 BFF: 48"))
    r = detect_field_parity("x.mp4", info())
    assert r.parity == "auto"
    assert r.confidence == 0.0


def test_sin_salida_de_idet_se_queda_en_auto(monkeypatch):
    fake_idet(monkeypatch, "")
    assert detect_field_parity("x.mp4", info()).parity == "auto"


def test_el_resumen_de_paridad_es_legible(monkeypatch):
    fake_idet(monkeypatch, IDET_BFF)
    texto = detect_field_parity("x.mp4", info()).summary()
    assert "BFF" in texto and "confianza" in texto


# ------------------------------------------------------------- recorte -----

def test_cropdetect_toma_el_recuadro_mas_conservador(monkeypatch):
    """Una escena oscura no debe hacer que se recorte imagen real."""
    salidas = iter([
        "[Parsed_cropdetect_0 @ 0x1] x1:79 x2:560 crop=480:480:80:0\n",
        "[Parsed_cropdetect_0 @ 0x1] x1:99 x2:540 crop=440:480:100:0\n",  # recorta mas
        "[Parsed_cropdetect_0 @ 0x1] x1:79 x2:560 crop=480:480:80:0\n",
        "[Parsed_cropdetect_0 @ 0x1] x1:79 x2:560 crop=480:480:80:0\n",
    ])
    monkeypatch.setattr(analysis, "_run_filter", lambda *a, **k: next(salidas))
    r = analysis.detect_crop("x.mp4", info())
    assert (r.left, r.right) == (80, 160)  # el minimo de cada lado, no el de 100
    assert r.any


def test_el_recorte_siempre_queda_par(monkeypatch):
    monkeypatch.setattr(
        analysis, "_run_filter",
        lambda *a, **k: "crop=481:479:81:1\n")
    r = analysis.detect_crop("x.mp4", info())
    for v in (r.left, r.top, r.right, r.bottom):
        assert v % 2 == 0, "el pipeline exige multiplos de 2"


def test_sin_bordes_el_recorte_es_cero(monkeypatch):
    monkeypatch.setattr(analysis, "_run_filter", lambda *a, **k: "")
    r = analysis.detect_crop("x.mp4", info())
    assert not r.any
    assert (r.left, r.top, r.right, r.bottom) == (0, 0, 0, 0)


def test_cropdetect_ignora_recuadros_imposibles(monkeypatch):
    monkeypatch.setattr(analysis, "_run_filter", lambda *a, **k: "crop=9999:9999:0:0\n")
    assert not analysis.detect_crop("x.mp4", info()).any


# --------------------------------------------------------------- color -----

def test_color_sd_sin_etiquetas_supone_bt601(monkeypatch):
    monkeypatch.setattr(
        subprocess, "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 1, b"", b"sin streams aqui"))
    c = analysis.detect_color("x.mp4", info())
    assert c.matrix == "bt601"
    assert c.range == "limited"
    assert not c.detected_matrix


def test_color_lee_las_etiquetas_del_archivo(monkeypatch):
    linea = (b"  Stream #0:0: Video: h264 (High), yuv420p(pc, bt709), 1920x1080\n")
    monkeypatch.setattr(
        subprocess, "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 0, b"", linea))
    c = analysis.detect_color("x.mp4", VideoInfo(Path("x"), 1920, 1080, 30, 1, 1))
    assert c.range == "full" and c.detected_range
    assert c.matrix == "bt709" and c.detected_matrix


def test_color_reconoce_smpte170m_como_bt601(monkeypatch):
    linea = b"  Stream #0:0: Video: h264, yuv420p(tv, bt470bg/bt470bg/smpte170m), 720x480\n"
    monkeypatch.setattr(
        subprocess, "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 0, b"", linea))
    c = analysis.detect_color("x.mp4", info())
    assert c.matrix == "bt601" and c.detected_matrix
    assert c.range == "limited" and c.detected_range


def test_el_resumen_de_color_dice_si_fue_detectado_o_supuesto():
    assert "supuesto" in ColorInfo().summary()
    assert "etiquetado" in ColorInfo(detected_range=True, detected_matrix=True).summary()


def test_el_resumen_de_recorte_calcula_la_imagen_util():
    texto = CropResult(80, 0, 80, 2).summary(640, 362)
    assert "480x360" in texto


# --------------------------------------------- resolucion de captura -------

def vi(w, h, fps=29.97, sar=1.0) -> VideoInfo:
    return VideoInfo(Path("x.mp4"), w, h, fps, 1000, 40.0, sar)


def test_una_captura_ntsc_nativa_pasa_sin_quejas():
    c = check_capture(vi(720, 480))
    assert c.severity == "ok"
    assert not c.rescaled
    assert c.standard_name == "NTSC D1"


def test_una_captura_pal_nativa_pasa_sin_quejas():
    c = check_capture(vi(720, 576, fps=25.0))
    assert c.severity == "ok"
    assert c.system == "pal"
    assert c.expected_lines == 576


def test_las_486_lineas_tambien_valen():
    assert check_capture(vi(720, 486)).severity == "ok"


def test_detecta_el_reescalado_del_caso_real():
    """640x362 con 480x360 utiles: el caso que motivo este aviso."""
    c = check_capture(vi(640, 362), CropResult(80, 0, 80, 2))
    assert c.rescaled
    assert c.severity == "grave"
    assert (c.content_width, c.content_height) == (480, 360)
    assert c.lost_pct == pytest.approx(50, abs=3)
    assert "720x480" in c.summary()


def test_un_lienzo_mas_grande_sin_perder_lineas_es_solo_aviso():
    """Meter NTSC en un lienzo 16:9 agrega barras, no tira resolucion."""
    c = check_capture(vi(854, 480), CropResult(87, 0, 87, 0))
    assert not c.rescaled
    assert c.severity == "aviso"
    assert "no se perdio detalle" in c.summary()


def test_el_sistema_se_deduce_de_los_fps_no_del_alto():
    """El alto es justo lo que puede venir falseado por el reescalado."""
    assert check_capture(vi(640, 360, fps=29.97)).system == "ntsc"
    assert check_capture(vi(640, 360, fps=25.0)).system == "pal"
    assert check_capture(vi(640, 360, fps=59.94)).system == "ntsc"


def test_unos_fps_raros_no_disparan_el_aviso():
    c = check_capture(vi(640, 360, fps=12.0))
    assert c.system == "desconocido"
    assert not c.rescaled, "sin saber el sistema no se puede acusar de nada"


def test_el_pixel_no_cuadrado_cuenta_para_la_imagen_util():
    c = check_capture(vi(720, 480, sar=8 / 9))
    assert c.content_width == 640  # 720 * 8/9


# ------------------------------------- el reescalado matiza la paridad -----

def _analisis(parity_kind: str, capture_w: int, capture_h: int, crop=None):
    fake_idet(_MP := None, None) if False else None
    from vhs_upscaler.analysis import ParityResult
    parity = {
        "progressive": ParityResult("progressive", 1, 1, 98, 0, 0.98),
        "bff": ParityResult("bff", 1, 95, 2, 2, 0.98),
    }[parity_kind]
    crop = crop or CropResult(0, 0, 0, 0)
    cap = check_capture(vi(capture_w, capture_h), crop)
    return Analysis(parity, crop, ColorInfo(), cap)


def test_progresivo_sobre_captura_reescalada_lleva_matiz():
    a = _analisis("progressive", 640, 362, CropResult(80, 0, 80, 2))
    aviso = a.parity_caveat()
    assert "NO es concluyente" in aviso
    assert "resolucion nativa" in aviso


def test_progresivo_sobre_captura_nativa_no_lleva_matiz():
    assert _analisis("progressive", 720, 480).parity_caveat() == ""


def test_un_veredicto_entrelazado_nunca_lleva_matiz():
    """Si hay peinado, lo que dice idet es un hecho, no una sospecha."""
    assert _analisis("bff", 640, 362, CropResult(80, 0, 80, 2)).parity_caveat() == ""


def test_el_resumen_completo_incluye_el_aviso_de_captura():
    a = _analisis("progressive", 640, 362, CropResult(80, 0, 80, 2))
    texto = a.summary(vi(640, 362))
    assert "captura reescalada" in texto
    assert "NO es concluyente" in texto


# ------------------------------------------------------ deteccion real -----

@pytest.mark.slow
def test_analisis_real_sobre_un_clip_sintetico(sample_video):
    """testsrc es progresivo y ocupa todo el cuadro: eso debe salir."""
    resultado = analysis.analyze(str(sample_video), probe_video(sample_video))
    assert resultado.parity.parity in ("progressive", "auto")
    assert not resultado.crop.any
    assert resultado.color.matrix == "bt601"  # 640x480 es SD
    # 640x480 es una resolucion de captura NTSC valida: no debe quejarse.
    assert resultado.capture.severity == "ok"
    assert not resultado.capture.rescaled
