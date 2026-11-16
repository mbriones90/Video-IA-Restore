"""Metodo de limpieza: hqdn3d (FFmpeg) frente a MVTools (VapourSynth).

El caso que motivo esto: una capturadora que desentrelaza por hardware y no
se puede desactivar. El material llega progresivo, QTGMC entero ya no se
puede usar (volveria a partir en campos lo que ya son fotogramas enteros),
pero su limpieza -el degrain compensado por movimiento- si.
"""

from __future__ import annotations

import pytest

from vhs_upscaler.pipeline import ProcessOptions, build_decode_filters
from vhs_upscaler.vs_backend import (
    effective_denoise_method, uses_vapoursynth, write_script,
)


def opts(**kw) -> ProcessOptions:
    base = dict(input_path="x.mp4", output_path="y.mp4")
    base.update(kw)
    return ProcessOptions(**base)


# ---------------------------------------------------------- que ruta usa ---

def test_qtgmc_usa_vapoursynth():
    assert uses_vapoursynth(opts(deinterlace=True, deinterlace_method="qtgmc"))


def test_mvtools_sobre_progresivo_usa_vapoursynth():
    """El caso de la capturadora que desentrelaza por hardware."""
    o = opts(deinterlace=False, denoise=True, denoise_method="mvtools")
    assert uses_vapoursynth(o)
    assert effective_denoise_method(o)[0] == "mvtools"


def test_hqdn3d_no_necesita_vapoursynth():
    assert not uses_vapoursynth(opts(deinterlace=True, deinterlace_method="bwdif",
                                     denoise=True, denoise_method="hqdn3d"))


def test_sin_denoise_y_sin_qtgmc_no_hay_grafo():
    assert not uses_vapoursynth(opts(deinterlace=False, denoise=False,
                                     denoise_method="mvtools"))


# ------------------------------------------- combinacion que no se puede ---

def test_bwdif_con_mvtools_degrada_a_hqdn3d_con_aviso():
    """Nunca perder el desentrelazado en silencio: degrada la limpieza."""
    o = opts(deinterlace=True, deinterlace_method="bwdif",
             denoise=True, denoise_method="mvtools")
    metodo, aviso = effective_denoise_method(o)
    assert metodo == "hqdn3d"
    assert "bwdif" in aviso and "QTGMC" in aviso
    assert not uses_vapoursynth(o), "se habria perdido el desentrelazado"


def test_yadif_con_mvtools_tambien_degrada():
    assert effective_denoise_method(
        opts(deinterlace=True, deinterlace_method="yadif",
             denoise=True, denoise_method="mvtools"))[0] == "hqdn3d"


def test_al_degradar_si_se_aplica_hqdn3d_en_los_filtros():
    filtros = build_decode_filters(
        opts(deinterlace=True, deinterlace_method="bwdif",
             denoise=True, denoise_method="mvtools", denoise_strength=4))
    assert any(f.startswith("hqdn3d") for f in filtros), "se quedo sin limpieza"


def test_con_mvtools_de_verdad_no_se_duplica_la_limpieza():
    """Si limpia el grafo, FFmpeg no debe limpiar tambien."""
    filtros = build_decode_filters(
        opts(deinterlace=False, denoise=True, denoise_method="mvtools",
             denoise_strength=4))
    assert not any(f.startswith("hqdn3d") for f in filtros)


def test_una_combinacion_valida_no_genera_aviso():
    assert effective_denoise_method(
        opts(deinterlace=True, deinterlace_method="qtgmc",
             denoise=True, denoise_method="mvtools"))[1] == ""


# ------------------------------------------------- el grafo que se emite ---

def leer(o, tmp_path) -> str:
    ruta = write_script(o, tmp_path / "entrada.mp4", source_height=480)
    try:
        return ruta.read_text(encoding="utf-8")
    finally:
        ruta.unlink(missing_ok=True)


def test_el_grafo_de_qtgmc_activa_el_modo_denoise(tmp_path):
    """El bug original: sin denoise_mode, denoise_func no se llamaba nunca."""
    texto = leer(opts(deinterlace=True, deinterlace_method="qtgmc", denoise=True),
                 tmp_path)
    assert "NoiseProcessMode.DENOISE if True" in texto
    assert "if True:" in texto  # la rama de QTGMC


def test_el_grafo_progresivo_no_mete_qtgmc(tmp_path):
    texto = leer(opts(deinterlace=False, denoise=True, denoise_method="mvtools"),
                 tmp_path)
    # La rama de QTGMC queda desactivada y se llama al degrain directamente.
    assert "if False:" in texto
    assert "clip = _ezdenoise(clip, tr=" in texto


def test_el_degrain_pide_vectores_por_delta_y_direccion(tmp_path):
    """mv.Analyse no tiene parametro 'tr': hay que pedirlos uno a uno."""
    texto = leer(opts(deinterlace=False, denoise=True, denoise_method="mvtools"),
                 tmp_path)
    assert "mv.Analyse(tr=" not in texto, "la API no acepta tr, reventaria"
    assert "isb=True, delta=1" in texto
    assert "isb=False, delta=1" in texto
    assert "Degrain1(sup, bw1, fw1" in texto


def test_la_fuerza_alta_sube_el_radio_temporal(tmp_path):
    suave = leer(opts(deinterlace=False, denoise=True, denoise_method="mvtools",
                      denoise_strength=3), tmp_path)
    fuerte = leer(opts(deinterlace=False, denoise=True, denoise_method="mvtools",
                       denoise_strength=8), tmp_path)
    assert "_ezdenoise(clip, tr=1)" in suave
    assert "_ezdenoise(clip, tr=2)" in fuerte


def test_la_fuerza_llega_al_umbral_del_degrain(tmp_path):
    texto = leer(opts(deinterlace=False, denoise=True, denoise_method="mvtools",
                      denoise_strength=5), tmp_path)
    assert "thsad=500" in texto   # 200 + 5*60


def test_el_metodo_entra_en_la_huella_del_trabajo():
    from vhs_upscaler.pipeline import job_fingerprint

    a = job_fingerprint(opts(denoise=True, denoise_method="hqdn3d"))
    b = job_fingerprint(opts(denoise=True, denoise_method="mvtools"))
    assert a != b, "cambiar de limpieza debe invalidar lo ya procesado"


# ---------------------------------------------------- se ejecuta de verdad -

@pytest.mark.slow
def test_el_grafo_progresivo_renderiza(sample_video, tmp_path):
    import subprocess

    from vhs_upscaler.vs_backend import find_vspipe

    vspipe = find_vspipe()
    if vspipe is None:
        pytest.skip("VapourSynth no disponible")
    o = opts(input_path=str(sample_video), deinterlace=False, denoise=True,
             denoise_method="mvtools", denoise_strength=4)
    ruta = write_script(o, sample_video, source_height=480)
    try:
        proc = subprocess.run([vspipe, "--start", "0", "--end", "9", str(ruta), "-"],
                              capture_output=True,
                              creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    finally:
        ruta.unlink(missing_ok=True)
    assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")[-400:]
    assert len(proc.stdout) == 10 * 640 * 480 * 3
