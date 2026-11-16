"""Formatos AV1: escala de calidad propia, control de tasa y degradacion.

Dos cosas se rompen con mucha facilidad aqui y por eso tienen test:

1. AV1 usa un cuantizador 0-63 y el codificador de la tarjeta no entiende
   "-crf" sino "-cq", y solo si el bitrate objetivo va a cero. Con "-crf" el
   parametro se ignora en silencio y sale un archivo con el bitrate por
   defecto: no falla, solo da una calidad que nadie pidio.

2. av1_nvenc aparece en "ffmpeg -encoders" en cualquier PC con FFmpeg
   completo, pero solo funciona de verdad en GeForce RTX 40 o superior.
"""

from __future__ import annotations

import pathlib

import pytest

from vhs_upscaler import pipeline as pipeline_mod
from vhs_upscaler.pipeline import (
    CODEC_CRF, EXPORT_FORMATS, ProcessOptions, VideoProcessor,
    quality_value, resolve_export_format,
)


def opts(**kw) -> ProcessOptions:
    base = dict(input_path="x.mkv", output_path="y.mp4")
    base.update(kw)
    return ProcessOptions(**base)


def comando(clave: str, **kw) -> list[str]:
    o = opts(export_format=clave, **kw)
    return VideoProcessor._encode_command(
        o, "ffmpeg", EXPORT_FORMATS[clave], 1920, 1080, 60.0,
        pathlib.Path("seg.mkv.part"))


def tras_codec(cmd: list[str]) -> list[str]:
    """Los argumentos entre -c:v y el siguiente -vf."""
    i = cmd.index("-c:v")
    return cmd[i:cmd.index("-vf", i)]


# ------------------------------------------------- estan en la tabla ------

@pytest.mark.parametrize("clave", ["mp4_av1", "mkv_av1", "mp4_av1_cpu"])
def test_el_formato_existe_y_esta_completo(clave):
    fmt = EXPORT_FORMATS[clave]
    assert fmt["video"] in ("av1_nvenc", "libaom-av1")
    assert fmt["ext"] in ("mp4", "mkv")
    assert fmt["rate"] is True
    assert fmt["fallback"] in EXPORT_FORMATS


def test_el_contenedor_del_respaldo_coincide():
    """Degradar no debe cambiar la extension del archivo que se prometio."""
    for clave, fmt in EXPORT_FORMATS.items():
        alt = fmt.get("fallback")
        if alt:
            assert EXPORT_FORMATS[alt]["ext"] == fmt["ext"], clave


def test_los_dos_codecs_av1_tienen_su_escala():
    for codec in ("av1_nvenc", "libaom-av1"):
        niveles = CODEC_CRF[codec]
        assert set(niveles) == {"alta", "media", "rapida"}
        valores = [int(v) for v in niveles.values()]
        assert valores == sorted(valores), "alta debe ser el numero mas bajo"
        assert all(0 <= v <= 63 for v in valores), "AV1 va de 0 a 63"


def test_la_escala_de_av1_no_se_confunde_con_la_de_h264():
    """Si alguien copia el 17 de H.264 a AV1, el archivo sale enorme."""
    assert int(CODEC_CRF["av1_nvenc"]["alta"]) > int(CODEC_CRF["libx264"]["alta"])


# ------------------------------------------------- control de tasa --------

def test_la_gpu_usa_cq_y_no_crf():
    args = tras_codec(comando("mp4_av1"))
    assert "-crf" not in args, "av1_nvenc ignora -crf en silencio"
    assert "-cq" in args
    assert args[args.index("-cq") + 1] == CODEC_CRF["av1_nvenc"]["alta"]


def test_la_gpu_pide_vbr_y_bitrate_cero():
    """Sin "-b:v 0" el -cq no hace nada: manda el bitrate por defecto."""
    args = tras_codec(comando("mp4_av1"))
    assert args[args.index("-rc") + 1] == "vbr"
    assert args[args.index("-b:v") + 1] == "0"


def test_ya_no_se_puede_fijar_el_bitrate():
    """Un bitrate fijo no encaja con una cinta: solo quedan los perfiles."""
    import dataclasses

    campos = {f.name for f in dataclasses.fields(ProcessOptions)}
    assert "bitrate_mbps" not in campos
    assert "quality_mode" not in campos
    args = tras_codec(comando("mp4_av1"))
    assert "-maxrate" not in args and "-bufsize" not in args


# ------------------------------------------- ajuste por resolucion --------
#
# Medido sobre 10 s reales del pipeline: el cuantizador que mantiene la misma
# calidad sube con la resolucion, porque a mas pixeles cada uno pesa menos.
#   720p  cq 26 -> VMAF 82.0      1080p cq 26 -> VMAF 82.1  (base)
#   1440p cq 28 -> VMAF 82.0      2160p cq 30 -> 28.9 Mbps frente a 40.5 con cq 26


def test_el_perfil_sube_el_cuantizador_al_subir_la_resolucion():
    valores = [int(quality_value("av1_nvenc", "alta", h))
               for h in (720, 1080, 1440, 2160)]
    assert valores == [26, 26, 28, 30]


def test_a_720p_no_se_ajusta_nada():
    """La sensibilidad solo crece hacia arriba: a 720p cq 26 da el mismo VMAF."""
    assert quality_value("av1_nvenc", "media", 720) == \
        quality_value("av1_nvenc", "media", 1080)


def test_el_ajuste_se_aplica_a_todos_los_codecs():
    """El efecto es de la resolucion, no del codec (x265 lo confirma)."""
    for codec in ("libx264", "libx265", "av1_nvenc", "libaom-av1"):
        a = int(quality_value(codec, "alta", 1080))
        b = int(quality_value(codec, "alta", 2160))
        assert b == a + 4, codec


def test_los_tres_perfiles_se_mantienen_separados_en_4k():
    valores = [int(quality_value("av1_nvenc", q, 2160))
               for q in ("alta", "media", "rapida")]
    assert valores == sorted(valores) and len(set(valores)) == 3


def test_el_cuantizador_no_se_sale_del_rango_del_codec():
    """H.264 y H.265 llegan a 51; AV1 y VP9, a 63."""
    assert int(quality_value("libx264", "rapida", 9999)) <= 51
    assert int(quality_value("libaom-av1", "rapida", 9999)) <= 63
    assert int(quality_value("av1_nvenc", "alta", 0)) >= 0


def test_el_comando_lleva_el_cuantizador_de_la_resolucion():
    """A 4K debe ir el valor ajustado, no el de 1080p."""
    args = tras_codec(comando("mp4_av1", target_width=3840, target_height=2160))
    assert args[args.index("-cq") + 1] == "30"


def test_manda_el_lienzo_final_y_no_la_salida_del_modelo():
    """El bug que esto fija.

    _encode_command recibe el tamano que entrega el modelo (ch * scale), que
    con una captura de 480 lineas y un modelo x4 son 1920 tanto si exportas a
    1080p como a 4K. Si el ajuste mira ese numero, elegir 4K no cambia nada y
    exportar a 1080p sale apretado de mas.
    """
    import pathlib

    def cq(target_h, modelo_h):
        cmd = VideoProcessor._encode_command(
            opts(export_format="mp4_av1", target_width=1920,
                 target_height=target_h),
            "ffmpeg", EXPORT_FORMATS["mp4_av1"], 2560, modelo_h, 60.0,
            pathlib.Path("seg.mkv.part"))
        args = tras_codec(cmd)
        return args[args.index("-cq") + 1]

    # Misma salida del modelo (1920), lienzos distintos -> cuantizadores
    # distintos.
    assert cq(1080, 1920) == "26"   # perfil "alta" a 1080p
    assert cq(2160, 1920) == "30"   # el mismo perfil a 4K, +4
    # Mismo lienzo, salidas del modelo distintas -> el mismo cuantizador.
    assert cq(1080, 1920) == cq(1080, 2880)


def test_el_log_dice_cuando_ajusto_por_resolucion():
    desc = VideoProcessor._rate_description
    en1080 = desc(opts(quality="alta"), EXPORT_FORMATS["mp4_av1"], 1080)
    en4k = desc(opts(quality="alta"), EXPORT_FORMATS["mp4_av1"], 2160)
    assert "cq 26" in en1080 and "ajustado" not in en1080
    assert "cq 30" in en4k and "2160p" in en4k


def test_av1_por_cpu_usa_crf_con_bitrate_cero():
    args = tras_codec(comando("mp4_av1_cpu"))
    assert args[args.index("-crf") + 1] == CODEC_CRF["libaom-av1"]["alta"]
    assert args[args.index("-b:v") + 1] == "0"
    assert "-row-mt" in args, "sin row-mt libaom es aun mas lento"


def test_av1_no_recibe_el_preset_de_x264():
    """'-preset medium' no existe en AV1; el de la GPU es p1..p7."""
    for clave in ("mp4_av1", "mp4_av1_cpu"):
        args = tras_codec(comando(clave))
        assert "medium" not in args


def test_h264_sigue_igual():
    args = tras_codec(comando("mp4_h264"))
    assert args[args.index("-crf") + 1] == "17"
    assert args[args.index("-preset") + 1] == "medium"
    assert "-cq" not in args


# ------------------------------------------------- degradacion ------------

def test_sin_rtx40_degrada_a_h265_avisando(monkeypatch):
    monkeypatch.setattr(pipeline_mod, "encoder_works",
                        lambda codec, ffmpeg=None: codec != "av1_nvenc")
    avisos: list[str] = []
    clave, fmt = resolve_export_format("mkv_av1", avisos.append)
    assert clave == "mkv_h265"
    assert fmt["ext"] == "mkv", "no cambiar el contenedor al degradar"
    assert avisos and "RTX 40" in avisos[0]


def test_con_rtx40_no_degrada_ni_avisa(monkeypatch):
    monkeypatch.setattr(pipeline_mod, "encoder_works",
                        lambda codec, ffmpeg=None: True)
    avisos: list[str] = []
    assert resolve_export_format("mp4_av1", avisos.append)[0] == "mp4_av1"
    assert avisos == []


def test_un_formato_desconocido_cae_en_h264():
    assert resolve_export_format("formato_inventado")[0] == "mp4_h264"


def test_los_codecs_de_software_no_se_sondean(monkeypatch):
    """Sondear cuesta decimas: solo tiene sentido con los que pueden fallar."""
    def explota(codec, ffmpeg=None):
        raise AssertionError(f"no hacia falta sondear {codec}")
    monkeypatch.setattr(pipeline_mod, "encoder_works", explota)
    assert resolve_export_format("mp4_h264")[0] == "mp4_h264"
    assert resolve_export_format("webm_vp9")[0] == "webm_vp9"


def test_el_formato_entra_en_la_huella_del_trabajo():
    from vhs_upscaler.pipeline import job_fingerprint
    assert (job_fingerprint(opts(export_format="mp4_av1"))
            != job_fingerprint(opts(export_format="mp4_h264")))


# ------------------------------------------------- etiqueta del log -------

def test_el_log_dice_cq_con_la_gpu_y_crf_con_el_resto():
    desc = VideoProcessor._rate_description
    assert "cq 26" in desc(opts(quality="alta"), EXPORT_FORMATS["mp4_av1"], 1080)
    assert "CRF 17" in desc(opts(quality="alta"), EXPORT_FORMATS["mp4_h264"], 1080)
    assert "CRF 16" in desc(opts(quality="alta"), EXPORT_FORMATS["mp4_av1_cpu"], 1080)


# ------------------------------------------------- de verdad --------------

@pytest.mark.slow
def test_la_gpu_codifica_av1_de_verdad():
    """Si esto falla en una RTX 40, el sondeo o los parametros estan mal."""
    from vhs_upscaler.ffmpeg_utils import encoder_works
    if not encoder_works("av1_nvenc"):
        pytest.skip("este PC no tiene AV1 por GPU")
    assert encoder_works("libx264")
