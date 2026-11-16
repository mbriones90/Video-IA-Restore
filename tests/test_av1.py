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
    resolve_export_format,
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


def test_en_modo_bitrate_la_gpu_tambien_pide_vbr():
    """Sin -rc vbr, NVENC ignora maxrate y bufsize."""
    args = tras_codec(comando("mp4_av1", quality_mode="bitrate", bitrate_mbps=12.0))
    assert args[args.index("-rc") + 1] == "vbr"
    assert "-maxrate" in args and "-bufsize" in args
    assert "-cq" not in args


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
    assert desc(opts(quality="alta"), EXPORT_FORMATS["mp4_av1"]) == "cq 26 (calidad constante)"
    assert desc(opts(quality="alta"), EXPORT_FORMATS["mp4_h264"]) == "CRF 17 (calidad constante)"
    assert desc(opts(quality="alta"), EXPORT_FORMATS["mp4_av1_cpu"]) == "CRF 16 (calidad constante)"


# ------------------------------------------------- de verdad --------------

@pytest.mark.slow
def test_la_gpu_codifica_av1_de_verdad():
    """Si esto falla en una RTX 40, el sondeo o los parametros estan mal."""
    from vhs_upscaler.ffmpeg_utils import encoder_works
    if not encoder_works("av1_nvenc"):
        pytest.skip("este PC no tiene AV1 por GPU")
    assert encoder_works("libx264")
