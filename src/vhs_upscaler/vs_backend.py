"""Backend opcional VapourSynth + QTGMC para desentrelazado de calidad profesional.

QTGMC (Quadratic Temporal Gaussian Motion Compensated) es el estandar de oro
para VHS: desentrelazado compensado por movimiento, denoise magnetico suave
(EZDenoise), limpieza de crominancia (ChromaNoise) y doble tasa temporal
(29.97i -> 59.94p). Requiere VapourSynth y plugins; ver README.

Si vspipe no esta disponible, el pipeline degrada a bwdif automaticamente.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

from .paths import ROOT_DIR

# Grafo VapourSynth: ffms2 (indexado determinista) -> QTGMC -> CropRel
# (overscan/head-switching) -> conversion a RGB24 (SD: matriz BT.601).
# Los tokens __X__ se reemplazan al generar el script.
VPY_TEMPLATE = """import vapoursynth as vs
from vapoursynth import core
from vsdeinterlace import QTempGaussMC


def _ezdenoise(clip, tr=1):
    # EZDenoise: degrain temporal compensado por movimiento (MVTools).
    #
    # Es la limpieza buena para VHS: en vez de promediar pixeles en el mismo
    # sitio (que arrastra estelas en el movimiento), sigue cada bloque de un
    # fotograma al siguiente y promedia a lo largo de su trayectoria. El grano
    # magnetico, que no tiene trayectoria, se cancela; el detalle real se queda.
    #
    # Ojo con la API: mv.Analyse NO tiene parametro 'tr'. Hay que pedir un
    # juego de vectores por cada delta y direccion, y pasarselos todos a
    # Degrain<N> en el orden exacto (bw1, fw1, bw2, fw2...).
    sup = clip.mv.Super(pel=2, sharp=2)
    bw1 = sup.mv.Analyse(isb=True, delta=1)
    fw1 = sup.mv.Analyse(isb=False, delta=1)
    if tr <= 1:
        return clip.mv.Degrain1(sup, bw1, fw1, thsad=__THSAD__)
    bw2 = sup.mv.Analyse(isb=True, delta=2)
    fw2 = sup.mv.Analyse(isb=False, delta=2)
    return clip.mv.Degrain2(sup, bw1, fw1, bw2, fw2, thsad=__THSAD__)


clip = core.ffms2.Source(r"__INPUT__")
# SeparateFields (dentro de QTGMC) exige dimensiones pares incluso en el plano
# de croma: en YUV420 la altura debe ser multiplo de 4 (p. ej. 486 NO vale).
# Se anaden bordes a multiplos de 4 y se retiran tras desentrelazar, junto
# con el recorte de overscan. El borde es par y solo abajo/derecha: la paridad
# de campos no se altera.
pad_r = (-clip.width) % 4
pad_b = (-clip.height) % 4
if pad_r or pad_b:
    clip = core.std.AddBorders(clip, right=pad_r, bottom=pad_b)

if __QTGMC__:
    # Paridad: BFF por defecto para NTSC (__TFF__ = True para TFF/PAL)
    #
    # denoise_mode es IMPRESCINDIBLE: por defecto QTGMC usa IDENTIFY, que solo
    # separa el ruido para restaurarlo luego, y con los dos *_noise_restore a
    # 0 el resultado es que NO limpia nada. Sin esta linea, denoise_func ni
    # siquiera se llama.
    clip = QTempGaussMC(
        denoise_mode=(QTempGaussMC.NoiseProcessMode.DENOISE if __DENOISE_ON__
                      else QTempGaussMC.NoiseProcessMode.IDENTIFY),
        denoise_tr=__DENOISE_TR__,
        denoise_func=_ezdenoise,    # limpieza de grano magnetico de cinta
        sharpen_strength=0.0,       # nitidez artificial al minimo: sin halos
        motion_blur_fps_divisor=__DIV__,  # 1 = doble tasa (59.94p), 2 = original
    ).deinterlace(clip, tff=__TFF__)
elif __DENOISE_ON__:
    # Material YA progresivo (la capturadora desentrelazo por hardware, o la
    # cinta venia asi): se aprovecha SOLO la limpieza de QTGMC, sin volver a
    # desentrelazar. Pasar material progresivo por QTGMC seria destructivo:
    # su SeparateFields partiria cada fotograma entero en dos medios cuadros
    # que nunca fueron campos, y devolveria aliasing vertical inventado.
    clip = _ezdenoise(clip, tr=__DENOISE_TR__)

# Recorte de overscan / ruido de cabezales (multiples de 2) + retirada del borde
clip = core.std.CropRel(clip, left=__L__, top=__T__, right=__R__ + pad_r, bottom=__B__ + pad_b)

# --- Correccion de color EN 16 BITS ---------------------------------------
# Antes esto se hacia frame a frame en numpy, sobre RGB de 8 bits ya
# cuantizado: cada ajuste de contraste o saturacion abria huecos en el
# histograma y aparecian bandas. Aqui se sube a 16 bits, se ajusta, y solo
# al final se baja con difusion de error. Ademas no cuesta CPU del proceso
# principal: lo hace vspipe en sus propios hilos.
if __COLOR__:
    clip = clip.resize.Point(format=vs.YUV444P16,
                             matrix=__MATRIX__,
                             range_in=__RANGE__, range=vs.ColorRange.RANGE_FULL)
    brillo, contraste, saturacion = __BRIGHT__, __CONTRAST__, __SATURATION__
    # Luma: (x - medio) * contraste + medio + brillo, todo en escala 0-65535.
    clip = core.std.Expr(clip, [
        f"x 32768 - {contraste} * 32768 + {brillo * 65535 / 500.0} + 0 65535 clamp",
        f"x 32768 - {saturacion} * 32768 + 0 65535 clamp",
        f"x 32768 - {saturacion} * 32768 + 0 65535 clamp",
    ])
    clip = clip.resize.Bicubic(format=vs.RGB24, matrix_in=__MATRIX__,
                               range_in=vs.ColorRange.RANGE_FULL,
                               dither_type="error_diffusion")
else:
    # Salida RGB para el motor de escalado. La matriz y el rango salen del
    # analisis del archivo (SD NTSC = BT.601/ST170M y rango limitado por
    # defecto); acertar aqui es lo que separa unos negros correctos de unos
    # negros aplastados.
    clip = clip.resize.Bicubic(format=vs.RGB24,
                               matrix_in=__MATRIX__,
                               range_in=__RANGE__,
                               dither_type="error_diffusion")
clip.set_output()
"""


def find_vspipe() -> str | None:
    """Busca vspipe en ./runtime/bin, ./runtime, el venv activo y el PATH.

    El paquete pip de VapourSynth instala vspipe.exe en .venv\\Scripts; cuando
    la app corre con "uv run" esa carpeta ya esta en el PATH, pero al invocar
    el python del venv directamente no, asi que se comprueba explicitamente.
    """
    for candidate in (ROOT_DIR / "runtime" / "bin" / "vspipe.exe",
                      ROOT_DIR / "runtime" / "vspipe.exe",
                      Path(sys.executable).parent / "vspipe.exe"):
        if candidate.is_file():
            return str(candidate)
    return shutil.which("vspipe")


def qtgmc_available() -> bool:
    return find_vspipe() is not None


def script_fps(vspipe: str, script_path: Path) -> float | None:
    """Pregunta a vspipe el fps EXACTO del grafo (fraccion, p. ej. 60000/1001).

    Sin esto se le pasa al codificador el fps redondeado de OpenCV (30.00 en
    vez de 29.97): en una cinta de 2 h el audio se desincroniza varios segundos.
    """
    import re
    import subprocess

    try:
        out = subprocess.run(
            [vspipe, "--info", str(script_path), "-"],
            capture_output=True, text=True, timeout=120,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    m = re.search(r"^FPS:\s*(\d+)\s*/\s*(\d+)", out.stdout or "", re.MULTILINE)
    if not m:
        return None
    num, den = int(m.group(1)), int(m.group(2))
    return num / den if den else None


# Como se nombra cada matriz y cada rango dentro del script generado.
_MATRIX_EXPR = {
    "bt709": "vs.MatrixCoefficients.MATRIX_BT709",
    "bt601": "vs.MatrixCoefficients.MATRIX_ST170_M",
}
_RANGE_EXPR = {
    "full": "vs.ColorRange.RANGE_FULL",
    "limited": "vs.ColorRange.RANGE_LIMITED",
}


MVTOOLS_INCOMPATIBLE = (
    "La limpieza MVTools corre dentro del grafo de VapourSynth, que no tiene "
    "bwdif ni yadif. Con uno de esos desentrelazados no se pueden combinar: se "
    "usa hqdn3d. Para MVTools, elige QTGMC como desentrelazado, o desactiva el "
    "desentrelazado si el material ya viene progresivo."
)


def effective_denoise_method(opts) -> tuple[str, str]:
    """Metodo de limpieza que se va a usar de verdad, y el aviso si cambio.

    El grafo de VapourSynth sustituye por completo a la cadena de filtros de
    FFmpeg: si se pidiera el degrain MVTools junto a bwdif, el desentrelazado
    desapareceria sin avisar. Antes que perderlo en silencio se degrada la
    limpieza, que es lo menos grave de los dos, y se dice por que.
    """
    if not opts.denoise or opts.denoise_method != "mvtools":
        return opts.denoise_method, ""
    if opts.deinterlace and opts.deinterlace_method in ("bwdif", "yadif"):
        return "hqdn3d", MVTOOLS_INCOMPATIBLE
    return "mvtools", ""


def uses_vapoursynth(opts) -> bool:
    """Si este trabajo necesita el grafo de VapourSynth.

    Dos motivos independientes: desentrelazar con QTGMC, o limpiar con el
    degrain compensado por movimiento (que tambien sirve, y es lo unico que
    se puede aprovechar, cuando el material ya viene progresivo).
    """
    if opts.deinterlace and opts.deinterlace_method == "qtgmc":
        return True
    if not opts.denoise:
        # Sin limpieza que hacer no hay motivo para arrancar vspipe, aunque el
        # metodo guardado siga siendo mvtools de una vez anterior.
        return False
    return effective_denoise_method(opts)[0] == "mvtools"


def write_script(opts, input_path: Path, source_height: int = 480) -> Path:
    """Escribe el grafo .vpy en un archivo temporal y devuelve su ruta."""
    script = VPY_TEMPLATE
    # "auto" => BT.601 en SD (una captura de VHS siempre lo es) y rango
    # limitado, que es lo que entrega cualquier capturadora por componente.
    matrix = opts.color_matrix if opts.color_matrix in _MATRIX_EXPR else \
        ("bt601" if source_height <= 576 else "bt709")
    rango = opts.color_range if opts.color_range in _RANGE_EXPR else "limited"
    replacements = {
        # Ruta ABSOLUTA: ffms2 resuelve las relativas respecto al directorio
        # del .vpy (una carpeta temporal), no respecto al CWD del proceso.
        "__INPUT__": str(input_path.resolve()).replace("\\", "/"),
        "__TFF__": "True" if opts.field_parity == "tff" else "False",
        "__DIV__": "1" if opts.qtgmc_double_fps else "2",
        "__THSAD__": str(200 + int(opts.denoise_strength) * 60),
        "__QTGMC__": "True" if (opts.deinterlace
                                and opts.deinterlace_method == "qtgmc") else "False",
        "__DENOISE_ON__": "True" if opts.denoise else "False",
        # tr 1 o 2 segun la fuerza pedida: tr=2 mira dos fotogramas a cada
        # lado y limpia mas, a cambio de arrastrar algo mas en el movimiento.
        "__DENOISE_TR__": "2" if int(opts.denoise_strength) >= 5 else "1",
        "__L__": str(opts.crop_left), "__T__": str(opts.crop_top),
        "__R__": str(opts.crop_right), "__B__": str(opts.crop_bottom),
        "__MATRIX__": _MATRIX_EXPR[matrix],
        "__RANGE__": _RANGE_EXPR[rango],
        "__COLOR__": "True" if opts.color_correct else "False",
        "__BRIGHT__": f"{opts.brightness:d}",
        "__CONTRAST__": f"{opts.contrast / 100.0:.4f}",
        "__SATURATION__": f"{opts.saturation / 100.0:.4f}",
    }
    for token, value in replacements.items():
        script = script.replace(token, value)
    fd, tmp = tempfile.mkstemp(prefix="vhs_qtgmc_", suffix=".vpy")
    os.close(fd)  # mkstemp deja el descriptor abierto: cerrarlo antes de escribir
    Path(tmp).write_text(script, encoding="utf-8")
    return Path(tmp)
