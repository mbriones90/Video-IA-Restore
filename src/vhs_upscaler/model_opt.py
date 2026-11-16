"""Variante fp16 del modelo ONNX, generada y cacheada automaticamente.

Los pesos oficiales vienen en fp32 y con el lote fijo a 1. En una GPU moderna
la mitad del trabajo de una convolucion fp32 es desperdicio: los tensor cores
trabajan en fp16. Medido en RTX 4070 con `realesr-general-x4v3`, 720x480 x4:

    fp32: 73 ms/frame
    fp16: 50 ms/frame   (1.47x)

El lote tambien se hace dinamico, aunque NO se usa en el pipeline: medido,
agrupar 2 o 4 frames no mejora nada (la GPU ya se satura con un frame SD de
1.4 Mpx que sale a 5.5 Mpx). Se deja abierto por si en el futuro se procesa
material mas pequeno, donde el coste fijo de lanzar los kernels si pesaria.

La conversion se hace UNA vez por modelo y queda en `models/<nombre>.fp16.onnx`.
Si faltan `onnx`/`onnxconverter-common`, o la conversion falla, se sigue con
fp32 sin molestar al usuario: es una optimizacion, no un requisito.
"""

from __future__ import annotations

from pathlib import Path

from .logging_setup import get_logger

log = get_logger(__name__)

FP16_SUFFIX = ".fp16.onnx"
# Dispositivos cuyo runtime aprovecha fp16. La CPU lo emula y va MAS LENTA.
FP16_DEVICES = ("cuda", "directml")


def fp16_path(model_path: Path) -> Path:
    return model_path.with_suffix("").with_suffix(FP16_SUFFIX) \
        if model_path.suffix == ".onnx" else Path(str(model_path) + FP16_SUFFIX)


def _make_batch_dynamic(model) -> None:
    """Sustituye el lote fijo (1) por una dimension simbolica."""
    for vi in list(model.graph.input) + list(model.graph.output):
        shape = vi.type.tensor_type.shape
        if not shape.dim:
            continue
        d = shape.dim[0]
        if d.HasField("dim_value"):
            d.ClearField("dim_value")
        d.dim_param = "batch"


def build_fp16(model_path: Path, out_path: Path | None = None, log=log.info) -> Path | None:
    """Convierte un ONNX fp32 a fp16. Devuelve la ruta, o None si no se pudo."""
    out_path = out_path or fp16_path(model_path)
    try:
        import warnings

        import onnx
        from onnxconverter_common import float16
    except ImportError:
        log("Optimizacion fp16 no disponible (falta el paquete 'onnx'). Se usa fp32.")
        return None

    try:
        model = onnx.load(str(model_path))
        _make_batch_dynamic(model)
        with warnings.catch_warnings():
            # Avisa de constantes diminutas que se truncan al minimo de fp16;
            # es esperado y no afecta al resultado visible.
            warnings.simplefilter("ignore")
            # keep_io_types=True: entrada y salida siguen siendo float32 y los
            # Cast a fp16 quedan DENTRO del grafo, o sea en la GPU. Con
            # keep_io_types=False habria que convertir en numpy, y convertir
            # 16.6 M de float16 a float32 por frame cuesta mas (35 ms) que todo
            # lo que ahorra la GPU (23 ms).
            model16 = float16.convert_float_to_float16(model, keep_io_types=True)
        tmp = out_path.with_suffix(out_path.suffix + ".part")
        onnx.save(model16, str(tmp))
        tmp.replace(out_path)
        log(f"Modelo fp16 generado: {out_path.name}")
        return out_path
    except Exception as exc:
        log(f"No se pudo generar la variante fp16 ({exc}). Se usa fp32.")
        try:
            out_path.with_suffix(out_path.suffix + ".part").unlink(missing_ok=True)
        except OSError:
            pass
        return None


def resolve_model(model_path: Path, device: str, enabled: bool = True,
                  log=log.info) -> tuple[Path, bool]:
    """Devuelve (ruta_a_usar, es_fp16) para este dispositivo.

    Genera la variante fp16 la primera vez y la reutiliza despues. La cache se
    invalida si el ONNX original es mas nuevo que la variante.
    """
    if not enabled or device not in FP16_DEVICES:
        return model_path, False
    cached = fp16_path(model_path)
    try:
        fresh = cached.exists() and cached.stat().st_mtime >= model_path.stat().st_mtime
    except OSError:
        fresh = False
    if fresh:
        return cached, True
    log("Generando variante fp16 del modelo (solo la primera vez)...")
    built = build_fp16(model_path, cached, log=log)
    return (built, True) if built else (model_path, False)
