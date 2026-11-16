"""Motores de escalado: ONNX Runtime (CPU / CUDA / DirectML), ncnn (Vulkan) y bicubico.

La interfaz es siempre la misma: recibe un frame BGR y devuelve el frame
escalado x4. El resto del programa no sabe que backend hay debajo.

El escalado se hace por bloques (tiles) con solape para:
- No agotar la memoria de GPU con frames grandes.
- Evitar costuras: cada bloque se infiere con contexto extra y solo se
  conserva la parte central.
"""

from __future__ import annotations

import atexit
from pathlib import Path

import numpy as np

from .logging_setup import get_logger
from .model_opt import resolve_model
from .models import BICUBIC_ID, get_model, is_downloaded
from .paths import MODELS_DIR

_log = get_logger(__name__)

PROVIDERS_BY_DEVICE = {
    "cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"],
    "directml": ["DmlExecutionProvider", "CPUExecutionProvider"],
    "cpu": ["CPUExecutionProvider"],
}

# TensorRT se prueba solo si el usuario lo pide con VHS_TENSORRT=1: requiere
# ~2 GB de wheels de NVIDIA que no se instalan por defecto y construye un
# "engine" por cada resolucion la primera vez (tarda minutos). Si no esta,
# onnxruntime lo descarta y sigue con CUDA sin fallar.
def _providers_for(device: str) -> list[str]:
    import os

    base = list(PROVIDERS_BY_DEVICE.get(device, PROVIDERS_BY_DEVICE["cpu"]))
    if device == "cuda" and os.environ.get("VHS_TENSORRT") == "1":
        base.insert(0, "TensorrtExecutionProvider")
    return base


class BicubicUpscaler:
    """Respaldo sin IA: util para probar el pipeline sin descargar modelos."""

    scale = 4
    device = "cpu"
    warning = None

    def upscale(self, frame: np.ndarray) -> np.ndarray:
        import cv2

        h, w = frame.shape[:2]
        return cv2.resize(frame, (w * self.scale, h * self.scale), interpolation=cv2.INTER_CUBIC)


class TiledUpscaler:
    """Base comun: recorre el frame por tiles con solape y sin costuras."""

    scale = 4
    tile_size = 256
    overlap = 16
    fixed_tile = None  # (h, w) si el modelo exige tamano fijo de entrada

    def _log(self, msg: str) -> None:
        getattr(self, "log", print)(msg)

    def _run_tile(self, tile: np.ndarray) -> np.ndarray:
        """tile BGR uint8 -> SR BGR uint8 (escala xN). Implementa cada backend."""
        raise NotImplementedError

    def upscale(self, frame: np.ndarray) -> np.ndarray:
        # tile_size <= 0 => un solo pase con el frame completo: para material SD
        # (720x480 o menos) es ~2x mas rapido que trocear, porque evita tanto el
        # solape recalculado como el coste fijo de numpy por tile. Si la GPU se
        # queda sin memoria se degrada a tiles y ya no se reintenta.
        if self.fixed_tile is None and self.tile_size <= 0:
            try:
                return self._run_tile(frame)
            except Exception as exc:
                self.tile_size = 256
                self._log(f"Frame completo no cabe en la GPU ({exc}). "
                          f"Se continua por bloques de {self.tile_size} px.")
        h, w = frame.shape[:2]
        s = self.scale
        if self.fixed_tile is not None:
            # Pasos independientes por eje: el tile (nucleo + solape) nunca debe
            # exceder el tamano fijo de entrada del modelo en ninguna dimension.
            step_y = max(self.fixed_tile[0] - 2 * self.overlap, 32)
            step_x = max(self.fixed_tile[1] - 2 * self.overlap, 32)
        else:
            step_y = step_x = self.tile_size
        out = np.empty((h * s, w * s, 3), dtype=np.uint8)
        ov = self.overlap
        for y0 in range(0, h, step_y):
            for x0 in range(0, w, step_x):
                x1, y1 = min(x0 + step_x, w), min(y0 + step_y, h)
                ex0, ey0 = max(0, x0 - ov), max(0, y0 - ov)
                ex1, ey1 = min(w, x1 + ov), min(h, y1 + ov)
                tile = frame[ey0:ey1, ex0:ex1]
                sr = self._run_tile(tile)
                cx0, cy0 = (x0 - ex0) * s, (y0 - ey0) * s
                out[y0 * s:y1 * s, x0 * s:x1 * s] = sr[cy0:cy0 + (y1 - y0) * s,
                                                       cx0:cx0 + (x1 - x0) * s]
        return out


def _preload_cuda_libs(ort, log=print) -> None:
    """Carga las DLL de CUDA/cuDNN que vienen en wheels de NVIDIA.

    onnxruntime-gpu >= 1.21 se compila contra CUDA 13 + cuDNN 9, pero no anade
    sus carpetas al buscador de DLL de Windows. Sin esto, cargar el proveedor
    CUDA falla con "cublasLt64_13.dll which is missing" (error 126) y la
    inferencia cae silenciosamente a CPU. preload_dlls() existe desde ORT 1.21;
    en versiones anteriores se anaden las carpetas a mano.
    """
    preload = getattr(ort, "preload_dlls", None)
    if callable(preload):
        try:
            preload(cuda=True, cudnn=True, msvc=True)
            return
        except TypeError:  # firmas antiguas sin algun argumento
            try:
                preload()
                return
            except Exception as exc:
                log(f"Aviso: preload_dlls() fallo ({exc}).")
        except Exception as exc:
            log(f"Aviso: preload_dlls() fallo ({exc}).")
    # Respaldo manual: registrar los directorios bin/lib de los paquetes nvidia-*
    try:
        import os
        import site

        for base in site.getsitepackages():
            nvidia = Path(base) / "nvidia"
            if not nvidia.is_dir():
                continue
            for sub in nvidia.iterdir():
                for name in ("bin", "lib"):
                    d = sub / name
                    if d.is_dir():
                        os.add_dll_directory(str(d))
    except Exception as exc:
        log(f"Aviso: no se pudieron registrar las DLL de CUDA ({exc}).")


class OnnxUpscaler(TiledUpscaler):
    def __init__(self, model_path: Path, scale: int, device: str,
                 tile_size: int = 0, overlap: int = 16, log=print, fp16: bool = True):
        import onnxruntime as ort

        if device == "cuda":
            _preload_cuda_libs(ort, log=log)

        self.log = log
        self.scale = scale
        self.tile_size = tile_size
        self.overlap = overlap
        self.warning = None
        self.device = device

        # En CUDA/DirectML se usa la variante fp16 (se genera y cachea sola):
        # ~1.5x mas rapida y sin diferencia visible en material VHS.
        # La E/S del grafo sigue siendo float32 aunque por dentro calcule en
        # fp16, asi que nada cambia del lado de numpy.
        model_path, self.fp16 = resolve_model(model_path, device, enabled=fp16, log=log)

        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        # Deja que ORT fusione y reordene el grafo al maximo antes de ejecutar.
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = _providers_for(device)
        try:
            self.session = ort.InferenceSession(str(model_path), sess_options=opts, providers=providers)
        except Exception as exc:
            if device == "cpu":
                raise
            self.warning = (
                f"No se pudo iniciar el backend '{device}' ({exc}). Se usara CPU como respaldo."
            )
            log(self.warning)
            self.device = "cpu"
            # La CPU emula fp16 por software y va MAS lenta: se vuelve a fp32.
            model_path, self.fp16 = resolve_model(model_path, "cpu", enabled=False, log=log)
            self.session = ort.InferenceSession(
                str(model_path), sess_options=opts, providers=PROVIDERS_BY_DEVICE["cpu"]
            )

        # ORT NO lanza excepcion si el proveedor pedido falla: simplemente lo
        # descarta y se queda con CPU. Hay que comprobarlo explicitamente.
        active = self.session.get_providers()
        if "TensorrtExecutionProvider" in active:
            log("TensorRT activo: la primera pasada tarda (construye el engine).")
        wanted = PROVIDERS_BY_DEVICE.get(device, PROVIDERS_BY_DEVICE["cpu"])[0]
        if self.device != "cpu" and wanted not in active:
            self.warning = (
                f"El proveedor '{wanted}' no se pudo activar; onnxruntime usara "
                f"{active[0]}. El escalado sera mucho mas lento. Ejecuta sync.bat "
                f"para reinstalar el runtime del backend '{device}'."
            )
            log(self.warning)
            self.device = "cpu"

        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        # Si el modelo tiene tamano de entrada fijo, los tiles deben respetarlo.
        self.fixed_tile = None
        if len(inp.shape) == 4 and isinstance(inp.shape[2], int) and isinstance(inp.shape[3], int):
            self.fixed_tile = (inp.shape[2], inp.shape[3])

    def _run_tile(self, tile: np.ndarray) -> np.ndarray:
        th, tw = tile.shape[:2]
        if self.fixed_tile is not None:
            fh, fw = self.fixed_tile
            if th < fh or tw < fw:
                import cv2
                tile = cv2.copyMakeBorder(
                    tile, 0, max(0, fh - th), 0, max(0, fw - tw), cv2.BORDER_REPLICATE
                )
        # BGR uint8 (H,W,3) -> RGB float32 (1,3,H,W) sin transposiciones ni
        # copias intermedias: se escribe plano a plano en el buffer definitivo.
        h, w = tile.shape[:2]
        # Toda la aritmetica de numpy va en float32: numpy no tiene SIMD para
        # float16 y operar en fp16 aqui multiplicaba por 6 el tiempo por frame.
        # El modelo fp16 convierte por dentro, en la GPU.
        tensor = np.empty((1, 3, h, w), dtype=np.float32)
        tensor[0, 0] = tile[:, :, 2]  # R
        tensor[0, 1] = tile[:, :, 1]  # G
        tensor[0, 2] = tile[:, :, 0]  # B
        tensor *= np.float32(1.0 / 255.0)
        out = self.session.run(None, {self.input_name: tensor})[0][0]
        # Vuelta a BGR uint8, de nuevo plano a plano y recortando in situ.
        oh, ow = out.shape[1], out.shape[2]
        np.clip(out, 0.0, 1.0, out=out)
        out *= np.float32(255.0)
        sr = np.empty((oh, ow, 3), dtype=np.uint8)
        sr[:, :, 0] = out[2]  # B
        sr[:, :, 1] = out[1]  # G
        sr[:, :, 2] = out[0]  # R
        return sr[: th * self.scale, : tw * self.scale]


def _find_ncnn_blobs(param_path: Path) -> tuple[str, str]:
    """Extrae los nombres de blob de entrada/salida del archivo .param."""
    in_blobs, used_as_input = [], set()
    last_output = ["out0"]
    try:
        lines = param_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return "in0", last_output[0]
    for line in lines[2:]:  # saltar magic y conteo de capas
        parts = line.split()
        if len(parts) < 4:
            continue
        layer_type, n_in, n_out = parts[0], int(parts[2]), int(parts[3])
        blobs = parts[4:]
        ins, outs = blobs[:n_in], blobs[n_in:n_in + n_out]
        used_as_input.update(ins)
        if layer_type == "Input":
            in_blobs.extend(outs)
        if outs:
            last_output = outs
    in_name = in_blobs[0] if in_blobs else "in0"
    for candidate in (last_output[-1] if last_output else "out0", "out0"):
        if candidate not in used_as_input:
            return in_name, candidate
    return in_name, "out0"


class NcnnUpscaler(TiledUpscaler):
    """Backend ncnn con Vulkan: mucho mas rapido que DirectML en iGPU AMD."""

    def __init__(self, param_path: Path, bin_path: Path, scale: int,
                 tile_size: int = 256, overlap: int = 16, log=print):
        import ncnn

        self.ncnn_module = ncnn
        self.scale = scale
        self.tile_size = tile_size
        self.overlap = overlap
        self.warning = None

        self.net = ncnn.Net()
        if ncnn.get_gpu_count() > 0:
            self.device = "vulkan"
            self.net.opt.use_vulkan_compute = True
            # fp16 en runtime: la mayoria de GPUs modernas (incluida Radeon
            # 8060S) lo soporta y duplica la velocidad.
            self.net.opt.use_fp16_packed = True
            self.net.opt.use_fp16_storage = True
            self.net.opt.use_fp16_arithmetic = True
        else:
            self.device = "cpu"
            self.warning = "Sin GPU Vulkan detectada: ncnn correra en CPU."
            log(self.warning)
        self.net.load_param(str(param_path))
        self.net.load_model(str(bin_path))
        self.in_blob, self.out_blob = _find_ncnn_blobs(param_path)
        # Red de seguridad: si nadie llama a close(), que al menos la instancia
        # Vulkan se cierre antes de que el interprete empiece a desmontar
        # modulos. Sin esto el proceso acaba en SIGSEGV con el trabajo ya hecho.
        atexit.register(self.close)

    def close(self) -> None:
        """Suelta la red y cierra Vulkan. Idempotente: close() y atexit a la vez."""
        from .hardware import close_vulkan

        net, self.net = getattr(self, "net", None), None
        if net is not None:
            net.clear()
        close_vulkan(getattr(self, "ncnn_module", None))

    def _run_tile(self, tile: np.ndarray) -> np.ndarray:
        th, tw = tile.shape[:2]
        rgb = np.ascontiguousarray(tile[:, :, ::-1].transpose(2, 0, 1)).astype(np.float32) / 255.0
        ex = self.net.create_extractor()
        ex.input(self.in_blob, self.ncnn_module.Mat(rgb))
        _, out = ex.extract(self.out_blob)
        arr = out.numpy()  # (3, h*4, w*4) en 0..1
        sr = np.clip(arr.transpose(1, 2, 0) * 255.0, 0, 255).astype(np.uint8)[:, :, ::-1]
        return sr[: th * self.scale, : tw * self.scale]


def create_upscaler(model_id: str, device: str, log=print, fp16: bool = True):
    """Fabrica el motor de escalado. La CPU siempre queda como respaldo."""
    if model_id == BICUBIC_ID:
        return BicubicUpscaler()
    entry = get_model(model_id)
    if device == "vulkan":
        param = MODELS_DIR / entry.ncnn_param if entry.ncnn_param else None
        binp = MODELS_DIR / entry.ncnn_bin if entry.ncnn_bin else None
        if not (param and param.exists() and binp and binp.exists()):
            raise RuntimeError(
                f"Faltan los archivos ncnn del modelo '{entry.name}'. Ejecuta: "
                f"uv run python tools/make_ncnn.py --model {entry.id}"
            )
        log(f"Modelo: {entry.name} | Dispositivo: vulkan (ncnn)")
        return NcnnUpscaler(param, binp, entry.scale, log=log)
    if not is_downloaded(entry):
        raise RuntimeError(
            f"El modelo '{entry.name}' no esta descargado. Ejecuta iniciar.bat "
            f"o: uv run python -m vhs_upscaler.tools.download_models --backend {device}"
        )
    log(f"Modelo: {entry.name} | Dispositivo: {device}")
    return OnnxUpscaler(entry.path, entry.scale, device, log=log, fp16=fp16)
