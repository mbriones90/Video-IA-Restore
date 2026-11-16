"""Pipeline de procesamiento: un video a la vez.

Flujo:
  FFmpeg/vspipe (decodifica + filtros VHS) -> frames BGR en crudo
    -> motor de escalado (IA o bicubico)
      -> FFmpeg (escala final SIN deformar + codificacion)
        -> concatenado de segmentos + audio original

Tuberia en memoria (stdout -> stdin), sin archivos temporales masivos.
La proporcion original NUNCA se deforma: la salida se completa con barras
negras (pillarbox/letterbox) cuando hace falta. No hay zoom ni recorte
automatico; el recorte de overscan es una opcion del usuario (manual o
propuesto por `analysis.detect_crop`).

Dos decisiones de diseno que explican casi todo lo de abajo:

1. **Las tres etapas corren a la vez.** Decodificar (QTGMC va a ~13 fps),
   escalar (la GPU va a ~14 fps) y codificar son independientes, asi que van
   en hilos distintos unidos por colas acotadas. En serie el total es la suma;
   solapados es el maximo. En una cinta de 2 h eso son horas de diferencia.

2. **Se procesa por segmentos y se puede reanudar.** Una cinta larga tarda
   horas; si falla al 80 % no se tira el trabajo. Cada segmento se codifica
   solo-video a `output/.jobs/<id>/`, y al final se concatenan con copia de
   flujo y se les pega el audio del original. Al reanudar se saltan los
   segmentos ya terminados.
"""

from __future__ import annotations

import hashlib
import json
import os
import queue
import subprocess
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .ffmpeg_utils import VideoInfo, get_ffmpeg_exe, probe_video
from .inference import create_upscaler
from .logging_setup import get_logger
from .paths import OUTPUT_DIR, ROOT_DIR

log_file = get_logger(__name__)

QUALITY_PRESETS = {
    # crf: menor = mejor calidad. preset: mas lento = mejor compresion.
    "alta": {"crf": "17", "preset": "medium"},
    "media": {"crf": "20", "preset": "medium"},
    "rapida": {"crf": "23", "preset": "veryfast"},
}

# Formatos de exportacion final (tras el escalado).
# rate=True admite CRF o bitrate objetivo.
EXPORT_FORMATS = {
    "mp4_h264": {
        "label": "MP4 - H.264 (compatibilidad total)",
        "video": "libx264", "audio": "aac", "ext": "mp4", "rate": True,
        "pix_fmt": "yuv420p",
    },
    "mp4_h265": {
        "label": "MP4 - H.265 (mejor compresion)",
        "video": "libx265", "audio": "aac", "ext": "mp4", "rate": True,
        "pix_fmt": "yuv420p", "tag": "hvc1",
    },
    "mkv_h264": {
        "label": "MKV - H.264 (contenedor flexible)",
        "video": "libx264", "audio": "aac", "ext": "mkv", "rate": True,
        "pix_fmt": "yuv420p",
    },
    "mkv_h265": {
        "label": "MKV - H.265 (contenedor flexible, mejor compresion)",
        "video": "libx265", "audio": "aac", "ext": "mkv", "rate": True,
        "pix_fmt": "yuv420p",
    },
    "webm_vp9": {
        "label": "WebM - VP9 (ideal para web)",
        "video": "libvpx-vp9", "audio": "libopus", "ext": "webm", "rate": True,
        "pix_fmt": "yuv420p",
    },
}

# CRF optimo por codec segun calidad (VP9 usa una escala distinta a x264/x265).
CODEC_CRF = {
    "libx264": {"alta": "17", "media": "20", "rapida": "23"},
    "libx265": {"alta": "18", "media": "21", "rapida": "24"},
    "libvpx-vp9": {"alta": "30", "media": "33", "rapida": "36"},
}

# Carpeta de trabajo de los segmentos (permite reanudar un trabajo cortado).
JOBS_DIR = OUTPUT_DIR / ".jobs"
# Duracion por defecto de cada segmento, en segundos de video de origen.
DEFAULT_SEGMENT_SECONDS = 300


class ProcessingCancelled(Exception):
    pass


@dataclass
class ProcessOptions:
    input_path: str
    output_path: str
    model_id: str = "realesr-general-x4v3"
    device: str = "cpu"
    export_format: str = "mp4_h264"
    quality: str = "alta"
    quality_mode: str = "crf"  # crf | bitrate
    bitrate_mbps: float = 16.0  # bitrate objetivo (Mbps) cuando quality_mode = "bitrate"
    deinterlace: bool = True
    deinterlace_method: str = "bwdif"  # bwdif | yadif | qtgmc
    field_parity: str = "auto"  # auto | bff | tff
    crop_left: int = 0
    crop_right: int = 0
    crop_top: int = 0
    crop_bottom: int = 0
    chroma_denoise: bool = True
    qtgmc_double_fps: bool = True
    denoise: bool = True
    denoise_strength: int = 3
    color_correct: bool = False
    brightness: int = 0
    contrast: int = 100
    saturation: int = 110
    deband: bool = False
    grain: bool = False  # grano de pelicula ligero (anti "oleoso")
    grain_strength: int = 6  # intensidad del grano (1-20)
    audio_mono: bool = False  # unificar audio estereo a mono (canal central)
    interpolate_fps: int = 0
    target_width: int = 1920
    target_height: int = 1080
    source_sar: float = 1.0  # proporcion del pixel del origen (NTSC 720x480 = 8/9)
    start_seconds: float = 0.0  # > 0: modo prueba (desde este segundo)
    max_seconds: float = 0.0    # > 0: limita la duracion procesada
    # --- Color ---
    color_range: str = "auto"   # auto | limited | full (rango del ORIGEN)
    color_matrix: str = "auto"  # auto | bt601 | bt709 (matriz del ORIGEN)
    # --- Estabilizacion ---
    deflicker: bool = False     # corrige el "bombeo" de brillo entre frames
    deflicker_window: int = 5   # frames de la media movil
    # --- Exposicion automatica por escena ---
    auto_exposure: bool = False        # nivela escenas de dia y de noche
    auto_exposure_target: int = 56     # mediana de luma objetivo (0-255)
    auto_exposure_strength: int = 80   # 0-100: cuanto de la correccion se aplica
    auto_exposure_smoothing: int = 30  # frames de la media movil del exponente
    # --- Rendimiento y reanudacion ---
    fp16: bool = True           # usa la variante fp16 del modelo en GPU
    segment_seconds: int = DEFAULT_SEGMENT_SECONDS  # 0 = un solo tramo
    resume: bool = True         # reaprovecha segmentos ya terminados


# ---------------------------------------------------------------------------
# Filtros y dimensiones
# ---------------------------------------------------------------------------

def cropped_dimensions(width: int, height: int, opts: ProcessOptions) -> tuple[int, int]:
    """Valida y aplica el recorte de overscan / head-switching (multiplos de 2)."""
    for name in ("crop_left", "crop_right", "crop_top", "crop_bottom"):
        v = getattr(opts, name)
        if v % 2 != 0:
            raise ValueError(f"El recorte '{name}' debe ser multiplo de 2 (valor recibido: {v}).")
    w = width - opts.crop_left - opts.crop_right
    h = height - opts.crop_top - opts.crop_bottom
    if w < 64 or h < 64:
        raise ValueError("El recorte deja una imagen demasiado pequena (minimo 64x64).")
    return w, h


def build_decode_filters(opts: ProcessOptions) -> list[str]:
    """Filtros de limpieza VHS, en el orden correcto (antes de escalar).

    Si el metodo elegido es QTGMC pero se usa la ruta FFmpeg (respaldo cuando
    falta VapourSynth, o vista previa de un frame), se aplica bwdif como
    equivalente aproximado: nunca se procesa sin desentrelazar por accidente.
    """
    filters = []
    if opts.color_range in ("limited", "full"):
        # Le dice a swscale como interpretar el YUV del origen. Si se equivoca,
        # los negros salen aplastados (o lavados) y ya no hay vuelta atras.
        src = "tv" if opts.color_range == "limited" else "pc"
        filters.append(f"scale=in_range={src}:out_range=pc")
    if opts.deinterlace:
        method = opts.deinterlace_method if opts.deinterlace_method in ("bwdif", "yadif") else "bwdif"
        parity = opts.field_parity if opts.field_parity in ("bff", "tff") else "auto"
        filters.append(f"{method}=mode=send_frame:parity={parity}:deint=all")
    if any(getattr(opts, n) for n in ("crop_left", "crop_right", "crop_top", "crop_bottom")):
        l, r, t, b = opts.crop_left, opts.crop_right, opts.crop_top, opts.crop_bottom
        filters.append(f"crop=iw-{l}-{r}:ih-{t}-{b}:{l}:{t}")
    if opts.denoise and opts.denoise_strength > 0:
        s = float(opts.denoise_strength)
        chroma_tmp = s * 2.0 if opts.chroma_denoise else s * 1.125
        filters.append(f"hqdn3d={s:.2f}:{s * 0.75:.2f}:{chroma_tmp:.2f}:{s * 0.75:.2f}")
    if opts.color_correct:
        filters.append(
            f"eq=brightness={opts.brightness / 500:.4f}"
            f":contrast={opts.contrast / 100:.4f}"
            f":saturation={opts.saturation / 100:.4f}"
        )
    if opts.deband:
        filters.append("deband")
    if opts.interpolate_fps:
        filters.append(f"minterpolate=fps={opts.interpolate_fps}:mi_mode=mci")
    return filters


def build_output_filter(opts: ProcessOptions) -> str:
    """Ajusta al lienzo de salida conservando la proporcion y rellenando con negro.

    Si hay grano, se aplica SOBRE la imagen escalada (antes del pad), para que
    las barras negras queden limpias. La conversion RGB -> YUV se fija a BT.709
    con rango limitado, que es justo lo que luego se etiqueta en el contenedor.
    """
    tw, th = opts.target_width, opts.target_height
    parts = []
    if abs(opts.source_sar - 1.0) > 0.01:
        # A pixeles cuadrados ANTES de encajar en el lienzo: 'scale' razona en
        # pixeles, no en proporcion visible, asi que si no se hace aqui una
        # captura NTSC marcada 8:9 sale estirada un 12 %.
        parts.append(f"scale=round(iw*{opts.source_sar:.6f}/2)*2:ih,setsar=1")
    parts += [
        f"scale={tw}:{th}:force_original_aspect_ratio=decrease"
        f":force_divisible_by=2:flags=lanczos"
        f":out_color_matrix=bt709:out_range=tv",
    ]
    if opts.grain and opts.grain_strength > 0:
        # Grano de pelicula ligero: quita el aspecto "oleoso" de la IA.
        parts.append(f"noise=alls={int(opts.grain_strength)}:allf=t+u")
    parts.append(f"pad={tw}:{th}:(ow-iw)/2:(oh-ih)/2:color=black")
    parts.append("setsar=1")
    return ",".join(parts)


def add_grain(frame_bgr: np.ndarray, strength: int) -> np.ndarray:
    """Grano gaussiano en memoria (equivalente aproximado para vistas previas)."""
    if strength <= 0:
        return frame_bgr
    rng = np.random.default_rng()
    noise = rng.normal(0.0, float(strength) * 1.5, frame_bgr.shape).astype(np.float32)
    return np.clip(frame_bgr.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def output_fps(opts: ProcessOptions, info: VideoInfo, qtgmc: bool = False) -> float:
    if qtgmc:
        return info.fps * (2.0 if opts.qtgmc_double_fps else 1.0)
    return float(opts.interpolate_fps) if opts.interpolate_fps else info.fps


def effective_duration(opts: ProcessOptions, info: VideoInfo) -> float:
    dur = max(0.0, info.duration - opts.start_seconds)
    if opts.max_seconds > 0:
        dur = min(dur, opts.max_seconds)
    return dur


def fit_to_canvas(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """Equivalente en OpenCV del filtro de salida (para vistas previas)."""
    import cv2

    h, w = frame.shape[:2]
    scale = min(width / w, height / h)
    nw, nh = max(2, round(w * scale / 2) * 2), max(2, round(h * scale / 2) * 2)
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    x, y = (width - nw) // 2, (height - nh) // 2
    canvas[y:y + nh, x:x + nw] = resized
    return canvas


def apply_rgb_eq(frame_bgr: np.ndarray, brightness: int, contrast: int, saturation: int) -> np.ndarray:
    """Correccion de color en RGB (respaldo; la ruta QTGMC la hace en 16 bits)."""
    f = frame_bgr[:, :, ::-1].astype(np.float32)
    b = brightness / 500.0 * 255.0
    c = contrast / 100.0
    s = saturation / 100.0
    f = (f - 128.0) * c + 128.0 + b
    if s != 1.0:
        luma = f[:, :, 0] * 0.299 + f[:, :, 1] * 0.587 + f[:, :, 2] * 0.114
        f = luma[:, :, None] + (f - luma[:, :, None]) * s
    return np.clip(f, 0, 255)[:, :, ::-1].astype(np.uint8)


class Deflicker:
    """Corrige el "bombeo" de brillo tipico del VHS, frame a frame.

    El control automatico de ganancia del reproductor hace que el brillo medio
    suba y baje entre fotogramas; tras escalar con IA se nota como un parpadeo.
    Se normaliza la luma media de cada frame hacia la media movil de los
    ultimos N, que es lo que hace el filtro `deflicker` de FFmpeg, pero aqui
    sirve igual para la ruta QTGMC, donde los frames ya no pasan por FFmpeg.

    NO corrige el jitter horizontal de cabezal (time-base error): eso solo se
    arregla de verdad capturando la senal RF (vhs-decode), no sobre el video
    ya digitalizado.
    """

    def __init__(self, window: int = 5):
        self.window = max(2, int(window))
        self._means: deque[float] = deque(maxlen=self.window)

    def apply(self, frame_bgr: np.ndarray) -> np.ndarray:
        # La media de los canales BGR basta como aproximacion de luma y es
        # mucho mas barata que convertir el frame entero a YUV.
        mean = float(frame_bgr.mean())
        if mean <= 1.0:  # fundido a negro: no tocar, amplificaria ruido
            self._means.append(mean)
            return frame_bgr
        if not self._means:
            self._means.append(mean)
            return frame_bgr
        target = sum(self._means) / len(self._means)
        self._means.append(mean)
        gain = target / mean
        # Tope del 12 %: un corte de plano cambia el brillo de verdad y no hay
        # que "corregirlo"; solo se suaviza la oscilacion del AGC.
        gain = min(max(gain, 0.88), 1.12)
        if abs(gain - 1.0) < 0.002:
            return frame_bgr
        out = frame_bgr.astype(np.float32)
        out *= np.float32(gain)
        np.clip(out, 0, 255, out=out)
        return out.astype(np.uint8)


class AutoExposure:
    """Nivela el brillo entre escenas de dia y de noche, sin tocar negros ni blancos.

    Por que gamma y no un estirado de niveles: medido sobre una cinta familiar
    real, una escena oscura y una clara tienen el MISMO punto de negro (p1=14)
    y el MISMO punto de blanco (p99=229); lo unico que cambia es la mediana
    (28 frente a 56). No hay histograma que estirar, asi que `normalize` (y
    cualquier auto-levels) no solo no arregla nada: oscurece todavia mas.
    Lo que hay que mover son los medios tonos, y eso es gamma.

    La curva se aplica sobre el rango normalizado entre p1 y p99, de modo que
    el negro sigue en su sitio y las altas luces no se queman: en la misma
    prueba, el vestido blanco de la escena se mantuvo sin recortar.

    Tres detalles que separan esto de un simple `eq=gamma`:

    - **Suavizado temporal del EXPONENTE, no del pixel.** Si cada fotograma
      eligiera su propio gamma, la imagen latiria.
    - **Reinicio en los cortes de plano.** Sin esto se arrastra la exposicion
      de la escena anterior durante el medio segundo siguiente al corte.
    - **Tope de correccion.** Una escena nocturna a proposito debe seguir
      pareciendo nocturna, no gris lavada.
    """

    def __init__(self, target: int = 56, strength: int = 80, smoothing: int = 30):
        self.target = float(min(max(target, 16), 160))
        self.strength = float(min(max(strength, 0), 100)) / 100.0
        self._inv: deque[float] = deque(maxlen=max(2, int(smoothing)))
        self._prev_small: np.ndarray | None = None
        # Topes del exponente (1/gamma): por debajo de 1 aclara, por encima
        # oscurece. Son ASIMETRICOS a proposito.
        #
        # Aclarar una escena oscura recupera caras que no se veian: merece la
        # pena hasta gamma 2.2 (inv 0.45). Oscurecer una escena ya bien
        # expuesta, en cambio, comprime las sombras contra el negro y hace
        # desaparecer lo que hay en segundo plano; probado sobre una escena
        # diurna de la cinta, bajarla al mismo nivel que una nocturna convertia
        # la ropa oscura del fondo en un borron negro y hundia el p1 de 10 a 1.
        #
        # Asi que el objetivo funciona practicamente como SUELO: se aclara
        # mucho si hace falta y se oscurece solo lo justo para quitar un brillo
        # exagerado. Una escena soleada debe seguir pareciendo soleada.
        self.min_inv, self.max_inv = 0.45, 1.12
        # Diferencia media entre miniaturas que se considera corte de plano.
        self.cut_threshold = 24.0

    @staticmethod
    def _thumb(frame_bgr: np.ndarray) -> np.ndarray:
        """Miniatura en gris: las estadisticas no necesitan el frame entero."""
        import cv2

        small = frame_bgr[::4, ::4]
        return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    def _is_cut(self, thumb: np.ndarray) -> bool:
        prev, self._prev_small = self._prev_small, thumb
        if prev is None or prev.shape != thumb.shape:
            return True
        diff = np.abs(thumb.astype(np.int16) - prev.astype(np.int16)).mean()
        return bool(diff > self.cut_threshold)

    def measure(self, frame_bgr: np.ndarray) -> float | None:
        """Exponente 1/gamma para este fotograma, o None si no hay que tocarlo."""
        thumb = self._thumb(frame_bgr)
        if self._is_cut(thumb):
            self._inv.clear()  # escena nueva: no arrastrar la exposicion anterior
        black, white = np.percentile(thumb, (1.0, 99.0))
        median = float(np.median(thumb))
        # Fundido a negro, pantalla plana o cinta sin senal: nada que nivelar.
        if white - black < 24.0 or median <= black + 1.0:
            return None
        span = white - black
        x = (median - black) / span
        y = (self.target - black) / span
        x = min(max(x, 0.02), 0.98)
        y = min(max(y, 0.02), 0.98)
        inv = float(np.log(y) / np.log(x))
        # strength interpola hacia 1.0 (sin correccion).
        inv = 1.0 + (inv - 1.0) * self.strength
        inv = min(max(inv, self.min_inv), self.max_inv)
        self._inv.append(inv)
        return sum(self._inv) / len(self._inv)

    def apply(self, frame_bgr: np.ndarray) -> np.ndarray:
        import cv2

        inv = self.measure(frame_bgr)
        if inv is None or abs(inv - 1.0) < 0.01:
            return frame_bgr
        thumb = self._prev_small
        black, white = np.percentile(thumb, (1.0, 99.0))
        span = max(white - black, 1.0)
        x = np.arange(256, dtype=np.float32)
        # La curva actua SOLO entre el negro y el blanco medidos; fuera de ese
        # tramo se deja el valor tal cual. Recortar con clip en vez de esto
        # aplastaba contra el negro todo lo que cayera por debajo de p1 (que es
        # el 1 % mas oscuro, mas lo que la miniatura no alcanza a ver): medido,
        # subia el negro de 14 a 28 al aclarar y lo hundia de 10 a 1 al
        # oscurecer. Los extremos deben quedarse donde estan.
        t = np.clip((x - black) / span, 0.0, 1.0)
        curva = black + (t ** inv) * span
        dentro = (x > black) & (x < white)
        lut = np.where(dentro, curva, x)
        lut = np.clip(lut, 0, 255).astype(np.uint8)
        # La misma curva en los tres canales: mantiene el tono y sube un punto
        # la saturacion de los medios, que en VHS desvaido sienta bien. Tocar
        # solo la luma dejaria el croma corto y la imagen saldria deslavada.
        return cv2.LUT(frame_bgr, lut)


def build_auto_exposure(opts: ProcessOptions) -> AutoExposure:
    """Crea el corrector a partir de las opciones (lo usan pipeline y vista previa)."""
    return AutoExposure(
        target=opts.auto_exposure_target,
        strength=opts.auto_exposure_strength,
        smoothing=opts.auto_exposure_smoothing,
    )


# ---------------------------------------------------------------------------
# Entorno de subprocesos
# ---------------------------------------------------------------------------

def process_env() -> dict:
    """Clona el entorno y antepone ./runtime/bin al PATH (resolucion de .dll)."""
    env = os.environ.copy()
    runtime_bin = ROOT_DIR / "runtime" / "bin"
    if runtime_bin.is_dir():
        env["PATH"] = str(runtime_bin) + os.pathsep + env.get("PATH", "")
    return env


def _flags() -> int:
    return getattr(subprocess, "CREATE_NO_WINDOW", 0)


def _stderr_pumper(proc: subprocess.Popen, buffer: deque, name: str) -> threading.Thread:
    def pump():
        # stderr viaja en BYTES (stdout debe seguir en binario para los frames):
        # se decodifica aqui para que los mensajes de error sean legibles.
        for line in proc.stderr:
            if isinstance(line, bytes):
                line = line.decode("utf-8", errors="replace")
            line = line.strip()
            if line:
                buffer.append(f"[{name}] {line}")

    t = threading.Thread(target=pump, daemon=True)
    t.start()
    return t


def _read_exact(pipe, size: int) -> bytes:
    chunks = []
    remaining = size
    while remaining > 0:
        chunk = pipe.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _wait_or_kill(proc: subprocess.Popen, timeout: float) -> None:
    """Espera a que el subproceso termine; si se resiste, lo mata sin fallar."""
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass


def _unlink_retry(path: Path, attempts: int = 10) -> None:
    """Borra un archivo que un subproceso quiza acabe de soltar.

    En Windows el handle de FFmpeg puede seguir vivo unos milisegundos tras
    terminar el proceso, y os.unlink falla con WinError 32.
    """
    for i in range(attempts):
        try:
            path.unlink(missing_ok=True)
            return
        except OSError:
            time.sleep(0.1 * (i + 1))
    log_file.warning("No se pudo borrar el temporal %s", path)


class PauseToken:
    """Permite pausar y reanudar el procesamiento sin perder nada de lo hecho.

    Por dentro es un Event que esta PUESTO mientras se trabaja; pausar lo
    limpia y el hilo de escalado se queda esperando. Como las colas del
    pipeline son acotadas, parar ese hilo frena solos al decodificador y al
    codificador: se llenan sus colas, se bloquean escribiendo en la tuberia y
    ahi se quedan, vivos y sin gastar CPU. No hay nada que guardar ni que
    rehacer al reanudar, y los subprocesos no se enteran.

    Es distinto de cancelar: cancelar tira el segmento a medias (aunque
    conserva los tramos ya terminados, ver `Job`), mientras que pausar deja
    todo exactamente donde estaba.

    Ademas lleva la cuenta del tiempo detenido, porque si no la velocidad y el
    tiempo restante que ve el usuario saldrian absurdos tras una pausa larga.
    """

    def __init__(self) -> None:
        self._running = threading.Event()
        self._running.set()
        self._paused_at: float | None = None
        self._accumulated = 0.0
        self._lock = threading.Lock()

    # -- control --------------------------------------------------------
    def pause(self) -> None:
        with self._lock:
            if self._paused_at is None:
                self._paused_at = time.monotonic()
            self._running.clear()

    def resume(self) -> None:
        with self._lock:
            if self._paused_at is not None:
                self._accumulated += time.monotonic() - self._paused_at
                self._paused_at = None
            self._running.set()

    def toggle(self) -> bool:
        """Cambia de estado y devuelve True si queda en pausa."""
        if self.is_paused:
            self.resume()
            return False
        self.pause()
        return True

    @property
    def is_paused(self) -> bool:
        return not self._running.is_set()

    @property
    def paused_seconds(self) -> float:
        """Total detenido, incluida la pausa en curso."""
        with self._lock:
            extra = (time.monotonic() - self._paused_at) if self._paused_at else 0.0
            return self._accumulated + extra

    # -- uso desde el pipeline -------------------------------------------
    def wait(self, cancel_event: threading.Event | None = None) -> None:
        """Bloquea mientras este pausado. Cancelar durante la pausa funciona."""
        if self._running.is_set():
            return  # camino rapido: ni una llamada de mas por fotograma
        while not self._running.wait(0.2):
            if cancel_event is not None and cancel_event.is_set():
                raise ProcessingCancelled()


def run_ffmpeg(args: list[str], what: str) -> None:
    """Corre FFmpeg y lanza un RuntimeError legible si falla."""
    proc = subprocess.run(args, capture_output=True, creationflags=_flags(), env=process_env())
    if proc.returncode != 0:
        detail = proc.stderr.decode("utf-8", errors="replace").strip().splitlines()[-15:]
        raise RuntimeError(f"Fallo al {what}:\n" + "\n".join(detail))


# ---------------------------------------------------------------------------
# Vista previa de un frame
# ---------------------------------------------------------------------------

def extract_filtered_frame(opts: ProcessOptions, second: float) -> np.ndarray:
    """Extrae un frame con los filtros VHS aplicados (vista previa real)."""
    ffmpeg = get_ffmpeg_exe()
    info = probe_video(opts.input_path)
    cw, ch = cropped_dimensions(info.width, info.height, opts)
    filters = build_decode_filters(opts)
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error",
           "-ss", f"{second:.3f}", "-i", str(opts.input_path), "-map", "0:v:0"]
    if filters:
        cmd += ["-vf", ",".join(filters)]
    cmd += ["-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]
    proc = subprocess.run(cmd, capture_output=True, creationflags=_flags(), env=process_env())
    if proc.returncode != 0 or not proc.stdout:
        raise RuntimeError(f"FFmpeg no pudo extraer el frame: {proc.stderr.decode(errors='replace')}")
    expected = cw * ch * 3
    data = proc.stdout[:expected]
    if len(data) < expected:
        raise RuntimeError("Frame incompleto al extraer la vista previa.")
    return np.frombuffer(data, dtype=np.uint8).reshape(ch, cw, 3).copy()


# ---------------------------------------------------------------------------
# Trabajos reanudables
# ---------------------------------------------------------------------------

@dataclass
class Segment:
    index: int
    start: float      # segundo absoluto del ORIGEN donde empieza
    duration: float   # segundos de origen que cubre
    frames: int       # frames de salida esperados (o producidos, si done)
    done: bool = False


@dataclass
class Job:
    job_id: str
    input_path: str
    output_path: str
    fingerprint: str
    fps_out: float
    width: int
    height: int
    segments: list[Segment] = field(default_factory=list)

    @property
    def work_dir(self) -> Path:
        return JOBS_DIR / self.job_id

    @property
    def state_path(self) -> Path:
        return self.work_dir / "state.json"

    def segment_path(self, seg: Segment) -> Path:
        return self.work_dir / f"seg_{seg.index:04d}.mkv"

    @property
    def pending(self) -> list[Segment]:
        return [s for s in self.segments if not s.done]

    def save(self) -> None:
        self.work_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(".json.part")
        tmp.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        tmp.replace(self.state_path)

    @classmethod
    def load(cls, state_path: Path) -> "Job | None":
        try:
            data = json.loads(state_path.read_text(encoding="utf-8"))
            segs = [Segment(**s) for s in data.pop("segments", [])]
            return cls(segments=segs, **data)
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return None


# Opciones que, si cambian, invalidan los segmentos ya calculados. El nombre
# del archivo de salida NO entra: renombrar la salida no debe obligar a repetir
# horas de proceso.
_FINGERPRINT_FIELDS = (
    "input_path", "model_id", "device", "export_format", "quality", "quality_mode",
    "bitrate_mbps", "deinterlace", "deinterlace_method", "field_parity",
    "crop_left", "crop_right", "crop_top", "crop_bottom", "chroma_denoise",
    "qtgmc_double_fps", "denoise", "denoise_strength", "color_correct",
    "brightness", "contrast", "saturation", "deband", "grain", "grain_strength",
    "interpolate_fps", "target_width", "target_height", "start_seconds",
    "max_seconds", "color_range", "color_matrix", "deflicker", "deflicker_window",
    "source_sar", "auto_exposure", "auto_exposure_target", "auto_exposure_strength",
    "auto_exposure_smoothing",
)


def job_fingerprint(opts: ProcessOptions) -> str:
    """Huella de los ajustes que afectan a los pixeles del resultado."""
    payload = {name: getattr(opts, name) for name in _FINGERPRINT_FIELDS}
    try:
        payload["mtime"] = Path(opts.input_path).stat().st_mtime_ns
    except OSError:
        payload["mtime"] = 0
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def plan_segments(opts: ProcessOptions, info: VideoInfo, fps_out: float) -> list[Segment]:
    """Parte el trabajo en tramos de `segment_seconds` del video de ORIGEN."""
    total = effective_duration(opts, info)
    size = float(opts.segment_seconds) if opts.segment_seconds > 0 else 0.0
    if size <= 0 or total <= size:
        return [Segment(0, opts.start_seconds, total, max(1, int(round(total * fps_out))))]
    segments, index, offset = [], 0, 0.0
    while offset < total - 1e-6:
        dur = min(size, total - offset)
        segments.append(Segment(
            index=index, start=opts.start_seconds + offset, duration=dur,
            frames=max(1, int(round(dur * fps_out))),
        ))
        offset += dur
        index += 1
    return segments


def find_resumable_jobs() -> list[Job]:
    """Trabajos a medias que se pueden reanudar (para avisar en la interfaz)."""
    jobs = []
    if not JOBS_DIR.is_dir():
        return jobs
    for state in sorted(JOBS_DIR.glob("*/state.json")):
        job = Job.load(state)
        if job and job.pending and any(s.done for s in job.segments):
            jobs.append(job)
    return jobs


def cleanup_job(job: Job) -> None:
    import shutil

    shutil.rmtree(job.work_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Procesador principal
# ---------------------------------------------------------------------------

class VideoProcessor:
    """Escala un video completo, por segmentos y con las etapas solapadas."""

    def run(self, opts: ProcessOptions, progress=None, log=print,
            cancel_event: threading.Event | None = None,
            pause: "PauseToken | None" = None) -> Path:
        from . import vs_backend

        cancel_event = cancel_event or threading.Event()
        pause = pause or PauseToken()
        progress = progress or (lambda done, total: None)

        ffmpeg = get_ffmpeg_exe()
        info = probe_video(opts.input_path)
        cw, ch = cropped_dimensions(info.width, info.height, opts)
        log(f"Entrada: {info.summary()}")

        # --- Desentrelazado: ruta QTGMC (VapourSynth) o ruta FFmpeg ---
        use_qtgmc = bool(opts.deinterlace and opts.deinterlace_method == "qtgmc")
        vspipe = vs_backend.find_vspipe() if use_qtgmc else None
        if use_qtgmc and vspipe is None:
            log("QTGMC no disponible (falta VapourSynth en ./runtime o en el PATH). "
                "Se usa bwdif como respaldo. Instalacion: ver README, seccion QTGMC.")
            use_qtgmc = False

        script_path: Path | None = None
        if use_qtgmc:
            if opts.interpolate_fps:
                log("Aviso: la interpolacion de fps no aplica con QTGMC "
                    "(ya duplica la tasa temporal a 59.94p). Se omite.")
            if opts.deband:
                log("Aviso: debanding no disponible en la ruta QTGMC; se omite.")
            script_path = vs_backend.write_script(opts, Path(opts.input_path),
                                                  source_height=info.height)
            log(f"Grafo VapourSynth: {script_path.name} | QTGMC "
                f"({'59.94p doble tasa' if opts.qtgmc_double_fps else '29.97p'})")

        try:
            return self._run_job(opts, info, cw, ch, ffmpeg, vspipe, script_path,
                                 use_qtgmc, progress, log, cancel_event, pause)
        finally:
            if script_path is not None:
                script_path.unlink(missing_ok=True)

    # -- Orquestacion por segmentos -----------------------------------------

    def _run_job(self, opts, info, cw, ch, ffmpeg, vspipe, script_path, use_qtgmc,
                 progress, log, cancel_event, pause) -> Path:
        fps_out = output_fps(opts, info, qtgmc=use_qtgmc)
        if use_qtgmc and script_path is not None:
            exact = graph_fps(vspipe, script_path)
            if exact and abs(exact - fps_out) > 1e-6:
                log(f"FPS exacto del grafo VapourSynth: {exact:.5f} "
                    f"(se descarta el valor aproximado {fps_out:.5f})")
                fps_out = exact

        fmt = EXPORT_FORMATS.get(opts.export_format, EXPORT_FORMATS["mp4_h264"])
        output = Path(opts.output_path).with_suffix("." + fmt["ext"])
        output.parent.mkdir(parents=True, exist_ok=True)

        fingerprint = job_fingerprint(opts)
        job = self._prepare_job(opts, info, fps_out, cw, ch, fingerprint, output)
        total_frames = sum(s.frames for s in job.segments)
        done_frames = sum(s.frames for s in job.segments if s.done)

        if len(job.segments) > 1:
            log(f"Trabajo {job.job_id}: {len(job.segments)} segmentos de "
                f"{opts.segment_seconds}s | pendientes: {len(job.pending)}")
            if done_frames:
                log(f"Reanudando: {len(job.segments) - len(job.pending)} segmentos ya "
                    f"estaban listos ({done_frames} frames, "
                    f"{done_frames / max(total_frames, 1):.0%} del total).")

        if opts.start_seconds > 0 or opts.max_seconds > 0:
            rango = f"{opts.start_seconds:.0f}s" + \
                (f" +{opts.max_seconds:.0f}s" if opts.max_seconds > 0 else " hasta el final")
            log(f"Modo prueba: procesando desde el segundo {rango}")
        log(f"Salida: {output.name} | {opts.target_width}x{opts.target_height} | "
            f"{fps_out:.3f} fps | {fmt['video']} | {self._rate_description(opts, fmt)}")
        if not use_qtgmc:
            filtros = build_decode_filters(opts)
            if filtros:
                log(f"Filtros VHS: {', '.join(filtros)}")
        if opts.deflicker:
            log(f"Estabilizacion de brillo activa (ventana de {opts.deflicker_window} frames).")
        if opts.auto_exposure:
            log(f"Exposicion automatica activa: mediana objetivo "
                f"{opts.auto_exposure_target}, intensidad {opts.auto_exposure_strength}%, "
                f"suavizado {opts.auto_exposure_smoothing} frames.")

        upscaler = create_upscaler(opts.model_id, opts.device, log=log, fp16=opts.fp16)
        if upscaler.warning:
            log(upscaler.warning)
        if getattr(upscaler, "fp16", False):
            log("Modelo en fp16: calculo de media precision en la GPU (~1.3x mas rapido).")

        started = time.monotonic()
        for seg in job.segments:
            if seg.done:
                continue
            if cancel_event.is_set():
                raise ProcessingCancelled()
            if len(job.segments) > 1:
                log(f"--- Segmento {seg.index + 1}/{len(job.segments)} "
                    f"({seg.start:.0f}s +{seg.duration:.0f}s) ---")

            base = done_frames

            def seg_progress(d: int, _t: int, base=base) -> None:
                progress(min(base + d, total_frames), total_frames)

            produced = self._encode_segment(
                job, seg, opts, cw, ch, ffmpeg, vspipe, script_path, use_qtgmc,
                fps_out, fmt, upscaler, seg_progress, cancel_event, pause,
            )
            seg.frames = produced
            seg.done = True
            job.save()
            done_frames += produced
            total_frames = max(total_frames, done_frames)
            progress(done_frames, total_frames)

        log("Uniendo segmentos y agregando el audio original...")
        final = self._finalize(job, opts, fmt, output)

        detenido = pause.paused_seconds
        elapsed = max(time.monotonic() - started - detenido, 1e-6)
        extra = f" (+{detenido / 60:.1f} min en pausa)" if detenido > 1.0 else ""
        log(f"Completado: {done_frames} frames en {elapsed / 60:.1f} min"
            f"{extra} ({done_frames / elapsed:.1f} fps) -> {final}")
        cleanup_job(job)
        progress(total_frames, total_frames)
        return final

    @staticmethod
    def _prepare_job(opts, info, fps_out, cw, ch, fingerprint, output) -> Job:
        job_id = hashlib.sha256(
            (str(Path(opts.input_path).resolve()) + fingerprint).encode("utf-8")
        ).hexdigest()[:12]
        existing = Job.load(JOBS_DIR / job_id / "state.json")
        if existing and opts.resume and existing.fingerprint == fingerprint:
            # Un segmento marcado como listo cuyo archivo desaparecio hay que
            # rehacerlo: lo que hay en disco manda sobre el JSON.
            for seg in existing.segments:
                if seg.done and not existing.segment_path(seg).exists():
                    seg.done = False
            existing.output_path = str(output)
            return existing
        if existing:
            cleanup_job(existing)
        job = Job(
            job_id=job_id, input_path=str(opts.input_path), output_path=str(output),
            fingerprint=fingerprint, fps_out=fps_out,
            width=cw * 4, height=ch * 4,
            segments=plan_segments(opts, info, fps_out),
        )
        job.work_dir.mkdir(parents=True, exist_ok=True)
        job.save()
        return job

    @staticmethod
    def _rate_description(opts: ProcessOptions, fmt: dict) -> str:
        if not fmt["rate"]:
            return "all-intra (sin tasa variable)"
        if opts.quality_mode == "bitrate":
            mbps = max(0.5, float(opts.bitrate_mbps))
            return f"{mbps:g} Mbps (pico {mbps * 1.5:g} Mbps)"
        codec = fmt["video"]
        crf = CODEC_CRF.get(codec, CODEC_CRF["libx264"]).get(
            opts.quality, QUALITY_PRESETS.get(opts.quality, QUALITY_PRESETS["alta"])["crf"])
        return f"CRF {crf} (calidad constante)"

    # -- Un segmento ---------------------------------------------------------

    def _encode_segment(self, job, seg, opts, cw, ch, ffmpeg, vspipe, script_path,
                        use_qtgmc, fps_out, fmt, upscaler, progress, cancel_event,
                        pause) -> int:
        out_path = job.segment_path(seg)
        tmp_path = out_path.with_suffix(".mkv.part")
        tmp_path.unlink(missing_ok=True)

        decode_cmd, decode_is_rgb = self._decode_command(
            opts, seg, ffmpeg, vspipe, script_path, use_qtgmc, fps_out)
        encode_cmd = self._encode_command(
            opts, ffmpeg, fmt, cw * upscaler.scale, ch * upscaler.scale, fps_out, tmp_path)

        decoder = subprocess.Popen(decode_cmd, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, creationflags=_flags(),
                                   env=process_env())
        encoder = subprocess.Popen(encode_cmd, stdin=subprocess.PIPE,
                                   stderr=subprocess.PIPE, creationflags=_flags(),
                                   env=process_env())
        err_dec, err_enc = deque(maxlen=40), deque(maxlen=40)
        _stderr_pumper(decoder, err_dec, "decode")
        _stderr_pumper(encoder, err_enc, "encode")

        # El .part NUNCA se borra dentro de los 'except': en Windows FFmpeg
        # todavia lo tiene abierto y os.unlink falla con WinError 32, lo que
        # enmascaraba el error real (p. ej. una cancelacion limpia).
        done = 0
        fallo: BaseException | None = None
        try:
            done = self._pump(decoder, encoder, opts, cw, ch, decode_is_rgb,
                              upscaler, seg.frames, progress, cancel_event, pause)
        except OSError as exc:
            # Si el codificador muere al arrancar, Windows devuelve EINVAL al
            # escribir en su stdin en vez de EPIPE: en ambos casos lo util es
            # su stderr, no el errno.
            fallo = RuntimeError(
                "El codificador se cerro inesperadamente:\n" + "\n".join(err_enc))
            fallo.__cause__ = exc
        except BaseException as exc:  # noqa: BLE001 - se re-lanza mas abajo
            fallo = exc
        finally:
            if decoder.poll() is None:
                decoder.terminate()
            # Cerrar stdin marca el EOF: el codificador termina de escribir el
            # archivo por si solo (equivale al SIGPIPE en POSIX).
            try:
                if encoder.stdin and not encoder.stdin.closed:
                    encoder.stdin.close()
            except OSError:
                pass
            if (fallo is not None or cancel_event.is_set()) and encoder.poll() is None:
                encoder.terminate()
            _wait_or_kill(decoder, timeout=15)
            _wait_or_kill(encoder, timeout=180)

        if fallo is None:
            if cancel_event.is_set():
                fallo = ProcessingCancelled()
            elif encoder.returncode != 0:
                fallo = RuntimeError("Error al codificar:\n" + "\n".join(err_enc))
            elif done == 0:
                fallo = RuntimeError(
                    "El decodificador no produjo ningun frame. Detalle:\n"
                    + "\n".join(err_dec))
        if fallo is not None:
            _unlink_retry(tmp_path)
            raise fallo
        if decoder.returncode:
            log_file.warning("decoder rc=%s: %s", decoder.returncode, " | ".join(err_dec))
        tmp_path.replace(out_path)
        return done

    def _pump(self, decoder, encoder, opts, cw, ch, decode_is_rgb,
              upscaler, expected, progress, cancel_event, pause) -> int:
        """Las tres etapas a la vez: leer / escalar / escribir.

        Colas cortas a proposito: cada frame escalado ocupa ~17 MB a 1080p, asi
        que lo que interesa es que ninguna etapa espere, no acumular buffer.
        """
        frame_size = cw * ch * 3
        q_raw: queue.Queue = queue.Queue(maxsize=6)
        q_sr: queue.Queue = queue.Queue(maxsize=4)
        errors: list[BaseException] = []
        deflicker = Deflicker(opts.deflicker_window) if opts.deflicker else None
        exposure = build_auto_exposure(opts) if opts.auto_exposure else None

        def read_frames():
            try:
                while not cancel_event.is_set():
                    raw = _read_exact(decoder.stdout, frame_size)
                    if len(raw) < frame_size:
                        break
                    if decode_is_rgb:
                        # vspipe escribe RGB24 en PLANOS separados y en orden
                        # GBR (el GBRP de FFmpeg), no entrelazado. Leerlo como
                        # packed BGR daba la imagen repetida en rejilla 3x3.
                        planes = np.frombuffer(raw, dtype=np.uint8).reshape(3, ch, cw)
                        frame = np.stack((planes[1], planes[0], planes[2]), axis=-1)
                    else:
                        frame = np.frombuffer(raw, dtype=np.uint8).reshape(ch, cw, 3).copy()
                    q_raw.put(frame)
            except BaseException as exc:  # noqa: BLE001 - se re-lanza en el hilo principal
                errors.append(exc)
            finally:
                q_raw.put(None)

        def write_frames():
            try:
                while True:
                    sr = q_sr.get()
                    if sr is None:
                        return
                    # memoryview evita una copia extra (varios MB por cuadro)
                    encoder.stdin.write(memoryview(sr))
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)
                # Seguir vaciando la cola o el productor se queda bloqueado.
                while q_sr.get() is not None:
                    pass

        reader = threading.Thread(target=read_frames, daemon=True)
        writer = threading.Thread(target=write_frames, daemon=True)
        reader.start()
        writer.start()

        done = 0
        try:
            while True:
                # Un solo punto de pausa, y aqui: parar el hilo que escala
                # frena por si solo al decodificador y al codificador, porque
                # las colas son acotadas. Pausar en tres sitios distintos solo
                # abriria carreras.
                pause.wait(cancel_event)
                if cancel_event.is_set():
                    raise ProcessingCancelled()
                if errors:
                    raise errors[0]
                frame = q_raw.get()
                if frame is None:
                    break
                # Orden deliberado: primero el deflicker, que quita el temblor
                # de alta frecuencia del AGC, y luego la exposicion, que decide
                # el nivel de la escena sobre una senal ya estable.
                if deflicker is not None:
                    frame = deflicker.apply(frame)
                if exposure is not None:
                    frame = exposure.apply(frame)
                q_sr.put(upscaler.upscale(frame))
                done += 1
                progress(done, expected)
        finally:
            q_sr.put(None)
            writer.join(timeout=180)
            reader.join(timeout=15)
        if errors:
            raise errors[0]
        return done

    # -- Construccion de comandos -------------------------------------------

    @staticmethod
    def _decode_command(opts, seg, ffmpeg, vspipe, script_path, use_qtgmc, fps_out):
        if use_qtgmc:
            # vspipe: salida raw sin cabecera; --start/--end en frames de SALIDA
            # (con QTGMC doble tasa la salida va al doble de fps).
            cmd = [vspipe]
            start_frame = int(round(seg.start * fps_out))
            end_frame = int(round((seg.start + seg.duration) * fps_out))
            if start_frame > 0:
                cmd += ["--start", str(start_frame)]
            if seg.duration > 0:
                # --end es INCLUSIVO en vspipe: sin el -1, cada segmento
                # repetiria el primer frame del siguiente.
                cmd += ["--end", str(max(start_frame, end_frame - 1))]
            cmd += [str(script_path), "-"]
            return cmd, True

        filters = build_decode_filters(opts)
        cmd = [ffmpeg, "-hide_banner", "-loglevel", "error"]
        if seg.start > 0:
            cmd += ["-ss", f"{seg.start:.3f}"]
        cmd += ["-i", str(opts.input_path)]
        if seg.duration > 0:
            cmd += ["-t", f"{seg.duration:.3f}"]
        cmd += ["-map", "0:v:0"]
        if filters:
            cmd += ["-vf", ",".join(filters)]
        cmd += ["-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]
        return cmd, False

    @staticmethod
    def _encode_command(opts, ffmpeg, fmt, up_w, up_h, fps_out, out_path: Path):
        """Codifica SOLO video a un .mkv intermedio (el audio se pega al final)."""
        codec = fmt["video"]
        q = QUALITY_PRESETS.get(opts.quality, QUALITY_PRESETS["alta"])
        rate_args: list[str] = []
        if fmt["rate"]:
            if opts.quality_mode == "bitrate":
                mbps = max(0.5, float(opts.bitrate_mbps))
                rate_args = ["-b:v", f"{mbps:g}M", "-maxrate", f"{mbps * 1.5:g}M",
                             "-bufsize", f"{mbps * 2:g}M"]
            else:
                crf = CODEC_CRF.get(codec, CODEC_CRF["libx264"]).get(opts.quality, q["crf"])
                rate_args = ["-crf", crf]
                if codec == "libvpx-vp9":
                    rate_args += ["-b:v", "0", "-deadline", "good",
                                  "-cpu-used", "4", "-row-mt", "1"]
        cmd = [
            ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s:v", f"{up_w}x{up_h}", "-r", f"{fps_out:.6f}", "-i", "pipe:0",
            "-an",
            "-c:v", codec, *rate_args,
        ]
        if codec in ("libx264", "libx265"):
            cmd += ["-preset", q["preset"]]
        cmd += [
            "-vf", build_output_filter(opts),
            "-pix_fmt", fmt["pix_fmt"],
            # Etiquetado de color: sin esto el reproductor adivina. A 1080p casi
            # siempre acierta con BT.709 (que es lo que producimos), pero un
            # archivo bien etiquetado no deja nada al azar.
            "-color_primaries", "bt709", "-color_trc", "bt709",
            "-colorspace", "bt709", "-color_range", "tv",
            # El segmento se escribe a "<nombre>.mkv.part" y FFmpeg NO puede
            # deducir el contenedor de esa extension: hay que decirselo, o se
            # cierra nada mas arrancar y escribir en su stdin da EINVAL.
            "-f", "matroska",
            str(out_path),
        ]
        return cmd

    # -- Union de segmentos + audio -----------------------------------------

    @staticmethod
    def _finalize(job: Job, opts, fmt, output: Path) -> Path:
        ffmpeg = get_ffmpeg_exe()
        parts = [job.segment_path(s) for s in job.segments if s.done]
        if not parts:
            raise RuntimeError("No se genero ningun segmento.")

        if len(parts) == 1:
            video_src = parts[0]
        else:
            listing = job.work_dir / "segments.txt"
            listing.write_text(
                "".join(f"file '{p.as_posix()}'\n" for p in parts), encoding="utf-8"
            )
            video_src = job.work_dir / "joined.mkv"
            run_ffmpeg([
                ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                "-f", "concat", "-safe", "0", "-i", str(listing),
                "-c", "copy", str(video_src),
            ], "unir los segmentos")

        # Audio del ORIGEN, desplazado al mismo punto que el video.
        cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", str(video_src)]
        if opts.start_seconds > 0:
            cmd += ["-ss", f"{opts.start_seconds:.3f}"]
        cmd += ["-i", str(opts.input_path), "-map", "0:v:0", "-map", "1:a?"]
        total = sum(s.duration for s in job.segments)
        if total > 0:
            cmd += ["-t", f"{total:.3f}"]
        # Las etiquetas de color se repiten aqui: el remux con "copy" conserva
        # lo que va dentro del flujo H.264, pero no siempre lo replica en la
        # cabecera del contenedor, y es ahi donde mira el reproductor.
        cmd += ["-c:v", "copy",
                "-color_primaries", "bt709", "-color_trc", "bt709",
                "-colorspace", "bt709", "-color_range", "tv"]
        if fmt.get("tag"):
            cmd += ["-tag:v", fmt["tag"]]
        cmd += ["-c:a", fmt["audio"]]
        if opts.audio_mono:
            cmd += ["-ac", "1"]
        if fmt["audio"] == "aac":
            cmd += ["-b:a", "192k"]
        if fmt["ext"] == "mp4":
            cmd += ["-movflags", "+faststart"]
        cmd += ["-shortest", str(output)]
        run_ffmpeg(cmd, "agregar el audio")
        return output


def graph_fps(vspipe, script_path):
    """Envoltura sustituible en las pruebas: fps exacto del grafo VapourSynth."""
    from . import vs_backend

    if vspipe is None or script_path is None:
        return None
    return vs_backend.script_fps(vspipe, script_path)
