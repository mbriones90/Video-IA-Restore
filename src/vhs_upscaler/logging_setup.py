"""Registro a fichero rotativo, ademas del log visible en la interfaz.

Un proceso de 3 horas que falla en el minuto 150 no cabe en el `deque` de 30
lineas que guarda el pipeline ni en el cuadro de texto de la GUI. Aqui queda
todo en `output/logs/vhs_upscaler.log`, con rotacion para no crecer sin fin.

Uso tipico:

    from .logging_setup import get_logger, log_to

    log = get_logger(__name__)
    log.info("mensaje")

El pipeline recibe un callable `log=` (para la GUI). `log_to(callable)`
devuelve un callable equivalente que ademas escribe en el fichero.
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path

from .paths import ROOT_DIR

LOG_DIR = ROOT_DIR / "output" / "logs"
LOG_PATH = LOG_DIR / "vhs_upscaler.log"
_configured = False


def setup_logging(level: int = logging.INFO, console: bool = False) -> Path | None:
    """Configura el logger raiz una sola vez. Devuelve la ruta del fichero."""
    global _configured
    if _configured:
        return LOG_PATH
    _configured = True

    root = logging.getLogger("vhs_upscaler")
    root.setLevel(level)
    root.propagate = False
    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-7s [%(name)s] %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        # 5 ficheros de 2 MB: suficiente para varias cintas completas.
        handler = logging.handlers.RotatingFileHandler(
            LOG_PATH, maxBytes=2 * 1024 * 1024, backupCount=5, encoding="utf-8"
        )
        handler.setFormatter(fmt)
        root.addHandler(handler)
    except OSError:
        # Carpeta de solo lectura o disco lleno: la app debe seguir funcionando.
        return None
    if console:
        stream = logging.StreamHandler(sys.stderr)
        stream.setFormatter(fmt)
        root.addHandler(stream)
    return LOG_PATH


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    short = name.rsplit(".", 1)[-1]
    return logging.getLogger(f"vhs_upscaler.{short}")


def log_to(sink=None, name: str = "pipeline"):
    """Envuelve un callable de log para que ademas quede en el fichero.

    `sink` es el destino visible (print en CLI, una Signal de Qt en la GUI).
    Si es None, solo se escribe al fichero.
    """
    logger = get_logger(name)

    def _log(message: str) -> None:
        try:
            logger.info(message)
        except Exception:
            pass
        if sink is not None:
            sink(message)

    return _log
