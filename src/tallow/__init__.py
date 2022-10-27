import logging

from .common.importing import try_import_and_raise
from .common.logging import TallowLogger, TqdmLoggingHandler

logging.setLoggerClass(TallowLogger)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO,
    handlers=[TqdmLoggingHandler()],
)


try_import_and_raise("torch")

from .common import seeds

# isort: list
__all__ = ["seeds"]
