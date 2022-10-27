import logging
import sys
from typing import TextIO, Type

from tqdm import tqdm


class TqdmLoggingHandler(logging.StreamHandler):
    def __init__(
        self,
        stream: TextIO = sys.stdout,
        tqdm_class: Type[tqdm] = tqdm,
    ):
        super(TqdmLoggingHandler, self).__init__(stream)
        self.tqdm_class = tqdm_class

    def emit(self, record):
        try:
            msg = self.format(record)
            self.tqdm_class.write(msg, file=self.stream)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:  # noqa pylint: disable=bare-except
            self.handleError(record)


class TallowLogger(logging.getLoggerClass()):
    pass
