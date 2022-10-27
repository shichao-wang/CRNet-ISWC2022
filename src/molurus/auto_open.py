import gzip
import typing
from typing import BinaryIO, Literal, Optional, TextIO

__all__ = ["auto_open", "ModeType", "AVAILABLE_MODES"]

ModeType = Literal["rt", "wt", "rb", "wb"]
AVAILABLE_MODES = typing.get_args(ModeType)


@typing.overload
def auto_open(
    filename: str,
    mode: ModeType = Literal["rt", "wt"],
    encoding: Optional[str] = None,
    errors: Optional[str] = None,
    newline: Optional[str] = None,
) -> TextIO:
    pass


@typing.overload
def auto_open(
    filename: str,
    mode: ModeType = Literal["rb", "wb"],
    encoding: Optional[str] = None,
    errors: Optional[str] = None,
    newline: Optional[str] = None,
) -> BinaryIO:
    pass


def auto_open(
    filename: str,
    mode: ModeType = "rt",
    encoding: Optional[str] = None,
    errors: Optional[str] = None,
    newline: Optional[str] = None,
):
    """
    Automatically open the file by it's suffix.
    now support plain file, .gz file.
    :param filename:
    :param mode:
    :param encoding:
    :param errors:
    :param newline:
    :return:
    """
    if filename.endswith(".gz"):  # gzip file
        return gzip.open(
            filename=filename,
            mode=mode,
            encoding=encoding,
            errors=errors,
            newline=newline,
        )
    else:  # plain file
        plain_mode = mode.replace("t", "")
        return open(
            file=filename,
            mode=plain_mode,
            encoding=encoding,
            errors=errors,
            newline=newline,
        )
