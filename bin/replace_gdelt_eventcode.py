import argparse
from io import StringIO
from typing import Mapping, TextIO

from tkgl.datasets import load_quadruples


def replace_codes(file: str, code_dict: Mapping[str, str]):
    quads = load_quadruples(file, bidirectional=False)
    buf = StringIO()
    for quad in quads:
        quad_string = "\t".join(
            [
                quad["subj"],
                code_dict[str.strip(quad["rel"])],
                quad["obj"],
                quad["mmt"],
            ]
        )
        buf.write(quad_string + "\n")

    with open(file, "w") as fp:
        fp.write(buf.getvalue())


def load_gdelt_code(stream: TextIO):
    cdict = {}
    for line in stream:
        line_splits = line.strip().split("\t")
        code, name = line_splits
        cdict[code] = name
    return cdict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("codefile")
    parser.add_argument("filelist", nargs=argparse.ONE_OR_MORE)
    args = parser.parse_args()

    code_dict = load_gdelt_code(open(args.codefile))

    for gdelt_file in args.filelist:
        replace_codes(gdelt_file, code_dict)


if __name__ == "__main__":
    main()
