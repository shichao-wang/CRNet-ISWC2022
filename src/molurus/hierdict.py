import argparse
import ast
import json
import warnings
from collections import UserDict
from typing import Any, Dict, Iterable, Iterator, List, Mapping, TextIO

import yaml


def load(stream: TextIO) -> "HierDict":
    data = yaml.safe_load(stream)
    return HierDict(data)


def dump(hierdict: "HierDict", stream: TextIO):
    return yaml.dump(hierdict.data, stream)


class HierDict(UserDict):

    # basic operators
    def __getitem__(self, key: str) -> Any:
        ks = key.split(".")
        d = self.data
        for k in ks:
            d = d[k]
        return d

    def __setitem__(self, key: str, item: Any) -> None:
        *pk, fk = key.split(".")
        d = self.data
        for k in pk:
            d.setdefault(k, {})
            d = d[k]
        d[fk] = item

    def __delitem__(self, key: str) -> None:
        _recursive_delitem(self.data, key.split("."))

    def __iter__(self) -> Iterator[str]:
        return _recursive_iter(self.data)

    def __len__(self) -> int:
        return len(_recursive_iter(self.data))

    def __contains__(self, key: object) -> bool:
        return key in iter(self)

    def __hash__(self) -> int:
        data = json.dumps(self.data, sort_keys=True)
        return hash(data)

    # operators
    def replace(self, other: Mapping):
        for k, v in other.items():
            if k in self:
                print("replace ", k)
                self[k] = v
            else:
                warnings.warn("replace missing key (%s) ignore it" % k)


def _recursive_iter(dict: Dict) -> Iterator[str]:
    for k, v in dict.items():
        if isinstance(v, Mapping):
            for sk in _recursive_iter(v):
                yield k + "." + sk
        else:
            yield k


def _recursive_delitem(dict: Dict, keys: Iterable[str]):
    if len(keys) == 1:
        del dict[keys[0]]
        return

    ck, *ks = keys
    _recursive_delitem(dict[ck], ks)
    if len(dict[ck]) == 0:
        del dict[ck]


class ArgumentParser(argparse.ArgumentParser):
    def __init__(self, prog: str = None) -> None:
        super().__init__(prog)
        self.add_argument("cfg", type=argparse.FileType("r"))
        self.add_argument("overrides", nargs=argparse.REMAINDER)


def _parse_overrides(overrides_string: List[str]) -> Mapping:
    def set_override(__key: str, __value):
        if len(__value) == 1:
            __value = __value[0]
        overrides[__key] = __value

    overrides = {}
    key, value = None, []
    for arg in overrides_string:
        if str.startswith(arg, "--"):
            if key is not None:
                set_override(key, value)
            key, value = arg[2:], []
        else:
            try:
                v = ast.literal_eval(arg)
            except Exception:
                v = arg
            value.append(v)

    if value:
        set_override(key, value)

    return overrides


def parse_args(stream: TextIO, overrides: List[str] = None):
    cfg = load(stream)
    if overrides:
        overrides = _parse_overrides(overrides)
        cfg.replace(overrides)
    return cfg
