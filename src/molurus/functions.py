import importlib
import inspect
from typing import Callable, Mapping, TypedDict, TypeVar

ArgType = TypeVar("ArgType")
ReturnType = TypeVar("ReturnType")
FunctionType = Callable[[ArgType], ReturnType]

__all__ = ["build_fn_kwargs", "smart_instantiate"]


def _is_var_kwargs(p: inspect.Parameter):
    return p.kind == inspect.Parameter.VAR_KEYWORD


def build_fn_kwargs(
    function: FunctionType, *dict_args: Mapping, **kwargs
) -> Mapping:
    """
    Apply kwargs to function.
    :param function:
    :param kwargs:
    :return:
    """
    full_kwargs = {}
    for dict_arg in dict_args:
        full_kwargs.update(dict_arg)
    full_kwargs.update(kwargs)

    signature = inspect.signature(function)
    parameters = signature.parameters
    if len(parameters) == 0:
        return {}

    if any(p for p in parameters.values() if _is_var_kwargs(p)):
        return full_kwargs

    needed_args = list(parameters)
    if needed_args[0] == "self":
        needed_args = needed_args[1:]

    output = {k: v for k, v in full_kwargs.items() if k in needed_args}
    for name, p in parameters.items():
        if p.default != inspect._empty:
            output.setdefault(name, p.default)

    return output


class SupportsInstantiate(TypedDict, total=False):
    _class: str


def instantiatable(dict: Mapping):
    if isinstance(dict, Mapping):
        return "_class" in dict
    else:
        return False


def smart_instantiate(dict_obj: SupportsInstantiate, **kwargs):
    assert instantiatable(dict_obj)
    class_string = dict_obj.pop("_class")
    modulename, classname = class_string.rsplit(".", 1)
    module = importlib.import_module(modulename)
    clazz = getattr(module, classname)

    init_kwargs = build_fn_kwargs(clazz, kwargs)
    for k, v in dict_obj.items():
        if instantiatable(v):
            init_kwargs[k] = smart_instantiate(v, **kwargs)
        else:
            init_kwargs[k] = v

    return clazz(**init_kwargs)


def smart_call(func_or_clazz, *dict_args: Mapping, **kwargs):
    call_kwargs = build_fn_kwargs(func_or_clazz, *dict_args, **kwargs)
    return func_or_clazz(**call_kwargs)
