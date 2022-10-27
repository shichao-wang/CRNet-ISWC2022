import argparse
import dataclasses
import enum
import typing
from dataclasses import MISSING, Field, fields, is_dataclass
from typing import Dict, Mapping, Sequence, Type, TypeVar

from . import functions

DataClass = TypeVar("DataClass")


@typing.overload
def parse(dataclass_type: Type[DataClass]) -> DataClass:
    pass


@typing.overload
def parse(dataclass_types: Mapping[str, Type[DataClass]]) -> DataClass:
    pass


def parse(dataclass_type_or_types):
    if isinstance(dataclass_type_or_types, Mapping):
        parser = argparse.ArgumentParser()
        sub_parsers = parser.add_subparsers(dest="command")
        for command, dataclass_type in dataclass_type_or_types.items():
            parser = sub_parsers.add_parser(command, add_help=False)
            make_argument_parser(dataclass_type, parser)
        args, _ = parser.parse_known_args()
        dataclass_type = dataclass_type_or_types[args.command]
        delattr(args, "command")
        return dataclass_type(**vars(args))
    else:
        return parse_into_dataclass(dataclass_type_or_types)


def parse_into_dataclass(dataclass_type: Type[DataClass]) -> DataClass:
    if not is_dataclass(dataclass_type):
        raise ValueError()
    parser = make_argument_parser(dataclass_type)
    args = parser.parse_args()
    print(args)
    kwargs = functions.build_fn_kwargs(dataclass_type, **vars(args))
    return dataclass_type(**kwargs)


def set_default_and_required(kwargs: Dict, dataclass_field: Field):
    new_kwargs = kwargs.copy()
    # del kwargs

    new_kwargs.setdefault("help", "")
    if new_kwargs.get("required", False):  # default is not required
        del new_kwargs["default"]
    elif dataclass_field.default_factory is not MISSING:
        new_kwargs["default"] = dataclass_field.default_factory()
        new_kwargs["help"] += "DEFAULT is %s" % str(new_kwargs["default"])
    elif dataclass_field.default is not MISSING:
        new_kwargs["default"] = dataclass_field.default
        new_kwargs["help"] += "DEFAULT is %s" % str(new_kwargs["default"])
    else:
        new_kwargs["required"] = True
    return new_kwargs


def field_general_kwargs(field: dataclasses.Field):
    kwargs = field.metadata.copy()
    kwargs.setdefault("help", "")
    kwargs.setdefault("type", field.type)

    if kwargs.get("required", False):  # default is not required
        del kwargs["default"]
    elif field.default_factory is not MISSING:
        kwargs["default"] = field.default_factory()
    elif field.default is not MISSING:
        kwargs["default"] = field.default
    else:
        kwargs["required"] = True

    if "default" in kwargs:
        kwargs["help"] += "DEFAULT is %s" % str(kwargs["default"])

    return kwargs


def add_boolean_argument(
    parser: argparse.ArgumentParser, field: dataclasses.Field
):
    kwargs = field_general_kwargs(field)
    del kwargs["type"]
    # kwargs["action"] = "store_true"  # https://stackoverflow.com/a/26992149

    # required
    # required = kwargs.pop("required", False)
    # default
    default = kwargs.pop("default", None)

    arg_name = replace_underline(field.name)

    parser.add_argument(
        "--%s" % arg_name, action="store_true", dest=field.name, **kwargs
    )
    parser.add_argument(
        "--no-%s" % arg_name, action="store_false", dest=field.name, **kwargs
    )
    if default is not None:
        parser.set_defaults(**{arg_name: default})


def add_enum_argument(
    parser: argparse.ArgumentParser, field: dataclasses.Field
):
    kwargs = field_general_kwargs(field)
    kwargs["action"] = EnumAction

    parser.add_argument("--%s" % replace_underline(field.name), **kwargs)


def add_literal_argument(
    parser: argparse.ArgumentParser, field: dataclasses.Field
):
    kwargs = field_general_kwargs(field)
    kwargs["type"] = str
    kwargs["choices"] = typing.get_args(field.type)
    parser.add_argument("--%s" % replace_underline(field.name), **kwargs)


def add_sequence_argument(
    parser: argparse.ArgumentParser, field: dataclasses.Field
):
    kwargs = field_general_kwargs(field)
    kwargs["type"] = typing.get_args(field.type)[0]
    parser.add_argument(
        "--%s" % replace_underline(field.name),
        nargs=argparse.ZERO_OR_MORE,
        **kwargs
    )


def add_primitive_argument(
    parser: argparse.ArgumentParser, field: dataclasses.Field
):
    kwargs = field_general_kwargs(field)
    parser.add_argument("--%s" % replace_underline(field.name), **kwargs)


def make_argument_parser(
    dataclass_type: Type[DataClass], parser: argparse.ArgumentParser = None
) -> argparse.ArgumentParser:
    parser = parser or argparse.ArgumentParser(
        getattr(dataclass_type, "__program__", dataclass_type.__name__)
    )
    field: Field
    for field in fields(dataclass_type):
        if not field.init:
            continue

        origin_type = typing.get_origin(field.type)
        if origin_type is None:  # primitive type
            if issubclass(field.type, enum.Enum):
                add_enum_argument(parser, field)
            elif field.type is bool:
                add_boolean_argument(parser, field)
            else:
                add_primitive_argument(parser, field)

        elif origin_type == typing.Literal:
            add_literal_argument(parser, field)
        elif origin_type == list or origin_type == tuple:
            add_sequence_argument(parser, field)
        elif (
            origin_type == typing.Union
            and len(typing.get_args(field.type)) == 2
        ):
            raise NotImplementedError("Do not support Optional argument.")
        else:
            msg = "Unsupported argument type `%s(origin=%s)`" % (
                field.type,
                origin_type,
            )
            raise NotImplementedError(msg)

    return parser


def replace_underline(string: str):
    return string.replace("_", "-")


class EnumAction(argparse.Action):
    def __init__(
        self,
        option_strings: Sequence[str],
        dest: str,
        default: enum.Enum,
        type: enum.Enum,
        required: bool = False,
        help: str = None,
        metavar: str = None,
    ) -> None:
        super().__init__(
            option_strings,
            dest,
            default=default,
            type=str,
            choices=type._member_names_,
            required=required,
            help=help,
            metavar=metavar,
        )
        self._enum_type = type

    def __call__(self, parser, namespace, values, option_string=None):
        values = self._enum_type[values] if isinstance(values, str) else values
        setattr(namespace, self.dest, values)
