import importlib


def try_import_and_raise(module_name: str):
    try:
        importlib.import_module(name=module_name)
    except ImportError:
        raise ImportError(f"Please install {module_name} first.")
