from .auto import make_cfg
from .configuration import DefaultCfg
from .datasaver import create_datafolder, save_cfg, save_data, make_comment
from .tools import make_sweep


def reload_zcutools():
    import importlib
    from types import ModuleType

    excluded = ["qick", "numpy", "matplotlib.pyplot", "importlib", "scipy"]
    visited = set()

    def reload(module, depth=0, level=0):
        if level > depth:
            return

        nonlocal visited
        if module in visited:
            return
        visited.add(module)

        print(" " * level + module.__name__)
        for attr_name in sorted(dir(module)):
            attr = getattr(module, attr_name)
            if isinstance(attr, ModuleType) and attr.__name__ not in excluded:
                reload(attr, depth, level + 1)

        importlib.reload(module)

    import zcu_tools.analysis as zf
    import zcu_tools.auto as za
    import zcu_tools.datasaver as zd
    import zcu_tools.program as zp
    import zcu_tools.schedule as zs
    import zcu_tools.tools as zt

    print("reloaded:")

    reload(zt)
    reload(zd)
    reload(za)
    reload(zp, 4)
    reload(zs, 4)
    reload(zf, 3)


__all__ = [
    "reload_zcutools",
    "DefaultCfg",
    "make_cfg",
    "save_cfg",
    "save_data",
    "make_comment",
    "create_datafolder",
    "make_sweep",
]
