"""SMART module defaults shared by library creation and deferred cfg seeds."""

from collections.abc import Callable
from copy import deepcopy

from zcu_tools.gui.cfg import CfgSectionValue, ReferenceValue
from zcu_tools.gui.session.parameters import ModuleSlotError
from zcu_tools.gui.session.types import SessionEnv
from zcu_tools.resources.entry import RoleResolutionError

from .cfg_schemas import module_cfg_to_value


def resolve_library_reference(
    path: str,
    env: SessionEnv,
    *,
    kind: str,
    fallback: Callable[[SessionEnv], ReferenceValue] | None,
) -> ReferenceValue | None:
    """Build a fresh SMART reference from one fixed parameter/library capture.

    ``path`` is an authored component/role module slot or concrete library root.
    ``env`` supplies the fixed capture; no live selection or hardware read occurs.
    Module materialization uses that capture's library and asset capabilities.
    ``kind`` must be "module"; other kinds and empty paths raise ValueError
    before lookup. ``fallback`` is a pure factory called only for bound unset,
    null or dangling slots, missing direct roots, or a missing concrete capture.
    None disables such a reference. Its result is detached before returning.

    Unbound aliases, including $ paths without a capture, instead keep a linked
    empty section without calling fallback. Available modules retain the authored
    path and carry their existing materialized cfg values. Source failures,
    invalid paths, malformed modules and asset/materialization errors propagate;
    only ModuleSlotError triggers fallback. Subsequent source changes do not
    mutate returned inputs; callers may resolve again against a new capture.
    """
    if kind != "module":
        raise ValueError(f"Unsupported reference kind {kind!r}; expected 'module'")
    if not path.strip():
        raise ValueError("Library reference path must not be empty")
    parameters = env.parameters
    if parameters is None and path.startswith("$"):
        return ReferenceValue(chosen_key=path, value=CfgSectionValue())
    if parameters is not None:
        try:
            key = parameters.library_key(path)
        except RoleResolutionError:
            # Keep the authored link so refresh can later supply its source.
            return ReferenceValue(chosen_key=path, value=CfgSectionValue())
        except ModuleSlotError:
            # Only the path owner's explicit unset/dangling slot error recovers.
            key = None
        if key is not None:
            try:
                parameters.library.module_spec(key)
            except KeyError:
                # A valid direct root can name a not-yet-created entry.
                pass
            else:
                _, value = module_cfg_to_value(parameters.library.get_module(key))
                return ReferenceValue(chosen_key=path, value=value)
    return deepcopy(fallback(env)) if fallback is not None else None
