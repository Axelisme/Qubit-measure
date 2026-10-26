"""Mux adapter naming from saved configuration, never live group bindings."""

from pydantic import JsonValue
from zcu_tools.resources.entry import SaveContext, default_save_stem


def _participant_stem(context: SaveContext, tones: JsonValue, *, field: str) -> str:
    cfg = context.cfg
    participants = None if cfg is None else cfg.get("participants")
    if not isinstance(tones, list) or not isinstance(participants, list):
        return default_save_stem(context)
    names: list[str] = []
    for index in participants:
        if not isinstance(index, int) or not 0 <= index < len(tones):
            return default_save_stem(context)
        tone = tones[index]
        name = tone.get(field) if isinstance(tone, dict) else None
        if not isinstance(name, str) or not name:
            return default_save_stem(context)
        names.append(name)
    if not names:
        return default_save_stem(context)
    return f"{context.name.replace('/', '.')}_{'+'.join(names)}_{context.run_id[-6:]}"


def mux_save_stem(context: SaveContext) -> str:
    """List participating qubits from cfg.tones[].name in participants order.

    Join names with + and append the run_id tail after the dotted adapter path.
    Missing cfg or names uses default role naming; never read live context.
    Onetone readout-only scans use mux_readout_save_stem instead.
    """
    tones = None if context.cfg is None else context.cfg.get("tones")
    return _participant_stem(context, tones, field="name")


def mux_readout_save_stem(context: SaveContext) -> str:
    """List onetone participants by cfg.readout.tones[].label, joined with +.

    Readout-only scans may have no qubits, so use resonator labels rather than
    inventing qubit names. Missing cfg or labels uses default role naming.
    Other mux adapters use mux_save_stem for qubit names. Never read live context.
    """
    readout = None if context.cfg is None else context.cfg.get("readout")
    tones = readout.get("tones") if isinstance(readout, dict) else None
    return _participant_stem(context, tones, field="label")
