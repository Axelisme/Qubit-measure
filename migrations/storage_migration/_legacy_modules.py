"""Evidence-based legacy module shapes, separate from current Spec validation."""

from copy import deepcopy
from dataclasses import dataclass, field

from zcu_tools.datafile import JsonObject


@dataclass(frozen=True)
class LegacyModuleCandidate:
    """Current-shape candidate with original paths for flattened unknown fields.

    values is the candidate for cfg_model, not a validated Spec. unrecognized
    contains original fields with no structural counterpart. origins maps each
    flattened candidate field to its literal original source path.
    """

    values: JsonObject
    unrecognized: JsonObject = field(default_factory=dict)
    origins: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def original_unknown(self, unknown: JsonObject) -> JsonObject:
        """Return all unexpressed fields in the original legacy nesting."""
        result = deepcopy(self.unrecognized)
        for key, value in unknown.items():
            path = self.origins.get(key, (key,))
            if len(path) == 1:
                result[path[0]] = value
            else:
                branch = result.setdefault(path[0], {})
                if not isinstance(branch, dict):
                    raise ValueError("Conflicting original module field paths")
                branch[path[1]] = value
        return result


def _readout_pulse(pulse: JsonObject, readout: JsonObject) -> LegacyModuleCandidate:
    values: JsonObject = {}
    origins: dict[str, tuple[str, ...]] = {}
    pulse_raw: JsonObject = {}
    for key, value in pulse.items():
        if key in {"type", "desc", "gen_ch"}:
            pulse_raw[key] = value
        elif key == "ro_ch":
            if value != readout.get("ro_ch"):
                pulse_raw[key] = value
        else:
            destination = "gen_ch" if key == "ch" else key
            values[destination] = value
            origins[destination] = ("pulse_cfg", key)
    return LegacyModuleCandidate(values, pulse_raw, origins)


def _base_readout(raw: JsonObject) -> LegacyModuleCandidate:
    pulse = raw.get("pulse_cfg")
    readout = raw.get("ro_cfg")
    if not isinstance(pulse, dict) or not isinstance(readout, dict):
        raise ValueError("Legacy base readout requires pulse_cfg and ro_cfg mappings")
    if "trig_offset" not in readout:
        raise ValueError("Legacy base readout is missing ro_cfg.trig_offset")
    if "ro_freq" in readout and readout["ro_freq"] != pulse.get("freq"):
        raise ValueError(
            f"Legacy base readout has pulse_cfg.freq={pulse.get('freq')!r} and "
            f"ro_cfg.ro_freq={readout['ro_freq']!r}; the current Spec has one freq"
        )
    generator = _readout_pulse(pulse, readout)
    values = {**generator.values, "type": "readout/pulse"}
    origins = generator.origins
    remainder: JsonObject = {
        key: value
        for key, value in raw.items()
        if key not in {"type", "desc", "pulse_cfg", "ro_cfg"}
    }
    if "desc" in raw:
        values["desc"] = raw["desc"]
    pulse_raw = generator.unrecognized
    readout_raw: JsonObject = {}
    for key, value in readout.items():
        if key in {"ro_ch", "ro_length", "trig_offset"}:
            if key in values:
                pulse_raw[key] = values[key]
            values[key] = value
            origins[key] = ("ro_cfg", key)
        elif key != "ro_freq":
            readout_raw[key] = value
    if pulse_raw:
        remainder["pulse_cfg"] = pulse_raw
    if readout_raw:
        remainder["ro_cfg"] = readout_raw
    return LegacyModuleCandidate(values, remainder, origins)


def legacy_module_candidate(raw: JsonObject) -> LegacyModuleCandidate:
    """Translate only proven historical shapes, retaining unknown source fields.

    Missing type is pulse only with ch/nqz/freq/gain/waveform present, as required
    before Pulse.auto_fill's defaults in 285e53290. Names never select a family.
    Historical base and bath discriminators follow 4a6d288ff's reader. Bath's
    three children are known Pulse positions. Invalid base structure or unequal
    generator/ADC frequencies raise ValueError; no semantic guess is made.
    Current-shape values pass through for the cfg_model owner to validate.
    """
    values = deepcopy(raw)
    if (
        "type" not in values
        and {"ch", "nqz", "freq", "gain", "waveform"} <= values.keys()
    ):
        values["type"] = "pulse"
    if values.get("type") == "base":
        return _base_readout(values)
    if values.get("type") == "bath":
        values["type"] = "reset/bath"
        for key in ("cavity_tone_cfg", "qubit_tone_cfg", "pi2_cfg"):
            child = values.get(key)
            if isinstance(child, dict):
                child.setdefault("type", "pulse")
    return LegacyModuleCandidate(values)
