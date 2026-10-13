"""Directed rejection data for native cfg backfill owned by Form."""

from dataclasses import dataclass

from pydantic import JsonValue

from zcu_tools.gui.form import FormErrors, FormPath


@dataclass(frozen=True)
class CfgInputDifference:
    """One native dump mismatch after lowering a Load candidate.

    path is a cfg field path with integer sequence indexes. loaded and would_run
    are the respective JSON-mode native model-dump values; None can also denote
    an absent member, explained by reason. reason describes value/membership
    mismatch without applying a tolerance or guessing an inverse transform.
    """

    path: FormPath
    loaded: JsonValue
    would_run: JsonValue
    reason: str


@dataclass(frozen=True)
class CfgInputRejected:
    """A rejected native Load candidate; editable input is never changed.

    errors explains invalid input or roundtrip failures at consuming cfg paths.
    differences lists exact native dump mismatches, or is empty when no valid
    candidate could be compared. The host forwards these diagnostics to callers.
    """

    errors: FormErrors
    differences: tuple[CfgInputDifference, ...] = ()


def dump_differences(
    loaded: JsonValue, would_run: JsonValue, path: FormPath = ()
) -> tuple[CfgInputDifference, ...]:
    """Compare two native JSON dumps exactly and return deterministic field paths.

    Dicts compare by keys, sequences by indexes, and leaves by Python equality.
    Missing members carry a membership reason; no numeric tolerance is applied.
    """
    differences: list[CfgInputDifference] = []
    if isinstance(loaded, dict) and isinstance(would_run, dict):
        for key, value in loaded.items():
            if key not in would_run:
                differences.append(
                    CfgInputDifference((*path, key), value, None, "missing")
                )
            else:
                differences.extend(
                    dump_differences(value, would_run[key], (*path, key))
                )
        for key in would_run.keys() - loaded.keys():
            differences.append(
                CfgInputDifference((*path, key), None, would_run[key], "extra")
            )
    elif isinstance(loaded, list) and isinstance(would_run, list):
        for index in range(max(len(loaded), len(would_run))):
            if index >= len(would_run):
                differences.append(
                    CfgInputDifference((*path, index), loaded[index], None, "missing")
                )
            elif index >= len(loaded):
                differences.append(
                    CfgInputDifference((*path, index), None, would_run[index], "extra")
                )
            else:
                differences.extend(
                    dump_differences(loaded[index], would_run[index], (*path, index))
                )
    elif loaded != would_run:
        differences.append(
            CfgInputDifference(path, loaded, would_run, "value mismatch")
        )
    return tuple(differences)
