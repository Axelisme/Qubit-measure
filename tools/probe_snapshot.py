"""Save caller-provided probe outputs and compare saved snapshots.

File I/O stays here rather than in experiment stages (ADR-0072).
"""

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from pathlib import Path
from zipfile import BadZipFile

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel


def _leaves(value: object, prefix: str, arrays: dict[str, NDArray[np.generic]]) -> None:
    if is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            _leaves(getattr(value, field.name), f"{prefix}.{field.name}", arrays)
    elif isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("snapshot mapping keys must be strings")
            _leaves(item, f"{prefix}.{key}", arrays)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _leaves(item, f"{prefix}[{index}]", arrays)
    elif value is not None:
        array = np.asarray(value)
        if array.dtype.kind not in "biufcSU":
            raise TypeError(f"unsupported snapshot leaf at {prefix}: {array.dtype}")
        if prefix in arrays:
            raise ValueError(f"duplicate snapshot array name: {prefix}")
        arrays[prefix] = array


def _jsonable(value: object) -> object:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="python")
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: getattr(value, field.name) for field in fields(value)}
    if isinstance(value, (np.ndarray, np.generic)):
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise TypeError("object arrays cannot be fingerprinted")
        return {
            "shape": array.shape,
            "dtype": np.lib.format.dtype_to_descr(array.dtype),
            "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
        }
    raise TypeError(f"unsupported snapshot input: {type(value).__name__}")


def _local_name(name: str) -> None:
    if not name or name in (".", "..") or "/" in name or "\\" in name:
        raise ValueError(f"snapshot names must be single path components: {name!r}")


def write_snapshot(
    directory: Path,
    label: str,
    value: object,
    pngs: Mapping[str, bytes] | None = None,
    *,
    inputs: object,
) -> None:
    """Write one new directory/label with arrays.npz, inputs.json and named PNGs.

    value contains arrays, numeric/text scalars, dataclass instances, string-keyed
    mappings or nested lists/tuples of these. None contributes no arrays. Array
    keys use label.field and [index] paths, or label.value for a root leaf.
    Object arrays are rejected. pngs maps
    single-component .png filenames to encoded bytes, or is None for no figures.
    inputs is required, including an explicit None when there are no inputs. It
    accepts JSON values, dataclasses, Pydantic models and NumPy arrays/scalars;
    arrays contribute shape, dtype and a SHA-256 of all C-order content bytes.
    Caller supplies every cfg, seed, option and environment setting to compare.
    Raise TypeError/ValueError for unsupported values or names, FileExistsError
    for an existing label and OSError for I/O failures. A failed write can leave
    a partial directory; discard it rather than treating it as a snapshot.
    """
    _local_name(label)
    arrays: dict[str, NDArray[np.generic]] = {}
    prefix = (
        label
        if is_dataclass(value) or isinstance(value, (Mapping, list, tuple))
        else f"{label}.value"
    )
    _leaves(value, prefix, arrays)
    fingerprint = json.dumps(inputs, default=_jsonable, sort_keys=True, indent=2)
    for name in pngs or {}:
        _local_name(name)
        if not name.endswith(".png"):
            raise ValueError(f"snapshot figure must have a .png suffix: {name!r}")
    target = directory / label
    target.mkdir(parents=True)
    np.savez(target / "arrays.npz", allow_pickle=False, **arrays)
    (target / "inputs.json").write_text(fingerprint, encoding="utf-8")
    for name, data in (pngs or {}).items():
        (target / name).write_bytes(data)


def _pixels(path: Path) -> NDArray[np.uint8]:
    from PIL import Image

    with Image.open(path) as image:
        return np.asarray(image.convert("RGBA"))


def _array_status(
    old: NDArray[np.generic],
    new: NDArray[np.generic],
    rtol: float | None,
    atol: float | None,
) -> str:
    if old.shape != new.shape or old.dtype != new.dtype:
        return "DIFFERENT"
    components = (
        ((old.real, new.real), (old.imag, new.imag))
        if old.dtype.kind == "c"
        else ((old, new),)
    )
    if all(
        np.array_equal(a, b, equal_nan=old.dtype.kind in "fc") for a, b in components
    ):
        return "SAME"
    if (
        old.dtype.kind in "fc"
        and (rtol is not None or atol is not None)
        and all(
            np.allclose(
                a,
                b,
                rtol=0 if rtol is None else rtol,
                atol=0 if atol is None else atol,
                equal_nan=True,
            )
            for a, b in components
        )
    ):
        return "WITHIN TOL"
    return "DIFFERENT"


def _arrays(before: Path, after: Path, rtol: float | None, atol: float | None) -> str:
    with (
        np.load(before / "arrays.npz", allow_pickle=False) as a,
        np.load(after / "arrays.npz", allow_pickle=False) as b,
    ):
        statuses = {
            _array_status(a[key], b[key], rtol, atol)
            if key in a.files and key in b.files
            else "DIFFERENT"
            for key in set(a.files) | set(b.files)
        }
    if "DIFFERENT" in statuses:
        return "DIFFERENT"
    return "WITHIN TOL" if "WITHIN TOL" in statuses else "SAME"


def _figures(before: Path, after: Path) -> str:
    names = {p.name for p in before.glob("*.png")} | {
        p.name for p in after.glob("*.png")
    }
    for name in names:
        a, b = before / name, after / name
        if (
            not a.is_file()
            or not b.is_file()
            or not np.array_equal(_pixels(a), _pixels(b))
        ):
            return "DIFFERENT"
    return "SAME"


def compare(
    before: Path,
    after: Path,
    *,
    rtol: float | None = None,
    atol: float | None = None,
) -> int:
    """Print one row per case directory in before/after; return the comparison code.

    Read snapshots written by write_snapshot. Arrays must have identical shape,
    dtype and NaN-equal values; complex components compare separately. PNGs
    compare decoded RGBA pixels. Explicit finite nonnegative rtol/atol allow
    floating/complex component errors <= atol + rtol * abs(after), with an omitted
    tolerance set to zero, and print WITHIN TOL for non-exact matches. Integers
    and text remain exact. Return 0 when outputs (exact or within tolerance)
    and input fingerprints match, 1 for changed/missing outputs or cases,
    or 3 for input-only differences. Code 1 takes priority over 3. Empty roots or
    unreadable/malformed snapshots raise; main translates these errors to code 2.
    """
    for tolerance in (rtol, atol):
        if tolerance is not None and (not math.isfinite(tolerance) or tolerance < 0):
            raise ValueError("tolerances must be finite and nonnegative")
    labels = sorted(
        {p.name for p in before.iterdir() if p.is_dir()}
        | {p.name for p in after.iterdir() if p.is_dir()}
    )
    if not labels:
        raise ValueError("no snapshot cases to compare")
    print(f"{'case':<24} {'arrays':<12} {'figures':<12} {'inputs':<12} verdict")
    code = 0
    for label in labels:
        a, b = before / label, after / label
        if not a.is_dir() or not b.is_dir():
            print(f"{label:<24} MISSING")
            code = 1
            continue
        arrays, figures = _arrays(a, b, rtol, atol), _figures(a, b)
        inputs = [
            json.dumps(
                json.loads((p / "inputs.json").read_text(encoding="utf-8")),
                sort_keys=True,
            )
            for p in (a, b)
        ]
        same_inputs = inputs[0] == inputs[1]
        verdict = arrays if same_inputs else "INPUTS DIFFER"
        if arrays == "DIFFERENT" or figures == "DIFFERENT":
            verdict, code = "DIFFERENT", 1
        elif not same_inputs and code != 1:
            code = 3
        input_status = "SAME" if same_inputs else "DIFFERENT"
        print(f"{label:<24} {arrays:<12} {figures:<12} {input_status:<12} {verdict}")
    return code


def main(argv: list[str] | None = None) -> int:
    """Compare snapshot roots from argv, or sys.argv when None; never exit the process.

    Return 0 for equality or help, 1 for output/case differences, 2 for usage,
    I/O or format errors, and 3 for input-only differences requiring human review.
    Errors print to stderr; comparison tables and help print to stdout.
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Exit codes: 0 outputs match (exact or within explicit tolerance) and inputs match; "
            "1 arrays, PNGs or cases differ/missing "
            "(takes priority over 3); 2 usage, I/O or snapshot format error; "
            "3 outputs match but inputs differ (human review required). -h returns 0."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser(
        "compare", help="compare two saved snapshot roots", epilog=parser.epilog
    )
    command.add_argument("before", type=Path)
    command.add_argument("after", type=Path)
    command.add_argument(
        "--rtol", type=float, help="relative component tolerance; omitted value is 0"
    )
    command.add_argument(
        "--atol", type=float, help="absolute component tolerance; omitted value is 0"
    )
    try:
        args = parser.parse_args(argv)
    except SystemExit as error:
        return error.code if isinstance(error.code, int) else 2
    try:
        return compare(args.before, args.after, rtol=args.rtol, atol=args.atol)
    except (OSError, ValueError, TypeError, KeyError, EOFError, BadZipFile) as error:
        print(f"probe_snapshot: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
