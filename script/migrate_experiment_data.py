from __future__ import annotations

import argparse
import os
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from zcu_tools.experiment.v2.twotone.time_domain.cpmg import (
    CPMG_Result,
    load_cpmg_grouped_result,
    save_cpmg_grouped_result,
)
from zcu_tools.utils.datasaver import format_ext


@dataclass(frozen=True)
class ConverterSpec:
    convert: Callable[[Path, Path], None]
    validate: Callable[[str], object]


def migrate_experiment_data(
    *,
    experiment: str,
    input_path: Path,
    output_path: Path,
    overwrite: bool = False,
) -> Path:
    try:
        spec = CONVERTERS[experiment]
    except KeyError:
        supported = ", ".join(sorted(CONVERTERS))
        raise ValueError(
            f"unsupported experiment {experiment!r}; supported: {supported}"
        ) from None

    if not input_path.is_file():
        raise FileNotFoundError(f"input file does not exist: {input_path}")

    output_path = Path(format_ext(str(output_path)))
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"output file already exists: {output_path}; pass --overwrite to replace it"
        )
    if not output_path.parent.exists():
        raise FileNotFoundError(
            f"output directory does not exist: {output_path.parent}"
        )

    temp_path = _make_temp_path(output_path)
    try:
        spec.convert(input_path, temp_path)
        spec.validate(str(temp_path))
        os.replace(temp_path, output_path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise

    return output_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Migrate legacy experiment data into current HDF5 format."
    )
    parser.add_argument(
        "--experiment",
        required=True,
        choices=sorted(CONVERTERS),
        help="Experiment converter to run.",
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Legacy input file.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output HDF5 file.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output file if it already exists.",
    )
    args = parser.parse_args(argv)

    try:
        migrated = migrate_experiment_data(
            experiment=args.experiment,
            input_path=args.input,
            output_path=args.output,
            overwrite=args.overwrite,
        )
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    print(migrated)
    return 0


def _make_temp_path(output_path: Path) -> Path:
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        suffix=".hdf5",
        dir=output_path.parent,
    )
    os.close(fd)
    return Path(temp_name)


def _convert_cpmg_npz(input_path: Path, output_path: Path) -> None:
    with np.load(input_path) as data:
        missing = {"times", "lengths", "signals2D"} - set(data.files)
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise ValueError(
                f"legacy CPMG npz is missing required arrays: {missing_text}"
            )

        result = CPMG_Result(
            ns=np.asarray(data["times"], dtype=np.int64),
            delays=np.asarray(data["lengths"], dtype=np.float64),
            signals=np.asarray(data["signals2D"], dtype=np.complex128),
        )
        comment = _npz_comment(data)

    save_cpmg_grouped_result(str(output_path), result, comment=comment)


def _npz_comment(data: np.lib.npyio.NpzFile) -> str:
    if "comment" not in data.files:
        return ""
    comment = data["comment"]
    if comment.shape == ():
        return str(comment.tolist())
    return str(comment)


CONVERTERS: dict[str, ConverterSpec] = {
    "twotone/cpmg": ConverterSpec(
        convert=_convert_cpmg_npz,
        validate=load_cpmg_grouped_result,
    )
}


if __name__ == "__main__":
    raise SystemExit(main())
