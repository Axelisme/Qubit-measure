"""Snapshot files and the in-process comparison CLI share one public contract."""

from dataclasses import dataclass
from io import BytesIO

import numpy as np
import probe_snapshot as snapshot
import pytest
from numpy.typing import NDArray
from PIL import Image


@dataclass
class _Result:
    signal: NDArray[np.float64]


def _png(color: str, *, compress_level: int = 6) -> bytes:
    buffer = BytesIO()
    Image.new("RGBA", (2, 2), color).save(
        buffer, format="PNG", compress_level=compress_level
    )
    return buffer.getvalue()


@pytest.fixture
def pair(tmp_path):
    """Save two independent roots and return their in-process compare argv."""

    def save(
        before,
        after,
        *,
        inputs_before=None,
        inputs_after=None,
        pngs_before=None,
        pngs_after=None,
    ):
        a, b = tmp_path / "before", tmp_path / "after"
        snapshot.write_snapshot(a, "case", before, pngs_before, inputs=inputs_before)
        snapshot.write_snapshot(b, "case", after, pngs_after, inputs=inputs_after)
        return ["compare", str(a), str(b)]

    return save


def test_saved_arrays_pngs_and_inputs_can_be_compared(tmp_path, capsys):
    data = np.array([1.0, np.nan, 3.0])
    for side, compression in (("before", 0), ("after", 9)):
        snapshot.write_snapshot(
            tmp_path / side,
            "run",
            _Result(data),
            {"live.png": _png("red", compress_level=compression)},
            inputs={"seed": 7},
        )
    with np.load(tmp_path / "before/run/arrays.npz", allow_pickle=False) as arrays:
        np.testing.assert_array_equal(arrays["run.signal"], data)
    assert (tmp_path / "before/run/live.png").read_bytes() == _png(
        "red", compress_level=0
    )
    assert (tmp_path / "before/run/inputs.json").is_file()
    assert (
        snapshot.main(["compare", str(tmp_path / "before"), str(tmp_path / "after")])
        == 0
    )
    output = capsys.readouterr().out
    assert "run" in output
    assert "SAME" in output


@pytest.mark.parametrize("option,amount", [("--rtol", "0.01"), ("--atol", "0.02")])
def test_tolerance_is_opt_in_and_reported(pair, capsys, option, amount):
    argv = pair(np.array([1.0, np.nan]), np.array([1.005, np.nan]))
    assert snapshot.main(argv) == 1
    assert snapshot.main([*argv, option, amount]) == 0
    assert "WITHIN TOL" in capsys.readouterr().out


@pytest.mark.parametrize(
    "before,after",
    [
        (np.array([1.0]), np.array([2.0])),
        (np.array([1.0]), np.array([[1.0]])),
        (np.array([1.0], dtype=np.float32), np.array([1.0], dtype=np.float64)),
        ({"a": np.array([1])}, {"b": np.array([1])}),
        (np.array([np.nan]), np.array([1.0])),
        (np.array([complex(np.nan, 1)]), np.array([complex(np.nan, 2)])),
        (np.array([1]), np.array([2])),
        ("accepted", "rejected"),
    ],
)
def test_changed_values_shapes_dtypes_or_array_names_fail(pair, before, after):
    assert snapshot.main(pair(before, after)) == 1


@pytest.mark.parametrize(
    "before,after",
    [
        ({"seed": 7}, {"seed": 8}),
        ({"source": np.array([1.0])}, {"source": np.array([2.0])}),
        ({"source": np.array([1.0])}, {"source": np.array([[1.0]])}),
        (
            {"source": np.array([1], dtype=np.int32)},
            {"source": np.array([1], dtype=np.int64)},
        ),
        (None, {"cfg": None}),
    ],
)
def test_input_only_differences_require_review(pair, before, after):
    assert snapshot.main(pair(1.0, 1.0, inputs_before=before, inputs_after=after)) == 3


def test_structured_input_dtype_is_part_of_fingerprint(pair):
    before = np.array([(1.0,)], dtype=[("signal", "f8")])
    after = np.array([(1.0,)], dtype=[("other", "f8")])
    assert snapshot.main(pair(1.0, 1.0, inputs_before=before, inputs_after=after)) == 3


def test_mapping_order_and_array_memory_layout_do_not_change_inputs(pair):
    source = np.arange(4.0).reshape(2, 2)
    argv = pair(
        None,
        None,
        inputs_before={"seed": 7, "source": source},
        inputs_after={"source": np.asfortranarray(source), "seed": 7},
    )
    assert snapshot.main(argv) == 0


def test_output_difference_takes_priority_over_input_difference(pair):
    argv = pair(1.0, 2.0, inputs_before={"seed": 7}, inputs_after={"seed": 8})
    assert snapshot.main(argv) == 1


def test_tolerated_output_with_changed_inputs_still_needs_review(pair, capsys):
    argv = pair(1.0, 1.001, inputs_before={"seed": 7}, inputs_after={"seed": 8})
    assert snapshot.main([*argv, "--rtol", "0.01"]) == 3
    assert "WITHIN TOL" in capsys.readouterr().out


@pytest.mark.parametrize(
    "pngs_after", [{"plot.png": _png("blue")}, {}, {"renamed.png": _png("red")}]
)
def test_changed_pixels_or_figure_names_fail(pair, pngs_after):
    argv = pair(1.0, 1.0, pngs_before={"plot.png": _png("red")}, pngs_after=pngs_after)
    assert snapshot.main([*argv, "--atol", "100"]) == 1


def test_missing_case_fails_even_if_other_case_inputs_differ(tmp_path, pair):
    argv = pair(1.0, 1.0, inputs_before=7, inputs_after=8)
    snapshot.write_snapshot(tmp_path / "before", "extra", 1.0, inputs=None)
    assert snapshot.main(argv) == 1


@pytest.mark.parametrize("argv", [["-h"], ["compare", "-h"]])
def test_help_returns_zero_without_exiting(argv):
    assert snapshot.main(argv) == 0


@pytest.mark.parametrize(
    "argv", [[], ["unknown"], ["compare"], ["compare", "missing-a", "missing-b"]]
)
def test_invalid_cli_usage_or_roots_return_two(argv):
    assert snapshot.main(argv) == 2


@pytest.mark.parametrize("filename", ["arrays.npz", "inputs.json", "plot.png"])
def test_malformed_snapshot_returns_two(tmp_path, pair, filename):
    argv = pair(
        1.0,
        1.0,
        pngs_before={"plot.png": _png("red")},
        pngs_after={"plot.png": _png("red")},
    )
    (tmp_path / "after" / "case" / filename).write_bytes(b"not a snapshot")
    assert snapshot.main(argv) == 2


@pytest.mark.parametrize("tolerance", ["-1", "nan", "inf"])
def test_invalid_tolerance_returns_two(pair, tolerance):
    assert snapshot.main([*pair(1.0, 1.0), "--rtol", tolerance]) == 2


def test_writer_refuses_to_overwrite_existing_case(tmp_path):
    snapshot.write_snapshot(
        tmp_path, "case", 1.0, {"plot.png": _png("red")}, inputs=None
    )
    with pytest.raises(FileExistsError):
        snapshot.write_snapshot(tmp_path, "case", 2.0, inputs=None)
    assert snapshot.main(["compare", str(tmp_path), str(tmp_path)]) == 0


@pytest.mark.parametrize("label", ["file", "allow_pickle"])
def test_case_names_are_not_numpy_keyword_arguments(tmp_path, label):
    snapshot.write_snapshot(tmp_path, label, 1.0, inputs=None)
    assert snapshot.main(["compare", str(tmp_path), str(tmp_path)]) == 0


def test_nested_outputs_and_dataclass_inputs_roundtrip(pair):
    outputs = {"fit": (_Result(np.array([1.0])), ["accepted", None, 2.0])}
    assert (
        snapshot.main(
            pair(
                outputs,
                outputs,
                inputs_before=_Result(np.array([1.0])),
                inputs_after=_Result(np.array([1.0])),
            )
        )
        == 0
    )


@pytest.mark.parametrize(
    "label,pngs",
    [
        ("../escaped", {}),
        ("case", {"../escaped.png": b"x"}),
        ("case", {"not-png": b"x"}),
    ],
)
def test_names_cannot_escape_the_snapshot_directory(tmp_path, label, pngs):
    with pytest.raises(ValueError, match="snapshot"):
        snapshot.write_snapshot(tmp_path, label, 1.0, pngs, inputs=None)


def test_unsupported_output_is_rejected_before_writing(tmp_path):
    with pytest.raises(TypeError, match="unsupported snapshot leaf"):
        snapshot.write_snapshot(tmp_path, "case", object(), inputs=None)


def test_unsupported_inputs_are_not_replaced_by_repr(tmp_path):
    with pytest.raises(TypeError, match="unsupported snapshot input"):
        snapshot.write_snapshot(tmp_path, "case", 1.0, inputs=object())
