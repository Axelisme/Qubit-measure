"""Public sample validation and NPZ bytes contract; no filesystem or formula owner."""

from io import BytesIO

import numpy as np
import pytest
from zcu_tools.core.waveform_format import (
    MAX_ARB_WAVEFORM_SAMPLES,
    WaveformFile,
    WaveformFormatError,
    WaveformSamples,
    decode_npz,
    encode_npz,
    validate_samples,
)


def _npz(
    *,
    time: object = (0.0, 1.0),
    idata: object = (0.0, 0.5),
    qdata: object = (0.0, 0.0),
    recipe_json: object | None = None,
) -> bytes:
    stream = BytesIO()
    fields = {
        "time": np.asarray(time),
        "idata": np.asarray(idata),
        "qdata": np.asarray(qdata),
    }
    if recipe_json is not None:
        fields["recipe_json"] = np.asarray(recipe_json)
    np.savez(stream, allow_pickle=True, **fields)
    return stream.getvalue()


def test_codec_round_trip_preserves_samples_and_opaque_recipe_text() -> None:
    recipe = ' { "future_recipe": {"formula": "not a formula"} } '
    source = WaveformFile(
        WaveformSamples(
            time=np.array([0.0, 0.2, 1.0]),
            idata=np.array([0.0, 0.3, 0.5]),
            qdata=np.array([0.0, 0.4, 0.0]),
        ),
        recipe_json=recipe,
    )
    result = decode_npz(encode_npz(source))
    np.testing.assert_array_equal(result.samples.time, source.samples.time)
    np.testing.assert_array_equal(result.samples.idata, source.samples.idata)
    np.testing.assert_array_equal(result.samples.qdata, source.samples.qdata)
    assert result.recipe_json == recipe


def test_codec_encodes_required_zero_q_and_decodes_it_as_none() -> None:
    samples = validate_samples(time=[0, 1], idata=[0, 1], qdata=None)
    payload = encode_npz(WaveformFile(samples))
    with np.load(BytesIO(payload), allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["qdata"], [0.0, 0.0])
    result = decode_npz(payload)
    assert result.samples.qdata is None
    assert result.recipe_json is None
    assert result.samples.time.dtype == np.float64
    assert result.samples.idata.dtype == np.float64


@pytest.mark.parametrize(
    ("time", "idata", "qdata", "reason"),
    [
        ([0, 1], ["bad", 0], None, "array_not_numeric"),
        ([0, 1], np.array([0.0, 1.0j]), None, "array_not_numeric"),
        ([0, 1], [[0, 1]], None, "array_not_1d"),
        ([0, 1], [0, np.nan], None, "array_non_finite"),
        ([0, np.inf], [0, 1], None, "array_non_finite"),
        ([0, 1], [0, 1], [0, np.inf], "array_non_finite"),
        ([0, 1], [0], [0, 0], "array_shape_mismatch"),
        ([0], [0], None, "sample_count_too_small"),
        ([1, 2], [0, 1], None, "time_must_start_at_zero"),
        ([0, 0], [0, 1], None, "time_not_strictly_increasing"),
        ([0, 1, 0.5], [0, 0, 0], None, "time_not_strictly_increasing"),
        ([0, 1], [0, 0.8], [0, 0.8], "amplitude_out_of_range"),
    ],
)
def test_sample_constraints_raise_classified_errors(
    time: object, idata: object, qdata: object, reason: str
) -> None:
    with pytest.raises(WaveformFormatError) as error:
        validate_samples(time=time, idata=idata, qdata=qdata)
    assert error.value.reason == reason


def test_sample_count_accepts_upper_boundary_and_rejects_one_more() -> None:
    count = MAX_ARB_WAVEFORM_SAMPLES
    result = validate_samples(
        time=np.arange(count, dtype=np.float64),
        idata=np.zeros(count),
        qdata=None,
    )
    assert result.time.size == count
    with pytest.raises(WaveformFormatError) as error:
        validate_samples(
            time=np.arange(count + 1, dtype=np.float64),
            idata=np.zeros(count + 1),
            qdata=None,
        )
    assert error.value.reason == "sample_count_too_large"
    assert error.value.data.get("sample_count") == count + 1


@pytest.mark.parametrize(
    ("time", "idata", "qdata"), [([0, 1], [0, 2], [0, 0]), ([0, 0], [0, 1], [0, 0])]
)
def test_decoder_applies_sample_constraints(
    time: object, idata: object, qdata: object
) -> None:
    with pytest.raises(WaveformFormatError):
        decode_npz(_npz(time=time, idata=idata, qdata=qdata))


@pytest.mark.parametrize("missing", ["time", "idata", "qdata"])
def test_decoder_rejects_missing_required_key(missing: str) -> None:
    stream = BytesIO()
    fields = {"time": [0, 1], "idata": [0, 1], "qdata": [0, 0]}
    del fields[missing]
    np.savez(stream, allow_pickle=True, **fields)
    with pytest.raises(WaveformFormatError) as error:
        decode_npz(stream.getvalue())
    assert error.value.reason == "missing_npz_key"
    assert error.value.data.get("missing_keys") == [missing]


def test_decoder_rejects_unknown_key() -> None:
    stream = BytesIO()
    np.savez(stream, time=[0, 1], idata=[0, 1], qdata=[0, 0], extra=[1])
    with pytest.raises(WaveformFormatError) as error:
        decode_npz(stream.getvalue())
    assert error.value.reason == "unknown_npz_key"
    assert error.value.data.get("unknown_keys") == ["extra"]


@pytest.mark.parametrize(
    "recipe", ['{"broken":', np.array(["{}"]), np.array(3), np.array(b"{}")]
)
def test_decoder_rejects_non_scalar_string_or_invalid_json(recipe: object) -> None:
    with pytest.raises(WaveformFormatError) as error:
        decode_npz(_npz(recipe_json=recipe))
    assert error.value.reason == "invalid_recipe_json"


def test_decoder_rejects_malformed_or_pickle_archive_with_cause() -> None:
    for payload in (b"not an archive", _npz(qdata=np.array([object(), object()]))):
        with pytest.raises(WaveformFormatError) as error:
            decode_npz(payload)
        assert error.value.reason == "npz_read_failed"
        assert error.value.__cause__ is not None


def test_encoder_rejects_invalid_samples_and_json_before_emitting_bytes() -> None:
    invalid = WaveformFile(
        WaveformSamples(np.array([0.0, 1.0]), np.array([0.0, 2.0]), None)
    )
    with pytest.raises(WaveformFormatError) as error:
        encode_npz(invalid)
    assert error.value.reason == "amplitude_out_of_range"
    valid_samples = validate_samples(time=[0, 1], idata=[0, 1], qdata=None)
    with pytest.raises(WaveformFormatError) as error:
        encode_npz(WaveformFile(valid_samples, recipe_json="not JSON"))
    assert error.value.reason == "invalid_recipe_json"


def test_validation_isolates_a_read_from_mutable_input_samples() -> None:
    time = np.array([0.0, 1.0])
    idata = np.array([0.0, 1.0])
    result = validate_samples(time=time, idata=idata, qdata=None)
    time[1] = 9.0
    idata[1] = 0.0
    np.testing.assert_array_equal(result.time, [0.0, 1.0])
    np.testing.assert_array_equal(result.idata, [0.0, 1.0])
