"""Fast-flux routes at the public decoded FTF YAML boundary."""

import pytest
from zcu_tools.simulators.ftf.config import parse_ftf
from zcu_tools.simulators.single_fluxonium.config.schema import FluxLine

from tests.simulators.ftf.config._data import dual_readout_document
from tests.simulators.ftf.config._data import ftf_document


@pytest.mark.parametrize(
    "loops",
    [(), ("f1",), ("coupler",), ("f2",), ("f1", "coupler", "f2")],
)
def test_flux_lines_preserve_single_line_calibration_and_optional_loops(
    loops: tuple[str, ...],
) -> None:
    lines = {
        loop: {
            "name": f"{loop}_flux",
            "gen_ch": 4 + index,
            "flux_phi0_per_gain": -0.02,
            "delay_ns": 3.0,
            "tau_ns": 8.0,
            "distortion_terms": [{"amplitude": -0.1, "tau_ns": 12.0}],
        }
        for index, loop in enumerate(loops)
    }
    settings = parse_ftf(ftf_document() | {"flux_lines": lines})
    expected = {
        loop: FluxLine.model_validate(line).model_dump(mode="json")
        for loop, line in lines.items()
    }
    assert settings.flux_lines.model_dump(mode="json", exclude_none=True) == expected
    assert parse_ftf(settings.model_dump(mode="json")) == settings
    baseline = parse_ftf(ftf_document())
    assert baseline.flux_lines.f1 is None
    assert baseline.flux_lines.coupler is None
    assert baseline.flux_lines.f2 is None
    assert settings.model_dump(exclude={"flux_lines"}) == baseline.model_dump(
        exclude={"flux_lines"}
    )


@pytest.mark.parametrize("loops", [("f1", "coupler"), ("coupler", "f2"), ("f1", "f2")])
def test_flux_lines_reject_shared_generator_with_both_loops_and_channel(
    loops: tuple[str, str],
) -> None:
    first, second = loops
    lines = {
        loop: {"name": f"{loop}_flux", "gen_ch": 7, "flux_phi0_per_gain": 0.02}
        for loop in loops
    }
    with pytest.raises(ValueError, match="gen_ch") as error:
        parse_ftf(ftf_document() | {"flux_lines": lines})
    message = str(error.value)
    assert f"flux_lines.{first}" in message
    assert f"flux_lines.{second}" in message
    assert "gen_ch" in message
    assert "7" in message


@pytest.mark.parametrize("loop", ["f1", "coupler", "f2"])
@pytest.mark.parametrize(
    "binding,channel,operator",
    [
        ("drive_lines.mw", 2, "charge"),
        ("drive_lines.mw", 2, "flux"),
        ("readout.drive", 0, "charge"),
        ("readout.mux", 1, "charge"),
    ],
)
def test_flux_lines_reject_drive_and_readout_generator_reuse(
    loop: str, binding: str, channel: int, operator: str
) -> None:
    document = dual_readout_document()
    if operator == "flux":
        document["drive_lines"] = {
            "mw": {
                "gen_ch": 2,
                "operator": "flux",
                "flux_drive_phi0_per_gain": {
                    "f1": {"real": 0.02, "imag": 0.0},
                    "coupler": {"real": 0.0, "imag": 0.0},
                    "f2": {"real": 0.0, "imag": 0.0},
                },
            }
        }
    document["flux_lines"] = {
        loop: {"name": "park", "gen_ch": channel, "flux_phi0_per_gain": 0.02}
    }
    with pytest.raises(ValueError, match="gen_ch") as error:
        parse_ftf(document)
    message = str(error.value)
    assert f"flux_lines.{loop}.gen_ch" in message
    assert f"{binding}.gen_ch" in message
    assert f"channel {channel}" in message


def test_flux_lines_allow_free_generators_with_mux_and_repeated_names() -> None:
    settings = parse_ftf(
        dual_readout_document()
        | {
            "flux_lines": {
                loop: {"name": "mw", "gen_ch": channel, "flux_phi0_per_gain": 0.02}
                for loop, channel in (("f1", 4), ("coupler", 5), ("f2", 6))
            }
        }
    )
    assert settings.flux_lines.model_dump(exclude_none=True) == {
        loop: FluxLine(name="mw", gen_ch=channel, flux_phi0_per_gain=0.02).model_dump()
        for loop, channel in (("f1", 4), ("coupler", 5), ("f2", 6))
    }
    assert parse_ftf(settings.model_dump(mode="json")) == settings


def test_flux_lines_allow_null_loops_and_default_instantaneous_response() -> None:
    settings = parse_ftf(
        ftf_document()
        | {
            "flux_lines": {
                "f1": {"name": "park", "gen_ch": 100, "flux_phi0_per_gain": 0.0},
                "coupler": None,
                "f2": None,
            }
        }
    )
    line = settings.flux_lines.f1
    assert line is not None
    assert line.gen_ch == 100
    assert line.flux_phi0_per_gain == 0.0
    assert line.delay_ns == 0.0
    assert line.tau_ns == 0.0
    assert line.distortion_terms == ()
    assert settings.flux_lines.coupler is None
    assert settings.flux_lines.f2 is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("name", ""),
        ("gen_ch", -1),
        ("gen_ch", True),
        ("gen_ch", 1.5),
        ("flux_phi0_per_gain", float("nan")),
        ("delay_ns", -1.0),
        ("tau_ns", -1.0),
        ("distortion_terms", [{"amplitude": 0.1, "tau_ns": 0.0}]),
        ("phi0_per_gain", 0.02),
    ],
)
def test_flux_line_validation_reports_nested_field_path(
    field: str,
    value: object,
) -> None:
    line = {"name": "park", "gen_ch": 4, "flux_phi0_per_gain": 0.02, field: value}
    with pytest.raises(ValueError, match=rf"flux_lines\.f1\.{field}"):
        parse_ftf(ftf_document() | {"flux_lines": {"f1": line}})


@pytest.mark.parametrize("lines", [None, {"other": {}}, {"f1": {}}])
def test_flux_lines_reject_unknown_loops_and_incomplete_lines(lines: object) -> None:
    with pytest.raises(ValueError, match="flux_lines"):
        parse_ftf(ftf_document() | {"flux_lines": lines})
