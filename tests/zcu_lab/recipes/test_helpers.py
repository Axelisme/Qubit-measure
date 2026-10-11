"""Domain helpers through public cfg reads/edits and the recording GUI seam."""

from contextlib import closing
from copy import deepcopy
from pathlib import Path
from typing import Literal

import pytest
from zcu_tools.mcp.measure.execution_reply import SummaryParameter
from zcu_tools.mcp.measure.recipe import (
    RecipeGenerator,
    RecipeNeedsParameters,
    RecipeSession,
)
from zcu_tools.mcp.measure.session import GuiRpcError

from tests.mcp.measure._recipe_execution_support import definition
from tests.mcp.measure._recipe_field_support import SweepGui as _SweepGui
from tests.mcp.measure._recipe_parameter_support import SweepGui
from tests.mcp.measure._recipe_support import (
    LookbackGui,
    publish_reference_locks,
    recipe_client,
    scalar,
)
from tests.mcp.measure._support import make_client
from zcu_lab.recipes.helpers import (
    FrequencySweepCalibration,
    set_flux_sweep,
    set_frequency,
    set_frequency_sweep,
    use_flux_device,
)


@pytest.mark.parametrize("single", [False, True])
@pytest.mark.parametrize("frequency", [None, 6200.0])
def test_author_frequency_hides_both_readout_shapes(tmp_path, single, frequency):
    gui = LookbackGui()
    gui.md["$resonator.freq"] = 6100.0
    readout = gui.publication["tree"]["children"]["modules"]["children"]["readout"]
    if single:
        readout["children"] = {"ro_freq": scalar(5000.0)}
    with recipe_client(tmp_path, gui) as client:
        tab = RecipeSession(client.context).open_tab("lookback")
        set_frequency(
            tab,
            "modules.readout",
            frequency,
            calibration="$resonator.freq",
            required="frequency_mhz",
        )
    expected = frequency if frequency is not None else 6100.0
    if single:
        assert readout["children"]["ro_freq"]["input"]["resolved"] == expected
    else:
        assert (
            readout["children"]["pulse_cfg"]["children"]["freq"]["input"]["resolved"]
            == expected
        )
        assert (
            readout["children"]["ro_cfg"]["children"]["ro_freq"]["input"]["resolved"]
            == expected
        )


@pytest.mark.parametrize("prefer_library", [True, False])
def test_author_frequency_only_uses_a_library_when_requested(tmp_path, prefer_library):
    gui = LookbackGui()
    gui.md["$resonator.freq"] = 6100.0
    readout = gui.publication["tree"]["children"]["modules"]["children"]["readout"]
    readout["ref"] = "R1.readout.readout"
    publish_reference_locks(readout)
    gui.library["R1.readout.readout"] = {}

    with recipe_client(tmp_path, gui) as client:
        tab = RecipeSession(client.context).open_tab("lookback")
        set_frequency(
            tab,
            "modules.readout",
            calibration="$resonator.freq",
            prefer_library=prefer_library,
            required="frequency_mhz",
        )
    assert readout["children"]["pulse_cfg"]["children"]["freq"]["input"][
        "resolved"
    ] == (5000.0 if prefer_library else 6100.0)


def test_author_frequency_missing_calibration_does_not_partially_edit(tmp_path):
    gui = LookbackGui()
    before = deepcopy(gui.publication)
    with recipe_client(tmp_path, gui) as client:
        tab = RecipeSession(client.context).open_tab("lookback")
        with pytest.raises(RecipeNeedsParameters) as error:
            set_frequency(
                tab,
                "modules.readout",
                calibration="$resonator.freq",
                required="frequency_mhz",
            )
        assert [item.parameter for item in error.value.missing] == ["frequency_mhz"]
    assert gui.publication == before


def test_author_qubit_frequency_uses_the_named_calibration(tmp_path):
    gui = LookbackGui()
    gui.md.update({"$resonator.freq": 6100.0, "$qubit.freq": 4500.0})
    gui.publication["tree"]["children"]["modules"]["children"]["qub_pulse"] = {
        "kind": "reference",
        "valid": True,
        "readonly": False,
        "mode": None,
        "ref": None,
        "children": {"freq": scalar(4000.0)},
    }
    with recipe_client(tmp_path, gui) as client:
        tab = RecipeSession(client.context).open_tab("lookback")
        set_frequency(
            tab,
            "modules.qub_pulse.freq",
            calibration="$qubit.freq",
            required="frequency_mhz",
        )
    assert (
        gui.publication["tree"]["children"]["modules"]["children"]["qub_pulse"][
            "children"
        ]["freq"]["input"]["resolved"]
        == 4500.0
    )


@pytest.mark.parametrize("frequency", [float("nan"), float("inf"), True])
def test_author_rejects_nonfinite_or_boolean_frequency_before_editing(
    tmp_path, frequency
):
    gui = LookbackGui()
    with recipe_client(tmp_path, gui) as client:
        tab = RecipeSession(client.context).open_tab("lookback")
        with pytest.raises(ValueError, match="finite"):
            set_frequency(
                tab,
                "modules.readout",
                frequency,
                calibration="$resonator.freq",
                required="frequency_mhz",
            )
    assert gui.publication["cfg_ref"]["revision"] == "2"


@pytest.mark.parametrize("center", [None, 6500.0])
@pytest.mark.parametrize("span", [None, 30.0])
@pytest.mark.parametrize("expts", [None, 7])
def test_author_frequency_sweep_retains_gui_width_and_omitted_count(
    tmp_path, center, span, expts
):
    gui = _SweepGui()
    with recipe_client(tmp_path, gui) as client:
        tab = RecipeSession(client.context).open_tab("lookback")
        set_frequency_sweep(
            tab,
            "sweep.freq",
            calibration=FrequencySweepCalibration(
                "$resonator.freq", "$resonator.kappa", 5.0
            ),
            center_mhz=center,
            span_mhz=span,
            expts=expts,
        )
    inputs = gui.publication["tree"]["children"]["sweep"]["children"]["freq"]["inputs"]
    expected_center = 6100.0 if center is None else center
    expected_span = 20.0 if span is None else span
    assert inputs["start"]["resolved"] == expected_center - expected_span / 2
    assert inputs["stop"]["resolved"] == expected_center + expected_span / 2
    assert inputs["expts"]["resolved"] == (41 if expts is None else expts)
    assert inputs["start"]["mode"] == "direct"
    assert inputs["stop"]["mode"] == "direct"


def test_author_frequency_sweep_reports_all_missing_inputs_before_edit(tmp_path):
    gui = _SweepGui()
    gui.md.clear()
    with recipe_client(tmp_path, gui) as client:
        tab = RecipeSession(client.context).open_tab("lookback")
        with pytest.raises(RecipeNeedsParameters) as error:
            set_frequency_sweep(
                tab,
                "sweep.freq",
                calibration=FrequencySweepCalibration(
                    "$resonator.freq", "$resonator.kappa", 5.0
                ),
            )
        assert [item.parameter for item in error.value.missing] == [
            "center_mhz",
            "span_mhz",
        ]
    assert gui.publication["cfg_ref"]["revision"] == "2"


@pytest.mark.parametrize(
    "start,stop", [(None, 1.0), (1.0, None), (1.0, 1.0), (float("inf"), 1.0)]
)
def test_author_flux_sweep_rejects_invalid_explicit_endpoints(tmp_path, start, stop):
    gui = _SweepGui()
    with (
        recipe_client(tmp_path, gui) as client,
        pytest.raises(ValueError, match="endpoints"),
    ):
        set_flux_sweep(
            RecipeSession(client.context).open_tab("lookback"),
            "sweep.flux",
            start=start,
            stop=stop,
        )
    assert gui.publication["cfg_ref"]["revision"] == "2"


def test_author_flux_sweep_requires_calibrations_only_for_omitted_endpoints(tmp_path):
    gui = _SweepGui()
    gui.md.clear()
    with recipe_client(tmp_path, gui) as client:
        tab = RecipeSession(client.context).open_tab("lookback")
        set_flux_sweep(tab, "sweep.flux", start=-2.0, stop=3.0)
    inputs = gui.publication["tree"]["children"]["sweep"]["children"]["flux"]["inputs"]
    assert inputs["start"]["resolved"] == -2.0
    assert inputs["stop"]["resolved"] == 3.0


@pytest.mark.parametrize("unit", ["A", "V", "native"])
def test_author_flux_device_writes_validated_unit_into_cfg(tmp_path, unit):
    gui = _SweepGui()
    gui.device = {
        "unit": unit,
        "type_name": "FakeDevice" if unit == "native" else "YOKOGS200",
        "info": {"type": "FakeDevice" if unit == "native" else "YOKOGS200"},
    }
    previous = "V" if unit == "A" else "A"
    gui.publication["tree"]["children"]["flux_unit"] = scalar(previous)
    before_sweep = deepcopy(gui.publication["tree"]["children"]["sweep"])
    with recipe_client(tmp_path, gui) as client:
        tab = RecipeSession(client.context).open_tab("lookback")
        use_flux_device(tab, "coil", unit=unit)
    assert gui.publication["tree"]["children"]["flux_unit"]["input"]["resolved"] == unit
    assert (
        gui.publication["tree"]["children"]["dev"]["children"]["flux_dev"]["input"][
            "resolved"
        ]
        == "coil"
    )
    assert gui.publication["tree"]["children"]["sweep"] == before_sweep


def test_author_flux_device_checks_units_without_changing_coordinates(tmp_path):
    gui = _SweepGui()
    with recipe_client(tmp_path, gui) as client:
        tab = RecipeSession(client.context).open_tab("lookback")
        use_flux_device(tab, "coil", unit="A")
    assert (
        gui.publication["tree"]["children"]["dev"]["children"]["flux_dev"]["input"][
            "resolved"
        ]
        == "coil"
    )


@pytest.mark.parametrize(
    "device,unit",
    [
        (
            {"unit": "A", "type_name": "YOKOGS200", "info": {"type": "YOKOGS200"}},
            "native",
        ),
        ({"unit": "A", "type_name": "YOKOGS200", "info": {"type": "YOKOGS200"}}, "V"),
        (
            {
                "unit": "native",
                "type_name": "FakeDevice",
                "info": {"type": "FakeDevice"},
            },
            "A",
        ),
        (
            {"unit": "none", "type_name": "FakeDevice", "info": {"type": "Other"}},
            "native",
        ),
    ],
)
def test_author_flux_device_rejects_mismatched_or_unconfirmed_units_before_edit(
    tmp_path, device, unit
):
    gui = _SweepGui()
    gui.device = device
    with recipe_client(tmp_path, gui) as client:
        with pytest.raises(GuiRpcError) as error:
            use_flux_device(
                RecipeSession(client.context).open_tab("lookback"), "coil", unit=unit
            )
        assert error.value.reason == "invalid_device"
    assert gui.publication["cfg_ref"]["revision"] == "2"


def test_author_fake_flux_requires_explicit_native_opt_in(tmp_path):
    gui = _SweepGui()
    gui.device = {
        "unit": "native",
        "type_name": "FakeDevice",
        "info": {"type": "FakeDevice"},
    }
    with recipe_client(tmp_path, gui) as client:
        use_flux_device(
            RecipeSession(client.context).open_tab("lookback"), "coil", unit="native"
        )
    assert gui.publication["cfg_ref"]["revision"] == "4"


@pytest.mark.parametrize(
    "name,unit", [("", "A"), ("coil", ""), (" ", "A"), ("coil", " ")]
)
def test_author_flux_device_rejects_blank_assertions_before_edit(tmp_path, name, unit):
    gui = _SweepGui()
    with recipe_client(tmp_path, gui) as client:
        tab = RecipeSession(client.context).open_tab("lookback")
        before = len(client.transport.sent)
        with pytest.raises(ValueError, match="must be non-empty"):
            use_flux_device(tab, name, unit=unit)
        assert len(client.transport.sent) == before
    assert gui.publication["cfg_ref"]["revision"] == "2"


def test_author_run_captures_asserted_flux_unit_without_conversion(tmp_path):
    gui = _SweepGui()
    with recipe_client(tmp_path, gui) as client:
        tab = RecipeSession(
            client.context,
            summary_parameters=(
                SummaryParameter("flux", "sweep.flux"),
                SummaryParameter("device", "dev.flux_dev"),
                SummaryParameter("unit", "flux_unit"),
            ),
        ).open_tab("lookback")
        use_flux_device(tab, "coil", unit="A")
        set_flux_sweep(tab, "sweep.flux", start=0.005, stop=-0.001)
        capture = tab.run().snapshot()
        assert capture is not None
        fields = capture.actual["fields"]
        assert fields["dev.flux_dev"] == {
            "value": "coil",
            "source": "explicit",
        }
        assert fields["sweep.flux"]["value"] == {
            "start": 0.005,
            "stop": -0.001,
            "expts": 19,
        }
        assert fields["flux_unit"] == {"value": "A", "source": "explicit"}
        assert fields["sweep.flux"].get("source") == {
            "start": "explicit",
            "stop": "explicit",
            "expts": "gui_default",
        }


@pytest.mark.parametrize("type_name", ["FakeDevice", "OtherDevice"])
def test_author_flux_device_uses_declared_native_coordinate(tmp_path, type_name):
    gui = _SweepGui()
    gui.device = {"unit": "native", "type_name": type_name, "info": {"type": type_name}}
    before_sweep = deepcopy(gui.publication["tree"]["children"]["sweep"])
    with recipe_client(tmp_path, gui) as client:
        tab = RecipeSession(client.context).open_tab("lookback")
        try:
            use_flux_device(tab, "coil", unit="native")
        except GuiRpcError as error:
            if error.reason == "invalid_device" and str(error) in (
                "Fake flux requires confirmed native coordinates",
                "Flux device has no physical unit",
            ):
                pytest.fail(f"declared native coordinate must be accepted: {error}")
            raise
    children = gui.publication["tree"]["children"]
    assert children["dev"]["children"]["flux_dev"]["input"]["resolved"] == "coil"
    assert children["flux_unit"]["input"]["resolved"] == "native"
    assert children["sweep"] == before_sweep
    assert gui.publication["cfg_ref"]["revision"] == "4"


def test_author_flux_device_rejects_unknown_coordinate_before_edit(tmp_path):
    gui = _SweepGui()
    gui.device = {
        "unit": "none",
        "type_name": "FakeDevice",
        "info": {"type": "FakeDevice"},
    }
    with recipe_client(tmp_path, gui) as client:
        tab = RecipeSession(client.context).open_tab("lookback")
        before = deepcopy(gui.publication)
        with pytest.raises(GuiRpcError) as error:
            use_flux_device(tab, "coil", unit="native")
        assert error.value.reason == "invalid_device"
        assert not [call for call in client.transport.sent if call[0] == "tab.edit_cfg"]
    assert gui.publication == before
    assert gui.publication["cfg_ref"]["revision"] == "2"


def frequency_sweep(
    session: RecipeSession,
    *,
    calibration: Literal["resonator", "qubit"],
) -> RecipeGenerator:
    """Resolve current calibration to numeric endpoints with full-span factor six."""
    tab = session.open_tab("lookback")
    set_frequency_sweep(
        tab,
        "sweep",
        calibration=FrequencySweepCalibration(
            f"${calibration}.freq", f"${calibration}.kappa", 6.0
        ),
    )
    yield tab.run()


def flux_sweep(
    session: RecipeSession, *, start: float | None = None, stop: float | None = None
) -> RecipeGenerator:
    """Prepare explicit flux endpoints; reject partial or equal inputs."""
    tab = session.open_tab("lookback")
    set_flux_sweep(tab, "sweep", start=start, stop=stop)
    yield tab.run()


@pytest.mark.parametrize("calibration", ["resonator", "qubit"])
def test_frequency_sweep_preserves_dotted_linewidth_expression(
    tmp_path: Path, calibration: Literal["resonator", "qubit"]
) -> None:
    center = f"${calibration}.freq"
    width = f"${calibration}.kappa"
    gui = SweepGui(f"{center} - 3 * {width}", f"{center} + 3 * {width}", 970.0, 1030.0)
    gui.md = {center: 1000.0, width: 10.0}
    recipe = definition(
        frequency_sweep,
        {
            "type": "object",
            "properties": {"calibration": {"type": "string"}},
            "required": ["calibration"],
            "additionalProperties": False,
        },
    )
    client = make_client(tmp_path, gui, recipes=(recipe,))
    with closing(client.context.session):
        reply = client.call("sample", {"calibration": calibration})
        assert reply.data["status"] == "finished"
        assert gui.ran
        inputs = gui.publication["tree"]["children"]["sweep"]["inputs"]
        assert inputs["start"]["resolved"] == 970.0
        assert inputs["stop"]["resolved"] == 1030.0
        assert inputs["expts"]["resolved"] == 51
        assert {center, width} <= set(gui.queried_paths)


@pytest.mark.parametrize(
    "start,stop,status",
    [(0.1, None, "failed"), (0.1, 0.1, "failed")],
)
def test_flux_sweep_uses_role_expressions_and_rejects_invalid_explicit_endpoints(
    tmp_path: Path, start: float | None, stop: float | None, status: str
) -> None:
    low = "$qubit.flux_half - ($qubit.flux_int - $qubit.flux_half) / 2"
    high = "$qubit.flux_half + ($qubit.flux_int - $qubit.flux_half) / 2"
    gui = SweepGui(low, high, 0.1, 0.5)
    gui.md = {"$qubit.flux_half": 0.3, "$qubit.flux_int": 0.7}
    recipe = definition(
        flux_sweep,
        {
            "type": "object",
            "properties": {
                "start": {"type": ["number", "null"]},
                "stop": {"type": ["number", "null"]},
            },
            "additionalProperties": False,
        },
    )
    client = make_client(tmp_path, gui, recipes=(recipe,))
    with closing(client.context.session):
        reply = client.call("sample", {"start": start, "stop": stop})
        assert reply.data["status"] == status
        assert gui.ran is (status == "finished")
        inputs = gui.publication["tree"]["children"]["sweep"]["inputs"]
        assert inputs["start"]["raw"] == low
        assert inputs["stop"]["raw"] == high
