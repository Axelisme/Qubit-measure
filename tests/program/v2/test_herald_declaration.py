"""Herald declarations at the public pulse-readout/compiled-program seam."""

import numpy as np
import pytest
from qick.asm_v2 import QickParam
from zcu_tools.cfg_model import CircleClassifier, LinearClassifier, ReadoutClassifier
from zcu_tools.program.herald import herald_ground_mask
from zcu_tools.program.v2 import ModularProgramV2, make_mock_soccfg
from zcu_tools.program.v2.modules import (
    Branch,
    ConstWaveformCfg,
    Join,
    Module,
    PulseReadout,
    Repeat,
    SoftDelay,
)

_DEFAULT_CLASSIFIER = LinearClassifier(angle=0.0, threshold=0.5)


class ReadThrice(Module):
    """Initialize one readout once and emit three reads per shot."""

    name = "three_reads"

    def __init__(self, readout: PulseReadout) -> None:
        self.readout = readout

    def init(self, prog: ModularProgramV2) -> None:
        self.readout.init(prog)

    def run(
        self, prog: ModularProgramV2, t: float | QickParam = 0.0
    ) -> float | QickParam:
        return self.readout.run(prog, self.readout.run(prog, self.readout.run(prog, t)))


def make_readout(
    name: str = "readout",
    *,
    ro_ch: int = 0,
    classifier: ReadoutClassifier | None = _DEFAULT_CLASSIFIER,
) -> PulseReadout:
    return PulseReadout(
        name,
        PulseReadout.Cfg(
            gen_ch=ro_ch,
            ro_ch=ro_ch,
            nqz=1,
            freq=100.0,
            gain=0.1,
            waveform=ConstWaveformCfg(length=0.5),
            ro_length=1.0,
            classifier=classifier,
        ),
    )


def make_control_scope(control: str, readout: PulseReadout) -> Module:
    """Place a readout in one repeated body or in the first selected case."""
    return (
        Repeat("repeat_scope", 2).add_content(readout)
        if control == "Repeat"
        else Branch("branch_scope", readout, None)
    )


def test_same_readout_herald_precedes_terminal_on_its_adc() -> None:
    readout = make_readout()
    prog = ModularProgramV2(make_mock_soccfg(), [ReadThrice(readout)])

    herald = readout.herald_spec(prog, occurrence=0)
    second_candidate = readout.herald_spec(prog, occurrence=1)

    assert herald.ro_ch == 0
    assert herald.read_index == 0
    assert herald.reads_per_shot == 3
    assert herald.classifier == LinearClassifier(angle=0.0, threshold=0.5)
    assert second_candidate.read_index == 1


def test_adc_index_follows_join_emission_not_module_tree_or_adc_position() -> None:
    readout = make_readout()
    prefix = make_readout("prefix")
    terminal = make_readout("terminal")
    other_adc = make_readout("other_adc", ro_ch=1)
    parallel = Join([SoftDelay("wait", 2.0), readout], prefix)
    prog = ModularProgramV2(
        make_mock_soccfg(n_readouts=2), [other_adc, parallel, terminal]
    )

    herald = readout.herald_spec(prog, occurrence=0)

    assert herald.ro_ch == 0
    assert herald.read_index == 1
    assert herald.reads_per_shot == 3


def test_occurrence_is_required_even_for_a_single_emission() -> None:
    readout = make_readout()
    prog = ModularProgramV2(make_mock_soccfg(), [readout, make_readout("terminal")])
    missing_keyword: dict[str, int] = {}
    with pytest.raises(TypeError, match="occurrence"):
        readout.herald_spec(prog, **missing_keyword)


@pytest.mark.parametrize("occurrence", [-1, 3, True])
def test_invalid_occurrence_reports_the_instance_ordinal(occurrence: int) -> None:
    readout = make_readout()
    prog = ModularProgramV2(make_mock_soccfg(), [ReadThrice(readout)])
    with pytest.raises(ValueError, match="occurrence"):
        readout.herald_spec(prog, occurrence=occurrence)


def test_selected_read_must_have_a_later_read_on_the_same_adc() -> None:
    readout = make_readout()
    other_adc = make_readout("other_adc", ro_ch=1)
    prog = ModularProgramV2(make_mock_soccfg(n_readouts=2), [readout, other_adc])
    with pytest.raises(ValueError, match="later.*ADC"):
        readout.herald_spec(prog, occurrence=0)


@pytest.mark.parametrize("control", ["Repeat", "Branch"])
def test_herald_inside_control_flow_is_rejected(control: str) -> None:
    readout = make_readout()
    terminal = make_readout("terminal")
    scope = make_control_scope(control, readout)
    prog = ModularProgramV2(
        make_mock_soccfg(), [scope, terminal], sweep=[("branch_scope", 2)]
    )
    with pytest.raises(ValueError, match=control):
        readout.herald_spec(prog, occurrence=0)


@pytest.mark.parametrize("control", ["Repeat", "Branch"])
@pytest.mark.parametrize("after_herald", [False, True])
def test_control_flow_reads_on_herald_adc_are_rejected(
    control: str, after_herald: bool
) -> None:
    readout = make_readout()
    other_read = make_readout("other_read")
    terminal = make_readout("terminal")
    scope = make_control_scope(control, other_read)
    modules = [readout, scope, terminal] if after_herald else [scope, readout, terminal]
    prog = ModularProgramV2(make_mock_soccfg(), modules, sweep=[("branch_scope", 2)])
    with pytest.raises(ValueError, match=rf"ADC 0.*macro [0-9]+.*{control}.*scope"):
        readout.herald_spec(prog, occurrence=0)


@pytest.mark.parametrize("control", ["Repeat", "Branch"])
def test_control_flow_on_other_adc_does_not_restrict_herald(control: str) -> None:
    readout = make_readout()
    other_read = make_readout("other_read", ro_ch=1)
    terminal = make_readout("terminal")
    scope = make_control_scope(control, other_read)
    prog = ModularProgramV2(
        make_mock_soccfg(n_readouts=2),
        [scope, readout, terminal],
        sweep=[("branch_scope", 2)],
    )

    herald = readout.herald_spec(prog, occurrence=0)

    assert herald.ro_ch == 0
    assert herald.read_index == 0
    assert herald.reads_per_shot == 2


def test_declared_read_count_must_match_trigger_emission_count() -> None:
    readout = make_readout()
    prog = ModularProgramV2(make_mock_soccfg(), [ReadThrice(readout)])
    prog.set_reads_per_shot(4)
    with pytest.raises(ValueError, match="ADC 0.*4.*3"):
        readout.herald_spec(prog, occurrence=0)


def test_missing_classifier_is_not_a_herald_calibration() -> None:
    readout = make_readout(classifier=None)
    prog = ModularProgramV2(make_mock_soccfg(), [ReadThrice(readout)])
    with pytest.raises(ValueError, match="classifier"):
        readout.herald_spec(prog, occurrence=0)


@pytest.mark.parametrize(
    "classifier,signals,expected",
    [
        (
            CircleClassifier(g_center=0j, e_center=2 + 0j, radius=1.5),
            [[0j, 0.5j, 1 + 0j], [1.2 + 0j, -1.5 + 0j, -1.6 + 0j]],
            [[True, True, False], [False, False, False]],
        ),
        (
            LinearClassifier(angle=np.pi / 2, threshold=0.5),
            [[1j, -1j, -0.5j]],
            [[True, False, True]],
        ),
    ],
)
def test_shared_ground_mask_uses_normalized_adc_geometry(
    classifier: ReadoutClassifier,
    signals: list[list[complex]],
    expected: list[list[bool]],
) -> None:
    mask = herald_ground_mask(np.array(signals, dtype=np.complex128), classifier)
    np.testing.assert_array_equal(mask, expected)


def test_circle_without_post_radius_fails_at_declaration_and_classification() -> None:
    classifier = CircleClassifier(g_center=0j, e_center=2 + 0j)
    readout = make_readout(classifier=classifier)
    prog = ModularProgramV2(make_mock_soccfg(), [ReadThrice(readout)])
    with pytest.raises(ValueError, match="radius.*GE Post"):
        readout.herald_spec(prog, occurrence=0)
    with pytest.raises(ValueError, match="radius.*GE Post"):
        herald_ground_mask(np.array([0j]), classifier)
