"""Tests for rabi adapter module writeback items and default-value seed.

Verifies that AmpRabiAdapter and LenRabiAdapter emit ModuleWriteback items
(in addition to the existing MetaDictWriteback items) when a cfg_snapshot is
available, and that the module value trees have the correct field overrides.
Also verifies that AmpRabiAdapter.make_default_value seeds the gain sweep
upper bound from md key 'pi_gain' (not the old 'pi_amp').
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from zcu_tools.experiment.v2.twotone.rabi.amp_rabi import (
    AmpRabiCfg,
    AmpRabiResult,
)
from zcu_tools.experiment.v2.twotone.rabi.len_rabi import (
    LenRabiCfg,
    LenRabiResult,
)
from zcu_tools.experiment.v2_gui.adapters.twotone.rabi.amp_rabi import (
    AmpRabiAdapter,
    AmpRabiAnalyzeResult,
)
from zcu_tools.experiment.v2_gui.adapters.twotone.rabi.len_rabi import (
    LenRabiAdapter,
    LenRabiAnalyzeResult,
)
from zcu_tools.gui.app.main.adapter import (
    DirectValue,
    EvalValue,
    MetaDictWriteback,
    ModuleWriteback,
    SweepValue,
    WritebackRequest,
)
from zcu_tools.meta_tool import MetaDict
from zcu_tools.program.v2 import PulseCfg
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg
from zcu_tools.program.v2.twotone import TwoToneModuleCfg

# ---------------------------------------------------------------------------
# Fixtures — shared across both adapters
# ---------------------------------------------------------------------------


def _make_qub_pulse(gain: float = 0.5, length: float = 0.1) -> PulseCfg:
    """Construct a minimal PulseCfg with a const waveform for testing."""
    return PulseCfg(
        type="pulse",
        waveform=ConstWaveformCfg(style="const", length=length),
        ch=0,
        nqz=1,
        freq=3000.0,
        gain=gain,
    )


def _make_amp_rabi_run_result(gain: float = 0.5) -> AmpRabiResult:
    """AmpRabiResult with a real cfg_snapshot containing qub_pulse."""
    import numpy as np

    qub_pulse = _make_qub_pulse(gain=gain)
    modules = MagicMock(spec=TwoToneModuleCfg)
    modules.qub_pulse = qub_pulse
    cfg = MagicMock(spec=AmpRabiCfg)
    cfg.modules = modules
    return AmpRabiResult(
        amps=np.array([0.0, 0.5, 1.0]),
        signals=np.array([0.0, 1.0, 0.0], dtype=complex),
        cfg_snapshot=cfg,
    )


def _make_len_rabi_run_result(length: float = 0.1) -> LenRabiResult:
    """LenRabiResult with a real cfg_snapshot containing qub_pulse."""
    import numpy as np

    qub_pulse = _make_qub_pulse(length=length)
    modules = MagicMock(spec=TwoToneModuleCfg)
    modules.qub_pulse = qub_pulse
    cfg = MagicMock(spec=LenRabiCfg)
    cfg.modules = modules
    return LenRabiResult(
        lengths=np.array([0.0, 0.1, 0.2]),
        signals=np.array([0.0, 1.0, 0.0], dtype=complex),
        cfg_snapshot=cfg,
    )


def _make_ctx() -> MagicMock:
    ctx = MagicMock()
    ctx.ml = MagicMock()
    ctx.qub_name = "Q1"
    return ctx


# ---------------------------------------------------------------------------
# AmpRabiAdapter — writeback items
# ---------------------------------------------------------------------------


class TestAmpRabiWriteback:
    def _items(self, pi_amp: float = 0.4, pi2_amp: float = 0.2) -> list:
        run_result = _make_amp_rabi_run_result(gain=0.5)
        analyze_result = AmpRabiAnalyzeResult(
            pi_gain=pi_amp,
            pi_gain_err=0.01,
            pi2_gain=pi2_amp,
            pi2_gain_err=0.01,
            figure=MagicMock(),
        )
        req = WritebackRequest(
            run_result=run_result,
            analyze_result=analyze_result,
            ctx=_make_ctx(),
        )
        return list(AmpRabiAdapter().get_writeback_items(req))

    def test_returns_four_items(self) -> None:
        # 2 md items + 2 module items
        items = self._items()
        assert len(items) == 4

    def test_md_items_present(self) -> None:
        items = self._items(pi_amp=0.4, pi2_amp=0.2)
        md_items = [it for it in items if isinstance(it, MetaDictWriteback)]
        by_name = {it.target_name: it for it in md_items}
        # Scalar gains write back as 'pi_gain'/'pi2_gain' (single_qubit.md);
        # the module items keep the 'pi_amp'/'pi2_amp' names.
        assert "pi_gain" in by_name
        assert by_name["pi_gain"].proposed_value == pytest.approx(0.4)
        assert "pi2_gain" in by_name
        assert by_name["pi2_gain"].proposed_value == pytest.approx(0.2)

    def test_module_items_present(self) -> None:
        items = self._items()
        mod_items = [it for it in items if isinstance(it, ModuleWriteback)]
        assert len(mod_items) == 2
        names = {it.target_name for it in mod_items}
        assert names == {"pi_amp", "pi2_amp"}

    def test_module_items_have_edit_schema(self) -> None:
        items = self._items()
        for item in items:
            if isinstance(item, ModuleWriteback):
                assert item.edit_schema is not None

    def test_pi_amp_module_gain_overridden(self) -> None:
        items = self._items(pi_amp=0.4)
        pi_mod = next(
            it
            for it in items
            if isinstance(it, ModuleWriteback) and it.target_name == "pi_amp"
        )
        # The gain field in the value tree must equal pi_amp.
        gain_val = pi_mod.edit_schema.value.fields["gain"]  # type: ignore[union-attr]
        assert isinstance(gain_val, DirectValue)
        assert gain_val.value == pytest.approx(0.4)

    def test_pi2_amp_module_gain_overridden(self) -> None:
        items = self._items(pi2_amp=0.2)
        pi2_mod = next(
            it
            for it in items
            if isinstance(it, ModuleWriteback) and it.target_name == "pi2_amp"
        )
        gain_val = pi2_mod.edit_schema.value.fields["gain"]  # type: ignore[union-attr]
        assert isinstance(gain_val, DirectValue)
        assert gain_val.value == pytest.approx(0.2)

    def test_module_item_descriptions(self) -> None:
        items = self._items()
        mods = {
            it.target_name: it.description
            for it in items
            if isinstance(it, ModuleWriteback)
        }
        assert mods["pi_amp"] == "amp pi pulse"
        assert mods["pi2_amp"] == "amp pi/2 pulse"

    def test_no_module_items_when_no_snapshot(self) -> None:
        """No cfg_snapshot → only md items (e.g. loaded from file)."""
        import numpy as np

        run_result = AmpRabiResult(
            amps=np.array([0.0, 0.5, 1.0]),
            signals=np.array([0.0, 1.0, 0.0], dtype=complex),
            cfg_snapshot=None,
        )
        analyze_result = AmpRabiAnalyzeResult(
            pi_gain=0.4,
            pi_gain_err=0.01,
            pi2_gain=0.2,
            pi2_gain_err=0.01,
            figure=MagicMock(),
        )
        req = WritebackRequest(
            run_result=run_result,
            analyze_result=analyze_result,
            ctx=_make_ctx(),
        )
        items = list(AmpRabiAdapter().get_writeback_items(req))
        assert len(items) == 2
        assert all(isinstance(it, MetaDictWriteback) for it in items)


# ---------------------------------------------------------------------------
# LenRabiAdapter — writeback items
# ---------------------------------------------------------------------------


class TestLenRabiWriteback:
    def _items(
        self, pi_len: float = 0.05, pi2_len: float = 0.025, rabi_f: float = 10.0
    ) -> list:
        run_result = _make_len_rabi_run_result(length=0.1)
        analyze_result = LenRabiAnalyzeResult(
            pi_len=pi_len,
            pi_len_err=0.001,
            pi2_len=pi2_len,
            pi2_len_err=0.001,
            rabi_f=rabi_f,
            rabi_f_err=0.1,
            figure=MagicMock(),
        )
        req = WritebackRequest(
            run_result=run_result,
            analyze_result=analyze_result,
            ctx=_make_ctx(),
        )
        return list(LenRabiAdapter().get_writeback_items(req))

    def test_returns_five_items(self) -> None:
        # 3 md items (pi_len, pi2_len, rabi_f) + 2 module items
        items = self._items()
        assert len(items) == 5

    def test_md_items_present(self) -> None:
        items = self._items(pi_len=0.05, pi2_len=0.025, rabi_f=9.5)
        md_items = [it for it in items if isinstance(it, MetaDictWriteback)]
        by_name = {it.target_name: it for it in md_items}
        assert by_name["pi_len"].proposed_value == pytest.approx(0.05)
        assert by_name["pi2_len"].proposed_value == pytest.approx(0.025)
        assert by_name["rabi_f"].proposed_value == pytest.approx(9.5)

    def test_rabi_f_md_item(self) -> None:
        items = self._items(rabi_f=12.3)
        rabi_item = next(
            it
            for it in items
            if isinstance(it, MetaDictWriteback) and it.target_name == "rabi_f"
        )
        assert rabi_item.proposed_value == pytest.approx(12.3)

    def test_module_items_present(self) -> None:
        items = self._items()
        mod_items = [it for it in items if isinstance(it, ModuleWriteback)]
        assert len(mod_items) == 2
        names = {it.target_name for it in mod_items}
        # len_rabi produces length-calibrated modules 'pi_len'/'pi2_len'
        # (single_qubit.md); amp_rabi produces the separate 'pi_amp'/'pi2_amp'.
        assert names == {"pi_len", "pi2_len"}

    def test_pi_len_module_length_overridden(self) -> None:
        items = self._items(pi_len=0.05)
        pi_mod = next(
            it
            for it in items
            if isinstance(it, ModuleWriteback) and it.target_name == "pi_len"
        )
        # The waveform field is a WaveformRefValue; its .value.fields["length"]
        # holds the overridden DirectValue.
        from zcu_tools.gui.app.main.adapter import WaveformRefValue

        wav_ref = pi_mod.edit_schema.value.fields["waveform"]  # type: ignore[union-attr]
        assert isinstance(wav_ref, WaveformRefValue)
        length_val = wav_ref.value.fields["length"]
        assert isinstance(length_val, DirectValue)
        assert length_val.value == pytest.approx(0.05)

    def test_pi2_len_module_length_overridden(self) -> None:
        items = self._items(pi2_len=0.025)
        pi2_mod = next(
            it
            for it in items
            if isinstance(it, ModuleWriteback) and it.target_name == "pi2_len"
        )
        from zcu_tools.gui.app.main.adapter import WaveformRefValue

        wav_ref = pi2_mod.edit_schema.value.fields["waveform"]  # type: ignore[union-attr]
        assert isinstance(wav_ref, WaveformRefValue)
        length_val = wav_ref.value.fields["length"]
        assert isinstance(length_val, DirectValue)
        assert length_val.value == pytest.approx(0.025)

    def test_module_item_descriptions(self) -> None:
        items = self._items()
        mods = {
            it.target_name: it.description
            for it in items
            if isinstance(it, ModuleWriteback)
        }
        assert mods["pi_len"] == "len pi pulse"
        assert mods["pi2_len"] == "len pi/2 pulse"

    def test_no_module_items_when_no_snapshot(self) -> None:
        import numpy as np

        run_result = LenRabiResult(
            lengths=np.array([0.0, 0.1, 0.2]),
            signals=np.array([0.0, 1.0, 0.0], dtype=complex),
            cfg_snapshot=None,
        )
        analyze_result = LenRabiAnalyzeResult(
            pi_len=0.05,
            pi_len_err=0.001,
            pi2_len=0.025,
            pi2_len_err=0.001,
            rabi_f=10.0,
            rabi_f_err=0.1,
            figure=MagicMock(),
        )
        req = WritebackRequest(
            run_result=run_result,
            analyze_result=analyze_result,
            ctx=_make_ctx(),
        )
        items = list(LenRabiAdapter().get_writeback_items(req))
        assert len(items) == 3
        assert all(isinstance(it, MetaDictWriteback) for it in items)


# ---------------------------------------------------------------------------
# AmpRabiAdapter — make_default_value gain-sweep seed
# ---------------------------------------------------------------------------


def _make_ctx_with_md(**md_values: float) -> MagicMock:
    """A ctx whose md is a real MetaDict seeded with md_values.

    md_has_key uses ctx.md.get() internally, so a real MetaDict is required;
    a bare MagicMock would make every key appear present.
    """
    ctx = MagicMock()
    md = MetaDict()
    for key, value in md_values.items():
        setattr(md, key, value)
    ctx.md = md
    ctx.ml = MagicMock()
    ctx.qub_name = "Q1"
    return ctx


def _gain_sweep_stop(ctx: MagicMock) -> float | EvalValue:
    """Extract the gain sweep stop edge from AmpRabiAdapter.make_default_value."""
    val = AmpRabiAdapter().make_default_value(ctx)
    sweep_section = val.fields["sweep"]
    assert isinstance(sweep_section, type(val)), (
        f"Expected CfgSectionValue, got {type(sweep_section)}"
    )
    gain_sweep = sweep_section.fields["gain"]
    assert isinstance(gain_sweep, SweepValue), (
        f"Expected SweepValue, got {type(gain_sweep)}"
    )
    return gain_sweep.stop


class TestAmpRabiDefaultValueGainSeed:
    def test_pi_gain_present_yields_eval_expr(self) -> None:
        # When md has pi_gain, the stop edge must be an EvalValue referencing
        # "2.0 * pi_gain" so the GUI keeps the live md-linked expression.
        ctx = _make_ctx_with_md(pi_gain=0.4)
        stop = _gain_sweep_stop(ctx)
        assert isinstance(stop, EvalValue)
        assert stop.expr == "2.0 * pi_gain"

    def test_pi_gain_absent_yields_fallback_float(self) -> None:
        # When md has no pi_gain, fall back to 2.0 * 0.5 = 1.0.
        ctx = _make_ctx_with_md()
        stop = _gain_sweep_stop(ctx)
        assert isinstance(stop, float)
        assert stop == pytest.approx(1.0)

    def test_pi_amp_does_not_seed_gain_sweep(self) -> None:
        # Old md key 'pi_amp' is now reserved for pulse MODULE names; it must
        # not trigger the md-linked eval expression.  With only pi_amp present
        # the stop must be the plain fallback float, not an EvalValue.
        ctx = _make_ctx_with_md(pi_amp=0.9)
        stop = _gain_sweep_stop(ctx)
        assert isinstance(stop, float), (
            "pi_amp must not seed the gain sweep; expected plain fallback float"
        )


# ---------------------------------------------------------------------------
# LenRabiAdapter — make_default_value length-sweep seed
# ---------------------------------------------------------------------------


def _length_sweep_stop(ctx: MagicMock) -> float | EvalValue:
    """Extract the length sweep stop edge from LenRabiAdapter.make_default_value."""
    val = LenRabiAdapter().make_default_value(ctx)
    sweep_section = val.fields["sweep"]
    assert isinstance(sweep_section, type(val)), (
        f"Expected CfgSectionValue, got {type(sweep_section)}"
    )
    length_sweep = sweep_section.fields["length"]
    assert isinstance(length_sweep, SweepValue), (
        f"Expected SweepValue, got {type(length_sweep)}"
    )
    return length_sweep.stop


class TestLenRabiDefaultValueLengthSeed:
    def test_pi_len_present_yields_eval_expr(self) -> None:
        # When md has pi_len, the stop edge must be an EvalValue referencing
        # "4.0 * pi_len" so the GUI keeps the live md-linked expression.
        ctx = _make_ctx_with_md(pi_len=0.05)
        stop = _length_sweep_stop(ctx)
        assert isinstance(stop, EvalValue)
        assert stop.expr == "4.0 * pi_len"

    def test_pi_len_absent_yields_fallback_float(self) -> None:
        # When md has no pi_len, md_eval_scaled returns 4.0 * 0.1 = 0.4
        # (factor * fallback, where fallback=0.1 from the adapter).
        ctx = _make_ctx_with_md()
        stop = _length_sweep_stop(ctx)
        assert isinstance(stop, float)
        assert stop == pytest.approx(0.4)
