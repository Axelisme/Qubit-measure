"""Tests for rabi adapter module writeback items.

Verifies that AmpRabiAdapter and LenRabiAdapter emit ModuleWriteback items
(in addition to the existing MetaDictWriteback items) when a cfg_snapshot is
available, and that the module value trees have the correct field overrides.
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
    MetaDictWriteback,
    ModuleWriteback,
    WritebackRequest,
)
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
            pi_amp=pi_amp,
            pi2_amp=pi2_amp,
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
        assert "pi_amp" in by_name
        assert by_name["pi_amp"].proposed_value == pytest.approx(0.4)
        assert "pi2_amp" in by_name
        assert by_name["pi2_amp"].proposed_value == pytest.approx(0.2)

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
            pi_amp=0.4, pi2_amp=0.2, figure=MagicMock()
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
            pi2_len=pi2_len,
            rabi_f=rabi_f,
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
            pi_len=0.05, pi2_len=0.025, rabi_f=10.0, figure=MagicMock()
        )
        req = WritebackRequest(
            run_result=run_result,
            analyze_result=analyze_result,
            ctx=_make_ctx(),
        )
        items = list(LenRabiAdapter().get_writeback_items(req))
        assert len(items) == 3
        assert all(isinstance(it, MetaDictWriteback) for it in items)
