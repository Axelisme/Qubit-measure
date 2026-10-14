from __future__ import annotations

import time
from collections.abc import Mapping
from pathlib import Path
from typing import ClassVar, TypeAlias

from zcu_tools.experiment.cfg_assembler import ExperimentCfgValue
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.gui.app.measure.adapter import (
    AdapterCapabilities,
    AnalysisMode,
    RunRequest,
    SessionEnv,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.gui.app.measure.schema_builder import (
    MeasureCfgBuilder,
    MeasureCfgDefinition,
)
from zcu_tools.gui.app.measure.seeds import custom, path_seed
from zcu_tools.gui.app.measure.spec_helpers import make_pulse_readout_module_spec
from zcu_tools.gui.cfg import (
    ReferenceValue,
    SweepValue,
    make_custom_reference_key,
    make_default_value,
)

from zcu_lab.v2._support.measure.defaults.role_factories import ROLE_FACTORIES
from zcu_lab.v2._support.measure.seeds import res_freq_range
from zcu_lab.v2.onetone.power_dep.core import PowerDepCfg, PowerDepExp, PowerDepResult

OneTonePowerDepRunResult: TypeAlias = RunRecord[PowerDepCfg, PowerDepResult]


def _blank_readout(ctx: SessionEnv) -> ReferenceValue:
    """Seed fresh inline readout from the tab's fixed resonator role.

    Sweep-owned generator/ADC frequencies and gain stay zero; const pulse length
    stays 1 us and nqz stays 2. Captured wiring channels fall back to zero;
    ext.default_probe_length minus 0.1 us and ext.time_of_fly plus 0.05 us use
    absent/null fallbacks 0.9 and 0.55 us. Role/source errors propagate.
    """
    spec = make_pulse_readout_module_spec().allowed[0]
    return (
        ReferenceValue(
            make_custom_reference_key(spec.label or "Readout"),
            make_default_value(spec),
        )
        .with_field("pulse_cfg.freq", 0.0)
        .with_field("pulse_cfg.gain", 0.0)
        .with_field("pulse_cfg.nqz", 2)
        .with_field("pulse_cfg.waveform.length", 1.0)
        .with_field(
            "pulse_cfg.ch",
            path_seed("$resonator.wiring.ch", fallback=0).resolve(ctx),
        )
        .with_field("ro_cfg.ro_freq", 0.0)
        .with_field(
            "ro_cfg.ro_ch",
            path_seed("$resonator.wiring.ro_ch", fallback=0).resolve(ctx),
        )
        .with_field(
            "ro_cfg.ro_length",
            path_seed(
                "$resonator.ext.default_probe_length",
                fallback=0.9,
                expr="$resonator.ext.default_probe_length - 0.1",
            ).resolve(ctx),
        )
        .with_field(
            "ro_cfg.trig_offset",
            path_seed(
                "$resonator.ext.time_of_fly",
                fallback=0.55,
                expr="$resonator.ext.time_of_fly + 0.05",
            ).resolve(ctx),
        )
    )


class OneTonePowerDepAdapter(BaseAdapter[PowerDepCfg, OneTonePowerDepRunResult]):
    experiment_tag: ClassVar[str | None] = PowerDepExp.persistence_tag()
    exp_cls = PowerDepExp
    ExpCfg_cls = PowerDepCfg
    capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities(
        analysis=AnalysisMode.NONE, load_data=True
    )

    guide_path: ClassVar[Path | None] = Path(__file__).with_name("guide.md")

    @classmethod
    def cfg_definition(cls) -> MeasureCfgDefinition:
        return (
            MeasureCfgBuilder(
                role_factories=ROLE_FACTORIES,
            )
            .field(
                "modules.readout",
                spec=make_pulse_readout_module_spec()
                .lock_literal("pulse_cfg.freq", 0.0)
                .lock_literal("ro_cfg.ro_freq", 0.0)
                .lock_literal("pulse_cfg.gain", 0.0),
                default=custom(_blank_readout, description="inline power readout"),
            )
            .relax_delay(1.0)
            .sweep(
                "gain",
                label="Gain (a.u.)",
                default=SweepValue(start=0.001, stop=1.0, expts=101),
            )
            .sweep(
                "freq",
                label="Freq (MHz)",
                default=res_freq_range(expts=201),
            )
            .float(
                "earlystop_snr",
                label="Early-stop SNR (0 disables)",
                default=0.0,
                decimals=3,
            )
            .reps(1000)
            .rounds(100)
            .build()
        )

    def build_exp_cfg(
        self, raw_cfg: Mapping[str, ExperimentCfgValue], req: RunRequest
    ) -> PowerDepCfg:
        """Assemble accepted cfg values with req's captured device snapshot.

        raw_cfg is the resolved experiment tree from cfg acceptance. Positive
        numeric earlystop_snr enables stopping; nonpositive values disable it.
        Return the typed cfg without hardware I/O; assembly errors propagate.
        """
        cfg_raw = dict(raw_cfg)
        cfg_raw.pop("earlystop_snr", None)
        cfg = super().build_exp_cfg(cfg_raw, req)
        cfg.earlystop_snr = self._earlystop_snr(raw_cfg)
        return cfg

    def _earlystop_snr(self, raw_cfg: Mapping[str, ExperimentCfgValue]) -> float | None:
        value = raw_cfg.get("earlystop_snr")
        if not isinstance(value, (int, float)):
            return None
        snr = float(value)
        if snr <= 0:
            return None
        return snr

    def run(
        self,
        req: RunRequest,
        raw_cfg: Mapping[str, ExperimentCfgValue],
        *,
        context: RunContext,
    ) -> OneTonePowerDepRunResult:
        """Acquire a power/frequency map from accepted cfg values.

        req supplies captured device settings for assembly; context supplies
        live SoC/device handles, cancellation and plots for acquisition.
        Return this run's typed cfg and result. Acquisition errors propagate.
        """
        cfg = self.build_exp_cfg(raw_cfg, req)
        result = PowerDepExp().run(cfg, context=context)
        return RunRecord(cfg=cfg, result=result)

    def make_filename_stem(self, ctx: SessionEnv) -> str:
        return f"{ctx.res_name}_gain_{time.strftime('%H%M')}"
