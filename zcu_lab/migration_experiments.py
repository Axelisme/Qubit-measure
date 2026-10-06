"""Fixed native declarations for offline migration, never a runtime registry."""

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from zcu_tools.datafile import CfgSnapshot, VariableSchema, validate_cfg_snapshot
from zcu_tools.experiment import (
    AxesSpec,
    ExpCfgModel,
    GroupedAxesSpec,
    load_run,
    native_schemas,
)

from zcu_lab.v2.fake.freq import core as fake_freq
from zcu_lab.v2.fake.signal import core as fake_signal
from zcu_lab.v2.fastflux.distortion.acc_phase import (
    core as fastflux_distortion_acc_phase,
)
from zcu_lab.v2.fastflux.distortion.freq import core as fastflux_distortion_freq
from zcu_lab.v2.fastflux.distortion.phase import core as fastflux_distortion_phase
from zcu_lab.v2.fastflux.mist import core as fastflux_mist
from zcu_lab.v2.fastflux.t1 import core as fastflux_t1
from zcu_lab.v2.fastflux.twotone import core as fastflux_twotone
from zcu_lab.v2.jpa.auto_optimize import core as jpa_auto_optimize
from zcu_lab.v2.jpa.check import core as jpa_check
from zcu_lab.v2.jpa.flux import core as jpa_flux
from zcu_lab.v2.jpa.flux_onetone import core as jpa_flux_onetone
from zcu_lab.v2.jpa.freq import core as jpa_freq
from zcu_lab.v2.jpa.power import core as jpa_power
from zcu_lab.v2.lookback import core as lookback
from zcu_lab.v2.mist.flux_dep import core as mist_flux_dep
from zcu_lab.v2.mist.power_dep.drive_freq import core as mist_power_dep_drive_freq
from zcu_lab.v2.mist.power_dep.single_trace import core as mist_power_dep_single_trace
from zcu_lab.v2.onetone.flux_dep import core as onetone_flux_dep
from zcu_lab.v2.onetone.freq import core as onetone_freq
from zcu_lab.v2.onetone.power_dep import core as onetone_power_dep
from zcu_lab.v2.onetone.sa import core as onetone_sa
from zcu_lab.v2.singleshot.ac_stark import core as singleshot_ac_stark
from zcu_lab.v2.singleshot.amp_rabi import core as singleshot_amp_rabi
from zcu_lab.v2.singleshot.check import core as singleshot_check
from zcu_lab.v2.singleshot.ge import core as singleshot_ge
from zcu_lab.v2.singleshot.len_rabi import core as singleshot_len_rabi
from zcu_lab.v2.singleshot.mist.freq import core as singleshot_mist_freq
from zcu_lab.v2.singleshot.mist.power import core as singleshot_mist_power
from zcu_lab.v2.singleshot.mist.power_freq import core as singleshot_mist_power_freq
from zcu_lab.v2.singleshot.mist.pre_freq import core as singleshot_mist_pre_freq
from zcu_lab.v2.singleshot.reset_check import core as singleshot_reset_check
from zcu_lab.v2.singleshot.t1.t1 import core as singleshot_t1_t1
from zcu_lab.v2.singleshot.t1.t1_with_tone import core as singleshot_t1_t1_with_tone
from zcu_lab.v2.twotone.ac_stark import core as twotone_ac_stark
from zcu_lab.v2.twotone.allxy import core as twotone_allxy
from zcu_lab.v2.twotone.ckp import core as twotone_ckp
from zcu_lab.v2.twotone.dispersive import core as twotone_dispersive
from zcu_lab.v2.twotone.fluxdep import core as twotone_fluxdep
from zcu_lab.v2.twotone.freq import core as twotone_freq
from zcu_lab.v2.twotone.power_dep import core as twotone_power_dep
from zcu_lab.v2.twotone.rabi.amp_rabi import core as twotone_rabi_amp_rabi
from zcu_lab.v2.twotone.rabi.len_rabi import core as twotone_rabi_len_rabi
from zcu_lab.v2.twotone.rb import core as twotone_rb
from zcu_lab.v2.twotone.reset.bath.freq import core as twotone_reset_bath_freq
from zcu_lab.v2.twotone.reset.bath.length import core as twotone_reset_bath_length
from zcu_lab.v2.twotone.reset.bath.phase import core as twotone_reset_bath_phase
from zcu_lab.v2.twotone.reset.dual_tone.freq import core as twotone_reset_dual_tone_freq
from zcu_lab.v2.twotone.reset.dual_tone.length import (
    core as twotone_reset_dual_tone_length,
)
from zcu_lab.v2.twotone.reset.dual_tone.power import (
    core as twotone_reset_dual_tone_power,
)
from zcu_lab.v2.twotone.reset.rabi_check import core as twotone_reset_rabi_check
from zcu_lab.v2.twotone.reset.single_tone.freq import (
    core as twotone_reset_single_tone_freq,
)
from zcu_lab.v2.twotone.reset.single_tone.length import (
    core as twotone_reset_single_tone_length,
)
from zcu_lab.v2.twotone.ro_optimize.auto_optimize import (
    core as twotone_ro_optimize_auto_optimize,
)
from zcu_lab.v2.twotone.ro_optimize.freq import core as twotone_ro_optimize_freq
from zcu_lab.v2.twotone.ro_optimize.freq_gain import (
    core as twotone_ro_optimize_freq_gain,
)
from zcu_lab.v2.twotone.ro_optimize.length import core as twotone_ro_optimize_length
from zcu_lab.v2.twotone.ro_optimize.power import core as twotone_ro_optimize_power
from zcu_lab.v2.twotone.time_domain.cpmg import core as twotone_time_domain_cpmg
from zcu_lab.v2.twotone.time_domain.t1 import core as twotone_time_domain_t1
from zcu_lab.v2.twotone.time_domain.t2echo import core as twotone_time_domain_t2echo
from zcu_lab.v2.twotone.time_domain.t2ramsey import core as twotone_time_domain_t2ramsey
from zcu_lab.v2.twotone.zigzag import core as twotone_zigzag
from zcu_lab.v2.twotone.zigzag_sweep import core as twotone_zigzag_sweep

ResultT = TypeVar("ResultT")
CfgT = TypeVar("CfgT", bound=ExpCfgModel)


@dataclass(frozen=True)
class MigrationExperiment:
    """Bind a concrete typed reader to its generic schema without erasing its cfg.

    source_tag and cfg_type are the exact historical experiment/cfg identities;
    neither determines the other. native_tag is the canonical spec tag written
    into native RunMetadata; it may differ from source_tag only by an explicit
    migration declaration. schemas is the generic
    disk-unit variable declaration. validate_cfg checks a historical CfgSnapshot
    for storage format, this pair's cfg_type, major version and concrete model.
    It returns None without changing raw values and raises ValueError for
    unconvertible cfg.
    validate_native reads an exact native Path
    with the corresponding typed load_run, returns None and propagates failures.
    These bindings perform no acquisition, registration or live-context capture.
    """

    source_tag: str
    native_tag: str
    cfg_type: str
    schemas: tuple[VariableSchema, ...]
    validate_cfg: Callable[[CfgSnapshot], None]
    validate_native: Callable[[Path], None]


def _declaration(
    spec: AxesSpec[ResultT, CfgT] | GroupedAxesSpec[ResultT, CfgT] | None,
    *,
    source_tag: str | None = None,
) -> MigrationExperiment:
    if spec is None:
        raise ValueError("Missing native experiment declaration")

    def validate_cfg(snapshot: CfgSnapshot) -> None:
        validate_cfg_snapshot(snapshot)
        if snapshot.cfg_type != spec.cfg_type.__name__:
            raise ValueError(f"cfg_type must be {spec.cfg_type.__name__}")
        if int(snapshot.schema_version.split(".")[0]) != int(
            spec.cfg_schema_version.split(".")[0]
        ):
            raise ValueError(
                f"cfg_schema_version major must match {spec.cfg_schema_version}"
            )
        spec.cfg_type.model_validate(deepcopy(snapshot.values), extra="ignore")

    def validate(path: Path) -> None:
        load_run(path, spec=spec)

    return MigrationExperiment(
        source_tag=spec.tag if source_tag is None else source_tag,
        native_tag=spec.tag,
        cfg_type=spec.cfg_type.__name__,
        schemas=native_schemas(spec),
        validate_cfg=validate_cfg,
        validate_native=validate,
    )


# Explicit core declarations, including notebook-only experiments. No discovery,
# import of GUI definitions, mutable registry or kind registration occurs here.
# Changes to this declaration surface also bump the concrete mapping revisions.
MIGRATION_EXPERIMENTS = (
    _declaration(fake_freq.FakeFreqExp.AXES_SPEC),
    _declaration(fake_signal.FakeExp.AXES_SPEC),
    _declaration(fastflux_distortion_acc_phase.AccPhaseExp.AXES_SPEC),
    _declaration(fastflux_distortion_freq.FreqExp.AXES_SPEC),
    _declaration(fastflux_distortion_phase.PhaseExp.AXES_SPEC),
    _declaration(fastflux_mist.MistExp.AXES_SPEC),
    _declaration(fastflux_t1.T1Exp.AXES_SPEC),
    _declaration(fastflux_twotone.TwoToneExp.AXES_SPEC),
    _declaration(jpa_auto_optimize.AutoOptimizeExp.AXES_SPEC),
    _declaration(jpa_check.CheckExp.AXES_SPEC),
    _declaration(jpa_flux.FluxExp.AXES_SPEC),
    _declaration(jpa_flux_onetone.OneToneFluxExp.AXES_SPEC),
    _declaration(jpa_freq.FreqExp.AXES_SPEC),
    _declaration(jpa_power.PowerExp.AXES_SPEC),
    _declaration(lookback.LookbackExp.AXES_SPEC),
    _declaration(mist_flux_dep.FluxDepExp.AXES_SPEC),
    _declaration(mist_power_dep_drive_freq.DriveFreqExp.AXES_SPEC),
    _declaration(mist_power_dep_single_trace.PowerDepExp.AXES_SPEC),
    _declaration(onetone_flux_dep.FluxDepExp.AXES_SPEC),
    _declaration(onetone_freq.FreqExp.AXES_SPEC),
    _declaration(onetone_power_dep.PowerDepExp.AXES_SPEC),
    _declaration(onetone_sa.SA_FreqExp.AXES_SPEC),
    _declaration(singleshot_ac_stark.AcStarkExp.AXES_SPEC),
    _declaration(singleshot_amp_rabi.AmpRabiExp.AXES_SPEC),
    _declaration(singleshot_check.CheckExp.AXES_SPEC),
    _declaration(singleshot_ge.GE_Exp.AXES_SPEC),
    _declaration(singleshot_len_rabi.LenRabiExp.AXES_SPEC),
    _declaration(singleshot_mist_freq.FreqDepExp.AXES_SPEC),
    _declaration(singleshot_mist_power.PowerExp.AXES_SPEC),
    _declaration(singleshot_mist_power_freq.FreqPowerExp.AXES_SPEC),
    _declaration(singleshot_mist_pre_freq.PreFreqExp.AXES_SPEC),
    _declaration(singleshot_reset_check.ResetCheckExp.AXES_SPEC),
    _declaration(singleshot_t1_t1.T1Exp.AXES_SPEC),
    _declaration(singleshot_t1_t1_with_tone.T1WithToneExp.AXES_SPEC),
    _declaration(twotone_ac_stark.AcStarkExp.AXES_SPEC),
    _declaration(twotone_ac_stark.AcStarkRamseyExp.AXES_SPEC),
    _declaration(twotone_allxy.AllXY_Exp.AXES_SPEC),
    _declaration(twotone_ckp.CKP_Exp.AXES_SPEC),
    _declaration(twotone_dispersive.DispersiveExp.AXES_SPEC),
    _declaration(twotone_fluxdep.FreqFluxExp.AXES_SPEC),
    _declaration(twotone_freq.FreqExp.AXES_SPEC),
    _declaration(twotone_power_dep.PowerExp.AXES_SPEC),
    _declaration(twotone_rabi_amp_rabi.AmpRabiExp.AXES_SPEC),
    _declaration(twotone_rabi_len_rabi.LenRabiExp.AXES_SPEC),
    _declaration(twotone_rb.RB_Exp.AXES_SPEC),
    _declaration(twotone_reset_bath_freq.FreqGainExp.AXES_SPEC),
    _declaration(twotone_reset_bath_length.LengthExp.AXES_SPEC),
    _declaration(twotone_reset_bath_phase.PhaseExp.AXES_SPEC),
    _declaration(twotone_reset_dual_tone_freq.FreqExp.AXES_SPEC),
    _declaration(twotone_reset_dual_tone_length.LengthExp.AXES_SPEC),
    _declaration(twotone_reset_dual_tone_power.PowerExp.AXES_SPEC),
    _declaration(twotone_reset_rabi_check.RabiCheckExp.AXES_SPEC),
    _declaration(twotone_reset_single_tone_freq.FreqExp.AXES_SPEC),
    _declaration(twotone_reset_single_tone_length.LengthExp.AXES_SPEC),
    _declaration(twotone_ro_optimize_auto_optimize.AutoOptExp.AXES_SPEC),
    _declaration(twotone_ro_optimize_freq.FreqExp.AXES_SPEC),
    _declaration(
        twotone_ro_optimize_freq_gain.FreqGainExp.AXES_SPEC,
        source_tag="twotone/ge/ro_optimize/freq",
    ),
    _declaration(twotone_ro_optimize_length.LengthExp.AXES_SPEC),
    _declaration(twotone_ro_optimize_power.PowerExp.AXES_SPEC),
    _declaration(twotone_time_domain_cpmg.CPMG_Exp.AXES_SPEC),
    _declaration(twotone_time_domain_t1.T1Exp.AXES_SPEC),
    _declaration(twotone_time_domain_t1.T1WithToneExp.AXES_SPEC),
    _declaration(twotone_time_domain_t1.ScanT1WithToneExp.AXES_SPEC),
    _declaration(twotone_time_domain_t2echo.T2EchoExp.AXES_SPEC),
    _declaration(twotone_time_domain_t2ramsey.T2RamseyExp.AXES_SPEC),
    _declaration(twotone_zigzag.ZigZagExp.AXES_SPEC),
    _declaration(twotone_zigzag_sweep.ZigZagScanExp.AXES_SPEC),
)
