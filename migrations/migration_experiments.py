"""Fixed native declarations for offline migration, never a runtime registry."""

from functools import partial

import numpy as np

from migrations.experiment_declaration import (
    declare_migration_experiment as _declaration,
)
from migrations.legacy_labber import LegacyAxisSchema, LegacyVariableSchema
from migrations.legacy_run_values import (
    MUX_COHERENCE_SCHEMA,
    project_mux_coherence,
    project_unknown_detune,
)
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
from zcu_lab.v2.mux.amp_rabi import core as mux_amp_rabi
from zcu_lab.v2.mux.len_rabi import core as mux_len_rabi
from zcu_lab.v2.mux.onetone import core as mux_onetone
from zcu_lab.v2.mux.singleshot_ge import core as mux_singleshot_ge
from zcu_lab.v2.mux.t1 import core as mux_t1
from zcu_lab.v2.mux.t2echo import core as mux_t2echo
from zcu_lab.v2.mux.t2ramsey import core as mux_t2ramsey
from zcu_lab.v2.mux.twotone_spectrum import core as mux_twotone_spectrum
from zcu_lab.v2.onetone.flux_dep import core as onetone_flux_dep
from zcu_lab.v2.onetone.freq import core as onetone_freq
from zcu_lab.v2.onetone.power_dep import core as onetone_power_dep
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
from zcu_lab.v2.singleshot.t1.t1_with_tone_sweep import core as singleshot_t1_tone_sweep
from zcu_lab.v2.two_qubit.chevron import core as ftf_chevron
from zcu_lab.v2.two_qubit.conditional_phase import core as ftf_conditional_phase
from zcu_lab.v2.two_qubit.conditional_spectroscopy import (
    core as ftf_conditional_spectroscopy,
)
from zcu_lab.v2.two_qubit.constructive_phase import core as ftf_constructive_phase
from zcu_lab.v2.two_qubit.cz_amplitude import core as ftf_cz_amplitude
from zcu_lab.v2.two_qubit.idle_zz import core as ftf_idle_zz
from zcu_lab.v2.two_qubit.joint_readout import core as ftf_joint_readout
from zcu_lab.v2.two_qubit.local_z import core as ftf_local_z
from zcu_lab.v2.two_qubit.rb import core as ftf_rb
from zcu_lab.v2.twotone.ac_stark import core as twotone_ac_stark
from zcu_lab.v2.twotone.ac_stark import ramsey_core as twotone_ac_stark_ramsey
from zcu_lab.v2.twotone.allxy import core as twotone_allxy
from zcu_lab.v2.twotone.ckp import core as twotone_ckp
from zcu_lab.v2.twotone.dispersive import core as twotone_dispersive
from zcu_lab.v2.twotone.flux_dep import core as twotone_fluxdep
from zcu_lab.v2.twotone.freq import core as twotone_freq
from zcu_lab.v2.twotone.power_dep import core as twotone_power_dep
from zcu_lab.v2.twotone.rabi.amp_rabi import core as twotone_rabi_amp_rabi
from zcu_lab.v2.twotone.rabi.len_rabi import core as twotone_rabi_len_rabi
from zcu_lab.v2.twotone.reset.bath.freq_gain import core as twotone_reset_bath_freq
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
from zcu_lab.v2.twotone.ro_optimize.auto import (
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
from zcu_lab.v2.twotone.time_domain.t1 import (
    scan_t1_with_tone as twotone_time_domain_scan_t1_with_tone,
)
from zcu_lab.v2.twotone.time_domain.t1 import (
    t1_with_tone as twotone_time_domain_t1_with_tone,
)
from zcu_lab.v2.twotone.time_domain.t2echo import core as twotone_time_domain_t2echo
from zcu_lab.v2.twotone.time_domain.t2ramsey import core as twotone_time_domain_t2ramsey
from zcu_lab.v2.twotone.zigzag import core as twotone_zigzag
from zcu_lab.v2.twotone.zigzag_scan import core as twotone_zigzag_sweep

# Historical disk schemas precede shot moments and run values. Neither the
# current native variables nor the Labber export projection defines these logs.
_SINGLE_COHERENCE_SCHEMA = (
    LegacyVariableSchema(
        variable="data",
        axes=(LegacyAxisSchema(name="Time", unit="s", dtype=np.dtype("float64")),),
        signal_name="Signal",
        signal_unit="a.u.",
        signal_dtype=np.dtype("complex128"),
    ),
)

_MIST_GAIN_SCHEMA = (
    LegacyVariableSchema(
        variable="data",
        axes=(
            LegacyAxisSchema(
                name="Drive Power", unit="a.u.", dtype=np.dtype("float64")
            ),
        ),
        signal_name="Signal",
        signal_unit="a.u.",
        signal_dtype=np.dtype("complex128"),
    ),
)

# Explicit core declarations, including notebook-only experiments. No discovery,
# import of GUI definitions, mutable registry or kind registration occurs here.
# Changes to this declaration surface also bump the concrete mapping revisions.
MIGRATION_EXPERIMENTS = (
    _declaration(fastflux_distortion_acc_phase.AccPhaseExp.AXES_SPEC),
    _declaration(fastflux_distortion_freq.FreqExp.AXES_SPEC),
    _declaration(fastflux_distortion_phase.PhaseExp.AXES_SPEC),
    _declaration(fastflux_mist.MistExp.AXES_SPEC),
    _declaration(fastflux_t1.T1Exp.AXES_SPEC),
    _declaration(fastflux_twotone.TwoToneExp.AXES_SPEC),
    _declaration(ftf_chevron.ChevronExp.AXES_SPEC),
    _declaration(ftf_conditional_phase.ConditionalPhaseExp.AXES_SPEC),
    _declaration(ftf_conditional_spectroscopy.SpectroscopyExp.AXES_SPEC),
    _declaration(ftf_constructive_phase.ConstructivePhaseExp.AXES_SPEC),
    _declaration(ftf_cz_amplitude.CZAmplitudeExp.AXES_SPEC),
    _declaration(ftf_idle_zz.IdleZZExp.AXES_SPEC),
    _declaration(ftf_joint_readout.JointReadoutExp.AXES_SPEC),
    _declaration(ftf_local_z.LocalZExp.AXES_SPEC),
    _declaration(ftf_rb.RBExp.AXES_SPEC),
    _declaration(jpa_auto_optimize.AutoOptimizeExp.AXES_SPEC),
    _declaration(jpa_check.CheckExp.AXES_SPEC),
    _declaration(jpa_flux.FluxExp.AXES_SPEC),
    _declaration(jpa_flux_onetone.OneToneFluxExp.AXES_SPEC),
    _declaration(jpa_freq.FreqExp.AXES_SPEC),
    _declaration(jpa_power.PowerExp.AXES_SPEC),
    _declaration(lookback.LookbackExp.AXES_SPEC),
    _declaration(mist_flux_dep.FluxDepExp.AXES_SPEC),
    _declaration(mist_power_dep_drive_freq.DriveFreqExp.AXES_SPEC),
    _declaration(
        mist_power_dep_single_trace.PowerDepExp.AXES_SPEC,
        schemas=_MIST_GAIN_SCHEMA,
    ),
    _declaration(mux_amp_rabi.MuxAmpRabiExp.AXES_SPEC),
    _declaration(mux_twotone_spectrum.MuxFreqExp.AXES_SPEC, source_tag="mux/freq"),
    _declaration(mux_len_rabi.MuxLenRabiExp.AXES_SPEC),
    _declaration(mux_onetone.MuxOnetoneExp.AXES_SPEC, source_tag="mux/onetone"),
    _declaration(mux_singleshot_ge.MuxGEExp.AXES_SPEC),
    _declaration(mux_t1.MuxT1Exp.AXES_SPEC),
    _declaration(
        mux_t2echo.MuxT2EchoExp.AXES_SPEC,
        source_tag="mux/time_domain/t2echo",
        schemas=MUX_COHERENCE_SCHEMA,
        project_payload=partial(
            project_mux_coherence, spec=mux_t2echo.MuxT2EchoExp.AXES_SPEC
        ),
    ),
    _declaration(
        mux_t2ramsey.MuxT2RamseyExp.AXES_SPEC,
        source_tag="mux/time_domain/t2ramsey",
        schemas=MUX_COHERENCE_SCHEMA,
        project_payload=partial(
            project_mux_coherence, spec=mux_t2ramsey.MuxT2RamseyExp.AXES_SPEC
        ),
    ),
    _declaration(onetone_flux_dep.FluxDepExp.AXES_SPEC),
    _declaration(onetone_freq.FreqExp.AXES_SPEC),
    _declaration(onetone_power_dep.PowerDepExp.AXES_SPEC),
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
    _declaration(singleshot_t1_tone_sweep.T1WithToneSweepExp.AXES_SPEC),
    _declaration(twotone_ac_stark.AcStarkExp.AXES_SPEC),
    _declaration(twotone_ac_stark_ramsey.AcStarkRamseyExp.AXES_SPEC),
    _declaration(twotone_allxy.AllXY_Exp.AXES_SPEC),
    _declaration(twotone_ckp.CKP_Exp.AXES_SPEC),
    _declaration(twotone_dispersive.DispersiveExp.AXES_SPEC),
    _declaration(twotone_fluxdep.FreqFluxExp.AXES_SPEC),
    _declaration(twotone_freq.FreqExp.AXES_SPEC),
    _declaration(twotone_power_dep.PowerExp.AXES_SPEC),
    _declaration(twotone_rabi_amp_rabi.AmpRabiExp.AXES_SPEC),
    _declaration(twotone_rabi_len_rabi.LenRabiExp.AXES_SPEC),
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
    _declaration(twotone_time_domain_t1_with_tone.T1WithToneExp.AXES_SPEC),
    _declaration(twotone_time_domain_scan_t1_with_tone.ScanT1WithToneExp.AXES_SPEC),
    _declaration(
        twotone_time_domain_t2echo.T2EchoExp.AXES_SPEC,
        source_tag="twotone/ge/t2echo",
        schemas=_SINGLE_COHERENCE_SCHEMA,
        project_payload=partial(
            project_unknown_detune, spec=twotone_time_domain_t2echo.T2EchoExp.AXES_SPEC
        ),
    ),
    _declaration(
        twotone_time_domain_t2ramsey.T2RamseyExp.AXES_SPEC,
        schemas=_SINGLE_COHERENCE_SCHEMA,
        project_payload=partial(
            project_unknown_detune,
            spec=twotone_time_domain_t2ramsey.T2RamseyExp.AXES_SPEC,
        ),
    ),
    _declaration(twotone_zigzag.ZigZagExp.AXES_SPEC),
    _declaration(twotone_zigzag_sweep.ZigZagScanExp.AXES_SPEC),
)
