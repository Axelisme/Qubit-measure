"""Explicit user-owned measure catalog; importing it performs no registration."""

from zcu_tools.gui.app.measure.registry import Registry
from zcu_tools.resources.entry.registry import ComponentRegistry

from zcu_lab.components import register_all as register_components
from zcu_lab.v2.fastflux.distortion.acc_phase.gui import (
    DEFINITION as ACC_PHASE_DEFINITION,
)
from zcu_lab.v2.fastflux.distortion.freq.gui import (
    DEFINITION as FAST_FLUX_FREQ_DEFINITION,
)
from zcu_lab.v2.fastflux.distortion.phase.gui import DEFINITION as PHASE_DEFINITION
from zcu_lab.v2.fastflux.mist.gui import DEFINITION as MIST_DEFINITION
from zcu_lab.v2.fastflux.t1.gui import DEFINITION as FAST_FLUX_T1_DEFINITION
from zcu_lab.v2.fastflux.twotone.gui import DEFINITION as TWO_TONE_DEFINITION
from zcu_lab.v2.jpa.auto_optimize.gui import DEFINITION as JPA_AUTO_OPTIMIZE_DEFINITION
from zcu_lab.v2.jpa.check.gui import DEFINITION as JPA_CHECK_DEFINITION
from zcu_lab.v2.jpa.flux.gui import DEFINITION as JPA_FLUX_DEFINITION
from zcu_lab.v2.jpa.flux_onetone.gui import DEFINITION as JPA_FLUX_ONE_TONE_DEFINITION
from zcu_lab.v2.jpa.freq.gui import DEFINITION as JPA_FREQ_DEFINITION
from zcu_lab.v2.jpa.power.gui import DEFINITION as JPA_POWER_DEFINITION
from zcu_lab.v2.lookback.gui import DEFINITION as LOOKBACK_DEFINITION
from zcu_lab.v2.mist.flux_dep.gui import DEFINITION as MIST_FLUX_DEP_DEFINITION
from zcu_lab.v2.mist.power_dep.drive_freq.gui import DEFINITION as DRIVE_FREQ_DEFINITION
from zcu_lab.v2.mist.power_dep.single_trace.gui import (
    DEFINITION as MIST_TRACE_DEFINITION,
)
from zcu_lab.v2.mux.amp_rabi.gui import DEFINITION as MUX_AMP_RABI_DEFINITION
from zcu_lab.v2.mux.len_rabi.gui import DEFINITION as MUX_LEN_RABI_DEFINITION
from zcu_lab.v2.mux.onetone.gui import DEFINITION as MUX_ONETONE_DEFINITION
from zcu_lab.v2.mux.onetone_flux_global.gui import (
    DEFINITION as ONETONE_FLUX_GLOBAL_DEFINITION,
)
from zcu_lab.v2.mux.onetone_flux_local.gui import (
    DEFINITION as ONETONE_FLUX_LOCAL_DEFINITION,
)
from zcu_lab.v2.mux.singleshot_ge.gui import DEFINITION as MUX_GE_DEFINITION
from zcu_lab.v2.mux.t1.gui import DEFINITION as MUX_T1_DEFINITION
from zcu_lab.v2.mux.t2echo.gui import DEFINITION as MUX_T2_ECHO_DEFINITION
from zcu_lab.v2.mux.t2ramsey.gui import DEFINITION as MUX_T2_RAMSEY_DEFINITION
from zcu_lab.v2.mux.twotone_flux_global.gui import (
    DEFINITION as TWOTONE_FLUX_GLOBAL_DEFINITION,
)
from zcu_lab.v2.mux.twotone_flux_local.gui import (
    DEFINITION as TWOTONE_FLUX_LOCAL_DEFINITION,
)
from zcu_lab.v2.mux.twotone_spectrum.gui import DEFINITION as MUX_FREQ_DEFINITION
from zcu_lab.v2.onetone.flux_dep.gui import DEFINITION as ONE_TONE_FLUX_DEP_DEFINITION
from zcu_lab.v2.onetone.freq.gui import DEFINITION as ONE_TONE_FREQ_DEFINITION
from zcu_lab.v2.onetone.power_dep.gui import DEFINITION as ONE_TONE_POWER_DEP_DEFINITION
from zcu_lab.v2.onetone.range_find.gui import DEFINITION as RANGE_FIND_DEFINITION
from zcu_lab.v2.singleshot.ac_stark.gui import DEFINITION as SS_AC_STARK_DEFINITION
from zcu_lab.v2.singleshot.amp_rabi.gui import DEFINITION as SS_AMP_RABI_DEFINITION
from zcu_lab.v2.singleshot.check.gui import DEFINITION as CHECK_DEFINITION
from zcu_lab.v2.singleshot.ge.gui import DEFINITION as GE_DEFINITION
from zcu_lab.v2.singleshot.len_rabi.gui import DEFINITION as SS_LEN_RABI_DEFINITION
from zcu_lab.v2.singleshot.mist.freq.gui import DEFINITION as MIST_FREQ_DEFINITION
from zcu_lab.v2.singleshot.mist.power.gui import DEFINITION as MIST_POWER_DEFINITION
from zcu_lab.v2.singleshot.mist.power_freq.gui import (
    DEFINITION as MIST_POWER_FREQ_DEFINITION,
)
from zcu_lab.v2.singleshot.mist.pre_freq.gui import (
    DEFINITION as MIST_PRE_FREQ_DEFINITION,
)
from zcu_lab.v2.singleshot.reset_check.gui import (
    DEFINITION as SS_RESET_CHECK_DEFINITION,
)
from zcu_lab.v2.singleshot.t1.t1.gui import DEFINITION as SS_T1_DEFINITION
from zcu_lab.v2.singleshot.t1.t1_with_tone.gui import (
    DEFINITION as SS_T1_TONE_DEFINITION,
)
from zcu_lab.v2.singleshot.t1.t1_with_tone_sweep.gui import (
    FREQ_DEFINITION as SS_T1_TONE_SWEEP_FREQ_DEFINITION,
)
from zcu_lab.v2.singleshot.t1.t1_with_tone_sweep.gui import (
    GAIN_DEFINITION as SS_T1_TONE_SWEEP_GAIN_DEFINITION,
)
from zcu_lab.v2.two_qubit.chevron.gui import DEFINITION as CHEVRON_DEFINITION
from zcu_lab.v2.two_qubit.conditional_phase.gui import (
    DEFINITION as CONDITIONAL_PHASE_DEFINITION,
)
from zcu_lab.v2.two_qubit.conditional_spectroscopy.gui import (
    DEFINITION as SPECTROSCOPY_DEFINITION,
)
from zcu_lab.v2.two_qubit.constructive_phase.gui import (
    DEFINITION as CONSTRUCTIVE_PHASE_DEFINITION,
)
from zcu_lab.v2.two_qubit.cz_amplitude.gui import DEFINITION as CZAMPLITUDE_DEFINITION
from zcu_lab.v2.two_qubit.idle_zz.gui import DEFINITION as IDLE_ZZ_DEFINITION
from zcu_lab.v2.two_qubit.joint_readout.gui import (
    DEFINITION as JOINT_READOUT_DEFINITION,
)
from zcu_lab.v2.two_qubit.local_z.gui import DEFINITION as LOCAL_Z_DEFINITION
from zcu_lab.v2.two_qubit.rb.gui import DEFINITION as TWO_QUBIT_RB_DEFINITION
from zcu_lab.v2.twotone.ac_stark.gui import DEFINITION as AC_STARK_DEFINITION
from zcu_lab.v2.twotone.ac_stark.gui import (
    RAMSEY_DEFINITION as AC_STARK_RAMSEY_DEFINITION,
)
from zcu_lab.v2.twotone.allxy.gui import DEFINITION as ALL_XY_DEFINITION
from zcu_lab.v2.twotone.ckp.gui import DEFINITION as CKP_DEFINITION
from zcu_lab.v2.twotone.dispersive.gui import DEFINITION as DISPERSIVE_DEFINITION
from zcu_lab.v2.twotone.fluxdep.gui import DEFINITION as FLUX_DEP_DEFINITION
from zcu_lab.v2.twotone.freq.gui import DEFINITION as FREQ_DEFINITION
from zcu_lab.v2.twotone.irb.gui import DEFINITION as IRB_DEFINITION
from zcu_lab.v2.twotone.power_dep.gui import DEFINITION as POWER_DEP_DEFINITION
from zcu_lab.v2.twotone.rabi.amp_rabi.gui import DEFINITION as AMP_RABI_DEFINITION
from zcu_lab.v2.twotone.rabi.len_rabi.gui import DEFINITION as LEN_RABI_DEFINITION
from zcu_lab.v2.twotone.rb.gui import DEFINITION as RB_DEFINITION
from zcu_lab.v2.twotone.reset.bath.freq.gui import (
    DEFINITION as BATH_FREQ_GAIN_DEFINITION,
)
from zcu_lab.v2.twotone.reset.bath.length.gui import (
    DEFINITION as BATH_LENGTH_DEFINITION,
)
from zcu_lab.v2.twotone.reset.bath.phase.gui import DEFINITION as BATH_PHASE_DEFINITION
from zcu_lab.v2.twotone.reset.dual_tone.freq.gui import (
    DEFINITION as DUAL_TONE_FREQ_DEFINITION,
)
from zcu_lab.v2.twotone.reset.dual_tone.length.gui import (
    DEFINITION as DUAL_TONE_LENGTH_DEFINITION,
)
from zcu_lab.v2.twotone.reset.dual_tone.power.gui import (
    DEFINITION as DUAL_TONE_POWER_DEFINITION,
)
from zcu_lab.v2.twotone.reset.rabi_check.gui import DEFINITION as RABI_CHECK_DEFINITION
from zcu_lab.v2.twotone.reset.single_tone.freq.gui import (
    DEFINITION as SINGLE_TONE_FREQ_DEFINITION,
)
from zcu_lab.v2.twotone.reset.single_tone.length.gui import (
    DEFINITION as SINGLE_TONE_LENGTH_DEFINITION,
)
from zcu_lab.v2.twotone.ro_optimize.auto_optimize.gui import (
    DEFINITION as RO_OPT_AUTO_DEFINITION,
)
from zcu_lab.v2.twotone.ro_optimize.freq.gui import DEFINITION as RO_OPT_FREQ_DEFINITION
from zcu_lab.v2.twotone.ro_optimize.freq_gain.gui import (
    DEFINITION as RO_OPT_FREQ_GAIN_DEFINITION,
)
from zcu_lab.v2.twotone.ro_optimize.length.gui import (
    DEFINITION as RO_OPT_LENGTH_DEFINITION,
)
from zcu_lab.v2.twotone.ro_optimize.power.gui import (
    DEFINITION as RO_OPT_POWER_DEFINITION,
)
from zcu_lab.v2.twotone.time_domain.cpmg.gui import DEFINITION as CPMG_DEFINITION
from zcu_lab.v2.twotone.time_domain.t1.gui import DEFINITION as T1_DEFINITION
from zcu_lab.v2.twotone.time_domain.t2echo.gui import DEFINITION as T2_ECHO_DEFINITION
from zcu_lab.v2.twotone.time_domain.t2ramsey.gui import (
    DEFINITION as T2_RAMSEY_DEFINITION,
)
from zcu_lab.v2.twotone.zigzag.gui import DEFINITION as ZIG_ZAG_DEFINITION
from zcu_lab.v2.twotone.zigzag_sweep.gui import (
    FREQ_DEFINITION as ZIG_ZAG_SCAN_FREQ_DEFINITION,
)
from zcu_lab.v2.twotone.zigzag_sweep.gui import (
    GAIN_DEFINITION as ZIG_ZAG_SCAN_GAIN_DEFINITION,
)

ADAPTERS: dict[str, object] = {
    "lookback": LOOKBACK_DEFINITION,
    "onetone/freq": ONE_TONE_FREQ_DEFINITION,
    "onetone/power_dep": ONE_TONE_POWER_DEP_DEFINITION,
    "onetone/flux_dep": ONE_TONE_FLUX_DEP_DEFINITION,
    "twotone/freq": FREQ_DEFINITION,
    "twotone/flux_dep": FLUX_DEP_DEFINITION,
    "twotone/power_dep": POWER_DEP_DEFINITION,
    "twotone/rabi/len_rabi": LEN_RABI_DEFINITION,
    "twotone/rabi/amp_rabi": AMP_RABI_DEFINITION,
    "twotone/t1": T1_DEFINITION,
    "twotone/t2ramsey": T2_RAMSEY_DEFINITION,
    "twotone/t2echo": T2_ECHO_DEFINITION,
    "singleshot/ge": GE_DEFINITION,
    "singleshot/check": CHECK_DEFINITION,
    "singleshot/len_rabi": SS_LEN_RABI_DEFINITION,
    "singleshot/amp_rabi": SS_AMP_RABI_DEFINITION,
    "singleshot/t1": SS_T1_DEFINITION,
    "singleshot/ac_stark": SS_AC_STARK_DEFINITION,
    "singleshot/reset_check": SS_RESET_CHECK_DEFINITION,
    "singleshot/t1_tone": SS_T1_TONE_DEFINITION,
    "singleshot/t1_tone_sweep_gain": SS_T1_TONE_SWEEP_GAIN_DEFINITION,
    "singleshot/t1_tone_sweep_freq": SS_T1_TONE_SWEEP_FREQ_DEFINITION,
    "twotone/dispersive": DISPERSIVE_DEFINITION,
    "twotone/ac_stark": AC_STARK_DEFINITION,
    "twotone/ac_stark_ramsey": AC_STARK_RAMSEY_DEFINITION,
    "twotone/ckp": CKP_DEFINITION,
    "twotone/cpmg": CPMG_DEFINITION,
    "twotone/allxy": ALL_XY_DEFINITION,
    "twotone/zigzag": ZIG_ZAG_DEFINITION,
    "twotone/zigzag_scan/gain": ZIG_ZAG_SCAN_GAIN_DEFINITION,
    "twotone/zigzag_scan/freq": ZIG_ZAG_SCAN_FREQ_DEFINITION,
    "twotone/rb": RB_DEFINITION,
    "twotone/irb": IRB_DEFINITION,
    "twotone/ro_optimize/freq": RO_OPT_FREQ_DEFINITION,
    "twotone/ro_optimize/power": RO_OPT_POWER_DEFINITION,
    "twotone/ro_optimize/length": RO_OPT_LENGTH_DEFINITION,
    "twotone/ro_optimize/freq_gain": RO_OPT_FREQ_GAIN_DEFINITION,
    "twotone/ro_optimize/auto": RO_OPT_AUTO_DEFINITION,
    "twotone/reset/single_tone/freq": SINGLE_TONE_FREQ_DEFINITION,
    "twotone/reset/single_tone/length": SINGLE_TONE_LENGTH_DEFINITION,
    "twotone/reset/dual_tone/freq": DUAL_TONE_FREQ_DEFINITION,
    "twotone/reset/dual_tone/length": DUAL_TONE_LENGTH_DEFINITION,
    "twotone/reset/dual_tone/power": DUAL_TONE_POWER_DEFINITION,
    "twotone/reset/bath/freq_gain": BATH_FREQ_GAIN_DEFINITION,
    "twotone/reset/bath/length": BATH_LENGTH_DEFINITION,
    "twotone/reset/bath/phase": BATH_PHASE_DEFINITION,
    "twotone/reset/check": RABI_CHECK_DEFINITION,
    "mux/onetone": MUX_ONETONE_DEFINITION,
    "mux/onetone_flux_global": ONETONE_FLUX_GLOBAL_DEFINITION,
    "mux/onetone_flux_local": ONETONE_FLUX_LOCAL_DEFINITION,
    "mux/twotone_flux_global": TWOTONE_FLUX_GLOBAL_DEFINITION,
    "mux/twotone_flux_local": TWOTONE_FLUX_LOCAL_DEFINITION,
    "onetone/range_find": RANGE_FIND_DEFINITION,
    "mux/twotone_spectrum": MUX_FREQ_DEFINITION,
    "mux/len_rabi": MUX_LEN_RABI_DEFINITION,
    "mux/amp_rabi": MUX_AMP_RABI_DEFINITION,
    "mux/t1": MUX_T1_DEFINITION,
    "mux/t2ramsey": MUX_T2_RAMSEY_DEFINITION,
    "mux/t2echo": MUX_T2_ECHO_DEFINITION,
    "mux/singleshot_ge": MUX_GE_DEFINITION,
    "jpa/flux_onetone": JPA_FLUX_ONE_TONE_DEFINITION,
    "jpa/freq": JPA_FREQ_DEFINITION,
    "jpa/flux": JPA_FLUX_DEFINITION,
    "jpa/power": JPA_POWER_DEFINITION,
    "jpa/auto_optimize": JPA_AUTO_OPTIMIZE_DEFINITION,
    "jpa/check": JPA_CHECK_DEFINITION,
    "two_qubit/joint_readout": JOINT_READOUT_DEFINITION,
    "two_qubit/conditional_spectroscopy": SPECTROSCOPY_DEFINITION,
    "two_qubit/chevron": CHEVRON_DEFINITION,
    "two_qubit/cz_amplitude": CZAMPLITUDE_DEFINITION,
    "two_qubit/conditional_phase": CONDITIONAL_PHASE_DEFINITION,
    "two_qubit/local_z": LOCAL_Z_DEFINITION,
    "two_qubit/idle_zz": IDLE_ZZ_DEFINITION,
    "two_qubit/constructive_phase": CONSTRUCTIVE_PHASE_DEFINITION,
    "two_qubit/rb": TWO_QUBIT_RB_DEFINITION,
    "fastflux/twotone": TWO_TONE_DEFINITION,
    "fastflux/t1": FAST_FLUX_T1_DEFINITION,
    "fastflux/distortion/freq": FAST_FLUX_FREQ_DEFINITION,
    "fastflux/distortion/phase": PHASE_DEFINITION,
    "fastflux/distortion/acc_phase": ACC_PHASE_DEFINITION,
    "fastflux/mist": MIST_DEFINITION,
    "mist/flux_dep": MIST_FLUX_DEP_DEFINITION,
    "mist/power_dep/drive_freq": DRIVE_FREQ_DEFINITION,
    "mist/power_dep/single_trace": MIST_TRACE_DEFINITION,
    "singleshot/mist/freq": MIST_FREQ_DEFINITION,
    "singleshot/mist/power": MIST_POWER_DEFINITION,
    "singleshot/mist/power_freq": MIST_POWER_FREQ_DEFINITION,
    "singleshot/mist/pre_freq": MIST_PRE_FREQ_DEFINITION,
}

# Core-derived forms of migrated adapters, registered with their adapter names.


def register_all(
    registry: Registry,
    *,
    components: ComponentRegistry | None = None,
) -> None:
    """Register adapters into the caller-owned registry.

    The registry already owns its injected persistent template definitions.
    Library creation reads ``registry.templates`` directly.
    Pass the shared ComponentRegistry as ``components`` at startup to register
    component models and roles. Omit this startup receiver on reload to
    preserve its identity. Duplicate adapters raise ValueError.
    Registration does not build forms or validate templates; Open/Reset does.
    """
    if components is not None:
        register_components(components)
    for name, definition in ADAPTERS.items():
        registry.register(name, definition)
