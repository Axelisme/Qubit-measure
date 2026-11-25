"""Explicit user-owned measure catalog; importing it performs no registration."""

from collections.abc import Callable
from types import ModuleType

from zcu_tools.gui.app.measure.registry import Registry
from zcu_tools.resources.entry.registry import ComponentRegistry

from zcu_lab.components import register_all as register_components
from zcu_lab.v2.chain.amp_rabi import gui as chain_amp_rabi_gui
from zcu_lab.v2.chain.len_rabi import gui as chain_len_rabi_gui
from zcu_lab.v2.chain.t1 import gui as chain_t1_gui
from zcu_lab.v2.chain.t2echo import gui as chain_t2echo_gui
from zcu_lab.v2.chain.t2ramsey import gui as chain_t2ramsey_gui
from zcu_lab.v2.fastflux.distortion.acc_phase import gui as acc_phase_gui
from zcu_lab.v2.fastflux.distortion.freq import gui as distortion_freq_gui
from zcu_lab.v2.fastflux.distortion.phase import gui as distortion_phase_gui
from zcu_lab.v2.fastflux.mist import gui as mist_gui
from zcu_lab.v2.fastflux.t1 import gui as fastflux_t1_gui
from zcu_lab.v2.fastflux.twotone import gui as twotone_gui
from zcu_lab.v2.heralded.chevron import gui as heralded_chevron_gui
from zcu_lab.v2.heralded.conditional_phase import gui as heralded_conditional_phase_gui
from zcu_lab.v2.heralded.conditional_spectroscopy import (
    gui as heralded_conditional_spectroscopy_gui,
)
from zcu_lab.v2.heralded.cz_amplitude import gui as heralded_cz_amplitude_gui
from zcu_lab.v2.heralded.cz_optimize import gui as heralded_cz_optimize_gui
from zcu_lab.v2.heralded.idle_zz import gui as heralded_idle_zz_gui
from zcu_lab.v2.heralded.local_z import gui as heralded_local_z_gui
from zcu_lab.v2.heralded.rb import gui as heralded_rb_gui
from zcu_lab.v2.jpa.auto_optimize import gui as auto_optimize_gui
from zcu_lab.v2.jpa.check import gui as jpa_check_gui
from zcu_lab.v2.jpa.flux import gui as flux_gui
from zcu_lab.v2.jpa.flux_onetone import gui as flux_onetone_gui
from zcu_lab.v2.jpa.freq import gui as jpa_freq_gui
from zcu_lab.v2.jpa.power import gui as jpa_power_gui
from zcu_lab.v2.lookback import gui as lookback_gui
from zcu_lab.v2.mist.flux_dep import gui as mist_flux_dep_gui
from zcu_lab.v2.mist.power_dep.drive_freq import gui as drive_freq_gui
from zcu_lab.v2.mist.power_dep.single_trace import gui as single_trace_gui
from zcu_lab.v2.mux.amp_rabi import gui as mux_amp_rabi_gui
from zcu_lab.v2.mux.len_rabi import gui as mux_len_rabi_gui
from zcu_lab.v2.mux.onetone import gui as onetone_gui
from zcu_lab.v2.mux.onetone_flux_global import gui as onetone_flux_global_gui
from zcu_lab.v2.mux.onetone_flux_local import gui as onetone_flux_local_gui
from zcu_lab.v2.mux.singleshot_ge import gui as singleshot_ge_gui
from zcu_lab.v2.mux.t1 import gui as mux_t1_gui
from zcu_lab.v2.mux.t2echo import gui as mux_t2echo_gui
from zcu_lab.v2.mux.t2ramsey import gui as mux_t2ramsey_gui
from zcu_lab.v2.mux.twotone_flux_global import gui as twotone_flux_global_gui
from zcu_lab.v2.mux.twotone_flux_local import gui as twotone_flux_local_gui
from zcu_lab.v2.mux.twotone_spectrum import gui as twotone_spectrum_gui
from zcu_lab.v2.onetone.flux_dep import gui as onetone_flux_dep_gui
from zcu_lab.v2.onetone.freq import gui as onetone_freq_gui
from zcu_lab.v2.onetone.power_dep import gui as onetone_power_dep_gui
from zcu_lab.v2.onetone.range_find import gui as range_find_gui
from zcu_lab.v2.singleshot.ac_stark import gui as singleshot_ac_stark_gui
from zcu_lab.v2.singleshot.amp_rabi import gui as singleshot_amp_rabi_gui
from zcu_lab.v2.singleshot.check import gui as singleshot_check_gui
from zcu_lab.v2.singleshot.ge import gui as ge_gui
from zcu_lab.v2.singleshot.len_rabi import gui as singleshot_len_rabi_gui
from zcu_lab.v2.singleshot.mist.freq import gui as mist_freq_gui
from zcu_lab.v2.singleshot.mist.power import gui as mist_power_gui
from zcu_lab.v2.singleshot.mist.power_freq import gui as power_freq_gui
from zcu_lab.v2.singleshot.mist.pre_freq import gui as pre_freq_gui
from zcu_lab.v2.singleshot.qutrit_population import gui as qutrit_population_gui
from zcu_lab.v2.singleshot.reset_check import gui as reset_check_gui
from zcu_lab.v2.singleshot.t1 import gui as singleshot_t1_gui
from zcu_lab.v2.singleshot.t1_tone import gui as t1_tone_gui
from zcu_lab.v2.singleshot.t1_tone_sweep import gui as t1_tone_sweep_gui
from zcu_lab.v2.two_qubit.chevron import gui as two_qubit_chevron_gui
from zcu_lab.v2.two_qubit.conditional_phase import (
    gui as two_qubit_conditional_phase_gui,
)
from zcu_lab.v2.two_qubit.conditional_spectroscopy import (
    gui as two_qubit_conditional_spectroscopy_gui,
)
from zcu_lab.v2.two_qubit.constructive_phase import gui as constructive_phase_gui
from zcu_lab.v2.two_qubit.cz_amplitude import gui as two_qubit_cz_amplitude_gui
from zcu_lab.v2.two_qubit.cz_optimize import gui as two_qubit_cz_optimize_gui
from zcu_lab.v2.two_qubit.idle_zz import gui as two_qubit_idle_zz_gui
from zcu_lab.v2.two_qubit.joint_readout import gui as joint_readout_gui
from zcu_lab.v2.two_qubit.local_z import gui as two_qubit_local_z_gui
from zcu_lab.v2.two_qubit.rb import gui as two_qubit_rb_gui
from zcu_lab.v2.twotone.ac_stark import gui as twotone_ac_stark_gui
from zcu_lab.v2.twotone.ac_stark_ramsey import gui as ac_stark_ramsey_gui
from zcu_lab.v2.twotone.allxy import gui as allxy_gui
from zcu_lab.v2.twotone.chain import gui as chain_gui
from zcu_lab.v2.twotone.ckp import gui as ckp_gui
from zcu_lab.v2.twotone.dispersive import gui as dispersive_gui
from zcu_lab.v2.twotone.flux_dep import gui as twotone_flux_dep_gui
from zcu_lab.v2.twotone.freq import gui as twotone_freq_gui
from zcu_lab.v2.twotone.power_dep import gui as twotone_power_dep_gui
from zcu_lab.v2.twotone.rabi.amp_rabi import gui as rabi_amp_rabi_gui
from zcu_lab.v2.twotone.rabi.len_rabi import gui as rabi_len_rabi_gui
from zcu_lab.v2.twotone.raman_gf import gui as raman_gf_gui
from zcu_lab.v2.twotone.rb import gui as twotone_rb_gui
from zcu_lab.v2.twotone.reset.bath.freq_gain import gui as bath_freq_gain_gui
from zcu_lab.v2.twotone.reset.bath.length import gui as bath_length_gui
from zcu_lab.v2.twotone.reset.bath.phase import gui as bath_phase_gui
from zcu_lab.v2.twotone.reset.dual_tone.freq import gui as dual_tone_freq_gui
from zcu_lab.v2.twotone.reset.dual_tone.length import gui as dual_tone_length_gui
from zcu_lab.v2.twotone.reset.dual_tone.power import gui as dual_tone_power_gui
from zcu_lab.v2.twotone.reset.flux.gain import gui as gain_gui
from zcu_lab.v2.twotone.reset.flux.length import gui as flux_length_gui
from zcu_lab.v2.twotone.reset.rabi_check import gui as rabi_check_gui
from zcu_lab.v2.twotone.reset.single_tone.freq import gui as single_tone_freq_gui
from zcu_lab.v2.twotone.reset.single_tone.length import gui as single_tone_length_gui
from zcu_lab.v2.twotone.ro_optimize.auto import gui as auto_gui
from zcu_lab.v2.twotone.ro_optimize.chain_freq import gui as chain_freq_gui
from zcu_lab.v2.twotone.ro_optimize.chain_length import gui as chain_length_gui
from zcu_lab.v2.twotone.ro_optimize.chain_power import gui as chain_power_gui
from zcu_lab.v2.twotone.ro_optimize.freq import gui as ro_optimize_freq_gui
from zcu_lab.v2.twotone.ro_optimize.freq_gain import gui as ro_optimize_freq_gain_gui
from zcu_lab.v2.twotone.ro_optimize.length import gui as ro_optimize_length_gui
from zcu_lab.v2.twotone.ro_optimize.power import gui as ro_optimize_power_gui
from zcu_lab.v2.twotone.ro_optimize.rpi_freq import gui as rpi_freq_gui
from zcu_lab.v2.twotone.ro_optimize.rpi_length import gui as rpi_length_gui
from zcu_lab.v2.twotone.ro_optimize.rpi_power import gui as rpi_power_gui
from zcu_lab.v2.twotone.stirap.amplitude import gui as amplitude_gui
from zcu_lab.v2.twotone.stirap.delay import gui as delay_gui
from zcu_lab.v2.twotone.stirap.detuning import gui as detuning_gui
from zcu_lab.v2.twotone.time_domain.cpmg import gui as cpmg_gui
from zcu_lab.v2.twotone.time_domain.t1 import gui as time_domain_t1_gui
from zcu_lab.v2.twotone.time_domain.t1_cov import gui as t1_cov_gui
from zcu_lab.v2.twotone.time_domain.t2echo import gui as time_domain_t2echo_gui
from zcu_lab.v2.twotone.time_domain.t2ramsey import gui as time_domain_t2ramsey_gui
from zcu_lab.v2.twotone.zigzag import gui as zigzag_gui
from zcu_lab.v2.twotone.zigzag_scan import gui as zigzag_scan_gui


def register_all(
    registry: Registry,
    record_source: Callable[[str, ModuleType], None] | None = None,
    *,
    components: ComponentRegistry | None = None,
) -> None:
    """Register adapters into the caller-owned registry.

    Bind the chain_length and rpi_length Form factories to this registry's
    persistent template catalog (ADR-0073). Reload binds the catalog of the
    Registry passed to that call. Library creation reads registry.templates
    directly.

    Pass the shared ComponentRegistry as components at startup to register
    component models and roles. Omit it on reload to preserve that registry.
    If supplied, record_source receives each registered key and its direct
    declaration module after Registry accepts the adapter. Duplicate adapters
    raise ValueError. Registration does not build forms or validate templates;
    Open and Reset do.
    """
    if components is not None:
        register_components(components)
    # Keep the definition (or factory result) and its declaring module together.
    adapters: dict[str, tuple[object, ModuleType]] = {
        "lookback": (lookback_gui.DEFINITION, lookback_gui),
        "onetone/freq": (onetone_freq_gui.DEFINITION, onetone_freq_gui),
        "onetone/power_dep": (onetone_power_dep_gui.DEFINITION, onetone_power_dep_gui),
        "onetone/flux_dep": (onetone_flux_dep_gui.DEFINITION, onetone_flux_dep_gui),
        "twotone/freq": (twotone_freq_gui.DEFINITION, twotone_freq_gui),
        "twotone/chain": (chain_gui.DEFINITION, chain_gui),
        "twotone/stirap/delay": (delay_gui.DEFINITION, delay_gui),
        "twotone/stirap/detuning": (detuning_gui.DEFINITION, detuning_gui),
        "twotone/stirap/amplitude": (amplitude_gui.DEFINITION, amplitude_gui),
        "twotone/flux_dep": (twotone_flux_dep_gui.DEFINITION, twotone_flux_dep_gui),
        "twotone/power_dep": (twotone_power_dep_gui.DEFINITION, twotone_power_dep_gui),
        "twotone/rabi/len_rabi": (rabi_len_rabi_gui.DEFINITION, rabi_len_rabi_gui),
        "twotone/rabi/amp_rabi": (rabi_amp_rabi_gui.DEFINITION, rabi_amp_rabi_gui),
        "twotone/raman_gf": (raman_gf_gui.DEFINITION, raman_gf_gui),
        "twotone/time_domain/t1": (time_domain_t1_gui.DEFINITION, time_domain_t1_gui),
        "twotone/time_domain/t1_cov": (t1_cov_gui.DEFINITION, t1_cov_gui),
        "twotone/time_domain/t2ramsey": (
            time_domain_t2ramsey_gui.DEFINITION,
            time_domain_t2ramsey_gui,
        ),
        "twotone/time_domain/t2echo": (
            time_domain_t2echo_gui.DEFINITION,
            time_domain_t2echo_gui,
        ),
        "chain/amp_rabi": (chain_amp_rabi_gui.DEFINITION, chain_amp_rabi_gui),
        "chain/len_rabi": (chain_len_rabi_gui.DEFINITION, chain_len_rabi_gui),
        "chain/t1": (chain_t1_gui.DEFINITION, chain_t1_gui),
        "chain/t2ramsey": (chain_t2ramsey_gui.DEFINITION, chain_t2ramsey_gui),
        "chain/t2echo": (chain_t2echo_gui.DEFINITION, chain_t2echo_gui),
        "singleshot/ge": (ge_gui.DEFINITION, ge_gui),
        "singleshot/qutrit_population": (
            qutrit_population_gui.DEFINITION,
            qutrit_population_gui,
        ),
        "singleshot/check": (singleshot_check_gui.DEFINITION, singleshot_check_gui),
        "singleshot/len_rabi": (
            singleshot_len_rabi_gui.DEFINITION,
            singleshot_len_rabi_gui,
        ),
        "singleshot/amp_rabi": (
            singleshot_amp_rabi_gui.DEFINITION,
            singleshot_amp_rabi_gui,
        ),
        "singleshot/t1": (singleshot_t1_gui.DEFINITION, singleshot_t1_gui),
        "singleshot/ac_stark": (
            singleshot_ac_stark_gui.DEFINITION,
            singleshot_ac_stark_gui,
        ),
        "singleshot/reset_check": (reset_check_gui.DEFINITION, reset_check_gui),
        "singleshot/t1_tone": (t1_tone_gui.DEFINITION, t1_tone_gui),
        "singleshot/t1_tone_sweep/gain": (
            t1_tone_sweep_gui.GAIN_DEFINITION,
            t1_tone_sweep_gui,
        ),
        "singleshot/t1_tone_sweep/freq": (
            t1_tone_sweep_gui.FREQ_DEFINITION,
            t1_tone_sweep_gui,
        ),
        "twotone/dispersive": (dispersive_gui.DEFINITION, dispersive_gui),
        "twotone/ac_stark": (twotone_ac_stark_gui.DEFINITION, twotone_ac_stark_gui),
        "twotone/ac_stark_ramsey": (
            ac_stark_ramsey_gui.DEFINITION,
            ac_stark_ramsey_gui,
        ),
        "twotone/ckp": (ckp_gui.DEFINITION, ckp_gui),
        "twotone/time_domain/cpmg": (cpmg_gui.DEFINITION, cpmg_gui),
        "twotone/allxy": (allxy_gui.DEFINITION, allxy_gui),
        "twotone/zigzag": (zigzag_gui.DEFINITION, zigzag_gui),
        "twotone/zigzag_scan/gain": (zigzag_scan_gui.GAIN_DEFINITION, zigzag_scan_gui),
        "twotone/zigzag_scan/freq": (zigzag_scan_gui.FREQ_DEFINITION, zigzag_scan_gui),
        "twotone/rb": (twotone_rb_gui.DEFINITION, twotone_rb_gui),
        "twotone/ro_optimize/freq": (
            ro_optimize_freq_gui.DEFINITION,
            ro_optimize_freq_gui,
        ),
        "twotone/ro_optimize/power": (
            ro_optimize_power_gui.DEFINITION,
            ro_optimize_power_gui,
        ),
        "twotone/ro_optimize/length": (
            ro_optimize_length_gui.DEFINITION,
            ro_optimize_length_gui,
        ),
        "twotone/ro_optimize/freq_gain": (
            ro_optimize_freq_gain_gui.DEFINITION,
            ro_optimize_freq_gain_gui,
        ),
        "twotone/ro_optimize/auto": (auto_gui.DEFINITION, auto_gui),
        "twotone/ro_optimize/chain_power": (
            chain_power_gui.DEFINITION,
            chain_power_gui,
        ),
        "twotone/ro_optimize/chain_freq": (chain_freq_gui.DEFINITION, chain_freq_gui),
        "twotone/ro_optimize/chain_length": (
            chain_length_gui.build_adapter(registry.templates),
            chain_length_gui,
        ),
        "twotone/ro_optimize/rpi_power": (rpi_power_gui.DEFINITION, rpi_power_gui),
        "twotone/ro_optimize/rpi_freq": (rpi_freq_gui.DEFINITION, rpi_freq_gui),
        "twotone/ro_optimize/rpi_length": (
            rpi_length_gui.build_adapter(registry.templates),
            rpi_length_gui,
        ),
        "twotone/reset/single_tone/freq": (
            single_tone_freq_gui.DEFINITION,
            single_tone_freq_gui,
        ),
        "twotone/reset/single_tone/length": (
            single_tone_length_gui.DEFINITION,
            single_tone_length_gui,
        ),
        "twotone/reset/dual_tone/freq": (
            dual_tone_freq_gui.DEFINITION,
            dual_tone_freq_gui,
        ),
        "twotone/reset/dual_tone/length": (
            dual_tone_length_gui.DEFINITION,
            dual_tone_length_gui,
        ),
        "twotone/reset/dual_tone/power": (
            dual_tone_power_gui.DEFINITION,
            dual_tone_power_gui,
        ),
        "twotone/reset/bath/freq_gain": (
            bath_freq_gain_gui.DEFINITION,
            bath_freq_gain_gui,
        ),
        "twotone/reset/bath/length": (bath_length_gui.DEFINITION, bath_length_gui),
        "twotone/reset/bath/phase": (bath_phase_gui.DEFINITION, bath_phase_gui),
        "twotone/reset/rabi_check": (rabi_check_gui.DEFINITION, rabi_check_gui),
        "twotone/reset/flux/gain": (gain_gui.DEFINITION, gain_gui),
        "twotone/reset/flux/length": (flux_length_gui.DEFINITION, flux_length_gui),
        "mux/onetone": (onetone_gui.DEFINITION, onetone_gui),
        "mux/onetone_flux_global": (
            onetone_flux_global_gui.DEFINITION,
            onetone_flux_global_gui,
        ),
        "mux/onetone_flux_local": (
            onetone_flux_local_gui.DEFINITION,
            onetone_flux_local_gui,
        ),
        "mux/twotone_flux_global": (
            twotone_flux_global_gui.DEFINITION,
            twotone_flux_global_gui,
        ),
        "mux/twotone_flux_local": (
            twotone_flux_local_gui.DEFINITION,
            twotone_flux_local_gui,
        ),
        "onetone/range_find": (range_find_gui.DEFINITION, range_find_gui),
        "mux/twotone_spectrum": (twotone_spectrum_gui.DEFINITION, twotone_spectrum_gui),
        "mux/len_rabi": (mux_len_rabi_gui.DEFINITION, mux_len_rabi_gui),
        "mux/amp_rabi": (mux_amp_rabi_gui.DEFINITION, mux_amp_rabi_gui),
        "mux/t1": (mux_t1_gui.DEFINITION, mux_t1_gui),
        "mux/t2ramsey": (mux_t2ramsey_gui.DEFINITION, mux_t2ramsey_gui),
        "mux/t2echo": (mux_t2echo_gui.DEFINITION, mux_t2echo_gui),
        "mux/singleshot_ge": (singleshot_ge_gui.DEFINITION, singleshot_ge_gui),
        "jpa/flux_onetone": (flux_onetone_gui.DEFINITION, flux_onetone_gui),
        "jpa/freq": (jpa_freq_gui.DEFINITION, jpa_freq_gui),
        "jpa/flux": (flux_gui.DEFINITION, flux_gui),
        "jpa/power": (jpa_power_gui.DEFINITION, jpa_power_gui),
        "jpa/auto_optimize": (auto_optimize_gui.DEFINITION, auto_optimize_gui),
        "jpa/check": (jpa_check_gui.DEFINITION, jpa_check_gui),
        "two_qubit/joint_readout": (joint_readout_gui.DEFINITION, joint_readout_gui),
        "two_qubit/conditional_spectroscopy": (
            two_qubit_conditional_spectroscopy_gui.DEFINITION,
            two_qubit_conditional_spectroscopy_gui,
        ),
        "two_qubit/chevron": (two_qubit_chevron_gui.DEFINITION, two_qubit_chevron_gui),
        "two_qubit/cz_amplitude": (
            two_qubit_cz_amplitude_gui.DEFINITION,
            two_qubit_cz_amplitude_gui,
        ),
        "two_qubit/cz_optimize": (
            two_qubit_cz_optimize_gui.DEFINITION,
            two_qubit_cz_optimize_gui,
        ),
        "two_qubit/conditional_phase": (
            two_qubit_conditional_phase_gui.DEFINITION,
            two_qubit_conditional_phase_gui,
        ),
        "two_qubit/local_z": (two_qubit_local_z_gui.DEFINITION, two_qubit_local_z_gui),
        "two_qubit/idle_zz": (two_qubit_idle_zz_gui.DEFINITION, two_qubit_idle_zz_gui),
        "heralded/idle_zz": (heralded_idle_zz_gui.DEFINITION, heralded_idle_zz_gui),
        "two_qubit/constructive_phase": (
            constructive_phase_gui.DEFINITION,
            constructive_phase_gui,
        ),
        "two_qubit/rb": (two_qubit_rb_gui.DEFINITION, two_qubit_rb_gui),
        "heralded/conditional_spectroscopy": (
            heralded_conditional_spectroscopy_gui.DEFINITION,
            heralded_conditional_spectroscopy_gui,
        ),
        "heralded/conditional_phase": (
            heralded_conditional_phase_gui.DEFINITION,
            heralded_conditional_phase_gui,
        ),
        "heralded/cz_amplitude": (
            heralded_cz_amplitude_gui.DEFINITION,
            heralded_cz_amplitude_gui,
        ),
        "heralded/cz_optimize": (
            heralded_cz_optimize_gui.DEFINITION,
            heralded_cz_optimize_gui,
        ),
        "heralded/local_z": (heralded_local_z_gui.DEFINITION, heralded_local_z_gui),
        "heralded/rb": (heralded_rb_gui.DEFINITION, heralded_rb_gui),
        "heralded/chevron": (heralded_chevron_gui.DEFINITION, heralded_chevron_gui),
        "fastflux/twotone": (twotone_gui.DEFINITION, twotone_gui),
        "fastflux/t1": (fastflux_t1_gui.DEFINITION, fastflux_t1_gui),
        "fastflux/distortion/freq": (
            distortion_freq_gui.DEFINITION,
            distortion_freq_gui,
        ),
        "fastflux/distortion/phase": (
            distortion_phase_gui.DEFINITION,
            distortion_phase_gui,
        ),
        "fastflux/distortion/acc_phase": (acc_phase_gui.DEFINITION, acc_phase_gui),
        "fastflux/mist": (mist_gui.DEFINITION, mist_gui),
        "mist/flux_dep": (mist_flux_dep_gui.DEFINITION, mist_flux_dep_gui),
        "mist/power_dep/drive_freq": (drive_freq_gui.DEFINITION, drive_freq_gui),
        "mist/power_dep/single_trace": (single_trace_gui.DEFINITION, single_trace_gui),
        "singleshot/mist/freq": (mist_freq_gui.DEFINITION, mist_freq_gui),
        "singleshot/mist/power": (mist_power_gui.DEFINITION, mist_power_gui),
        "singleshot/mist/power_freq": (power_freq_gui.DEFINITION, power_freq_gui),
        "singleshot/mist/pre_freq": (pre_freq_gui.DEFINITION, pre_freq_gui),
    }
    for name, (definition, source) in adapters.items():
        registry.register(name, definition)
        if record_source is not None:
            record_source(name, source)
