"""Concrete legacy parameter and deferred module declarations for this lab.

Values stay in legacy working units. This profile explicitly creates R1, Q1 and
J1, with Q1.resonator pointing to R1; it does not infer topology from filenames.
Flux values require explicit unit keys in the same legacy JSON. No cfg is built
or evaluated, and no definitions are registered on import.
"""

from collections.abc import Mapping
from typing import Literal

from zcu_tools.resources.parameter_upgrade import ParameterDocument

from migrations.legacy_labber import LegacyVariableSchema
from migrations.storage_migration import KeyRule, MigrationMapping, ModuleRule

# Explicit legacy mappings preserve working units and fixed roots. Changes to rules, seeds,
# roles or native declarations require new revisions for both kind profiles.
# Never reuse a revision across profiles: resume compares this exact identity.
_KIND_REVISIONS = {"qubit/fluxonium": "1.26", "qubit/transmon": "1.27"}

_VALUE_PATHS = (
    ("r_f", "R1.freq"),
    ("rf_w", "R1.kappa"),
    ("res_ch", "R1.wiring.ch"),
    ("ro_ch", "R1.wiring.ro_ch"),
    ("qub_0_1_ch", "Q1.wiring.ch"),
    ("qub_ch", "Q1.wiring.ch"),
    ("lo_flux_ch", "Q1.wiring.lo_flux_ch"),
    ("timeFly", "R1.ext.time_of_fly"),
    ("res_edelay_calibration", "R1.ext.edelay"),
    ("theta0", "R1.ext.theta0"),
    ("res_probe_len", "R1.ext.default_probe_length"),
    ("q_f", "Q1.freq"),
    ("qf_w", "Q1.kappa"),
    ("t1", "Q1.t1"),
    ("t2r", "Q1.t2r"),
    ("t2e", "Q1.t2e"),
    ("rabi_f", "Q1.ext.rabi_freq"),
    ("t1_with_tone", "Q1.ext.t1_with_tone"),
    ("ac_stark_coeff", "Q1.ext.ac_stark_coeff"),
    ("cutoff", "Q1.ext.cutoff"),
    ("chi", "Q1.ext.chi"),
    ("best_ro_freq", "Q1.ext.best_ro_freq"),
    ("best_ro_gain", "Q1.ext.best_ro_gain"),
    ("best_ro_length", "Q1.ext.best_ro_length"),
    ("g_center", "Q1.ext.readout_cal.g_center"),
    ("e_center", "Q1.ext.readout_cal.e_center"),
    ("ge_radius", "Q1.ext.readout_cal.radius"),
    ("ge_s", "Q1.ext.readout_cal.sigma"),
    ("confusion_matrix", "Q1.ext.readout_cal.confusion_matrix"),
    ("fid", "Q1.ext.readout_cal.fidelity"),
    ("reset_f", "Q1.ext.reset_f"),
    ("resetf_w", "Q1.ext.reset_fwhm"),
    ("reset_f1", "Q1.ext.reset_f1"),
    ("reset_f2", "Q1.ext.reset_f2"),
    ("reset_gain1", "Q1.ext.reset_gain1"),
    ("reset_gain2", "Q1.ext.reset_gain2"),
    ("bathreset_freq", "Q1.ext.reset_bath_freq"),
    ("bathreset_gain", "Q1.ext.reset_bath_gain"),
    ("bathreset_max_phase", "Q1.ext.reset_bath_max_phase"),
    ("bathreset_min_phase", "Q1.ext.reset_bath_min_phase"),
    ("best_jpa_freq", "J1.pump_freq"),
    ("best_jpa_power", "J1.pump_power"),
    ("flux_unit", "general.ext.flux_unit"),
    ("local_flux_unit", "Q1.flux_unit"),
    ("jpa_flux_unit", "J1.flux_unit"),
)

_FLUX_PATHS = (
    ("cur_A", "general.ext.flux_value", "flux_unit"),
    ("flx_half", "Q1.flux_half", "flux_unit"),
    ("flx_period", "Q1.flux_period", "flux_unit"),
    ("flx_int", "Q1.flux_int", "flux_unit"),
    ("flux_bias", "Q1.ext.flux_bias", "flux_unit"),
    ("best_jpa_flux", "J1.flux", "jpa_flux_unit"),
)

_MODULE_KEY_PATHS = (
    ("pi_gain", "qubit.module.x180.gain", "Q1.pulses.pi_amp.gain"),
    (
        "pi_len",
        "qubit.module.x180.waveform.length",
        "Q1.pulses.pi_len.waveform.length",
    ),
    ("pi2_gain", "qubit.module.x90.gain", "Q1.pulses.pi2_amp.gain"),
    (
        "pi2_len",
        "qubit.module.x90.waveform.length",
        "Q1.pulses.pi2_len.waveform.length",
    ),
)

# Only these aliases have a documented corresponding formal destination.
# Removal still requires a successful same-context source write with equal value.
_ALIAS_TARGETS = {"readout_f": "R1.freq", "flx_bias": "Q1.ext.flux_bias"}

_REMOVED = (
    ("readout_f", "Use resonator.freq instead of the removed readout frequency alias"),
    ("flx_bias", "Obsolete alias; flux_bias holds the predictor's flux offset"),
    ("mA_c", "Obsolete flux value"),
    ("mA_e", "Obsolete flux value"),
    ("resetf1_w", "Removed reset width"),
    ("cur_jpa_A", "Obsolete JPA flux alias"),
    ("fake_peak", "Fake adapter uses component fields and ext"),
    ("log_scale", "Analysis parameter, not a stored component value"),
)

# Candidates follow existing role_table Lib priority, not source YAML order.
# Each legacy reset name keeps its fixed root without guessing module.reset.
# Calibration uses ext scalars, not per-scheme module slots.
_MODULE_NAMES = (
    ("qub_probe", "Q1.pulses.qub_probe", "Q1.module.probe"),
    ("res_probe", "R1.pulses.res_probe", "R1.module.probe"),
    ("pi_amp", "Q1.pulses.pi_amp", "Q1.module.x180"),
    ("pi_len", "Q1.pulses.pi_len", "Q1.module.x180"),
    ("pi2_amp", "Q1.pulses.pi2_amp", "Q1.module.x90"),
    ("pi2_len", "Q1.pulses.pi2_len", "Q1.module.x90"),
    ("readout_dpm", "R1.readout.readout_dpm", "Q1.module.readout"),
    ("readout_rf", "R1.readout.readout_rf", "Q1.module.readout"),
    ("readout", "R1.readout.readout", "Q1.module.readout"),
    ("res_readout", "R1.readout.res_readout", "Q1.module.readout"),
    ("readout_direct", "R1.readout.readout_direct", None),
    ("reset_10", "Q1.reset.reset_10", None),
    ("reset_120", "Q1.reset.reset_120", None),
    ("reset_none", "Q1.reset.reset_none", None),
    ("reset_bath", "Q1.reset.reset_bath", None),
    ("reset_bath_e", "Q1.reset.reset_bath_e", None),
)


def _convert_parameter_batch(document: ParameterDocument) -> ParameterDocument:
    # Keep lab imports at execution, not declaration: synthetic framework callers
    # replace this callback and never load concrete lab models.
    from zcu_lab.components.flux_upgrade import convert_storage_flux_batch

    return convert_storage_flux_batch(document)


def build_mapping(
    *,
    qubit_kind: Literal["qubit/fluxonium", "qubit/transmon"],
    data_schemas: Mapping[tuple[str, str], tuple[LegacyVariableSchema, ...]],
    native_tags: Mapping[tuple[str, str], str],
) -> MigrationMapping:
    """Return this lab's explicit R1/Q1/J1 conversion profile without any I/O.

    qubit_kind is the caller's declared historical kind, never inferred from
    chip/qubit names. data_schemas is supplied by the composition root's fixed
    native declarations. native_tags explicitly renames selected historical
    pairs for native metadata only; omitted pairs keep the historical tag.
    Seeds carry no guessed channels, flux units or pulse
    values. Keep complete readout_cal ext containers for dotted value writes.
    No entry registration or module cfg conversion occurs here. Unsupported
    kinds raise ValueError. Missing unit evidence is pending in the converter.
    """
    if qubit_kind not in _KIND_REVISIONS:
        raise ValueError(f"Unsupported migration qubit kind: {qubit_kind}")
    rules = (
        tuple(
            KeyRule(
                old_key=key,
                target_path=path,
                action="value",
                reason="Preserve legacy working units without conversion",
            )
            for key, path in _VALUE_PATHS
        )
        + tuple(
            KeyRule(
                old_key=key,
                target_path=path,
                action="value",
                requires_keys=(unit_key,),
                reason="Flux current or voltage requires an explicit unit key",
            )
            for key, path, unit_key in _FLUX_PATHS
        )
        + tuple(
            KeyRule(
                old_key=key,
                target_path=path,
                action="stderr",
                reason="Store standard error in the value's working units",
            )
            for key, path in (
                ("t1err", "Q1.t1"),
                ("t2r_err", "Q1.t2r"),
                ("t2e_err", "Q1.t2e"),
            )
        )
        + tuple(
            KeyRule(
                old_key=key,
                target_path=path,
                action="module",
                candidate_path=candidate,
                reason="Compare only the declared same-context library candidate",
            )
            for key, path, candidate in _MODULE_KEY_PATHS
        )
        + tuple(
            KeyRule(
                old_key=key,
                target_path=_ALIAS_TARGETS.get(key),
                action="remove",
                reason=reason,
            )
            for key, reason in _REMOVED
        )
    )
    return MigrationMapping(
        mapping_version=_KIND_REVISIONS[qubit_kind],
        parameter_transform=_convert_parameter_batch,
        components={
            "R1": {"kind": "resonator"},
            "Q1": {"kind": qubit_kind, "resonator": "R1", "ext": {"readout_cal": {}}},
            "J1": {"kind": "amplifier/jpa"},
        },
        roles={"qubit": "Q1", "resonator": "R1"},
        rules=rules,
        data_schemas=data_schemas,
        native_tags=native_tags,
        module_rules=tuple(
            ModuleRule(
                old_name=name,
                target_path=target,
                reference_path=reference,
                reason="Fixed reset root; downstream selection unchanged"
                if name in ("reset_10", "reset_120", "reset_bath", "reset_bath_e")
                else "Preserve the named legacy module as an explicit library candidate",
            )
            for name, target, reference in _MODULE_NAMES
        )
        + tuple(
            ModuleRule(
                old_name=name,
                target_path=f"{component}.waveforms.{name}",
                reference_path=None,
                kind="waveform",
                reason="Preserve the named legacy waveform as a formal library entry",
            )
            for name, component in (
                ("mist_waveform", "Q1"),
                ("qub_cos", "Q1"),
                ("qub_flat", "Q1"),
                ("ro_waveform", "R1"),
            )
        ),
    )
