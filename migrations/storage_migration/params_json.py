"""Offline conversion of legacy analysis handoffs into entry-scoped resources."""

from dataclasses import dataclass, replace
from pathlib import Path

from pydantic import TypeAdapter
from zcu_tools.datafile import JsonObject
from zcu_tools.resources.document_store import YamlMap, YamlValue
from zcu_tools.resources.entry import (
    ResultEntry,
    SetupView,
    component_registry,
    document_value,
    snapshot,
)
from zcu_tools.resources.predictor_file import (
    PredictorFile,
    PredictorSource,
    load_predictor_file,
    save_predictor_file,
    validate_predictor_file,
)
from zcu_tools.simulate.fluxonium.predict import FluxoniumParams

from .analysis_files import convert_analysis_files, proven_created_at
from .backup import backup_source_path
from .errors import MigrationInputError
from .models import KeyMappingItem, MigrationMapping
from .paths import contained_path
from .state import MigrationSession, record_pending

_JSON = TypeAdapter(JsonObject)
_YAML = TypeAdapter(YamlValue)
_ENERGIES = ("EJ", "EC", "EL")
_FLUX_FIELDS = ("flux_half", "flux_period")


@dataclass
class _Context:
    """Setup metadata, including seed values when no dry-run handle exists."""

    target: Path
    view: SetupView | None
    values: YamlMap

    def value(self, path: str) -> YamlValue:
        if self.view is not None:
            try:
                return document_value(snapshot(self.view), path)
            except AttributeError:
                return None
        return self.values.get(path)

    def set(self, path: str, value: YamlValue) -> None:
        if self.view is not None:
            with self.view.edit() as draft:
                draft.set(path, value)
        self.values[path] = value


def _setup_context(
    session: MigrationSession,
    entry: ResultEntry | None,
    mapping: MigrationMapping,
    name: str,
) -> _Context:
    seed = mapping.components[name]
    kind = seed["kind"]
    if not isinstance(kind, str):
        raise ValueError(f"{name}: seed requires kind")
    model = component_registry.get(kind).model_validate(seed)
    return _Context(
        target=session.manifest.report.destination.result_path / "setup.yaml",
        view=entry.setup if entry is not None and not session.dry_run else None,
        values={
            f"{name}.{key}": _YAML.validate_python(value, strict=True)
            for key, value in model.model_dump().items()
        },
    )


def _flux_unit(
    values: JsonObject, fit: JsonObject, assumed: str | None
) -> tuple[str | None, str | None]:
    if "flux_unit" not in values and "flux_unit" not in fit and assumed is not None:
        return assumed, None
    root_unit, fit_unit = values.get("flux_unit"), fit.get("flux_unit")
    if "flux_unit" in values and "flux_unit" in fit and root_unit != fit_unit:
        return None, "params.json flux_unit and fluxdep_fit.flux_unit disagree"
    unit = root_unit if "flux_unit" in values else fit_unit
    if not isinstance(unit, str) or unit not in ("A", "mA", "V", "mV"):
        return None, "Missing or unsupported explicit flux_unit (A/mA/V/mV)"
    return unit, None


def _write_predictor(path: Path, predictor: PredictorFile) -> None:
    # The destination belongs to this migration, but an interrupted write may
    # already have committed it. Never replace a conflicting document on resume.
    if path.is_symlink():
        raise MigrationInputError(f"{path}: predictor symlink conflict")
    if path.exists():
        try:
            existing = load_predictor_file(path)
        except ValueError as error:
            raise MigrationInputError(f"{path}: invalid predictor: {error}") from error
        if existing != predictor:
            raise MigrationInputError(
                f"{path}: predictor differs from planned legacy fit"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    save_predictor_file(path, predictor)


def _project_fit(
    session: MigrationSession,
    source: Path,
    name: str,
    values: JsonObject,
) -> list[KeyMappingItem]:
    fit = values.get("fluxdep_fit")
    if not isinstance(fit, dict):
        return []
    target = (
        session.manifest.report.destination.result_path / "predictors" / f"{name}.json"
    )
    energies = fit.get("params")
    unsupported = None
    if "model" in fit:
        unsupported = "fluxdep_fit.model"
    elif isinstance(energies, dict) and energies.keys() - set(_ENERGIES):
        unsupported = "fluxdep_fit.params"
    if unsupported is not None:
        return [
            KeyMappingItem(
                old_file=source,
                old_key=unsupported,
                new_file=None,
                new_path=None,
                action="pending",
                reason="Model or parameter names have no supported legacy fluxdep_fit contract; preserve without guessing",
            )
        ]
    params: JsonObject = {"flux_bias": 0.0}
    rows: list[KeyMappingItem] = []
    for field in (*_ENERGIES, *_FLUX_FIELDS):
        container = energies if field in _ENERGIES else fit
        if not isinstance(container, dict) or field not in container:
            continue
        value = container[field]
        reason = None
        try:
            FluxoniumParams.from_mapping({field: value})
        except ValueError:
            reason = "Expected a finite numeric value or null; no expression evaluation or coercion"
        else:
            params[field] = value
        key = (
            f"fluxdep_fit.params.{field}"
            if field in _ENERGIES
            else f"fluxdep_fit.{field}"
        )
        rows.append(
            KeyMappingItem(
                old_file=source,
                old_key=key,
                new_file=target,
                new_path=f"params.{field}",
                action="pending" if reason else "value",
                reason=reason or "Preserve legacy fluxonium parameter in native units",
            )
        )
    assumed = session.manifest.report.declarations.assume_flux_unit
    unit, unit_error = _flux_unit(values, fit, assumed)
    created_at = proven_created_at(fit.get("timestamp"))
    backup = session.manifest.report.backup_path
    original = (
        backup_source_path(backup, session.manifest.report.source, source)
        if backup is not None
        else source
    )
    for key, path, reason in (
        ("fluxdep_fit.flux_unit", "flux_unit", unit_error),
        (
            "fluxdep_fit.timestamp",
            "source.created_at",
            None
            if created_at is not None
            else f"No proven timezone-aware ISO 8601 timestamp; original at {original}#fluxdep_fit.timestamp",
        ),
    ):
        rows.append(
            KeyMappingItem(
                old_file=source,
                old_key=key,
                new_file=target,
                new_path=path,
                action="pending" if reason else "value",
                reason=reason
                or (
                    f"假設單位: {assumed}; operator supplied, no numeric conversion"
                    if key == "fluxdep_fit.flux_unit"
                    and assumed is not None
                    and "flux_unit" not in values
                    and "flux_unit" not in fit
                    else "Preserve explicit legacy source metadata"
                ),
            )
        )
    predictor = PredictorFile(
        model="fluxonium",
        params=FluxoniumParams.from_mapping(params),
        flux_unit=unit,
        source=PredictorSource(run_id=None, created_at=created_at),
    )
    validate_predictor_file(predictor)
    if not session.dry_run:
        _write_predictor(target, predictor)
    return rows


def _preserve_sections(
    context: _Context, source: Path, name: str, values: JsonObject, *, has_ext: bool
) -> list[KeyMappingItem]:
    rows: list[KeyMappingItem] = []
    ext: YamlMap = {}
    current = context.value(f"{name}.ext")
    if isinstance(current, dict):
        ext.update(current)
    legacy: YamlMap = {}
    existing = ext.get("legacy_params")
    if isinstance(existing, dict):
        legacy.update(existing)
    for key, value in values.items():
        legacy[key] = _YAML.validate_python(value, strict=True)
        rows.append(
            KeyMappingItem(
                old_file=source,
                old_key=key,
                new_file=context.target if has_ext else None,
                new_path=f"{name}.ext.legacy_params.{key}" if has_ext else None,
                action="pending",
                reason="Legacy analysis section and metadata preserved verbatim; review before formal use"
                if has_ext
                else "Destination component has no ext field; original handoff remains preserved",
            )
        )
    if has_ext:
        ext["legacy_params"] = legacy
        context.set(f"{name}.ext", ext)
    return rows


def convert_params_json(
    session: MigrationSession,
    entry: ResultEntry | None,
    mapping: MigrationMapping,
) -> None:
    """Preserve metadata and publish entry predictor/dispersive/T1 fit files.

    session owns baselines and report; entry is its destination handle or None in
    dry-run. mapping supplies seeds with exactly one qubit/* kind; otherwise the
    whole handoff stays pending. Call before points so they inherit raw metadata,
    then apply each context's meta_info normally. Legacy EJ/EC/EL fits use the
    documented fluxonium model and zero bias; flux_int is alignment-only. An
    explicit model is unsupported and stays pending. Unknown/invalid numbers
    remain unknown, with invalid values located in pending. Units only come from
    explicit root/fit flux_unit, or the declared assumption only when both keys
    are absent. Otherwise missing/conflicting units become null and pending,
    without conversion. Source run_id is unknown; only a timezone-aware ISO 8601
    timestamp supplies created_at. Preserve full raw sections and metadata in
    component.ext.legacy_params with located pending. Predictors remain entry
    resources and never participate in setup/point inheritance. Dispersive and T1
    file owners validate partial data and source evidence; invalid content stays
    pending with its original ext copy. plot_transitions joins the dispersive
    file without inventing fit values. Dry-run reports
    the same plan without writes. The parameters owner checkpoints completion;
    input/I/O failures propagate and sources are never changed here.
    """
    root = session.manifest.report.source.result_path
    candidate = root / "params.json"
    if not candidate.is_file():
        return
    source = contained_path(candidate, root)
    session.baseline(source)
    values = _JSON.validate_json(source.read_text(encoding="utf-8"), strict=True)
    names = [
        name
        for name, seed in mapping.components.items()
        if isinstance(seed.get("kind"), str) and str(seed["kind"]).startswith("qubit/")
    ]
    if len(names) != 1:
        record_pending(
            session,
            source,
            "params.json",
            "No unique migrated qubit component; keep the whole handoff pending without guessing",
        )
        return
    name = names[0]
    kind = mapping.components[name]["kind"]
    if not isinstance(kind, str):
        raise ValueError(f"{name}: seed requires kind")
    fields = set(component_registry.get(kind).model_fields)
    context = _setup_context(session, entry, mapping, name)
    rows = _project_fit(session, source, name, values)
    rows.extend(convert_analysis_files(session, entry, source, name, values))
    rows.extend(
        _preserve_sections(context, source, name, values, has_ext="ext" in fields)
    )
    report = session.manifest.report
    session.report(
        replace(
            report,
            key_mappings=(
                *[row for row in report.key_mappings if row.old_file != source],
                *rows,
            ),
        )
    )
    reasons: dict[str, list[str]] = {}
    for row in rows:
        if row.action == "pending":
            reasons.setdefault(row.old_key, []).append(f"{row.new_file}: {row.reason}")
    for key, messages in reasons.items():
        record_pending(session, source, key, "; ".join(messages))
    session.baseline(source)
