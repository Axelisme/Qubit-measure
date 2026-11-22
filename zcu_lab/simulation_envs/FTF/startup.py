"""Open or create the FTF entry and install offline resources in a measure GUI."""

import logging
import os
from pathlib import Path

import yaml
from pydantic import TypeAdapter
from zcu_tools.analysis.flux import DevicePeriod, offset_for_flux
from zcu_tools.device import DeviceManager
from zcu_tools.device.fake import FakeDeviceInfo
from zcu_tools.format_version import YamlMap
from zcu_tools.gui.app.measure.startup_context import StartupContext, simulation_workers
from zcu_tools.gui.session.aliases import AliasRow
from zcu_tools.gui.session.context_types import LibraryChange
from zcu_tools.program import build_soccfg
from zcu_tools.resources.context.library import ModuleLibrary
from zcu_tools.resources.entry import (
    DestinationExistsError,
    EntryTemplate,
    load_entry_template,
)
from zcu_tools.resources.predictor_file import (
    PredictorFile,
    PredictorFlux,
    PredictorSource,
    save_predictor_file,
)
from zcu_tools.simulate.fluxonium.predict import FluxoniumParams
from zcu_tools.simulators.ftf.config import load_ftf
from zcu_tools.simulators.ftf.config.schema import FTFSettings
from zcu_tools.simulators.ftf.environment import ftf

from zcu_lab.components.fluxonium import QubitSchema

PROFILE_PATH = Path(__file__).resolve().parents[1] / "profiles" / "qick_216_v2.yaml"


def startup(ctx: StartupContext) -> None:
    """Select ftf-demo/initial, seeding it from the template on first use.

    The template binds qubit1=Q1, qubit2=Q2, coupler=coupler and mux=MG12
    for two-qubit forms. Existing entries keep their aliases; update the point's
    Aliases if the current bindings are missing.
    The template seeds the half-flux components and module slots. The script
    installs the demo library in the entry's global documents before creating
    the point, so `initial` inherits both the components and their modules.

    ctx is the GUI's typed startup capability, called on its background worker
    after lab components/adapters are registered. Use the GUI's data project
    roots, not the shell cwd. An existing ftf-demo is reopened unchanged at its
    initial point, so later edits survive restarts; the template only seeds a
    new entry. Independent design guesses seed only missing predictor files.
    Unknown design qubits or mismatched device units raise ValueError.
    Template, entry, environment and binding failures propagate to
    the startup runner, which retains completed work and registered cleanup.

    Reserve one host CPU, with at least one simulation worker. Prepare common
    CPU paths and the OpenCL program in the background so the first acquisition
    need not compile them after startup work finishes. This script never acquires
    or controls hardware. Successfully registered drivers become GUI-owned.
    The close callback drains preparation and closes only untransferred drivers.
    """
    settings = load_ftf(Path(__file__).with_name("ftf.yaml"))
    template = load_entry_template(Path(__file__).with_name("entry-template.yaml"))
    try:
        ctx.setup.new_entry("ftf-demo")
    except DestinationExistsError as exists:
        ctx.progress("Reopening FTF demo entry")
        try:
            ctx.setup.open_entry("ftf-demo")
        except FileNotFoundError as missing:
            # Result and Database halves are created together; one alone is a
            # partial entry that startup must neither reuse nor overwrite.
            raise RuntimeError(
                f"ftf-demo is only partially present ({exists.filename} exists, "
                f"{missing.filename} is missing). Remove both ftf-demo directories "
                "under result/ and Database/ to start fresh, or restore the "
                "missing one."
            ) from missing
        ctx.setup.use_point("initial")
    else:
        ctx.progress("Seeding FTF demo entry from template")
        _seed_entry(ctx, template)

    ctx.progress("Building offline FTF resources")
    soccfg = build_soccfg(yaml.safe_load(PROFILE_PATH.read_text()))
    environment = ftf(
        settings,
        workers=simulation_workers(os.cpu_count()),
        warmup=True,
        soccfg=soccfg,
    )
    transferred: set[str] = set()

    def close_environment() -> None:
        errors: list[Exception] = []
        if environment.close_backend is not None:
            try:
                environment.close_backend()
            except Exception as error:
                logging.exception("FTF backend cleanup failed")
                errors.append(error)
        try:
            # Only the GUI closes transferred drivers. Rejected drivers remain
            # script-owned.
            for name in transferred:
                environment.devices.drop_device(name)
            environment.devices.close_all_devices()
        except Exception as error:
            logging.exception("FTF device cleanup failed")
            errors.append(error)
        if errors:
            raise ExceptionGroup("FTF resource cleanup failed", errors)

    ctx.on_close(close_environment)
    ctx.soc.install(environment.soc, environment.soccfg)
    for name, driver in environment.devices.get_all_devices().items():
        ctx.devices.register(name, driver)
        transferred.add(name)

    _seed_predictors(ctx, settings, template, environment.devices)
    if not settings.design:
        ctx.summary("FTF ready; design 未提供，未建立 predictor")


def _seed_predictors(
    ctx: StartupContext,
    settings: FTFSettings,
    template: EntryTemplate,
    devices: DeviceManager,
) -> None:
    """Validate authored guesses against template wiring, then save missing files."""
    if not settings.design:
        return
    predictors: dict[str, PredictorFile] = {}
    for name, design in settings.design.items():
        component = template.components.get(name)
        if not isinstance(component, QubitSchema):
            raise ValueError(f"design.{name}: no fluxonium qubit in entry-template")
        if component.flux_dev is None:
            raise ValueError(f"design.{name}: entry-template qubit has no flux_dev")
        info = devices.get_info(component.flux_dev)
        if not isinstance(info, FakeDeviceInfo) or info.value_unit != design.flux_unit:
            raise ValueError(
                f"design.{name}.flux_unit: must match device {component.flux_dev}"
            )
        predictors[name] = PredictorFile(
            model="fluxonium",
            params=FluxoniumParams(
                EJ=design.ej_ghz,
                EC=design.ec_ghz,
                EL=design.el_ghz,
            ),
            flux=PredictorFlux(
                offset=offset_for_flux(
                    {
                        component.flux_dev: DevicePeriod(
                            design.flux_period, design.flux_unit
                        )
                    },
                    {component.flux_dev: design.flux_half - design.flux_bias},
                    flux=0.5,
                ),
                period={component.flux_dev: design.flux_period},
                unit={component.flux_dev: design.flux_unit},
            ),
            source=PredictorSource(run_id=None, created_at=None),
        )
    directory = ctx.context.calls.call(ctx.context.control.get_predictors_path)
    if directory is None:
        raise RuntimeError("FTF predictor seeding requires an opened entry")
    directory.mkdir(parents=True, exist_ok=True)
    for name, predictor in predictors.items():
        path = directory / f"{name}.json"
        if not path.exists():
            save_predictor_file(path, predictor)


def _seed_entry(ctx: StartupContext, template: EntryTemplate) -> None:
    """Install global defaults before inheriting them into the initial point."""
    library = ModuleLibrary(Path(__file__).with_name("entry-library.yaml")).snapshot()
    if invalid := library.document().invalid_entries:
        raise ValueError(f"Invalid FTF demo module library: {invalid}")
    for name, component in template.components.items():
        fields = TypeAdapter(YamlMap).validate_python(
            component.model_dump(exclude_unset=True)
        )
        fields.pop("kind")
        ctx.context.add_component(name, component.kind, fields, point=None)

    def install_library() -> None:
        control = ctx.context.control
        target = control.read_context()
        for key in library.module_keys():
            control.apply_library_change(
                target,
                LibraryChange("module", "put", key, module=library.module_spec(key)),
            )
        for key in library.waveform_keys():
            control.apply_library_change(
                target,
                LibraryChange(
                    "waveform", "put", key, waveform=library.waveform_spec(key)
                ),
            )

    ctx.calls.call(install_library)
    ctx.setup.new_point("initial")
    ctx.context.apply_alias_rows(
        tuple(
            AliasRow(name, value) for name, value in template.aliases.aliases.items()
        ),
        point="initial",
    )
