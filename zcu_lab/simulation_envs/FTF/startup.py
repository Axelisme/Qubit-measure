"""Open or create the FTF entry and install offline resources in a measure GUI."""

from pathlib import Path

from pydantic import TypeAdapter
from zcu_tools.device import DeviceManager
from zcu_tools.device.fake import FakeDeviceInfo
from zcu_tools.experiment.ftf import ftf
from zcu_tools.format_version import YamlMap
from zcu_tools.gui.app.measure.startup_context import StartupContext
from zcu_tools.gui.session.aliases import AliasRow
from zcu_tools.resources.entry import (
    DestinationExistsError,
    EntryTemplate,
    load_entry_template,
)
from zcu_tools.resources.ftf import load_ftf
from zcu_tools.resources.ftf.schema import FTFSettings
from zcu_tools.resources.predictor_file import (
    PredictorFile,
    PredictorSource,
    save_predictor_file,
)
from zcu_tools.simulate.fluxonium.predict import FluxoniumParams

from zcu_lab.components.fluxonium import QubitSchema


def startup(ctx: StartupContext) -> None:
    """Select ftf-demo/initial, seeding it from the template on first use.

    The template binds f1=Q1, f2=Q2, coupler=coupler and mux=MG12 for FTF
    forms. Existing entries keep their aliases; add missing roles in Aliases.
    The template seeds MG12 mux gen1 and ADCs [1, 2] in resonator order.
    The GUI is then ready for the user to open experiment tabs.

    ctx is the GUI's typed startup capability, called on its background worker
    after lab components/adapters are registered. Use the GUI's data project
    roots, not the shell cwd. An existing ftf-demo is reopened unchanged at its
    initial point, so later edits survive restarts; the template only seeds a
    new entry. Independent design guesses seed only missing predictor files.
    Unknown design qubits or mismatched device units raise ValueError.
    Template, entry, environment and binding failures propagate to
    the startup runner, which retains completed work and registered cleanup.

    Build with one worker and no numeric warmup; this script never acquires or
    controls hardware. Successfully registered drivers become GUI-owned. The
    close callback releases borrowed references and closes only untransferred
    drivers, after background work drains and before GUI driver cleanup.
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
    environment = ftf(settings, workers=1, warmup=False)
    transferred: set[str] = set()

    def close_environment() -> None:
        # Runtime reads this manager; transferred drivers are borrowed until close.
        # Only the GUI closes them. Rejected drivers remain script-owned.
        for name in transferred:
            environment.devices.drop_device(name)
        environment.devices.close_all_devices()

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
        if component.flux_source is None:
            raise ValueError(f"design.{name}: entry-template qubit has no flux_source")
        info = devices.get_info(component.flux_source)
        if not isinstance(info, FakeDeviceInfo) or info.value_unit != design.flux_unit:
            raise ValueError(
                f"design.{name}.flux_unit: must match device {component.flux_source}"
            )
        predictors[name] = PredictorFile(
            model="fluxonium",
            params=FluxoniumParams(
                EJ=design.ej_ghz,
                EC=design.ec_ghz,
                EL=design.el_ghz,
                flux_half=design.flux_half,
                flux_period=design.flux_period,
                flux_bias=design.flux_bias,
            ),
            flux_unit=design.flux_unit,
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
    """Fill the just-created entry's setup and initial point from the template."""
    for name, component in template.components.items():
        fields = TypeAdapter(YamlMap).validate_python(
            component.model_dump(exclude_unset=True)
        )
        fields.pop("kind")
        ctx.context.add_component(name, component.kind, fields, setup=True)
    ctx.setup.new_point("initial")
    ctx.context.apply_alias_rows(
        tuple(AliasRow(name, value) for name, value in template.aliases.aliases.items())
    )
