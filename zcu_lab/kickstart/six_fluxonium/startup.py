"""Create or reopen six-fluxonium; no hardware connection or simulation loading."""

from pathlib import Path

from pydantic import TypeAdapter
from zcu_tools.format_version import YamlMap
from zcu_tools.gui.app.measure.startup_context import StartupContext
from zcu_tools.gui.session.aliases import AliasRow
from zcu_tools.resources.entry import DestinationExistsError, load_entry_template


def startup(ctx: StartupContext) -> None:
    """Select six-fluxonium/initial, creating template components and aliases on first use.

    ctx is the GUI startup capability, called after lab component registration.
    Use its project roots, not the shell cwd. A new entry gets the adjacent
    entry-template.yaml components in setup, then an initial point with those
    components and persisted aliases. Existing six-fluxonium/initial is opened unchanged.
    The template seeds Q1-Q6, R1-R6, Q1_bias-Q6_bias, coil_bias and MG123456.
    Aliases bind mux=MG123456, qubit=Q1, resonator=$qubit.resonator and
    flux_dev=$qubit.flux_source, selecting the scalar target's local DC line.
    Existing entries keep their aliases; add missing roles in Aliases.
    MG123456 uses mux gen0 and ADCs [0, 1, 2, 3, 4, 5] in resonator order.
    No SoC or device commands run, no simulation is loaded, and no close
    callback is registered.

    A partial entry raises RuntimeError with recovery instructions. Missing
    initial, template, validation and I/O errors propagate to the startup
    runner; completed operations remain in place without rollback.
    """
    try:
        ctx.setup.new_entry("six-fluxonium")
    except DestinationExistsError as exists:
        ctx.progress("Reopening six-fluxonium entry")
        try:
            ctx.setup.open_entry("six-fluxonium")
        except FileNotFoundError as missing:
            # Never overwrite a partial entry left by an interrupted creation.
            raise RuntimeError(
                f"six-fluxonium is only partially present ({exists.filename} exists, "
                f"{missing.filename} is missing). Remove both six-fluxonium directories "
                "under result/ and Database/ to start fresh, or restore the "
                "missing one."
            ) from missing
        ctx.setup.use_point("initial")
    else:
        ctx.progress("Seeding six-fluxonium entry from template")
        template = load_entry_template(Path(__file__).with_name("entry-template.yaml"))
        for name, component in template.components.items():
            fields = TypeAdapter(YamlMap).validate_python(
                component.model_dump(exclude_unset=True)
            )
            fields.pop("kind")
            ctx.context.add_component(name, component.kind, fields, setup=True)
        ctx.setup.new_point("initial")
        ctx.context.apply_alias_rows(
            tuple(
                AliasRow(name, value)
                for name, value in template.aliases.aliases.items()
            )
        )
