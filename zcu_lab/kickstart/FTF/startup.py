"""Create or reopen the FTF entry without connecting SoC or devices."""

from pathlib import Path

from pydantic import TypeAdapter
from zcu_tools.format_version import YamlMap
from zcu_tools.gui.app.measure.startup_context import StartupContext
from zcu_tools.gui.session.aliases import AliasRow
from zcu_tools.resources.entry import DestinationExistsError, load_entry_template


def startup(ctx: StartupContext) -> None:
    """Select FTF/initial, creating template components and aliases on first use.

    ctx is the GUI startup capability, called after lab component registration.
    Use its project roots, not the shell cwd. A new entry gets the adjacent
    entry-template.yaml components in setup, then an initial point with those
    components and persisted aliases. Existing FTF/initial is opened unchanged.
    The template seeds MG12 mux gen1 and ADCs [1, 2] in resonator order.
    No SoC or device commands run, and no close callback is registered.

    A partial entry raises RuntimeError with recovery instructions. Missing
    initial, template, validation and I/O errors propagate to the startup
    runner; completed operations remain in place without rollback.
    """
    try:
        ctx.setup.new_entry("FTF")
    except DestinationExistsError as exists:
        ctx.progress("Reopening FTF entry")
        try:
            ctx.setup.open_entry("FTF")
        except FileNotFoundError as missing:
            # Never overwrite a partial entry left by an interrupted creation.
            raise RuntimeError(
                f"FTF is only partially present ({exists.filename} exists, "
                f"{missing.filename} is missing). Remove both FTF directories "
                "under result/ and Database/ to start fresh, or restore the "
                "missing one."
            ) from missing
        ctx.setup.use_point("initial")
    else:
        ctx.progress("Seeding FTF entry from template")
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
