"""Persistent module input identities and their source/literal policies."""

from collections.abc import Mapping
from typing import Annotated, ClassVar, Literal

from pydantic import Field, JsonValue

from zcu_tools.cfg_model import ModuleSpec
from zcu_tools.gui.form import FormEnv, FormModel, InputPolicy, field_error


class KeyRef(FormModel):
    """Tagged key-only input, without source lookup or inline content.

    kind is always ref. name is a literal key string, including strings that
    look like expressions. Expr envelopes are invalid and never evaluated.
    Source validation, if required, belongs to a specialized ref subclass.
    """

    kind: Literal["ref"] = "ref"
    name: Annotated[str, InputPolicy(expression=False)]


class ModuleRef(KeyRef):
    """Library module/waveform reference without cached materialization.

    kind selects ref. name is an actual LibrarySnapshot key, not a role or alias.
    spec_kind selects the library namespace, module by default; generated waveform
    fields use waveform. A used ref needs captured library; missing/invalid names
    fail at the consuming path, never fall back to defaults.
    """

    spec_kind: ClassVar[Literal["module", "waveform"]] = "module"

    @classmethod
    def check_sources(cls, data: Mapping[str, JsonValue], env: FormEnv) -> None:
        """Require a captured Spec in spec_kind for a literal authored name.

        Missing/non-string names remain ordinary validation errors. Missing
        capture or rejected key raises field_error at name. Unexpected source
        exceptions propagate; this never materializes a native module.
        """
        name = data.get("name")
        if not isinstance(name, str):
            return
        if env.library is None:
            raise field_error(("name",), "Missing captured library")
        try:
            if cls.spec_kind == "waveform":
                env.library.waveform_spec(name)
            else:
                env.library.module_spec(name)
        except KeyError as exc:
            raise field_error(
                ("name",),
                f"{cls.spec_kind.capitalize()} {name!r} is not in the captured library",
            ) from exc
        except ValueError as exc:
            raise field_error(("name",), str(exc)) from exc


class ModuleInline(FormModel):
    """Detached inline Spec, not a runtime cfg or linked library cache.

    kind selects inline. module is the ordinary cfg_model ModuleSpec preserving
    its own references. Adapter to_cfg owns materialization/family requirements.
    """

    kind: Literal["inline"] = "inline"
    module: ModuleSpec


type ModuleInput = Annotated[ModuleRef | ModuleInline, Field(discriminator="kind")]
"""Reference or inline Spec with wire {kind: inline, module: {...}}."""
