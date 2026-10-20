"""Explicit synthetic branch replacement for tests about host/view behaviour."""

from collections.abc import Mapping
from typing import ClassVar, Literal, get_args, get_origin

from pydantic import BaseModel, JsonValue
from zcu_tools.gui.form import (
    FormCommandContext,
    FormCommandSnapshot,
    FormConversionResult,
    FormEnv,
    FormModel,
    FormPath,
    InputConverter,
    static_defaults,
)
from zcu_tools.gui.form.commands import ReadonlyFormInput


class RebuildingForm(FormModel):
    """Synthetic forms explicitly opting into fresh branch defaults.

    branch_models maps relative selector paths to permitted replacement models.
    Tests using this base exercise rebuilding views/host behaviour, rather than
    Core's no-converter selection identity. It is not used by application code.
    """

    branch_models: ClassVar[Mapping[FormPath, tuple[type[BaseModel], ...]]] = {}

    @classmethod
    def converters(cls, env: FormEnv) -> Mapping[FormPath, InputConverter]:
        return {
            path: fresh_branches(env, models)
            for path, models in cls.branch_models.items()
        }


def fresh_branches(env: FormEnv, models: tuple[type[BaseModel], ...]) -> InputConverter:
    """Declare a synthetic default-producing converter with explicit branches."""

    def convert(
        _old: FormCommandSnapshot,
        selection: ReadonlyFormInput,
        _context: FormCommandContext,
    ) -> FormConversionResult:
        if selection is None:
            return FormConversionResult(None, None)
        if not isinstance(selection, Mapping):
            raise ValueError("Expected a synthetic model selector")
        for model in models:
            defaults: dict[str, JsonValue] = (
                model.default_input(env).data
                if issubclass(model, FormModel)
                else static_defaults(model)
            )
            if all(
                key in model.model_fields
                and (
                    value in get_args(model.model_fields[key].annotation)
                    if get_origin(model.model_fields[key].annotation) is Literal
                    else True
                )
                for key, value in selection.items()
            ):
                defaults.update(
                    {
                        key: value
                        for key, value in selection.items()
                        if isinstance(value, (str, int, float, bool)) or value is None
                    }
                )
                return FormConversionResult(model, defaults)
        raise ValueError("No declared synthetic branch matches selection")

    return InputConverter(convert)
