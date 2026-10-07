"""Explicit legacy Labber import for the offline storage migration caller."""

from pathlib import Path

import h5py

from .grouped import GROUPED_VERSION_ATTR, load_grouped_labber_data
from .labber import load_labber_data
from .labber_schema import cast_labber_values
from .models import LabberPayload
from .native import validate_experiment_payload
from .native_models import ExperimentPayload, VariableSchema


def load_legacy_labber_payload(
    source: Path, *, schema: tuple[VariableSchema, ...]
) -> ExperimentPayload:
    """Read supported legacy Labber layouts using an explicit disk schema.

    source is an exact existing Labber file; schema names its experiment's
    variables, axes, signal labels, SI/discrete units and numeric dtypes.
    Marked grouped files use the canonical/streaming grouped reader; unmarked
    files must declare one variable and use the single-log reader. No filename
    or channel guesses, unit conversion or runtime fallback is performed.
    Return a schema-validated generic payload with source metadata preserved.
    Numeric reader containers are cast through the Labber dtype contract.
    Unsupported layout/schema raises ValueError; filesystem errors propagate.
    This migration-only entry never modifies source.
    """
    if not schema:
        raise ValueError(f"{source}: missing disk schema")
    with h5py.File(source, "r") as file:
        grouped = GROUPED_VERSION_ATTR in file.attrs
    if grouped:
        loaded = load_grouped_labber_data(
            str(source), required_variables=[item.variable for item in schema]
        )
        variables = loaded.variables
        metadata = loaded.metadata
    else:
        if len(schema) != 1:
            raise ValueError(
                f"{source}: unmarked file cannot identify multiple variables"
            )
        single = load_labber_data(str(source))
        variables = {schema[0].variable: single.payload}
        metadata = single.metadata
    converted = {
        item.variable: LabberPayload(
            (
                variables[item.variable].data.name,
                variables[item.variable].data.unit,
                cast_labber_values(
                    variables[item.variable].z,
                    item.signal_dtype,
                    context=f"{source}/{item.variable}",
                ),
            ),
            [
                (
                    actual.name,
                    actual.unit,
                    cast_labber_values(
                        actual.values,
                        declared.dtype,
                        context=f"{source}/{item.variable}/{actual.name}",
                    ),
                )
                for actual, declared in zip(
                    variables[item.variable].axes, item.axes, strict=True
                )
            ],
            timestamps=variables[item.variable].timestamps,
        )
        for item in schema
    }
    payload = ExperimentPayload(
        variables=converted,
        metadata=metadata,
        representation="grouped" if grouped else "single",
    )
    validate_experiment_payload(payload, schema=schema)
    return payload
