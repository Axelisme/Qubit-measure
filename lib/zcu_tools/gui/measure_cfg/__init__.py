"""Measure-domain program cfg shapes shared by GUI applications."""

from .catalog import (
    PROGRAM_SHAPES,
    ProgramCfgKind,
    ProgramShape,
    ProgramShapeCatalog,
    ProgramSpecPolicy,
    UnknownProgramShapeError,
)
from .materializer import (
    ProgramMaterializationPolicy,
    materialize_program_module,
    materialize_program_waveform,
)

__all__ = [
    "PROGRAM_SHAPES",
    "ProgramCfgKind",
    "ProgramMaterializationPolicy",
    "ProgramShape",
    "ProgramShapeCatalog",
    "ProgramSpecPolicy",
    "UnknownProgramShapeError",
    "materialize_program_module",
    "materialize_program_waveform",
]
