"""Synthetic tagged module input for core source and editing contracts.

The pulse subset keeps an unresolved waveform key, without admitting the
retired untagged key-or-inline union of the persistent PulseSpec.
"""

from typing import Annotated

from pydantic import Field
from zcu_tools.cfg_model import PulseSpec
from zcu_tools.gui.form_adapter import ModuleInline, ModuleRef


class KeyPulseSpec(PulseSpec):
    """A pulse with a literal child reference and no inline waveform branch."""

    waveform: str


class SyntheticModuleInline(ModuleInline):
    """Use the core defaults hook with a supported, key-only pulse subset."""

    module: KeyPulseSpec


type SyntheticModuleInput = Annotated[
    ModuleRef | SyntheticModuleInline, Field(discriminator="kind")
]
