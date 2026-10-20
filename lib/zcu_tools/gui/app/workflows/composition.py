"""Caller-owned workflow sources shared by Behavior and embedded Window."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from zcu_tools.cfg_model import TemplateDef
from zcu_tools.gui.catalog_loader import SourceCatalogLoader
from zcu_tools.gui.session.alias_presets import AliasPresets

from .attachment import Attachment
from .catalog import WorkflowCatalog


@dataclass(frozen=True)
class WorkflowComposition[C]:
    """Workflow/form sources and presentation catalogs; owns no live session.

    registrations is a nonempty ordered collection of coherent attachments
    sharing opaque context C. The first is default; Behavior and Window reject
    empty or duplicate names when constructed. templates is the caller's
    persistent module/waveform template catalog for Inspect's New action;
    an empty mapping is valid. alias_presets retains the full named catalog
    of ordered alias rows. None means no catalog; an explicit mapping must
    contain "default" (possibly empty), and Behavior validates row syntax.
    loader optionally supplies replacement WorkflowCatalog instances through
    bounded caller-configured namespaces; None disables source reload.
    Constructing this record performs no validation, I/O or Qt construction.
    """

    registrations: Sequence[Attachment[C]]
    templates: Mapping[str, TemplateDef]
    alias_presets: AliasPresets | None = None
    loader: SourceCatalogLoader[WorkflowCatalog[C]] | None = None
