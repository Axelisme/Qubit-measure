"""Detached authored-input projections; the Form owner retains its native codec."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TabInput:
    """Opaque authored input produced by FormInput.to_json(), not a session schema.

    form_json preserves raw data, expressions, ordinary invalid drafts and
    explicit selections. It contains no model class, sources or results. Callers
    may transfer/store it unchanged; only the Form owner decodes and restores it.
    Malformed JSON is rejected by new_tab before publication. Incompatible paths
    use the Form owner's restoration policy and are logged as discarded paths.
    """

    form_json: str


@dataclass(frozen=True)
class StartupTab:
    """One live tab's detached authored input in visible workspace order.

    adapter_name is its registered catalog name. input is the current opaque
    FormInput encoding, including invalid drafts; modifying the live tab later
    cannot change this value. It does not include sources, results or focus.
    """

    adapter_name: str
    input: TabInput
