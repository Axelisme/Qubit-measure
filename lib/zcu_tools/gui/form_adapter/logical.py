"""Detached logical form contracts; hosts own capture, guards and publication."""

from dataclasses import dataclass
from typing import Literal

from pydantic import JsonValue

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.core.element import ElementIdentity
from zcu_tools.gui.expected_error import FailedPreconditionError, InvalidInputError
from zcu_tools.gui.form import (
    FormDescription,
    FormErrors,
    FormInput,
    FormModel,
    FormValue,
    FormWritePath,
)


@dataclass(frozen=True)
class ModuleSelector:
    """Generator-owned annotation metadata, without environment or state.

    inline_field names the generated content wrapper. specs is the nonempty
    persistent inline family; ref_specs is its nonempty captured reference
    subset. source_kind chooses the module/waveform library namespace.
    """

    inline_field: str
    specs: tuple[type[ConfigBase], ...]
    ref_specs: tuple[type[ConfigBase], ...]
    source_kind: Literal["module", "waveform"]


@dataclass(frozen=True)
class LogicalEdit:
    """One logical field and its exact authored JSON replacement, Expr retained.

    A model/container target replaces its entire raw subtree without merging
    defaults; leaf targets replace only that leaf. Generated selector-bearing
    containers require the dedicated selection operations instead. Paths in a
    batch must be nonoverlapping. No old branch is evaluated or converted.
    """

    path: FormWritePath
    value: JsonValue


@dataclass(frozen=True)
class LogicalEditCandidate:
    """Complete detached input candidate and content-change flag, not valid cfg.

    input retains the incoming model specialization. changed compares authored
    content only. Install only in the same host-guarded turn/current input/env.
    """

    input: FormInput[FormModel]
    changed: bool


@dataclass(frozen=True)
class LogicalSelector:
    """Target selector identity; never inherited by ordinary descendants.

    mode distinguishes explicit key/inline/null from missing and unresolved
    input. key is the actual string only for key mode, otherwise None.
    available_keys lists captured family-compatible keys in library order;
    it says nothing about declaration locks.
    """

    mode: Literal["key", "inline", "disabled", "missing", "unresolved"]
    key: str | None
    available_keys: tuple[str, ...]


@dataclass(frozen=True)
class LogicalSource:
    """Nearest captured ref supplying a value, not authored input.

    selector_path locates that ref logically; key is its actual library key.
    source_kind identifies its namespace. spec_path locates the requested value
    from that persistent Spec root, without generated wrapper names.
    """

    selector_path: FormWritePath
    key: str
    source_kind: Literal["module", "waveform"]
    spec_path: FormWritePath


@dataclass(frozen=True)
class LogicalFieldObservation:
    """One detached logical read from a fixed input/environment.

    description uses logical paths for target and direct children. input_present
    distinguishes authored missing from explicit null; input is authored JSON,
    never projection data. value_status distinguishes available (including null),
    missing and unavailable; only available has a meaningful ordinary value.
    selector describes only a generated target. source identifies the nearest
    supplying captured ref, or None. readonly means a ref ancestor was crossed,
    without altering declaration fixed/selection_fixed metadata. errors contains
    only relevant directed errors and root/ancestor failures, or None. element
    is the declaring FixedElements identity from this same input, or None for
    fields outside a declared element. It is not inferred from labels or refs.
    """

    description: FormDescription
    input_present: bool
    input: JsonValue
    value_status: Literal["available", "missing", "unavailable"]
    value: FormValue
    selector: LogicalSelector | None
    source: LogicalSource | None
    readonly: bool
    errors: FormErrors | None
    element: ElementIdentity | None = None


type LogicalEditReason = Literal[
    "form_path",
    "form_option",
    "ref_ancestor",
    "module_selector",
    "library_key",
    "module_state",
    "form_structure",
    "form_source",
]


class LogicalEditError(InvalidInputError):
    """Rejected logical mutation; no prefix of a batch has been installed.

    reason_code classifies path/option/ref ancestry/selector/key/state/structure/source
    admission. path is the full request; edit_index is its original zero-based
    batch index, or None for selector operations. selector_path identifies the
    blocking selector, or None. available_keys is None when irrelevant; an empty
    tuple means an actual empty menu. message is diagnostic only.
    """

    def __init__(
        self,
        message: str,
        *,
        reason_code: LogicalEditReason,
        path: FormWritePath,
        edit_index: int | None = None,
        selector_path: FormWritePath | None = None,
        available_keys: tuple[str, ...] | None = None,
    ) -> None:
        """Store the structured rejection and diagnostic message."""
        super().__init__(message, reason_code=reason_code)
        self.path = path
        self.edit_index = edit_index
        self.selector_path = selector_path
        self.available_keys = available_keys


class LogicalSourceError(FailedPreconditionError):
    """A requested dereference/copy cannot read its captured Spec.

    reason_code is library_source. path is the full logical request;
    selector_path locates the failing ref; key is its literal string or None.
    source_kind chooses the captured namespace. Expected library failures retain
    their cause; unexpected provider failures propagate unchanged.
    """

    def __init__(
        self,
        message: str,
        *,
        path: FormWritePath,
        selector_path: FormWritePath,
        key: str | None,
        source_kind: Literal["module", "waveform"],
    ) -> None:
        """Store the consuming request and captured-source identity."""
        super().__init__(message, reason_code="library_source")
        self.path = path
        self.selector_path = selector_path
        self.key = key
        self.source_kind = source_kind
