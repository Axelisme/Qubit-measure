from __future__ import annotations

import pytest
from zcu_tools.gui.app.main.services.cfg_editor import CfgEditorError
from zcu_tools.gui.app.main.services.guard import GuardError
from zcu_tools.gui.app.main.services.load import LoadDataError
from zcu_tools.gui.expected_error import (
    ExpectedError,
    ExpectedErrorCategory,
    FailedPreconditionError,
    InvalidInputError,
)
from zcu_tools.gui.session.ports import OperationConflictError
from zcu_tools.gui.session.services.connection import SoCConnectionError
from zcu_tools.gui.session.services.context import (
    MdValueError,
    MlEntryValidationError,
)
from zcu_tools.gui.session.services.device import DeviceRegistrationError
from zcu_tools.gui.session.services.predictor import (
    PredictorLoadError,
    PredictorNotLoaded,
)
from zcu_tools.gui.session.value_lookup import (
    DuplicateValueKey,
    MissingValue,
    ProviderError,
    UnavailableValue,
    ValueLookupError,
    ValueTypeError,
)


def test_expected_error_categories_are_closed_and_wire_independent() -> None:
    assert list(ExpectedErrorCategory) == [
        ExpectedErrorCategory.INVALID_INPUT,
        ExpectedErrorCategory.FAILED_PRECONDITION,
    ]
    assert [category.value for category in ExpectedErrorCategory] == [
        "invalid_input",
        "failed_precondition",
    ]


@pytest.mark.parametrize(
    ("error", "category", "reason"),
    [
        (
            InvalidInputError("bad input"),
            ExpectedErrorCategory.INVALID_INPUT,
            "",
        ),
        (
            FailedPreconditionError("not ready"),
            ExpectedErrorCategory.FAILED_PRECONDITION,
            "",
        ),
        (
            FailedPreconditionError("not ready", reason_code="no_context"),
            ExpectedErrorCategory.FAILED_PRECONDITION,
            "no_context",
        ),
    ],
)
def test_runtime_compatible_expected_error_contract(
    error: ExpectedError,
    category: ExpectedErrorCategory,
    reason: str,
) -> None:
    assert isinstance(error, RuntimeError)
    assert error.category is category
    assert error.reason_code == reason
    assert str(error) == error.args[0]


def test_plain_runtime_error_does_not_opt_in() -> None:
    assert not isinstance(RuntimeError("programmer bug"), ExpectedError)


@pytest.mark.parametrize(
    "error",
    [
        GuardError("guard", reason_code="no_run_result"),
        LoadDataError("load", reason_code="invalid_data_file"),
        OperationConflictError("busy"),
        PredictorLoadError("bad predictor"),
        PredictorNotLoaded("no predictor"),
    ],
)
def test_named_failed_precondition_errors_preserve_runtime_ancestry(
    error: ExpectedError,
) -> None:
    assert isinstance(error, RuntimeError)
    assert error.category is ExpectedErrorCategory.FAILED_PRECONDITION


@pytest.mark.parametrize(
    "error",
    [
        CfgEditorError("bad editor"),
        MlEntryValidationError("bad module"),
        MissingValue("missing", "missing value"),
        ValueTypeError("wrong", "wrong type"),
    ],
)
def test_named_invalid_input_errors_opt_in_explicitly(error: ExpectedError) -> None:
    assert error.category is ExpectedErrorCategory.INVALID_INPUT


def test_md_value_error_preserves_value_error_ancestry() -> None:
    error = MdValueError("bad value")

    assert isinstance(error, ValueError)
    assert isinstance(error, ExpectedError)
    assert error.category is ExpectedErrorCategory.INVALID_INPUT
    assert error.reason_code == ""


def test_unavailable_value_is_failed_precondition() -> None:
    error = UnavailableValue("device.flux", "device is unavailable")

    assert isinstance(error, ExpectedError)
    assert error.category is ExpectedErrorCategory.FAILED_PRECONDITION


@pytest.mark.parametrize(
    "error",
    [
        ValueLookupError("key", "base error"),
        ProviderError("key", "owner", RuntimeError("provider bug")),
        DuplicateValueKey("key", "duplicate"),
        SoCConnectionError("connect failed"),
        DeviceRegistrationError("registration failed"),
    ],
)
def test_operational_invariant_and_mixed_parent_errors_do_not_opt_in(
    error: BaseException,
) -> None:
    assert not isinstance(error, ExpectedError)
