"""Recipe source reads distinguish absent registrations from other RPC failures."""

from pathlib import Path

import pytest
from pydantic import BaseModel, Field
from zcu_tools.gui.form import FormModel
from zcu_tools.gui.remote.errors import ErrorCode, RemoteError
from zcu_tools.mcp.measure.recipe_api import RecipeSession, recipe
from zcu_tools.mcp.measure.recipe_contracts import (
    MissingParameter,
    RecipeNeedsParameters,
)
from zcu_tools.mcp.measure.session import GuiRpcError

from ._script_recipe_support import SyntheticGui
from ._support import make_client


class FrequencyInputs(FormModel):
    frequency_mhz: float | None = Field(None, description="Explicit frequency in MHz")


class FrequencyReport(BaseModel):
    frequency_mhz: float = Field(description="Selected frequency in MHz")


@pytest.mark.parametrize(
    ("registered", "source_value", "explicit", "expected"),
    [
        (False, None, None, None),
        (False, None, 7001.0, 7001.0),
        (True, 7002.0, None, 7002.0),
        (True, None, None, None),
        (True, 7002.0, 7001.0, 7001.0),
    ],
)
def test_optional_frequency_source_or_explicit_input(
    tmp_path: Path,
    registered: bool,
    source_value: float | None,
    explicit: float | None,
    expected: float | None,
) -> None:
    gui = SyntheticGui()

    def respond(method: str, params: dict[str, object]) -> dict[str, object]:
        if method == "value.read":
            assert params == {"key": "$resonator.freq"}
            if not registered:
                raise RemoteError(
                    ErrorCode.INVALID_PARAMS,
                    "Unregistered calibration",
                    reason="unregistered_source",
                    data={"path": "$resonator.freq"},
                )
            return {"value": source_value}
        return gui.respond(method, params)

    @recipe("synthetic", "Require an explicit frequency or optional calibration")
    async def synthetic(
        session: RecipeSession, inputs: FrequencyInputs
    ) -> FrequencyReport:
        tab = session.open_tab("synthetic")
        frequency = inputs.frequency_mhz
        if frequency is None:
            frequency = session.context.read("$resonator.freq", missing_ok=True)
        if not isinstance(frequency, (int, float)):
            raise RecipeNeedsParameters(
                (MissingParameter("frequency_mhz", "Provide a frequency in MHz"),)
            )
        tab.set(("frequency",), frequency)
        await tab.run()
        return FrequencyReport(frequency_mhz=frequency)

    client = make_client(tmp_path, respond, recipes=(synthetic,))
    try:
        execution = client.context.session.recipes.start(
            client.context, "synthetic", {"frequency_mhz": explicit}
        )
        execution.wait(30)  # The synthetic responder completes every operation.
        snapshot = execution.snapshot()
        if expected is None:
            assert snapshot.status == "needs_parameters"
            assert snapshot.missing == ("frequency_mhz",)
            assert snapshot.outcome is None
            assert gui.runs == 0
        else:
            assert snapshot.status == "completed"
            assert snapshot.outcome is not None
            assert snapshot.outcome.report == FrequencyReport(frequency_mhz=expected)
            assert gui.runs == 1
        if explicit is not None:
            assert not any(
                method == "value.read" for method, _ in client.transport.sent
            )
    finally:
        client.context.session.close()


@pytest.mark.parametrize(
    ("code", "reason", "missing_ok"),
    [
        (ErrorCode.INVALID_PARAMS, "unregistered_source", False),
        (ErrorCode.INVALID_PARAMS, "", True),
        (ErrorCode.PRECONDITION_FAILED, "unavailable_source", True),
        (ErrorCode.CONTROLLER_ERROR, "provider_error", True),
        (ErrorCode.TIMEOUT, "", True),
    ],
)
def test_context_read_preserves_errors_outside_optional_missing_registration(
    tmp_path: Path, code: ErrorCode, reason: str, missing_ok: bool
) -> None:
    def respond(method: str, params: dict[str, object]) -> dict[str, object]:
        assert method == "value.read"
        raise RemoteError(code, "Source read failed", reason=reason)

    @recipe("synthetic", "Preserve source errors")
    async def synthetic(
        session: RecipeSession, inputs: FrequencyInputs
    ) -> FrequencyReport:
        with pytest.raises(GuiRpcError) as error:
            session.context.read("$resonator.freq", missing_ok=missing_ok)
        assert error.value.code == code.value
        assert (error.value.reason or "") == reason
        return FrequencyReport(frequency_mhz=7001.0)

    client = make_client(tmp_path, respond, recipes=(synthetic,))
    try:
        execution = client.context.session.recipes.start(
            client.context, "synthetic", {}
        )
        execution.wait(30)  # The source rejection is delivered synchronously.
        assert execution.snapshot().status == "completed"
    finally:
        client.context.session.close()
