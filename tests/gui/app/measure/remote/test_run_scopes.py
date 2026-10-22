"""Tab result completion facts cross State, render snapshot and RPC unchanged."""


def test_tab_snapshot_keeps_partial_fact_without_metadata_and_resets_on_replacement(
    form_rpc,
) -> None:
    fixture, tab_id, rpc = form_rpc
    state = fixture.state
    state.update_tab_result(tab_id, object(), partial_result=True)
    snapshot = fixture.ctrl.get_tab_snapshot(tab_id)
    assert snapshot.run is not None
    assert snapshot.run.partial_result
    response = rpc.call("tab.snapshot", {"tab_id": tab_id})
    assert response["ok"], response
    assert response["result"]["tabs"][0]["result_state"]["partial_result"] is True
    assert response["result"]["tabs"][0]["raw_data"]["run_id"] is None
    state.update_tab_result(tab_id, object())
    response = rpc.call("tab.snapshot", {"tab_id": tab_id})
    assert response["ok"], response
    assert response["result"]["tabs"][0]["result_state"]["partial_result"] is False
