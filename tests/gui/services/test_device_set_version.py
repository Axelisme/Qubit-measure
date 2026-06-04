"""Device-set cardinality version key (set-membership concurrency guard).

A per-device ``device:<name>`` version cannot reveal a *newly added* device to
an opt-in concurrency check (the agent never declared a key for a device that
did not exist when it read versions). ``devices:__set__`` advances on set
grow / shrink so a whole-set dependant (run.start) detects it.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from zcu_tools.gui.app.main.state import (
    DEVICE_SET_VERSION_KEY,
    DeviceState,
    DeviceStatus,
    State,
)


def _make_state() -> State:
    # State only stores the ExpContext; these tests never touch it, so a stand-in
    # is sufficient (matches the existing test_state.py / test_device_snapshot.py
    # convention of State(MagicMock())).
    return State(MagicMock())


def _dev(name: str, status: DeviceStatus = DeviceStatus.CONNECTED) -> DeviceState:
    return DeviceState(
        name=name,
        type_name="YOKOGS200",
        address="addr",
        status=status,
        remember=False,
        info=None,
    )


def test_adding_a_new_device_bumps_set_version():
    st = _make_state()
    assert st.version.get(DEVICE_SET_VERSION_KEY) == 0
    st.put_device(_dev("yoko"))
    assert st.version.get(DEVICE_SET_VERSION_KEY) == 1
    st.put_device(_dev("sgs"))
    assert st.version.get(DEVICE_SET_VERSION_KEY) == 2


def test_removing_a_device_bumps_set_version():
    st = _make_state()
    st.put_device(_dev("yoko"))
    before = st.version.get(DEVICE_SET_VERSION_KEY)
    st.remove_device("yoko")
    assert st.version.get(DEVICE_SET_VERSION_KEY) == before + 1


def test_status_transition_does_not_bump_set_version():
    """Re-putting / status-editing an existing member leaves the set unchanged."""
    st = _make_state()
    st.put_device(_dev("yoko"))
    set_ver = st.version.get(DEVICE_SET_VERSION_KEY)
    dev_ver = st.version.get("device:yoko")

    # Status transition reuses the existing entry (set membership unchanged).
    st.set_device_status("yoko", DeviceStatus.SETTING_UP)
    assert st.version.get(DEVICE_SET_VERSION_KEY) == set_ver  # set unchanged
    assert st.version.get("device:yoko") == dev_ver + 1  # member moved

    # put_device on an existing name is a status transition, not a new member.
    st.put_device(_dev("yoko", status=DeviceStatus.CONNECTED))
    assert st.version.get(DEVICE_SET_VERSION_KEY) == set_ver  # still unchanged


def test_info_and_remember_edits_do_not_bump_set_version():
    st = _make_state()
    st.put_device(_dev("yoko"))
    set_ver = st.version.get(DEVICE_SET_VERSION_KEY)
    st.set_device_remember("yoko", True)
    assert st.version.get(DEVICE_SET_VERSION_KEY) == set_ver
