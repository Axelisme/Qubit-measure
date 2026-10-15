from pathlib import Path

import pytest
from zcu_tools.gui.session.services import arb_waveform
from zcu_tools.gui.session.state import SessionState
from zcu_tools.gui.session.types import SessionEnv
from zcu_tools.gui.session.ui.arb_waveform_dialog import ArbWaveformDialog
from zcu_tools.resources.waveform_assets import ArbWaveformDatabase


@pytest.fixture
def entry_waveform_dialog(qapp, tmp_path: Path):
    """Bind the waveform service to an entry installed by the real context owner."""
    from zcu_tools.gui.event_bus import BaseEventBus
    from zcu_tools.gui.session.services.context import ContextService
    from zcu_tools.resources.entry import ResultEntry

    state = SessionState(SessionEnv(parameters=None, soc=None, soccfg=None))
    context = ContextService(
        state, BaseEventBus(), bind_assets=ArbWaveformDatabase.info_at
    )
    entry = ResultEntry.create(
        "entry", result_root=tmp_path / "result", database_root=tmp_path / "Database"
    )
    context.install_entry(entry)
    assets = arb_waveform.ArbWaveformService(state)
    dialog = ArbWaveformDialog(assets)
    dialog.show()
    qapp.processEvents()
    try:
        yield dialog, assets
    finally:
        dialog.close()
        dialog.deleteLater()
        qapp.processEvents()
