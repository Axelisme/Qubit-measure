"""Accumulated acquisition progress through the public context-local factory."""

from collections.abc import Callable

from zcu_tools.gui.session.pbar_host import ProgressBar
from zcu_tools.gui.session.ports import ProgressEvent, ProgressEventKind
from zcu_tools.gui.session.services.progress import BoundProgressFactory
from zcu_tools.program.v2.modular import ModularProgramV2
from zcu_tools.program.v2.modules.readout import PulseReadoutCfg
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg
from zcu_tools.progress_bar import use_pbar_factory

from tests.program._acquisition_soc import make_acquisition_soc


class RecordingTransport:
    """Capture progress at its external delivery port without altering routing."""

    def __init__(self) -> None:
        self.events: list[ProgressEvent] = []
        self.receiver: Callable[[ProgressEvent], None] | None = None

    def emit(self, event: ProgressEvent) -> None:
        self.events.append(event)
        if self.receiver is not None:
            self.receiver(event)

    def set_receiver(self, receiver: Callable[[ProgressEvent], None]) -> None:
        self.receiver = receiver


def _acquire_two_records() -> None:
    soc, soccfg = make_acquisition_soc()
    program = ModularProgramV2(
        soccfg,
        modules=[
            PulseReadoutCfg(
                gen_ch=0,
                nqz=1,
                gain=0.1,
                waveform=ConstWaveformCfg(length=1.0),
                ro_ch=0,
                ro_length=1.0,
                freq=5000.0,
            ).build("ro")
        ],
        reps=500,
    )
    # One-round acquisitions expose reps progress rather than a rounds bar.
    for _ in range(2):
        program.acquire(soc, progress=True)


def test_acquire_routes_each_accumulated_reps_bar_through_factory() -> None:
    transport = RecordingTransport()
    bound = BoundProgressFactory(transport, operation_id=7)
    calls: list[dict[str, object]] = []
    bars: list[ProgressBar] = []

    def factory(**kwargs: object) -> ProgressBar:
        calls.append(kwargs)
        bar = bound(**kwargs)
        bars.append(bar)
        return bar

    with use_pbar_factory(factory):
        _acquire_two_records()

    reps_bars = [bar for bar in bars if bar.total == 500]
    assert len(reps_bars) == 2
    assert [bar.n for bar in reps_bars] == [500, 500]
    reps_calls = [call for call in calls if call.get("total") == 500]
    assert [call.get("leave") for call in reps_calls] == [False, False]
    closed = [
        event
        for event in transport.events
        if event.total == 500 and event.kind is ProgressEventKind.CLOSE
    ]
    assert len(closed) == 2
    assert [event.n for event in closed] == [500, 500]


def test_acquire_reps_emit_create_update_close_through_bound_factory() -> None:
    transport = RecordingTransport()
    factory = BoundProgressFactory(transport, operation_id=7)
    with use_pbar_factory(factory):
        _acquire_two_records()

    reps_events = [event for event in transport.events if event.total == 500]
    handles = {event.handle_id for event in reps_events}
    assert len(handles) == 2
    for handle in handles:
        kinds = [event.kind for event in reps_events if event.handle_id == handle]
        assert kinds[0] is ProgressEventKind.CREATE
        assert ProgressEventKind.UPDATE in kinds
        assert kinds[-1] is ProgressEventKind.CLOSE
