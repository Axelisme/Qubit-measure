"""QICK transport adapter requiring an explicitly assembled simulation binding."""

import numpy as np
from qick import QickConfig

from zcu_tools.core.simulation import AcquireRequest

from .binding import SimulationBinding
from .signals import pack_signals


class SimulationSoc(QickConfig):
    """Bound offline SoC; no implicit noise mode and no driver ownership.

    routes is a tuple of (generator, ADC) pairs. The fixed capability profile is
    supplied by make_mock_soccfg; using an unbound route fails before acquisition.
    Getters retain the last completed realization, never execute or resample it.
    """

    def __init__(
        self,
        soccfg: QickConfig,
        binding: SimulationBinding,
        *,
        routes: tuple[tuple[int, int], ...],
    ) -> None:
        super().__init__(soccfg.get_cfg())
        self.binding = binding
        self.routes = routes
        for gen, adc in routes:
            if not 0 <= gen < len(self["gens"]) or not 0 <= adc < len(self["readouts"]):
                raise ValueError(f"invalid simulation route {(gen, adc)}")
        self._request: AcquireRequest | None = None
        self._round_index = 0
        self._accumulated = {}
        self._decimated = {}
        self._poll_done = False

    def install_request(self, request: AcquireRequest) -> None:
        """Admit detached controls only when every used channel is wired.

        request is borrowed until release_request. Validate every execution's
        generator and ADC routes before retaining it or invoking the backend.
        ValueError identifies an unbound generator's source module/channel, or
        an unbound generator/ADC route. This method does not generate signals.
        """
        bound_gens = {gen for gen, _ in self.routes}
        for template in request.templates:
            for event in template.events:
                if np.any(event.active) and event.gen_ch not in bound_gens:
                    raise ValueError(
                        f"{event.source!r} generator {event.gen_ch} is not bound"
                    )
            for window in template.readouts:
                if any(
                    (int(gen), window.channel) not in self.routes
                    for gen in np.unique(window.generator_channels)
                ):
                    raise ValueError("program uses an unbound generator/ADC route")
        self._request = request
        self.binding.last_request = request
        self._round_index = 0

    def release_request(self) -> None:
        self._request = None

    def _execute_round(self) -> None:
        if self._request is None:
            raise RuntimeError("simulation has no active acquisition request")
        signals = self.binding.sample_round(self._request, self._round_index)
        self._accumulated, self._decimated = pack_signals(self._request, signals)
        self._round_index += 1

    def start_tproc(self) -> None:
        self._execute_round()

    def start_readout(self, total_shots, counter_addr, ch_list, reads_per_shot) -> None:
        if self._request is None:
            raise RuntimeError("simulation has no active acquisition request")
        expected = [(w.channel, w.reads) for w in self._request.readouts]
        if (
            total_shots != self._request.executions
            or list(zip(ch_list, reads_per_shot, strict=True)) != expected
        ):
            raise ValueError("QICK readout layout differs from simulation request")
        del counter_addr
        self._poll_done = False
        self._execute_round()

    def poll_data(self, totaltime=0.1, timeout=None):
        del totaltime, timeout
        if self._poll_done or self._request is None:
            return []
        self._poll_done = True
        return [
            (
                self._request.executions,
                ([self._accumulated[w.channel] for w in self._request.readouts], {}),
            )
        ]

    def get_tproc_counter(self, addr: int) -> int:
        del addr
        return 2**31 - 1

    def get_decimated(self, ch, address, length):
        return self._decimated[ch][address : address + length].copy()

    def get_accumulated(self, ch, address, length):
        return self._accumulated[ch][address : address + length].copy()

    def stop_tproc(self, *args, **kwargs) -> None: ...
    def start_src(self, *args, **kwargs) -> None: ...
    def reload_mem(self, *args, **kwargs) -> None: ...
    def clear_tproc_counter(self, *args, **kwargs) -> None: ...
    def prepare_round(self, *args, **kwargs) -> None: ...
    def cleanup_round(self, *args, **kwargs) -> None: ...
    def load_bin_program(self, *args, **kwargs) -> None: ...
    def load_envelope(self, *args, **kwargs) -> None: ...
    def load_weights(self, *args, **kwargs) -> None: ...
    def load_pulse(self, *args, **kwargs) -> None: ...
    def load_pulses(self, *args, **kwargs) -> None: ...
    def set_nyquist(self, *args, **kwargs) -> None: ...
    def set_mixer_freq(self, *args, **kwargs) -> None: ...
    def config_mux_gen(self, *args, **kwargs) -> None: ...
    def configure_readout(self, *args, **kwargs) -> None: ...
    def config_mux_readout(self, *args, **kwargs) -> None: ...
    def config_avg(self, *args, **kwargs) -> None: ...
    def config_buf(self, *args, **kwargs) -> None: ...
    def enable_buf(self, *args, **kwargs) -> None: ...
