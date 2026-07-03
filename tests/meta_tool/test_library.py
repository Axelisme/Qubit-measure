from __future__ import annotations

from typing import Any

from zcu_tools.meta_tool.library import (
    ModuleCfgFactory,
    ModuleLibrary,
    WaveformCfgFactory,
)


def test_module_library_load_passes_self_to_cfg_factories(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "module_cfg.yaml"
    path.write_text(
        "\n".join(
            [
                "waveforms:",
                "  wav:",
                "    style: const",
                "    length: 1.0",
                "modules:",
                "  mod:",
                "    type: reset/none",
            ]
        ),
        encoding="utf-8",
    )
    calls: list[tuple[str, ModuleLibrary | None]] = []

    def fake_waveform_from_raw(raw: object, *, ml: ModuleLibrary | None = None) -> Any:
        calls.append(("waveform", ml))
        return raw

    def fake_module_from_raw(raw: object, *, ml: ModuleLibrary | None = None) -> Any:
        calls.append(("module", ml))
        return raw

    monkeypatch.setattr(
        WaveformCfgFactory, "from_raw", staticmethod(fake_waveform_from_raw)
    )
    monkeypatch.setattr(
        ModuleCfgFactory, "from_raw", staticmethod(fake_module_from_raw)
    )

    ml = ModuleLibrary(path)

    assert calls == [("waveform", ml), ("module", ml)]
