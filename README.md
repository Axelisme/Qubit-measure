# ZCU-Tools

**Last updated:** 2026-07-02 — experiment runtime map

ZCU-Tools 是 ZCU216/QICK 平台上的量子量測工具集。工作站端負責 GUI、
Notebook、MCP automation、資料分析與模擬；ZCU 板端只跑 Pyro server，讓工作站
透過 QICK 控制 FPGA 與量測流程。

## Runtime Profiles

- Python 3.13（repo 預設）：GUI、量測 runtime、MCP bridge，使用 NumPy 2.x。
- Python 3.12：`design` / `quantum-metal` / Ansys stack，使用 NumPy 1.26.x。
- ZCU 板端：PYNQ Python 3.8，只使用 `script/start_server.py`；這個腳本維持
  Python 3.8 相容。

## Install

```bash
uv sync --extra gui
uv sync --extra client
uv sync --extra all
uv sync --python 3.12 --extra all  # includes design stack when needed
```

`qick` 由 `client` extra 從 upstream Git 安裝。板端若不使用同一個 uv 環境，需在
板上另行安裝或放置 QICK。

## Main Entry Points

```bash
.venv/bin/python script/start_server.py --port <port> --soc v2
.venv/bin/python script/run_measure_gui.py
.venv/bin/python script/run_fluxdep_gui.py
.venv/bin/python script/run_dispersive_gui.py
.venv/bin/python script/run_autofluxdep_gui.py
.venv/bin/python script/generate_fluxonium_sample.py --help
.venv/bin/python script/migrate_experiment_data.py --help
```

`run_measure_gui.py` 是主要量測 GUI；`fluxdep` / `dispersive` GUI 負責
Fluxonium 參數與讀出腔擬合；`autofluxdep` GUI 負責自動 flux sweep workflow。
Notebook 仍可直接呼叫 `zcu_tools.experiment.v2` 與 `zcu_tools.notebook` helper。

## Package Map

- `zcu_tools.program.v2`：QICK ASM / modular pulse / IR / mock SoC。
- `zcu_tools.experiment.v2`：Notebook 與 GUI adapter 共用的實驗實作、
  Schedule-based acquisition runtime、canonical persistence。
- `zcu_tools.experiment.v2_gui`：把 experiment 包成 measure-gui adapter。
- `zcu_tools.gui`：Qt GUI framework、shared session core、shared remote transport。
- `zcu_tools.mcp`：GUI-facing MCP bridge 與 agent-memory server。
- `zcu_tools.meta_tool`：`ExperimentManager`、`MetaDict`、`ModuleLibrary`、arbitrary
  waveform asset store。
- `zcu_tools.device`：儀器 driver 與 `GlobalDeviceManager`。
- `zcu_tools.analysis` / `zcu_tools.notebook.analysis`：GUI-neutral analysis kernel
  與 notebook-facing workflow。
- `zcu_tools.simulate.fluxonium`：Fluxonium prediction engine。
- `zcu_tools.liveplot`：Notebook / experiment runtime live plotting。
- `zcu_tools.utils.datasaver`：Labber-style HDF5 persistence facade。

## Data Layout

- `result/<chip>/<qub>/params.json` 是 project scope 的身分與 handoff 檔。
- `result/<chip>/<qub>/...` 放分析輸出、圖片、GUI state 與 context-local metadata。
- `Database/<chip>/<qub>/...` 放 canonical experiment data file。
- `ModuleLibrary` 與 `MetaDict` 由 `ExperimentManager` 管理；GUI 和 Notebook 共用同一個
  project/context 概念。

Experiment runtime 只讀寫 canonical HDF5。Legacy artifact 透過 migration script 轉換，
不作為一般 runtime loading format。

## Documentation Map

- `CLAUDE.md` / `AGENTS.md`：repo 操作規則、agent 角色、文件更新規則。
- `docs/CONTEXT.md`：跨模組 domain language。
- `docs/adr/`：跨模組架構決策。
- 各 `lib/**/README.md` 與 `tests/README.md`：模組 cheat-sheet；修改該模組前先讀。

## Quality Gates

```bash
uv run pyright
uv run pytest -n auto
uv run ruff check --select I --fix
uv run ruff format
```

本 repo 執行 Python 腳本時使用 `.venv/bin/python`。測試放在 `tests/`，路徑對應被測
模組，使用 pytest，避免外部狀態依賴。
