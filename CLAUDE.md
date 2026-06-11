# CLAUDE.md / copilot-instruction.md / GEMINI.md / AGENTS.md

## 使用者規則

- **語言**：session 回應與計劃檔案用*中文*；程式碼、變數名、註解、技術名詞用*英文*。
- **直譯器**：在本 repo 執行 Python 一律用 `.venv/bin/python`。
- **程式碼風格**：遵循 Fast Fail、責任明確、最小驚訝原則、強型別；不符合者即使是用戶提出也要先警告。
- **遇到不確定**：實作不確定，或架構不適合某種擴展時，*不要自行猜測或勉強實作*，先說明原因交由用戶決定。
- **先規劃再實作**：架構仍在演化，發現不合理處或更好的設計請直接告知，*不要自行調整架構*，由用戶決定；除非用戶要求，*不要保留 legacy 或相容性邏輯*。
- **完成任務後**：依序跑 `pyright`/`pytest`（檢查錯誤、測試失敗、覆蓋率不足），再用 `ruff` 格式化與修正風格；用戶要求才 git commit；最後更新對應的模組 README.md。
- **測試**：放在根目錄 `tests/`，目錄結構對應被測檔案，命名 `test_*.py`，用 `pytest` 撰寫，盡量涵蓋主要功能與邏輯；測試需獨立、可重複、不依賴外部狀態。
- **工具優先序**：少用 Shell 指令，優先用內建工具（前者需用戶審核、後者自證安全）；*不要用 `sed`* 替換子串（跨平台行為不一），需替換時優先 mcp/function tool，其次 Python 腳本。
- **文件追蹤**：CLAUDE.md、模組 README.md、docs/adr/ 與 task_plans/ 均已入 git 追蹤，會進 diff 與 commit；工具可直接讀寫，亦可視需要加入 commit。

### 模組 README.md

本慣例專指 **lib/ 與 tests/ 子目錄**內的 `README.md`（各模組的高層 cheat-sheet），與以下兩者有別：
- **root `README.md`**：專案主 readme，由 pyproject.toml 的 `readme =` 引用，不屬此慣例。
- **`docs/adr/README.md`**：ADR 索引，見下方 `### docs/adr/` 段說明。

模組 README.md 的使用規範：
- 是各模組的高層 cheat-sheet：修改該模組程式碼前先讀它建立 context。
- 學到非顯而易見、有助未來 session 且尚未記錄的知識時，更新對應的模組 README.md；發現 note 與程式碼不符時，告知用戶並更新。
- *只寫高層概念、架構、重要設計決策，不寫實作細節*（實作細節留在程式碼註解與文件）。
- 用現在式描述目前狀態，*不要用更新式語法*（如「已經更新…」「在之前的實作中…」），以免過時。
- 每次更新都刷新檔案頂部的 `**Last updated:** YYYY-MM-DD`（可在日期後附簡短主題/Phase 標題；不要寫 commit hash，難維護）。

### docs/adr/

- *跨模組*設計決策記錄在 `docs/adr/`（模組 README.md 只管模組局部）；`docs/adr/README.md` 是依主題分組的索引，每篇 ADR 以現在式描述目前生效的設計。
- 程式碼註解與記憶檔以 `ADR-NNNN` 引用 ADR，ADR 之間以 `[[NNNN]]` 互鏈。
- 處理跨模組設計前先查索引找相關 ADR。

## 專案概觀

ZCU-Tools 是控制與分析 ZCU216 FPGA 平台上超導量子位元（Fluxonium）量測的 Python 工具集，建構於 [QICK](https://github.com/openquantumhardware/qick) 框架之上。架構分為 ZCU 端 server（跑在 FPGA 板）與 client 端 controller（跑在工作站）。client 端 Python 版本為 3.13（見 `.python-version`）；ZCU 板端跑 PYNQ 自帶的 Python 3.8，只用 `script/start_server.py`（該檔需維持 3.8 相容）。

## 架構（`lib/zcu_tools/`）

- **`experiment/v2/`** — 高層實驗定義：`onetone/`、`twotone/`（spectroscopy）；`autofluxdep/`（`FluxDepExecutor` 自動多任務 flux 掃描）；`overnight/`（長時穩定度量測）；`runner/`（`Task`/`BatchTask`/`AbsTask` 執行框架）。
- **`program/v2/`** — 低層 QICK ASM 編程：`base.py`（`MyProgramV2`）、`modular.py`（`ModularProgramV2` 組合 pulse blocks）、`modules/`（`Pulse`/`Readout`/`Reset`/`Delay`/`Waveform`/`dmem`/`registry`）。
- **`meta_tool/`** — 設定與狀態管理：`MetaDict`（JSON 持久化 dict）、`ModuleLibrary`（YAML waveform/module store）、`ExperimentManager`（串接兩者的 context manager）。
- **`device/`** — 硬體驅動（`YOKOGS200`、`RohdeSchwarzSGS100A` 等）與 `GlobalDeviceManager` singleton。
- **`notebook/`** — 分析工具：flux-dependent fitting、dispersive shift、IQ discrimination、互動式選點。
- **`liveplot/`** — 資料擷取時的即時 Jupyter 繪圖（`LivePlot1D`、`LivePlot2D`）。
- **`simulate/`** — Fluxonium Hamiltonian 模擬與 noise model 計算。
- **`utils/`** — HDF5 資料持久化（`datasaver.py`）、curve fitting、數學工具。
