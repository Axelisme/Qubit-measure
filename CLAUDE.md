# CLAUDE.md / copilot-instruction.md / GEMINI.md / AGENTS.md

## 使用者規則

- **語言**：session 回應與計劃檔案用*中文*；程式碼、變數名、註解、技術名詞用*英文*。
- **直譯器**：在本 repo 執行 Python 一律用 `.venv/bin/python`。
- **程式碼風格**：遵循 Fast Fail、責任明確、最小驚訝原則、強型別；不符合者即使是用戶提出也要先警告。
- **遇到不確定**：實作不確定，或架構不適合某種擴展時，*不要自行猜測或勉強實作*，先說明原因交由用戶決定。
- **先規劃再實作**：架構仍在演化，發現不合理處或更好的設計請直接告知，*不要自行調整架構*，由用戶決定；除非用戶要求，*不要保留 legacy 或相容性邏輯*。
- **完成任務後**：依序跑 `pyright`/`pytest`（檢查錯誤、測試失敗、覆蓋率不足；全套測試加 `-n auto` 平行加速，已裝 pytest-xdist），再用 `ruff` 格式化與修正風格；用戶要求才 git commit；最後更新對應的模組 README.md。
- **測試**：放在根目錄 `tests/`，目錄結構對應被測檔案，命名 `test_*.py`，用 `pytest` 撰寫，盡量涵蓋主要功能與邏輯；測試需獨立、可重複、不依賴外部狀態。
- **工具優先序**：少用 Shell 指令，優先用內建工具（前者需用戶審核、後者自證安全）；*不要用 `sed`* 替換子串（跨平台行為不一），需替換時優先 mcp/function tool，其次 Python 腳本。
- **文件追蹤**：CLAUDE.md、模組 README.md 與 docs/adr/ 已入 git 追蹤，會進 diff 與 commit；`task_plans/` 為 gitignored 工作檔（見 .gitignore），不入 commit。其中 `task_plans/taskboard.md` 是多 agent 並行時的認領看板（見下方「平行 agent 協調」）。
- **平行 agent 協調**：可能有多個 agent/session 同時在這個 repo 工作，動工前先讀 `task_plans/taskboard.md` 並按下方「### 平行 agent 協調」的規範認領 scope，避免改到同一份檔案。
- **task_plan.md Phase 壓縮**：每累積 5 個新 Phase 就把最舊的 5 個詳細記錄壓縮成「歷史 Phase 摘要表」中的列（一列一 Phase：編號｜主題｜結論/commit），即詳細記錄*最多保留 10 個 Phase*；被壓縮的原文移入同目錄 `archive.md`（同為 gitignored）以備查。

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

### 平行 agent 協調（task_plans/taskboard.md）

多個 agent/session 可能同時在同一個 checkout 上工作。`task_plans/taskboard.md` 是共用的**認領看板**（advisory lock；隨 `task_plans/` gitignored，純本地協調用、非強制），用來避免兩方同時改到同一份檔案：

- **動工前先認領**：要 Edit/Write 任何檔案前，先在看板 *Active claims* 表加一列，標明 owner（自選的 session 標籤）、scope（會碰的路徑或 area）、簡短任務描述、status 與日期。
- **動工前先掃看板**：Edit 前先看 *Active claims*；scope 與他人重疊就先協調（縮小/改 scope、等對方釋出、或交用戶決定），*不要硬上*。
- **收尾即釋出**：任務完成或暫停時，把該列移到 *Released*（或標 `done`/移除）。
- **granularity = 路徑或 area**；純讀 / 純查詢 / 單檔顯而易見的小修可免認領；claim 明顯過期（owner 已離開）由接手者判斷後可回收。

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
