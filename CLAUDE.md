# CLAUDE.md / copilot-instruction.md / GEMINI.md / AGENTS.md

## 角色判定（最先讀）

本 repo 的 agent 分兩種角色，先判定自己屬於哪一種：

- **MEASUREMENT（操作儀器跑量測）**：任務是透過 `measure-gui` MCP 工具（`mcp__measure-gui__gui_*`）驅動量測。
  → 遵循 `run-measure-gui` skill 與該 MCP server 的 instructions；持久量測知識（記錄/疑難排解/驗收清單）
  寫入 `agent-memory`（你的筆記本），不用其它或內建 memory 機制；你是 **operator 不是 developer**，
  **不要讀或改 `lib/` 原始碼**（要驗證就放寬參數重跑量測、看圖判斷，不要 grep 實作——引用實作細節有過時風險）；
  下方「## 使用者規則」起的開發守則**不適用**，只有「語言」規則照常（中文回應）。GUI「Agent」按鈕啟動的
  session 一律是此角色。
- **DEVELOPMENT（開發/維護程式碼）**：其餘所有任務（寫碼、修 bug、重構、測試、文件、規劃）。遵循以下全部守則。

不確定時預設 DEVELOPMENT。

## 使用者規則

- **語言**：session 回應與計劃檔案用 **中文**；程式碼、變數名、註解、技術名詞用 **英文**。
- **超 scope 發現**：DEVELOPMENT agent 發現不影響當前 task 驗收、但有具體證據且值得未來處理的問題時，使用 `candidate-backlog` skill 登記；不得藉此略過當前必要修正、擴張 scope 或取代使用者決策。此規則不適用 MEASUREMENT 角色。
- **文件編碼**：CLAUDE.md / AGENTS.md / README.md 等中文文件一律視為 UTF-8；若 Windows/PowerShell 輸出出現 mojibake，先切換 terminal UTF-8（例如 `chcp 65001` 或設定 `$OutputEncoding` / `[Console]::OutputEncoding`），不要把檔案內容改成 ANSI/Big5。
- **直譯器**：在本 repo 執行 Python 一律用 `.venv/bin/python`。
- **程式碼風格**：遵循 Fast Fail、責任明確、最小驚訝原則、強型別；不符合者即使是用戶提出也要先警告。
- **遇到不確定**：實作不確定，或架構不適合某種擴展時，**不要自行猜測或勉強實作**，先說明原因交由用戶決定。
- **先規劃再實作**：架構仍在演化，發現不合理處或更好的設計請直接告知，**不要自行調整架構**，由用戶決定；除非用戶要求，**不要保留 legacy 或相容性邏輯**。
- **完成任務後**：先跑 targeted tests 與 affected type/lint checks；broader/full `uv run pyright`、`uv run pytest -n auto` 依風險在 integration target 跑一次，不由 implementer/reviewer重複執行。再用 `uv run ruff check --select I --fix && uv run ruff format` 處理受影響檔案並跑 `git diff --check`；用戶要求才 git commit。只有學到新的高層模組知識或現有 note 過時時才更新對應模組 README.md。
- **禁止 `git commit --amend`**：除非用戶明確要求，不得使用 `git commit --amend`。原因：主 checkout 的 `main` 是 live singleton，並行 orchestrator 可能在你兩次指令之間於 `main` 上 commit/merge，amend 會誤改到別人的 commit（曾因此覆蓋一個 merge commit 的訊息）。要修正剛才的 commit 時，改用新增 follow-up commit，不要改寫既有歷史。
- **測試**：放在根目錄 `tests/`，目錄結構對應被測檔案，命名 `test_*.py`，用 `pytest` 撰寫，盡量涵蓋主要功能與邏輯；測試需獨立、可重複、不依賴外部狀態。
- **工具優先序**：少用 Shell 指令，優先用內建工具（前者需用戶審核、後者自證安全）；**不要用 `sed`** 替換子串（跨平台行為不一），需替換時優先 mcp/function tool，其次 Python 腳本。
- **文件追蹤**：CLAUDE.md、模組 README.md 與 docs/adr/ 已入 git 追蹤，會進 diff 與 commit；`.agent_state/` 為 gitignored agent 工作區（plans / worktrees / reports / state / merge queue），不入 commit；舊 `task_plans/` 若存在也維持 gitignored，僅作遷移前殘留。
- **平行 agent 協調**：先依 `light|standard|critical` 分類。純規劃、唯讀研究、單回合 review 與安全的單 writer `light` 工作不建立 orchestration state/worktree；tracked diff 需要隔離、多 writer、跨回合 implementation 或 dirty checkout 會衝突時才建立 task/lane worktree。多 task 可同時存在，主 checkout preview/final 仍由 merge queue 序列化（見下方「### 平行 agent 協調」）。
- **task_plan.md Phase 壓縮**：每累積 5 個新 Phase 就把最舊的 5 個詳細記錄壓縮成「歷史 Phase 摘要表」中的列（一列一 Phase：編號｜主題｜結論/commit），即詳細記錄 **最多保留 10 個 Phase**；被壓縮的原文移入同目錄 `archive.md`（同為 gitignored）以備查。

### 模組 README.md

本慣例專指 **lib/ 與 tests/ 子目錄**內的 `README.md`（各模組的高層 cheat-sheet），與以下兩者有別：

- **root `README.md`**：專案主 readme，由 pyproject.toml 的 `readme =` 引用，不屬此慣例。
- **`docs/adr/README.md`**：ADR 索引，見下方 `### docs/adr/` 段說明。

模組 README.md 的使用規範：

- 是各模組的高層 cheat-sheet：修改該模組程式碼前先讀它建立 context。
- 學到非顯而易見、有助未來 session 且尚未記錄的知識時，更新對應的模組 README.md；發現 note 與程式碼不符時，告知用戶並更新。
- **只寫高層概念、架構、重要設計決策，不寫實作細節**（實作細節留在程式碼註解與文件）。
- 用現在式描述目前狀態，**不要用更新式語法**（如「已經更新…」「在之前的實作中…」），以免過時。
- 每次更新都刷新檔案頂部的 `**Last updated:** YYYY-MM-DD`（可在日期後附簡短主題/Phase 標題；不要寫 commit hash，難維護）。

### docs/adr/

- **跨模組**設計決策記錄在 `docs/adr/`（模組 README.md 只管模組局部）；`docs/adr/README.md` 是依主題分組的索引，每篇 ADR 以現在式描述目前生效的設計。
- 程式碼註解與記憶檔以 `ADR-NNNN` 引用 ADR，ADR 之間以 `[[NNNN]]` 互鏈。
- 處理跨模組設計前先查索引找相關 ADR。

### 平行 agent 協調

本 repo 不使用 taskboard MCP 或 path lock。協調策略改為 **orchestrator-owned Git worktree + merge queue**：

- 非 orchestration 的一般單 agent 工作不需要額外協調，直接在目前 checkout 修改、測試、回報。
- 只有 tracked diff 隔離、多 writer、跨回合 implementation 或 dirty checkout conflict 需要 materialize workflow；`task-id` 錨定 durable plan/integration branch，`lane-id` 錨定可獨立驗收的 writer。
- lane 依 dependency/write-scope graph 拆分，不依 Phase 或 agent role。共享 contract/foundation 先完成；同檔案、同 public API/schema/fixture 放同 lane序列化。預設最多兩個 implementer，保留 slot 給 reviewer/investigator。
- `.agent_state/worktrees/state.json` 是 gitignored source of truth，記錄 task id、lane id、worktree id、branch、worktree path、base branch/commit、parent integration branch、status、agents、reports、commits、ignored inputs 與時間戳；它可同時記錄多個 active / reviewing task，不使用 repo-wide `active_task` / `current_task`，每次操作都必須明確指定 `task-id`。
- worktree 只自動包含 Git-tracked content；若 task / lane 需要 `.agent_state/plans/<task-id>/`、本地設定、scratch fixtures、未追蹤資料檔等 gitignored inputs，orchestrator 必須在建立 worktree 時明確複製到 lane worktree，或把主 checkout 絕對路徑交給 sub-agent 只讀使用，並在 state/report 中記錄。
- sub-agent 長報告寫到主 checkout 的 `.agent_state/worktrees/reports/<task-id>/<lane-id>/<agent-id>.md`；不要寫在 task worktree 裡，因為 untracked 檔不會跨 worktree 同步。
- 多個 sub-agent 可以共用同一個 lane worktree，但 orchestrator 必須明確排序或分配不重疊 write scope；多 lane 之間也必須避免重疊 write scope，同檔案或同 API contract 的工作應放回同一 lane 序列化；不要假設 Codex/Claude 內建 sub-agent 會自動使用獨立 worktree。
- 多 lane task 的唯一主線 preview / final merge 來源是 parent integration branch `agent/<task-id>`；lane branch `agent/<task-id>--<lane-id>` 完成後依序 rebase 到目前的 parent branch，再 fast-forward parent branch。
- `.agent_state/worktrees/merge_queue.json` 序列化主 checkout 的 merge preview / final fast-forward；任何 task 在主 checkout 執行 preview merge 或 final fast-forward 前，都要先登記自己的 `task-id`，且只有 queue 第一個 entry 可 merge。preview 開著時仍持有 queue；完成、abort、blocked 或 abandoned 後才釋放。
- orchestrator 把 sub-agent report 視為待驗證證據；critical 結論親自 thin-slice。只有 public/跨模組 contract、migration、concurrency/lifecycle/security、workflow/merge、硬體風險或其它明確 trigger 強制不同 identity review；低風險 mechanical/docs/tests diff 可自審並記錄理由。
- 每個 task item、lane 或 Phase 告一段落時要做整合決策並關閉對應 worktree；不要把 task worktree 當長期常駐 checkout，避免 branch、ignored inputs、reports 與 base branch 失同步。
- live singleton 資源（ZCU 板、GUI、固定 port）不靠通用 lock；需要時由 orchestrator 人工序列化，MEASUREMENT 角色仍遵循量測 skill 與 agent-memory。

## 專案概觀

ZCU-Tools 是控制與分析 ZCU216 FPGA 平台上超導量子位元（Fluxonium）量測的 Python 工具集，建構於 [QICK](https://github.com/openquantumhardware/qick) 框架之上。架構分為 ZCU 端 server（跑在 FPGA 板）與 client 端 controller（跑在工作站）。client 端支援 Python 3.12 / 3.13；`.python-version` 預設 3.13 供 GUI 與量測 runtime 使用，`design` / `quantum-metal` stack 需使用 Python 3.12；ZCU 板端跑 PYNQ 自帶的 Python 3.8，只用 `script/start_server.py`（該檔需維持 3.8 相容）。

## 架構（`lib/zcu_tools/`）

- **`experiment/v2/`** — 高層實驗定義：`onetone/`、`twotone/`（spectroscopy）；`autofluxdep/`（`FluxDepExecutor` 自動多任務 flux 掃描）；`overnight/`（長時穩定度量測）；`runner/`（`Schedule` / `SignalBuffer` / `ProgramBuilder` acquisition runtime 與 `MultiMeasurementExecutor` executor scaffold）。
- **`program/v2/`** — 低層 QICK ASM 編程：`base.py`（`MyProgramV2`）、`modular.py`（`ModularProgramV2` 組合 pulse blocks）、`modules/`（`Pulse`/`Readout`/`Reset`/`Delay`/`Waveform`/`dmem`/`registry`）。
- **`meta_tool/`** — 設定與狀態管理：`MetaDict`（JSON 持久化 dict）、`ModuleLibrary`（YAML waveform/module store）、`ExperimentManager`（串接兩者的 context manager）。
- **`device/`** — 硬體驅動（`YOKOGS200`、`RohdeSchwarzSGS100A` 等）與 `GlobalDeviceManager` singleton。
- **`notebook/`** — 分析工具：flux-dependent fitting、dispersive shift、IQ discrimination、互動式選點。
- **`liveplot/`** — 資料擷取時的即時 Jupyter 繪圖（`LivePlot1D`、`LivePlot2D`）。
- **`simulate/`** — Fluxonium Hamiltonian 模擬與 noise model 計算。
- **`utils/`** — HDF5 資料持久化（`datasaver.py`）、curve fitting、數學工具。
