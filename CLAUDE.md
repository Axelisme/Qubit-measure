# Repository agent instructions

## 先判定角色

開始工作前選定一個角色：

- 任務透過 `measure-gui` MCP 操作儀器或執行量測時，使用 MEASUREMENT。
- 寫程式、修 bug、重構、測試、文件、規劃及其他工作，使用 DEVELOPMENT。
- 無法判定時，使用 DEVELOPMENT。

兩種角色的 session 回應與計劃檔案用中文；程式碼、變數名、註解與技術名詞用英文。

## MEASUREMENT

MEASUREMENT agent 是 operator：

1. 第一次操作前讀取 `run-measure-gui` skill 與 `measure-gui` MCP server instructions。
2. 透過 `measure-gui` MCP 工具驅動量測。
3. 將記錄、疑難排解與驗收清單等持久知識只寫入 `agent-memory`。
4. 以量測資料驗證結果；需要更多證據時，放寬參數、重跑並判讀圖形。

`lib/` implementation 不屬於量測證據。MEASUREMENT agent 不讀、不搜尋、不修改或引用其中的實作。

處理 agent launch 或 lifecycle 時讀 [ADR-0024](docs/adr/0024-embedded-agent-session-architecture.md)。外部 CLI 或 MCP workflow 擁有啟動流程，GUI 不提供 launch UI。

MEASUREMENT 只套用本節與語言規則。以下規則屬於 DEVELOPMENT。

## DEVELOPMENT

### 1. 建立必要 context

只讀取目前工作需要的文件：

- 需要判斷 Runtime Profiles 或全 repo 概觀時，讀 root `README.md`。
- 實作、重構或 review 程式碼前，讀 [程式碼品質](docs/code-quality.md)，用於設計取捨與審查裁決。
- 修改 `lib/` 或 `tests/` 內的模組前，讀該路徑適用的 module `README.md`。
- 處理或記錄跨模組設計時，先讀 `docs/adr/README.md` 的格式與索引，再讀相關 ADR。

完成條件：每個預計修改的模組都有適用 context；跨模組決策已找到相關 ADR，或已停下請使用者決定。

### 2. 守住決策與 scope 邊界

在既有架構內實作。需求有實質歧義、架構不適合擴展、沒有可靠實作路徑，或發現更好的架構時，說明原因並請使用者決定。除非使用者要求，不加入 legacy 或相容性邏輯。

程式碼遵循 Fast Fail、責任明確、最小驚訝與強型別原則。使用者要求若違反這些原則，先指出風險。

具體且值得後續處理的 scope 外問題使用 `candidate-backlog` skill 登記。它不改變當前 scope，也不取代必要修正或使用者決策。

完成條件：實作範圍、使用者決策與 scope 外發現已清楚區分。

### 3. 使用正確環境

受管理 lane 是協作流程建立的 worktree。Orchestrator 建立 lane 後、派發任何 role 前執行：

```bash
uv sync --directory <lane> --locked
```

成功後才派發 role；失敗時停止。所有 worktree 的 Python interpreter 與 Python entry-point 工具一律使用：

```bash
uv run --directory <worktree> --no-sync -- <command>
```

非 Python 的獨立執行檔直接呼叫，不經 `uv run`。

`<lane>/.venv` 由該 lane 專用並隨 worktree 清除。修改 tracked dependency files 後，Orchestrator 先重跑 locked bootstrap，roles 再使用 `--no-sync`。`--no-sync` 不自動修復環境：環境與 lockfile 不符時讓指令失敗，由 Orchestrator 決定是否重跑 bootstrap。本 repo 的 Python 指令不使用 worktree 外的 interpreter。

完成條件：指定 interpreter 可用；受管理 lane 的 locked bootstrap 成功。

### 4. 實作與測試

優先使用內建工具；沒有合適工具且使用者核准時才用 Shell。子字串替換先用 function 或 MCP 工具，其次用 Python script，不用 `sed`。

測試遵循以下契約；新增 regression、拆分或搬遷測試前，先讀 [tests/README.md](tests/README.md) 的套件結構、fixture 與搬遷規則，找出既有行為的 owner：

- 測試位於 root `tests/`，檔名使用 `test_*.py`，以 `pytest` 涵蓋本次變更的主要行為與邏輯。
- 測試目錄的路徑對應被測模組：含 `test_*.py` 的目錄必須對應一個實際存在的模組目錄。對應是模組層級，檔名不受約束。`script` 與 `tools` 對應 repo root 的同名目錄，其餘對應 `lib/zcu_tools/` 之下。`contract` 與 `parity` 為保留名稱，豁免該段及其以下，但其前的路徑前綴仍須對應；新增保留名稱需使用者同意。
- 路徑對應以 `tools/check_test_path_correspondence.py` 判定。它目前對既有目錄回報非零；判讀對象是本次改動觸及的路徑，不是總數。
- 測試保持獨立、可重複、不依賴外部狀態。修改 module-level 集合、cache、registry 等 mutable state 後必須還原；共用狀態者在 module 前後比對，並由 guard 指出污染者。
- 同一棵 tree 因測試選集或順序得出不同結論時，將該不穩定視為缺陷，不以偶然通過的結果驗收。

#### 暫存檔生命週期

| 生命週期 | 位置與清理方式 |
| --- | --- |
| 不長過 worktree | 放在 worktree 內。 |
| 不長過 task | 放在 `.agent_state/` 下的 per-task 目錄。 |
| 必須使用 `/tmp` | 任務結束前刪除。 |

`/tmp` 是 RAM-backed tmpfs，不讓平行 lanes 共用具名路徑。

### 5. 完成前驗證

依序完成：

1. 執行 targeted tests 與 affected type and lint checks。
2. 對受影響檔案執行 Ruff import sorting：`check --select I --fix`，再執行 formatter。
3. Formatter 若改變程式碼，重跑受影響的 tests、type checks 與 lint checks。
4. 依風險執行 broader 或 full Pyright，以及 parallel pytest。
5. 執行 `git diff --check`。

受管理 lane 的 Python 指令繼續遵循第 3 節。完成回報列出無法執行或依風險略過的檢查及原因。

### 文件規範

#### Module README

Module `README.md` 專指 `lib/` 與 `tests/` 子目錄中的高層 cheat-sheet。只在取得新的高層知識時更新；發現 note 過時則告知使用者並更新。內容限於高層概念、架構與重要決策，實作細節留在程式碼或其他文件。使用現在式，並刷新 `**Last updated:** YYYY-MM-DD`；日期可附主題或 Phase，不寫 commit hash。

#### 編碼與 agent 工作檔

中文文件使用 UTF-8。Windows 或 PowerShell 顯示 mojibake 時，先將 terminal 切換為 UTF-8，例如 `chcp 65001` 或設定 `$OutputEncoding`，不改寫檔案編碼。

Agent plans 與 worktree 工作檔放在 gitignored `.agent_state/`。若舊 `task_plans/` 仍存在，維持 gitignored，只作為遷移前殘留。
