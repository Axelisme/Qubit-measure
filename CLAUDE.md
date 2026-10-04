# Repository agent instructions

## 先判定角色

開始工作前選定一個角色：

- 任務透過 `measure-gui` MCP 操作儀器或執行量測時，使用 MEASUREMENT。
- 寫程式、修 bug、重構、測試、文件、規劃及其他工作，使用 DEVELOPMENT。
- 無法判定時，使用 DEVELOPMENT。

兩種角色的 session 回應與計劃檔案用中文；程式碼、變數名、註解與技術名詞用英文。

## MEASUREMENT

MEASUREMENT agent 是在目標與 policy 內自主工作的實驗協作者：

1. 讀取 [.agents/skills/run-measure-gui/SKILL.md](.agents/skills/run-measure-gui/SKILL.md)。第一次操作前讀取 `measure-gui` MCP server instructions，開始或恢復時核對 GUI 現況。
2. 透過 `measure-gui` MCP 操作 GUI 與儀器。遵守適用的硬體限制與當次授權，不以分析腳本旁路控制硬體。
3. 任務以 cwd 為準，目標、policy、戰略與進度存於 `.agent_state/measurement-tasks/<task-id>/`。可重用經驗存於 `measure_knowledge/`，依 skill 初始化缺失入口；共享 symlink 由使用者管理。
4. 可使用原生檔案、搜尋與 Git 工具維護上述內容，並建立或執行離線分析腳本。Python 使用 `uv run --directory <repo> --no-sync -- <command>`，沿用 repo 環境，不自行安裝依賴。一次性腳本與衍生輸出留在任務目錄，保留原始資料。
5. 以量測證據建立工作假說，比較分析、補量測與改道的資訊及成本。需要使用者裁決目標、policy、風險或必要背景時求助。

`lib/` implementation 不屬於量測證據。MEASUREMENT agent 不讀、不搜尋、不修改或引用其中的實作。

處理 agent launch 或 lifecycle 時讀 [Remote／Transport ADR](docs/adr/0068-remote-transport.md)。外部 CLI 或 MCP workflow 擁有啟動流程，GUI 不提供 launch UI。

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

寫程式碼時遵守下列規則。每條都附檢查方式；使用者要求若違反其中一條，先指出風險。

- **放置。** 新程式碼依 [模組定位](docs/adr/draft/module-boundaries.md) 選 package；所有類別都不合才放 `utils`。檢查：說得出這段程式碼屬於哪一列定位。
- **依賴。** 新增跨 package import 後，`lint-imports` 必須通過。`.importlinter` 的 `ignore_imports` 是待清除的債務，不是先例：不新增、不放寬，也不仿照其中的依賴寫新程式碼。檢查：交付時列出這次新增的跨 package 依賴邊，以及涵蓋它的 contract；沒有 contract 涵蓋的邊要說明理由。
- **不重做別的模組的工作。** 需要的能力屬於另一個模組、而它沒有提供時，擴充那個模組，或停下來問。檢查：呼叫端沒有在驗證被呼叫者的輸出格式、組被呼叫者的內部版面，或修補它回傳的資料。
- **抽象要有現存的理由。** 新增 class、Protocol、registry 或 factory 時，指出目前已存在的第二個使用者，或它集中隱藏的複雜度；只有一個使用者就直接呼叫。檢查：交付時能為每個新抽象說出這個理由。
- **黑箱（black box）。** 每個公開 class、function、method 都是黑箱：只看它的簽名與 docstring，就知道每個參數填什麼、產生什麼效果、何時失敗。公開型別的每個欄位在型別本身寫明意義與可能的值。檢查：只靠簽名與 docstring，為每個公開項目寫出一次呼叫；只看型別本身，說得出每個欄位的意義。填不出的參數、說不出的效果或欄位，補進 docstring 或改名。
- **局部性（Locality of Behaviour）。** 一個函式的行為，看函式本身，加上它呼叫的黑箱的簽名與文件就知道。檢查：理解某個函式需要打開另一個函式的實作時，調整邊界、命名或介面文件。
- **無聊程式碼（boring code）。** 用一眼就懂的寫法。sentinel 物件、互相遞迴、只用到一半的回傳值這類需要另外解釋的寫法，換成無聊的寫法；確實需要時，在原地加一行理由。檢查：diff 中每處這類寫法都已換掉，或旁邊有理由。
- **Fast Fail。** 契約外的輸入在進入點 raise，不以預設值、空資料或 `None` 掩蓋。只有能恢復、轉譯或隔離失敗的地方才 catch。檢查：每個 `except` 都做其中一件事。
- **強型別。** 模組邊界用具名型別（dataclass、pydantic model、TypedDict）傳遞資料，不用無結構的 `dict` 或 `Any`。不新增 `cast`、`# type: ignore` 或 Pyright suppression。檢查：diff 中沒有新增的 suppression；無法避免時在交付中列出位置與理由。

細節與審查裁決見 [程式碼品質](docs/code-quality.md)。

具體且值得後續處理的 scope 外問題使用 `candidate-backlog` skill 登記。它不改變當前 scope，也不取代必要修正或使用者決策。

完成條件：實作範圍、使用者決策與 scope 外發現已清楚區分；`lint-imports` 通過，上述規則的檢查都有答案，例外已列出理由。

### 3. 使用正確環境

開發環境建立或重新同步時，一律包含 `quality` dependency group。受管理 lane 是協作流程建立的 worktree。Orchestrator 建立 lane 後、派發任何 role 前執行：

```bash
uv sync --directory <lane> --locked --group quality
```

成功後才派發 role；失敗時停止。所有 worktree 的 Python interpreter 與 Python entry-point 工具一律使用：

```bash
uv run --directory <worktree> --no-sync -- <command>
```

非 Python 的獨立執行檔直接呼叫，不經 `uv run`。

`<lane>/.venv` 由該 lane 專用並隨 worktree 清除。修改 tracked dependency files 後，Orchestrator 先重跑 locked bootstrap，roles 再使用 `--no-sync`。`--no-sync` 不自動修復環境：環境與 lockfile 不符時讓指令失敗，由 Orchestrator 決定是否重跑 bootstrap。本 repo 的 Python 指令不使用 worktree 外的 interpreter。

Worktree 只隔離檔案，不隔離 ZCU／儀器、GUI subprocess 或固定 port 等共享資源。並行工作若會用到同一個 live resource，先安排使用順序，不能因為在不同 lane 就假定互不影響。Worktree 也不帶來硬體操作授權。

完成條件：指定 interpreter 可用且已安裝 `quality` group；受管理 lane 的 locked bootstrap 成功。

### 4. 實作與測試

優先使用內建工具；沒有合適工具且使用者核准時才用 Shell。子字串替換先用 function 或 MCP 工具，其次用 Python script，不用 `sed`。

測試遵循以下契約；新增、拆分或搬遷測試前，先讀 [tests/README.md](tests/README.md) 的套件結構、fixture 與搬遷規則，找出既有行為的 owner：

- 測試位於 root `tests/`，檔名使用 `test_*.py`，以 `pytest` 涵蓋本次變更的主要行為與邏輯。
- 測試目錄的路徑對應被測模組：含 `test_*.py` 的目錄必須對應一個實際存在的模組目錄。對應是模組層級，檔名不受約束。`scripts` 與 `tools` 對應 repo root 的同名目錄，其餘對應 `lib/zcu_tools/` 之下。`contract` 與 `parity` 為保留名稱，豁免該段及其以下，但其前的路徑前綴仍須對應；新增保留名稱需使用者同意。
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
