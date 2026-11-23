# Repository agent instructions

**讀者：** agent。依任務選定角色，遵循量測或開發工作的授權、操作與驗證規則。

## 先判定角色

開始工作前選定一個角色：

- 任務透過 `measure-gui` MCP 操作儀器或執行量測時，使用 MEASUREMENT。
- 寫程式、修 bug、重構、測試、文件、規劃及其他工作，使用 DEVELOPMENT。
- 無法判定時，使用 DEVELOPMENT。

兩種角色的 session 回應與計劃檔案用中文；程式碼、變數名、註解與技術名詞用英文。

## MEASUREMENT

MEASUREMENT agent 是在目標與 policy 內自主工作的實驗協作者：

1. 讀取 [.agents/skills/measure-task/SKILL.md](.agents/skills/measure-task/SKILL.md)。操作 GUI 前讀取 [.agents/skills/run-measure-gui/SKILL.md](.agents/skills/run-measure-gui/SKILL.md)。
2. 透過 `measure-gui` MCP 操作 GUI 與儀器。遵守適用的硬體限制與當次授權，不以分析腳本旁路控制硬體。
   讀寫目前 point 的參數、library 或 aliases 時，先讀 [Point 檔案讀寫](lib/zcu_tools/mcp/measure/README.md#point-檔案讀寫)。
   依該流程直接讀檔，取得 holder 後編輯，再經 `rpc_call` 呼叫 `context.reload` 驗證安裝。
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
- 修改或 review 程式前，從 repo root 起，讀該檔案路徑上每一層目錄的 `README.md`，並照其中的指引讀相關文件。
- 處理或記錄跨模組設計時，先讀 `docs/adr/README.md` 的格式與索引，再讀相關 ADR。

完成條件：每個預計修改的模組都有適用 context；跨模組決策已找到相關 ADR，或已停下請使用者決定。

### 2. 守住決策與 scope 邊界

在既有架構內實作。需求有實質歧義、架構不適合擴展、沒有可靠實作路徑，或發現更好的架構時，說明原因並請使用者決定。除非使用者要求，不加入 legacy 或相容性邏輯。

寫程式碼時遵守下列規則。每條都附檢查方式；使用者要求若違反其中一條，先指出風險。

- **放置。** 新程式碼依 [模組定位](docs/adr/0072-module-boundaries.md) 選 package；所有類別都不合才放 `utils`。檢查：說得出這段程式碼屬於哪一列定位。
- **依賴。** 新增跨 package import 後，`lint-imports` 必須通過。`.importlinter` 的 `ignore_imports` 是待清除的債務，不是先例：不新增、不放寬，也不仿照其中的依賴寫新程式碼。檢查：交付時列出這次新增的跨 package 依賴邊，以及涵蓋它的 contract；沒有 contract 涵蓋的邊要說明理由。
- **不重做別的模組的工作。** 需要的能力屬於另一個模組、而它沒有提供時，擴充那個模組，或停下來問。檢查：呼叫端沒有在驗證被呼叫者的輸出格式、組被呼叫者的內部版面，或修補它回傳的資料。
- **抽象要有現存的理由。** 新增 class、Protocol、registry 或 factory 時，指出目前已存在的第二個使用者，或它集中隱藏的複雜度；只有一個使用者就直接呼叫。檢查：交付時能為每個新抽象說出這個理由。
- **黑箱（black box）。** 每個公開 class、function、method 都是黑箱：只看它的簽名與 docstring，就知道每個參數填什麼、產生什麼效果、何時失敗。公開型別的每個欄位在型別本身寫明意義與可能的值。檢查：只靠簽名與 docstring，為每個公開項目寫出一次呼叫；只看型別本身，說得出每個欄位的意義。填不出的參數、說不出的效果或欄位，補進 docstring 或改名。
- **局部性（Locality of Behaviour）。** 一個函式的行為，看函式本身，加上它呼叫的黑箱的簽名與文件就知道。檢查：理解某個函式需要打開另一個函式的實作時，調整邊界、命名或介面文件。
- **無聊程式碼（boring code）。** 用一眼就懂的寫法。sentinel 物件、互相遞迴、只用到一半的回傳值這類需要另外解釋的寫法，換成無聊的寫法；確實需要時，在原地加一行理由。檢查：diff 中每處這類寫法都已換掉，或旁邊有理由。
- **Fast Fail。** 契約外的輸入在進入點 raise，不以預設值、空資料或 `None` 掩蓋。只有能恢復、轉譯或隔離失敗的地方才 catch。檢查：每個 `except` 都做其中一件事。
- **強型別。** 公開函式的參數與回傳用具名型別（dataclass、pydantic model、TypedDict）。上層資訊往下層傳時，能直接傳上層原物件就傳原物件；沒有現成型別的少量附加資訊可以用 `dict`，consumer 讀不到需要的 key 就 fast fail。不要為了跨層傳遞，在每一層各定義一個對應的類別。新程式碼不新增 `cast` 或 suppression，不放寬型別配合錯誤。
  已明確授權清洗的既有未處置診斷，先逐位置嘗試修除，確定保留後才用指定實際 rule 的 suppression。既有 suppression 按種類抽樣試清，展開條件見 [程式碼品質](docs/code-quality.md#既有容忍項先嘗試修除理由留在原地)。
  Ruff 用 `# noqa: PLR0913  # tolerated: <reason>`，Pyright 用 `# pyright: ignore[reportPrivateUsage]  # tolerated: <reason>`。Suppression 留在診斷行，理由寫同行，放不下時放正上方的 `# tolerated: <reason>`。禁止無 rule 的 noqa、type ignore 或 pyright ignore。
  既有 `cast` 與設定項也先嘗試清除。保留時分別用 `# tolerated(cast): <reason>` 與 `# tolerated(config): <reason>`，放原行行尾或正上方。保留設定值不變，不新增 `cast`，不新增或放寬設定。
  File-size／radon／capability 保留項用 `# tolerated(<metric>): <reason>`，放相關行或檔案開頭。理由只寫原地，不交容忍名冊。
  檢查：每個既有容忍項有嘗試證據、指定標記與具體理由；例外只適用已授權清洗的既有項目，新程式碼沒有新增 `cast` 或 suppression。

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

框架 UI 或 `zcu_lab/` 實驗定義有改動時，實作與 review 前讀 [UI 驗證](tests/README.md#ui-驗證)，依框架 ComponentDriver／實驗定義離線 probe 分支驗收。實驗定義的畫面改動也走 probe 分支；兩者都改時，兩種驗收都要做。

優先使用內建工具。子字串替換先用 function 或 MCP 工具，其次用 Python script，不用 `sed`。

測試遵循以下契約；新增、拆分或搬遷測試前，先讀 [tests/README.md](tests/README.md) 的套件結構、fixture 與搬遷規則，找出既有行為的 owner：

- 測試位於 root `tests/`，檔名使用 `test_*.py`，以 `pytest` 涵蓋本次變更的主要行為與邏輯。
- 測試目錄的路徑對應被測模組：含 `test_*.py` 的目錄必須對應一個實際存在的模組目錄。對應是模組層級，檔名不受約束。`apps`、`mcp_servers`、`migrations`、`scripts`、`tools` 與 `zcu_lab` 對應 repo root 的同名目錄，其餘對應 `lib/zcu_tools/` 之下。`contract` 與 `parity` 為保留名稱，豁免該段及其以下，但其前的路徑前綴仍須對應；新增保留名稱需使用者同意。
- 路徑對應以 `tools/check_quality.py test-paths` 判定。判讀對象是本次改動觸及的路徑；若有既有違規，不以全庫總數判定本次改動。
- 測試保持獨立、可重複、不依賴外部狀態。修改 module-level 集合、cache、registry 等 mutable state 後必須還原；共用狀態者在 module 前後比對，並由 guard 指出污染者。
- 同一棵 tree 因測試選集或順序得出不同結論時，將該不穩定視為缺陷，不以偶然通過的結果驗收。

#### 暫存檔生命週期

| 生命週期 | 位置與清理方式 |
| --- | --- |
| 不長過 worktree | 放在 worktree 內。 |
| 一張票 | `.agent_state/plans/<task>/scratch/<ticket>/`，收票時刪除。 |
| 一個 task | `.agent_state/plans/<task>/scratch/`，task 封存時清理。 |
| 跨 task 的 effort | `.agent_state/maps/<effort>/scratch/`，effort 結束時清理。 |
| 永久 | 進 git：數據放在使用它的工具旁，結論寫進 ADR、README 或討論紀錄。 |
| 必須使用 `/tmp` | 任務結束前刪除。 |

`/tmp` 是 RAM-backed tmpfs，不讓平行 lanes 共用具名路徑。

`.agent_state/` 最上層只有 `plans`、`maps`、`archives`、`worktrees`、`backlog`、`measurement-tasks`、`ratchet`、`benchmark` 與 `quality`，不放散檔，也不放沒有主人的 scratch。調查與研究的產出放在提出問題的 task 或 map 的 scratch；要留存的寫成討論紀錄。每個 worktree 有主人與移除時機：票的 lane 在收票或放棄時、integration 在落地後、spike 在報告被接受時、小修改在合併後。移除包含 worktree、分支、設定 `PYTHONPYCACHEPREFIX` 時其下對應該 worktree 絕對路徑的 bytecode 目錄，與空掉的上層目錄。保留下來的證據以能重新核對結論為準，原始大檔在 task 封存時刪除。

### 5. 完成前驗證

依序完成：

1. 執行 targeted tests 與 affected type and lint checks。
2. 執行 `tools/gate.py --base <ref>`。預設會對受影響檔案執行 Ruff import sorting 與 formatter，再執行其餘 gate 檢查。
3. Formatter 若改變程式碼，重跑受影響的 tests、type checks 與 lint checks。
4. 依風險執行 broader Pyright 與 parallel pytest；全樹型別以 `tools/check_ratchet.py --base <ref> --detector pyright --full-pyright` 和 baseline 比較，原始全樹 Pyright 的既有錯誤照實列出即可，見 [tools/README.md](tools/README.md)。
5. 執行 `git diff --check`。

受管理 lane 的 Python 指令繼續遵循第 3 節。完成回報列出無法執行或依風險略過的檢查及原因。

### 文件規範

使用者面向的文件（README、guide、教學、docstring）每句寫操作會發生什麼或會產生什麼狀態，不用品質形容詞代替輸入、動作與結果。

#### 讀者宣告

每份 tracked markdown 文件第一段的第一行宣告讀者與用途，格式固定：`**讀者：** <讀者>。<一句用途>`，英文文件用 `**Readers:** <readers>. <purpose>`。讀者取 operator（在 GUI／MCP 做量測的人）、lab-author（擴充 `zcu_lab/` 的人）、framework-dev（維護 `lib/`、`tests/`、`tools/` 的人）、agent（coding 或 measurement agent）之一或組合。ADR 與討論快照不逐篇宣告，由 `docs/adr/README.md` 與 `docs/discussions/INDEX.md` 各宣告一次。

套用例：
- `zcu_lab/v2/mux/twotone_spectrum/guide.md` 第一行：`**Readers:** operator, agent. How to run the mux qubit frequency sweep from the GUI or MCP and read its results.`
- module README：`**讀者：** framework-dev。session 層的 owner、接縫與重要決策速查。`（放在 `**Last updated:**` 行之後）
- `zcu_lab/tutorial/01-components.md`：`**讀者：** lab-author。三步新增自己的 component。`

`tools/check_doc_readers.py` 判定：非 ADR、非討論的 markdown 第一段含 `**讀者：**` 或 `**Readers:**` 且讀者詞在四個之內。

#### Module README

Module `README.md` 專指 `lib/` 與 `tests/` 子目錄中的高層 cheat-sheet。只在取得新的高層知識時更新；發現 note 過時則告知使用者並更新。內容限於高層概念、架構與重要決策，實作細節留在程式碼或其他文件。使用現在式，並刷新 `**Last updated:** YYYY-MM-DD`；日期可附主題或 Phase，不寫 commit hash。

#### 文件之間的關聯

文件與程式碼的關聯由下層指向上層，全部可以用搜尋查到：

- Module README 列出管轄它的 ADR，以 `ADR-NNNN` 引用；沒有 ADR 管轄時寫明沒有。程式碼受某篇 ADR 約束的位置，以註解引用 `ADR-NNNN`。
- 文件指向程式碼時連到檔案或寫出名稱，不釘行號：`#L123` 會隨編輯錯位。
- 改動 tracked 程式碼後、交付前，執行 `tools/doc_relations.py <base>`。它列出相關文件（改動路徑上的 README、這些 README 與改動程式引用的 ADR、引用同一批 ADR 的其他文件），以及提到被刪除、改名或改動的名稱與檔案路徑的文件行。文件對被改對象另有稱呼時，用 `--term` 補上。每一筆給一個處置：無關、已一致、已更新。用法見 [tools/README.md](tools/README.md#document-relations)。
- `tools/doc_relations.py --check` 檢查相對連結的目標與 `ADR-NNNN` 是否存在。

#### 文檔組織與搜尋

讓不熟 repo 的人或 agent 花最少時間與力氣找到所需粒度的資訊。

- **粒度分層。** 依〈讀者宣告〉選入口：operator、lab-author、framework-dev 或 agent 從各自會讀的文件找操作、契約與決策。實作細節由程式碼交代，維護理由留在相關程式的註解。各層分工，說法一致，跨層用連結代替重抄。例如，skopt `n_jobs=1` 的理由留在 [auto-optimize 的 `core.py`](zcu_lab/v2/twotone/ro_optimize/auto_optimize/core.py) 註解；operator guide 只說操作與結果。
- **判斷依據。** 讓讀者從文件找到會影響判斷的事實：是什麼、由誰負責、為何如此。要追到實作時，連到程式碼或寫明可 grep 的名稱。例如，lab-author 從 [`experiment-authoring.md` 的存檔軸段](zcu_lab/v2/experiment-authoring.md#存檔軸用關鍵字固定座標不塞進-result)找結果格式；需要判斷舊檔的 cfg 相容性時，入口指向 [ADR-0069〈Cfg 欄位改名與 schema 版本〉](docs/adr/0069-experiment-core-and-frontends.md#cfg-欄位改名與-schema-版本)。
- **橫向互指。** 一個模組的契約依賴另一個模組的定義時，從雙方的契約段落互連到對應段落。例如，FTF environment、ADC 與 Program simulation 在各自說明 integrated statistics 的段落互指，讓讀者直接對照環境產出的統計、ADC 的邊際計數與 Program 消費的統計格式。
- **限定語。** 同一詞在不同模組代表不同資料時，每個模組在自己的文件寫明限定語或固定搜尋詞。例如，`joint` 可以指 FTF 的 herald history × terminal state 聯合分布，也可以指 `two_qubit/joint_readout` 的雙 qubit IQ 聯合分類；ADC 的 per-event marginals 則是各讀取事件的邊際計數。
- **實測數字。** 文件與 agent 工作檔（dossier、決策、spec、dispatch、報告）引用實測的耗時、誤差、計數或 telemetry 值時，在數字旁標明量測 commit、輸入或設定，以及範圍：單輪或總和、實測或外推。一般常數照常引用。數字要驅動派工、裁決或驗收前，先執行 `git log <commit>..HEAD -- <被測路徑>`；有改動就先重量，或寫明改動為何不影響這個數字。`complete=false` 的 telemetry 列中，未走到的階段可能記為 0 或空值，不代表耗時為零。引用過去裁決或驗收的狀態前，先搜尋其編號，確認沒有被後續決策取代。例如，引用 chain_freq 的 materialize 時寫「325 s（`4a2dea8`，11 點、187 輪總和）」；`git log 4a2dea8..HEAD -- lib/zcu_tools/simulate/adc` 列出 `b6bdb8024`，就先重量再派優化。
- **Grep 橋接。** 動態組出的持久名稱若能列舉，就在組字串的程式旁以註解列出完整名稱。含 hash 等無法列出全名的名稱，則寫明可 grep 的固定前綴。例如，two_qubit 用 `cz_{role}` 組出 `cz_qubit1` 與 `cz_qubit2`，就在組字串處列出這兩個 slot 名，讓人用完整名稱搜到產生處。
- **調查筆記。** 調查或裁決時，在 map 內照該 map 的 Standing order 記筆記。在 map 外，只用現有的 `candidate-backlog` 登記範圍外的文檔組織發現，將 `area` 設為 `doc-organization`。記下觀察、可重現的證據（如檔案路徑或搜尋詞）、對查找的影響與想取得的結果。沒有該 skill 時，在回報中附上這些欄位與 `area`，交由 backlog owner 登記。例如，FTF 的 integrated statistics 只連到 ADC README 開頭時，記下兩邊的段落與 ADC 的 `ProjectionChain` 搜尋詞，說明讀者無法直接對照契約，希望連結直達對應段落。
- **重構時機。** 在 map 內照該 map 的 Standing order 彙整。在 map 外，從上次彙整起計算 `doc-organization` 候選項，依既有的 task 納入紀錄排除已彙整的項目。累積 5 筆候選項，或同一類缺陷出現 2 次時，交由負責排程的人彙整成一個文檔重構 task。一起修具體文件與這類查找的規範或搜尋詞。驗收時重走觸發筆記的查找路徑，確認能從 README 連結、ADR 索引或寫明的 grep 詞找到所需事實。例如，integrated statistics 有兩處缺契約互指，就一起補；從 ADC 的契約段落應能直達 FTF 與 Program 的對應段落。

#### 編碼與 agent 工作檔

中文文件使用 UTF-8。Windows 或 PowerShell 顯示 mojibake 時，先將 terminal 切換為 UTF-8，例如 `chcp 65001` 或設定 `$OutputEncoding`，不改寫檔案編碼。

Agent plans 與 worktree 工作檔放在 gitignored `.agent_state/`。

#### 只依賴 git 內的內容

Tracked 檔案（程式碼、註解、測試、repo 文件、skills）要讀者讀取、執行或依賴的文件、腳本、工具、資料與證據，都放在 git 裡，或是外部公開來源。需要的工具或腳本加進 repo；依據來自未入庫的工作檔時，把結論與必要事實寫進正文。Tracked 檔案可以寫出 git 外的位置，供程式或 agent 存放新產生的檔案與快取，例如 `.agent_state/` 下的計畫、暫存檔與 ratchet 快取；建立與使用該位置的程式或流程在 git 裡，位置不存在時由它們重建。檢查：清空所有 gitignored 內容後，照 tracked 文件操作仍走得通。

#### 使用者決策的引用

`.agent_state/` 的決策編號（`Dnn`、`decisions/NNNN`）只在該 task 內有效，各 task 各自編號。Tracked 檔案（程式碼、註解、測試、repo 文件）直接寫出理由；屬於架構規則的引用 `ADR-NNNN`。涉及跨模組的元件定位、職責或邊界裁決寫進 ADR 的決策段，附裁決日期與「」原話。

取捨原則依影響範圍放置，判準類似 Python 的 PEP：內部但架構重大的改變也寫（例如 PEP 659），小改進不寫。符合任一條的寫進該領域 ADR 的決策段：約束或可能約束多個底層模組；決定一個子系統內所有部件共同遵守的架構；難以反悔或需要留下被否決方案的理由。其餘只有擁有者自己用、可在一個 task 內反悔的取捨，寫進擁有它的 README 的決策段；個別元件的閾值與校準常數寫在模組 README 或程式旁。每條包含規則、一個例子、適用邊界、「」原話與日期，以及出處的討論紀錄。已核准但尚未落實的方向寫進 `docs/adr/draft/`；`docs/adr/README.md` 只列 ADR 原則的摘要與指標。使用者 2026-11-23：「搬到README，adr只放跨模塊決策」；「但從一體性來看，evolution與內部kernel規則確實值得進adr」；「採用建議」。

#### 先例

做設計決策前，先查同類問題過去怎麼裁、repo 現在怎麼解：ADR、`docs/discussions/INDEX.md`，以及程式碼中已解決同類問題的寫法。

| 強度 | 來源 |
| --- | --- |
| 有拘束力 | 使用者親自裁定的決策、ADR、原則 |
| 僅供參考 | 既有程式碼的寫法、使用者尚未覆核的代裁 |
| 不得援引 | 已標記為債務的寫法 |

對每個先例表態：遵循；區別，並寫出事實上哪裡不同；或建議推翻，並寫出理由。有拘束力的先例只能由使用者推翻。推翻後，依舊先例寫的程式碼在同一個 task 一起修改，或登記為債務；同一類問題不並存兩種沒有紀錄的寫法。同類裁決多次一致時，提議歸納成一條原則。

#### 討論紀錄

`docs/discussions/` 保存值得留下的討論快照：外部方案評估、選項比較、被否決的方案與理由，以及推導過程。它同時是先例的彙編。討論產生上述內容時，agent 提議記錄，使用者同意後才寫；使用者明確要求時直接寫。

新增或刪除一個概念、改變公開介面或行為的意義、改變資料格式、跨越模組邊界、或難以反悔的決策，在落地時一律寫一篇。內容取自決策前的調查：問題、成因、比較過的選項、查到的先例與對每個先例的立場、裁決。

- 每個討論濃縮成一篇 `YYYY-MM-DD-slug.md`，並在 `INDEX.md` 依主題加一行：日期、標題連結、問題類型、一句結論。問題類型是一個短詞（例如「刪除或收窄」「資料入庫」），讓同類決策可以被搜尋到。
- 快照不擁有決策。決策寫進 ADR，討論只連結；討論中的事實由程式碼、ADR 或 README 擁有時，附路徑而不複製。
- 快照必須 self-contained：只連結 tracked 檔案或外部公開來源，不引用 `.agent_state/` 或其他 gitignored、本機路徑。當時依據未入庫的工作檔時，把結論與必要事實寫進正文，並註明原始工作檔未入庫。
- Frontmatter 記 `date`、`status`（`active`／`superseded`）、`superseded_by`、`related`，以及 `basis`：當時依據的本 repo commit 與外部來源版本。
- 正文依序寫結論、問題、考慮過的選項與比較、被否決的選項與理由、留下的問題、依據。保留使用者的判斷與修正；省略工具輸出、對話往返與可從程式碼讀到的事實。
- 寫完後不改寫成現在式。發現錯誤時在文末加「後記」；被新討論或 ADR 取代時改 `status` 並填 `superseded_by`。移除本機路徑、補成 self-contained 不算改寫。
