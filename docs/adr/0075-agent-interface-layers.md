---
status: accepted
---

# ADR-0075 — Agent 介面分層

**狀態：** accepted。使用者於 2026-10-16 以「都採用建議」確認歸檔。本篇記錄入口分層及已落實的 recipe 契約；workflow 的 agent yield 仍是方向，見「現況與後續」。不新增工具、不改 method policy，也不授權硬體操作。
**關聯：** [[0068]] 的 remote 與 measure 固定工具、[[0062]] 的 workflow run、[[0063]] 的保存權威、[[0069]] 的核心與前端；控制權見 [[0070]]，workflow engine 見 [[0062]]。

## 問題

Measure、workflows、fluxdep 與 dispersive 的 MCP 各自提供工具，[[0068]] 也寫了「日常量測優先 recipe、排查使用 RPC」的指引。這些入口位於哪一層、agent 應該在哪裡介入、層與層如何交接參數，目前分散在各篇 ADR 和 module README，沒有一份文件寫在同一處。缺少這份定位時，新增工具容易放錯層，agent 也不知道從哪一層開始。

## 決策

2026-10-12 使用者裁決 MCP 的定位：「提供agent從底層到抽象層的方法，最底層是實驗環境與硬體操作，這對用戶與agent都不可直接控制，上一層是gui/notebook層級控制，包含儀器控制，實驗建立，完整cfg控制，實驗參數控制等。再上一層是recipe，將常用實驗包裝層一個參數量小的自動gui操作，其餘參數用自動推導的方式，只在需要agent時yield給agent，這樣正常使用agent只需要傳入真的需要其判斷的資訊即可。再上一層是workflow gui/mcp，包裝常用實驗流程進一次實驗執行，比如overnight實驗，或是t1 overflux實驗，只需要要一次設定，就能完成跨多組實驗的流程。同層的還有跨實驗整合分析層，比如fluxdep_fit gui/mcp，支援跨多次量測數據甚至跨多qubit的數據整合分析。」

同日使用者裁定 L1 與 L2 的差別：「L1中，每個實驗更像是一組多部流程，agent可以在中間介入。L2中實驗就是一個整體，agent只能控制輸入與接受輸出。」隨後補充：「L2不是完全無法介入中間，而是中間明確需要agent時使用yield來與agent協作。」

同日使用者裁定 L3 與自動推導的歸屬：「L3的workflow也是yield來協作。」「自動推導規則會由workflow/recipe內部，或是gui adapter的初始參數生成負責。」

### 四層

| 層 | 內容 | Agent 入口 | Agent 的控制粒度 |
| --- | --- | --- | --- |
| L0 實驗環境與硬體 | SoC、儀器 driver、program 執行 | 沒有直接入口，只能經由 L1 的 owner | 不適用 |
| L1 GUI／Notebook | 儀器控制、實驗建立、完整 cfg、實驗參數、單次分析與保存 | Measure 的 `rpc_list`／`rpc_describe`／`rpc_call`、`device_setup`、`simulation_initialize`、`tab_analyze`、`tab_interact`、`apply_writeback` | 一次實驗是多步流程，agent 可以在任兩步之間介入 |
| L2 Recipe | 常用單次實驗的自動 GUI 操作，參數少，其餘自動推導 | Measure 的各 recipe 工具，搭配 `recipe_guide`、`status`、`wait`、`answer`、`finish_early`、`cancel` | 一次實驗是一個整體。Agent 控制輸入、接受輸出，中途只在 recipe 明確 yield 時協作 |
| L3 Workflow | 一次設定，執行跨多組實驗的流程，例如 overnight、T1 over flux | Workflows MCP 的 Start、Pause、Resume、Stop 與 tunables | 現行以 Pause／Stop 和 tunables 介入；由 workflow 明確 yield 給 agent 是尚未落實的方向 |
| L3 跨實驗整合分析 | 整合多次量測的資料做聯合分析，例如 flux-dependence fit | Fluxdep MCP 的決策點工具；dispersive MCP 只讀 | 分析步驟由 agent 驅動，順序不固定；agent 的判斷落在宣告的決策點，程式能判斷的事不交給 agent |

Notebook 是使用者在 L1 的入口，agent 目前不經 Notebook 操作。

### Agent 介入點

正常使用時，agent 只提供真正需要它判斷的資訊。自動推導得出的參數由 recipe 或 workflow 補齊，agent 不必逐一傳入。Recipe 在已宣告的判斷交接處停下，把資訊和選項交給 agent。Workflow 的同類 agent 交接尚未落實。

L1 與 L2 的差別在於誰決定介入點。L1 由 agent 決定，它可以在任兩步之間停下、讀取或修改。L2 由 recipe 決定，只在明確需要 agent 的位置交還控制權，其餘步驟自動執行。交接可以落在輸入端（缺少必要參數）、中途（互動分析交接 tab，agent 以 `tab_interact` 完成選取）或輸出端（寫回確認）。Agent 需要在 recipe 沒有交接的位置調整步驟時，改用 L1。

Recipe 作者以兩種 `yield from` helper 區分目的。`tab.run_step()` 與 `run.analyze_step(...)` 是工作步驟，等待 GUI 操作並回傳具名結果，不表示每一步都要 agent 回答。分析需要互動時，driver 才交接 tab。`ask(run.propose_writeback())` 則明確交還控制權，等待 agent 或使用者回答寫回問題。`ask` 目前只接受已捕捉的 `WritebackQuestion`，回答本身不寫入資料。工作步驟與提問結果都保留 status；取消以 cancelled 結果交給作者收尾並 return，不改為例外。Operation 由 helper 交給 driver，不直接 yield，也不可迭代。

L3 workflow 的核准方向是採用與 L2 相同的 agent 協作方式，由 workflow 決定在哪裡交還控制權。一次執行跨越多次實驗，交接可出現在實驗之間或之內。現行 `yield from env.<effect>` 是交給 Engine 的工作請求，不是交給 agent 的提問，也不表示 workflow 已提供 `ask`。

2026-10-21 使用者修正 L2 與 L3 的定位與協作模型（推導見 [Recipe 接口重設計](../discussions/2026-10-21-recipe-interface-redesign.md)、[三角色協定](../discussions/2026-10-21-three-role-protocol.md)、[Workflow 定位與承載](../discussions/2026-10-21-workflow-positioning-and-hosting.md)；介面以實作 task 落檔為準）：

- Recipe 是操作 GUI tab 的腳本，可開多個 tab、串多個實驗；「參數少、單次實驗」不再是定位。原話：「recipe是一組操作gui的定位，其實不只能跑一次實驗，是可以做到類似workflow的行為的」。Recipe 與 workflow 的區別在執行基底（GUI tab 對 Engine）與誰持有迴圈，兩者都是腳本，輸入都以 pydantic `FormModel` 宣告，由框架投影成 GUI Form 或 MCP schema。原話：「該統一的是使用者與作者的認知與慣例」。
- 三角色協定：agent、腳本、runtime。腳本等 runtime 是 `Wait(op)`，腳本等 agent 是 `Ask(question)`，agent 主動通知腳本是 `Signal(kind, payload)`。Recipe 驅動器與 workflow Engine 底層共用這份協定，`tab.run()`、`env.run()` 等都是糖。原話：「我心智模型是三個角色：agent/recipe/runtime，recipe等agent就是Ask，recipe等runtime就是Wait。agent主動送訊息給recipe就是signal之類的」；「Engine底層應該也要用Ask/Wait模式，其他都是糖」。
- Recipe 執行分兩層：驅動引擎與實驗語義解耦，只處理 yield 流程；便利層提供 `run`／`analyze`／`ask_writeback` 等語法。原話：「recipe執行要分兩層，一層是驅動引擎，與實驗語義解耦，只負責處理yield的流程。另外一層利用前者，提供run/analysis/ask_writeback等便利語法」。Recipe inputs 的欄位由作者定義意義，框架不設帶 overrides 的基底。原話：「overrides不該出現在base class中，超過負責的範疇了」。
- Workflow 由 measure-gui 承載（同程序、同 session、同 RUN lease、同 MCP server），步驟不經 measure tab 執行。原話：「我也傾向B+D」（B 嵌入 measure-gui）。

上述取代本節「Recipe 作者以兩種 `yield from` helper 區分目的」一段的 `run_step`／`analyze_step`／`ask` 具名契約，以及「L3 workflow 的核准方向」一段中「現行 `yield from env.<effect>` 是交給 Engine 的工作請求」的描述；新契約由實作 task 寫回本篇與 measure MCP README。

跨實驗整合分析與 workflow 同屬 L3，但不由 workflow 決定 yield。Fluxdep 是獨立、可選的分析 app，不併入 measure 或 workflows。Agent 在資料可用性、對齊、採樣、躍遷假設、驗收與剔除六個決策點介入，各步可以重做。程式處理前置條件、計算與發布；agent 判讀證據並選下一步。人與 agent 共駕同一份 application state，寫入由 [[0070]] 的 holder 控制。

上層不重做下層的工作。Recipe 經由 L1 的 owner 操作 GUI。Workflow 經實驗 runtime 執行並自行保存，分工見 [[0062]]。跨實驗分析讀取下層已保存的資料。

### Generator 協作與主動訊息

Recipe 與 workflow 保留 generator，由 driver／Engine 掌握推進時機，不改成 async/await 或普通阻塞函式。工作步驟使用有型別的 `yield from` helper，與要求 agent 判斷的交接分開。Workflow effect 的精確回傳型別讓作者不必在每個呼叫點收窄共同的 send 聯集。使用者於 2026-10-06 裁定：「同意，一律 yield from」。

目前只支援兩類主動訊息。控制命令如 cancel、finish_early、Stop、Pause，在 Run 進行中透過協作式 StopSignal 生效；它們不是立即停止或 cleanup 完成的證明。帶資料的 workflow tunables 在步驟邊界生效，當前 step 保留自己的 capture。要重做當前步驟，先 Pause，再 Apply、Resume，見 [[0062]]。不支援帶資料且須在 Run 中即時反應，或腳本同時等待多個來源的訊息。使用者於 2026-10-16 裁定：「好，那先不支援c，做ab就好，順便把這討論紀錄到discussion」。這裡的 a、b、c 分別指上述控制命令、步驟邊界資料與 Run 中即時資料需求。

Recipe 的取消結果交給作者，讓作者仍能保存 partial 或補充說明，再自行 return。使用者於 2026-10-04 說明理由：「但我認為可以考慮讓recipe定義者自己決定在cancel或finish要採取什麼行動，這樣顯式的處理也比較符合boring code」。2026-10-16 再以「2A」選擇 helper 回傳具名 status、作者收尾，而非以專用例外終止 recipe。Driver 將具名結果放在 operation.delivery，以 None 推進 generator；真正的失敗仍 throw 回 helper 呼叫處。欄位與可用 helper 由 [measure MCP owner](../../lib/zcu_tools/mcp/measure/README.md#recipe-與-execution) 維護。

若出現 Run 中即時資料或同時等待多個來源的需求，再評估 async。這不新增 web／headless consumer，也不改 [[0053]] 的範圍。

### 層與層之間以共享狀態交接

各層之間不靠 agent 搬運數值。下層寫入、上層讀取的是同一份共享狀態：

- 工作點的參數與 module library，由 writeback 更新。
- Ledger 與 Experiment Data File，保存權威見 [[0063]]。
- `params.json` 的 typed handoff，例如 fluxdep 寫入、dispersive 讀取的 fit 結果。

Agent 可以讀這些狀態來判斷，但後續實驗取用參數時，以共享狀態為準，不以 agent 對話中記住的數值為準。

### 自動推導的 owner

自動推導參數的規則有兩類 owner：

- Recipe 與 workflow 在內部推導自己流程需要的參數。
- GUI adapter 在建立實驗時生成初始參數，例如從工作點或 module library 取預設值。

這兩類規則各自屬於自己的 owner，不抽到與前端無關的共用位置。同一個量在不同 owner 中可以有不同的推導方式，各 owner 對自己的規則負責。

Recipe framework 只提供通用 cfg 操作，不知道 calibration、實驗欄位的意義或取值退回順序。實驗 helper 留在 zcu_lab，以通用操作組成流程。原裁決日期未記錄，使用者於 2026-10-16 map 結案時確認沿用，原話「都採用建議」。原裁決原話為「框架應該連calibration這個概念都不該知道」。

Recipe 只曝露入口需要 agent 決定的參數。曝露多少、如何對到 aliases 或 cfg，由 recipe 作者決定，framework 不生成一份通用實驗參數面。原裁決日期未記錄，使用者於 2026-10-16 map 結案時確認沿用，原話「都採用建議」。原裁決原話為「recipe的參數應該是這個實驗入口需要agent需要agent決定的參數，其他都隱藏起來，所以要曝露多少是這個recipe的設計決定。」

### 選層

Agent 先找符合目標的 L3 workflow 或跨實驗分析，再找 L2 recipe，最後才用 L1。L1 也用於 recipe 無法涵蓋的細部設定、排查與修正。這是指引，不是權限模式，每一層的公開入口都可以直接呼叫，與 [[0068]] 一致。

降到下層時，agent 要說得出上層缺少什麼。降層的原因若反覆出現，應補進上層，不讓 agent 長期在 L1 手動重組同一流程。

## MCP 設計原則

這些原則適用於 GUI app 的 agent MCP，不要求 MCP 完整覆蓋 GUI 或 RPC。2026-10-12 使用者裁決分工：「前者(MCP)面向agent，設計上要考慮agent使用負擔，後者(RPC)是控制接口，設計上要簡單基礎通用。前者透過組裝後者來兼顧雙方需求」。同日裁決操作粒度：「對agent來說，tool本身就是意圖，送什麼tool就切換到什麼階段。」

### RPC 是控制接口，MCP 面向 agent

RPC 提供簡單、基礎、通用的控制，回完整資料，出圖使用獨立 method。MCP 組裝 RPC、分派階段、摘要回覆並選擇出圖時機。GUI 是唯一的業務真相，MCP 不保存另一份業務狀態或驗證規則。傳輸及控制身份分工見 [[0068]]、[[0070]]；MCP→GUI remote 的依賴由 C13 約束。

### 從決策點列能力

先列 agent 必須判斷的事和需要的證據，再列完成這些判斷所需的工具。其餘 GUI 能力留在 RPC。Fluxdep 的決策點是資料可用性、對齊、採樣、躍遷假設、驗收和剔除。2026-10-12 使用者裁定範圍：「mcp的功能不必完全覆蓋gui的能力，只要能滿足：agent用mcp開啟或連上fluxdep_fit gui，讀取/加入/移除新數據，錨定對稱點，生成採樣點，Export數據集與採樣結果，Restore之前的數據集與採樣結果，Filter過濾此採樣點，附帶參數擬合目標參數，查看結果。每一步不一定照順序，也不一定只有一次。」

### 在決策點回證據圖

程式不能代為判斷的事回圖，機械命令回數字摘要。設計者給每個工具決定預設出圖時機，agent 可以要求 image=true。證據圖與 GUI 共用 renderer，兩個前端看同一份資料。Fluxdep 的 finish 回階段圖，fit 回 Show 疊圖。過期或沒有 recorded quality 的結果不回疊圖，這項限制由 fluxdep 的結果契約決定。

### Tool 即意圖

Fluxdep 對某個對象送階段工具，就進入該階段。Agent 不管理 open、cancel 或 context id。階段以編輯命令與 finish 組成；finish 接受並回證據圖，未編輯直接 finish 表示照原樣接受。切換顯示不取消別的對象草稿，GUI 跟隨 agent 操作的對象。Context lifetime 與同對象換階段的草稿處理仍由 GUI owner 決定。這不取消 measure 的 recipe cancel 入口。

### 命令直接表達動作

模式是命令的參數，不另設隱藏模式。能由其他命令表達的操作不另設命令。Fluxdep 的每個編輯階段以 finish 結束。它的 move_line 明填角色、位置與 conjugate；MCP 不暴露 swap_lines、set_tool、perform_on_all、clear 或 cancel。

### 不遷就 GUI 的操作方式

2026-10-14 使用者裁決：「mcp不要遷就gui的操作方式，能夠直接設定setup/device/inspect/...就不用打開dialog，也不要設定dialog focus自動跟隨」。

- Setup、Device、Inspect（參數、modules、aliases）等 dialog 編輯的內容，MCP 直接以 RPC 讀寫負責保存的 owner，不經過 dialog。
- Agent 的操作不開啟 dialog，也不切換 dialog 的分頁或焦點。`dialog.screenshot` 只擷取使用者已開的 dialog 目前顯示的頁，不改畫面。
- 本節只管 dialog。Agent 操作分頁時 GUI 跟隨的規則不變，使用者同日確認：「tab跟隨我保留」。

### 只回判斷所需的摘要

逐點陣列留在 RPC 與匯出檔。MCP 回計數、範圍與差量；changes 只列變動欄位的 [before, after]，invalidated 投影原生 publication 的下游失效。點或 mask 的新增／移除計數用 [added, removed]，不與欄位差量混用。參數 schema、範圍與領域詞彙放工具描述，不在每次回覆重複。精確值讀匯出檔。

### 長操作看 agent 中途能否影響結果

操作進行中有 agent 能判斷且能影響結果的事，就回 token 並提供進度與提早停止。沒有時，MCP 輪詢 RPC 到結束，不設整體上限，也不把 token 交給 agent。代價是同一 stdio server 等待期間不處理其他呼叫。Fluxdep 的 auto_align 與 search 採後者；measure Run 可看進度與 early_stop，採前者。

### 人與 agent 共駕

Measure／fluxdep 的 GUI 與 agent 共用同一份 application state，同時只有 holder 寫入，見 [[0070]]。狀態、圖與失效標示由同一 owner 提供，MCP 回覆能在 GUI 核對。控制轉移不取消已接受的工作，拒絕後讀現況再決定，不重送或回滾。本篇不把 holder enforcement 擴張到其他 app。

### 先做簡單設計，再按需加法

第一版只實作能力清單所需的命令與回覆。覆寫、選項與細部欄位有具體需求才加入。Fluxdep 的摘要與組裝需求來自實際分析中逐點陣列、schema 與重讀的成本，不把成本估算當驗收門檻。

### 判讀依據放知識庫

Skill 放流程與完成條件，每個決策點保留最小判準，並指向知識庫入口。完整判讀依據、反例與圖例放 measure_knowledge，不由單一前端擁有。GUI、MCP 與 Notebook 的 skill 可以共用它；skill 在沒有知識庫時仍能依最小判準運作。

### Fluxdep 的依賴模型不是跨 app 原則

2026-10-12 使用者對將失效與品質提升成通用範式裁定：「這點還不確定，先不進範式。」Fluxdep 的 alignment → points → Filter → fit 依賴、recorded quality、過期圖與匯出限制是 app 的局部契約，見 [[0068]] 及 app README。其他 app 的失效模型由各自的需求決定。

## 現況與後續

- Workflow 的 agent yield 尚未落實。現行 L3 workflow 把接受度寫在 workflow 定義與 tunables 中，校準失敗或 fit 不被接受時提交具名 reason，agent 可用 Pause／Stop 與 tunables 介入，不能回答 workflow 的 agent 提問。2026-10-12 使用者說明另有 task 正在重構 workflow 並實作此協作。Yield 的介面以該實作為準，本篇不先行規定。
- 核准方向是無人回應時停在 yield。2026-10-12 使用者裁定：「就卡住，有需求再加設計。」Workflow yield 後沒有 agent 或使用者回應時，流程停在該處等待。套用預設決定、逾時或跳過等機制，等有具體需求再設計。因此含 yield 的 workflow 在無人值守時可能整段停住，overnight 流程要靠宣告的接受度減少 yield。
- L0 邊界只靠政策維持。Agent 有 shell 時，技術上仍可繞過 MCP 碰觸硬體。目前由 MEASUREMENT 角色規則約束「不以分析腳本旁路控制硬體」。2026-10-12 使用者裁定：「有需要再加設計。」出現具體失敗類別時，再升級成權限設定或執行期強制。

## 取捨

- **自動化往下推，而不是讓 agent 編排每一步。** 另一種做法是只提供 L1 粒度的工具，由 agent 決定每一步參數、每一步之後的判斷與下一步。那樣新增實驗的成本低，agent 看得到每一步的完整資料。本篇選擇自動化往下推。Recipe 與 workflow 的步驟由程式碼決定，agent 不必逐步選擇操作，減少模型推理與工具往返。代價是每個 recipe 與 workflow 都要寫程式碼，自動推導的規則必須正確，agent 也較晚察覺推導規則沒有涵蓋的異常。
- **保留 L1 完整入口。** 只提供 recipe 與 workflow 會讓少見情況無路可走。保留 L1 的代價是工具面較大，agent 可能過早降層，所以選層指引要求說明降層原因。
- **推導規則留在各 owner，而不是抽成共用規則。** 共用規則能讓同一個量只有一種推導方式，但會把 recipe、workflow 與 GUI adapter 綁在同一份定義上，任何一方調整推導都要顧及其他使用者。留在各 owner 讓每個流程能依自己的需要推導，代價是同一個量的推導可能重複或不一致，需要時由各 owner 自行對齊。
- **以共享狀態交接，而不是由 agent 傳值。** 這讓 GUI、Notebook 與 agent 看到同一份校準結果，也讓 workflow 能無人值守地延續參數。代價是寫回需要明確的接受流程與 provenance。

本篇不新增 package 依賴方向。MCP 經 GUI remote 的方向沿 [[0068]] 與 C13；framework 不反向 import 使用者定義沿 [[0072]] 與 C16；workflow Engine 不依賴前端沿 [[0062]] 與 C18。這些既有 `.importlinter` contracts 不因介面層級而放寬。

## 相關文件

- [measure MCP README](../../lib/zcu_tools/mcp/measure/README.md)：固定工具、recipe 與 execution 的使用契約。
- [fluxdep MCP README](../../lib/zcu_tools/mcp/fluxdep/README.md)：跨量測整合分析的決策點工具。
- [zcu_lab workflows README](../../zcu_lab/workflows/README.md)：具體 workflow 的校準、接受度與保存。
- [run-measure-gui skill](../../.agents/skills/run-measure-gui/SKILL.md)：agent 量測時的操作規則。
- [討論紀錄](../discussions/2026-10-12-agent-interface-layers.md)：本篇的推導過程與被否決的選項。
