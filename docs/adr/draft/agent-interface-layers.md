---
status: draft
---

# Agent 介面分層

**狀態：** 使用者已說明定位，本篇待 review 後轉為 accepted。本篇整理既有入口的定位，不新增工具、不改 method policy，也不授權硬體操作。
**關聯：** [[0068]] 的 remote 與 measure 固定工具、[[0062]] 的 workflow run、[[0063]] 的保存權威、[[0069]] 的核心與前端；控制權見 [控制身份 draft](control-identity.md)，workflow engine 見 [workflows draft](workflows-engine.md)。

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
| L3 Workflow | 一次設定，執行跨多組實驗的流程，例如 overnight、T1 over flux | Workflows MCP 的 Start、Pause、Resume、Stop 與 tunables | 一次設定後自動執行，中途只在 workflow 明確 yield 時協作 |
| L3 跨實驗整合分析 | 整合多次量測的資料做聯合分析，例如 flux-dependence fit | Fluxdep MCP 的 pipeline 與互動工具；dispersive MCP 只讀 | 分析步驟由呼叫者驅動 |

Notebook 是使用者在 L1 的入口，agent 目前不經 Notebook 操作。

### Agent 介入點

正常使用時，agent 只提供真正需要它判斷的資訊。自動推導得出的參數由 recipe 或 workflow 補齊，agent 不必逐一傳入。自動流程遇到需要判斷的狀況時，停在判斷交接處，把資訊和選項交給 agent。

L1 與 L2 的差別在於誰決定介入點。L1 由 agent 決定，它可以在任兩步之間停下、讀取或修改。L2 由 recipe 決定，recipe 在明確需要 agent 的位置 yield，其餘步驟自動執行。Yield 可以落在輸入端（缺少必要參數）、中途（互動分析交接 tab，agent 以 `tab_interact` 完成選取）或輸出端（寫回確認）。Agent 需要在 recipe 沒有 yield 的位置調整步驟時，改用 L1。

L3 workflow 與 L2 用同一種協作方式。Workflow 決定在哪裡 yield，其餘步驟自動執行。差別在於 workflow 的一次執行跨越多次實驗，yield 可以出現在任一次實驗之間或之內。

上層不重做下層的工作。Recipe 經由 L1 的 owner 操作 GUI，workflow 經由同一套 Run 與保存流程執行實驗，跨實驗分析讀取下層已保存的資料。

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

### 選層

Agent 先找符合目標的 L3 workflow 或跨實驗分析，再找 L2 recipe，最後才用 L1。L1 也用於 recipe 無法涵蓋的細部設定、排查與修正。這是指引，不是權限模式，每一層的公開入口都可以直接呼叫，與 [[0068]] 一致。

降到下層時，agent 要說得出上層缺少什麼。降層的原因若反覆出現，應補進上層，不讓 agent 長期在 L1 手動重組同一流程。

## 現況與後續

- **Workflow 的 yield 由進行中的重構實作。** 現行 L3 workflow 把接受度寫在 workflow 定義與 tunables 中，校準失敗或 fit 不被接受時提交具名 reason，agent 只能 Pause 或 Stop。2026-10-12 使用者說明另有 task 正在重構 workflow 並實作此協作。Yield 的介面以該實作為準，本篇不先行規定。
- **無人回應時停在 yield。** 2026-10-12 使用者裁定：「就卡住，有需求再加設計。」Workflow yield 後沒有 agent 或使用者回應時，流程停在該處等待。套用預設決定、逾時或跳過等機制，等有具體需求再設計。因此含 yield 的 workflow 在無人值守時可能整段停住，overnight 流程要靠宣告的接受度減少 yield。
- **L0 邊界只靠政策維持。** Agent 有 shell 時，技術上仍可繞過 MCP 碰觸硬體。目前由 MEASUREMENT 角色規則約束「不以分析腳本旁路控制硬體」。2026-10-12 使用者裁定：「有需要再加設計。」出現具體失敗類別時，再升級成權限設定或執行期強制。

## 取捨

- **自動化往下推，而不是讓 agent 編排每一步。** 另一種做法是只提供 L1 粒度的工具，由 agent 決定每一步參數、每一步之後的判斷與下一步。那樣新增實驗的成本低，agent 看得到每一步的完整資料。本篇選擇自動化往下推。Recipe 與 workflow 的步驟由程式碼決定，同一份設定得到同一串操作，token 用量與執行時間不隨模型改變。代價是每個 recipe 與 workflow 都要寫程式碼，自動推導的規則必須正確，agent 也較晚察覺推導規則沒有涵蓋的異常。
- **保留 L1 完整入口。** 只提供 recipe 與 workflow 會讓少見情況無路可走。保留 L1 的代價是工具面較大，agent 可能過早降層，所以選層指引要求說明降層原因。
- **推導規則留在各 owner，而不是抽成共用規則。** 共用規則能讓同一個量只有一種推導方式，但會把 recipe、workflow 與 GUI adapter 綁在同一份定義上，任何一方調整推導都要顧及其他使用者。留在各 owner 讓每個流程能依自己的需要推導，代價是同一個量的推導可能重複或不一致，需要時由各 owner 自行對齊。
- **以共享狀態交接，而不是由 agent 傳值。** 這讓 GUI、Notebook 與 agent 看到同一份校準結果，也讓 workflow 能無人值守地延續參數。代價是寫回需要明確的接受流程與 provenance。

## 相關文件

- [measure MCP README](../../lib/zcu_tools/mcp/measure/README.md)：固定工具、recipe 與 execution 的使用契約。
- [fluxdep MCP README](../../lib/zcu_tools/mcp/fluxdep/README.md)：跨量測整合分析的 pipeline。
- [zcu_lab workflows README](../../zcu_lab/workflows/README.md)：具體 workflow 的校準、接受度與保存。
- [run-measure-gui skill](../../.agents/skills/run-measure-gui/SKILL.md)：agent 量測時的操作規則。
- [討論紀錄](../../discussions/2026-10-12-agent-interface-layers.md)：本篇的推導過程與被否決的選項。
