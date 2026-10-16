---
date: 2026-10-12
status: active
superseded_by: null
related:
  - docs/adr/0075-agent-interface-layers.md
  - docs/discussions/2026-10-12-qca-frontend-evaluation.md
basis:
  - main@9c956097f
---

# Agent 介面分層的推導

## 結論

measure-gui 相關的 MCP 依抽象程度分成四層：L0 實驗環境與硬體、L1 GUI／Notebook、L2 recipe、L3 workflow 與跨實驗分析。設計方向是把自動化往下推，agent 只在流程宣告的判斷交接處介入，層與層之間以工作點、ledger、資料檔等共享狀態交接。決策本身寫在 [ADR-0075 Agent 介面分層](../adr/0075-agent-interface-layers.md)，詞彙寫在 [CONTEXT](../CONTEXT.md) 的「Agent 介面」一節。本篇記錄推導過程與對照。

## 問題

評估 QCA 時（見 [QCA 評估](2026-10-12-qca-frontend-evaluation.md)），兩者的差異最後歸結到「agent 在流程中扮演什麼角色」。現有的 measure、workflows、fluxdep、dispersive MCP 各自提供工具，但各入口的定位、agent 的介入點與層間交接沒有寫在同一處。

## 考慮過的選項與比較

### Agent 編排每一步

QCA 的模型：agent 決定每一步的參數、每一步之後的判斷與下一步，參數由 LLM 在實驗之間搬運，workflow 的每個 node 由一個 subagent 執行。

- 新增實驗的成本低，一個回傳 dict 的函式就是一個實驗。
- Agent 看得到每一步的完整資料。
- Token 用量、執行時間與結果都取決於模型；流程不能無人值守；參數搬運與單位換算容易出錯。

### 自動化往下推

本 repo 的模型：recipe 自動推導參數，workflow 一次設定後自己執行，agent 只在需要判斷時被叫出來。

- 步驟由程式碼決定，同一份設定得到同一串操作。
- 每個 recipe 與 workflow 都要寫程式碼，推導規則必須正確。
- Agent 較晚察覺推導規則沒有涵蓋的異常。

採用此選項。逐層對照如下：

| 本 repo 的層 | 本 repo 的實作 | QCA 的對應 |
| --- | --- | --- |
| L0 實驗環境與硬體 | 只能經由 L1 owner | 沒有邊界，script 直接操作硬體，agent 另有 shell |
| L1 GUI／Notebook | `rpc_*`、`device_setup`、`tab_analyze`、`tab_interact` | 與 L0 合併在 script 裡 |
| L2 Recipe | 自動推導，在宣告處 yield | 沒有，參數全由 LLM 傳入 |
| L3 Workflow | 一次設定，跨多組實驗 | LLM 逐 node 執行與判斷 |
| L3 跨實驗分析 | fluxdep、dispersive | 沒有，只能臨時寫分析腳本 |
| 層間共享狀態 | 工作點、library、ledger、`data.h5` | 只有 workflow context 與 LLM 記憶 |

## 被否決的選項與理由

- **Agent 編排每一步**：結果依賴模型，無法無人值守，參數搬運不可靠。
- **只提供 recipe 與 workflow，不保留 L1**：少見情況會無路可走。改為保留 L1，並要求 agent 降層時說明上層缺什麼。
- **把自動推導規則抽到與前端無關的共用位置**：會把 recipe、workflow 與 GUI adapter 綁在同一份定義上。使用者裁定推導規則留在各 owner。
- **無人回應 yield 時套用預設決定或跳過**：使用者裁定先停在 yield 等待，有需求再設計。

## 討論中使用者的判斷與修正

1. 使用者先說明分層定位：L0 對使用者與 agent 都不可直接控制；L1 包含儀器控制、實驗建立、完整 cfg 與實驗參數；L2 recipe 參數少、其餘自動推導、只在需要時 yield；L3 workflow 一次設定完成跨多組實驗，同層還有跨實驗整合分析。
2. 同意 `tab_analyze` 與 `tab_interact` 放在 L1，並說明 L1 的實驗是 agent 可中途介入的多步流程，L2 的實驗是只控制輸入與接受輸出的整體。
3. 隨後修正：L2 不是完全無法中途介入，而是在明確需要 agent 時用 yield 協作。區別因此從「能否介入」改為「誰決定介入點」：L1 由 agent 決定，L2 由 recipe 決定。這也解決了 recipe 互動分析位於中途的矛盾。
4. L3 workflow 同樣以 yield 協作。另有 task 正在重構 workflow 並實作此協作。
5. 自動推導由 workflow／recipe 內部，或 GUI adapter 的初始參數生成負責。
6. 無人回應 yield 時就停住；L0 的強制機制有需要再設計。
7. ADR 中不提 QCA，QCA 只作為討論的對照。

## 留下的問題

- Workflow yield 的介面以進行中的重構為準。記錄當日查閱本機 workflow 任務規格，沒有找到 agent 判斷交接的設計；現有的 `yield from env.<effect>` 是 engine 的 effect 機制，不是交給 agent 的判斷交接。實作 task 的 spec 可能需要補上。
- 含 yield 的 workflow 在無人值守時可能整段停住，overnight 流程需靠宣告的接受度減少 yield。
- L0 邊界只靠 MEASUREMENT 角色規則維持。

## 依據

- [ADR-0075 Agent 介面分層](../adr/0075-agent-interface-layers.md)
- [ADR-0068](../adr/0068-remote-transport.md) 的 recipe 優先與 RPC 排查指引
- [zcu_lab workflows README](../../zcu_lab/workflows/README.md) 的接受度與 reason 提交
- [measure MCP README](../../lib/zcu_tools/mcp/measure/README.md) 的 recipe 互動交接
- [run-measure-gui skill](../../.agents/skills/run-measure-gui/SKILL.md)
