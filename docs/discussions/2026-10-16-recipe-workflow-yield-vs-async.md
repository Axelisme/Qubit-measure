---
date: 2026-10-16
status: active
superseded_by: null
related:
  - docs/adr/0075-agent-interface-layers.md
  - docs/adr/0053-owner-scheduler-and-gate-presence.md
  - docs/adr/0062-experiment-workflow.md
  - docs/discussions/2026-10-12-agent-interface-layers.md
basis:
  - "repo@8e0f25cd1，討論時的 main HEAD"
  - "Python 3.13；repo 內沒有 asyncio 或 qasync 依賴"
---

# Recipe 與 workflow 的協作模型：generator、阻塞函式或 async

## 結論

Recipe 與 workflow 保留 generator 的 coroutine 模型，由 driver 掌控進度。不改成 async/await，也不改成普通阻塞函式。

Agent 或使用者主動送給執行中腳本的訊息，只支援兩類：

- (a) 控制命令：cancel、finish_early、Stop、Pause。Run 進行中就會生效。
- (b) 帶資料的訊息：workflow tunables。在步驟邊界生效。

不支援 (c)：帶資料、而且 Run 進行中就要反應的訊息。

Recipe 改用和 workflow 一樣的有型別 `yield from` helper，並把「工作步驟」與「交還控制權」分成兩種呼叫。決策目前記在 storage-redesign map 的決策紀錄，之後寫進 [ADR-0075 Agent 介面分層](../adr/0075-agent-interface-layers.md)。

## 問題

使用者對 generator 提出兩點疑問：

- async/await 是否比較原生。
- `Generator[Y, S, R]` 的所有 yield 共用同一個 send 型別，而且 effect 需要 driver 特別處理。

使用者的設計脈絡是兩個角色的互動。程序化的腳本負責確定、穩定的內容。遇到需要抽象判斷的地方，腳本交還控制權並提交資訊給 agent 或使用者；對方判斷後回覆，腳本再繼續。這個互動模型很像 IO 的 async。

使用者另外指出第二個管道：agent 或使用者根據 liveplot 等即時資訊，主動傳訊息給腳本。

## 現況

### 兩套 generator

Recipe 位於 `zcu_lab/recipes/`，共 11 個檔、32 處 yield。

- 它 yield 三種 operation：`RunOperation`、`AnalyzeOperation`、`WritebackQuestion`。
- 每次 yield 收回的都是同一個 tuple 型別 `RecipeDelivery`（`lib/zcu_tools/mcp/measure/recipe.py`），所以 11 個 recipe 都重複寫了 `assert isinstance(run, RecipeRun)`。
- Driver 在每個 execution 的 daemon thread 上用 `send` 與 `throw` 推進 generator（`recipe_execution.py`）。
- 等 agent 回覆時，thread 停在 `Condition.wait_for`。Generator 沒有跨 MCP 呼叫暫停，也從不序列化。

Workflow step 位於 `zcu_lab/workflows/`，共 23 處 `yield from`。

- 它有三種 effect：Run、Device、Wait。send 型別是 `None`。
- `yield from env.run(...)` 直接回傳有型別的 `Completed | Failed`，整套沒有 cast。
- 執行在專用的 QThread 上。

Repo 內沒有 asyncio。ADR-0053 把 asyncio adapter 列為非目標，留給 web 或 headless consumer 出現時再處理。

2026-10-06 的 workflow 提案曾否決「workflow 寫成普通函式，暫停靠 `env.run` 內阻塞」。當時使用者要求保留 `yield` 來控制 workflow 的進行。

### 兩種 yield

這次討論把現有的 yield 分成兩種：

- 工作步驟：Run、Analyze、Device、Wait。這是腳本自己的確定性工作，交給 driver 執行。
- 交還控制權：例如 `WritebackQuestion`。腳本提交資訊，等 agent 或使用者判斷。

Recipe 用同一個 send 型別承載這兩種 yield，這是型別薄弱的根本原因。

### 主動訊息管道

| 類型 | 現有機制 | 生效時機 |
| --- | --- | --- |
| (a) 控制命令 | MCP cancel、finish_early，GUI Stop，workflow Pause、Stop | 共用一個 stop event。acquire 迴圈在 round 內輪詢它，裝置 ramp 與 `wait_until` 也會檢查它 |
| (b) 帶資料，邊界生效 | workflow tunables，一批 JSON，帶 revision | 每次呼叫 step 前讀一份快照。要讓當前這步生效，先 Pause、修改，再 Resume 重做這一步 |
| (c) 帶資料，Run 中反應 | 沒有 | — |

實際使用紀錄只有一次主動介入：agent 看了 live 圖後 cancel，再存下部分結果。Pause、tunables 與成功的 finish_early 都沒有使用紀錄。2026-10-06 的提案把跳過點、修改 flux 清單、回頭重跑列在範圍外，也否決了「等當前點量完再停」。

### 多來源等待已經存在

Workflow 等裝置設定完成時，agent 可能送出停止命令，這就是同時等兩件事。現在的做法是把協作式取消 token（`StopSignal`）傳進阻塞呼叫，例如 `setup_device(name, cfg, signal)` 與 `wait_until` 的 `event.wait(remaining)`。

停止是一種特殊的第二來源，因為它的語意永遠是「中止這一步」。所以由 driver 與 port 統一處理，腳本作者看不到。

改用 async 後，等待那一側可以寫成 `asyncio.wait({device_task, stop})`。但 `device_task` 底下仍是包在 `to_thread` 裡的阻塞硬體呼叫，asyncio 無法中斷它。要讓 ramp 停下來，仍然必須把 token 傳進去。

## 考慮過的選項

| 選項 | 內容 | 成本 | 型別 |
| --- | --- | --- | --- |
| 有型別的 `yield from` helper（採用） | Recipe 比照 workflow，`run = yield from tab.run_step(...)`、`decision = yield from ask(question)` | 改 recipe 框架與 11 個 recipe | 每種呼叫各有回傳型別，與 `await` 等效 |
| 普通阻塞函式 | 每一步直接呼叫，在 worker thread 上阻塞，取消時丟例外 | 改兩個 driver、55 處呼叫點、約 150 行測試 | 完整 |
| async/await | `await agent.ask(...)`，每個 execution thread 跑自己的 loop | driver 約 4000 行改寫，每個阻塞呼叫要包 `to_thread`，要推翻 ADR-0053 | 完整 |

## 被否決的選項與理由

- **async/await**：它的主要好處是讓腳本作者依「哪個來源先到」分支，也就是 (c)。目前沒有 (c) 需求，而 (a)、(b) 用 token 與快照就能處理。型別上的好處，有型別的 `yield from` helper 也拿得到。換到 async，成本與 ADR-0053 衝突都不划算。
- **普通阻塞函式**：advisor 一度建議過，但它與使用者先前「保留 yield 控制進行」的裁決衝突，所以撤回。
- **支援 (c)**：(c) 確實比較靈活，但腳本的行為會取決於訊息到達的時間，可重現性與測試都會變差，也削弱了「腳本負責確定性內容」的角色分工。Pause、修改 tunables、Resume 重做這一步，已經能涵蓋多數想在執行中改參數的情況，代價只是損失當前這一步的進度。

## 留下的問題

- 什麼時候重新評估 async：出現 (c) 類需求，或 ADR-0053 預留的 web 或 headless consumer。
- 如果只需要讓腳本在自己選的點接收帶資料訊息，先考慮 `update = yield from env.receive(SpanUpdate)` 這類接收點。接收點由腳本宣告，訊息寫進 journal，仍在 generator 模型內，不需要 async。
- ADR-0075 Agent 介面分層提到 workflow 的 agent yield 由另一項重構實作，但目前沒有相應程式碼。實作時，它的回覆屬於「交還控制權」這一類，應該用有型別的 `ask` 呼叫。

## 依據

- 調查紀錄：`.agent_state/maps/storage-redesign/scratch/yield-vs-async-research.md`，含檔案與行號。
- `lib/zcu_tools/mcp/measure/recipe.py`、`recipe_execution.py`
- `lib/zcu_tools/experiment/workflows/execution.py`、`env.py`、`effects.py`、`models.py`、`clock.py`
- `docs/adr/0053-owner-scheduler-and-gate-presence.md`
- `.agent_state/proposals/2026-10-06-autofluxdep-workflow-engine.md`
