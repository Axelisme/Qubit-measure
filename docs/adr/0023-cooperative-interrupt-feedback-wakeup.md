# ADR-0023 — Cooperative interrupt：user feedback 喚醒 pending wait

## 脈絡

要讓使用者**委派一個 Claude agent 操作 GUI**（經 `gui_*` MCP 工具）並能**即時插話糾正**，且限定在**訂閱認證 + CLI subprocess**（非 API key）下。查證結論：乾淨的 turn-中途 `interrupt()`（Agent SDK）只能 API key（訂閱 OAuth 用於 SDK 被官方禁止）；CLI subprocess 只能做 turn-之間打斷或粗暴 SIGINT。

## 決策

採 **cooperative interrupt**，不做 OS-level interrupt：

利用「agent 在一個 turn 裡會卡在長 `wait` 上等 tool result」這個天然 parking point。整條 wait 的阻塞點是**單一共用點** `OperationHandles.await_outcome`（IO worker thread 上的 `threading.Event.wait()`，`off_main_thread=True`；鏈：mcp `_await_operation_by_key` → RPC `operation.await` → dispatch `_h_operation_await` → `Controller.await_operation` → `await_outcome`）。

給這個 await **加第二喚醒源 = 一個 thread-safe user-feedback inbox**：pending wait 在「operation 完成 / inbox 有 feedback / timeout」三者之一時返回，payload 帶**判別欄位** `{reason: "completed"|"user_feedback"|"timeout", result?, feedback?}`。agent（持 operation_id）下一個 reasoning step 即把 feedback 當高優先指令重新規劃，並可選擇 `cancel` 或讓 operation 續跑。

因為阻塞點唯一，第二喚醒源加一處即讓**所有 `*_wait` 受益**。

## 分層（依 ADR-0002 三層）

- **mechanism（RPC/gui）**：`await_outcome` 第二喚醒源 + thread-safe inbox + dispatch payload 加 `reason/feedback`。inbox 寫入來自 GUI 主執行緒（使用者打字 widget）、讀取在 IO worker；**不放 State**（State 主執行緒寫入不變式，見 ADR-0007），由 Controller 持有 thread-safe inbox。
- **簿記（mcp）**：`mcp/measure` 把判別 payload 原樣透傳。
- **語義（agent）**：`*_wait` 工具 description / SKILL 明示「`reason=user_feedback` → 重規劃、持 operation_id 可 cancel/續跑」。

## residual / 邊界

- agent 當下沒在 wait（推理中／連打瞬時呼叫／idle）：feedback 留在 inbox，下個 wait 進入時立即帶出；GUI 提示「將於下個等待點生效」。被動 piggyback（折進下個 reply）作為 residual 通道。
- 真正暴走（不 wait）：留 **SIGINT** 當硬停逃生口。
- 提早返回後 operation 仍在跑（op_id 有效），cancel/續跑由 agent 決定（與既有 handle 模型一致）。

## 與既有設計關係

- **擴展 ADR-0002**：在既有 async-handle await 上補一個返回條件，非新並發機制。
- **ADR-0019（Operation facets）**：wait 屬 Handle facet 的 await，語義不變、只多一個喚醒源。
- **取代/具體化** `project_gui_agent_ux` 擱置的 user→agent gui-inbox：本 ADR 是**主動喚醒**實現；該記憶描述的「反向複用 piggyback」降為被動 residual 通道（兩者互補，非同一機制）。
