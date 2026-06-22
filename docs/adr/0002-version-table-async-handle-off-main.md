---
status: accepted
---

# 並發感知 = 資源版本表 + RPC-as-proxy async handle + off-main blocking handler

**狀態：** accepted（已實作；版本表 guard + 三層分工 + operation handle 全在線）。
**關聯：** 承 [[0001]]（Permit/Lease；async handle 是 lease 的延伸）；off-main 契約被 [[0003]]（shutdown）、[[0014]]（shared transport）援用；external-refresh Reaction 模式見 [[0004]]。

## 脈絡

GUI 有兩個平級 client（Qt View、remote RPC agent）並發驅動同一批受保護操作。並發感知要解三件事：(1) guard 不能擋 agent 自己剛做的事、(2) async 操作（device.setup / run / connect）完成要能通知 agent、(3) 主線程不能阻塞（卡 event loop → 死鎖）。

## 決策

三條**正交**的線。

### 1. 資源版本表（只服務 guard）

- **粒度（中粒度）**：`context`、`soc`、`device:<name>`、每 tab 的 `tab:<id>:cfg` / `:result` / `:save_path` / `tab:<id>`（存在性）、`editor:<id>`。tab 資源綁 `tab_id`（uuid4，永不重用）→ 無 key 撞名。
- **版本號 = per-resource 單調遞增整數**（非 wall-clock）。`VersionTable` 是 `State` 的一個區塊。
- **bump 責任歸資源 owner service，且在「資源實際被寫」同點、必在主線程**：這是更普遍不變式的推論——**所有 `State` 寫入只在主線程；worker 唯一允許的副作用是 emit Qt signal**。sync 操作在 service mutator bump；async 操作（device/run/connect worker）在 worker 回主線的 **terminal Qt slot** bump（與 `update_tab_result` 同處）。不靠 origin、不靠 emit/release 順序——只靠「在主線、在資源被寫處」。
- **bump = 狀態真的變了，不含「值未變的快取同步」**：讀取衍生的快取更新若值未變則不 bump、不 emit（否則純讀 spurious 推進版本號、誤使他人 `expected_versions` 失效）；讀到外部來源真的變了才 bump + emit。
- **guard 用 optional `expected_versions`（類 HTTP If-Match）**：`run.start` / `save.*` / `editor.commit` 帶可選參數，server 在主線 `_dispatch._run()` 單一同步序列內**原子**比對，不符→`PRECONDITION_FAILED` + 回當前版本。依賴消失（tab close 刪 entry）= 視同 stale 擋下。

### 2. RPC-as-proxy 持 async handle

「一個 async 操作」收斂成 async-task 心智模型的 handle，責任分離：`_OperationExclusion`（互斥）+ `_OperationRegistry`（per-token status/outcome/Event，LRU 保留）+ `OperationGate`（façade）。共用同一 operation token。

- `OperationOutcome`（中性 frozen：finished/failed/cancelled + error，不帶 result）。`release(lease, outcome)` 分派：exclusion 即移（讓出 hardware）+ registry settle（保留供 late awaiter）。
- device/run 啟動回 `operation_id`（= token）；terminal slot 帶 outcome release。
- 查詢/等待：`poll(token)`（非阻塞）/ `await_outcome(token, timeout)`（off-main 阻塞）；**查無 token = 視為已完成立即返回**（不 hang）。
- analyze 也納入 handle（`OperationKind.ANALYZE`，never-conflict、handle-only）。

> **修訂（2026-06-17）：`connect` 已改為同步，不再是 async handle。** SoC connect 收斂成同步 `soc.connect` wire RPC
> （on-main marshalled 執行，由 `make_soc_proxy` 的 ~1s connect-scoped timeout 界定上界故不會 hang），**不再 mint
> operation_id、不走 await/poll**——SoC connect 快且有界，handle 無增益。現存 async handle 僅 `device.setup` / `run`
> （+`analyze`）。`SOC_CONNECT` Lease 仍取用以協調兩個 connect 入口（GUI 按鈕 + `soc.connect` wire RPC）。
> connect-scoped 的 1s timeout 必須 save/restore global `COMMTIMEOUT` **並**重置回傳 proxy 的 `_pyroTimeout`，
> 否則量測 proxy 會殘留 1s cap（見 `remote/pyro.py`）。

### 3. off-main blocking handler

`MethodSpec.off_main_thread`（預設 False）。`_dispatch` 看到 True 不 marshal 上主線，在 IO worker thread 直接執行。受**嚴格契約**：只能做 thread-safe 等待，**不得碰 main-thread-owned 狀態**（版本表 / change-related / CfgEditor / `_snapshots`）、不需要 stale guard。`operation.await` 即此類。

## 三層分工（脊椎）

- **RPC = mechanism**：持版本表、提供 `resources.versions`、guard 原子比對、回 `operation_id`。
- **mcp = policy + 簿記 + 翻譯**：持 `last_seen`、知道每操作依賴哪些資源、組 `expected_versions`、收 `PRECONDITION_FAILED`；持「語義 key→最新 operation_id」對照（device→name / run→tab_id；connect 已改同步 `soc.connect`、無 operation_id）。版本號與 operation_id **只在 RPC↔mcp 之間流動**，從回傳 strip 掉。
- **agent（LLM）= 只收語義**：
  - resource-change 感知 = **樂觀 + guard 撞牆**：mcp 把 `PRECONDITION_FAILED`（帶 `data.stale` 資源身份）翻成「tab X cfg 過時了」。agent **不 subscribe event、不看版本號**。
  - async 完成 = **poll / wait 操作句柄**：泛型 `gui_op_poll(handle)`（非阻塞）涵蓋 run / device / soc（Phase 171 把 per-op `gui_run_poll` / `gui_device_poll` 收斂成單一泛型工具，START reply 直接外露 handle）；`gui_op_wait(handle)` 回 `{status, waited_seconds}`，timeout 不 raise，內部對 handle 發 `operation.await`。

## 演化（被取代的設計，保留脈絡）

- **Phase 92/93 origin tracking**（`_originating_state` → EventBus `current_origin` / `acting_as` / lease `origin`）：靠「分辨某筆變動是不是 agent 自己造成的」讓 stale guard 放行 agent。**已取代**——其正確性依賴「每個 emit 都正確標 origin」，而 origin 標記容易漏（controller 層 Qt slot 內轉發 emit 已實證漏標）。重新框定為「**版本變了沒**」而非「**誰**改了」即根治。隨之全拆 `current_origin` / `acting_as` / lease `origin` / emit `origin=` / change buffer / `change_categories.py`。
- **Phase 120c agent 面收斂**：agent 不再曝露 EventBus event（移除 `gui_events_*`），改為上述「樂觀 + guard 撞牆 / poll-wait 句柄」；diagnostic piggyback 保留。GUI 端 EventBus push 全保留（[[0013]] / [[0014]]）。
- **保留自 Phase 93**：off-main handler（本決策 §3）與 `device.wait_setup` 死鎖修復——`wait_setup` 曾在主線 `threading.Event.wait()` 阻塞 event loop → 等不到 Qt queued signal → 死鎖；改 off-main + `gate.await_outcome`。其後 by-name `wait_setup` 由 per-domain operation handle 取代。

## 替代方案與否決理由

- **bump 綁進 `EventBus.emit`（一點涵蓋）**：emit 與資源被寫不必然同點（async emit 在同步窗外）；定為「資源 owner service 在主線 bump」。
- **per-connection 計數抵銷**（begin +1 / terminal −1）：依賴「一次操作恰 1 begin + 1 terminal」的脆性前提，與另一機制並存邏輯雜。版本表一套機制治兩種窗，更收斂。
- **agent 拿裸版本號自己 diff**：違三層分工，版本號是 mcp 簿記非 agent 關注。
- **兩套並存（版本表 + origin/change-buffer）**：兩套通知會漂移，全面取代。
- **`processEvents` 轉 event loop 解死鎖**：重入反模式，CONTEXT.md 明文 avoid。

## 範圍

`experiment/v2_gui/`（adapter 層）與 `gui/` 並列為預設可改範圍；device bump 點在 `GlobalDeviceManager` 的呼叫處（gui 外），須留意。
