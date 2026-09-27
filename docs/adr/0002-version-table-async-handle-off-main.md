---
status: accepted
---

# 並發感知：資源版本表與 off-main blocking handler（handle 模型沿革）

**狀態：** accepted（資源版本表 guard 與 RPC off-main handler 仍有效；下文的舊 handle 模型已由 [[0066]] 接替）。
**關聯：** Permit／Lease 見 [[0001]]；handle、lease、取消與關閉的現行分界見 [[0066]]；shared transport 見 [[0014]]，external-refresh Reaction 見 [[0067]]。

## 脈絡

GUI 有兩個平級 client（Qt View、remote RPC agent）並發驅動同一批受保護操作。資源版本 guard 避免誤擋 agent 自己剛做的事；需要等待 operation 完成的 RPC handler 不能阻塞主線程（否則卡 event loop → 死鎖）。Operation 的建立與完成由 [[0066]] 說明。

## 決策

版本 guard 與 off-main handler 是兩條**正交**的線；§2 留存已被接替的 handle 模型沿革。

### 1. 資源版本表（只服務 guard）

- **粒度（中粒度）**：`context`、`soc`、`device:<name>`、每 tab 的 `tab:<id>:cfg` / `:result` / `:save_path` / `tab:<id>`（存在性）、`editor:<id>`。tab 資源綁 `tab_id`（uuid4，永不重用）→ 無 key 撞名。
- **版本號 = per-resource 單調遞增整數**（非 wall-clock）。`VersionTable` 是 `State` 的一個區塊。
- **bump 責任歸資源 owner service，且在「資源實際被寫」同點、必在 State owner loop**：worker 不直接提交 `State`；同步操作在 service mutator bump，背景操作在回到 owner 的 terminal policy 寫入時 bump。不靠 origin、不靠 emit/release 順序——只靠「由 owner 在資源被寫處 bump」。
- **bump = 狀態真的變了，不含「值未變的快取同步」**：讀取衍生的快取更新若值未變則不 bump、不 emit（否則純讀 spurious 推進版本號、誤使他人 `expected_versions` 失效）；讀到外部來源真的變了才 bump + emit。
- **guard 用 optional `expected_versions`（類 HTTP If-Match）**：`run.start` / `save.*` / `editor.commit` 帶可選參數，server 在主線 `_dispatch._run()` 單一同步序列內**原子**比對，不符→`PRECONDITION_FAILED` + 回當前版本。依賴消失（tab close 刪 entry）= 視同 stale 擋下。

### 2. Operation handle（歷史設計）

本節原本把 handle 當作 lease 的延伸，經 `_OperationRegistry` 與 `_OperationExclusion` 由 `OperationGate` 一起管理；也曾把 connect 描述成沒有 handle 的同步 wire 操作。這些是當時的模型，**不再是現行 operation 決策**。現行 `OperationRunner` 分別建立 handle 與可選的 exclusion lease；handle 不證明持有 lease。GUI connect 也可經 runner 同時使用兩者，wire `soc.connect` 則走同步路徑。建立、settle、取消及關閉的責任以 [[0066]] 和 [session owner](../../lib/zcu_tools/gui/session/README.md) 為準。

### 3. off-main blocking handler

`MethodSpec.off_main_thread`（預設 False）。`_dispatch` 看到 True 不 marshal 上主線，在 IO worker thread 直接執行。受**嚴格契約**：只能做 thread-safe 等待，**不得碰 main-thread-owned 狀態**（版本表 / change-related / CfgEditor / `_snapshots`）、不需要 stale guard。`operation.await` 即此類。

## 三層分工（脊椎）

- **RPC = mechanism**：持版本表、提供 `resources.versions`、原子比對 `expected_versions`。Operation handle 的 wire 投影由現行 [Remote owner](0060-measure-agent-interface-shared-gui-view.md) 說明。
- **mcp = policy + 簿記 + 翻譯**：持 `last_seen`、知道每操作依賴哪些資源、組 `expected_versions`、收 `PRECONDITION_FAILED`。版本號只在 RPC↔mcp 之間流動。
- **agent（LLM）= 不看資源版本號**：mcp 把 `PRECONDITION_FAILED`（帶 `data.stale` 資源身份）翻成「tab X cfg 過時了」。Operation handle 的 agent 呈現與 poll／wait 工具不由本文規定，見 [Remote owner](0060-measure-agent-interface-shared-gui-view.md) 與 [[0066]]。

## 演化（被取代的設計，保留脈絡）

- **Phase 92/93 origin tracking**（`_originating_state` → EventBus `current_origin` / `acting_as` / lease `origin`）：靠「分辨某筆變動是不是 agent 自己造成的」讓 stale guard 放行 agent。**已取代**——其正確性依賴「每個 emit 都正確標 origin」，而 origin 標記容易漏（controller 層 Qt slot 內轉發 emit 已實證漏標）。重新框定為「**版本變了沒**」而非「**誰**改了」即根治。隨之全拆 `current_origin` / `acting_as` / lease `origin` / emit `origin=` / change buffer / `change_categories.py`。
- **Phase 120c agent 面收斂**：agent 不再曝露 EventBus event（移除 `gui_events_*`），改為「樂觀 + guard 撞牆 / poll-wait 句柄」；diagnostic piggyback 保留。GUI 端 EventBus push 全保留（[[0013]] / [[0014]]）。現行 handle 的呈現見 [Remote owner](0060-measure-agent-interface-shared-gui-view.md)。
- **保留自 Phase 93**：off-main handler（本決策 §3）修復主線等待死鎖；舊 `device.wait_setup` 曾在主線 `threading.Event.wait()` 阻塞 event loop → 等不到 Qt queued signal。當時改成 off-main + `gate.await_outcome`，後來的 handle／await 分界見 [[0066]]。

## 替代方案與否決理由

- **bump 綁進 `EventBus.emit`（一點涵蓋）**：emit 與資源被寫不必然同點（async emit 在同步窗外）；定為「資源 owner service 在主線 bump」。
- **per-connection 計數抵銷**（begin +1 / terminal −1）：依賴「一次操作恰 1 begin + 1 terminal」的脆性前提，與另一機制並存邏輯雜。版本表一套機制治兩種窗，更收斂。
- **agent 拿裸版本號自己 diff**：違三層分工，版本號是 mcp 簿記非 agent 關注。
- **兩套並存（版本表 + origin/change-buffer）**：兩套通知會漂移，全面取代。
- **`processEvents` 轉 event loop 解死鎖**：重入反模式，[Operation ADR](0066-operation-lifecycle.md) 說明 owner-loop 不能阻塞等待完成。

## 範圍

本文保留版本 guard 與 off-main handler 的局部契約；operation 執行範圍與 owner 分界見 [[0066]]。
