# ADR-0026 — operation abstraction：統一 OperationRunner + scope-as-adapter-input + State write ports

關聯 [[0019]]（operation = token + facets）、[[0025]]（跨線程 channel）、[[0004]]/[[0005]]（service 角色 / port）、[[0017]]（worker plotting）、[[0007]]（device state→State）。

## 脈絡

每個長時 operation（run / analyze / device setup / soc connect）的生命週期是**同一段機制**：

```
ensure_can_start(若有 exclusion facet) → 開 handle（註冊 cancel hook）→ 註冊 exclusion lease
→ 綁 per-op progress factory → submit 給 bg + scope → 終局：domain 副作用 → discard progress
→ settle handle → release lease
```

但這段機制目前在 `run.py`、`staged_analyze.py`、`device.py`、`connection.py` **各抄一份**，每份又把 op 專屬的領域政策（run 的 cancel-partial 判讀、analyze 的 `set_tab_analyzing`、device 的 rollback、writeback teardown/compute）和機制揉在一起。三個結構問題：

1. **機制重複**：四份生命週期骨架，新增 op 要再抄一次、修 bug 要改四處。
2. **runner 語義外洩到 bg**：舊 `BackgroundService._entered` 把「`stop_event → experiment runner stop scope`」寫死。experiment runner 的協作中斷語義，**不該**是泛用 off-main 執行器的知識——它會讓 bg 認得「這是個 v2 runner experiment」。
3. **State 被各 service 直接寫**：`run.py` / `staged_analyze.py` 直接呼 `state.set_tab_running` / `update_tab_result` / `set_tab_analyzing`，service 綁死具體 State，而非契約。

`_StagedAnalyzeService` 看似現成的共用骨架，但 **run 無法繼承它**——它把「`set_tab_analyzing` ＋ 無 exclusion」烤進骨架，run 需要 exclusion 且不 set analyzing。共用要靠**組合**而非繼承。

## 決策

### 1. `OperationRunner`：唯一的 kind-agnostic 生命週期機制（組合，非繼承）

抽出一個 `OperationRunner`，負責**上述整段機制且只有機制**。每個 op 不再各自編排生命週期，而是把一份 **`OperationSpec`** 交給 runner：

```
OperationSpec(
    exclusion: ExclusionRequest | None,   # kind/owner/resource_id；None = analyze（無 exclusion facet）
    owner_id: str,                        # progress 容器 owner（tab_id / device_name）
    wants_progress: bool,                 # True → runner mint pbar factory 並注入 work
    cancel_hook: CancelHook | None,       # 如何取消（handles.create 用；run/device = stop_event.set）
    work: Callable[[ProgressFactory | None], Any],  # off-main thunk；runner 注入 minted pbar factory
    run_in_pool: bool,
    interpret: Callable[[BgOutcome], OperationOutcome],  # bg done/error → 終局語義（run 的 cancel-partial 在此）
    on_begin: Callable[[int], None] | None,             # 樂觀寫 + 標 busy（token 傳入供記錄）
    on_terminal: Callable[[OperationOutcome, Any], None],# 領域副作用（State 寫入、writeback、rollback）
)
```

`work` 只收 runner 唯一擁有、且必須在 token mint 之後才能 mint 的 **progress factory**；`figure_container` 與 `stop_event` 是 op policy 的 **closure 細節**（policy 建 work thunk 時就知道），**不**放進 spec——這讓 runner 真正 figure/stop-agnostic（不再像舊 `_entered` 那樣認得 figure facet）。`OffMainScopes` 隨之退場（三欄位分別化為 closure / 注入參數 / closure）。

- runner 透過 **port** 使用協作者（`ExclusionGate` / `ProgressHub` / `BackgroundExecutor` / handle+channel），**只認契約不認行為**。
- run / analyze / device / soc-connect 退化成**極薄的 policy 物件**：組 `OperationSpec`、填 `interpret` / `on_terminal`、宣告要哪些 facet。這就是「每個 service 封裝一種邏輯」——領域政策留在各 op，機制收斂到 runner 一處。
- autofluxdep RUN 同樣是 `OperationRunner` client：flux-sweep domain loop 保留在 autofluxdep，但 QThread、running/stop flags、progress、cancel 與 terminal outcome 不另寫平行 lifecycle。RUN 使用 app-local operation kind（例如 autofluxdep run kind），透過同一 `ExclusionGate` / `ProgressHub` / handle channel interface 組合 facet；cancel 是協作停止，攜帶 `Stop(reason)` 意圖並保留已完成的 partial results，terminal outcome 可為 `cancelled` 而非 failure rollback。per-node fit/result summary 仍是 autofluxdep domain state/query，不塞進 generic progress。
- **不可化約的領域邏輯**（run 的 cancel 帶 partial result 判讀、writeback compute、device rollback）以 `interpret` / `on_terminal` callback 留在各 op，**不**塞進 runner（避免 god-object）。

### 2. scope-as-adapter-input：`BackgroundService` 退化純執行器

`BackgroundService` 變回**純 off-main 執行器**——`submit(work, *, run_in_pool, on_done, on_error)` 只把 thunk 丟 worker thread / pool、把結果 marshal 回主線程，**移除 `_entered` 的 facet 知識**（連 `scopes` 參數一併拿掉）。scope 的 wiring 改由 **op policy 的 work thunk** 負責（experiment adapter 的 `run(request, schema)` 簽名不動——它是 domain code，不該為 GUI scope 改簽名）：

```
# run policy 建構的 work thunk（runner 注入 pbar_factory；figure/stop 由 closure 捕獲）
def work(pbar_factory):
    with figure_ambient(live_container):       # app 層 helper：routing + liveplot（[[0017]]）
        with progress_ambient(pbar_factory):   # session 層 helper：pbar ContextVar
            with schedule_stop_scope(StopSignal(stop_event)):  # op 專屬：run policy 自己接 experiment stop scope
                return adapter.run(request, schema)  # experiment adapter 簽名不動
```

- 通用 ambient 拆成兩個 helper，尊重層次：**`progress_ambient`（session 層**，純 `progress_bar` ContextVar，無 Qt——run 與 device setup 用）＋ **`figure_ambient`（app 層**，routing + `QtLivePlotBackend`，Qt——run 與 analyze 用）。session 層的 device 只能用 `progress_ambient`，不會跨層拉 Qt。
- **op 專屬**的 `stop_event → schedule_stop_scope(StopSignal(...))` 由 **run policy** 的 work thunk 接——experiment runner 的 cancellation 語義只留在 run policy 一處，不再外洩 bg。`Schedule` 與 executor `MeasurementContext` 都讀同一個 ambient stop scope，避免 GUI cancel 與底層執行路徑分叉。
- 不需要 figure / 不接受 stop 的 op（analyze thunk 只用 `figure_ambient`、device thunk 只用 `progress_ambient`、connect 兩者都不用）**靜默不用**對應 helper 即可：每個 work thunk 對內有定義行為的自由，對 bg 統一成無參 thunk。

### 3. State → 窄 write port

run/analyze 對 State 的寫入收斂成窄契約（比照 [[0005]]/0021 既有 `ports.py` 的 `ContextWritePort` / `WritebackQueryPort`）：

- `TabResultWritePort`：`set_tab_running` / `update_tab_result` / `clear_tab_results`（run policy 消費）。
- `TabAnalyzeWritePort`：`set_tab_analyzing` / `update_tab_analyze` / `update_tab_post_analyze`（analyze policy 消費）。

runner 與 policy 只認 port，不認具體 `State`；`State` 是唯一 implementer（仍守主線程寫入不變式 [[0007]]）。

### 4. gate / progress 維持 port 兄弟（不併入 runner）

`OperationGate`（exclusion policy）與 `ProgressService`（progress 容器）**不**折進 runner——它們與 runner 只共享一個 int token，各自有獨立職責與生命週期。runner 透過 `ExclusionGate` / `ProgressHub` port 使用它們。progress 是高頻 pull、性質與「一次性互動事件」不同，**不**走 [[0025]] 的 channel。

### 5. `ConnectionService` 拆成 SoC ＋ Predictor

`connection.py` 目前把兩種無關職責混在一起：

- `SoCConnectionService`：SoC 連線 op（硬體 facet）。移除自有的 `_ConnectWorker` QThread，改用 `bg.submit` 成為 `OperationRunner` 的 client（無 cancellation point，故 `cancel=None`）。
- `PredictorService`：predictor load / predict（純計算，非硬體 op，不進 runner、不進 channel）。

### 6. `DeviceService` 保留為領域 service

device 連線/斷線/設定的領域邏輯（rollback、`ActiveDeviceOperation` 簿記、snapshot）夠豐富，**保留** `DeviceService`；其生命週期段走 `OperationRunner`。`GlobalDeviceManager` 抽 `DeviceRegistryPort`，讓 DeviceService 依契約而非 singleton。

### 7. `SaveService` 留在抽象外

save 同步、無 handle、不經 `operation.await`（[[0019]] 明示）、無「中途停 save」需求——**不**納入 operation abstraction。未來 save 若改 async/cancellable 再採用。

### 8. agent-facing handle 外露 + 泛型 op wait/poll

mcp 介面層對 agent 揭露 operation 的策略改變（純揭露策略，不動本 ADR §1–§7 的任何機制）：

- **(a) START 顯式回 `handle`**：每個短等 START（run / analyze / post-analyze / device）的 reply 一律帶 `handle`（pending 與 finished 都帶）。`handle` ＝ wire `operation_id`（int），對 agent 是**不透明 token**，唯一用途是餵給泛型 `gui_op_poll` / `gui_op_wait`。wait/poll 路徑改吃 handle 直打 wire `operation.await(operation_id, timeout)` / `operation.progress(operation_id)`——這兩個 wire 入口本就 **op-agnostic**（只吃一個 int id）。`_OP_BY_KEY{semantic-key → id}` 從 wait/poll 主路徑退場，僅留作 `gui_debug_operations` 的 **latest-handle-per-resource 投影**（debug-only）；mcp 端 app-local `MeasureMcpSession` 捕捉 START reply 的 handle 並維護此 debug projection，但不再 strip `operation_id`——改 rename 為 `handle` 保留進 reply。

- **(b) cancel 維持 op-specific（無法泛型化）**：controller 沒有、也不該有 `cancel(operation_id)` wire——三個 cancel 走三條領域路徑：`cancel_run()` 是 keyless singleton（run_svc.stop_event）、`cancel_analyze(tab_id)` 需先 `unmount_interactive_analysis(tab_id)` 做 View teardown 再 settle picker、device cancel 走 device stop_event/rollback（另議，P2-B）。`OperationChannel` 雖有 per-token cancel_hook（[[0025]]），但 interactive 的 **View teardown 不在 hook 內**，故無法用「fire hook by id」統一。因此 cancel 仍按 name/tab_id 定址，START 的 product fold 必須繼續攜帶 resource 身分（tab_id/device name）讓 agent 能 cancel。

- **(c) product fold 歸宿遷移**：泛型 `gui_op_poll` / `gui_op_wait` **只回 status**（外加 progress / feedback / cancelled reason）——單憑 handle 無從得知 op kind，無從 fold figure / summary / snapshot。故 finished product fold 全數移入 **(i) 各 op-specific START 的短等 finished 分支**（既有 `_start_op_with_short_wait` 已是此形）與 **(ii) 既有 typed getter**（`gui_tab_get_analyze_result` / `gui_tab_get_post_analyze_result` / `gui_tab_get_current_figure` / `gui_device_snapshot`）。這是「6 支 per-op wait/poll → 2 支泛型」相對「每 op 各自 wait/poll」唯一真正的**行為遷移**：finished-after-degrade（pending 後才 finished）不再由 wait/poll 自動補 figure，agent 改呼 getter。

wire `operation.await` / `operation.progress` / `operation.cancel` 契約**不動**（仍 op-agnostic 吃 int id）；[[0025]] §適用性表不動（事件集 / channel 不變）——本節僅聲明 wait/poll 仍消費同一 channel，只是定址改由 handle 直達。

## 後果

- 新增一個長時 op = 寫一份 `OperationSpec` policy，機制零重複。
- Experiment runner stop scope 不再外洩執行器層；bg 變成 app 無關的純執行器（更易測）。
- runner / policy 對 State / gate / progress / bg 全部依 port——可注入 fake 單元測試，無需 Qt / 真硬體。
- **concurrency-critical**：runner 是所有 op 共用的生命週期核心，動它必經 sub-agent review；與 [[0025]] channel 落地一併驗證。
- `OffMainScopes` 退場（其 figure/pbar/stop 三欄位分別化為 work thunk 的 closure / runner 注入參數 / closure）；`BackgroundExecutor.submit` 的 `scopes` 參數移除。此型別是 session-core 共用（[[0020]]），故 autofluxdep 等共用 app 的呼叫點一併遷移。
- **分子階段落地**：2a State write ports（純加 Protocol，zero behaviour change）→ 2b bg 退化純執行器＋`progress_ambient`/`figure_ambient` helper（concurrency-adjacent）→ 2c `OperationRunner` 抽取＋run/analyze/post/device-setup 改 client（concurrency-critical）。connect 改 `bg.submit` 留 §5 的 ConnectionService 拆分階段。

## 拒絕的替代方案

- **單一 god-Service 把所有 op 領域邏輯吞進去**：違反單一職責；run/analyze/device 的終局語義本就不同，硬塞成一坨會比現狀更難維護。正解是「機制統一（runner）＋政策分立（per-op spec）」。
- **把 `_StagedAnalyzeService` 用繼承一般化成共用 runner**：它把 analyze 專屬政策（`set_tab_analyzing`、無 exclusion）烤進骨架，run 無法套用；繼承會把某一 op 的政策洩進共用層。改用組合。
- **保留 `bg._entered` 的 scope-entering**：讓泛用執行器認得 experiment runner stop scope（runner 語義），是 §2 要消除的外洩。
