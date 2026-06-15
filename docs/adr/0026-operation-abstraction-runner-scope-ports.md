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
2. **runner 語義外洩到 bg**：`BackgroundService._entered` 把「`stop_event → ActiveTask`」寫死。`ActiveTask` 是 `experiment/v2/runner` library 的協作中斷語義，**不該**是泛用 off-main 執行器的知識——它讓 bg 認得「這是個 v2 runner experiment」。
3. **State 被各 service 直接寫**：`run.py` / `staged_analyze.py` 直接呼 `state.set_tab_running` / `update_tab_result` / `set_tab_analyzing`，service 綁死具體 State，而非契約。

`_StagedAnalyzeService` 看似現成的共用骨架，但 **run 無法繼承它**——它把「`set_tab_analyzing` ＋ 無 exclusion」烤進骨架，run 需要 exclusion 且不 set analyzing。共用要靠**組合**而非繼承。

## 決策

### 1. `OperationRunner`：唯一的 kind-agnostic 生命週期機制（組合，非繼承）

抽出一個 `OperationRunner`，負責**上述整段機制且只有機制**。每個 op 不再各自編排生命週期，而是把一份 **`OperationSpec`** 交給 runner：

```
OperationSpec(
    exclusion: ExclusionRequest | None,   # kind/owner/resource_id；None = analyze（無 exclusion facet）
    owner_id: str,                        # progress 容器 owner（tab_id / device_name）
    wants_progress: bool,
    figure_container: FigureContainer | None,
    cancel: CancelHook | None,            # 如何取消（見 §2 channel）
    work: Callable[[OffMainScopes], Any], # off-main thunk（adapter 包裝）
    interpret: Callable[[BgOutcome], OperationOutcome],  # bg done/error → 終局語義（run 的 cancel-partial 在此）
    on_terminal: Callable[[OperationOutcome, Any], None],# 領域副作用（State 寫入、writeback、rollback）
)
```

- runner 透過 **port** 使用協作者（`ExclusionGate` / `ProgressHub` / `BackgroundExecutor` / handle+channel），**只認契約不認行為**。
- run / analyze / device / soc-connect 退化成**極薄的 policy 物件**：組 `OperationSpec`、填 `interpret` / `on_terminal`、宣告要哪些 facet。這就是「每個 service 封裝一種邏輯」——領域政策留在各 op，機制收斂到 runner 一處。
- **不可化約的領域邏輯**（run 的 cancel 帶 partial result 判讀、writeback compute、device rollback）以 `interpret` / `on_terminal` callback 留在各 op，**不**塞進 runner（避免 god-object）。

### 2. scope-as-adapter-input：`BackgroundService` 退化純執行器

`BackgroundService` 變回**純 off-main 執行器**——只把 thunk 丟 worker thread / pool、把結果 marshal 回主線程，**移除 `_entered` 的 facet 知識**。scope 的 wiring 改由 **adapter** 負責：

```
adapter.run(request, schema, scope):              # scope = OffMainScopes
    with scope.enter_gui():                        # 通用 GUI ambient：figure routing+liveplot+pbar（app 提供 helper）
        with ActiveTask(scope.stop_event):        # operation 專屬：run-adapter 自己接 ActiveTask
            return run_experiment(request, schema)
```

- 通用 GUI ambient（figure routing＋liveplot[[0017]]＋pbar ContextVar）由 scope 提供的 helper 進入——adapter 呼用、不重抄 Qt 知識。
- **operation 專屬**的 `stop_event → ActiveTask` 由 **run-adapter** 接——`ActiveTask`（runner 語義）只留在 run-adapter 一處，不再外洩 bg。
- 不需要 figure / 不接受 stop 的 op（analyze 不接 ActiveTask、connect 不畫圖）**靜默不用**對應欄位即可：對外統一成 scope，對內各 adapter 有定義行為的自由。

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

## 後果

- 新增一個長時 op = 寫一份 `OperationSpec` policy，機制零重複。
- `ActiveTask` 不再外洩執行器層；bg 變成 app 無關的純執行器（更易測）。
- runner / policy 對 State / gate / progress / bg 全部依 port——可注入 fake 單元測試，無需 Qt / 真硬體。
- **concurrency-critical**：runner 是所有 op 共用的生命週期核心，動它必經 sub-agent review；與 [[0025]] channel 落地一併驗證。
- `OffMainScopes` docstring 目前那句「entering logic stays in BackgroundService」隨 §2 過時，遷移時一併修。

## 拒絕的替代方案

- **單一 god-Service 把所有 op 領域邏輯吞進去**：違反單一職責；run/analyze/device 的終局語義本就不同，硬塞成一坨會比現狀更難維護。正解是「機制統一（runner）＋政策分立（per-op spec）」。
- **把 `_StagedAnalyzeService` 用繼承一般化成共用 runner**：它把 analyze 專屬政策（`set_tab_analyzing`、無 exclusion）烤進骨架，run 無法套用；繼承會把某一 op 的政策洩進共用層。改用組合。
- **保留 `bg._entered` 的 scope-entering**：讓泛用執行器認得 `ActiveTask`（runner 語義），是 §2 要消除的外洩。
