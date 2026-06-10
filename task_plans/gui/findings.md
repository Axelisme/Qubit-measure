# GUI 架構 Review Findings

**Review date:** 2026-05-25  
**範圍：** `lib/zcu_tools/gui/`、`lib/zcu_tools/experiment/v2_gui/` 與對應 GUI 測試

## 已確認背景

- 既定設計要求 `State` 是純被動的 shared/per-tab SSOT，`services/` 擁有邏輯與副作用，`ui/` 僅顯示狀態與派送使用者操作。
- Phase 60 將 device setup 改為 `DeviceService` 擁有的背景 operation，並由 snapshot model 支援 dialog 關閉後繼續執行；這是本輪首先驗證的責任邊界。
- 工作樹在 review 開始時為乾淨狀態，分支為 `gui2`；目前 HEAD 為 `c08d256d`，`AI_NOTE.md` 起初仍標示較舊的 `fe2c2371`，本輪已同步修正。

## 待查證問題

- UI dialog 是否直接觸碰硬體執行、檔案持久化或 operation lifecycle，而非經由 Controller/Service。
- `Controller` 是否仍為薄 façade，或開始承載 workflow/state mutation。
- `State` 是否僅表達狀態與 guard invariant，而未執行 IO 或非同步 operation。
- EventBus、Qt worker ownership 與 progress model 是否造成 View 與 Service 的雙重真相來源。

## Findings

### 已確認文件缺口

- `task_plan.md` 宣告「詳細設計長文：見同目錄 `architecture.md`」，但 `task_plans/gui/architecture.md` 不存在。這使 Phase 60 之後的責任邊界只能由 `AI_NOTE.md` 與實作反推，architecture review 缺少單一可核對的 design reference。

### 初步符合邊界的區域

- `State` 實作只有 context/tab/busy/device-setup identity 的狀態 mutation 與 invariant guard，未執行 IO、Qt worker 或 persistence。
- `DeviceService` 持有 `_DeviceSetupWorker` 及 `DeviceSetupProgressModel`，`DeviceDialog` 所需 snapshot 由 service 對外發布，符合 background operation 由 service ownership 的 Phase 60 方向。
- `RunService`、`AnalyzeService`、`SaveService` 各自建立 immutable request、啟動 runner、回收 busy state並 emit event；目前未看到 view 直接啟動 runner。

### 待深查風險

- `ContextService` 直接 mutation `MetaDict` / `ModuleLibrary` 後不顯式 `dump()`；已由 `meta_tool/AI_NOTE.md` 確認兩者 write API 透過 `SyncFile.auto_sync("write")` 自動保存，並非資料遺失風險。
- `WritebackService` 在 `ctx.ml.register_*()` 已自動寫回後又於 `has_persistence` 時呼叫 `ctx.ml.dump()`；屬不一致且多餘的 persistence policy，但不會漏存。

### Finding A - High: Device 建立與實體連線發生在 View，繞過 operation guard

- `ui/device_dialog.py:_instantiate_device()` 直接 `pyvisa.ResourceManager()` 並呼叫 driver constructor；`DeviceDialog._on_add_clicked()` 是先建立 instance，之後才呼叫 `Controller.register_device()`。
- `device/base.py:BaseDevice.__init__()` 的 constructor 立即呼叫 `rm.open_resource(address)` 並執行 IDN query；`YOKOGS200.__init__()` 亦會再查詢 mode。這不是單純建立 immutable config，而是已經對硬體執行 IO。
- `DeviceService._require_device_mutation_available()` 的 run/setup exclusion 僅在 `register_device()` 被呼叫時才生效。若 run 正在執行，UI 仍可能先連上硬體，再被 service 拒絕登錄，留下未納管 session / 外部副作用。
- 建議方向：以 typed `RegisterDeviceRequest`（type/name/address）交給 `DeviceService`；service 在 guard 通過後建立並註冊 device，並明確負責建立失敗時的 resource cleanup。若連線可能阻塞，沿用 setup 的 worker/outcome pattern。

### Finding B - High: Connection/Prediction Service 名義上存在，實際 execution 仍由 Dialog 執行

- `ConnectionService` docstring 宣稱封裝 SoC connection 與 predictor settings，但公開實作只接受已完成建立的 `soc` / `predictor` 並寫入 `State`。
- `SetupDialog._on_connect_clicked()` 直接呼叫 `make_mock_soc()` / `make_soc_proxy(ip, port)`；真實 remote connection 在 GUI event handler 同步執行，View 負責 domain failure policy，也可能凍結 Qt event loop。
- `PredictorDialog._on_accepted()` 直接呼叫 `FluxoniumPredictor.from_file()`，`_on_predict_clicked()` 又直接執行 `predict_freq()`；View 同時承擔檔案載入、domain computation 與 render。
- 建議方向：將 connect/load/predict 轉為 `ConnectionService` 的 typed commands/queries；真實 SoC 連線若可阻塞則以 service-owned worker 處理，Dialog 只 render snapshot/outcome。

### Finding C - Medium: Context 建立流程由 View 直接讀硬體 global singleton

- `SetupDialog._refresh_device_list()` / `_detect_unit()` / `_on_new_ctx_clicked()` 直接 import `GlobalDeviceManager`；其中 `_detect_unit()` 可能呼叫 `YOKOGS200.get_mode()`，`_on_new_ctx_clicked()` 直接呼叫 `dev.get_value()`，再把讀值結果送給 `Controller.new_context()`。
- `DeviceRefLiveField._refresh_validity()` 及 `DeviceRefWidget._refresh_combo()` 也直接讀 `GlobalDeviceManager`，因此 device registry 的 query source 同時存在於 service、model 與 view。
- 這讓 context/View/model 依賴硬體 registry 及可能具 IO 的 query；`DeviceService` 的存取政策、錯誤轉換與未來 mock/async 策略無法集中套用於這些路徑。
- 建議方向：新增 application-level operation，例如 `Controller.new_context_from_device(name, unit, clone)`，由 service 協作取得 device snapshot 並交由 `ContextService` 建立 context。

### Finding D - Medium: UI operation handler 的 broad catch 破壞 Fast Fail 契約

- `SetupDialog._on_connect_clicked()` 與 `DeviceDialog._on_add_clicked()` 都在執行連線/硬體建立後捕捉 `Exception` 並僅更新 status label；`_MlConfigDialog._on_save()` 亦捕捉所有錯誤轉成 modal validation message。
- 對可預期的 user/transport/validation failure 顯示錯誤是合理的，但 broad catch 也會吞掉 contract violation、程式錯誤或不支援的 domain state，繞過 `install_global_exception_hook()` 所宣告的 fail-fast 行為。
- 建議方向：operation 移入 service 後以 typed expected failure/outcome 回傳給 View；未分類的錯誤保留為 uncaught failure 或由 Controller 統一呈現。

### Finding E - Low: 詳細 architecture 文件連結失效

- `task_plan.md` 指向不存在的 `task_plans/gui/architecture.md`；Phase 60 的 operation ownership 決策目前只能從分散的 phase 記錄與 `AI_NOTE.md` 還原。
- 建議方向：恢復或移除該連結，並在單一架構文件中明確列出允許 UI 承擔的 input parsing / render 責任，以及不得出現在 UI 的 IO / domain execution / persistence operation。

### 補充觀察

- `MainWindow` 的 run/analyze/save/writeback handlers 只讀取表單並呼叫 Controller；目前沒有直接執行 adapter、runner 或 persistence。
- `InspectDialog` 的 library editor 在 UI 內負責 schema lowering 與 domain object validation，但實際 mutation 仍透過 `Controller.set_ml_*()`；相較上述硬體/connection 路徑，這是較低優先的 presentation-model 邊界整理議題。
- `lib/zcu_tools/experiment/v2_gui/adapters/` 未直接依賴 Qt、UI 或 QThread；experiment adapter 反向污染 GUI execution lifecycle 的問題未出現。
- `tests/gui/ui/test_device_dialog.py`、`test_setup_dialog.py`、`test_predictor_dialog.py` 目前直接 mock UI 內的 device/SoC/predictor 建立與運算，等於固定現行穿透行為；沒有對應 `ConnectionService` 測試驗證這些 operation 由 service ownership。

## 建議處理順序

1. 先收斂 `DeviceService` 的 registration API，使 guard 包覆 device construction/resource acquisition，避免既有 exclusion contract 可被 UI 繞過。
2. 再擴充 `ConnectionService` 為真正 application service，承接 SoC connect、predictor load/predict；對可能阻塞的 connect 定義 background operation lifecycle。
3. 將 `SetupDialog` / `DeviceRefLiveField` / `DeviceRefWidget` 的 `GlobalDeviceManager` 直接 query 改為 typed service/controller query 或 injected read model。
4. 以 expected failure 型別取代 View 的 broad catch，並恢復單一 architecture document 以固定允許/禁止的邊界。

## Review 結論

`State`、device setup background lifecycle、run/analyze/save pipeline 與 adapter 方向仍大致整潔；目前架構退化主要集中在近期加入的 setup/device/predictor UI 功能。它們已從「render + dispatch」擴張成 domain object construction、hardware/remote IO、domain computation 與 failure policy owner，應優先在擴充更多 GUI 功能前收斂。

---

## Phase 69 Re-review（2026-05-26）

### 基準與範圍

- 現行 `HEAD`：`a0086379` (`fix(gui): align adapter defaults with single_qubit notebook conventions`)；review 起始時 tracked working tree clean。
- 範圍：`lib/zcu_tools/gui/`、`lib/zcu_tools/experiment/v2_gui/`、`tests/gui/`、`tests/experiment/v2_gui/`。
- 既有 Phase 61 findings 屬歷史紀錄；Phase 62 已宣稱修復其中八項問題，本輪僅報告現行程式碼仍可重現或由後續功能新引入的責任問題。
- 重點增量：`services/session_persistence.py`、`services/startup_persistence.py`、tab restore、memory-only device UX、sweep direct/eval UX 與最新 adapter shared defaults。

### 待查證主題

- persistence service 是否真正 owns serialization / IO，或由 view/controller 重新承擔 workflow。
- startup/device session 與 live device registry 是否存在兩套互相競爭的 state source。
- `Controller` 是否因 restore/persistence/operation orchestration 超出 thin facade。
- LiveModel 與 widget 是否正在承擔 domain lowering、session migration 或跨服務 state mutation。
- adapter shared helpers 是否已形成可擴充契約，或大量 adapter 仍重複 orchestration/mapping。

### 核心掃描候選問題（待精讀後定級）

- `state.py:State.add_tab()` 目前直接執行 `adapter.make_default_cfg(ctx)`，與 architecture 宣告的 passive state container 不一致。
- `controller.py:restore_tabs_from_session()` / `persist_tabs_session()` 直接遍歷及 mutation `State`、呼叫 session codec 並 broad-catch 錯誤；session application workflow 沒有落在 service。
- `services/tab.py:get_tab_save_paths()` 先呼叫 `refresh_tab_save_paths()` 後才回傳，query API 具有 mutation；需再核對 default save path 是否也造成 filesystem side effect。
- `ui/setup_dialog.py` 的 startup context 路徑直接建 `MetaDict()` / `ModuleLibrary()`，疑似重新越過 Phase 62 的 View/Service domain construction 邊界。
- `services/device.py` 雖接管 driver construction，但 `register_device()` 仍是 UI handler 同步呼叫；需確認 VISA constructor 是否具阻塞 IO，判斷是否違反 async-shaped operation 契約。
- `services/connection.py` / `services/run.py` 的 lock 需比對 `architecture.md` 所稱 run 與 SoC connect 排斥是否落實。

### 已確認現行問題（第一輪）

- **Query 觸發 IO / 建目錄：** `services/tab.py:99-110` 的 `get_tab_save_paths()` 先呼叫 `refresh_tab_save_paths()`；該方法呼叫 adapter `make_save_paths()`。基底 `adapter/protocol.py:70-87` 的 `make_default_save_paths()` 會執行 `create_datafolder()` 與 `os.makedirs()`。因此 `MainWindow.refresh_tab_save_paths()` 這類 render refresh 透過 `get_*` API 寫入 state 與 filesystem，違反最小驚訝並讓 preview/restore 產生副作用。
- **State 非被動容器：** `state.py:71-86` 的 `add_tab()` 呼叫 `adapter.make_default_cfg(ctx)`；建立 domain schema 的失敗路徑被藏在資料容器 mutation 內，與 `architecture.md` 宣告 State pure/passive 不一致。
- **Controller 已成 session workflow owner：** `controller.py:179-263` 直接遍歷 state、轉換 schema、建立/restore tab、更新 override/active state、發 event，且以 `except Exception` 跳過損壞 tab 或整次 IO failure。這不只是 façade delegation，也會讓使用者在無回饋下遺失尚未成功還原/持久化的分頁。
- **Startup persistence ownership 破碎：** `ui/setup_dialog.py:291-328` 在 View 建立 `MetaDict` / `ModuleLibrary` 並依序呼叫 context、startup persistence、project setup；`ui/device_dialog.py:425-462` 也在 register/drop/forget 之外額外調 persistence。尤其 `controller.py:504-506` 已於 `forget_device()` 移除 persisted entry，而 `device_dialog.py:455-458` 再移除一次，產生重複寫入。
- **同步 registration/reconnect 仍會凍結 View：** `services/device.py:230-273,334-342` 從 Qt handler 同步進入 driver factory；factory `:106-123` 建 VISA driver，而 `device/base.py:44-70` 會 `open_resource()` 與 `*IDN?` query，`device/yoko.py:39-42` 再 query mode。Service ownership 已正確，但 async-shaped UX 契約仍未涵蓋具阻塞 IO 的 registration/reconnect。
- **Operation lock 實作與設計文件矛盾：** `architecture.md` 宣告 run 排斥 SoC connect；但 `services/connection.py:166-231` 僅 guard concurrent connect，未檢查 `State.is_run_active()` 或 device setup；`services/run.py:39-82` 也不檢查 active connect。run 中可啟動 remote connection 並替換 live context 的 SoC handles。
- **Persistence failure 被靜默降級：** `services/startup_persistence.py:71-143` 將 malformed load 與 write failure 只記 warning 並回傳/繼續；`SessionPersistenceService` 的 exception 又由 `controller.py:179-224` broad-catch。使用者看到已套用/已關窗，無法知道設定或 session 沒有保存。
- **Framework 反向依賴 experiment adapter layer：** `gui/adapter/protocol.py:46-53` 的 `AbsExpAdapter.run()` 直接 import `zcu_tools.experiment.v2_gui.adapters.shared.require_soc_handles`；helper 本身只有 `RunRequest` non-None guard (`experiment/v2_gui/adapters/shared/real_experiment.py:3-11`)。這使通用 `gui` framework 依賴其 consumer package，與 `AI_NOTE.md` 的「反向不成立」契約直接矛盾。
- **Adapter base 強型別在擴充點消失：** `gui/adapter/protocol.py:35-64` 以 generic 宣告 result/analysis/params，卻把 `exp_cls`、`build_exp_cfg()`、預設 `get_analyze_params()` 與 `analyze()` 的型別標成 `Any`。新增 adapter 時 type checker 無法保證 cfg type、experiment 與 analysis contract 對齊。
- **Sweep invariant 由 widget 而非 model 擁有：** `live_model.py:241-334` 的 `SweepLiveField` 只保存 `_expts/_step`；`ui/fields/common.py:468-508` 才依編輯來源計算 canonical `step/expts`。非 UI 建立、restore 或 programmatic set 的 `SweepValue` 可保留互相不一致的欄位，而 lowering 又只信任 `expts`；同步規則無法由 session/adapter 共同重用。
- **Startup context 保留過期 active label：** `services/context.py:128-161` 以 `dataclasses.replace()` 替換 startup context 時未重設 `active_label`。若先 `use_context()` 再套用 startup context，新的 `md/ml/result_dir/database_path` 將與舊 label 組合；`adapter/protocol.py:70-87` 會依該 label 產生 save 路徑。
- **Legacy session eval migration 產生 invalid expression：** `services/session_persistence.py:291-293,337-348` 將 legacy `"=r_f - 10"` 原樣放進 `EvalValue.expr`；`expression.py:26-34` 直接以 `ast.parse(expr, mode="eval")` 解析，前導 `=` 必定失敗。本輪以 `.venv/bin/python` 重現得到 `Invalid expression syntax: '=r_f - 10'`；現有 `test_session_persistence_restores_sweep_eval_edges` 僅斷言保存含 `=` 字串，反而固定錯誤行為。
- **Drop 不釋放 hardware resource：** `services/device.py:275-285` 以 `GlobalDeviceManager.drop_device(name)` 實作 UI 的 “Drop/Disconnect”；但 `device/manager.py:26-32` 只從 dict 刪除 object，沒有呼叫 `BaseDevice.close()`。持久化 UX 支援反覆 Drop/Reconnect，因此 VISA session 可能仍保持開啟或累積。

## Phase 71 Planning Research（2026-05-26）

### Phase 70 提交基準

- Phase 70 correctness foundation 已提交為 `9787c768` (`refactor(gui): establish correctness foundation`)。
- `ContextReadiness`、strict session/startup persistence 與 device close lifecycle 已成為 Phase 71 的前置 contract，不應在 operation refactor 中回退。

### Operation ownership 現況

- `RunService.start_run()` 仍以 `State.is_device_setup_active()` / `State.is_run_active()` 做局部 guard，terminal path 透過 state flag 與 event 清除狀態，尚無可組合的 lease。
- `ConnectionService.start_connect()` 僅以 `_active_worker` 防止 concurrent remote connect；mock connect 直接套用 context，remote success 也會於 callback 中替換 `soc/soccfg`，兩者均未排斥 active run。
- `DeviceService` 仍以 `_require_device_mutation_available()` 拒絕 run/setup 期間的 synchronous mutation；registration/reconnect/drop 會在 caller thread 做 driver IO 或 close。
- `State` 同時保存 `running_tab_id` 與 `active_device_setup_name`，表示 busy policy 已散佈於 passive state 與 services；Phase 71 應建立 centralized `OperationGate` 作為 exclusion authority，而非再增加 state flags。

### Phase 71 規劃約束

- `OperationGate` 必須涵蓋 `RUN`、`SOC_CONNECT` 與 device mutations，worker success/failure/cancel 均 exactly-once release。
- Device registration/reconnect/disconnect/set-value 應使用 service-owned worker，避免 Qt handler 執行 VISA/network IO；既有 setup worker 必須改為取得相同 gate lease。
- 同一 device mutation active 時，hardware read 不可被 View refresh 重新觸發；應以 service-owned last-known snapshot 作 render source。
- `DeviceDialog._on_add_clicked()` 目前在 sync registration 回傳後呼叫 `save_startup_device()`；async 化後必須由 Controller 在 successful terminal outcome 執行 persistence，否則 View 無法判定 transaction 成功。
- `DeviceDialog` 目前只 render 單一 `DeviceSetupSnapshot` 並在 setup active 時 disable 全清單；若要允許不同 device mutation 並行，必須先新增 multi-operation snapshot/progress/cancel UX，而非只放寬 gate rule。
- `SetupDialog` 於 remote connect 發起前保存 IP/port，該資料語意為 connection preference 而非 live connection success，可在 Phase 71 保留。

## DDD/Hexagonal Service 角色重構 Research（2026-05-30，M1–M5）

### 起因與根因

- 用戶反覆指出三處不適：service「什麼都能做」（責任太鬆）、state 私有狀態歸屬不清、依賴互呼卡循環。
- 根因（非表面）：`gui/services/` 14 個 service 按**話題**聚合（「跟 X 有關的都丟 XService」），而非按**角色**聚合 → 單一類揉多責任、依賴複雜。

### 查證（外部權威，逐字引用）

- 來源：Context7 `/sairyss/domain-driven-hexagon`（High reputation，綜合 DDD/Hexagonal/Clean）。
- **App Service**（逐字）：「orchestrate... **Contain no domain-specific business logic**... Uses **ports** to declare dependencies on infrastructural services... **Should not depend on other application services (cyclic dependencies)**.」
- **Aggregate Root**（逐字）：「contains other entities/value objects **and all logic to operate them**... is a **gateway**... references from outside should **only** go to the aggregate root... reference other aggregates **via id**, avoid direct object reference.」
- **Driving/Primary Adapter**（逐字）：「user-facing interfaces... **User can be either a person OR another server**」→ 印證本專案「View + RemoteControlService = 兩個平級 client」就是兩個 driving adapter。
- **Driven/Secondary Adapter**（逐字）：「persistence/brokers/emails/3rd-party... **implementation of ports**... **not supposed to be called directly... only through ports**.」
- 目錄：範例**否定** layered 平鋪（「harder navigation and tight coupling」），推薦 vertical-slice。

### 三大系統性違規（= 用戶三處不適的根，全有逐字依據）

1. **貧血 Aggregate**：`TabState`/`DeviceState`/`CfgEditorSession` 是哑 dataclass，行為全在 service。
2. **App-service 互依**：`tab_view→tab/writeback/context`、`workspace→tab`、`startup→context/device`。
3. **基礎設施未經 port**：service 直接 import 具體 infra（persistence/driver/IOManager）或借 `_EditorCtrl` 整個 Controller。

### 關鍵判斷（grill 沉澱，記錄理由）

- **gui2 平行模塊方案否決**（用戶拍板）：78 re-export 檔 + class identity 脆弱不變式 + M3 原子翻牌 + 雙份維護 = 在重新發明 git 已提供的「平行/可回退」。改**原地改 + 每 milestone 一 commit**保底。
- **service 互依「三問規則」**（ADR-0007）：每條「A 用到 B」的邊只問 A 對 B 的意圖 —— Query（都讀 State，無 edge）/ Command（構造注入，單向）/ Reaction（EventBus，依賴反轉）。撞到成環 = 把 Reaction 誤當 Command。比「分層表」抗需求變化（局部判定 vs 全局結構）。
- **狀態放哪兩軸正交**（ADR-0007）：軸 1「進不進 State」= 除 owner 外誰要讀；軸 2「persist 投不投影」= 重啟有無意義。**不可序列化只影響軸 2**（`ExpContext.soc` 進 State、persist 跳過 = 自洽）。早期把兩軸坍縮成一條是 bug，已修正。
- **過度分類修正**（M1）：Runner/AnalyzeRunner/SaveDataRunner 與 connection 的 SoC **本就已 DI / 已 callable seam 隔離**，非「直接 instantiate infra」——Explore 的關鍵字掃描過度列為違規。Runner 是 Qt-signal QObject bridge（非外部系統），包 Protocol 會洩漏 Qt，**不包**。connection 無改動。
- **GlobalDeviceManager 延後**（M1）：worker-thread hardware-I/O 邊界、測試以 fixture cleanup 控制；包 `DeviceRegistryPort` 需動 9 處 worker，收益邊際。`DriverFactoryPort` 已涵蓋有測試價值的 driver 構造 seam。
- **未越界**（M3）：State mutator + version bump 仍是 State 職責（主線寫入不變式），aggregate 謂詞 query-only。需 narrow Optional 處（writeback）保留直接 `is None`（謂詞非 type-guard）。
- **M5 vertical-slice 不做**：三違規已由 M1–M4 全解；橫切 domain service（guard/operation_gate/run/device）不能乾淨切片，高 churn 零行為收益。

### 文件產出

- `docs/adr/0007-service-dependency-three-questions.md`、`docs/adr/0008-service-roles-ddd-hexagonal.md`、`gui/CONTEXT.md`（角色詞彙）、`gui/AI_NOTE.md`（角色段 + ports 層）。
