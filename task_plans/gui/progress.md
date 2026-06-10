# GUI 架構 Review Progress

## 2026-05-25

- 啟動 Phase 61 responsibility-boundary review。
- 讀取 `task_plan.md` 與 `lib/zcu_tools/gui/AI_NOTE.md`，確認目標邊界：View render/dispatch、Controller façade、Service 副作用 ownership、State 被動 SSOT。
- 確認工作樹乾淨且目前分支為 `gui2`；取得 knowledge graph 現行 GUI symbols 與 Phase 60 device setup 相關入口。
- 讀取 `State`、`Controller`、`DeviceService`、`RunService`、`AnalyzeService`、`SaveService`、`ContextService`、`WritebackService`，初步確認 operation service ownership，並標記 persistence 契約待核對。
- 錯誤：`task_plans/gui/architecture.md` 不存在，無法讀取計劃檔所引用的詳細設計文；改以 `AI_NOTE.md` 與現行程式碼為 review 基準。
- 錯誤：第一次寫入 `findings.md` 的 patch 段落定位不符，未套入；已以精確 `## Findings` 段落重新套用。
- 核對 `meta_tool/AI_NOTE.md`：`MetaDict` / `ModuleLibrary.register_*` 自動持久化，排除 ContextService 漏存疑慮；標記 `WritebackService` 額外 `dump()` 為不一致但非資料損失。
- 確認高風險穿透：`DeviceDialog` 在 service guard 前建立/連接硬體；`SetupDialog` 在 UI thread 執行 SoC 連線；`PredictorDialog` 直接載入/運算 predictor。
- 確認中風險穿透：`SetupDialog` 建立 context 前直接透過 `GlobalDeviceManager` 查詢 device value。
- 橫向掃描確認 `DeviceRefLiveField` / `DeviceRefWidget` 亦直接讀 `GlobalDeviceManager`；device query source 尚未集中在 service。
- 檢查 adapter 依賴，未見 `experiment/v2_gui/adapters/` 反向 import Qt/UI/threading；MainWindow 常規 run/analyze/save handlers 亦維持 controller dispatch。
- 檢查測試：dialog tests 直接 mock UI 內的 device/SoC/predictor 建立與運算，尚無 `ConnectionService` ownership 測試。
- 更正 `AI_NOTE.md` 的 current commit、DeviceDialog Add ownership 與 UI exception boundary 敘述；另確認 `app.py` 現行 View attach API 與 note 有落差，將同步修正。
- 將 broad `except Exception` 對 global Fast Fail policy 的衝突列為獨立中嚴重度 finding。
- 整理建議處理順序與結論，Phase 61 review 完成。
- 驗證：本任務未修改 Python product/test code，僅更新 review planning records 與 `AI_NOTE.md`；因此未執行 `pytest` / `basedpyright` / `ruff`。
- 依需求將 review 交付內容整理至 `task_plans/gui/review_gui.md`，包含分級 findings、影響、建議修復順序與測試面觀察。

## 2026-05-26（Phase 69）

- 啟動 adapter 擴充前的 GUI 架構 re-review，使用 `planning-with-files` skill 維護調查紀錄。
- 讀取現行 `task_plan.md`、`architecture.md`、Phase 61 `findings.md`/`progress.md` 與 `lib/zcu_tools/gui/AI_NOTE.md`；Phase 62 已建立明確 View → Controller → Service → State / Adapter 邊界，本輪以其作為評估契約。
- 確認 `HEAD=a0086379`、tracked working tree 起始為 clean；`task_plans/` 與 `AI_NOTE.md` 為 `.gitignore` 內的刻意規劃/知識檔。
- 建立檢查範圍與高風險增量：Phase 64–68 的 tab session/startup persistence、memory-only device、sweep UX，以及新增 real adapter/shared defaults。
- 掃描 `State`、`Controller`、全部 services 的 symbols/import hotspots 與相關測試，辨識候選缺口：State 直接執行 adapter、Controller 編排 session workflow、query/mutation 混用、startup context 的 View domain construction、同步 device registration 以及 operation lock 文件/實作一致性。
- 精讀 `controller.py`、`tab.py`、`device.py`、`connection.py`、`setup_dialog.py`、`device_dialog.py`、persistence services 與 adapter save-path base contract；確認 query 造成建目錄 IO、session workflow 位於 façade、startup/device persistence 分裂、同步 VISA registration/reconnect、run/connect lock 缺口與 silent persistence failure。
- 精讀 LiveModel/sweep widget、adapter base/shared/real implementations、ContextService/IOManager 與測試；確認 framework 反向 import adapter consumer、base `Any` 型別洞、sweep invariant 位於 widget、startup stale `active_label`、legacy `"=expr"` restore invalid 及 device Drop 未 close resource。
- 驗證：以 `PYTHONPATH=lib .venv/bin/python` 還原 legacy sweep expression 並呼叫 evaluator，重現 `restored_expr='=r_f - 10'; evaluation_error=Invalid expression syntax: '=r_f - 10'`。
- 錯誤：新增第二輪 findings 的第一次 `apply_patch` 因段落定位上下文不匹配而未套用；改以現存 `Persistence failure` 行為定位後完成寫入。
- 完成正式交付 `task_plans/gui/review_gui.md`：整理 11 項分級 findings、target architecture、API 收斂方向、Phase A–E 重構順序、必補測試與三項需決策事項。
- 發現 `lib/zcu_tools/gui/AI_NOTE.md` 的 current contract 記載與 Phase 62 後程式碼不符；同步校正 device construction/registry query/persistence ownership 說明，並依規則將該 note 與 root `AI_NOTE.md` metadata 更新為 `2026-05-26 | a0086379`。
- 驗證：`PYTHONPATH=lib .venv/bin/python -m pytest tests/gui tests/experiment/v2_gui -q --tb=short` 通過（411 passed）。
- 驗證：`.venv/bin/ruff check lib/zcu_tools/gui lib/zcu_tools/experiment/v2_gui tests/gui tests/experiment/v2_gui` 通過。
- 驗證缺口：`.venv/bin/python -m pyright lib/zcu_tools/gui lib/zcu_tools/experiment/v2_gui tests/gui tests/experiment/v2_gui` 失敗，原因為 `.venv` 未安裝 `pyright` module；未以其他 interpreter 繞過專案規則。
- Phase 69 review complete；未修改 Python product/test implementation。

## 2026-05-26（Phase 69 補充）

- 依使用者要求擴充 `task_plans/gui/review_gui.md` 的架構建議參考章節，將先前口頭建議具體化為 future-state contract。
- 新增分層執行模型、`TabViewSnapshot`、`OperationGate` API/互斥矩陣、device lifecycle/snapshot、`ContextReadiness`、persistence transaction、pure save/sweep model 與 typed adapter contract skeleton。
- 本補充僅記錄後續重構方案，未變更目前 implementation 或既有 current-contract `AI_NOTE.md`。
- 補充文件後驗證：`PYTHONPATH=lib .venv/bin/python -m pytest tests/gui tests/experiment/v2_gui -q --tb=short` 通過（411 passed）。
- 補充文件後驗證：`.venv/bin/ruff check lib/zcu_tools/gui lib/zcu_tools/experiment/v2_gui tests/gui tests/experiment/v2_gui` 通過。
- 補充文件後驗證缺口：`.venv/bin/python -m pyright lib/zcu_tools/gui lib/zcu_tools/experiment/v2_gui tests/gui tests/experiment/v2_gui` 無法執行，`.venv` 未安裝 `pyright` module；此為既有環境缺口。

## 2026-05-26（Phase 70）

- 使用者核准開始 GUI 正規化實作；建立 `task_plans/gui/refractor_plan.md`，固定完整階段、API 方向與測試 gate。
- 已確認 policies：startup context 僅為 `DRAFT`、real run/save 要求 `ACTIVE`；session/startup cache version bump 後拒絕舊檔；device async 範圍為 mutation operations，同一 device mutation 時同步 read 禁止。
- 第一實作切片限定於 correctness foundation：context readiness/stale label、device close/double persistence、cache legacy rejection/persistence error reporting；operation gate 與 async worker 不在同一切片混入。
- 使用者補充 `pyright` 可直接執行；確認 executable 位於 `/home/axel/.local/share/nvim/mason/bin/pyright`，本輪型別 gate 改用直接 `pyright` command。
- 實作 `ContextReadiness`：startup context 進入 `DRAFT` 並清空 `active_label`；file-backed context 進入 `ACTIVE`；Controller 的 real run/save entry points 拒絕非 active context。
- 修正 draft UX eligibility：`TabInteractionState.has_active_context` 讓 View 在 `DRAFT` 預先停用 Run/Save，同時保留 Analyze/Writeback。
- 實作 device disconnect correctness：`drop_device()` 先 close live driver，再轉為 memory-only entry；`DeviceDialog` 移除 duplicated preference removal；`FakeDevice.close()` 明確為 no-op。
- 實作 strict persistence foundation：session/startup cache payload versions 升為 `2` / `3`、atomic write、typed errors、legacy expression/sweep shape rejection；Controller 顯示 repository failure 與 rejected restored tabs。
- 新增/調整 regression tests，包含 context readiness、draft run/save、device close/single forget transaction、session legacy/version rejection、startup version/write failure 與 restore error visibility。
- 驗證：`PYTHONPATH=lib .venv/bin/python -m pytest tests/gui tests/experiment/v2_gui -q --tb=short` 通過（420 passed）。
- 驗證：`.venv/bin/ruff format <Phase 70 changed Python files>`、`.venv/bin/ruff check --fix <Phase 70 changed Python files>`、full scope `.venv/bin/ruff check ...` 與 `git diff --check` 通過。
- 驗證：`pyright --pythonpath .venv/bin/python lib/zcu_tools/device/fake.py lib/zcu_tools/gui lib/zcu_tools/experiment/v2_gui tests/gui tests/experiment/v2_gui` 為 0 errors；仍顯示 22 項既有 dependency source warnings（`typing_extensions` / `yaml`）。
- Phase 70 correctness foundation 完成；下一階段為 centralized `OperationGate` 與 device mutation async lifecycle。

## 2026-05-26（Phase 71 規劃）

- Phase 70 tracked code/tests 已提交：`9787c768` (`refactor(gui): establish correctness foundation`)；`task_plans/` 與 `AI_NOTE.md` 依專案規則維持 ignored。
- 開始以 plan-only 範圍細化 Phase 71；不修改 product code。
- 初步核對確認：run、SoC connect、device mutation 仍各自擁有局部 operation flags/guards；SoC connect 與 run 尚無統一 exclusion；device registration/reconnect/drop 仍具有同步 IO 邊界。
- 確認 async device migration 的相依：`DeviceDialog` 目前在 sync registration 回傳後保存 remembered device，Phase 71 必須把 successful terminal persistence orchestration 移到 Controller boundary。
- 在 `refractor_plan.md` 新增 Phase 71 detailed plan，拆為 gate core、existing async integration、snapshot/read boundary、blocking mutation migration、View cleanup/delivery 五個 slices。
- 標記 D71-1 待確認：建議本階段先採全域單一 device mutation；若允許不同 device 並行，需同步擴大 progress/snapshot/View contract。
- 使用者選擇 D71-1 方案 A；Phase 71 的 operation contract 固定為全域同時間僅允許一個 device mutation，開始實作。
- 71-A 至 71-E 第一輪 product/test 遷移完成後執行 targeted tests；首次失敗為四項 migrated-test assumption mismatch（Yoko request type、gate conflict message、dialog live-read stub，以及前者中斷 cleanup 造成後續 worker teardown），修正測試資料與 assertion，不回退 async/gate contract。
- Full validation 前的 failure-path audit 發現初次 connect failure 會留下未持久化 memory-only snapshot，以及 pre-worker event error 可能遺留 lease；已收緊 rollback/release contract 並新增回歸測試。測試更新時一次同型 assertion 誤套至正常 disconnect case，依 contract 立即更正。
- 實作 centralized `OperationGate` 並由 `Controller` 注入 `RunService`、`ConnectionService`、`DeviceService`；方案 A 固定為全域僅一個 device mutation，run/SoC/device conflict policy 不再依賴 `State` flags。
- 實作 async device command contract：connect/reconnect/disconnect/set-value/setup 均由 `DeviceService` worker 擁有，terminal callback 更新 cached `DeviceSnapshot` 並 release lease；同裝置 mutation active 時 explicit live read 直接拒絕。
- 遷移 Controller/View responsibility：remembered device persistence 僅在 async terminal success 協調；`DeviceDialog` 只送 typed commands 與 render snapshots；移除 sync façade 與 `State.active_device_setup_name`。
- 驗證：`PYTHONPATH=lib .venv/bin/python -m pytest tests/gui tests/experiment/v2_gui -q --tb=short` 通過（429 passed）。
- 驗證：`.venv/bin/ruff check lib/zcu_tools/device/fake.py lib/zcu_tools/gui lib/zcu_tools/experiment/v2_gui tests/gui tests/experiment/v2_gui` 與 `git diff --check` 通過。
- 驗證：`pyright --pythonpath .venv/bin/python lib/zcu_tools/device/fake.py lib/zcu_tools/gui lib/zcu_tools/experiment/v2_gui tests/gui tests/experiment/v2_gui` 為 0 errors；仍有 22 項既有 dependency source warnings（`typing_extensions` / `yaml`）。
- Phase 71 完成；尚存的 idle explicit device live-read UI blocking 與 save/workflow/query responsibility 於後續 phase 處理。

## 2026-05-26（Phase 72）

- 使用者確認 Phase 71 已自行提交；現行 `HEAD=f02b9527`，product commit `0d215ed7` 為 Phase 72 baseline，其後兩筆提交僅維護 `.gitignore`。
- 核對既有 Phase 3 規劃與程式碼：`Controller` 仍自行 restore/persist session 與 emit tab lifecycle events；`State.add_tab()` 仍呼叫 `adapter.make_default_cfg()`；`SetupDialog` 仍直接建立 `MetaDict` / `ModuleLibrary` 並拼接 startup/project persistence。
- 定義 Phase 72 slices：`WorkspaceService`、passive `State`、`StartupService` typed transaction、Controller connection subscription façade，以及 regression/full validation。
- 文件同步第一次 patch 因 `task_plan.md` 現有段落格式與預期上下文不同而未套用；依實檔 Phase 71/決策分界重新插入，未覆寫歷史紀錄。
- 實作 `WorkspaceService` 與 typed `RestoreReport`：tab create/close/session persist/restore、active tab restore、tab lifecycle events 不再由 `Controller` 編排。
- 實作 passive tab construction boundary：`State.add_tab()` 只保存完整 `TabState`，`TabService` 負責 adapter creation 與 default cfg construction。
- 實作 `StartupService` 與 `StartupProjectRequest` / `StartupConnectionRequest`：service 建立 startup `MetaDict` / `ModuleLibrary`、協調 project/connection/device preference；`SetupDialog` 只 dispatch typed request。
- 以 `Controller.bind_connection_outcome()` 取代 View 取得 concrete `ConnectionService`；移除 legacy startup persistence façade 呼叫與對其不存在 API 的 mock assertions。
- 新增 `test_workspace.py` / `test_startup.py`，調整 Controller/State/SetupDialog/DeviceDialog regression tests。
- 驗證：targeted Phase 72 suite 通過（59 passed）；full `PYTHONPATH=lib .venv/bin/python -m pytest tests/gui tests/experiment/v2_gui -q --tb=short` 通過（436 passed）。
- 驗證：full scope `.venv/bin/ruff check ...`、`git diff --check` 通過；direct `pyright --pythonpath .venv/bin/python ...` 為 0 errors，僅保留 22 個既有 dependency source warnings。
- Phase 72 完成；下一個實作範圍為 pure save-path/query 與 sweep model contract。

## 2026-05-26（Phase 73）

- 核對 `AbsExpAdapter.make_default_save_paths()`：目前透過 `create_datafolder()` 與 `os.makedirs(image_dir)` 在 path suggestion 時建立 filesystem directories。
- 核對 `TabService.get_tab_save_paths()`：query 呼叫 `refresh_tab_save_paths()` 並 mutation `suggested_save_paths`，render 路徑同時具 filesystem 與 state side effects。
- 核對 sweep flow：canonical arithmetic 位於 `SweepWidget._on_ui_changed()`，與 save-path 為獨立 model migration；Phase 73 先只處理 pure save-path/actual-save boundary，避免一次變更兩類行為契約。
- 實作 `utils.datasaver.get_datafolder_path()` pure helper；adapter save-path suggestion 不再呼叫 `mkdir`。
- 移除 `TabState.suggested_save_paths` 與 `TabService.refresh_tab_save_paths()` state mutation；getter 只回傳 override 或 pure suggestion。
- 實作 `SaveService` actual command parent-directory creation，新增 pure query 與 data/image save boundary tests。
- 驗證：focused save-path tests 通過（38 passed）；full GUI/adapters suite 通過（439 passed）；`ruff` 與 `git diff --check` 通過。
- 驗證：納入 `lib/zcu_tools/utils/datasaver.py` 的 direct `pyright` 為 0 errors / 25 warnings；相較原 scope 多出的 3 項 warning 為該既有 utility 的 `typing_extensions` / `requests` source warnings。
- Phase 73 完成；下一步為 sweep canonical arithmetic 的 pure model 遷移。

## 2026-05-26（Phase 74）

- 啟動 sweep pure model 遷移；第一個探索用 `rg` 指令因 shell quote 組合錯誤未執行，改以單純 pattern 重跑後確認現有 tests 無 `expts=0` 合法行為假設。
- 現況確認：`SweepWidget._on_ui_changed()` 擁有 start/stop/expts/step 重算；`SweepLiveField.set_value()` 接受任意 step 而不 canonicalize；lowering 僅以 expts 為準。
- 實作無 Qt dependency 的 `SweepEditor`，集中 numeric bounds、`expts` / `step` intent 與 canonical result 計算；`SweepValue` 對 `expts < 1` Fast Fail。
- 將 `SweepLiveField` 定位為 UI intent/model synchronization boundary；`SweepWidget` 移除 arithmetic，僅 dispatch `update_expts()` / `update_step()` 並 render model result。
- 為 stale input step、step/expts edits、unresolved `EvalValue` 與 invalid expts 補 regression tests；調整 form round-trip 以 canonical step 作為預期。
- 驗證：focused sweep/model/form suite 通過（115 passed）；full GUI/adapters suite 通過（446 passed）；`ruff` 與 `git diff --check` 通過。
- 驗證：納入 `lib/zcu_tools/utils/datasaver.py` 的 direct `pyright` 為 0 errors / 25 warnings；皆為既有 optional dependency source warnings。
- Phase 74 完成；後續可進入 `TabViewSnapshot` query 或 typed adapter contract migration。

## 2026-05-26（Phase 75）

- 核對 `MainWindow` render path：interaction state 由 View 呼叫多個 `Controller.has_*()` / `is_*()` 組合，content refresh 另行查詢 analyze/writeback/save paths/figure。
- 實作 `TabViewService` 與 frozen `TabViewSnapshot`，集中 tab render read model；snapshot save path 沿用 Phase 73 pure query contract。
- 將 analyze parameter instance initialization 從 render getter 移至 run success boundary，確保 `get_tab_snapshot()` 是 pure query。
- 遷移 `MainWindow` 以每個 tab/event 一份 snapshot 渲染；刪除被取代的 scattered Controller/TabService render getters，並補 session initial render 可顯示 saved paths。
- 新增 snapshot pure query、run-terminal preparation 與 MainWindow single-query tests。
- 驗證：focused snapshot/controller/window suite 通過（44 passed）；full GUI/adapters suite 通過（449 passed）；`ruff` 與 `git diff --check` 通過。
- 驗證：extended-scope direct `pyright` 為 0 errors / 25 warnings；皆為既有 optional dependency source warnings。
- Phase 75 完成；Phase 4（pure save/sweep/render snapshot）全部完成，下一步為 typed adapter contract migration。

## 2026-05-26（Phase 76）

- 盤點 typed adapter boundary，確認唯一 framework reverse import 是 `AbsExpAdapter.run()` 動態匯入 experiment shared `require_soc_handles()`；`OneTonePowerDepAdapter` 亦直接使用同一 helper。
- 將 SoC request invariant 移至 `gui.adapter.validation.require_soc_handles()`，base adapter 與 custom run adapter 統一依賴 framework API。
- 刪除 experiment shared 中失去責任的 `real_experiment.py` 與 export，不建立 legacy alias；新增 framework validator regression test。
- 驗證：`rg` gate 確認 `lib/zcu_tools/gui` 無 `zcu_tools.experiment.v2_gui` product import。
- 驗證：focused adapter suite 通過（73 passed）；full GUI/adapters suite 通過（450 passed）；`ruff` 與 `git diff --check` 通過。
- 驗證：extended-scope direct `pyright` 為 0 errors / 25 warnings；皆為既有 optional dependency source warnings。
- Phase 76 完成；four-generic adapter protocol 與 `AdapterCapabilities` 留待獨立 Phase 77 migration。

## 2026-05-30（DDD/Hexagonal Service 角色重構 M1–M5）

**基準：** `ac6be233`（乾淨）。路線：原地改 `gui/`，每 milestone 一 commit 保底，失敗 `git revert`。每階段既有測試零回歸（行為等價證明）+ 新測試驗 ADR 修正 + MCP live。adapter/DTO/ui/remote 零牽連。

### M0（鷹架，已廢棄）
- 曾建 `lib/zcu_tools/gui2/` 78 re-export 檔 + `app_gui2.py` + `run_gui --gui2`。用戶決定「git 即保底」後**全刪**，回 `ac6be233`。不留痕跡。

### M1：port 邊界（解違規 3） — commit `bbf409b8`
- **Status:** complete
- Actions：新 `services/ports.py`（`StartupStorePort`/`SessionStorePort`/`ProjectIOPort`/`DriverFactoryPort`，interface segregation）；`startup`/`workspace`/`context`/`device` 改依賴 port 型別，具體服務 structurally 滿足，`build_app_services` 注入點不變。過度分類修正：Runner/connection 本就 DI（不改）；GlobalDeviceManager 延後。
- Files：`ports.py`(新)、`context.py`/`startup.py`/`workspace.py`/`device.py`、`tests/.../test_ports.py`(新)。
- 驗證：706 passed（702+4）；pyright 0；ruff clean。

### M2：cfg_editor aggregate（解違規 1 範本） — commit `c0b895f1`
- **Status:** complete
- Actions：`_EditorSession`（哑 dataclass）升 `CfgEditorSession` aggregate root（set_field/get/lower/commit/is_headless 行為上身）；`CfgEditorService` 收斂為 Repository；commit 經新 `ModuleLibraryWritePort`（廢 `_EditorCtrl`-as-whole-Controller）；`CfgEditorHost` 組合三 facet 給 composition root。
- Files：`cfg_editor.py`、`ports.py`、`app_services.py`、`tests/.../test_cfg_editor.py`。
- 驗證：708 passed；pyright 0；ruff clean。MCP live：editor open→set_field(scalar+eval)→commit(落 ml `m2_gauss_wf`)→discard(teardown→unknown)。

### M3：device/tab aggregate 謂詞（解違規 1） — commit `275cb438`
- **Status:** complete
- Actions：`DeviceState` 加 `is_memory_only`/`is_connected`/`is_live`；`TabState` 加 `is_busy`/`has_run_result`/`has_analyze_result`/`has_figure`。散落 field-poke 改走謂詞（device/state/tab_view/guard）。未越界：mutator+version bump 仍 State 職責；writeback 保留直接 `is None`（narrow）。
- Files：`state.py`、`device.py`、`tab_view.py`、`guard.py`、`writeback.py`、`tests/gui/test_state.py`。
- 驗證：708 passed；pyright 0；ruff clean。MCP live：device 狀態機 + tab interaction（run 中 is_running、cancel 後 has_run_result）。

### M4：斷 app-service 互依（解違規 2） — commit `ad45dc58`
- **Status:** complete
- Actions：三 offender 按三問 —— `tab_view→context/tabs`（pure query）→ 直接讀 State aggregate（`ExpContext` 加 has_context/is_draft/is_active/has_soc 謂詞 + `TabState.effective_save_paths`），edge 消失；`tab_view→writeback`/`workspace→tab`/`startup→context+device` → 窄 port（`WritebackQueryPort`/`TabLifecyclePort`/`StartupContextPort`/`RememberedDevicePort`）。`DeviceMemoryInfo` 移到 ports.py 斬斷 startup→device import。AST gate 守。
- Files：`ports.py`、`tab_view.py`/`workspace.py`/`startup.py`/`context.py`/`tab.py`/`device.py`/`app_services.py`、`adapter/types.py`、`state.py`、`tests/.../test_tab_view.py`、`test_app_service_decoupling.py`(新)。
- 驗證：711 passed（含 AST gate）；pyright 0；ruff clean。MCP live：startup apply / tab snapshot / session persist+restore 全經新 port。

### M5：目錄 vertical-slice 重排
- **Status:** complete（決定不做）
- 三違規已由 M1–M4 全解；橫切 domain service 不能乾淨切片，高 churn 零行為收益。`gui/services/` 保持平鋪。決定記於 ADR-0008。

### 最終全鏈驗證（MCP live）
- launch → connect_mock → state_check 四旗標 true → device connect/set_value/disconnect → editor aggregate commit → tab_view 謂詞 snapshot → run→analyze→save_both → session persist+restore。調試日誌**零錯誤零異常**（僅 benign exception-hook INFO）。
- 全套：711 passed；pyright 0；ruff clean。tracked 交付 = `bbf409b8`→`ad45dc58` 四 commit。

## Test Results（M1–M5）
| Test | 範圍 | Expected | Actual | Status |
|------|------|----------|--------|--------|
| pytest tests/gui tests/experiment/v2_gui | 全套 + 新測試 | 零回歸 | 711 passed | ✓ |
| test_ports | M1 fake port 注入 | port 可注入 | 4 passed | ✓ |
| test_cfg_editor | M2 aggregate 行為 | session 帶行為 | 30 passed | ✓ |
| test_state | M3 device/tab 謂詞 | 謂詞正確 | passed | ✓ |
| test_app_service_decoupling | M4 AST gate | 無 app-svc 互依 import | passed | ✓ |
| pyright lib/zcu_tools/gui | 全 | 0 errors | 0 errors | ✓ |
| MCP live 全鏈 | 端到端 | 行為等價、零錯誤 | 零錯誤 | ✓ |
