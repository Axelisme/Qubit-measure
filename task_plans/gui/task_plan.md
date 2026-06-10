# 任務計畫：v2 GUI 框架與 Adapter 層

**最後更新：** 2026-06-09（**ADR-0019 operation 機制全落地**：Phase 146 BackgroundService 抽取 + Phase 147 OperationGate→Exclusion/Handles 拆 + Phase 148 device execution 進 bg（GUI 24、WIRE 21 不變、live MCP smoke 綠）；interactive picker live-mirror 修復（matplotlib timer，`7e2df7d7`）。前次 2026-06-04：第三個 tool_gui：**dispersive-fit-gui** 完成 —— `gui/app/dispersive/` 移植 dispersive.md 的 g/bare_rf 擬合；單流程面板、全 matplotlib、worker compute/record、read-only RPC（port 8767）；骨架共用領域各寫；dispersive 67 + shared gate 12 + fluxdep 191、pyright 0、ruff clean、live socket smoke 綠；skill 三副本；ADR-0017 worker 畫圖 marshal。前次 Phase 133 步驟 B+ 第二批：plot substrate（mpl backend+host+routing+container）抽到 `gui/plotting/`（b959e9d9）——client/host/container 職責切分、訊號在 host、三分歧和解（show raise/close+shutdown/ContextVar）、R4 fluxdep worker 自進 routing_scope；刪 7 舊檔淨刪 475 行；gui 935+fluxdep 191/pyright 0/ruff clean；live smoke 兩 app 綠。第一批（ffe7f52f）：`gui/remote/*`+`version_table`+wire_version 拆分。event_bus 待定。步驟 A：兩 GUI 搬進 `app/{main,fluxdep}/`（c766ae0e/099bbee6）；Phase 132：liveplot 去 gui 認知化 — LivePlotBackend ABC + 註冊機制 + 統一渲染路徑 + qt backend 搬進 gui，gui2:4378766b；Phase 131：mpl backend setup 簡化 + app.py 歸位 gui/ + wiring 上提 + registry 合併 + mcp run_gui 路徑 fix，gui2:c76e5fef）

## 目標

在 `lib/zcu_tools/gui/` 維持**不含實驗知識**的通用 GUI 框架（adapter 契約為 generic-free 的 `ExpAdapterProtocol`），並在 `lib/zcu_tools/experiment/v2_gui/` 以 `BaseAdapter` + `CfgSchema` 驅動實驗流程（共用實作 + 四 generic 強型別）。Controller 保持為 GUI façade；實驗層不直接碰 Qt 或 threading。

## 重構準則

- **Fast Fail**：錯誤在資料/型別/初始化邊界提早失敗，不延後到深層 runtime
- **責任明確**：模組對外只暴露必要契約；不允許 façade 穿透 service 私有實作
- **最小驚訝**：函式簽名、欄位名稱、狀態更新與實際效果一致；禁止保留「看起來有用但無效」的 API
- **強型別**：以 dataclass / protocol / generic / union 表達狀態與資料流；避免 `Any` 擴散
- **無 Legacy**：不保留已被推翻的過渡 API、不兼容舊邏輯

## 已完成階段濃縮（Phase 1–99）

每行一個 phase（群）：主題 + 關鍵產出/決策 + commit（如有）。詳細 scope/驗證已精簡；現行設計契約見下方「已做決策」表，個別模塊細節見 `AI_NOTE.md` 與 `docs/adr/`。

| Phase | 主題 | 關鍵產出 / 決策 |
| --- | --- | --- |
| 1–32 | 基礎框架 | `AbsExpAdapter`、`CfgSchema`、Spec/Value 型別樹、視窗 / dialog、Controller services 拆分 |
| 33–34 | EventBus 與事件刷新 | `EventBus`、`TabInteractionState`、移除舊 timer 刷新；Controller façade 收斂 |
| 35–36 | LiveModel 反應式表單 | `LiveModelEnv`、`SectionLiveField`、`ControllerProtocol` 取代直接 `md/ml` 綁定 |
| 36.5–37.6 | 風格對齊 / 測試 / UI 收斂 | offscreen headless session；Module/Waveform 摺疊 UI；shared helpers |
| 38–43 | Writeback / Run / Analyze / Request 邊界 | typed `WritebackItem`/`WritebackService`；immutable requests；AST eval scalar；集中 refresh |
| 44–49 | EventBus 強型別 / Hardening / Fast Fail | `Payload` dataclass；adapter contract 凍結；`install_global_exception_hook()` |
| 50–51 | Inspect Modules 管理 | Module/Waveform CRUD、Modify 模式、`LiteralSpec` discriminator |
| 52–55 | QThread lifecycle / Optional ModuleRef | `deleteLater`；三條 outcome 路徑；`ModuleRefSpec.optional` + UI「None」 |
| 56–59 | Spec helpers / AnalyzeParams / 真實 Adapter / Thread teardown | `shared/spec_helpers.py`；adapter-owned dataclass；lookback + onetone adapters；segfault 修復 |
| 60–62 | Device setup 背景化 / GUI review / Findings 修復 | service-owned progress；`RegisterDeviceRequest`；`ConnectionService` 升格 |
| 63–69 | TwoTone Adapters / Session Persist / Sweep / Startup Persist / Re-review | 8 個 twotone adapters；`SessionPersistenceService`；SweepCfg 四欄同步；`StartupPersistenceService` |
| 70 | Correctness Foundation | `ContextReadiness` DRAFT/ACTIVE；strict versioned persistence；`DeviceService.drop_device()` lifecycle |
| 71 | OperationGate + Async Device Mutation | `OperationGate` 全域互斥；`DeviceService` typed request + cached snapshot；async worker lifecycle |
| 72 | Application Workflow 收斂 | `WorkspaceService`；`StartupService`；`State.add_tab()` 只接受完整 `TabState` |
| 73 | Pure Save Path Query | `get_datafolder_path()` pure；`SaveService` command boundary 才 mkdir |
| 74 | Pure Sweep Model | `SweepEditor` pure canonical arithmetic；`SweepLiveField` intent application |
| 75 | Typed Tab View Snapshot | `TabViewService` / `TabViewSnapshot`；View 一次 snapshot 分派 |
| 76 | Framework Request Validation Boundary | `require_soc_handles()` 移回 `gui.adapter.validation`；反向依賴消除 |
| 77 | Typed Adapter Contract + State Cleanup | 四 generic `AbsExpAdapter`；`AdapterCapabilities`；`ExpContext.readiness` 唯一 SSOT |
| 78 | CfgSchema Draft-aware SSOT | `State.cfg_schema` committed SSOT；`start_run` 從 State 讀；keystroke 不發 `TAB_INTERACTION_CHANGED` |
| 79 | 構想筆記分檔（純文件） | `remote_control_draft.md`；`cfg_ssot_draft_aware_plan-1.md` 標 superseded |
| 80 | RemoteControlService 核心 | `services/remote/` 套件：stdlib NDJSON RPC over TCP；marshal 回 main thread 走 `QObject`+queued `Signal`（非 `QTimer.singleShot`）；token auth；loopback 預設免 auth、`--control-allow-external` 強制 token |
| 81a | Event Push + Dialog API + View Query | per-client writer thread + queue；`EVENT_SERIALIZERS`（requery hint，不上複合物件）；dialog 一律 non-modal 經 `MainWindow._open_dialogs` registry；`view.snapshot/screenshot`；mcp_server reader-thread 分流 reply/push |
| 81b | cfg.set_field + context/device query | `path_resolver.resolve_and_set`（scalar/sweep edge/moduleref/deviceref）；`cfg.set_field` mutate live LiveModel 維持 WYSIWYG；md/ml/device wire 只送 scalar/names |
| 87 | 雙 Client 對稱重構 | `GuardService` + 型別化 Permit（Run/Save/Analyze/Writeback）；`ViewQueryService`；`build_app_services()`+frozen `AppServices`；`method_specs.py`(Qt-free)+`ParamSpec` 為 wire 型別 SSOT，MCP tool 從 `METHOD_SPECS` 生成。見 ADR-0001 |
| 88 | MCP Agent 體驗改進 | `run.progress.percent`；移除 `RUN_FINISHED/FAILED` 併入 `RUN_LOCK_CHANGED.outcome`；`GuardError.reason_code`→`ErrorEnvelope.reason`；`tab.list_paths`；connect/startup 改 ParamSpec 生成 |
| 89 | Writeback workflow + Adapter spec 查詢 | adapter 拆 `cfg_spec()`(靜態)+`make_default_value(ctx)`；`adapter.cfg_spec`/`analyze_spec`(未建 tab 可查)；`writeback.preview/apply`(stable key 比對、apply 重算為準)。+4 工具 |
| 90 | RPC headless CfgEditor session | `CfgEditorService` headless LiveModel owner（editor_id 索引）；`editor.open/set_field/get/commit/discard`；斷線回收。見 ADR-0002（後被 0003 supersede） |
| 91 | tab+agent 共享 CfgEditor session | editor 升 widget+agent 共享 cfg draft SSOT；headless/delegated 兩 kind，delegated 共用同一 `SectionLiveField` instance（WYSIWYG）；editor 專屬變更流（`editor_changed`/`editor_closed`）；唯一 teardown 收口 `_remove` 防懸空 hook。gui2:b063bd18/744b1a2c/53dbfe58，見 ADR-0003 |
| 92 | ⚠ **SUPERSEDED by 94/95** | GUI 變動 buffer + stale guard（`change_buffer`/`change_categories`/`_guard_stale`/`_originating_state`）**已從代碼移除**，被資源版本表取代。原 gui2:ff11aee6/a8a08ebf。現行並發 guard 見 Phase 100 + 決策表 |
| 93 | async 來源歸屬（⚠ 部分 SUPERSEDED）+ off-main handler | 來源歸屬機制（`current_origin`/`acting_as`/`OperationLease.origin`）**已移除**，被 `expected_versions` 比對取代。**仍生效**：`device.wait_setup` 的 off-main blocking handler（`MethodSpec.off_main_thread`+`OperationGate.await_outcome`）修 main-thread 死鎖。gui2:1e91a353，見 ADR-0004 |
| 96–98 | Device 狀態下放 State（SSOT）+ persistence 投影 | 刪 `DeviceService._snapshots`；`State.devices` SSOT（含升格的 `remember`+`BaseDeviceInfo`）；`DeviceSnapshot` 變 read-time projection；persistence 改 `StartupService` 訂閱 `DEVICE_CHANGED` 從 State 讀 `remember=True` 整份 `replace_devices`（diff-guard+log-and-swallow）。gui2:feaa92a4/2b6ca032/0f80a83c，見 ADR-0006（修正 0005 的「device 不經 State」前提）|
| 99（原 M1–M5）| DDD/Hexagonal 角色重構 service 層 | 按**角色**非話題聚合，消三大違規：M1 `services/ports.py`（port 邊界）/ M2 `CfgEditorSession` 升 aggregate root + `CfgEditorService` 收斂 Repository / M3 device/tab/ExpContext 加查詢謂詞 / M4 斷 app-service 互依（窄 port 或直讀 State，AST gate `test_app_service_decoupling` 守）/ M5 不做。基準 `ac6be233`→交付 `bbf409b8`/`c0b895f1`/`275cb438`/`ad45dc58`，見 ADR-0007/0008 |

## Phase 100–120：濃縮（已完成，細節見對應 commit / memory / ADR）

每行一 phase（群）：主題 + 關鍵產出/決策 + commit/版本。現行設計契約見下方「已做決策」表，模塊細節見 `AI_NOTE.md` 與 `docs/adr/`。

| Phase | 主題 | 關鍵產出 / 決策 | commit / 版本 |
| --- | --- | --- | --- |
| 100 | version-table 並發盲區補完 | md/ml 語義寫入 bump `context`（單鍵，非拆細鍵）；`update_tab_analyze` bump `tab:<id>:analyze`；`writeback.apply` 入 `_GUARD_DEPS`；`set_context` 改純欄位替換（不誤封）；editor key 對稱 drop；四批含 3-agent 橫向審計 4 真問題全修 | gui2:6eda8d76 |
| 101 | 導航 + 契約註解（降認知負擔） | 純註解：façade 入口寫終點 mutator/bump key/emit、`set_md_attr` 設規範錨點列全三條寫 md/ml 路、`VersionTable.bump` 放 bump↔drop 配對表；plot/device Protocol 補語義契約。零邏輯改動 | gui2:9a7fca5e |
| 102 | device 集合基數 guard | `devices:__set__` 集合基數 version key（`put_device` 新成員 / `remove_device` bump），`run.start` `_GUARD_DEPS` 聲明 → 補 `device:*` glob 對「集合新增」失明。單一 mid-grained key，沿用 opt-in 協議 | gui2:4ff8394d |
| 103 | Phase 101 review 接續 | md/ml bump AST gate（`test_context_bump_gate.py` 把「寫 md/ml 必 bump context」結構化）；phase-named 測試改主題命名；Controller 瘦身評估後**不做**（façade 邊界已乾淨，0 處私有 reach-in） | gui2:96091792/b109bb01 |
| 104 | test↔source 結構對齊 | adapter test 移入 `tests/gui/adapter/`；`test_io_device` 拆成 io_manager + device_manager；tests/ 內 11 處 phase 編號移除。純 test 重組無 source 改動。**教訓**：描述 test 行為前先讀 body 勿臆測 | gui2:dcdc76b6 |
| 105 | 測試覆蓋率補完（75%→~82%） | 三批補 error_handler/wire/analyze/save/writeback/session_persistence/framing/events/adapter/context（+58 測試）。**教訓**：error_handler hook restore 必回 `sys.__excepthook__` 防跨 test 累積致 Qt teardown OOM | 1639 pass |
| 106 | Adapter Protocol 化 + 實驗側 BaseAdapter | `AbsExpAdapter` ABC → `ExpAdapterProtocol`（runtime_checkable、零 generic、10 成員）；experiment 側 `BaseAdapter[...]`（PEP 696 default）持共用實作 + no-analysis raise；刪 `NoAnalysisAdapterMixin`；`cfg_spec` 升 classmethod 不升 classvar（避 import-time eval）；14 adapter 遷移 | gui2:a881b91f（與 107 同 commit）|
| 107 | notebook 模式對齊 helper + RefValue override 持久化 | `proper_relax`/`md_writeback` helper 收斂樣板；**RefValue `is_overridden` 持久化（真 bug）**：MODIFIED 態原只活 runtime widget、persist 丟失 → reload 把改過的庫引用當純 LINKED；不容忍 legacy（直接加欄位） | gui2:a881b91f |
| 108 | Spec/Value fluent 覆寫 + LiteralSpec 鎖定 + 角色 default | ADR-0009：spec 層 `lock_literal`/`readonly`（回新 frozen）vs value 層 `with_field`（in-place 回 self）對稱但機制不同；鎖定走 LiteralSpec（widget 一律不畫）；角色 default 工廠系統（每角色一檔 blank+ref 版）；多輪實作收斂（刪 schema_overrides/readonly、defaults/ 套件重組、12 adapter 遷移） | gui2:a1d1d057 + 多 commit |
| 109 | device RPC 對稱補完 | `connect`/`disconnect` 回 operation handle（透傳 token）；`device.snapshot` 補 `info.to_dict()`（原誤判 not JSON-friendly 而剝掉）；**`set_value` 全鏈拆除**（無 ramp 危險捷徑，設值改走 `setup(updates={value})`）；wait by operation_id。**意外發現觸發 Phase 110**：set_value 是 agent 偷加非設計 | gui2:b94f3586，WIRE 2→3 |
| 110 | device setup schema discovery + GUI 偷懶 review | (A) `device.setup_spec` RPC（從 live pydantic `model_fields` 回欄位 schema，protected 標 unsettable）；(B) 4-agent 掃 gui/ 找「繞抽象/偷懶」共 F1–F11，逐項核實（多過度報告）：批一 ADR-0011 ml 寫入單一化（F1 Writeback/F2 Inspect、Context read/write port 拆分、移除 set_ml raw RPC）、批二 F4/F5/F6 去冗餘防禦（F3 延後 ADR-0013）、批三 ADR-0012 一等「停用態」value（DisabledRefValue 取代 None 哨兵）+ fake 目錄重組、批四 F11 孤兒刪 | d0df2f2b + 4 批 commit，WIRE→5 |
| 111 | Progress 大重構（Qt-free ProgressService） | 抽 `ProgressTransport` port（Qt 成 driven adapter `QtProgressTransport`）+ Qt-free 無鎖 `ProgressService`（owner→live op 映射）；handle=operation_id（復用 lease token）；progress 生死全走 transport（push 建/pop 退役）不訂閱 Gate；`ProgressContainer` 一等概念；View 用 owner_id attach 一次。OperationGate 不動 | gui2:90adbcec，GUI v7 |
| 112 | soc.info RPC | `soc.info` 回 soccfg description + 結構化 DAC/ADC channel/sample rate/freq range + is_mock；mcp 折進 connect 回傳。讓 agent 判斷 cfg ch/freq 是否硬體可行 | gui2:861e0930，WIRE 11/GUI 8/MCP 5 |
| 113 | batch convenience MCP 工具 | 純 mcp-side fan-out（不動 wire）：`gui_editor_set_fields`/`gui_context_set_md_attrs` 迴圈呼既有單筆 RPC；fail-fast 非原子。convenience→mcp 分層 | gui2:d5214b9c，MCP 6→7 |
| 114 | Adapter 行為導覽（AdapterGuide）+ 第三分頁 | `AdapterGuide` frozen dataclass 五欄（behavior/expects_md/expects_ml/typical_writeback/recommended）+ `guide()` classmethod + `adapter.guide` RPC + UI Guide 分頁；13 adapter 全覆寫（具體 key + 建議範圍 + optional）。定位=導覽散文非契約、不加漂移守衛 | gui2:b4b4ca9b/45d6fa2f，WIRE 11→12 |
| 115 | ro_optimize 5 adapter + readout_dpm role + liveplot 修復 | 5 個 readout 校準 adapter（freq/power/length/freq_gain/auto）writeback 寫 md 標量；`make_readout_dpm_default` role；liveplot 修 external-hosted plotter self-refresh bug + auto gridspec figure Qt no-op。length t0 optional analyze 延後 | gui2:69c18922，wire 不變 |
| 116 | 固定 figsize 出圖 + 2 bug | `figure_export.py` 固定 SAVE_FIGSIZE/DPI（出圖尺寸與視窗無關）；disabled optional module persist 崩潰修（`{__kind:disabled}` 對稱編解碼，ADR-0012 缺口）；auto-optimize skopt `n_jobs=1`（-1 反而慢 4x 且吃滿核卡 UI） | gui2:5dbb8599 |
| 117 | save path 單一來源 + 移除 gui_connect_mock | `StartupService.derive_project_paths` 單一 derive 點，消除 `apply_project` 雙重 re-scope（`chip/qub/chip/qub`）；data 檔名 `@label` 後綴；移除 mock-only 捷徑（agent 改走 connect_start+startup_apply+context_use 同 user 流程） | gui2:731f918c，MCP 8→10 |
| 118 | mcp_server 跨平台 + app.shutdown 優雅關閉 | launch python 改 `sys.executable`（跨平台）+ POSIX/Windows detach 分支；`app.shutdown` RPC 走正常 window-close（persist、無 OS kill）；`gui_stop` 先優雅關閉再超時才 kill；skill 從 symlink 改回複製 | gui2:f0111fb1，WIRE 12→13 |
| 119 | 統一 async-task cancel + Qt-free ShutdownCoordinator | ADR-0014：`_OperationRegistry` 補 cancel 動詞（持 worker stop_event）；RunWorker 自判 cancelled 刪 `_cancel_requested_tabs`；Qt-free `ShutdownCoordinator`（輪詢 poll 等停 + timeout 強關）+ `QtShutdownDriver`（QTimer 50ms）；消除 `_shutdown_waiting_for_device_setup` flag + 廣播訂閱 | gui2:5ed92fd2/3df4890a，WIRE 13 不變 |
| 120 | agent UX + token 噪音改善（smoke 後 6 項） | 分三批：120a 內部重構（`schema_to_dict`→`CfgSchema.to_raw_dict(md,ml)` 單一入口、analyze 改名）；120b wire 形狀（set 不回 cfg 回 `{valid,removed,added}`、`list_paths` 加 root/verbosity、cfg_spec ref-only 省 280→16 條）；120c **移除 agent 端所有 event**（樂觀預設+guard 撞牆取代「資源變了」event、blocking wait 取代「異步完成」event）+ guard 回語義化 stale 清單 + analyze 進 gate（agent 同步 mcp 內 await）+ per-domain async poll + wait 回 `{status,waited_seconds}` 不 raise。ADR-0005/0010 補 supersede | 9 commits，WIRE 13→16/GUI 10→13/MCP 11→18 |

## 已做決策（穩定設計契約，仍生效）

| 決策 | 理由 |
| --- | --- |
| `GuiEvent` 用 Enum 而非裸字串 | 降低拼字錯誤與事件漂移 |
| Controller 拆成 façade + services | 降低上帝類風險；View pull model |
| View 刷新事件驅動 | Controller 不耦合 `refresh_*` 介面 |
| `CfgFormWidget` 用 LiveModel | 移除全量遞迴讀表單與 widget 狀態同步 |
| `LiveModelEnv` + `ControllerProtocol` | 避免 `md` / `ml` 重綁定後引用失效 |
| analyze params 使用 adapter-owned dataclass instance | 後處理參數獨立於 `CfgSchema`，免 raw dict / cast |
| `WritebackItem.edit_template` 由 adapter 提供 | View 不持有實驗知識 |
| `ExpContext` `frozen=True` | 變更走 `dataclasses.replace()` |
| `build_exp_cfg()` 委派 `ctx.ml.make_cfg(...)` | adapter 低邏輯密度；驗證集中共用層 |
| 真實 experiment adapter run result 攜 `cfg_snapshot` | save data 不收 live cfg；adapter 自行重建 `last_cfg/last_result` |
| `Controller/State` 才是 live SSOT；snapshot 只在操作邊界建立 | 避免過度凍結 live 能力 |
| 同一時間僅允許一個 run | 硬體安全與 UI 可預測 |
| 全域錯誤處理走 QMessageBox | Fast Fail；防止 async/deep stack error 被吞 |
| `OperationGate` 為 run/SoC/device mutation 唯一 exclusion authority | 集中 hardware safety；移除散落 service guards |
| `TabViewSnapshot` 是 tab render 單一輸入 | View 不組合 query 或在讀取時觸發 hidden state mutation |
| `ExpContext.readiness` 為 context SSOT | 移除 `State.has_startup_context` mirror，避免雙重來源漂移 |
| `AdapterCapabilities` 為 SoC / analyze 需求單一聲明 | `RunService` 在 acquire 前 Fast Fail；UI 依 capability render；adapter 無需重複硬寫 |
| `State.cfg_schema` 為 committed SSOT；`LiveModel` 為 runtime draft | tab auto-commit / dialog local draft 都用同一 LiveModel 機制；run / save / persistence 一律讀 State，不讀 form |
| Cfg keystroke 不發 `TAB_INTERACTION_CHANGED` | 觸發 snapshot 全量重建會放大 N 倍（含 `MD_CHANGED` 連鎖 re-resolve）；UI validity 走獨立 form signal |
| `RemoteControlService` marshal 用 `QObject` + queued `Signal`，不用 `QTimer.singleShot` | `QTimer.singleShot` 從非 main thread 行為不可靠；queued connection 由 Qt 保證 receiver 在 owning thread 跑 |
| Remote `tab.update_cfg`「spec 預設為底 + raw 覆蓋」（缺漏 key 回 spec 預設）；partial 留給 Phase 81 `cfg.set_field` | `raw_to_schema` 以 base.spec 重建 value；partial 需要 path resolver，與本 phase 無關 |
| Remote event push 採 requery hint，複合物件不上 wire | wire schema 不隨 internal type 漂移；agent 接到「變了」自行 re-query |
| 所有 dialog 經 `MainWindow._open_dialogs` 單一 registry、一律 non-modal | UI click 與 remote 共用入口；modal 會凍 event loop 致 marshal deadlock |
| Guard 用型別化 Permit，受保護 service 方法只收 Permit；`GuardService` 單一發放 | 兩 client 共用同一 guard 路徑，pyright 擋漏檢；見 ADR-0001 |
| Permit=靜態前置（純憑證、無釋放）、Lease=動態互斥（有生命週期）；`is_tab_busy` 歸 Lease | 語義分離，避免 permit 變「檢查完就過期」的假保證 |
| `build_app_services()` + frozen `AppServices` 集中 service 建構；Controller 持 bundle | 解耦接線與轉發，Controller 維持守門 façade |
| `ViewQueryService` 擁有 remote-only View 投影；`set_field` 為 View-coupled WYSIWYG | 移出 Controller，消除對 remote `path_resolver` 反向 import |
| `method_specs.py`（Qt-free）+ `ParamSpec` 為 wire 型別唯一來源 | 執行期驗證 + MCP inputSchema 同源，永不漂移；handler 不再 `_require_*` |
| MCP tool 從 `METHOD_SPECS` 生成；override 僅 lifecycle/fan-out/檔案寫入/coercion | 消除三層鏡像；tool 集合由 `test_mcp_generation.py` 閘門守恆（現為 83 個，隨 phase 增減）|
| 並發 guard 採資源版本表（Phase 94/95，取代 Phase 92/93 的 change-buffer/origin）：`State.version`（`VersionTable`）per-resource 單調遞增，owner service 主線 bump；mcp `_GUARD_DEPS` 持依賴策略並附 `expected_versions`；RPC `_guard_versions` 主線原子比對，不符回 `PRECONDITION_FAILED(stale_version)` | 三層分離（RPC=mechanism / mcp=簿記+翻譯 / agent 只收語義）；通知面正交走 EventBus subscribe/poll，不帶版本號；md/ml 寫入 bump `context`、run/analyze result 各有 `tab:<id>:result`/`:analyze` key（Phase 100 補完）|

## Phase 121–130：濃縮（已完成，逐 phase 詳見對應 memory）

| Phase | 內容 | commit / 版本 |
| --- | --- | --- |
| 121 | TabState→Session 正名 + 合併 TabViewSnapshot+PersistedTab→TabSnapshot + `new_tab(from_dict)` 單一入口 + 整 tab widget attach/detach + closeEvent 單擊修復 | e3e26701..6a5c09f7，WIRE 16 不變 |
| 122 | fake/freq 模擬共振真值移出 cfg 進 adapter `__init__`（model_type+params），sweep 與真值脫鉤讓 fit 真盲掃 | 2e1df382/55aa86e7 |
| 123–125 | freq writeback 只留 r_f/rf_w；onetone（含 fake）不開 init_pulse/reset；stop 中斷 run 不自動切 Analysis（決策移到 `_on_bus_run_finished` 讀 outcome） | 67dd3769..365c4ac9 |
| 126 | PersistenceCaretaker（Memento+Caretaker，ADR-0015）：單檔 `gui_state_v1.json`(pydantic memento)、關閉才寫、startup 偏好進 `State.startup_prefs`、codec 進 `session_codec`；刪兩 PersistenceService + session.persist/restore RPC | 67920281，WIRE 16→17/GUI 13→14/MCP 18→19 |
| 127 | 消化 agent 反饋：精簡 RPC 回傳（connect 只折 soc description、run-finished tab 折 {tab_id,interaction}、context.new 回 {label}、save 回 {ok}）；save_both→save_result；刷新陳舊 SKILL | dae3bfea，WIRE 17→18/GUI→15/MCP→20 |
| 128 | context.new 去裸 value/unit 改 bind_device 驅動（白名單 unit + 只讀 device 值、Fast-Fail）+ clone_from label + UI clone 下拉 + 清 wire dead-code | ee7d14b9，WIRE 18→19/GUI→16 |
| (clean) | clean-start flag：`run_gui --clean` / `run_app(clean=)` / `gui_launch(clean=)` 啟動略過還原（`restore_all(load=False)`）；關閉仍 flush | 95af6c48，GUI 16→17/MCP 20→21 |
| 129 | agent 錯誤語意修正 + progress 折進 poll：cancel≠failed（`reason` 映 status）；run 開始即清 result（進行中/失敗時無 result，analyze fail-fast `no_run_result`）；context.new 無 project→`no_project`；progress 折進 `*_poll`（單一 `operation.progress` RPC，移除 run/device progress 工具） | gui2:36fa8b37，WIRE 19→20/GUI 17→18/MCP 21→22 |
| 130 | 消化獨立 agent 壓測 4 項：fake save 真存檔（`persist_data` default True）；cancelled partial result 是設計（只修文檔）；mcp tool error 末附 `reason:<tag>`；SweepValue step 髒值治本（`__init__` 加 `auto_norm` 構造時 canonicalize，SweepEditor opt-out） | gui2:616a4b23，GUI 18→19/MCP 22→23/WIRE 20 不變 |

## Phase 131：mpl backend setup 簡化 + app composition root 歸位（消化 user request）— ✅ 完成（gui2:c76e5fef，版本不變）

**起因**：user 質疑 `mpl_backend_setup.py` 的 import-order 脆弱感與 `os.environ` 旗標用途；逐步釐清後界定根因與責任，並順帶歸位錯置的 `app.py`、合併相鄰 registry、修一個 pre-existing bug。

- **mpl_backend_setup 重寫**：`ZCU_TOOLS_GUI_MPL_BACKEND` 環境變數旗標**砍除**——它非程式庫行為開關（全 codebase 無人讀），只是跨進程 idempotency 旗標，且跨進程語意是壞的（`matplotlib.use()` 不跨進程，子進程繼承旗標反而會漏設 backend，latent bug）。改**模組級 `_configured`** 做進程內 idempotency。保留 `matplotlib.pyplot` 已 import → **Fast-Fail**（backend 選擇只在 pyplot 首次 import 前生效，靜默失效會讓圖跑去開獨立視窗）。模組維持結構性 import-clean（零頂層 matplotlib，`import matplotlib` 在函式內延遲）。
- **責任界定（user 主導）**：「backend 在被使用前先設好」是**進程入口的責任**，不是業務模組的責任——業務模組（如 `utils/fitting/multi_decay.py`）top-level `import matplotlib.pyplot` 完全正當，不該為遷就 backend 設定而被迫延遲 import（曾考慮過此「治本」方案，被 user 否決為挖東牆補西牆）。正解：保證 `zcu_tools.gui` 套件 **import-clean**，腳本入口 `from zcu_tools.gui import configure_gui_matplotlib_backend` → 設定 → 再 import 會用 matplotlib 的東西，順序自然無陷阱。
- **gui 套件 re-export + 測試守衛**：`gui/__init__.py` re-export `configure_gui_matplotlib_backend`；新增 `test_gui_package_import_is_matplotlib_clean`（子進程斷言 `import zcu_tools.gui` 不洩漏 matplotlib），鎖死此不變式。
- **app.py 歸位**：`experiment/v2_gui/app.py` → `gui/app.py`（它是 GUI composition root，非實驗宣告；`experiment/` 只放實驗相關）。刪掉其內部 backend 設定呼叫（信任入口已設）。
- **wiring 上提（消除反向依賴）**：`run_app(registry, role_catalog, ...)` 改收**已填充**的容器；接線（建 Registry/RoleCatalog → `register_all`/`register_all_roles` → 傳入）上提到 entry script `run_gui.py`（composition root）。**gui 框架層不再 import `experiment.v2_gui`**——只有 `run_gui.py` 同時觸碰兩個模組，依賴方向單一。
- **registry 合併**：`role_catalog_registry.py` 併入 `registry.py`（職責相鄰：都是把 v2_gui adapter 層註冊進 gui 框架的契約；`register_all` adapter + `register_all_roles` role 同檔）。
- **bug fix（pre-existing）**：`mcp_server.py` 的 `gui_launch` 子進程啟動 `run_gui.py` 路徑仍指 repo root，但 commit `3f0bc09a` 已把它移到 `script/`——修正為 `repo_root/script/run_gui.py`，恢復 `gui_launch`。此 bug 與本次重構正交，3 個 `test_mcp_launch_connect` 失敗於乾淨 HEAD 即存在。

**驗證**：pyright 0 errors/warnings / ruff clean / `tests/gui`+`tests/experiment/v2_gui` 1051 pass（含修復後的 3 個 MCP launch 測試）。版本號（WIRE/GUI/MCP）不變（無 wire/RPC 語意改動）。

## Phase 132：liveplot 後端去 gui 認知化（grill-with-docs 定稿後實作）— ✅ 完成（gui2:4378766b，版本不變）

**起因**：user 要 liveplot/progress_bar 後端去 gui 認知化。研究發現 progress_bar 已是範本（protocol + `use_pbar_factory`/ContextVar + 零 gui import），只有 liveplot 反向 import gui。經 grill-with-docs 完整定稿。**終極目標**：setup 只 setup_matplotlib + setup_progressbar，liveplot 不需 setup；去除 liveplot 對 gui 的**認知**（不 import gui），非去除 gui→liveplot 依賴。

分三階段實作（每階段 pyright/pytest/ruff 全綠）：

- **階段 1（gui canvas 吸收跨線程）**：`GuiFigureCanvas` 覆寫 `draw_idle`——主線程直接畫、worker fire-and-forget marshalling（守 draw_idle 非阻塞契約），用 `plot_host.is_main_thread()`（新增）判斷。跨線程安全收進 canvas，liveplot 無感。
- **階段 2（統一渲染路徑）**：liveplot `make_plot_frame` 改 `plt.subplots()`（走 mpl_backend attach）；**刪 `create_figure_in_current_container`** + `create_requested` bridge + `_on_create` handler。run liveplot / 裸 plt / analysis 三條路匯流到 `plt.* → GuiFigureManager → attach 進 FigureContainer`。
- **階段 3（去認知化 + 註冊）**：新增 `LivePlotBackend` ABC（`liveplot/backend/base.py`）；jupyter/fallback 改 class（純 matplotlib）；`set_liveplot_backend`（ContextVar CM）+ `set_default_liveplot_backend`（process-wide）；`active_backend()` 選擇順序＝註冊優先 → 名稱兜底（nbagg→Jupyter / 其餘→Fallback），**與 matplotlib 名稱解耦**；刪 `backend/qt.py`、`has_current_container` 維度、孤兒 `set_figure_container`。Qt backend 搬進 `gui/adapters/qt_liveplot_backend.py`（`QtLivePlotBackend`，gui run worker 註冊）。兩處 jupyter 特例（autofluxdep grab_frame、jpa instant_plot）保留直接 import。

**核心決定（grilling）**：base 用 ABC（同 progress_bar 風格、runtime Fast-Fail，非 Protocol）；qt backend 保留（保有與 fallback 分歧的自由度）；跨線程安全押在 `GuiFigureCanvas.draw_idle` 覆寫（user 同意「對使用方透明」）；run liveplot 與 analysis 共用同一渲染路徑（方案2，連 draw 都收斂到 canvas）。

**驗證**：pyright 0 / ruff clean / 全量 1844 pass（+新增 liveplot backend 選擇測試 6）/ **import liveplot 零 gui 牽連（gui modules = NONE）**。版本號不變（無 wire/RPC 改動）。progress_bar 整個未動（已是範本）。

## Phase 133：fluxdep_gui 子計劃完成 + app/ 子模塊化 — ✅ 完成（共用層抽取由 post-Stage-E batch #1–#6 完成，commits 96f52a61…ba5d6703；header 曾過時標 🚧，2026-06-10 對時）

> **fluxdep_gui（`task_plans/tool_gui/`）是本 gui 計劃的子計劃**，不是無關的平行工具。它先以「複製一份 gui/ 機制」獨立演化，現在要回頭把共用層抽出來——這一步**直接改動本檔 scope**（`lib/zcu_tools/gui/`），所以在 gui 計劃裡正式立 Phase。fluxdep 側的領域實作細節仍在子計劃 `task_plans/tool_gui/{task_plan,progress}.md`。

### 子計劃現況（fluxdep_gui，已完成）

`lib/zcu_tools/fluxdep_gui/` — 獨立 Qt GUI 做 fluxonium **flux-dependence 能譜擬合**（移植 `notebook/analysis/fluxdep/`）。自己的 State / service / EventBus / Controller / 互動 widget / remote RPC / MCP server（`fluxdep-gui`）/ skill（`run-fluxdep-gui`）。

- **完整 pipeline（user 在 GUI 操作）**：載入 spectrum hdf5 → 拖線定 half/integer flux → 選點（OneTone 自動 / TwoTone brush mask）→ 累積多譜（可繼承對齊）→ 跨譜篩選 → 匯出 spectrums.hdf5 → 搜資料庫求 (EJ, EC, EL) → 匯出 params.json。
- **狀態（commits gui2 b36c13bf→87bdb923）**：v1（互動 pipeline + remote + MCP + skill）、v2（資料庫搜尋 + matplotlib 視覺化 + params.json + 能量模擬 ~100x/~150x 加速）完成；remote/MCP 轉只讀（87bdb923，agent 只觀測、user 驅動）。199 tests / pyright 0 / ruff clean。
- **採「先複製換獨立演化自由度」**：複製了 gui/ 的通用機制（version table / NDJSON-TCP RPC / worker / mpl backend / progress 後端），暫有兩份，修共通 bug 需手動同步——用戶接受此代價，**換取下一步收斂的設計依據**（複製期間看清哪些「剛好能共用」）。

### 共用層抽取（本 Phase 的 gui-scope 工作，🚧 待展開）

**目標架構（用戶定錨）**：

```
lib/zcu_tools/gui/
  <最基本共用工具>   # mpl 轉發後端 + 註冊機制、progress 後端、version table、RPC/worker 機制——剛好能共用就好，不強迫
  app/
    main/            # = 目前的 measure-gui
    fluxdep/         # = 目前的 fluxdep_gui
    <other apps...>  # 平行的獨立 app
```

**演化順序（重要，用戶定錨）**：

1. ~~當前~~：複製一份 gui/ → 改成 fluxdep 需求（**已完成**，獨立演化自由度已換到）。
2. **本 Phase**：兩 app 已穩定 → 把「剛好能共用」的工具抽到 `lib/zcu_tools/gui/` 頂層，現有 measure-gui 收進 `app/main/`、fluxdep 收進 `app/fluxdep/`。

> 判斷錨點：**先複製換自由度，需求穩定後再收斂共用層**——不在需求未明時過早抽象。現在兩 app 都穩定，收斂時機到。

#### 步驟 A：把兩 app 搬進 `gui/app/{main,fluxdep}/`（✅ 完成，**純搬遷、不抽共用**）

用戶定：**先只搬模塊、不抽共用層**。兩個 GUI 整包物理搬進 `app/`，檔案原樣搬、只改 import：

- `lib/zcu_tools/gui/` → `lib/zcu_tools/gui/app/main/`（measure-gui，commit c766ae0e）
- `lib/zcu_tools/fluxdep_gui/` → `lib/zcu_tools/gui/app/fluxdep/`（commit 099bbee6，對稱手法）

- **手法**：`git mv` 保歷史（85 rename）；新建極簡 `gui/__init__.py`（父，不 re-export、import-clean）+ `gui/app/__init__.py`；舊 hub `gui/__init__.py` 成 `app/main/__init__.py`。
- **import 三檢查點**（嚴禁交錯）：Stage A 全樹 + 外部 546 處 `zcu_tools.gui`→`zcu_tools.gui.app.main` 前綴替換（先求全絕對綠）→ Stage B 同組（=子目錄）絕對收成相對（24 處）→ ruff/format。
- **靜默破壞點（已處理）**：`module://zcu_tools.gui.mpl_backend` backend 字串 + test 斷言；`mcp_server.py` `Path(__file__).parents[4/5]`→`[6/7]`（插兩層、不報 import error 只 runtime 炸）；`.mcp.json` + **`.codex/config.toml`** 兩份 mcp_server.py 路徑；`run_gui.py:42` logger 字串。
- **順帶**：ADR-0008 gate（`test_app_service_decoupling`）原只偵測相對 import，被絕對 import 蒙蔽（analyze/run/save/writeback→guard/writeback 一直存在）；改成**也偵測絕對 import**揭露真相，但精確排除「型別 import」（`*Permit` 憑證 + TYPE_CHECKING 註解，非 runtime service 耦合）。pre-existing `_DummyAdapter` 缺 `guide()` 順手補（Phase 114 舊帳）。
- **驗證（main）**：`tests/gui` + `tests/experiment/v2_gui` **1050 passed**、pyright 0、ruff clean；入口鏈直驗（`parents[6]`=lib、run_gui import 鏈、mcp_server 載入 85 tools）。**live MCP smoke 待 `/mcp reconnect measure-gui`**。
- **fluxdep 搬遷（099bbee6，對稱）**：143 處 `zcu_tools.fluxdep_gui`→`zcu_tools.gui.app.fluxdep`；同類靜默破壞點（`module://...ui.mpl_backend` + 3 test 斷言、`parents[4/5]`→`[6/7]`、`.mcp.json` fluxdep 路徑、`run_fluxdep_gui.py` logger）；temp PID/log 檔名不改（不依賴 package 路徑）。`tests/fluxdep_gui` **199 passed**、pyright 0、ruff clean。**live smoke 待 `/mcp reconnect fluxdep-gui`**。
- **已知非回歸**：兩 app 測試各自隔離全綠（gui 927 / fluxdep 199）；跨包順序（fluxdep 先跑）下 8 個 `tests/gui` plot 測試失敗 = **pre-existing matplotlib backend test pollution**（兩個進程級 `module://` backend 互搶、先設的贏），搬移前 HEAD 同樣存在，與本次無關。
- **未做（仍待後續）**：共用層**完全未抽**（兩份機制仍各自獨立，兩 app 並排在 `app/` 下）。

#### 步驟 B+：共用層抽取（🚧 進行中——第一批已抽）

**共用層分析（逐檔 diff 兩 app，正規化 import 前綴）分三層**：

- **Tier 1（逐字相同）**：`remote/{errors,framing,param_spec}`、wire 機制、`VersionTable`、`utils/error_handler`。
- **Tier 2（抽核心+per-app seam）**：wire 版本常數、`service.py` 傳輸核心（卡 EventBus 決策）、mpl backend（seam=`module://` 字串）、`plot_host`（有真分歧需先和解）、`MethodSpec` 型別。
- **Tier 3（不共用）**：`event_bus`（設計上分歧:enum-key vs payload-type-key）、`dispatch`/`mcp_server`/`method_specs` 表、progress（已有 GUI 樹外 `BaseProgressBar` 共用基底）、`io_manager`（實際領域耦合）、領域 service、widget。
- **浮現的 latent drift（複製後各自漂移）**：EventBus 例外策略（main re-raise / fluxdep 吞）、`plot_host._on_activate`（main raise / fluxdep no-op）——抽 plot_host/event_bus 前須先和解。

**✅ 第一批已抽（commit ffe7f52f）**：5 個最低風險 Tier 1 → `lib/zcu_tools/gui/` 頂層共用層。

- `gui/remote/{errors,framing,param_spec,wire}.py`（兩 app 副本刪、import 改指共用）；`gui/version_table.py`（VersionTable 移出 state.py、state re-export）。
- wire 版本常數（WIRE_VERSION/GUI_VERSION）拆出各 app `services/remote/wire_version.py`（各 app 獨立契約版本）。
- `test_shared_layer.py` 守不變式（共用層 import-clean 零 Qt/mpl/app、VersionTable 單一來源、各 app wire 版本獨立）。
- import 風格：共用層內相對、跨層絕對 `zcu_tools.gui.remote.X`。淨刪 343 行（消重複副本）。gui 934 / fluxdep 199 tests、pyright 0、ruff clean。**✅ live smoke 已驗**（`/mcp reconnect` 後 handshake 正確報 main wire v20、fluxdep wire v1）。

**✅ 第二批已抽（plot substrate，commit b959e9d9）**：mpl backend + host + routing + container → `lib/zcu_tools/gui/plotting/`。

- 設計（用戶定 client/host/container 職責切分）：`backend.py`(client，攔 pyplot 轉發) / `host.py`(**單一主線程 bridge QObject 持訊號**+registry+lifecycle，**不放 Container**) / `container.py`(純 QStackedWidget wrapper 無訊號) / `routing.py`(ContextVar) / `setup.py`(BACKEND_NAME 一條共用路徑 + configure，import-clean)。
- 三分歧和解：show() 統一 raise（Fast-Fail，fluxdep 棄 silent attach-on-show）；close+shutdown guard 超集；routing 統一 ContextVar。通訊單向 worker→host（不預留雙向——互動圖走主線程自建 canvas 不經 backend）。
- **R4（fluxdep 頭號風險）**：ContextVar 在 QThreadPool worker 看不到主線程 set → `_SearchWorker` 自己在 `run()` 進 `routing_scope(diag_container)`（container 建構時捕獲）。subprocess pool-worker routing 測試守此。
- 刪 7 舊檔 + 過時 test，內容搬 `tests/gui/plotting/test_plotting.py`。淨刪 475 行。gui 935 / fluxdep 191、pyright 0、ruff clean。兩 app 入口鏈直驗用新 backend + worker routing。

**✅ 第三批已抽（service.py router scaffolding，commit `96f52a61`）**：`RemoteControlServiceBase`（+`SubscriptionCtx`）→ `gui/remote/control_service.py`。

- base 用 template-method 吃下三 app 共用的 EndpointRouter seam（`route` + events.* handlers）、`_dispatch_on_main`（marshal + off-main 分支 + `_guard`/`_after_success` 兩政策 seam）、EventBus subscribe/serialize/broadcast loop。
- **關鍵：泛型 event key 繞過了「先定 enum vs payload-type 才能抽 service.py」的 blocker** —— base 從不檢視 key（只丟給 `bus.subscribe(key,cb)` + `wire_event_name(key)`），故 main 的 GuiEvent-enum-key 與小 app 的 payload-type-key **共存即可**，service.py 傳輸核心無需先統一 event_bus。
- fluxdep/dispersive `RemoteControlAdapter` 塌成 ~55 行純 domain 綁定、**零 policy 覆寫**；main subclass 覆寫 policy seams（guard / editor lifecycle / diagnostic / `_get_bus` / render_view），保留具名 `_guard_versions`/`_track_editor_lifecycle`/`_reclaim_editors`（既有 unit test 不動）。順刪 dead `_safe_editor_id_for_owner`。
- 淨刪 735 行（三 service.py）。WIRE/GUI/MCP 版本不變（純內部重構）。remote 285 + gui 1033 + fluxdep 199 + dispersive 99 tests、pyright 0、ruff clean。

**event-key 方案評估（消化 user 提問「小 app 要不要也用 gui enum」）**：**不改**。`BaseEventBus` docstring 明載 type-key 是刻意設計——`subscribe(SomePayload, cb)` 無 overload 即自動推得 payload 型別；main 的 enum-key 正是為此型別安全才寫 546 行 `@overload` bus。小 app 改 enum-key = 倒退、且要多維護 enum↔type 對應表，零收益。base 泛型 key 已零成本橋接兩者，無「為統一而統一」壓力。**唯一合理統一是反方向**（見下 #6）。

**待做（後續批次，已定序）**：

- **#3 ProjectInfo + ProjectDialog — ✅ 完成（commit `5eabf5f4`）**：`ProjectInfo`+`default_result_dir`/`default_database_root`+`DEFAULT_CHIP`/`DEFAULT_QUBIT` → `gui/project.py`（Qt-free，stdlib only，兩 app `state.py` 經驗證語義逐字相同）；**開 `gui/widgets/` 共用 widget package**（用戶定方案 B，之後 #5 load_dialog base 進來）→ `gui/widgets/project_dialog.py` 單一 `ProjectDialog`，`db_label=` 參數化（fluxdep `"Database path"` / dispersive `"One-tone dir"`，Browse caption 由它導出）。兩 per-app `ui/project_dialog.py` **刪除**（無 app-specific glue）、全 importer 重接、**無 compat shim**。**發現第四 app `autofluxdep/` 也有自己的 ProjectInfo**（不依賴另兩 app、出 scope，留待未來 consolidation）。pyright 0 / ruff clean / fluxdep 199 + dispersive 99。**sub-agent（opus）實作、parent 復驗+commit**。
- **#1 off-main runner → `gui/background.py` — ✅ 完成（commit `754efa69`）**：`BackgroundRunner`（generic worker + pool + `submit(work,*,on_done,on_error,run_in_pool=True,enter:CM|None)`，**只 qtpy+stdlib**）下放共用。main `BackgroundService` **組合** 它 + 保留 `OffMainScopes`/`_entered`，`submit(work,scopes,...)` 委派 `runner.submit(...,enter=_entered(scopes))`——main call site 全不動、`NO_RESULT` 從共用 import（identity 共享）。6 個手刻 worker 改用 per-panel `BackgroundRunner`：fluxdep search 傳 `enter=ExitStack[routing_scope+use_pbar_factory]`（無 liveplot/ActiveTask）、其餘 `enter=None`。**generation-stamp「最新者勝」取消 + debounce 留在 panel/on_done**（不進 runner，與 main stop_event 範式刻意不合併）。`OperationHandles`/`OperationGate` 維持 main-only。淨減 ~212 行。pyright 0 / ruff clean / gui 1033 + fluxdep 199 + dispersive 99。**sub-agent（opus）實作；parent 復驗時抓到 inline 報 selector dangling 但命令列 pyright 0——確認是 mid-edit stale snapshot，最終態乾淨。**
- **#4 run_qt_app + BaseController — ✅ 完成（commit `7ee123ea`）**：`gui/run_app.py` `run_qt_app(*, controller_factory, window_factory, control=None, adapter_factory=None)`（import-clean，heavy import deferred）+ `gui/controller_base.py` `BaseController[StateT,BusT]`（Qt-free/domain-free generic，state/bus/project_root + `_emit`）。落點 **concept-named top-level**（非 `gui/runtime/` package —— background 已 top-level，不為分組 churn）。fluxdep/dispersive app.py 塌成 factory 接線、Controller 繼承 BaseController（12 façade call site 用 `_emit`）。**main 刻意不路由**：3 個 ordering mismatch（exception hook 要在 QApplication 前、caretaker/restore 要在 show 前、adapter 掛 window+closeEvent 生命週期）+ controller 是 1145 行 superset，留 standalone。砍掉 agent 預留但無 consumer 的 `on_built` hook（YAGNI）。pyright 0 / ruff clean / fluxdep 199 + dispersive 99。sub-agent（opus）實作 + parent 復驗。
- **#5 load_dialog 半邊 + 雜項 — ✅ 完成（commit `b26a129d`）**：`gui/widgets/load_dialog.LoadDataDialog`（共用 file row/transpose/preview/OK-gate，`_build_options` + `result_request` 兩 subclass seam；fluxdep 掛 Type/Inherit、dispersive 不加）；`nearest_existing` → `gui/project.py`；`contrast_limits` → `gui/app/fluxdep/ui/interactive/display.py`（留 gui scope，**不**搬 notebook/）；`gui/error_messages.py` framework（`normalize_raw`/`details_tail`/`friendly_from_rules`/`fit_io_redirect`，domain rule 各 app）。**跳過**（不值 indirection）：`_slider`（fluxdep 單用）、events serializer registry protocol（bodies 主導）。pyright 0 / ruff clean / fluxdep 199 + dispersive 99。sub-agent（opus）+ parent 復驗（inline 報 paths `_nearest_existing` undefined 但 grep 零殘留 + 命令列 pyright 0 = mid-edit stale）。
- **#6 main → `BaseEventBus` + type-key — ✅ 完成（commit `ba5d6703`）**：main 自製 **546→206 行**，退役 `@overload` bus + `_EventPayloadMap`，改建在共用 `BaseEventBus`、payload-type-key（三 app 統一，消 event-key 分歧）。GuiEvent enum 留作 wire-name 詞彙、每 Payload 帶 `EVENT: ClassVar[GuiEvent]`。call-site migration ~18 main + ~15 test 檔（subscribe/unsubscribe(GuiEvent.X,cb)→(XPayload,cb)、emit(GuiEvent.X,p)→emit(p)），**pyright 當完整性網**。`events.py` re-key by payload type、`wire_event_name=payload_type.EVENT.value` → **wire 名 byte-identical（runtime 驗 14==14）、WIRE_VERSION 不變**。**latent drift 和解**：BaseEventBus 吞+log subscriber 例外（不 re-raise）；唯二依賴 re-raise 的 production site 處理——device `_begin_operation` 刪不可達 rollback（emit 不再會 raise、lease 在真 terminal 釋放）、connection `_finish_success` 保留 finally；2 個 regression test 更新到新契約（failing View subscriber 不再 abort device connect / 從 _finish_success 傳播，state/lease 仍正確）。`gui/event_bus.py` docstring 更新（main 不再是 holdout）。pyright 0（main+tests）/ ruff clean / **tests/gui 1034**。sub-agent（opus）跑機械 migration + 在 re-raise blocker 正確停下交我裁決；parent 復驗（多次 inline stale ✘ 用命令列 pyright+grep 證偽）+ 決策（死碼移除 + 契約更新）+ commit。**至此三 app event 系統完全統一在 BaseEventBus。**
- `utils/error_handler`（低風險，可隨時隨手）。

**▶ 下一大塊（子計劃）：session-core extraction** —— autofluxdep 複用 measure-gui 的量測 session core（context+連線+多 device+setup/device dialog），抽到新 `gui/session/`。規劃完成、S1 待開工。完整 plan（鎖定決策/投查 boundary/責任錯置 pre-refactor/S1–S5）見 **`task_plans/gui/session_core_extraction.md`**。#1–#6 即為此鋪路。

#### 第三個 tool_gui：dispersive-fit-gui（✅ 完成）

`lib/zcu_tools/gui/app/dispersive/` — 移植 `notebook_md/analysis/dispersive.md` 的 fluxonium **色散位移 g / bare_rf 擬合**成第二個 tool_gui app（fluxdep 是第一個）。對標 fluxdep 結構：骨架共用（plotting/remote/version_table 三方共用、event_bus/RemoteControlAdapter/gui_pbar/error_messages/project_dialog 逐字複製），領域各寫（preprocess/predict/fit/viz）。設計參 ADR-0017（worker 畫圖 marshal）。

- **設計決策（用戶定）**：UI **單流程面板**（非 fluxdep 多-spectrum list+stacked，因 dispersive 是單 onetone 單一線性流程）；繪圖**全 matplotlib**（plotly→mpl 在 `services/viz.py` 改寫，不動 notebook）；auto-fit/preprocess **走 worker**（compute off-main / record main，守 State 不變式）。
- **領域依賴 fluxdep**：經同一 `params.json` 的不同 section 銜接 —— fluxdep 寫 `fluxdep_fit`、dispersive 讀它並寫 `dispersive={g,bare_rf}`（`update_result` 保留 fluxdep_fit）。
- **R4 不適用**：worker 只回資料（PreprocessResult/AutoFitResult）不在 worker 畫圖 → 不需 routing_scope；即時 tune 圖主執行緒同步畫（LRU-cached predict + set_data）。
- **Remote read-only**（fluxdep 模式）：5 純查詢 RPC + 3 lifecycle MCP tool（launch/connect/disconnect，無 stop 曝露），control port **8767**（避開 fluxdep 8766）。`.mcp.json` 加 dispersive-gui entry；`.codex/config.toml` 不動（同 fluxdep 無 entry）。
- **抓到並修的 robustness bug**：preprocess `gaussian_filter1d(σ=n_freq//30)` 小頻網格 σ=0 → scipy ZeroDivisionError；`_smooth_sigma` floor 至 1。
- **驗證**：dispersive **67 passed**（隔離）、shared-layer gate 12（含 dispersive mcp_server 無相對 import）、fluxdep 191 不破、pyright 0、ruff clean。入口鏈直驗選共用 backend + 建窗；live socket smoke 五 read RPC + wire handshake 全綠。skill `run-dispersive-gui` 三副本（cp 同步）。

**對 measure-gui 的既有外溢（子計劃期間）**：能譜加速那批改了 `lib/zcu_tools/simulate/`（`calculate_energy_vs_flux` 快路徑 + BLAS 釘執行緒）與 `script/generate_fluxonium_sample.py`——非 gui scope，但 measure 側若用到同函式會一起受惠。其餘子計劃改動全在 `lib/zcu_tools/fluxdep_gui/` 內。

## Phase 134：onetone/twotone run 收尾兩 bug 修復（消化 user report）— ✅ 完成（版本不變）

**起因**：user 報 onetone/flux_dep（與同類二維 adapter）run 結束時報「analysis 未定義」，且 rounds（第二個）pbar 每次 update 短暫消失再出現。逐一排查根因後治本。

- **Bug 1（analysis 未定義，同一契約 gap 三個出口）**：4 個宣告 `supports_analysis=False` 的 adapter（onetone/twotone × flux_dep/power_dep）run 完成後**故意沒有** analyze params，但三處對「run 完成 ⟹ 有 analyze params」做錯誤假設（[[feedback_dual_end_user_agent]]：同一 gap 兩端/多端都中）：
  1. **寫入端** `Controller._on_run_finished`（`controller.py`）無條件 `initialize_tab_analyze_params` → 打到 base adapter Fast-Fail `get_analyze_params`（`NotImplementedError`）。**修**：加 `if tab.adapter.capabilities.supports_analysis:` guard（決策點判斷，非在 service 內 silent no-op，維持 `initialize_tab_analyze_params` 對 `run_result is None` 的 Fast-Fail 乾淨）。
  2. **渲染端** `MainWindow.refresh_tab_analyze_form`（`ui/main_window.py`）：`has_run_result` 但 `analyze_params is None` → `raise RuntimeError("Run result has no initialized analyze parameters")`。**修**：`supports_analysis=False` 時提早 return（不弱化「支援 analyze 卻沒 params」這個真錯誤的 Fast-Fail）。
  3. **布局/導航端**（用戶回報「無法存檔」的根因，pre-existing bug 自 commit `ff5d8be9` 引入 AdapterCapabilities）：`update_interaction_state` 用 `setTabVisible(1, False)` 把整個第二分頁隱藏，但**Save 區塊（Save Data/Image/Result）與 Analyze/Writeback 同住該分頁** → 整頁一隱藏連存檔入口都沒了。`_on_bus_run_finished` 又無條件切該（已隱藏）分頁。**修（用戶定錨「只隱藏 analysis 區塊非整頁」+「run 完照常切過去」）**：(a) 不再 `setTabVisible`，改 `_analyze_section`/`analyze_btn`/`writeback_section` 各自 `setVisible(has_analysis)`，Save 區塊永遠可見；(b) 分頁標題動態化 `setTabText(1, "Analysis" if has_analysis else "Save")`（non-analysis 時頁內只剩 Save）；(c) `_on_bus_run_finished` 撤回 supports_analysis guard，run 完照常 `setCurrentIndex(1)`（non-analysis 落在 Save 頁，正合二維掃描存檔需求）。
  其餘 analyze 路徑（attach `populate_values`、analyze_form populate、dispatch RPC `_h_tab_get_analyze_params`/`_h_analyze_start`、按鈕 enable 條件）經掃確認已有 guard，安全。base docstring 本就聲明「`supports_analysis=False` 永不該被路由到 `get_analyze_params`」。**教訓**：[[feedback_capability_gap_all_exits.md]] 第一次只修報錯點（寫入/渲染/導航），漏了「Save 與 analysis 共處一分頁」這個**布局耦合**根因——隱藏 capability 不該一刀切整個容器，要分離無關職責的子區塊。
- **Bug 2（rounds pbar 閃爍，治本）**：根因兩層 ——(a) `Task.set_pbar_n`（外層每掃一點呼叫，`experiment/v2/runner/task.py`）設 `avg_pbar.total`，而 `ProgressBar.total` setter（`pbar_host.py`）**發 `CREATE`** 而非 `UPDATE`；(b) GUI 端 `ProgressContainer.apply`（`services/progress.py`）收 CREATE **整盤替換 model**（連 start_time 重置），`ProgressStack.render_models`（`ui/progress_stack.py`）每次 `reset_all()` 把 widget removeWidget+hide 再 push → 視覺上消失再出現。**修（治本三處）**：`total` setter 改 `_publish(force=True)`（發 UPDATE 帶新 total，不重建）；`ProgressEvent.total` 契約改「CREATE+UPDATE 都帶」；`ProgressContainer.apply` 的 UPDATE 分支加 `model.set_total`（+`ProgressBarModel.set_total`）增量更新不換 model；`render_models` 改「bar 數量不變則原地 setMaximum/setValue/setFormat、數量變才 reset」（抽 `_apply_model`）。真實路徑下 worker `_emit` 一律帶當前 total，故 UPDATE 帶 total 是既有不變式。
- **回歸測試**：Bug 1 ——`test_run_finished_skips_analyze_init_for_non_analysis_adapter`（寫入端不炸）、`test_refresh_analyze_form_skips_non_analysis_adapter_without_raising`（渲染端不 raise）、`test_non_analysis_adapter_run_auto_switches_to_second_tab`（run 完切第二分頁）、`test_non_analysis_adapter_hides_analysis_widgets_but_keeps_save`（**核心**：analyze section/btn 隱藏、Save 可見可用、分頁標題=Save）、`test_analysis_adapter_shows_analysis_widgets_and_labels_tab`（對照組）；Bug 2 ——`test_total_change_updates_widget_in_place_no_flicker`（total 連改時同一 QProgressBar 實例原地更新、不被 pop 回 pool）。
- **驗證**：`tests/gui` **942 passed**、pyright 0、ruff clean。版本號（WIRE/GUI/MCP）不變（純 GUI 內部行為修復，無 wire/RPC 改動）。改動全在 `gui/app/main/`（controller / ui/main_window / pbar_host / services{progress,ports} / ui/progress_stack）；`experiment/v2/runner/task.py` 觸發源僅讀未改（real fix 在 pbar_host total setter）。

## Phase 135：默認 result_dir/database_path 鍌定 repo root（消化 user report）— ✅ 完成（版本不變）

**起因**：user 報用 `.bat` 啟動時默認 result_dir/database_dir 以「script 起始路徑」算（`.bat` 的 `cd /d "%~dp0"` 把 cwd 設成 `script/`），希望固定為專案根目錄。

- **根因**：`derive_project_paths(chip, qub, root)` 是 pure 函式（root 注入、本身正確），但**三個調用點都傳 `os.getcwd()`**（`setup_dialog._prefill_from_persistence`/`_on_names_changed`、dispatch `_h_startup_apply`）→ `.bat` 下 cwd=`script/` → 默認落在 `script/result/...`。
- **修法（用戶定錨「入口腳本注入 repo root」，符合 [[project_gui_mpl_backend_import_clean]] base path 是進程入口責任）**：入口 `run_measure_gui.py` 用 `Path(__file__).parent.parent`（=repo root，與 `LOG_FILE` 同源）算 `project_root` → `run_app(project_root=)` → `_build_window` → `Controller(project_root=)`，存 `self._project_root`（None 回退 `os.getcwd()`，測試/`python -m` 從 repo root 跑行為不變）+ 暴露 `Controller.get_project_root()`。三調用點 `os.getcwd()` → `self._ctrl.get_project_root()` / `adapter.ctrl.get_project_root()`。刪 setup_dialog/dispatch 的 unused `import os`。
- **不在 service 內求 repo root**（駁回 agent 建議的 `Path(__file__).parents[N]`）：層數硬編碼脆弱（Phase 133 搬家就因 `parents[N]` 壞過）、且違反「base path 是入口責任非業務模組責任」。
- **回歸測試**：`test_get_project_root_returns_injected_root`（注入值回傳）、`test_get_project_root_falls_back_to_cwd_when_not_injected`（無注入回退 cwd）、改寫 `test_startup_apply_optional_dirs_default_to_project_root`（端到端：注入明確 root，斷言 startup.apply 默認路徑鍌在注入 root 非 cwd）。`ControllerFixture`/`Fixture` 加可選 `project_root`。
- **驗證（measure-gui）**：`tests/gui` **944 passed**、pyright 0、ruff clean。版本號不變（無 wire/RPC 語意改動，`project_root` 是進程內注入）。改動全在 `gui/app/main/`（controller/app/ui/setup_dialog/services/remote/dispatch）+ 入口 `script/run_measure_gui.py`。

**fluxdep + dispersive 對稱修復（user 要求順便修，同模式不同形狀）**：兩 app 的 `default_result_dir`/`default_database_root` 回**相對路徑字符串**（非 measure 的立即絕對化），最終在 `ui/paths.py:_nearest_existing` 用 `os.path.abspath` 相對 cwd 解析 → `.bat` 下相對 `script/`。

- **修法（user 定錨「初始化傳入 root_dir，用它生成 default_path」）**：`default_result_dir(chip, qub, root="")` 加 `root` 參數（非空時 `join(root, ...)` 絕對化，空時維持相對 = 向後兼容、測試不破）；`ProjectInfo` 加 `root_dir: str = ""` 欄位，`__post_init__` 用它生成絕對默認；`Controller(project_root=)` + `get_project_root()`；`run_app(project_root=)`；入口 `run_{fluxdep,dispersive}_gui.py` 算 `Path(__file__).parent.parent` 傳 `ProjectInfo(root_dir=project_root, ...)` + `run_app(project_root=)`；`ProjectDialog` 從 `project.root_dir` 派生（三處 default_* 調用 + `result_project()` 回傳帶 root_dir，避免絕對路徑被誤判 overridden）。dispersive `database_path` 是 `Database/<chip>/<qub>`（fluxdep 是 `result/<chip>/<qub>`）——差異不影響注入邏輯。
- **回歸測試**：兩 app 各 `test_project_info_root_dir_anchors_derived_defaults`（注入 root 絕對化）+ `test_project_info_empty_root_dir_keeps_relative_default`（回退相對）+ `Controller` 的 `get_project_root` 注入/回退兩個；fluxdep 另加 `test_auto_derivation_anchors_at_project_root_dir`（dialog 改名表單用 root 派生 + result_project 帶 root_dir）。`tests/fluxdep_gui`+`tests/dispersive_gui` **295 passed**、pyright 0、ruff clean。
- **bundled 搜索資料庫鍌定（user 要求一併修，僅 fluxdep）**：fluxdep `ui/paths.py` 的 `Database/simulation`（precomputed search db，repo 自帶共享資源、project-independent）原是 cwd-relative `_SIM_DB_DIR` 常量 → `.bat` 下找不到。**修**：拆成相對片段 `_SIM_DB_REL` + `_sim_db_dir(root="")`（root 非空 join、空維持相對）；`database_dir(project, root="")` / `default_database_file(project, root="")` 加 root 參數；兩個調用點（analyze_panel `_load_from_state` / `_on_browse_db`）傳 `self._ctrl.get_project_root()`（**復用既有注入鏈，不引入第二種 repo-root 求法**，避開 `parents[N]` 脆弱性）。回歸測試 `test_database_dir_anchors_bundled_at_injected_root` + `test_default_database_file_anchors_bundled_at_injected_root`。**dispersive 不涉及**（無 bundled search db；其 predict 是記憶體模擬不讀磁碟、project-derived 路徑已隨 ProjectInfo 絕對化）。`tests/fluxdep_gui` **198 passed**、pyright 0、ruff clean。

## Phase 136：value 樹永遠完整 + `None` 統一表「空」（消化 lookback persist bug）— ✅ 完成（版本不變，ADR-0021 supersede 0012）

**起因**：user 報 loopback cfg `Reset:None`/`Init Pulse:None` 關閉重啟變 `Reset:NoneReset`/`Init Pulse:Pulse`——停用態 persist 後遺失。經多輪 grill-with-docs 定錨，從表面 bug 挖到職責層級重構。

- **根因（職責層級）**：「停用 optional ref」概念有多種長相散多層漂移——`ModuleRefLiveField.is_enabled` 旁路 flag（`get_value` 隱形）、`SectionLiveField` reach-in 子層 flag 而**省略 key**（責任倒置）、`DisabledRefValue` marker（codec/init）。捕獲端用「省略 key」、還原端 `make_default_value` 對缺 optional ref key 回 enabled `allowed[0]` → 停用→重啟→變第一個選項。同源 bug：scalar 未填用 `DirectValue(value, is_unset)` 雙欄位（value 死欄位、產生點不一致）。
- **設計收斂（ADR-0021，supersede 0012）**：(1) value 樹**永遠完整**（每欄位都有 entry、無缺 key）→「缺 key→停用 or 給預設」多義反推消失；(2) 一切「空」統一 `None`，**刪 `DisabledRefValue` + `is_unset`**——停用 ref = 裸 `None`、未填 scalar = `DirectValue(None)`（包裝保 direct/eval 模式）；(3) 檢測「空」一律 value 自述（`fields[k] is None` / `dv.value is None`），不 isinstance sentinel、不讀 flag、不反推 spec；(4) `make_default_value` 是 helper、猜合理預設、optional ref 猜停用（`None`）、特殊需求走 OO 鏈式/工廠；(5)「停用→消失」只在 lowering（run/save 出口），persist 忠實序列化（停用 ref↔`{"__kind":"disabled"}`）。
- **為何推翻 0012 採其否決的 None 選項**：0012 否決 None 的前提是「value 樹有缺 key」（None＝不存在＝與停用混淆）；本 phase 消滅「缺 key」態 → None 一義（entry 在、值 None＝停用），否決理由不成立。0012 消除 8 twotone adapter 3 行 if 的成果保留（factory 回 None 仍一行，`fields` 型別放寬 `Optional`）。與 ADR-0009「呈現/判斷概念由消費層決策、不塞 spec」同源。
- **改動**：core 在 `gui/`（`adapter/{types,inheritance,lowering}`、`live_model`、`services/session_codec`、`cfg_schemas`、`services/remote/{path_resolver,dispatch}`、`ui/fields/common`）；scope 內含 `experiment/v2_gui`（用戶修正過 v2_gui 屬 gui scope）：8 個 `make_*_ref_default(optional=True)` 工廠回 `None`、6 個 twotone adapter `_module_fields` 型別放寬、8 處註解。**lookback 無需改**（省略 key 經 helper 完整兜底自動得 `None`＝停用，新建 tab 與 restore 一致）。
- **順帶治理 `_val`（cfg_schemas）**：原 `_val(cfg, key, default)` 雙欄位編「顯示型別預設當提示但標 unset」——這個「型别预设」是工廠**硬編**的（非 spec/make_default_value），且 default 被 is_unset 覆蓋形同死碼。改 `_val(cfg, key)`：缺 key → `DirectValue(None)`（unset），移除硬編 default（預設歸 spec/helper）。
- **驗證**：`tests/gui`+`tests/experiment/v2_gui` **1067 passed**、`tests/fluxdep_gui`+`tests/dispersive_gui` 298 passed（共用層不受影響）、pyright 0、ruff clean。新增回歸：停用 ref `get_value()` 回 `None`（補 0012 假安全感缺口——舊測試直接餵 marker，真路徑從不產生）、`set_value(None)` 設停用、disable→re-enable round-trip、`make_default_value` 完整性（`set(fields)==set(spec.fields)` + optional→None）、codec round-trip 停用還原仍 None、缺 key 還原 None、scalar 未填 round-trip 保 `DirectValue(None)`。版本號（WIRE/GUI/MCP）不變（純 value 表示 + persist 格式微調，舊檔 strict fallback）。
- **文件**：ADR-0021 新增 + 0012 改 `superseded by 0021`；CONTEXT.md term（Value 樹完整性 + 「空」表示）、AI_NOTE.md 同步（皆 gitignored 工作樹）。**未做 live MCP smoke**（需 user 在 GUI 操作 reset/init_pulse=None → relaunch 驗還原，誠實標記待驗）。

## Phase 137：adapter 必回傳完整 value 樹 + `CfgSchema.validate` 靜態合法性（grill-me 定稿）— ✅ 完成（版本不變，ADR-0022）

**起因**：討論「adapter 怎麼在 spec 固定欄位」（= `LiteralSpec`/`lock_literal`）→ 挖到根本：adapter 的 `make_default_value(ctx)`（~20 個各寫）可在任意深度省略 key（lookback 漏 reps/init_pulse/reset），產結構不完整 value 樹；ADR-0021 用 codec 回退補丁掩蓋。經 grill-me 逐分支定案。

**用戶定錨原則**：value 樹缺 key **必修**（抽 adapter helper 是「痛點 2」分開）；**不為 adapter 擦屁股**（框架不補齊，靠驗證強制）；value 樹**零例外完整**（含 LiteralSpec entry）；LiteralSpec 雙重來源用**驗證**化解（value==spec.value 漂移即 raise）。

- **`CfgSchema.validate(ml)`**（薄 wrapper → `lowering.validate_section`，復用 `_find_allowed_spec`）：靜態查（不需 md）——結構完整（每 spec key 有 entry）+ None 只能 optional ref + LiteralSpec value==spec.value + DirectValue scalar 型別（**int→float widen、float→int reject、bool/str 嚴格**）+ choices；**EvalValue 跳過**（型別 resolve 時定）；Fast-Fail。
- **呼叫點**（成品邊界）：`BaseAdapter.make_default_cfg` 產出後 `validate(ctx.ml)`（adapter 漏/錯當場 raise）；`to_raw_dict` lower 前 `self.validate(ml)`。**不放 `__post_init__`**（誤傷編輯中間態：cfg_form/editor/restore 大量建 CfgSchema 值可暫不合法）。靜態 vs 動態：required-有值/EvalValue-resolve 是動態、留 lowering（要 md）。
- **改 adapter**：lookback 補 `reps:DirectValue(1)`/`init_pulse:None`/`reset:None`；onetone/freq 在 value 端 `.with_field("pulse_cfg.freq",0.0).with_field("ro_cfg.ro_freq",0.0)`（lock_literal 鎖 0.0、value 要服從）；其餘 ~18 adapter **本就完整不動**（全 1068→ 只 2 處需改）。
- **移除 codec `_default_node_value` 回退**（ADR-0021 為修 reps-null 加的）：value 樹保證完整後 capture 不再缺 key；非 ref None 改 fast-fail（`SessionCodecError` incomplete value tree）。
- **LiteralSpec 設計釐清**（grill 關鍵）：`lock_literal` 鎖的欄位 value 帶矛盾值（freq 鎖 0.0 卻 EvalValue("r_f")）是 value 樹自相矛盾的**誤導值**，validate 抓它是對的——freq 實驗掃 freq、readout freq 被 lock，原測試斷言「鎖定 freq 是 EvalValue」是**驗無意義的值**（user 定錨：測試有問題），改成斷言 `DirectValue(0.0)`。
- **lower 加 validate 行為變嚴**：以前漏網不合法 cfg（不在 choices/型別不符）現在被擋；實證現有代碼 100% 型別嚴格 → 不炸。
- **驗證**：`tests/gui`+`tests/experiment/v2_gui`+`tests/fluxdep_gui`+`tests/dispersive_gui` **1387 passed**、pyright 0、ruff clean。新增 `test_validate.py`（20 個）。**live MCP smoke**：lookback/twotone tab 新建 + run 過 validate（待驗）。版本號不變（adapter 內部 + value 驗證，無 wire/RPC）。
- **文件**：ADR-0022 新增（schema validation 邊界、靜態 vs 動態、否決 gate-test/框架補齊/**post_init**/LiteralSpec-只查entry）；AI_NOTE 同步（validate 邊界、adapter 必完整、lock_literal value 服從）。**第二階段（validate 擴動態檢查）+ 痛點 2（有更好 default 值的 helper、覆寫變少）= backlog**。皆 gitignored。

## Phase 138：`CfgSchema.validate_dynamic` 動態合法性檢查（ADR-0022 Phase 2）— ✅ 完成（版本不變）

**起因**：ADR-0022 將 validate 分為靜態（Phase 137）與動態。動態檢查散在 lowering `_section_to_dict_inner` 各 branch，錯誤訊息是實作細節級非契約語義。目標收斂為獨立遞迴驗證。

- **`CfgSchema.validate_dynamic(md, ml)`**（`md` 必要參數）→ `lowering.validate_dynamic_section`：遞迴 Fast-Fail——scalar `DirectValue(None)` raise「is unset (no value to lower)」、`EvalValue` 嘗試 `evaluate_numeric_expr` + `coerce_eval_result`（失敗 raise 附原始錯誤）、`SweepValue` EvalValue edge 同理（type=float）、`DeviceRefSpec` 空/None raise「device not selected」、ref 遞迴（disabled optional skip）、CfgSectionSpec 遞迴、LiteralSpec skip。
- **`_validate_eval`**：獨立 helper（不復用 `_resolve_eval`——後者有 snapshot 優先/cross-check/log warning 語義，validate 只確認可 resolve）。
- **呼叫點**：`to_raw_dict` 在 `self.validate(ml)`（靜態）後、`_section_to_dict_inner` 前呼叫 `self.validate_dynamic(md, ml)`（`md is None` 時跳過）。**不在 `make_default_cfg`**（新建 cfg scalar 本就 unset）。
- **lowering 既有動態檢查保留**（方案 A，重複但安全——validate_dynamic 先給人讀的語義訊息，lowering 自己的檢查作安全網）。
- **不動 adapter/spec/value 型別/靜態 validate/make_default_cfg**。
- **驗證**：`tests/gui`+`tests/experiment/v2_gui` **1105 passed**（新增 `test_validate_dynamic.py` 16 個）、pyright 0、ruff clean。版本號不變（純內部驗證邏輯，無 wire/RPC）。

## Phase 139：`CfgBuilder` value 樹組裝工具（HANDOFF backlog 2 重定義，grill 定稿）— ✅ 完成（版本不變，ADR-0023）

**驗證**：`tests/gui`+`tests/experiment/v2_gui` **1122 passed**（新增 `test_cfg_builder.py` 17 個）、pyright 0（全 gui+v2_gui）、ruff clean。版本號不變（純 value 層組裝工具，無 wire/RPC）。

**全量遷移完成**（commit `5d66bf04` builder+3 pilot、`526bc079` 其餘 16）：全 20 個 adapter 的 `make_default_value` 統一走 CfgBuilder（含 fake stub）；`test_*_adapters.py` 自動回歸綠（含 `ignore_library_readout`）；遷移淨刪 ~200 行。映射：`make_<role>_default`→`.role(prefer_blank=True)`、`make_<role>_ref_default(ctx)`→`.role()`、`(optional=True)`→`.role(optional=True)`、EvalValue 邊界 sweep→`.set_sweep`、字面→`.sweep`、無條件禁用 optional ref→不寫（L1 blank 給 None）。

**Builder 自動填鎖定 literal 值（commit `64bda5b6`，C-raise，1125 passed +3）**：消除 adapter 重複宣告鎖定值（onetone/freq 刪 `.set(pulse_cfg.freq,0.0)`/`.set(ro_cfg.ro_freq,0.0)`——本與 `cfg_spec().lock_literal(...,0.0)` 同值寫兩遍）。`build()` 遍歷 spec 把每個 `LiteralSpec` leaf 對齊 `spec.value`，穿透已掛 ref 的 chosen shape（lowering 的 `_find_allowed_spec` 公開為 `find_allowed_spec` 供 builder 復用）；`.set` 碰 locked path 直接 raise（adapter 不該手設）。**LiteralSpec 一致性檢查點只在 validate**（lowering 本體對 LiteralSpec 直接用 `spec.value`、不看 value 不比對）——三道防線：CfgBuilder.build 事前對齊／`to_raw_dict` 的 validate 事後 raise（守手拼/codec restore/editor draft 等非-builder 路徑，含 value=None→raise）／lowering 信任 spec.value 產出。互補不冗餘。

**起因**：adapter 的 `make_default_value(ctx)` 現況是手拼 `CfgSectionValue(fields={...})` + 手動調各 `make_<role>_default(ctx)`（如 `twotone/freq.py:124-142` 17 行）。未來要把 `experiment/v2/` 全部實驗都做 adapter（50+），重複痛點放大。用戶要一個降低 adapter 實作難度的組裝工具。

**grill 過程定錨的事實（推翻 HANDOFF 舊描述）**：

- HANDOFF backlog 2 的「`default_value_from` + 明文 dict + type 字串比對 ref」**已被 ADR-0009 的 `shared/defaults/` per-role factory 主體取代**；HANDOFF 是舊快照。
- 預設值是**角色相關**非 field-name 相關（同名 `gain` 在 qub_probe=0.05、readout=0.1），扁平 dict 解不了；領域默認**大量是公式**（`trig_offset=timeFly+0.05`、`post_delay=5/(2π·rf_w)`），純資料表裝不下 → 必須是代碼（現 L2 factory 即是）。
- 「角色」已 first-class：`RoleCatalog`+`RoleEntry`，Inspect 建 module/waveform 用 L2 factory（第二消費者）→ **L2 簽名鎖定不能被 builder 收編**。

**三層架構（L1/L2 已存在，新增 L3）**：

- **L1** `make_default_value(spec)`（`inheritance.py`，gui 框架層，不動）：結構完整、值中性骨架。
- **L2** `make_<role>_default(ctx)` / `make_<role>_ref_default(ctx, optional=)`（`shared/defaults/`，領域層，**簽名不動**）：單角色 ModuleRefValue/WaveformRefValue + 領域默認（含公式 EvalValue + library 查找）。RoleCatalog/Inspect + Builder 三方消費。
- **L3** `CfgBuilder`（新增 `shared/cfg_builder.py`，**領域層**——須懂 ctx+role+library，不可污染框架層 `CfgSectionValue`）：flat-path fluent 組裝工具。

**`CfgBuilder` 契約**（in-place 持 ctx+spec+value，起手 = L1 blank 骨架）：

- `.scalars(**kwargs)`：頂層 scalar，**純顯式無內建表**（各 adapter relax_delay/rounds 本就不同，共用表藏意圖）；bad key fast-fail。
- `.role(path, role_id, *, optional=False)`：經單一 `ROLE_FACTORIES` 表（`defaults/__init__.py`，registry+builder 共用 source）直調 L2；`optional=True`→`_ref_default(ctx, optional=True)` 可回 None；非 ref spec / kind 不匹配 / 對 required ref 用 optional 都 fast-fail。
- `.set(path, value: ScalarLeafInput)`：path scalar 覆寫（複用 `with_field`），spec-aware fast-fail。
- `.sweep(path, start: float, stop: float, expts: int)`：**只收字面 float**，任一邊要 EvalValue → raise（提示走 set_sweep）。
- `.set_sweep(path, sweep: SweepValue)`：逃生口，邊界可含 EvalValue。
- `.build() -> CfgSectionValue`：回 value 樹，一次性（build 後 mutate raise）；**不 validate**（留 `make_default_cfg` 邊界）。
- **挂整節點能力 = Builder 私有 `_mount_node`**（下鑽 path 父節點塞整節點/None）；框架層 `CfgSectionValue.with_field` 保持 scalar-only 不動。

**守既有 ADR**：Builder 零鎖定（ADR-0009 決策 5，鎖定 100% 在 `cfg_spec().lock_literal`）；value in-place（決策 4）；value 樹完整 None-for-empty（ADR-0021）。

**關鍵風險（Q1，已定 Option A）**：adapter 經 `Registry.create()` 無參構造、`make_default_value(self, ctx)` 只收 ctx，**`ExpContext` 不帶 RoleCatalog**（catalog 只在 Controller）；且 catalog factory **永不回 None**（表達不了 optional→None，那能力只在未註冊的 `_ref_default`）。→ Builder **不走 RoleCatalog**，領域層直調 L2 函數 + 自己的 `ROLE_FACTORIES` 表（與 registry 共用 source，含 `_ref` 變體）。Inspect/catalog 與 Builder 是同一批 L2 函數的兩個平行消費者。

**新 ADR-0023**：記錄 value 層為何引入 builder（對照 0009 spec 層不用 builder）——spec 無 default 聚合邏輯、value 有，且領域邏輯（role/ctx/library）不能下沉到框架層數據容器 `CfgSectionValue`，Builder 是這堆邏輯的正確歸屬（領域層）。

**Pilot 遷移 3 個**（舊手拼寫法仍合法，漸進遷移無 big-bang）：`twotone/freq`（.scalars+.role 必填+optional-miss+sweep）、`lookback`（.role+post-mount .set+None optional+LiteralSpec reps=1）、`amp_rabi`（.set_sweep with eval 邊界）。其餘 ~16 adapter 待 pilot 通過、API 凍結後再遷。

**測試**：`tests/experiment/v2_gui/adapters/shared/test_cfg_builder.py`——結構完整性（無覆寫 == L1）、.scalars/.set/.role/.sweep 各路徑 + fast-fail、optional-miss→None、sweep 數學 + 非 float raise、一次性 build；pilot 等價性（builder 重現舊手拼樹）由既有 `test_*_adapters.py` 驅 `make_default_cfg` 自動回歸。

## Phase 140：掃描 adapter 鎖定被掃模組欄位（lock_literal 對齊 onetone/freq）— ✅ 完成（版本不變）

**狀態**：B 表 11 個 + 錯置回收的 `ro_optimize/length` 的 `ro_cfg.ro_length`（共 12 處 adapter 鎖定）已加 `lock_literal`；框架補上 `WaveformRefSpec` 的 lock 下鑽能力（(a) 遺漏）；波形內 length 三處依 (b) 維持 editable（不鎖）。`tests/gui`+`tests/experiment/v2_gui` **1129 passed**（adapter 等價 + 新增 4 個 WaveformRefSpec descent 測試）、pyright 0、ruff clean。

**C 表處置（grill 後定案）**：

- **錯置回收（其實是 B）**：`ro_optimize/length` 的掃描軸 `ro_cfg.ro_length` 是頂層 scalar、純 placeholder（`set_param("ro_length")` 逐點覆寫），照 B 鎖。自動推導的 pulse 波形 length（`max+0.11`）屬波形內，不鎖。
- **(a) 框架遺漏 → 已補（`gui/app/main/adapter/types.py`）**：`lock_literal`/`_with_override`/`_path_exists` 原只穿透 `CfgSectionSpec`+`ModuleRefSpec`，不穿透 `WaveformRefSpec`（不對稱）。補 `WaveformRefSpec._with_override`+`lock_literal`（mirror ModuleRefSpec）、`CfgSectionSpec._with_override` descent 加 `WaveformRefSpec`、`_path_exists` 同。框架現可鎖波形內 leaf（如 `qub_pulse.waveform.length`），與「是否真去鎖」脫鉤。測試見 `test_spec_value_fluent.py`（descent / chain-start / full-path / no-match raise）。
- **(b) 波形 length 不鎖（語意理由，用戶同意）**：`len_rabi` / `ro_optimize/auto` / `ro_optimize/length` 的 pulse 波形 length 不鎖。理由：gauss/drag 的 `set_param("length")` 會 `sigma_ratio=sigma/length; sigma=ratio*length` —— 靜態 length **定義脈衝形狀比例**並在掃描全程保持，非 don't-care placeholder（鎖 0.0 會 ZeroDivisionError）；且可鎖性隨 runtime 選的波形 shape 變（const 是 don't-care、gauss 有意義、arb 拒絕 set_param），靜態 spec 鎖無法 conditional。維持 editable 才正確。

**B 表交付細節**：改動全在 `experiment/v2_gui/adapters/`（屬 gui scope），每處加註「sweep 軸接管該欄→鎖起來不上表單」；make_default_value 不需改（無 `.set` 落在被鎖 path，role factory 內部設值由 `build()._align_literals` 覆蓋對齊）。AI_NOTE 既有 line 152 已記此原則（onetone/freq 範例），不需結構性更新。

**起因**：user 指出多個「掃 module 參數」的 adapter（如 `twotone/freq` 的 qub_pulse freq）沒把被掃欄位 `lock_literal`，應比照 `onetone/freq`。

**機制**：實驗在 run 時對模組呼叫 `set_param(...)` 把被掃欄位換成 sweep 軸（如 `twotone/freq` 的 `modules.qub_pulse.set_param("freq")`）。該欄位由 sweep 擁有、靜態值是 placeholder；不 lock 會讓表單顯示一個 run 時被靜默覆寫的可編輯欄位（違反最小驚訝）。`onetone/freq` 的作法：`cfg_spec()` 回傳的 ref spec 鏈上 `.lock_literal("pulse_cfg.freq", 0.0).lock_literal("ro_cfg.ro_freq", 0.0)`（PulseReadout 的 `set_param("freq")` 同時寫 pulse 與 ro 兩處 freq）。

**重要查核（修正初判，[[feedback_verify_real_path]]）**：lock_literal 與「library 連結模組」**不衝突**。四條 value 樹建構路徑都強制 `LiteralSpec` leaf = `spec.value`、忽略 payload/library 帶的值：① `CfgBuilder.build`→`_align_literals` ② LiveModel `LiteralLiveField.get_value` 恆回 spec.value（連結 library 走 `_rebuild_sub_field` 也經此） ③ codec restore `_node_value_from_raw` LiteralSpec→`DirectValue(spec.value)` ④ agent `raw_to_schema` 同上。`validate` 因此恆過。初判的「validate 失敗」來自一個**繞過全部真實路徑**的合成 repro（手 new `ModuleRefValue(value=module_cfg_to_value(lib))`），非任何 GUI 行為會產生 → 不是 blocker。

**待改 adapter（B：乾淨案例，被掃欄位是頂層 scalar 或 pulse-readout 子欄）**

| adapter | 被掃 (`set_param`) | 需鎖 path（相對 ref shape） |
| --- | --- | --- |
| `twotone/freq` | qub_pulse `freq` | `freq` |
| `twotone/power_dep` | qub_pulse `freq`+`gain` | `freq`, `gain` |
| `twotone/flux_dep` | qub_pulse `freq` | `freq` |
| `twotone/rabi/amp_rabi` | qub_pulse `gain` | `gain` |
| `onetone/power_dep` | readout `freq`+`gain` | `pulse_cfg.freq`+`ro_cfg.ro_freq`, `pulse_cfg.gain` |
| `onetone/flux_dep` | readout `freq` | `pulse_cfg.freq`+`ro_cfg.ro_freq` |
| `twotone/ro_optimize/freq` | readout `freq` | `pulse_cfg.freq`+`ro_cfg.ro_freq` |
| `twotone/ro_optimize/power` | readout `gain` | `pulse_cfg.gain` |
| `twotone/ro_optimize/freq_gain` | readout `freq`+`gain` | freq 兩處 + `pulse_cfg.gain` |
| `twotone/ro_optimize/auto` | readout `freq`+`gain`（+length 是 pulse 波形內、屬 C） | freq 兩處 + `pulse_cfg.gain`（length 不鎖）|
| `fake/freq` | readout freq（模擬） | 同 onetone，但 spec 用 `make_readout_module_spec()`（含 direct+pulse 兩 shape）→ 需處理 per-shape path |

**待改 adapter（C：棘手，被掃欄位在 waveform ref 內 / 被實驗自動推導）**

- `twotone/rabi/len_rabi`：`qub_pulse.set_param("length")` 寫進**波形**的 length（在 WaveformRef 多 shape 內），非頂層 scalar，鎖法待設計。
- `twotone/ro_optimize/length`：掃 `ro_cfg.ro_length`（可乾淨鎖）+ 實驗另自動把 pulse length 設 `max+0.11`（也在波形內）。
- `twotone/ro_optimize/auto` 的 length 同上（pulse length 自動推導）。

**不需改（被掃軸非表單重複欄位）**：`twotone/time_domain/{t1,t2ramsey,t2echo}`（delay 對應實驗內部 Delay module、未在表單暴露）；`lookback`（無 sweep）；`onetone/freq`（已正確）。

**待討論（如何正確宣告 ref 內屬性的 lock）**：

1. **per-shape path**：同一語意欄位在不同 allowed shape 路徑不同（pulse-readout=`pulse_cfg.freq`+`ro_cfg.ro_freq`、direct-readout=`ro_freq`）。`lock_literal` duck-type 套到「含該 path 的所有 shape」、無 shape 命中才 raise → 兩 shape 都要鎖需各列。只有 `fake/freq` 用雙 shape readout 且掃 readout，其餘掃 readout 的都是 pulse-only spec。
2. **語意**：`lock_literal`（值恆為某字面、漂移=bug）vs 一個專用「swept/hidden、值 don't-care」標記。操作上等價（四路徑都強制 spec.value、lowering 一律輸出 spec.value、run 再 set_param 覆寫），且 notebook 慣例就是 `freq: 0.0, # not used` → 傾向沿用 lock_literal(0.0)。
3. **UX**：連 library 校準模組（如 `qub_probe`/`readout_rf` 帶真實 freq）時，被鎖欄位在表單消失、tree 內被強制 0.0。對掃描實驗正確（該欄被掃），但屬有意識取捨。
4. **placeholder 值**：採 0.0（對齊 notebook + onetone/freq）。

## Phase 141：optional 標量框架特性 + Alt D 分組 + pulse `mixer_freq`（消化 pulse 欄位缺漏）— ✅ 完成（版本不變）

**起因**：user 發現 `make_pulse_spec()` 漏掉 `PulseCfg` 多個欄位（`mixer_freq`/`mux_*`/`mask`/`outsel`/`ro_ch`/`desc`）。審計發現全是 `Optional` 預設 None，漏掉的根因是框架兩個表達缺口：① 無「optional 標量」（None scalar 在 lowering 一律 raise，`required` 是反方向）② 無 list 欄位 spec。決策：不暴露 mux（多工範式正交）；先做 `mixer_freq`；`ro_ch` 由 PulseReadout autofill 處理（program scope，本 phase 不做）；UX 用 **Alt D**（摺疊 Advanced 區，不耦合 soc、不做反應式 show/hide）。

- **Step 1 — optional 標量特性**：`ScalarSpec` 加 `optional`（與 `required` 互斥，`__post_init__` 守）；`ScalarLiveField._refresh_validity` optional+None=valid；lowering `_section_to_dict_inner` optional+None→**省略 key**、`validate_dynamic` 同放行；`make_default_value`/`inherit_from` optional→`DirectValue(None)`；widget（`common.py`）optional 數值/字串 direct-mode 改 `QLineEdit`（empty=None、placeholder `(none)`、numeric validator），`optional+choices/bool` fast-fail；`read_scalar_widget` optional-aware。靜態 `_validate_scalar` 無需改（已對 None return）。
- **Step 2 — Alt D 分組 + mixer_freq**：`ScalarSpec` 加 `group`（純呈現）；`SectionWidget._build_children` 把非空 group 欄位收進預設摺疊的 `_CollapsibleSection`（抽 `_add_field_row`）；`make_pulse_spec` 加 `mixer_freq`（`optional=True, group="Advanced"`）。pulse 的 `nqz`/`phase` 也歸 `group="Advanced"`（非 optional、僅渲染分組）——主表單只剩 waveform/ch/freq/gain/pre_delay/post_delay。**value 樹不變形**（grouped 欄位仍頂層 leaf、照常 lower）。
- **連帶修**：`cfg_schemas._pulse_to_value` 補 `"mixer_freq": _val(cfg, "mixer_freq")`（module_cfg_to_value 也要產完整 value 樹，否則 validate 報 missing）——這是唯一被新欄位打到的既有轉換點，其餘（role factory / CfgBuilder / inheritance）由 L1 自動補 None。
- **未做（依決策）**：mux_*/mask（list 缺口 + 範式正交）、`outsel`/`gen_ch`/`desc`（同 optional 特性，將來要時直接加）、`ro_ch` autofill（program scope，待 user 同意動 `program/v2/modules/readout.py`）、動態 soc-driven show/hide（Alt C，成本高且需反應式框架，延後）。
- **驗證**：`tests/gui`+`tests/experiment/v2_gui` **1148 passed**（新增 optional 資料層 13 + live_model validity 2 + widget 3 + 分組 1）、`tests/fluxdep_gui`+`tests/dispersive_gui` 298（無共用層回歸）、pyright 0、ruff clean。AI_NOTE 同步 optional/group/widget 契約。

## Backlog：pulse mux 欄位（list 型別框架特性）— 🅿️ 待實作（已定設計、未動手）

**起因**：user 指出**有些 gen channel 需要 mux 資訊才能運行**（非 YAGNI，有真實硬體需求）。`make_pulse_spec()` 漏的 `PulseCfg.mux_freqs`/`mux_gains`/`mux_phases`（`Optional[list[float]]`）+ `mask`（`Optional[list[int]]`）目前**全 codebase 只存在於 model 定義**（無 experiment/library/notebook 設定過），但硬體層 `declare_gen` 會用（`pulse.py:82-90`）。要放進 pulse 的 Advanced group。

**框架缺口**：spec 系統是 7 spec × 6 value、**全 fixed-cardinality**，沒有任何 list/變長型別。mux 是第一個 list 欄位。

**方案 A（user 接受、推薦）—— 輕量 `ListSpec` + 逗號分隔 `QLineEdit`**：

- 不能只用 optional-字串欄位（`PulseCfg.mux_freqs` 要真 `list[float]`、欄位埋在共用 pulse module、多 adapter 共用）→ string→list 轉換必須在 **lowering 邊界**，故需一個「lower 成真 list」的 spec 型別，但可很薄。
- **value 複用 `DirectValue`**（裝 `list` 或 `None`=unset），不新增 value 型別（系統維持 6 value 型別；`DirectValue.value` 是 `Optional[Any]` 已可裝 list）。
- **widget = `QLineEdit`** 逗號分隔（讀 split+parse、空=None、格式化顯示），非變長 table。
- **改動面**（每處小，多是加一個 ListSpec 分支）：`types.py`（新 `ListSpec(label,item_type,optional,group)`）/ `inheritance.py`×2（optional→`DirectValue(None)`）/ `lowering.py`（emit list + static/dynamic validate item 型別 + optional 空省略）/ `session_codec.py`（round-trip）/ `live_model.py`（小 `ListLiveField`）/ `ui/fields`（`ListWidget` + 註冊）/ `path_resolver.py`（RPC 分支，**scope 待定**）/ `specs/pulse.py`（4 欄 ListSpec, group="Advanced"）/ `cfg_schemas._pulse_to_value`（補 4 欄，同 mixer_freq）/ 測試。
- 規模：比 mixer_freq 大（新 spec 型別走全棧分支）、比完整 table 小（無變長列 UI、無新 value 型別）。

**方案 B —— `QTableWidget` 表格**：4 條相關 list 是一張表（列=tone、欄=freq/gain/phase + active→mask），結構上防 desync，但工程最大（變長列 UI + 自訂 spec + lowering 把表拆回 4 list）。

**已知取捨（方案 A）**：逗號分隔=4 獨立欄位 → 三條 mux list 長度可能 desync；**不**在 GUI 做跨欄等長檢查（要動 `PulseCfg`，out of scope），長度不符由 `declare_gen` 在 run fast-fail。要「結構上不可能 desync」才需走方案 B。

**待 user 決定**：① 方案 A（逗號分隔，推薦）vs B（表格，防 desync）；② RPC `path_resolver` 是否含（agent 也能設 mux）還是先只做 GUI。`mux` 之外的 `outsel`/`gen_ch`/`desc` 仍走既有 optional 標量、要時再加；`ro_ch` autofill 屬 program scope（見 Phase 140 討論）。

## Phase 142：adapter `cfg_spec` 頂層欄位順序統一（消化 user report）— ✅ 完成（版本不變）

**起因**：user 發現 adapter 的 `cfg_spec()` 欄位順序彼此不一致（form 顯示順序混亂）。最初以為要逐實驗對齊 `notebook_md/single_qubit.md`，但 single_qubit 的能譜 cell 是 `sweep` 在前、時域 cell 是 `relax_delay` 在前——**user 澄清那些 sweep-在前的能譜 cell 是漏改的、不當基準**，要**統一成 `relax_delay` 在 `sweep` 前**。

- **canonical 順序（統一一套）**：`modules, [dev], relax_delay, sweep, [run-only kwargs], reps, rounds`。
  - `dev`（flux device section）緊接 `modules`。
  - `relax_delay` 永遠在 `sweep` 前。
  - run-only kwargs（`earlystop_snr` / `num_points`，在 notebook 是 `run()` kwargs、不在 `exp_cfg`）夾在 `sweep` 與 `reps` 之間。
  - `reps`,`rounds` 永遠最末（user 指定）。
- **modules 內順序**：`reset → pulses → readout`（readout 永遠最後、reset 最前＝執行序）。只有 **lookback** 違反（原 `readout,init_pulse,reset`），改為 `reset,init_pulse,readout`。其餘 adapter modules 順序本就正確。
- **改動範圍**：18 個 adapter `cfg_spec`（onetone freq/power_dep/flux_dep、twotone freq/power_dep/flux_dep、rabi len/amp、time_domain t1/t2ramsey/t2echo、ro_optimize freq/power/length/freq_gain/auto、fake/freq）+ lookback modules。`fake/stub`（純框架 stub、無實驗對應）未動。`fake/freq` 無 `relax_delay` 欄位 → `modules,sweep,reps,rounds`（無 relax 可前置，符合規則）。
- **scope**：只動 `cfg_spec()` 的 `fields` dict 順序（＝form 顯示順序）；`make_default_value` builder 呼叫照路徑定址、不需同步、未動。lowering 走 `ml.make_cfg` 按 key 映射到 pydantic 具名欄位，**順序不影響行為**（只影響表單）。
- **教訓**：① 初版誤信 notebook 為 ground truth、想保留「能譜 sweep 前 / 時域 relax 前」二分法，user 一句「統一 relax 在前、有些 cell 我忘了改」推翻——**notebook 不必然自洽，欄位順序這種約定要跟 user 確認單一 canonical**。② 機械批改後**務必跑結構化驗證**（逐檔印 top-level key order），別只靠 Edit 成功與測試綠（測試不檢查欄位順序，相關斷言皆 set 比較）。
- **驗證**：pyright 0/0、ruff clean、`tests/experiment/v2_gui`+`tests/gui` 全綠（adapter 309 / gui 1005）。版本號不變（無 wire/RPC 改動）。

## Phase 143：adapter spec 宣告簡化（IntSpec/FloatSpec + build_exp_spec）— ✅ 完成（版本不變）

**起因**：user 想簡化 `cfg_spec()` 宣告樣板。否決「ScalarSpec 預設 type=int」（type=int 42 vs type=float 40 幾乎五五波，靜默預設踩最小驚訝），改走顯式 sugar + 根組裝器。

- **`IntSpec`/`FloatSpec`**（框架層 `gui/app/main/adapter/types.py`，re-export 自 adapter package）：`ScalarSpec(label=..., type=int/float)` 的顯式 factory（顯式、可讀、無隱藏預設）。`FloatSpec` 帶 float-only 的 `decimals`。
- **`build_exp_spec` + `declare_modules_spec`/`declare_sweep_spec`/`declare_dev_spec`**（domain 層 `shared/spec_helpers.py`）：`build_exp_spec` 組裝 root cfg spec，固定吐出 canonical 順序 `modules, [dev], relax_delay, sweep, [extra], reps, rounds`，並自動補標準 `relax_delay`(float,3)/`reps`(int)/`rounds`(int)。**頂層欄位順序與三個標準 scalar 由單一所有者擁有**——Phase 142 的欄位順序不一致 bug 從此結構上不可能（adapter 不再手排）。
  - 參數：`sweep=None`（lookback 無 sweep 省略）、`sweep_label`（auto 的「Search bounds (min–max)」）、`dev`（flux_dep）、`extra`（earlystop_snr/num_points 等 run-only knob，夾在 sweep 與 reps 間）、`relax_delay=False`（fake/freq）、`reps=LiteralSpec(1)`（lookback 鎖定）。
- **18 個 adapter 遷移**到 build_exp_spec（fake/stub 無實驗對應不動）。行為等價：runtime 驗證 7 個代表案的 `cfg_spec().fields` 順序與 Phase 142 committed 結果完全一致。
- **手法**：foundation 我自己做 + onetone/freq pilot 驗證 API；其餘 17 個分 4 個 sonnet sub-agent 平行遷移（各帶精確 per-file 目標 + ruff/pyright 自檢）。ruff `--fix` 自動移除每檔 unused `ScalarSpec`。
- **驗證**：pyright 0/0/0、ruff clean、`tests/experiment/v2_gui`+`tests/gui` **1148 passed**。commit `ef0c9fe1`（Phase 142 = `433c6df1`）。

## Phase 144：消化測量 agent 實戰反饋（`agent-report.md`）— ✅ 完成（Step 1 `09bfa822`+stale-fix `84568640`、Step 2a `cea2b10a`、Step 2b `e020583b`、Step 2c agent-memory record 無 commit）

**起因**：一次 Q5_2D / Q1 / R1 resonator one-tone + flux-dependence 量測 session 後，測量 agent 寫了 `agent-report.md`（gitignored），列出 skill / MCP / codebase 三層缺口。逐項核對代碼後定案。

**核對校正（駁回 report 兩個前提）**：

- report 稱「flux_dep run 無圖可看」**不成立**：`gui_tab_figure_screenshot`（截 `_plot_stack` 當前 canvas，`main_window.py:1363`）對 `supports_analysis=False` 的 2D tab 一樣截得到 run 的 `LivePlot2DwithLine` colormap（run/analyze 共用同一 `_figure_container`）。→ 真缺口是 **skill 沒教 agent 截 run 圖**，非工具缺。
- report MCP #4（long-run wait 卡整個 turn）**已於 commit `2516904b`（同日）修畢**（server v24 + SKILL v14 三副本補 poll / background-agent 指引）→ 不再做。

**用戶定案**：

- **不做數值摘要 RPC**（原規劃 C1）：run result 是 ndarray，連摘要回 agent 都嫌佔 context；agent 一律**靠看圖**判斷 2D 結果。
- **flux_scan 設定僅作建議**：measure_setup.yaml 可選填、非權威、不當必填。
- **flux 選點互動工具**（原規劃 C2）：用戶要另立新工具設計，**本 phase 不做**（backlog；照 fluxdep/dispersive「user 驅動互動、agent 唯讀」哲學）。
- **新增：合併取圖工具為 `get_current_figure`**：取圖機制其實**已是單一 RPC**（`tab.figure_screenshot` 截當前 canvas，run colormap 或 analysis fit 皆可；run/wait/poll/analyze 都經 `_figure_path_if_any` 折回 `figure_path`，`server.py:822/857`）。工作是**改名 + 統一心智模型**讓 agent 操作直覺，非合併邏輯。

**Step 1（code/wire，一次版本 bump）— ✅ 完成（commit `09bfa822`；WIRE 20→21 / GUI 19→20 / MCP 24→25；tests/gui 1013 pass、pyright 0、ruff clean）。core 校正：path_resolver 的 docstring/AI_NOTE 早已聲稱 `.device` 可用是 aspirational，code 從未實作——本次補上使之成真。順帶（用戶要求，獨立 commit `84568640`）修了一個與本 phase 無關的 pre-existing stale 測試 `test_shared_layer.py::test_mcp_server_has_no_relative_imports`（MCP 搬遷 c8eb1a03 後仍指向舊 `gui/app/*/services/remote/mcp_server.py`，改指 `mcp/{measure,fluxdep,dispersive,agent_memory}/server.py`）。**live MCP smoke 待 `/mcp reconnect measure-gui` 後驗**（看 banner `wire v21; gui code v20, mcp code v25` + 改名工具 `gui_tab_get_current_figure` 生效 + deviceref `.device` set_field 成功 + save 回 `{data_path}`）。**：

- **get_current_figure 改名**：`tab.figure_screenshot`→`tab.get_current_figure`（method_specs + dispatch handler key + `_figure_path_if_any` 內部呼叫 + server instructions）、`gui_tab_figure_screenshot`→`gui_tab_get_current_figure`（mcp override + `_OVERRIDE_NAMES` + `test_mcp_generation` 名單）；description 重述為「回傳當前顯示的圖（run 2D colormap 或 analysis fit，誰在最上面截誰）」。
- **B1 DeviceRef 路徑兩端對齊（真 bug，[[feedback_dual_end_user_agent]]）**：`list_paths` 廣告 `dev.<x>.device`（kind=deviceref，`path_resolver.py:302`）但 `path_resolver` 無 `.device` 分支、只認裸 `dev.<x>`（`:112`）→ agent 照 discovery 設值必失敗。補 `.device` 分支（對稱 moduleref 的 `.ref`），令 discovery path == settable path。
- **B2 save.data 直接回解析路徑**：`safe_labber_filepath()` 在 `start_save_data` 同步算出 `_N.hdf5` 路徑（`services/save.py:57`）→ 沿 controller→handler 回 `{data_path}`，免事後從 diagnostic / snapshot.save_paths 撈。

**Step 2（經驗歸檔 —— 按「目的地」分流，非全塞 skill；Step 1 後做，引用最終名/行為）— ✅ 完成**。落地：(2a `cea2b10a`) onetone/twotone flux_dep `guide()` recommended 擴充（gain 取捨修正單面 0.005／dispersive 小→早收窗／reverse-sweep／截圖看圖）；(2b `e020583b`) run-measure-gui skill v14→15（**guide-first linchpin**＋決策邊界段＋`get_current_figure` 看 2D run 圖＋save 回路徑）、measure-setup v1→2（task-driven 擴充＋**可選**advisory flux_scan 區塊），三副本 `sync_skills.sh` 同步；(2c) agent-memory 寫 1 條 episodic record `records/Q5_2D/Q1/2026-06-08-onetone-freq-onetone-flux-dep`（surprise＋「目標是右側 dip」決策；校準值在 MetaDict、reusable 規則在 guide，**故不 seed solution 避免兩地重複**——doctrine），`agent_memory/` 是 gitignored runtime store 故無 commit。底下原計畫條列為當時定錨，實作依此。 詳細 ↓

報告裡的經驗依性質分四個歸屬。**不開子 skill**——領域層的家是 AdapterGuide（`gui_adapter_guide` 已是 per-experiment、agent-queryable 的「實驗說明」面），開子 skill 會與它重複：

- **通用流程 → skill**：
  - `run-measure-gui`：① **「跑實驗前先查 `gui_adapter_guide`」工作流步驟**（**linchpin**：domain 經驗住 AdapterGuide，唯有流程強制先讀 guide，那層知識才會被消費——skill 現只提及 guide 存在、未列為步驟）② **決策邊界段**（agent 可自決 vs 必問用戶：粗掃多 dip 選哪個目標、弱結構是否目標模式→必問）③ **用 `get_current_figure` 看 2D run 圖**（明說非 analysis 專屬，補認知缺口）。其餘（figure-first、ambiguity→停問、hm lookback 離共振）skill 既有，不重寫。
  - `measure-setup`：**task-driven 擴充**（換任務先擴檔、只問新欄位、不重問舊的）+ **可選** flux_scan 建議區塊（僅建議，Q2）。
- **具體實驗細節 → 實驗說明（AdapterGuide `guide()`，非 skill）**：onetone/flux_dep + twotone/flux_dep 的 `behavior`/`recommended` 擴充——連續 flux 掃反轉 start↔stop 省 ramp、dispersive shift 小→早收頻窗、SNR 差→gain 從 ~0.005 往 ~0.05（**修正現有 `recommended` 只寫 0.005 的單面建議**，註明 punch-out vs SNR 取捨）。`flx_half`/`flx_int`/`flx_period` 定義**已在 magic_names.md（行 41-43）**不動；其穩定 per-rig 值歸 context md（+ 可選 measure_setup）。
- **問題解決記錄 → agent-memory**：
  - **record**（episodic）`records/Q5_2D/Q1/2026-06-08-*`：粗掃 5.35–5.65 GHz 多 dip、user 選右側 dip、dispersive shift 小須收窄頻窗、gain 0.05 才看得清、反轉 flux 掃省時。（`r_f≈5550.73`/`rf_w≈8.40` 等校準值住 MetaDict，**不進 memory**。）
  - **solution**（context-free，provisional）：reverse-sweep / narrow-window-small-shift / raise-gain-poor-SNR 與上面 AdapterGuide 規則**同源**。Doctrine：owner 已 bless 的廣義規則放 guide/skill（curated、always-loaded），memory solution 由測量 agent 在 session 中自行 accrete（provisional→confirmed≥2）；**同一規則不兩地重複，confirmed 後從 memory「畢業」進 guide**。本批可選擇性 seed 1–2 條 confirmed solution 讓下次 `memory_search` 撈得到。
  - record/solution 的「何時寫」紀律**已在 agent-memory server instructions**，不需另寫 skill。
- bump `skill_version`、跑 `sync_skills.sh` 同步 `.claude`/`.agent`/`.codex` 三副本（[[feedback_skill_present_tense_sync]]）；guide 改動屬 `experiment/v2_gui`（gui scope）。

**scope**：全在 gui scope —— `gui/app/main/services/remote/`（wire）、`experiment/v2_gui`（v2_gui 屬 gui scope）、`mcp/measure`（wire 消費方 passthrough）、skills（本任務 deliverable）。**不動** `experiment/v2/`（result dataclass 只讀不改）。

**收尾**：pyright/pytest、ruff（先 `--select I --fix` 再 format）；live MCP smoke 需 `/mcp reconnect measure-gui` 後驗（[[feedback_mcp_fresh_server]]，靠 WIRE handshake + 觀察改名工具生效，非版本號）；commit 待用戶要求。

## Phase 145：INTERACTIVE analysis capability + flux-pick（flx_half/flx_int 互動選點）— ✅ 完成（S1 `bf249fef`、S2 `2e4fefcf`、S3a `e2b2b7a1`、S3b/c `746d7d96`、S4 `4b36bb63`、S5 `d13911a4`、S6 skill+docs）

**起因**：消化 agent-report MCP #2（Phase 144 deferred C2）。`onetone/twotone flux_dep` 跑完是 2D map，要從圖上取 `flx_half`/`flx_int`（sweet-spot flux 值，定 flux period），這步**需人判斷 + 點選**，agent 不能可靠自動決定。經多輪設計討論收斂。

**設計收斂（對話定錨）**：

- capability `supports_analysis: bool` → 判別式 `analysis: AnalysisMode = NONE | FIT | INTERACTIVE`。`INTERACTIVE` = 結果由人互動產生、延遲取得（**命名用本質「interactive」非後果「async」**；延遲是 INTERACTIVE 的推論性質，且不與未來「慢計算」混淆）。
- **cohesion（用戶堅持）**：客製分析的實驗智慧（seed 計算、線→flx 解讀、寫回）綁 **adapter**；base 預設 = 標準行為、子類覆寫 = 互動（**多型即註冊，無 gui-side kind→widget registry 表**）。
- **Qt-free 不變式守住**：adapter 只碰 matplotlib（在 gui 給的 axes 上畫 map + 掛 `TwoLinePicker`），Qt（canvas/chrome）全留 gui。「adapter 回傳 widget」被否決（破 Qt-free + 反轉依賴），改「gui 給 axes、adapter 在上面佈置」。
- **gui 對互動方式零認知（用戶定錨）**：gui host 是 generic 宿主——建主線程互動 canvas、給 `Axes`、給 generic「Done」、驅動 `InteractiveSession` 協議（`finish()->AnalyzeResult`），**不知道是線/點/什麼**。「這個分析用兩線選點」的知識**只在 adapter**（它選擇 instantiate `TwoLinePicker`）。新增別種互動 = 新 adapter 用別的原語，gui host 一行不改。互動專屬小按鈕（swap/auto-align）v1 由 `TwoLinePicker` 用 mpl 事件/鍵吃掉；要 GUI 按鈕則 session 暴露 generic `actions:[(label,callback)]`，host 照單渲染不懂語意。
- **互動原語 `TwoLinePicker` 住 analysis 層（非 gui）**：matplotlib 核心放 `notebook/analysis/fluxdep/interactive/` 那帶（InteractiveLines 鄰居），adapter import、notebook 包，**gui 永不 import**。fluxdep-gui 的 `line_picker.py` 是它自己的 Qt widget（不同結構）；兩者可共用的是**那個 mpl 核心**，非整個 widget。
- **INTERACTIVE 跑主線程、不走 worker（用戶定錨）**：無重計算（畫 map + 算 seed 都快）→ 不用 `AnalyzeWorker`/QThread/`routing_scope`；主線程同步掛 canvas + 畫 + 掛 picker，互動經 mpl 事件（本就主線程），結果在 finish callback 產生。FIT 維持 worker（有真計算）。analyze-start 邊界依 mode 分派。

**偵察修正（兩個關鍵，修正我對話中的話）**：

1. **fluxdep-gui 已有整套 Qt 互動選點 UI**（`gui/app/fluxdep/ui/interactive/{base,line_picker,find_points,...}.py`）→ 共用 widget 是「抽出/共用既有」非從零移植。
2. **互動 = 主執行緒 Case B**（自建 `Figure()+FigureCanvasQTAgg`+mpl_connect，範本 = dispersive `ui/tune_canvas.py`），**不是 routing_scope**（那是 worker 畫圖 Case A，只用於 FIT 圖）。`InteractiveLines` 自開 `plt.subplots` → 抽核心時要改成**接外部 axes**。

**偵察事實（實作依據，皆已定位）**：

- `AdapterCapabilities`（`adapter/types.py:65-70`）；`supports_analysis` 讀取 **4 處**：`controller.py:256`（run 完 gate analyze-params init）、`services/tab.py:141`（analyze_params RPC→[]）、`ui/main_window.py:572-579`（UI 可見性）、`:914`（refresh_tab_analyze_form skip）+ base.py docstring。
- ~12 adapter 宣告 capability（4 NONE = 2D scans、其餘 FIT 預設/顯式）。
- FIT analyze 生命週期：`AnalyzeService.start_analyze`→`AnalyzeWorker`(QThread)`with routing_scope(container): adapter.analyze(req)`→`_on_analyze_finished`→`WritebackService.compute_items_for_tab`→`State.update_tab_analyze(result,figure,items)`。
- 第二分頁 UI（`main_window.py:205-270`）：`_analyze_section`(analyze_form)/`analyze_btn`/`writeback_section`/`save_section`；`:572-596` 依 capability setVisible。
- writeback：`adapter.get_writeback_items()`→`MetaDictWriteback(target_name,proposed_value)`→compute/edit/apply（批次寫 md，bump 一次）。flux_dep 現無 writeback。
- agent RPC：`gui_analyze`（analyze.start→operation_id）、`gui_tab_get_analyze_result`（回 `to_summary_dict()` 或 None）。
- `InteractiveLines`：`__init__(signals,dev_values,freqs,flux_half=,flux_int=)` 自開 `plt.subplots`；結果在 `.flux_half/.flux_int` 屬性 + `get_positions(finish=True)` + `finish_interactive()`；Jupyter 耦合 = ipywidgets/IPython/FuncAnimation（chrome 不搬），純 mpl = mpl_connect handler/線狀態/diff_mirror（可抽）。

**分階段（每階段獨立可驗 + commit）**：

- **S1 — capability enum 遷移（純重構、零行為變）— ✅ 完成（commit `bf249fef`）**：加 `AnalysisMode` enum（NONE/FIT/INTERACTIVE，`adapter/types.py`，re-export `adapter/__init__`）；`supports_analysis: bool`→`analysis: AnalysisMode = FIT`（**無 legacy property**）；4 讀取點忠實譯（`controller:256`/`tab:141`/`main_window:572,914`→`analysis is (not) NONE`，非 `is FIT`——INTERACTIVE 特殊分支留 S3）；4 NONE adapter 宣告 `analysis=NONE`、FIT 用新預設；測試（main_window_ui/controller/adapter/toolchain）遷移（helper 保留 bool 便利、構造處映射 enum）。`tests/gui` **1013 pass**、`tests/experiment/v2_gui` 綠、pyright 0、ruff clean。無 wire/RPC 變→版本不變。AI_NOTE 同步。
- **S2 — 抽 `TwoLinePicker` 到 analysis 層（非 gui）— ✅ 完成（commit `2e4fefcf`，用戶選 (b)）**：核心**從 fluxdep 既有 `LinePickerWidget` 抽**（比 notebook 版乾淨：draw-on-event 非 FuncAnimation、純 helper 已抽），放 `notebook/analysis/fluxdep/interactive/two_line_picker.py`（吃 host 給的 `figure`+座標+toolbar 動作+`redraw` callback；owns 資料/雙圖/線狀態/drag/swap/auto_align/magnitude/conjugate；無 Qt/ipywidgets）。`fold_initial_lines`/`find_best_mirror_position` 移進核心、line_picker re-export（向後兼容）。**fluxdep `LinePickerWidget` 重構成薄 Qt chrome**（控制項 wire 到核心、redraw=info label+canvas、`get_result`=核心 positions）；行為不變（測試全綠，僅一個 white-box attr 隨狀態移進核心而改）。notebook `InteractiveLines`（FuncAnimation/ipywidgets）**留原樣**（另一範式，日後可改用核心）。新增 headless 核心測試（合成事件）。fluxdep+notebook **230 pass**、pyright 0、ruff clean、無 wire 變。**`InteractiveSession` 協議延到 S3**（與它的 gui-host 消費者一起定，避免先立空契約）。
- **S3 — gui generic 互動 host（Case B 主執行緒、零互動認知）**：契約定案（多輪 grill）——`InteractiveHost` port（`figure`/`redraw`/`run_background(compute,on_done)`，host 實作=Qt canvas+QThreadPool+marshal）+ `InteractiveSession`（`on_press/move/release`/`actions()->[(id,label)]`/`invoke_action(id)`/`info_text()`/`finish()->AnalyzeResultBase`）兩個 Qt-free Protocol 放 `gui/app/main/adapter/`；`adapter.setup_interactive_analysis(req: AnalyzeRequest, host) -> InteractiveSession`（沿用 AnalyzeRequest 帶 run_result+analyze_params 種子+md/ml）。**支援 analyze param**（INTERACTIVE 也有 params form，static 開關如 `force_magnitude` 移進 form、host 保持零互動認知）；session 編排 action（只把重步丟 `host.run_background`、任意點呼 `host.redraw`，顆粒度細）。三層職責：picker 純邏輯/threading 無認知 — session 編排、知互動不知 Qt — host 知 Qt+threading 不知互動。
  - **S3a ✅ 完成（commit `e2b2b7a1`）**：TwoLinePicker 改**被動**（方法只改狀態、不重畫）+ heavy `auto_align` 拆 `compute_aligned_positions`(純)/`apply_positions`(主線程改狀態)；fluxdep widget 改「驅動完自己 `_repaint`」（行為不變）。fluxdep+notebook 232 pass。
  - **S3b/S3c ✅ 完成（commit `746d7d96`）**：兩 Protocol（`InteractiveHost`/`InteractiveSession`，Qt-free，re-export）+ `ExpAdapterProtocol.setup_interactive_analysis` + BaseAdapter raising default；`AnalyzeService.start_interactive`(lease 無 worker)/`finish_interactive`(Done→`session.finish()`→既有 `_on_analyze_finished` 終端)；`Controller.analyze` 分派 FIT→worker / INTERACTIVE→`_start_interactive_analyze`(主線程經 View 掛載)；`RenderHost.mount_interactive_analysis`；`InteractiveAnalysisWidget`（View 的 InteractiveHost：canvas + 從 `actions()` 渲染 generic 鈕 + Done + canvas 事件轉發 + `run_background`=QThreadPool+queued-signal marshal）；`MainWindow.mount_interactive_analysis` 掛進 tab plot_stack。測試：service interactive 3 + widget 7。tests/gui 1023 pass。
- **S4 — adapter 側 ✅ 完成（commit `4b36bb63`）**：`shared/interactive_flux_pick.py`（`FluxPickParams(force_magnitude)`/`FluxPickResult(flx_half/int/period+figure)`/`FluxPickSession` 包 TwoLinePicker、auto_align 走 `host.run_background`/`build_flux_pick_session` 從 md seed）；onetone/flux_dep + twotone/flux_dep 宣告 INTERACTIVE + get_analyze_params(onetone force_magnitude=True / twotone False) + setup_interactive_analysis + get_writeback_items(3 MetaDict)；guide typical_writeback 更新。**端到端接通**（用戶 run→拖線→Done→flx_* 寫回 md）。session 單元測試 7。gui+v2_gui 1173 pass、pyright 0、ruff clean、無 wire 變。
- **S4 — adapter 側（onetone 先）**：`onetone/flux_dep` 宣告 `analysis=INTERACTIVE` + 覆寫 `setup_interactive_analysis`（給定 axes 上 `cast2real_and_norm` 畫 map + `TwoLinePicker(ax, seeds)`，seed = GUI-side 算 dip 極值）+ `get_writeback_items()` 回 flx_half/int/period 三個 `MetaDictWriteback`。`twotone/flux_dep` 沿用骨架（seed 不同、qub arc 較難）。
- **S5 — agent 介面 ✅ 完成（commit `d13911a4`，用戶定錨「analyze 走 run 的 degrade」）**：`gui_analyze` 不再 block，改 `_start_op_with_short_wait`（短等超時降級 pending，同 run/device/connect）；加 `gui_analyze_wait`/`gui_analyze_poll`（鏡像 run_wait/poll，鍵 `analyze:<tab>`）。**不特判 INTERACTIVE**：它「短等絕不 settle」自然 pending、慢 FIT 也受惠；agent 靠 guide 區分「提示用戶 vs 純等」。**無 wire 變**（analyze.start/operation.await/poll 已存在）—— mcp-side degrade + 2 工具。MCP 25→26、GUI 20→21（INTERACTIVE 是可觀測 GUI 變化）、WIRE 不變。description+instructions 全校對。read-only 守（選點一律 user）。測試：degrade-pending + analyze-poll running/finished。
- **S6 — 文件 ✅ 完成**：guide typical_writeback 已於 S4 更新（取代「recorded elsewhere」）；run-measure-gui skill v15→16（loop gui_analyze degrade + 「Interactive analysis」小節：run→analyze pending→提示用戶拖線+Done→`gui_analyze_poll`→`gui_tab_get_analyze_result` 讀 flx_*→`gui_writeback_apply`，選點一律 user），三副本 `sync_skills.sh` 同步；AI_NOTE（services/remote 版本號+analyze.start INTERACTIVE 分派、main 加 INTERACTIVE 契約+生命週期）。

**Phase 145 全綠收尾**：gui 1025 / v2_gui 143 / fluxdep+notebook 232 全 pass，pyright 0，ruff clean。**live MCP smoke 待 `/mcp reconnect measure-gui`**（[[feedback_mcp_fresh_server]]）驗 banner `wire v21; gui code v21, mcp code v26` + 跑 onetone/flux_dep→`gui_analyze` 回 pending→GUI 拖線 Done→`gui_analyze_poll` finished→flx_* 寫回。測試矩陣：TwoLinePicker 核心（合成事件，被動）、FluxPickSession（fake host）、InteractiveAnalysisWidget（actions/Done/event-forward/run_background marshal）、AnalyzeService start/finish_interactive、gui_analyze degrade+poll。

**待用戶定的開放點**：

1. **互動 sub-kind**：現在只做 line-pick；INTERACTIVE 要不要現在帶「哪種互動」（line/point）欄位，還是 hardcode line-pick、出現第二種再加？（傾向後者，YAGNI）。
2. **觸發**：INTERACTIVE run 完**自動掛** picker，還是按鈕/`gui_analyze` 顯式觸發？（FIT 是顯式 analyze；但互動「就是」分析，自動掛較順）。
3. **共用時機**：S2 現在就和 fluxdep-gui 既有 picker 收斂成單一 shared 核心，還是 measure 先自抽、之後再收斂？（fluxdep 版已存在 → 傾向現在就評估直接抽它）。
4. **寫回**：flx_half/int/period 走既有 writeback（`MetaDictWriteback`，可 review/edit/apply）—— 確認採此（vs 直寫 md）。

**明確不做（守邊界）**：不動 Phase 132 出圖路徑（FIT 圖仍 routing_scope）；不做「深度 B 全統一」（figure-to-adapter 取代所有 analyze、反轉 14 adapter）—— 獨立大重構，不綁本 feature；adapter 不碰 Qt。

**scope**：`gui/app/main`（capability 讀取點 / UI host / RPC）、`gui/`（shared `TwoLinePicker`）、`experiment/v2_gui`（adapter 宣告 + 覆寫，屬 gui scope）、`gui/app/fluxdep`（若共用 picker 收斂）、skill/guide。reuse `notebook/.../InteractiveLines` 核心（抽不改）。

## Phase 146：BackgroundService 抽取（ADR-0019 Phase A）— ✅ 完成（commit `897414c2`，與 147 同 commit）

**起因**：interactive 的 `run_background` 引出「off-main 執行該不該收斂成一個 service」討論，多輪 grill 收斂出 **ADR-0019**：Operation = token + opt-in facets（Exclusion/Handle/Progress/Cancel）+ 可插 execution strategy；洞察「async 相對 caller 不相對 thread」（interactive 也是 async job）。

**做法**：抽 `services/background.py` `BackgroundService.submit(work, scopes, *, run_in_pool, on_done, on_error)`（OffMain strategy）。三 per-op QThread worker + 三 runner（`runner.py` RunWorker/AnalyzeWorker/SaveDataWorker）收斂成一個 generic worker、`runner.py` 刪。`OffMainScopes` = 三正交 opt-in scope（`figure_container`=routing+liveplot **同一 facet co-dependent** / `pbar_factory` / `stop_event`）。**cancel 判讀上移 RunService**（持 stop_event 者自判 finished vs cancelled+partial）。interactive widget 經窄 `InteractiveHostEnv` port（Controller 實作、背後 bg pool）委派 run_background、不持自己 QThreadPool（passive host 收窄注入）。run/analyze(FIT)/save build thunk+scopes 後 bg.submit。GUI 21→22、WIRE 不變。

## Phase 147：OperationGate 拆 Exclusion / Handles（ADR-0019 Phase B）— ✅ 完成（commit `897414c2`）

**起因**：gate 把 exclusion 與 async handle 綁死，code 自招「analyze takes a lease only for the async handle」——poll/wait 不該是 lease 的事。三正交 concern（Exclusion / Handle / Execution），消費者集合不同（handle ⊋ exclusion）。

**做法**：`OperationGate` 瘦成純 **Exclusion**（`ensure_can_start`/`register`/`release`/`has_active`/`is_device_mutating`，keyed by token，不再 mint token/持 handle）；新 `services/operation_handles.py` `OperationHandles`（`create(stop_event)->token` mint operation_id + `settle`/`await_outcome`/`poll`/`cancel`/`cancel_all`/`live_count`）成正交 sibling。刪 `OperationLease`、`OperationOutcome` 移去 operation_handles、刪 `OperationKind.ANALYZE`。**analyze/interactive 只拿 Handle 不拿 exclusion**；run/device/connect `ensure_can_start→create→register`，終端 `settle(token)→release(token)`，cancel 走 `handles.cancel`。`ShutdownCoordinator`/`QtShutdownDriver`/`Controller.await_operation` 改吃 `OperationHandles`；`active_operation_count = handles.live_count`（現含 analyze/interactive）。各 service 持 `_active_token:int`（device 另存 `_active_kind`）。**正式 supersede ADR-0003 §一「生命週期綁死」**。GUI 22→23、WIRE 不變。test_operation_gate 拆 + 新 test_operation_handles；shutdown/qt_shutdown_driver 改 Handles。

## Phase 148：device execution 移植到 BackgroundService（ADR-0019 follow-up）— ✅ 完成（commit `326c0cdb`）

**做法**：device 兩個 QThread（`_DeviceCommandWorker`/`_DeviceSetupWorker`）收進 `bg.submit`。connect/disconnect → `_submit_command`（無 scope）；setup → `bg.submit` + **progress scope**（stop_event 由 work closure 捕捉、driver 直接 poll，**非** ActiveTask scope），cancel 判讀進 `DeviceService._on_setup_done`，「setup 可不可取消」改讀 `_active_kind`、刪 `_setup_worker`/`_command_worker` field。至此 run/analyze/save/interactive/**device** 全走同一 OffMain strategy。GUI 23→24、WIRE 不變。worker-level device 測試 → DeviceService-level failure/cancel 測試（走真實 bg+handle 路徑）。

**Phase 146–148 全綠收尾**：gui+v2_gui+fluxdep+notebook **1408 pass**、pyright 0、ruff clean。**live MCP smoke ✅**（banner `wire v21 (mcp==gui); gui code v24, mcp code v26`）：connect→run（bg+pbar+handle poll/wait）→analyze（FIT bg）→save（bg+resolved path）→device connect/setup（bg `_submit_command`+`_on_setup_done` 成功路徑）→device handle poll→`gui_stop`（shutdown 經 `OperationHandles`）皆綠。唯 device setup cancel-relabel 因 FakeDevice ramp 太快無法 live cancel，由單測 `test_device_setup_cancel_emits_setup_cancelled` 覆蓋。

## bug fix：interactive picker live mirror — ✅ 完成（commit `7e2df7d7`）

`TwoLinePicker` 只在 `on_release` 刷新 mirror-loss → 拖線時 mirror 不同步。根因：Phase 145/S2 把 picker 砍成「零 timer/repaint」過頭（原 `InteractiveLines` 用 `FuncAnimation` 500ms debounce 自帶）。修：picker 用 matplotlib **backend-agnostic** `canvas.new_timer`（throttle ~120ms）在被拖線最新位置重算 `diff_mirror` + `draw_idle`，**主線程、不碰 Qt** → measure-gui 與 fluxdep **一次都好**、host/port 不動、不擴 `InteractiveHost` port。off-main auto-align 維持 compute/apply。picker 不再全被動（自有此一互動 repaint）。Qt-free 不變式（adapter 用它）由 matplotlib 的 toolkit-agnostic timer 滿足。

## Phase 149：optional analyze params（`Optional[T]` = 留空/None）— ✅ 完成（commit `89a2ba27`）

**起因**：清 backlog `project_gui_optional_analyze_param`。`twotone/ro_optimize/length` 的 `t0: Optional[float]`（None=raw SNR max、數值=懲罰長讀出）框架原 `raise TypeError` 不支援。

**做法**：`_resolve_field_info` 認 `Optional[T]`(=`Union[T,None]`)→ strip None、flag `optional`、bare=T;`describe_analyze_params` 標 `optional:true`;`reconstruct`/`_coerce` 對 optional 的 None pass-through。form **複用既有 optional-scalar widget**（`make_value_widget(optional=True)` = QLineEdit 空「(none)」=None + numeric validator，**同 cfg form 的 ADR-0010**，非新 checkbox —— converge to existing abstraction）。adapter `length` 暴露 `t0` + guide 文案更新。**agent 端零 code**：`_h_analyze_start` 的 `dataclasses.replace(ap, **updates)` 本就吃 None，agent 從 `analyze_spec` 看 `optional:true` 傳 `null`。**WIRE 21 不變**（mcp verbatim 轉發 spec/updates 不解讀 optional）、GUI 24→25。測試:resolve/describe/reconstruct + form blank↔value round-trip。

## Phase 150：session-core 共用層抽取 + autofluxdep 複用 + Phase B（cfg-driven 模擬）— ✅ 完成（gui2，`3b89bacf`…`2e7846f0`）

**起因**：第二個 measurement-session app `autofluxdep`（拿某 flux 量到的 base context 衍生其他 flux 的掃描、自動跑 flux sweep）要**整套複用** measure 的「量測 session core」（context 系統 + SoC 連線 + 多 device + setup/device/inspect/predictor dialog），否則大量重抄 + 長期漂移。承 Phase 133（兩 GUI 搬進 `app/`）。詳細子計劃 `session_core_extraction.md` + `autofluxdep_phase_b.md`；跨模組決策見 **ADR-0020**。

**S1–S5：抽 `gui/session/` 共用層（measure 零行為變、wire 不變）**——值型別（ExpContext/Soc*）、`SessionState` slice、session events、connection/context/device/startup 服務 + `build_session_services`、共用 dialog（setup/device/predictor/inspect_base，吃 `SessionControllerPort`）。**app-local vs shared 邊界**：`OperationGate`（衝突 policy + app 自己的 RUN kind）、`BackgroundService`（measure 帶 QtLivePlot facet / autofluxdep 瘦版無 figure routing）各 app 自持；`OperationHandles`/`ProgressService`/`IOManager`/`QtProgressTransport` promote 成共用；service 只依窄 port（ExclusionGate/BackgroundExecutor/ProgressHub/ProjectIOPort），app 注入具體。`SessionControllerPort` 是共用 dialog 的 Controller 契約，各 app Controller **結構上**實作（pyright 在各自 dialog call site 驗 conformance）。

**autofluxdep 複用**：`AutoFluxDepState(SessionState)`；Controller 組 `build_session_services` + 注入 app-local infra + 實作 port；run 讀 `exp_context`（SSOT），**退役**自己的 `SetupResources`/`setup()`，改用共用 setup/device/predictor dialog。headless 測試轉 async connect（package-level `qapp` autouse + `connect_mock` QEventLoop，與 measure 一致，不留同步 setup() 旁路）。node_list 加 Devices/Predictor button + **flux-source picker**（從連上的 device 選 flux 源、標 unit）；`run_app` 注入 repo-root `project_root`。

**Phase B（cfg-driven 模擬，整段不碰真硬體）**：7 個 node（qubit_freq/lenrabi/ro_optimize/t1/t2ramsey/t2echo/mist）各 `Builder.make_cfg`（**在 `produce` 跑**，snapshot 在手——決策 A + D1）把當前 context 的 ml/md + drive 設定頭 params lower 成真 cfg（經 `ml.make_cfg`），再**從 cfg 驅動模擬** acquire（無硬體、無 mock 偵測）；空-ml/demo context fallback 純 snapshot 模擬（既有 run 測試不變）。**真 acquire**（per-point `setup_devices` + program `acquire` + `cfg.dev` 寫入 + mock/real 分支）**延未來獨立 phase**——屆時只換 `produce` 的合成那段。B-3 其餘 6 node 用 Workflow fan-out 產出、**逐 node 在主樹自驗整合**（worktree 的 editable-install 解析使 agent 自驗不可信，故內容逐一 pyright+full suite+diff 檢查 sim 邏輯才 commit）。

**收尾**：pyright 四樹 0 error、`tests/gui` + `tests/autofluxdep_gui` **1195 pass**、ruff clean。文檔同步（`gui/session` + `gui/app/autofluxdep` AI_NOTE、autofluxdep CONTEXT.md「Run path」、measure AI_NOTE、ADR-0020 + README）。**真 acquire / agent-RPC 接入 autofluxdep 未做**（本 phase 全模擬、無 remote layer）。

## Phase 151：gui 模塊 code-review 修復（4 findings）— ✅ 完成（commit `a8ea8e86`）

**起因**：全模塊 7-angle code review（2026-06-10）確認 4 個 findings：1 正確性 bug + 1 契約缺口 + 2 處逐字重複。

**F1（正確性）`gui/remote/framing.py` encode_line 大小檢查不對稱**：`len(payload)` 量 str 字元數、在 `.encode("utf-8")` 之前，而 `decode_line` 量位元組數 —— 多位元組字元（中文、例外訊息）可繞過出站上限、被對端拒收。**修法**：先 encode 再以位元組數檢查（`data = payload.encode("utf-8"); if len(data) > MAX_LINE_BYTES: raise`），與 decode 對稱。純內部防線修正，**WIRE 不變**。

**F2（契約）`app/main/services/remote/method_specs.py:292-293` device.connect/disconnect 空 ParamSpec**：handler 實吃 `type_name/name/address/remember`（connect）、`name/remember`（disconnect），但 spec 未宣告 → `control_service` 對空 spec 直接 pass-through 跳過 `validate_params`，且 spec 作為參數文件失真。**修法**：補宣告 ParamSpec（`remember` 為 optional BOOLEAN default True，對齊 `optional_bool(params, "remember", True)` 與 mcp 端「有才傳」行為）；需新增 `_bool_default` helper（JsonType.BOOLEAN 已支援）。語義對正確呼叫者不變 → **WIRE 21 不變**、GUI_VERSION bump；依慣例校對 mcp 端 description（mcp/ 在 scope 外，只 audit 不改，發現不符回報）。

**F3（重複）`gui/session/services/device.py:420-421 vs 441-442` YOKOGS200 mode→unit 逐字重複**：lenient `get_device_unit` 與 strict `get_device_unit_strict` 各有一份 `mode=="voltage"→"V" else "A"`。**修法**：抽單一 module-level helper（如 `_mode_dependent_unit(dev) -> Optional[str]`），兩函式先問它、None 再走各自 fallback/whitelist 路徑。下沉到 `device/` 層的方案在 gui scope 外，不做。

**F4（重複）fluxdep vs dispersive `services/remote/dispatch.py` 三 handler 逐字重複**：`_h_project_info` payload、`_h_state_check` 的 has_project（DEFAULT_CHIP/DEFAULT_QUBIT placeholder 判斷）兩 app word-for-word。**修法**：`gui/project.py`（ProjectInfo/DEFAULT_* 的 owner）新增 `project_info_payload(project)` 與 `is_real_project(project)` 兩 helper，兩 app handler 改為呼叫（converge to existing abstraction）；`_h_resources_versions` 本身一行、不值得抽，保留。

**驗收**：framing 多位元組邊界 regression test；device.connect/disconnect 錯型參數被 INVALID_PARAMS 拒絕的測試；unit helper 與 project helper 單元測試；pyright 0 error、全 `tests/gui*` pass、ruff（`check --select I --fix` + `format`）clean；同步相關 AI_NOTE.md。實作委派 sonnet sub-agent。

**結果**：四修全落地（framing encode-then-check、specs + `_bool_default` + GUI 26→27 changelog、`_mode_dependent_unit` helper、`gui/project.py` 新增 `project_info_payload`/`is_real_project` 兩 app 收斂）。測試 2549→2570（+21）、pyright 0 error、ruff clean。mcp audit 發現的 `gui_device_connect` `remember` description 錯標 "default false"（實際 True）已獲授權一併修正（含 prose 與 inputSchema 兩處；disconnect 端原本就正確）。app/main 走 `ctrl.has_project()` 抽象不同、未收斂（正確）。全數入 commit `a8ea8e86`。

## Phase 152：service 依賴介面化 + event 定義歸 domain — ✅ 完成（commit `6056fe22`）

**起因**：用戶提案「service 模塊化走 port + event 兩層化（domain 定義、app 組裝）」。評估結論（2026-06-10 兩輪調查）：(1) session/services 已是 port-only 標準、fluxdep/dispersive 的 State-only 模式符合 ADR-0004 Query 不需動；**真缺口在 app/main/services 的 concrete 依賴**。(2)「domain 定義事件、app 組裝」**已是現行架構**（session/events.py 即範例、bus 以 payload type 為 key），不改 bus 機制、不把 key 換 enum（會失去 handler 型別綁定）；要補的是 app/main 內 tab/run 兩個 domain 的事件定義仍混在一個 flat `GuiEvent`。呼叫方向不變（ctrl→service + bus reaction，ADR-0004），State 不 port 化，資料夾不做 vertical slice（ADR-0005 M5 既決）。

**A. 依賴介面化（app/main/services，行為零變）**
- A1 重標注為既有 port：`RunService`/`AnalyzeService`/`SaveService` 的 `BackgroundService` 參數 →`BackgroundExecutor`；`RunService` 的 `OperationGate` →`ExclusionGate`；`ProgressService` →`ProgressHub`。實作前先驗 method surface：若 service 用到超出 port 的方法（如 liveplot facet），按 Fast Fail 擴 port 或開 app-local 窄 port，**不硬塞**，超出即回報。
- A2 新 port（進 `services/ports.py`，維持集中式慣例）：`WritebackLifecyclePort`（`teardown_tab_items`/`compute_items_for_tab`，Run+Analyze 消費）、`CfgEditorPort`（`open_seeded`/`teardown`/`get_root`，Writeback 消費）。EventBus 與 State 維持 concrete（前者本身是共用抽象、後者是 ADR-0004 Query SSOT）。
- A3 補 AST gate：`test_app_service_decoupling` 的 `_APP_SERVICE_MODULES` 缺 `background`/`operation_gate`/`cfg_editor`；補入並加「runtime import concrete infra 類但已有對應 port」檢查（TYPE_CHECKING 註解 import 仍允許）。

**B. event 定義歸 domain（wire name 全不變 → WIRE/mcp 不動）**
- B1 `app/main/event_bus.py` 拆 `app/main/events/{tab.py, run.py}`：`TabEvent`（TAB_ADDED/CLOSED/CONTENT_CHANGED/INTERACTION_CHANGED + 4 payload）與 `RunEvent`（RUN_STARTED/FINISHED + 2 payload）。payload 模組是 leaf、無循環風險（已驗）；import 扇出 ~40-50 行（emitters: workspace/run/analyze/save/writeback/controller；subscribers: main_window/remote events；tests）。舊 `event_bus.py` 刪除不留 re-export（no-legacy 規則）。controller 直接 emit 的站點（TabContentChanged ×2、TabInteractionChanged ×2）**本 phase 不搬**（coordinator emit 合法，搬動屬行為調整）。
- B2 autofluxdep 泛名 `EventType` 改名（候選 `AutoFluxDepEvent`；是否進一步拆 Workflow/Run 兩 enum 待用戶定）；修 `gui/remote/control_service.py:27` 過時 docstring（仍稱 measure 用自製 enum bus）。fluxdep/dispersive 的 event 檔已是單一 domain，不動。
- B3 文檔：ADR 補一條「event ownership convention：domain module 擁有 enum+payload 定義、app 在 bus/EVENT_SERIALIZERS 層組裝；port 集中於各層 ports.py」（掛 ADR-0004/0005 系譜）；AI_NOTE 同步。

**驗收**：行為零變（全 `tests/gui*` 2570 pass 不增不減邏輯、僅 import/型別/新 port 測試）；pyright 0 error；AST gate 新規則有紅→綠測試；ruff clean。

**決策**：(D1，已定) autofluxdep 拆兩 domain enum（workflow-editing / run lifecycle），並比照 main 採 `events/` package 慣例；(D2) A1 若發現 port surface 不足，實作端停下回報、由用戶定擴共用 port 或開 main-local port——實作未觸發（既有 port surface 全數吻合）。

**結果**：全項落地（47 檔 + 2 新 events/ package）。A1 重標注全成（Run/Analyze/Save → `BackgroundExecutor`/`ExclusionGate`/`ProgressHub`）；A2 `WritebackLifecyclePort` + `CfgEditorPort` 進 `services/ports.py`；A3 AST gate 補 `background`/`operation_gate`/`cfg_editor` + `_CONCRETE_INFRA_NAMES` denylist（red proof 驗證過：違規 import 會 fail）。B1/B2 兩 app event 拆 domain module（`main/events/{tab,run}.py`、`autofluxdep/events/{workflow,run}.py`），舊 event_bus.py 刪除、`GuiEvent`/`EventType` 絕跡；wire name 全集由新 wire-name lock 測試鎖定（14 名）；B3 docstring 修正。文檔：ADR-0021（event ownership convention）+ README 索引 + remote/session AI_NOTE 同步。驗證：全套 2573 pass（baseline 2570 + 3 新測）×2 輪、pyright gui 樹 0 error（僅剩 2 個 scope 外既有錯誤：tests/experiment/v2_gui、tests/mcp）、ruff clean。**插曲**：實作 agent 中途被停，工作經 stash 完整恢復；其回報的「device 測試合跑 segfault」在最終狀態兩輪全跑皆不重現（baseline 亦乾淨），判定為其中間態程式碼或環境暫態，順序耦合根治列 Phase 153。

## Phase 153：測試套件速度優化 — ✅ 完成（commit `17abcf32`）

**起因**：全套 `tests/` 跑一輪 ~109s（2573 tests），拖慢每個 phase 的收尾驗證迴圈。

**Profile（2026-06-10，8 核機）**：收集僅 3.5s 非瓶頸；`tests/simulate` 59 tests 佔 **81.7s（75%）**——fluxonium scqubits 對照組單顆 2–6s、CPU-bound、彼此獨立；`tests/gui`+`tests/autofluxdep_gui` 1219 tests 僅 10.0s（GUI 不是問題，最慢單測 0.83s）；其餘 ~1295 tests 約 17s。`qapp` 已 session-scope、xdist 未安裝。

**方案**：pytest-xdist `-n auto` 並行（**不改任何測試語意**，預期 109s→~25s）。(1) `pyproject.toml` optional-dependencies 加 `pytest-xdist`；(2) 全套 `-n auto` 驗證——每個 worker 是獨立 process 各持 offscreen QApplication，理論安全；若 Qt 測試在 xdist 下不穩，fallback 拆兩段（CPU 套件並行 + Qt 套件序列）並提供單一入口腳本；(3) 連跑 ≥3 輪驗穩定性，xdist 的分發順序天然擾動執行序，正好曝露 Phase 152 報告過的 device 合跑 segfault 類順序耦合——若重現則根治（測試獨立性修復優先於跳過）；(4) 文檔化標準跑法。**不做**：scqubits 對照值 pin 成快照（換速度損失對照活庫的保真，需另議）、pytest-randomly 常駐隨機化。

**驗收**：全套 wall time ≤35s、≥3 輪連跑零 flake、序列模式（不帶 -n）仍全綠不退化、pyright/ruff clean。

**結果**：標準跑法 `pytest tests/ -n auto` = **~14.5s**（109s→7.4×，遠超目標；BLAS 在 xdist worker 內由 `tests/conftest.py` 頂部自動 pin 至 1 線程——未 pin 時 8 worker × 多線程 OpenBLAS 搶核只得 70.5s）。serial 模式不變（保留多線程 BLAS 覆蓋，`test_pinned_matches_blas_multithreaded` 的多線程面僅在 serial 行使、已註記）。

**順帶根治了潛伏的 Qt segfault（Phase 152 device 合跑 segfault 同根因）**：測試把持有 `BackgroundRunner` 的 QObject 當 local 丟棄，queued 跨線程 done-delivery（`QMetaCallEvent`）還在主線程佇列；物件 GC 後由下一個泵事件的測試引爆——crash 站點漂移是指紋。修法（opus investigator 確認，xdist 0/25、加壓 serial 0/18）：(1) production `BackgroundRunner` 加 `_pool_signals` 追蹤 + `quiesce(timeout_ms)`（join pool/worker + flush queued delivery），measure/autofluxdep BackgroundService 委派之；(2) 持有者 fixture teardown quiesce（test_controller `cf`、test_device_manager `_make_real_svc`、autofluxdep `app` fixture、fluxdep find_points/selector）；(3) **production close-path 同步堵**：dispersive MainWindow.closeEvent、fluxdep MainWindow.closeEvent + `_clear_editor`、`AnalyzePanelWidget.quiesce()`（含內嵌 SelectorWidget）——child widget closeEvent 不會 fire，全部掛在真正執行的 host 路徑。已 revert 的歧途：per-test `gc.collect()`（22s→104s 效能災難）、sleep 同步。

**驗證**：3+1 輪全套 `-n auto` 全綠（14.4–14.8s）、Qt 套件 serial 全綠、pyright 0 新 error、ruff clean。文檔：tests/AI_NOTE.md（標準跑法 + quiesce teardown 慣例）。**教訓**：間歇 crash 不要拿全套（110s/輪）當重現器——先縮小重現迴圈再迭代；sub-agent 因此燒掉 2.5h。

## Phase 154：維護小掃除（5 件小瑕疵）— ✅ 完成（commit `2ed4d403`）

**起因**：用戶指派清掉累積的小尾巴（2026-06-10 盤點）。predictor 成功路徑驗證不在內（等用戶提供檔案）。

- **A. pyright 清零**（scope 外既有 2 錯）：`tests/experiment/v2_gui/adapters/shared/test_cfg_builder.py:208`（EvalValue 不可指派給 `sweep(start: float)`）、`tests/mcp/agent_memory/test_store.py:59`（`'t1'` 字串 vs `exp_type: List[str]`）。修測試端型別（若是被測 API 簽名不合理則回報不擅動）。
- **B. smoke 0602 瑕疵 1**：`context_new` unit="mA" 裸 KeyError。**先查是否已過時**——context flux unit 現由 bind_device 推導（YOKOGS200→A/V、FakeDevice→none），舊「直接傳 unit」路徑可能已不存在；若 KeyError 路徑仍在則改為明確 INVALID_PARAMS/DeviceRegistrationError，已消失則記錄並更新 memory。
- **C. smoke 0602 瑕疵 2**：startup（setup）dialog 表單草稿殘留——重開 dialog 顯示上次未 apply 的草稿而非 active project 現值。修為每次開啟從 State 重置表單。
- **D. quiesce 理論缺口補齊**：fluxdep/dispersive controller 層測試 fixture 持有會跑真 worker 的 BackgroundRunner/Service 而無 teardown quiesce——按 tests/AI_NOTE 慣例補上（對齊 tests/gui 的 ControllerFixture pattern）。
- **E. Phase 133 header 對時** — ✅ 已改（本檔）。

**驗收**：pyright 全 repo 0 error；`pytest tests/ -n auto` 全綠；B/C 各有 regression test（C：開 dialog 兩次、中間改值不 apply，第二次開啟反映 State）；ruff clean；同步 AI_NOTE/memory。MCP 端版本同步（GUI_VERSION 27）需重啟 MCP server 後另行確認。

**結果**：**pyright 全 repo 首次 0 error**（A：兩處 `type: ignore` 錯位修正、production 簽名皆正確）。B 判定**已過時**：WIRE v19 起 context.new 無 unit 參數、unit 由 bind_device 經 `get_device_unit_strict()` whitelist 推導，裸 KeyError 結構上不可能（memory 已更新結案）。C 修復：`SetupDialog.showEvent` 每次 programmatic show 從 `startup_prefs` 重 seed + regression test；orchestrator 復驗時加固 **spontaneous show（最小化還原）不洗草稿** + 簽名改 `Optional[QShowEvent]`。D audit 結論：fluxdep/dispersive 所有跑真 worker 的 fixture 已有 quiesce（7 處清單在 agent 報告），零新增（無死碼）。E header 對時完成。驗證：2574 passed（+1 test）/ 14.4s、ruff clean。**收尾 smoke（MCP server 重啟後）**：handshake `wire v21 (mcp==gui)` + gui code v27 確認生效；device connect/disconnect 描述與 roundtrip 正確；**predictor 成功路徑首次全綠**（用戶提供 `result/Q3_2D[2]/Q1/params.json` 副本：load/info/predict×3（5825/5995/5593 MHz，物理合理）/clear）——0602 smoke 最後一個未測項關閉。

## 已知缺口與風險

（目前無 open 缺口；以下為已修復風險的歷史紀錄。）

| 項目 | 現況 |
| --- | --- |
| `gui` → `experiment/v2_gui` 反向依賴 | Phase 76 已修復；**Phase 131 再強化**：`run_app` 不再 import v2_gui registry，接線上提到 `run_gui.py`（composition root）；`require_soc_handles()` 屬 framework request validation |
| Save path query 副作用 | Phase 73 已修復；query pure，actual save command 才建立 parent directories |
| Sweep arithmetic 位於 Widget | Phase 74 已修復；canonical transform 位於 pure `SweepEditor` / `SweepLiveField` |
| version guard `device:*` glob 對「集合新增」失明 | **已修（Phase 102，gui2:4ff8394d）**：新增 `devices:__set__` 集合基數 version key，`State.put_device`（新名）/`remove_device` bump，`run.start` 的 `_GUARD_DEPS` 聲明之。agent 讀完 versions 後 user 新增/移除 run 依賴的設備 → 基數 key 移動 → run 被 guard 偵測。狀態/info/remember 編輯既有成員不動基數（走各自 `device:<name>` key）。比「集合基數＋成員」侵入式 wire 改動更輕：單一 mid-grained key，沿用既有 opt-in 協議 |

## 錯誤紀錄（保留教訓型）

| 錯誤 | 解法 |
| --- | --- |
| MCP 索引回傳工作樹不存在的 `tests/gui/conftest.py::_teardown_tqdm_monitor` | 以工作樹與 runtime 測試為權威 |
| `test_io_device.py:208` 未 narrow `DeviceInfo` union | 加入 `FakeDeviceInfo` runtime/type narrowing |
| Phase 71 initial connect failure phantom snapshot + pre-worker event failure lease leak | 收緊 rollback/release path 並加入 regression tests |

## 備註

- 本檔僅保留「目前仍可信」的 phase、決策與風險；舊過渡 API、已失效事件名、被推翻的驗收條件均移除
- 詳細設計契約見 `architecture.md`；實作 cheat sheet 見 `lib/zcu_tools/gui/AI_NOTE.md`
