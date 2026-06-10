# Session-core extraction:measure-gui ↔ autofluxdep 共用量測 session core

**Status:** S1 `3b89bacf`；S2a `66a5cf1e`；S2b `c463bc0a`；S2c-1(connection) `cdbe3361`；S2c-2(device, approach B) `8a65951b`；S2c-3(context P-b façade + apply_ml_writes callback) `b29127a4`；S2c-4(build_session_services) `e74193bd`；S2d(persistence 拆 + StartupService 搬) `720a32c5`。**🎉 S2 全完成。** S3-a(progress primitives→session) `a71eb107`；S3-b(SessionControllerPort + setup_dialog→session/ui) `1b165d50`；S3-c(device_dialog→session/ui + TrimDoubleSpinBox→gui/widgets) `68a7ed97`；S3-d(inspect 拆 base/measure) `590ea8e8`。**🎉 S3 全完成（四 dialog/widget 全進 session core，measure 零行為變）。**
S4-pre(ProgressService+IOManager→session) `8e9c308f`；S4-a(AutoFluxDepState(SessionState)) `b17564ac`；S4-b infra(QtProgressTransport→session/adapters) `5735f184`；**S4-b core(autofluxdep app-local OperationGate+瘦 BackgroundService `f327d006`；Controller 組 build_session_services+實作 SessionControllerPort, additive `c24a3f76`)。** **S4-b retire part 1(run/_build_tools 讀 exp_context、predictor 調和+run_predictor stash、刪 SetupResources、has_setup→has_soc `c057f409`)。** 用戶定**方案 B**（headless test 收斂 async connect）。**S4-c C1(測試轉 connect_mock + qapp autouse `cb9fa824`)；C2(切共用 setup_dialog + 退 setup()/舊 dialog，Fork1→(b) predictor UI 暫緩、Fork2→採完整 dialog `0f1b44ca`)。** **🎉 S4 完成**（autofluxdep 組共用 session services + 實作 SessionControllerPort + run 讀 exp_context + 用共用 setup_dialog）。**S5 收尾(用戶選 B)**：device_dialog 整合 done(`7a552503` node_list Devices button→共用 DeviceDialog)；互動 device setup 那半完成。**🎉 session-core 抽取/複用主線 S1–S5 reshape 全達標**(session 核心進 gui/session/、measure 零行為變、autofluxdep 完全複用)。**Phase B 進行中（計劃 `autofluxdep_phase_b.md`；用戶定 A=cfg 在 Builder/D1=produce 跑/B2 輕量；「先全 synthetic，真 acquire 最後」）**：B-0 接線(`e6ec2c88`)、B-1 qubit_freq cfg_maker(`73415b2d`)、B-2 qubit_freq cfg-driven sim(`96711cfd`)、B-4 PredictorDialog 共用化(`790f08f6`)、**B-3 其餘 6 node cfg_maker+cfg-driven sim(Workflow fan-out `fa3b9a07`..`1456ebb3`)** 完成。**用戶定整 Phase B 不碰真 acquire、produce 一律模擬(cfg 驅動)、無 mock 偵測**。B-5 flux-source picker + B-6 project_root(`2e7846f0`)。**🎉 Phase B(simulated)完成**：7 node 全 cfg-driven sim + 共用 setup/device/predictor dialog + flux-source picker + project_root；pyright 0 + 1195 綠。真 acquire(setup_devices+TwoToneProgram.acquire+cfg.dev+mock/real 分支)延未來獨立 phase(用戶定 Phase B 全模擬)。 **Branch:** gui2。
**S2c 全完成**：connection/device/context 三服務 + build_session_services 都進 session/，session 服務層成形可複用。StartupService 仍 measure（依賴 persistence_types，待 S2d 一起搬）。
S2 拆 sub-batch：S2a✅ ExclusionGate port+OperationKind 拆；S2b✅ SessionState slice(State 子類化、零 ripple)；S2c✅(1 connection / 2 device via ports BackgroundExecutor/ProgressHub) / 待(3 ContextService P-b façade + ProjectIO/ContextRead port / 4 startup + build_session_services)；S2d persistence 拆。
**教訓**：搬 module 必跑 pytest collection（pyright 漏抓 `__init__` re-export 斷裂）；高風險搬移自己做別委派（device 委派失敗過）。
本檔是 gui 主計劃(`task_plans/gui/task_plan.md` Phase 133 步驟 B+)的延伸子計劃。

## 目標

`autofluxdep`(`lib/zcu_tools/gui/app/autofluxdep/`,**開發中**,功能尚未寫全)要**複用 measure-gui 的「量測 session core」**:context 系統(MetaDict/ModuleLibrary)+ SoC 連線 + **多 device** 管理 + `setup_dialog`/`device_dialog`。各 app 疊自己的主界面(autofluxdep = flux_dev 選擇 + node-sweep 編排;measure = tabs/experiments)。

autofluxdep 的本質:拿 measure-gui 在**某個特定 flux 量到的 context**當 base,**衍生出其他 flux 值的掃描參數**,自動跑一段 flux sweep 的量測「nodes」。初始化要激活 context + 連 SoC;device 要支援 flux_dev 以外的儀器。

#1–#6 的共用層重構(`BackgroundRunner`/`RemoteControlServiceBase`/`ProjectInfo`/`gui/widgets`/`run_qt_app`/`BaseController`/`BaseEventBus`,見 memory `project_gui_shared_layer_batch`)就是為這步鋪路;這是收口。

## 鎖定決策(用戶定 + 我受權判定)

1. **autofluxdep 必須有 context 系統**(用戶定):它基於 base context 衍生不同 flux 的掃描參數。
2. **device 要支援多 device**(用戶定):flux_dev 以外可能有其他儀器要設。
3. **OperationGate 各 app 自持,不抽共用 singleton**(用戶傾向 + 我判定):measure/autofluxdep 是**獨立進程**,無跨 app 共享需求;gate 的 `OperationKind`+互斥 policy 帶 app 味。→ **`OperationHandles` 抽共用**(純 token mint/settle/await/poll/cancel,零 kind);**`OperationGate` 各 app 自持**;共用 `ConnectionService`/`DeviceService` 依賴**窄 exclusion port**(`ensure_can_start/register/release/has_active`),各 app 注入自己的 gate。device/connect kind 屬 session-core(隨 service)、run/sweep kind 各 app 自加。
4. **autofluxdep 編輯 ml:先不支援(measure-only),按難度延後**(用戶授權按難度):ml-edit(`_MlModifyDialog`/`_MlCreateDialog`)會拖整套 `CfgEditorService`+`CfgFormWidget`+`RoleCatalog`(真正 experiment-coupled)。autofluxdep 是消費 base context(ml 已填),八成不需自造 ml module。→ 共用 inspect = **md 編輯 + ml 檢視**;ml-edit 留 measure-only 擴充。
5. **共用 dialog 落點 `gui/session/ui/`**(session-coupled,非 `gui/widgets/` 純 dumb widget bucket)。
6. **VersionTable:單一共用實例**(兩 slice 同表 bump,保留跨 slice guard)。

## 投查結論(session-core boundary,2026-06-09 兩 opus agent 勘查)

### 乾淨可抽
- **dialog 耦合面乾淨**:`setup_dialog`(call 14 session-core ctrl 方法、訂 Context/Soc/Device event)+ `device_dialog`(call 16 session-core 方法、訂 3 device event)**零 tab/run/analyze 觸碰**;device 的 per-device 設定面板是**手刻 QFormLayout**(`_FakeDevicePanel`/`_YOKOGS200Panel`/`_SGS100APanel`/`_MemoryDevicePanel`,走 `BaseDeviceInfo.with_updates`),**不碰 cfg-editor**。
- **已共用/app-agnostic**:`BackgroundService`(已組合共用 `gui/background.BackgroundRunner`)、`OperationGate`/`OperationHandles`/`ProgressService`(純)、`BaseEventBus`(#6)。
- **services 乾淨**:`ConnectionService`/`ContextService`/`DeviceService` 無 experiment 觸達(只需 type-import 重導)。
- **State slice 可分**:`ExpContext`(md/ml/soc/soccfg/predictor/chip/qub/res/paths/readiness)+ `devices: dict[str,DeviceState]` + `startup_prefs` + 對應 version keys(`context`/`soc`/`device:<name>`/`devices:__set__`)—— `ExpContext` 零 tab 內容,與 tab slice 乾淨可分。
- **event payloads**:session-core = `Md/Ml/ContextSwitched/Soc/Predictor/Device{Changed,SetupStarted,SetupFinished}Payload`;experiment-only = `Tab*/Run*`。BaseEventBus 是 **payload-type-key**,移 payload 機械乾淨(各 app 自留 enum、import 共用 payload,`EVENT: ClassVar` 需一個 home enum)。

### 責任錯置(pre-refactor,先正位再抽)
- **P-a**:`ExpContext`/`SocHandle`/`SocCfgHandle`/`ContextReadiness` 錯放 `lib/zcu_tools/gui/app/main/adapter/types.py`(它們是 session-core 值型別,非 adapter type)→ 移共用層。`CfgSchema` 留 experiment。
- **P-b**:`ContextService` 的 ml-write 經 **CfgSchema lowering**(`set_ml_module_from_schema`/`set_ml_waveform_from_schema`/`apply_writes`,`services/context.py:251-325`)把乾淨的 ContextService 綁上 experiment cfg-tree。raw md(`set_md_attr`/`del_md_attr`)+ context-switch(`use/new_context`)乾淨 → 拆:raw + context-switch + ml-register(共用)/ CfgSchema-lowering façade(measure-side)。
- **P-c**:`AppPersistedState`/`StartupService` 跨 session(startup-prefs+devices)與 experiment(tabs)→ 拆 persistence projection。
- `build_app_services` 一把抓 15 service → 拆 `build_session_services` + experiment 接線。

### 其他糾纏
- **inspect_dialog 半 session 半 experiment**:md tab + ml 檢視乾淨;ml edit/create(`_MlModifyDialog`/`_MlCreateDialog`)拖 cfg-editor → 決策 4:ml-edit 留 measure。
- **單一 gate/handles 跨 run+device+connect**(measure 內,ADR-0019)→ 決策 3:handles 共用、gate 各 app 自持 + 注入 port。
- **VersionTable 一張表跨兩 slice** → 決策 6:單一共用實例。
- **autofluxdep 的 predictor ≠ measure 的 predictor**:autofluxdep `tools.Predictor`(`FluxoniumPredictorAdapter`,**自適應掃描引擎** + `calibrate` 閉環)vs measure `ConnectionService` 的 `FluxoniumPredictor`(freq 預測 dialog)。**勿混同**;setup 時載 params.json 的 `FluxoniumPredictor` 概念兩邊重疊(可共用 loading),但 autofluxdep 的自適應「use」是它自己的。

### ⚠️ Load-bearing 設計點 + 解法(per-point flux-set)
autofluxdep sweep **每個 flux 點要同步設 flux device 值,在 run worker thread 裡**;但共用 `DeviceService` 把 device mutation 當 gated/async/主線程終結 operation。
**解法(已定)**:**「互動設定」與「程式化掃描驅動」分離** —— 共用 `DeviceService`/`device_dialog` 只管**跑前互動設定**;**sweep 跑起後 flux device 由 worker 直驅**(走 `experiment/v2` runner 的 `setup_devices(cfg)` 慣例,**不經 DeviceService**),由 autofluxdep 自己的 `OperationGate` 確保 run 中互動 device 操作被排除。這跟 measure 一致(measure 的 run 也程式化驅動 dev sweep、不經 DeviceService)。衝突消解(config 與 run 是兩階段)。

## 落點 `gui/session/`(新概念 package,對標 `gui/remote`/`gui/plotting`)

```
gui/session/
  types.py     (P-a: ExpContext/SocHandle/SocCfgHandle/ContextReadiness)
  events.py    (session payloads + SessionEvent enum)
  state.py     (SessionState: exp_context+devices+startup_prefs+version keys; DeviceState/Status)
  ports.py     (ExclusionGate / DriverFactory / ProjectIO / ProgressTransport ports)
  operation_handles.py   (shared, pure — 從 main 移)
  controller_port.py     (SessionControllerPort — dialog 依賴的 Protocol)
  services/{connection,context,device,startup}.py + build_session_services
  ui/{setup_dialog, device_dialog, inspect_base}.py
```

## Phases(每 phase 獨立綠 + 可 commit;S1–S3 對 measure 零行為變)

| Phase | 內容 | 行為變? | 風險 |
|---|---|---|---|
| **S1** ✅ | **值型別+payloads 正位**:P-a 移 `ExpContext` 等出 `adapter/`;session payloads + `SessionEvent` enum 進 `gui/session/events.py`;`OperationHandles` 移共用。**adapter 邊界 principled re-export(選項1，非全重導):ExpContext/Soc 是 ExpAdapterProtocol 契約詞彙故 adapter re-export，blast radius 圈在 gui/ 內**;S2 要抽的 session-core 檔(state+connection/context/guard/ports)已直接指向 session.types。已 commit `3b89bacf` | 零(純搬+import) | 低 |
| **S2** | **抽 session services + SessionState**:`ConnectionService`/`ContextService`(P-b 拆 ml-write)/`DeviceService` + `SessionState` slice → `gui/session/`;`build_session_services(gate_port, handles, …)`;measure State 內嵌 SessionState、Controller 組合共用 session services + 注入自己的 gate;P-c 拆 persistence;拆 `build_app_services`。measure 行為零變 | 零 | **高(核心)** |
| **S3** | **抽 dialog + 定 port**:`SessionControllerPort` Protocol;`setup_dialog`+`device_dialog`+`inspect_base`(md+ml-view)→ `gui/session/ui/` 吃 port;measure Controller 實作 port、ml-edit 留 measure 擴充 | 零 | 中 |
| **S4** | **重塑 autofluxdep core**:State 長出 SessionState;Controller 組合共用 session services + `BaseController` + `run_qt_app`;自持一個 gate;退役 `SetupResources` 一次性模型,改用共用 context/connection/device;接共用 dialog。per-point flux-set 走上面解法 | autofluxdep 變(feature) | **高** |
| **S5** | **autofluxdep 主界面**:node-sweep + **flux_dev 選擇**(從共用 device list 選哪個當 flux 源);setup/device 走共用 dialog;接真 context 後 node cfg-derivation 用真 ml(node 接 `soc.acquire` 的 Phase B/C 可併入或獨立排) | autofluxdep 變 | 中 |

**依賴鏈**:S1→S2→S3 是 measure 內部「零行為變」抽取(獨立 review/commit、隨時可暫停);S4→S5 才動 autofluxdep。S1–S3 即使 autofluxdep 暫不接,也讓共用層成形、measure 程式更清晰。

## autofluxdep 要保留的領域核心(**別動,這是它存在的理由**)

`nodes/builder.py`(Builder/Node/Service/Provider/RunEnv)、`orchestrator.py`(requirement-resolver + `InfoStore`,**非** topo-sort)、`nodes/spec.py`(Dependency/ModuleDep)、`nodes/io.py`(Snapshot/Patch)、`derivation.py`(SmoothingService,跨點平滑)、`tools.py`(自適應 Predictor + calibrate)、`registry.py`、`nodes/*.py`(qubit_freq/lenrabi/ro_optimize/t1/t2ramsey/t2echo/mist)、node-sweep UI(`ui/main_window.py`/`node_list.py`/`node_detail.py`/`param_form.py`)。見 `lib/zcu_tools/gui/app/autofluxdep/CONTEXT.md`(若存在)+ 各 AI_NOTE。

執行期耦合(S4/S5 要接的真實路徑):node 的 `produce` 目前是**合成訊號**(Phase C),真路徑要 `TwoToneProgram(soccfg,cfg,sweep).acquire(soc,...)` + `cfg_maker(state, ml)`(需真 ml)+ 每點 `setup_devices(cfg)` 設 flux 值。`orchestrator.py` 已有 `ml=` seam 但 `start_run` 沒傳(`controller.py:387-393`)—— 接真 context 的 ml 是第一個整合點。

## 待用戶最終確認(handoff 時尚未明確點頭,但方向已認同)
- OperationGate 各 app 自持 + 共用 Handles + 注入 exclusion port(決策 3)
- ml-edit 先 measure-only(決策 4)
- 共用 dialog 放 `gui/session/ui/`(決策 5)
→ 用戶 /handoff 訊息「我打算從乾淨的 context 中實作」= 認同方向、要新 agent 從 S1 開工。新 agent 開工前可快速跟用戶 re-confirm。
