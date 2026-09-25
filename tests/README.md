# `tests/` — test suite

**Last updated:** 2026-09-25 — test structure policy

本頁是整個 `tests/` 套件新增、拆分與搬遷測試的結構規則，也保留硬體與 GUI 測試的領域注意事項。
測試行為與驗證流程以 [AGENTS.md](../AGENTS.md) 為準。

## 放置新增測試

1. 先找同一個可觀察行為或契約的既有測試檔。新增 regression 優先放在該檔；同一 owner
   有不同且穩定的責任時，可以拆成另一個 `test_*.py`，不按 ticket、phase 或 part 切檔。
   `T1` 等名稱若是領域實體或真實物理量，仍可作檔名的一部分，不能只靠字面判定為 ticket 名。
2. 目錄依 owning module，沿用 [AGENTS.md](../AGENTS.md) 的路徑對應：`tests/script/` 和
   `tests/tools/` 對應 repo root 同名目錄，其餘對應 `lib/zcu_tools/` 下的模組。
   `contract`、`parity` 是既有保留段，該段以下豁免，但前綴仍須對應；不因新案例建立新的豁免目錄。
   跨模組契約放在擁有整合行為的模組目錄，不按每個參與模組複製案例。
3. 案例按情境與預期結果命名，從模組接縫或公開契約檢查可觀察結果。
   文件、靜態內容、設定值及腳本旗標直接審閱；既有白箱或靜態檢查案例不因本頁宣稱已搬遷。

## Fixture 與測試支援碼

Fixture 放在能覆蓋使用者的最小合理目錄：只供一個檔案用的留在檔內，跨檔共用時
放最近的 `conftest.py`。`conftest.py` 管理 pytest 注入及生命週期；一般 builder、fake 或
recording adapter 放明確可 import 的支援模組，不把 `conftest.py` 當 library，也不從另一個
`test_*.py` 匯入。Fixture 的可見範圍取決於所在目錄，`scope=` 則決定實例的生命週期；
不要為了少建幾次物件把 fixture 提升到更廣的可見範圍。Root fixture 必須具有全套件用途。
相似 setup 只有在代表相同語意時才共用；不同責任不為減少行數而塞入帶多個旗標的萬用 fixture。

## 搬遷驗收

搬檔前列出舊檔到新檔的案例映射，確認每個情境、斷言與 marker 都有對應；搬檔後
比較 pytest collection 與 marker，核對 fixture lookup、覆寫與生命週期，並跑受影響選集及共同父目錄。
涉及共享狀態時，驗證不同執行順序與平行選集。先做純搬檔，再另外審查斷言改寫與重複案例刪除。
檔案移動可能改變 `conftest.py` 的 fixture lookup；案例總數相同不能取代上述核對。
本頁不代表現有測試已搬遷。

## 執行測試套件

受管理 worktree 的 Python 指令按 [AGENTS.md](../AGENTS.md) 使用 `uv run --directory <worktree> --no-sync --`。
選集與平行化由 pytest CLI 決定。開始前確認所用環境安裝 GUI/client 測試需要的 optional dependency。

pytest 全域 warning filter 只放第三方 noise（例如 `qick` / `scqubits` 在 Python 3.13
import 時的 invalid-escape `SyntaxWarning`）。本 repo 的 production warning 不應全域 suppress；
若 integration test 允許某個 fit fallback warning，使用 test-local `filterwarnings` marker
標註該測試的預期退化。

`-n auto` 啟動 pytest-xdist 多進程平行化。`tests/conftest.py` 在每個 worker 進程啟動時
（偵測到 `PYTEST_XDIST_WORKER`）把 `OMP_NUM_THREADS / OPENBLAS_NUM_THREADS / MKL_NUM_THREADS`
設為 `"1"`，避免 worker 與多執行緒 BLAS 相互過訂。Serial 跑法不設這些變數，適合 debug。
Qt GUI 子套件可選 `tests/gui tests/gui/app/autofluxdep tests/gui/app/fluxdep tests/gui/app/dispersive`。

### BackgroundRunner.quiesce() — 測試 teardown 必要模式

任何 fixture 若持有會透過 `BackgroundRunner` 提交 pool 或
thread worker 的物件，都必須在 teardown 呼叫 `quiesce()`，**才** `deleteLater()` / 放 GC。

原因：worker 的 `on_done` 是跨執行緒 queued signal；如果 runner 在 delivery 進到 main-thread queue
之後就被 GC，下一次 `processEvents()` 會把訊號分發到已釋放的 C++ 物件 → segfault。

```python
# pattern（見 tests/gui/test_controller.py 的 ControllerFixture.quiesce()
#           和 tests/gui/session/services/test_device_manager.py 的 _quiesce_services fixture）
@pytest.fixture
def my_widget(qapp):
    w = SomeWidgetThatOwnsBackgroundRunner(...)
    yield w
    w._debounce.stop()   # 停止任何 debounce timer，防止啟動新 worker
    w._runner.quiesce()  # join in-flight + flush queued deliveries
    w.deleteLater()
    qapp.processEvents()
```

如果 fixture 持有的是 `Controller`（它持有具體 `BackgroundRunner`），改呼叫
`ctrl._background_svc.quiesce()`（見 `tests/gui/test_controller.py`）。

### Qt timer waits

`qtpy.QtTest.QTest` 在 PyQt6 runtime 是 C++ namespace，不能實例化；但部分
stub 會把 `QTest.qWait(ms)` 視為 unbound instance method。GUI 測試需要等待
debounce timer 時，用本地 helper 包 `QEventLoop + QTimer.singleShot`，不要直接
呼叫 `QTest.qWait`，也不要用 `cast()` 或 type ignore 壓掉 stub 差異。

---

## 現有 owner 導覽

`tests/program/v2/` 擁有 QICK compile、IR、macro、module 與 simulator 行為；
`tests/experiment/v2/` 擁有排程與實驗資料流程；`tests/experiment/v2_gui/adapters/`
擁有 adapter 對設定與寫回的契約。`tests/gui/` 與各 app GUI 目錄擁有 UI、service、remote
接縫；`tests/mcp/` 擁有 MCP bridge 與操作契約。`tests/meta_tool/`、`tests/analysis/`、
`tests/notebook/`、`tests/utils/` 分別擁有其路徑對應模組的測試。
例如 `tests/program/v2/modules/test_registry.py` 測 `PulseRegistry` 的 pulse 定義 SHA256 去重，
與同名的其他 registry 測試無關。需要定位檔案時以目前目錄及程式 owner 為準。

---

## 兩層 Fixture 策略

### Level 1 — `make_mock_soccfg()`（實作在 `lib/zcu_tools/program/v2/mocksoc.py`，由 `tests/program/v2/conftest.py` 匯入）

用真實 `QickConfig` 從純 dict 建構，不需要任何硬體。  
所有 `ModularProgramV2.__init__` → `compile()` → `_initialize/_body` 都跑真實 QICK ASM。

**Gen 規格**（`axis_signal_gen_v6`，HAS_MIXER=False）：

- `fs=12288.0 MHz`（245.76 × 50），`f_dds=12288.0`，`b_dds=32`，`b_phase=32`
- `fs_mult=50`，`fs_div=1`，`fdds_div=1`，`samps_per_clk=16`
- `has_mixer=False`，`maxv=32766`，`maxv_scale=1.0`
- nqz=1 有效頻率範圍：0–6144 MHz（f_dds 拉高到 12288 給 6 GHz 級 readout 充足 headroom，免折疊）

**Readout 規格**（`axis_readout_v2`）：

- `fs=2457.6 MHz`（245.76 × 10），`f_dds=2457.6`，`b_dds=32`
- `fs_mult=10`，`fs_div=1`，`fdds_div=1`，`decimation=1`
- `f_output=307.2 MHz`（fs / (decimation × DOWNSAMPLING=8)）
- 有效頻率範圍：0–1228.8 MHz

**為什麼不用 `axis_sg_int4_v1`？**  
這個 gen 是 `HAS_MIXER=True`，`declare_gen()` 會強制要求 `mixer_freq`，  
但 `Pulse.init_pulse()` 只在 `cfg.mixer_freq is not None` 時才傳 `mixer_freq`，  
導致 `RuntimeError: generator N has a digital mixer, but no mixer_freq was defined`。

**哪些欄位必須手動補齊？**  
`QickConfig.__init__` 不計算任何欄位，欄位由 FPGA firmware driver 設定，  
所以 mock dict 必須包含：`has_mixer`、`maxv`、`maxv_scale`、`b_phase`、  
`fs_mult`、`fs_div`、`fdds_div`、`interpolation`（gen）；  
`fs_mult`、`fs_div`、`fdds_div`、`decimation`、`has_outsel`（readout）。  
缺少任何一個就會在 `declare_gen` / `add_readoutconfig` / `freq2reg` 時 `KeyError`。

### Level 2 — `MagicMock` (`tests/program/v2/modules/conftest.py`)

用於 `modules/` 下的**純單元測試**，只測試 cfg parsing / set_param / allow_rerun 等邏輯，  
不觸發真實 QICK 編譯。`mock_prog.soccfg` 是一個最小 dict，不需完整硬體欄位。

### IR 結構測試（`tests/program/v2/ir/`）

`BasicBlockNode` / `BlockNode` / `IRLoop` 可建立不依賴 MockSoc 的輸入。
經由負責該行為的 IR pipeline 接縫檢查輸出結構或編譯結果；不要以私有
`_optimize_tree()` 的回傳物件身分判斷是否發生變更，因為它可能原地修改 `BlockNode`。

### SNR scorer helpers

`tests/experiment/v2/utils/test_snr.py` 用解析的 g/e `mean` / `covariance` / `third_moment` fixture 測 `snr_as_signal` 與 `skew_penalty`，不靠隨機 samples 或 Monte Carlo 閾值；這能直接驗證 pooled-sigma SNR、shape mismatch penalty 與 one-sided skew penalty 的公式語意。

### Experiment v2 Schedule runtime tests

`tests/experiment/v2/runner/test_flow.py` 覆蓋 `SignalBuffer` / `Schedule` / `ProgramBuilder` 的 typed env、host scan、program-side sweep、buffer shape、stop checker、ProgramBuilder retry、failed attempt 後 stop 不再 retry、`ScheduleOutcome`、batch 與 raw conversion contract。`test_result_tree.py` 覆蓋 executor-owned ResultTree 的 node set、direct node env event / missing-env fast-fail、child buffer、per-measurement subscription、root broadcast、flush 與 ordinary SignalBuffer regression；`test_multi_executor.py` 覆蓋 `MultiMeasurementExecutor` template lifecycle、retry、error/stop partial result、figure close 與 `ComposedMeasurementBundle` delegation。個別 experiment module 更接近資料編排，不新增 migration-specific tests；若要測 QICK compile 行為，放到 `tests/program/v2/` 或既有 sim integration 測試。

`tests/experiment/v2/onetone/` 放 onetone domain-level pure behavior tests；例如 `freq`
的 homophasal helper 測端點保留與 resonator-circle phase 等距，不碰 GUI 或硬體。

### Autofluxdep typed context tests

`tests/experiment/v2/autofluxdep/test_info_tracker.py` 覆蓋 `FluxDepInfoTracker` 的 `current` / `first` / `last` snapshot、mutable value deepcopy、missing required field fast-fail、unknown field fast-fail 與 smoothing helper behavior。這組是純 Python unit test，不觸發 predictor、SoC 或 device setup。

### Device manager tests

`tests/device/test_manager_lock.py` 覆蓋 `GlobalDeviceManager` registry lock 只保護
registry dict、`get_info` 不被其它 device 的 setup ramp 阻塞、整批 name validation
fast-fail、以及 `setup_devices(..., cancel_signal=...)` / `device_setup_cancel_scope(...)`
的協作取消語意。取消測試使用真 `FakeDevice`，不註冊 MagicMock driver。

### 補充 — `make_mock_soc()`（`lib/zcu_tools/program/v2/mocksoc.py`）

若要測 `acquire()` / `poll_data()` 路徑（不只 compile），可用 `make_mock_soc()` 建立 `MockQickSoc`。  
它繼承 `QickConfig` 並提供 no-op 硬體控制方法，回傳 shape 正確的隨機資料，讓整段 acquire 流程可在無硬體下跑通。

### Program v2 simulator tests

`tests/program/v2/sim/test_params.py` 擁有 `SimParams` validation、coherence helper、
readout decay knobs 與 `Temp` + operating qubit frequency 到 Boltzmann equilibrium
population 的純 helper 測試。`tests/program/v2/sim/test_readout.py` 覆蓋
step-photon readout backaction closed form（pulse length vs readout length 分離）與
decimated excited-initial center；`test_bloch.py` 覆蓋 readout 後 amplitude damping map。
`tests/program/v2/sim/test_engine.py` 放 public simulator behavior：physics shape、
spectroscopy/Rabi/T1/T2、single-shot blob、readout scaling、readout-induced backaction、
decimated trace 與 branch smoke。效能相關 regression 先找 public simulator 的輸出與
可觀察的資源行為；private cache key、helper call count 或 routing threshold
不是新增測試的獨立驗收條件。既有 `test_engine_optimization_contract.py`
仍含白箱案例，本次文件整理不代表該檔已遷移。

### Shared GUI cfg import ownership tests

`tests/gui/measure_cfg/`鎖定closed 7 module + 6 waveform discriminator/label/order、program/v2
runtime parity、nested allowed sets、deep-fresh mutable containers、main/autoflux僅兩個policy差異、
strict root-only Mapping/typed inspection、missing/non-string/unknown與fresh-process import purity；materializer tests另鎖spec/value完整對齊、scalar
missing、nested complete default、required reference `allowed[0]`、missing style Const、explicit unknown與
Bath ghost rejection。`tests/gui/cfg/test_materialization.py`只測domain-free Spec walk與policy ports；main
全7+6及autoflux legal-but-unmaterializable subset由各app cfg tests鎖定。app binding/lowering tests鎖
shape lookup、converter與`to_dict()`精確call count；role tests鎖registration shape/value count、Controller
value→shape順序，以及既有26-entry metadata/value golden。

`tests/gui/cfg/binding/test_targets.py`鎖定list/resolve acceptance equality、canonical grammar、
legacy zero-mutation replacement、schema collision與production registry coverage。remote cfg tests
鎖定 target wire parity；service/writeback tests 鎖定 shape-only listing 與 batch net diff。
現有 AST import purity 案例屬於待整理的靜態檢查，新增 import 變動直接審閱。

measure adapter facade 不 forward `zcu_tools.gui.cfg.__all__` names；generic cfg imports
應指向 shared owner，autofluxdep app-local barrel 另有自己的 owner。
現有 `tests/gui/cfg/test_measure_import_contract.py` 與
`tests/gui/app/autofluxdep/test_cfg_import_contract.py` 含靜態 import 檢查；新增或修改
import 規則時直接 review 相關檔案，對外行為則由接縫測試驗證。

`tests/gui/cfg/test_schema_assembler.py`擁有domain-free paired Spec/Value construction contract：
path/parent conflict與batch preflight、default carrier、optional ref、locked alignment、choice binding、
caller alias隔離與one-shot build。domain role、Seed與app section policy不得進入這組shared tests。

### Experiment v2 GUI adapter tests

`tests/experiment/v2_gui/adapters/_support/test_schema_builder.py`鎖定context-free
`MeasureCfgBuilder` / `MeasureCfgDefinition`、`ModuleInit` role shape與materialization modes、typed Seed
resolution/path errors、module override/lock transactionality與definition isolation。
`tests/gui/app/main/adapter/test_adapter_definition.py` 驗證 empty/rich md/ml contexts 下的
adapter definition 可重複 instantiate；registry 數量與 static spec 宣告直接審閱。

Singleshot adapter 案例依 cfg、analysis 等穩定行為找 owner，不以歷史 Phase 切檔。
例如 GE、downstream、LenRabi/T1、AC-Stark/MIST/T1-tone-sweep 描述的是領域責任，
不是 ticket 命名。adapter 層 patch domain `run` / `analyze` 可作為 boundary isolation，
但 assertion 應驗證 adapter 對 cfg、centers、summary、writeback 的語意。

onetone adapter tests 覆蓋 real-hardware adapter 的 cfg lowering、md preflight 與 writeback
contract；`onetone/freq` 的 homophasal selector 只在 adapter 邊界注入 md fit params，runtime
取樣公式由 domain-level tests 擁有。`onetone/freq` writeback tests 覆蓋 MetaDict
`r_f` / `rf_w` / `theta0` 與 `readout_rf` ModuleLibrary writeback 的 no-snapshot gate、
pulse-readout schema、non-pulse skip，以及 default 仍不 adopt library readout。

twotone `ro_optimize` adapter tests 覆蓋 pulse-readout-only spec、GUI analyze-param
命名、MetaDict scalar writeback 與 `readout_dpm` ModuleLibrary writeback gate /
schema fields：no-snapshot md-only、current result 與 MetaDict 合併、缺值 skip。

### GUI remote/control tests

`tests/gui/remote/test_lazy_broadcast.py`以barrier鎖定two-phase recipient
selection：零matching recipient不build、多client只build一次、unsubscribe/disconnect送前重驗、
late subscribe不補收舊event，以及slow-client drop budget不阻塞healthy client或改變per-client order。
remote EventBus與cfg-editor focused tests另鎖serializer/current-path/encode lazy cost、failure logging、
`editor_closed` delivery cleanup；diagnostic測試確認fault channel仍不受subscription gate影響。

`tests/gui/app/main/ui/test_main_window_events.py`鎖定closed tab facts到Qt reaction的
完整call sequence、zero-reaction local edits與lazy單次snapshot；service與真實UI測試覆蓋
run去重、analysis start-rejected/failure/cancel retained-figure restore、load stale-canvas clear、
same-class form hydrate/cache，以及ModuleLibrary變更透過attached cfg draft更新run gate。

`tests/gui/test_expected_error.py`鎖定closed category、legacy RuntimeError/ValueError ancestry與
explicit concrete opt-in/exclusion；`tests/gui/app/main/services/remote/test_expected_error_wire_compat.py`
以exact `(code, message, reason, data)` tuple鎖定既有handler projection，並證
`ResultScopeError`分類不依賴reason prefix。

`test_expected_error_dispatch.py`走實際shared dispatch reply path，鎖定main/off-main category
mapping、direct structured `RemoteError` passthrough、generic `data=None`與translator failure
containment。`test_unexpected_handler_errors.py`以既有handler樣本確認ordinary RuntimeError、ProviderError、I/O與
invariant failure不被降級；unexpected dispatch測試另確認controller error log保留traceback。
新增 request handler 或修改分類時，直接審閱 catch 範圍與 method entry 的靜態宣告，
再用公開 request／reply 測試確認對外錯誤碼。

`tests/mcp/measure/`擁有measure MCP tool assembly、guard、operation、timeout、bundle、
view product及lifecycle／stdio行為。每個fixture建立自己的session／bridge／tool table，
透過recording Transport觀察RPC，不patch server globals或私有helpers。
`tests/gui/app/main/services/remote/test_remote_mcp_toolchain.py`保留GUI startup/device/save／guide
handler契約；同目錄的事件整合測試保留真socket，驗證EventBus→bridge→session的origin。
Shared exposure policy 的可觀察行為屬於 `tests/gui/remote/`。Schema 文字、tool inventory 與
script flags 用直接 review，不納入 pytest。

`tests/gui/_control_fakes.py` 提供 control facet tests 的 typed recording fakes。新增 `test_*_control.py` contract 時，偏好 recording fake + 表驅動 public forwarding contract；只在需要 Qt signal / event bus behavior 時測 event disposer、signal rebind 或 state transition，不把 `MagicMock.assert_called_once_with` 當成主要測試內容。

GUI widget tests use `tests/gui/_dialog_fakes.py::RecordingDialogPresenter` for
information messages, warnings, critical errors, confirmations, and destructive
confirmations exposed through the shared `DialogPresenter` port. Prefer
injecting this adapter into the widget under test over monkeypatching
`QMessageBox`; keep QFileDialog, QInputDialog, QMenu, or method monkeypatches
only when the tested object has no dialog-presenter boundary for that
interaction.

### Autofluxdep GUI tests

`tests/gui/app/autofluxdep/test_cfg_schema.py`另外鎖定`NodeSchemaBuilder`抽取到shared
`CfgSchemaAssembler`前後的spec/value/logical-path/persisted observable parity；autoflux domain
仍擁有logical projection與generation policy。

`tests/gui/app/autofluxdep/test_cfg_schema.py` 擁有 `NodeSchemaBuilder` public verbs、logical-key 格式、pulse module mutation、transactional build / compound declaration contract，以及 typed node cfg schema、OverridePlan serialization/validation、production registry snapshot leaf coverage、strict declared-patch application、pulse-readout shape restriction、real-acquire node `acquire_retry` generation knob 與 seam invariants。`test_cfg_import_contract.py` 的現有靜態檢查不作為新增測試模式。`test_node_defaults_helpers.py` 覆蓋 node module patch、sweep extraction、readout seed 與 timing seed/range helpers 的 owner-level behavior。`test_acquire_helpers.py` 覆蓋 Schedule/ProgramBuilder acquire helper 的 retry knob default/validation、completed/stopped/failed outcome handling，以及run snapshot nested alias、`SweepCfg`與ndarray freeze/thaw隔離。`test_cfg_maker.py` 覆蓋 node builder 的 cfg lowering 與 generation overrides；lenrabi 測試同時鎖定 drive-gain feedback 使用 `expected_pi_length` setpoint、auto sweep range 使用上一點 measured `pi_length`、first-pass fallback 使用 `pi_product_seed`；T1/T2/T2Echo 測試鎖定 auto decay sweep stop 受 generation `max_length` 上限控制。`test_orchestrator.py` 鎖定 `ModuleDep` alias/missing/node-produced precedence、run-start fallback capture 與 consumer mutation isolation；`test_run_body.py` 鎖定 production `RunSession` 以同一 run-local `ModuleLibrary` 做 cfg snapshot lowering 與 module source。`ui/test_node_cfg_form.py` 覆蓋 Default cfg / Generation split form、generated/initial decoration refresh 與 field path collection。`test_lenrabi_acquire.py` 覆蓋 lenrabi real-acquire smoke path 與 node-local fit gate helper：decay/non-decay fit 競賽、預期 candidate fit failure isolation、非預期 fit exception Fast Fail、不可信 fit 不送 feedback Patch、pi2 不可信時不產生成對 drive modules。

`tests/gui/app/autofluxdep/test_labber_browser_export.py` 覆蓋 Labber Browser sidecar contract、fixed-axis sidecar live streaming row writes 與 terminal qubit_freq sidecar export。

Autofluxdep real-acquire smoke tests 依賴 flux-aware `MockSoc` 的物理模型；測試 fixture 要讓
`connect_mock(..., sim_params=...)`、`mock_flux_predictor(sim_params)` 與 drive pulse calibration
使用同一個 `SimParams`。π / π/2 drive pulse 優先用 helper 依 `pi_gain_len / gain` 校準；當測試目標是
real acquire + fit 本身時，將 sweep/gain/relax 的 generation mode 固定，避免 feedback auto mode 覆寫測試輸入。

UI mechanics tests 的 `make_measurement_builder("qubit_freq")` 仍使用 production node type/name，
因此 fake Result 也必須符合 `qubit_freq` artifact/export contract（`QubitFreqResult`），不能用 generic
`Sweep1DResult` 冒充；否則 run 會在 terminal Labber Browser export Fast Fail，headless 測試可能卡在
`QMessageBox.warning` modal dialog。其它 fake 1D measurement type 可繼續用 `Sweep1DResult`。

### GUI device service tests

`GlobalDeviceManager` 是 production singleton，入口只接受 `BaseDevice` instance。GUI service unit tests 若用
`MagicMock` driver 來驗證 call interaction，應注入 `tests/gui/session/services/_device_fakes.py::FakeDeviceRegistry`，
不要把 mock driver 註冊進 global singleton。需要測 singleton CRUD 時改用真 `FakeDevice`。

`DeviceService.poll_device_info(name)` 測試應視為 best-effort off-main live-read contract：memory-only、
connect/disconnect 等非 setup mutation 會 skip；`SETTING_UP` 的 selected device 可 poll current driver
info 來刷新 cache/UI，但 late delivery 若已進入非 setup mutation 不可 bump/emit。

等待 async device connect 時避免裸 `QEventLoop.exec()` 只聽 success signal；測試 helper 應同時觀察
`operation_failed`，並用 bounded `processEvents()` loop，讓 connect failure 變成 assertion failure 而非 pytest hang。

### Golden / characterization tests

大型 golden/snapshot equality 可以保留作 characterization，但旁邊應有 focused semantic invariant 說明 load-bearing contract，例如 role set ownership、live `EvalValue` link、或 schema 宣告的是 user knobs 而非 derived cfg fields。變更 golden payload 時，先用 invariant 說明語意，再更新 payload。

測試不重寫 production default/value table。若 characterization 必須比對 default 或派生值，expected 從 production schema、constant 或 helper 取得；測試本身只寫語意 invariant 或明確測試輸入。

---

## 整合測試注意事項

### 頻率值

測試檔案 `test_modules_integration.py` 使用：

```python
GEN_FREQ = 1000.0   # MHz，nqz=1，axis_signal_gen_v6 有效
RO_FREQ  = 100.0    # MHz，axis_readout_v2 有效
```

### PulseReadout / gen_ch

`DirectReadoutCfg(gen_ch=X)` 需要 gen ch X 已被 `declare_gen` 過（即同一個程式中有對應的 `Pulse.init`），  
否則 `add_readoutconfig` 會找不到 `gen_chs[X]`，拋出 `KeyError`。

### Register-driven Repeat

```python
r = Repeat("r_cnt", "n_count")   # "r_cnt" = counter reg，"n_count" = count reg（sweep 建立）
r.add_content(...)
_make_prog(modules=[r], sweep=[("n_count", 4)])
```

`Repeat(name, n_reg)` 中的 `name` 是 `OpenLoopReg.preprocess` 新建的 counter register，  
`n_reg` 是已存在的 count register（由 sweep 的 `add_loop` 建立）。  
**不能**用同一個名字 `Repeat("x", "x") + sweep=[("x", N)]`——sweep 和 OpenLoopReg 都會試著建 `"x"`，造成 `NameError: register name 'x' already exists`。

### Branch

需搭配外部 sweep loop，且 `compare_reg` 名稱要與 sweep loop 名稱一致：

```python
Branch("sel", [...], [...])  # compare_reg 預設為 "sel"
_make_prog(modules=[b], sweep=[("sel", 2)])
```

### Join / merge_max_length UserWarning

`modules/util.py::merge_max_length()` 在兩個 length 值相等且無法 reduce 時發出 UserWarning（「Detected multiple overlapping lengths」）。在測試 `Join` 時，若兩個 branch 的 `SoftDelay` delay 值相同（例如都是 `0.0`），round 後都為 `0.0`，會觸發此 warning。解法：使用不同的 delay 值（例如 `0.1` vs `0.5`），讓 `merge_max_length` 可選出明確最大值。

### LoadValue 壓縮閾值

`auto_compress` 只在 `len(values) >= 30` 時啟動。  
測試中用 64 個元素驗證壓縮路徑，3 個元素驗證非壓縮路徑。

### IR pass 測試

- `tests/program/v2/ir/test_ir_builder.py` 覆蓋 `__META__` parsing、branch case identity、loop section parsing，以及 jump label reference 不被誤判成 label definition。
- `tests/program/v2/ir/test_ir_passes_optimization.py` 等 `ir/test_ir_passes_*.py` 涵蓋結構與 label validation、label DCE、branch case normalize、constant loop unroll、hoist/peephole/timing pass。新增案例從 pipeline 結果驗證 IR 行為。
- `PeepholePass` 應保留 `NOP`；驗證輸出時分辨 `NOP` 與可清除的 `IR_` internal annotations。
- `ConstantLoopUnrollPass` 測試使用顯式 `IRLoop.trip_count`，不要用 QICK 實際 loop asm shape 當作 unroll 偵測依據。

### estimate_* 函式測試注意事項

`estimate_body_scheduled_ticks` / `estimate_flat_size` / `estimate_body_cost` 的簽名接受 `list[IRNode]`，但 Python list 是 invariant：不能把 `list[BasicBlockNode]` 直接傳入，需要明確宣告型別 `nodes: list[IRNode] = [bb1, bb2]`。

### IR block 不變量

`BasicBlockNode.__post_init__` 拒絕在 `.insts` 中放 `JumpInst`。新增 pass 測試用有效的 block
經 pipeline 檢查可觀察結果；不為 unreachable 的非法 block 繞過建構器測私有方法。

### UnrollLoopPass — jump table 觸發條件

Register-driven loop（`n=Register`）+ `available_regs` 非空 + `k_final >= 2` + `body_size > 0` 才觸發 jump table。測試中：body 用 NopInst（size=1），`pmem_budget=1024`（k_budget=1024），`cost_default=1, cost_jump_flush=40`，body_cost=1，scheduled_ticks=0，slack=-1 < 0 → k_timing=max_unroll_factor=32；k_final=min(32,1024)=32 → 觸發。

### SimplifyDispatchPass — 回傳 BlockNode 而非 BasicBlockNode

`SimplifyDispatchPass` 對 `IRDispatch(k==2)` 回傳 `BlockNode([cond_bb, fallthrough_bb])`，兩個分支均顯式 jump，不依賴 fall-through：

- `cond_bb`：`JUMP target_labels[1] -if(NZ) -op(value_reg - #0)`（小 PMEM）；big-PMEM 用 `REG_WR s15 label + JUMP [s15]`
- `fallthrough_bb`：`JUMP target_labels[0]`（小 PMEM，`BranchEliminationPass` 可消除）；big-PMEM 永遠保留

經 pipeline 驗證時檢查兩個 branch 的目標，而不是依賴舊的 `BasicBlockNode` 回傳型別。

### DmemDispatchPass — 小/大 pmem 路徑

`DmemDispatchPass` 的 guard block 依 `needs_big_jump(pmem_capacity)` 分叉：
- `pmem_capacity <= 2048`：guard 是 label-mode `JumpInst`（`label=LabelRef(last)`, `addr=None`）
- `pmem_capacity > 2048`：guard 是 `RegWriteInst(dst=s15, src=LABEL)` + `JumpInst(addr=s15)`

驗證小 pmem 路徑時提供 `PipeLineContext(config=PipeLineConfig(pmem_capacity=512))`，經 IR pipeline 觀察生成結果。

### DeadWriteEliminationPass / ZeroDelayDCEPass — disable_opt guard

`block.disable_opt=True` 應保留未優化的行為。以有效 `BasicBlockNode(..., disable_opt=True)`
作 pipeline 輸入，比較產生的 IR，而非直接斷言 `_process_block` 的呼叫或回傳。

---

## 新增測試的快速指南

### GUI analyze params 測試

`tests/gui/app/main/adapter/test_analyze_params.py` 覆蓋 dataclass-based analyze params helper；`tests/gui/app/main/ui/test_analyze_form.py` 覆蓋 `AnalyzeFormWidget` 的 dataclass round-trip、hydrate 不 emit、使用者編輯 emit instance。新增 GUI adapter 測試時，analysis 參數應直接使用 adapter 回傳的 params dataclass instance，不要組 raw dict 或假設 `get_analyze_params()` 可迭代。

### measure-gui canonical result load 測試

load-result feature 的 targeted tests 分散在對應 ownership：
`tests/experiment/v2_gui/adapters/test_base_load.py` 鎖 adapter default load contract；
`tests/experiment/v2_gui/adapters/test_legacy_load.py` 鎖 adapter legacy single-file fallback；
`tests/gui/app/main/services/test_load.py` 鎖 state invalidation / version bump；
`tests/gui/app/main/ui/test_main_window_ui.py` 鎖 `Load Data...` button gate 與 file dialog；
`tests/gui/app/main/services/remote/` 鎖 `tab.load_data` dispatch、tool generation 與 MCP guard deps。
`tests/mcp/measure/`覆蓋operation handle與RPC timeout policy：bounded
GUI handler timeout應回傳狀態，transport timeout應被視為連線異常。

Program v2 的 compile regression 先找 `test_modules_integration.py` 中同一契約，
以 `_make_prog(modules=[...])` 建構程式，檢查編譯結果及該情境需區分的語意。
`tests/program/v2/modules/` 的 cfg parsing 或 set_param 案例可用 `mock_prog` 隔離硬體；
從 module 接縫驗證結果，只有呼叫順序本身屬於契約時才斷言順序。

**新增 soccfg 欄位**：若 QICK 版本升級導致新的 `KeyError`，在  
`lib/zcu_tools/program/v2/mocksoc.py` 的 `_build_mock_cfg()`（內含 `_gen()` / `_readout()`）補入對應欄位即可。

---

## Cfg 解析測試的 import 注意事項

`ModuleCfgFactory` / `WaveformCfgFactory` 目前是 `TypeAdapter(...).validate_python(...)` 的 thin wrapper；分派由 `ModuleCfg` / `WaveformCfg` 的 Union TypeAlias 決定。測試檔仍建議從 package 入口 import，避免直接碰 leaf base class：

```python
from zcu_tools.program.v2.modules import ModuleCfgFactory   # ✅ 使用 package 入口
# 不要單獨：from zcu_tools.program.v2.modules.base import AbsModuleCfg

cfg = ModuleCfgFactory.from_raw({"type": "readout/direct", "ro_ch": 0, ...})
```

呼叫 `ModuleCfgFactory.from_raw(raw, ml=...)` 進行解析與 context 傳遞。

**測試一個非 `Literal` 的 cfg（負面測試）**：若要在測試裡定義 `class UnknownCfg(AbsReadoutCfg): type: str = "unknown"` 來驗證 factory 錯誤路徑，重點是它不在 `ModuleCfg` Union 裡，因此 `ModuleCfgFactory.from_raw()` 不會 dispatch 到它。
