# `tests/` — test suite

**Last updated:** 2026-07-07 — ro_optimize analyze-param adapter tests

> 註：`test_registry.py` 測的是 `program/v2/modules/registry.py` 的 `PulseRegistry`（pulse 定義 SHA256 去重）。

這份筆記整理測試套件的結構、fixture 架構與新增測試時的注意事項。

---

## 執行測試套件

**標準快速跑法**（約 15 s）：

```bash
.venv/bin/python -m pytest tests/ -n auto
```

CI / agent 品質門檻使用 `uv run pytest -n auto`。`dev` dependency group 會同步安裝
`zcu-tools[gui]`，因此 bare full-suite pytest 具備 GUI/client 測試需要的 optional dependency；
若直接呼叫 `.venv/bin/python -m pytest`，先確認 venv 是透過 `uv` 的 dev group 同步。

`-n auto` 啟動 pytest-xdist 多進程平行化。`tests/conftest.py` 在每個 worker 進程啟動時
（偵測到 `PYTEST_XDIST_WORKER`）自動把 `OMP_NUM_THREADS / OPENBLAS_NUM_THREADS / MKL_NUM_THREADS`
設為 `"1"`，避免 8 worker × 多執行緒 BLAS 造成過訂（無此 pin 時約 70 s）。Serial 跑法不設
這些變數，BLAS 保持多執行緒（`pytest tests/` 約 109 s，適合 debug 或全覆蓋確認）。

**Qt GUI 子套件單獨跑**（約 22 s，不需 `-n auto`）：

```bash
.venv/bin/python -m pytest tests/gui tests/autofluxdep_gui tests/fluxdep_gui tests/dispersive_gui -q
```

### BackgroundRunner.quiesce() — 測試 teardown 必要模式

任何 fixture 若持有會透過 `BackgroundRunner` 提交 pool 或
thread worker 的物件，都必須在 teardown 呼叫 `quiesce()`，**才** `deleteLater()` / 放 GC。

原因：worker 的 `on_done` 是跨執行緒 queued signal；如果 runner 在 delivery 進到 main-thread queue
之後就被 GC，下一次 `processEvents()` 會把訊號分發到已釋放的 C++ 物件 → segfault。

```python
# pattern（見 tests/gui/test_controller.py 的 ControllerFixture.quiesce()
#           和 tests/gui/services/test_device_manager.py 的 _quiesce_services fixture）
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

## 目錄結構

```text
tests/
├── conftest.py                     # 頂層 fixture（空的 / 共用 path 設定）
├── program/
│   └── v2/
│       ├── conftest.py             # 從 program/v2 匯入 make_mock_soccfg() + mock_soccfg fixture
│       ├── test_compile.py         # 空程式 / sweep / reps 等編譯煙霧測試
│       ├── test_modules_integration.py  # 各 Module 對真實程式的整合測試
│       ├── ir/                     # IR 子系統單元測試（純 Python 物件，不需 MockSoc）
│       │   ├── test_ir_analysis.py
│       │   ├── test_ir_base.py
│       │   ├── test_ir_builder.py               # IRBuilder structure/meta parsing tests
│       │   ├── test_ir_estimate_funcs.py        # estimate_body_scheduled_ticks / flat_size / body_cost（IRLoop+IRBranch 所有分支）
│       │   ├── test_ir_hw_semantics.py          # WAVE/VOLATILE/GENERAL_REGS 常數驗證
│       │   ├── test_ir_instructions.py
│       │   ├── test_ir_linker.py
│       │   ├── test_ir_linker_wait.py
│       │   ├── test_ir_node.py
│       │   ├── test_ir_operands.py
│       │   ├── test_ir_passes_control_flow.py
│       │   ├── test_ir_passes_increg_merge.py
│       │   ├── test_ir_passes_optimization.py   # IR traversal/validation/optimization pass tests
│       │   ├── test_ir_passes_timeline.py
│       │   ├── test_ir_passes_unpack_branch.py  # UnpackIRBranchPass（含 _first_basic_block, _case_entry_label）
│       │   ├── test_ir_passes_unreachable.py    # UnreachableEliminationPass
│       │   ├── test_ir_range_hint.py
│       │   ├── test_ir_roundtrip.py
│       │   ├── test_ir_unroll_validation.py
│       │   ├── test_ir_validation.py
│       │   └── test_wave_alias_refactor.py
│       ├── macro/                  # macro 層單元測試（MagicMock prog）
│       │   ├── conftest.py         # mock_prog (pmem=512) + large_pmem_prog (pmem=4096)
│       │   ├── test_write_reg.py   # format_alu_op, WriteRegOp
│       │   ├── test_loop.py        # _needs_big_jump, _emit_cond_jump, OpenInnerLoop, CloseInnerLoop
│       │   ├── test_debug.py
│       │   ├── test_delay.py       # DelayRegAuto
│       │   ├── test_meta.py        # MetaMacro.translate
│       │   └── test_pulse_reg.py   # PulseByReg
│       └── modules/
│           ├── conftest.py         # MagicMock-based mock_prog fixture（f_time=430.08 MHz）
│           ├── test_control.py     # Repeat + Branch（含 large pmem path）
│           ├── test_delay.py       # SoftDelay, DelayAuto, Join
│           ├── test_dmem.py        # LoadValue.run（uncompressed/compressed）+ ScanWith
│           ├── test_pulse.py
│           ├── test_readout.py
│           ├── test_reset.py
│           ├── test_computed_pulse.py
│           ├── test_factory.py
│           ├── test_waveform.py
│           ├── test_registry.py
│           └── test_util.py
├── experiment/v2/                  # Experiment runtime、persistence / analysis tests
│   ├── autofluxdep/                # FluxDepInfoTracker typed context tests
│   └── runner/                     # Schedule runtime、ResultTree 與 MultiMeasurementExecutor tests
├── meta_tool/                      # SyncFile / QubitParams / arbitrary waveform persistence tests
├── analysis/
│   └── fluxdep/                    # Flux-Dependence Analysis kernel tests
├── mcp/                            # MCP bridge、call-log、timeout policy、remote schema / ARRAY param regression tests
├── notebook/analysis/fluxdep/      # Fluxonium 光譜分析模型測試
├── notebook/analysis/t1_curve/     # T1 curve Q-channel 與 t1_curve_fit tests
└── utils/fitting/                  # 曲線擬合工具測試
```

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

### Level 3 — IR 結構測試（`tests/program/v2/ir/`）

直接構建 `BasicBlockNode` / `BlockNode` / `IRLoop` 等 Python 物件，通過 `_optimize_tree()` 執行 pass，不需要 MockSoc 也不需要 MagicMock prog。適合測試 IR pass 的結構轉換行為。

```python
# 典型模式
from zcu_tools.program.v2.ir.pipeline import _optimize_tree, PipeLineContext, PipeLineConfig
ctx = PipeLineContext(config=PipeLineConfig(pmem_capacity=512), pmem_budget=1024)
result = _optimize_tree(root, [SomePass()], ctx)
```

`_optimize_tree()` 對 `BlockNode` 做 in-place mutation（`result is root`），不要依賴返回值是否為新物件來判斷 `changed`。

### SNR scorer helpers

`tests/experiment/v2/utils/test_snr.py` 用解析的 g/e `mean` / `covariance` / `third_moment` fixture 測 `snr_as_signal` 與 `skew_penalty`，不靠隨機 samples 或 Monte Carlo 閾值；這能直接驗證 pooled-sigma SNR、shape mismatch penalty 與 one-sided skew penalty 的公式語意。

### Experiment v2 Schedule runtime tests

`tests/experiment/v2/runner/test_flow.py` 覆蓋 `SignalBuffer` / `Schedule` / `ProgramBuilder` 的 typed env、host scan、program-side sweep、buffer shape、stop checker、ProgramBuilder retry、failed attempt 後 stop 不再 retry、`ScheduleOutcome`、batch 與 raw conversion contract。`test_result_tree.py` 覆蓋 executor-owned ResultTree 的 node set、direct node env event / missing-env fast-fail、child buffer、per-measurement subscription、root broadcast、flush 與 ordinary SignalBuffer regression；`test_multi_executor.py` 覆蓋 `MultiMeasurementExecutor` template lifecycle、retry、error/stop partial result、figure close 與 `ComposedMeasurementBundle` delegation。個別 experiment module 更接近資料編排，不新增 migration-specific tests；若要測 QICK compile 行為，放到 `tests/program/v2/` 或既有 sim integration 測試。

`tests/experiment/v2/onetone/` 放 onetone domain-level pure behavior tests；例如 `freq`
的 homophasal helper 測端點保留與 resonator-circle phase 等距，不碰 GUI 或硬體。

### Autofluxdep typed context tests

`tests/experiment/v2/autofluxdep/test_info_tracker.py` 覆蓋 `FluxDepInfoTracker` 的 `current` / `first` / `last` snapshot、mutable value deepcopy、missing required field fast-fail、unknown field fast-fail 與 smoothing helper behavior。這組是純 Python unit test，不觸發 predictor、SoC 或 device setup。

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
decimated trace 與 branch smoke。若測試需要 spy private `SimEngine` helper、numba
routing、cooperative cancel、population cache key 或 optimization call count，放在
`test_engine_optimization_contract.py`，並在 test name / docstring 說清楚它是白箱
optimization contract。Segment propagator LRU、單次 signal-grid prefix sequence cache
與 numba routing threshold 這類 private optimization contract 也放在同一檔。

### Experiment v2 GUI adapter tests

`tests/experiment/v2_gui/adapters/singleshot/_helpers.py` 集中 singleshot adapter 測試的 `ModuleLibrary` / context / request fixture。singleshot 測試檔名以 domain ownership 命名，例如 GE、downstream、LenRabi/T1、AC-Stark/MIST/T1-tone-sweep；不要再用歷史 Phase 編號命名。adapter 層 patch domain `run` / `analyze` 可作為 boundary isolation，但 assertion 應驗證 adapter 對 cfg、centers、summary、writeback 的語意。

onetone adapter tests 覆蓋 real-hardware adapter 的 cfg lowering、md preflight 與 writeback
contract；`onetone/freq` 的 homophasal selector 只在 adapter 邊界注入 md fit params，runtime
取樣公式由 domain-level tests 擁有。

twotone `ro_optimize` adapter tests 覆蓋 pulse-readout-only spec、GUI analyze-param
命名、MetaDict scalar writeback 與 `readout_dpm` ModuleLibrary writeback gate /
schema fields：no-snapshot md-only、current result 與 MetaDict 合併、缺值 skip。

### GUI remote/control tests

`tests/gui/services/remote/test_remote_mcp_toolchain.py` 保留 remote MCP startup/device/schema/wrapper 與 base async contract；bundle/stage tools 放在 `test_bundle_tools.py`，screenshot/debug/overview 放在 `test_screenshot_overview_tools.py`。新增 remote MCP 測試時優先放到對應 focused file；只有真正跨 toolchain 的行為才放回 `test_remote_mcp_toolchain.py`。

`tests/gui/_control_fakes.py` 提供 control facet tests 的 typed recording fakes。新增 `test_*_control.py` contract 時，偏好 recording fake + 表驅動 public forwarding contract；只在需要 Qt signal / event bus behavior 時測 event disposer、signal rebind 或 state transition，不把 `MagicMock.assert_called_once_with` 當成主要測試內容。

GUI widget tests use `tests/gui/_dialog_fakes.py::RecordingDialogPresenter` for
information messages, warnings, critical errors, confirmations, and destructive
confirmations exposed through the shared `DialogPresenter` port. Prefer
injecting this adapter into the widget under test over monkeypatching
`QMessageBox`; keep QFileDialog, QInputDialog, QMenu, or method monkeypatches
only when the tested object has no dialog-presenter boundary for that
interaction.

### Autofluxdep GUI tests

`tests/autofluxdep_gui/test_cfg_schema.py` 覆蓋 typed node cfg schema、OverridePlan serialization/validation、strict declared-patch application、pulse-readout shape restriction、real-acquire node `acquire_retry` generation knob 與 seam invariants。`test_acquire_helpers.py` 覆蓋 Schedule/ProgramBuilder acquire helper 的 retry knob default/validation、completed/stopped/failed outcome handling。`test_cfg_maker.py` 覆蓋 node builder 的 cfg lowering 與 generation overrides；lenrabi 測試同時鎖定 drive-gain feedback 使用 `expected_pi_length` setpoint、auto sweep range 使用上一點 measured `pi_length`、first-pass fallback 使用 `pi_product_seed`。`ui/test_node_cfg_form.py` 覆蓋 Default cfg / Generation split form、generated/initial decoration refresh 與 field path collection。`test_lenrabi_acquire.py` 覆蓋 lenrabi real-acquire smoke path 與 node-local fit gate helper：decay/non-decay fit 競賽、預期 candidate fit failure isolation、非預期 fit exception Fast Fail、不可信 fit 不送 feedback Patch、pi2 不可信時不產生成對 drive modules。

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
`MagicMock` driver 來驗證 call interaction，應注入 `tests/gui/services/_device_fakes.py::FakeDeviceRegistry`，
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
- `tests/program/v2/ir/test_ir_passes_optimization.py` 等 `ir/test_ir_passes_*.py` 覆蓋 traversal helper、structure validation、label reference validation、label DCE、branch case normalize、constant loop unroll、hoist/peephole/timing sanity pass。
- `PeepholePass` 測試必須保證不刪 `NOP`；目前只驗證清理 `IR_` internal annotations。
- `ConstantLoopUnrollPass` 測試使用顯式 `IRLoop.trip_count`，不要用 QICK 實際 loop asm shape 當作 unroll 偵測依據。

### estimate_* 函式測試注意事項

`estimate_body_scheduled_ticks` / `estimate_flat_size` / `estimate_body_cost` 的簽名接受 `list[IRNode]`，但 Python list 是 invariant：不能把 `list[BasicBlockNode]` 直接傳入，需要明確宣告型別 `nodes: list[IRNode] = [bb1, bb2]`。

### DeadTestEliminationPass — JumpInst in insts 路徑

`BasicBlockNode.__post_init__` 拒絕在 `.insts` 中放 `JumpInst`（不變量）。但 `_find_dead_indices` 的 line 80 確實處理了這種情況，適用於「直接呼叫 private method 傳入任意 list」的使用情境。測試必須繞過 BasicBlockNode 直接呼叫 `pass_._find_dead_indices(insts, branch=None)` 而非通過 chunk pipeline 路由。

### UnrollLoopPass — _maybe_build_jump_table 觸發條件

Register-driven loop（`n=Register`）+ `available_regs` 非空 + `k_final >= 2` + `body_size > 0` 才觸發 jump table。測試中：body 用 NopInst（size=1），`pmem_budget=1024`（k_budget=1024），`cost_default=1, cost_jump_flush=40`，body_cost=1，scheduled_ticks=0，slack=-1 < 0 → k_timing=max_unroll_factor=32；k_final=min(32,1024)=32 → 觸發。

### SimplifyDispatchPass — 回傳 BlockNode 而非 BasicBlockNode

`SimplifyDispatchPass` 對 `IRDispatch(k==2)` 回傳 `BlockNode([cond_bb, fallthrough_bb])`，兩個分支均顯式 jump，不依賴 fall-through：

- `cond_bb`：`JUMP target_labels[1] -if(NZ) -op(value_reg - #0)`（小 PMEM）；big-PMEM 用 `REG_WR s15 label + JUMP [s15]`
- `fallthrough_bb`：`JUMP target_labels[0]`（小 PMEM，`BranchEliminationPass` 可消除）；big-PMEM 永遠保留

測試應斷言 `isinstance(result, BlockNode)` 並分別驗證兩個 child block 的 branch 指令目標，不可依賴舊的 `BasicBlockNode` 回傳型別。

### DmemDispatchPass — 小/大 pmem 路徑

`DmemDispatchPass` 的 guard block 依 `needs_big_jump(pmem_capacity)` 分叉：
- `pmem_capacity <= 2048`：guard 是 label-mode `JumpInst`（`label=LabelRef(last)`, `addr=None`）
- `pmem_capacity > 2048`：guard 是 `RegWriteInst(dst=s15, src=LABEL)` + `JumpInst(addr=s15)`

測試 `DmemDispatchPass` 時用 `PipeLineContext(config=PipeLineConfig(pmem_capacity=512))` 強制小 pmem 路徑。

### DeadWriteEliminationPass / ZeroDelayDCEPass — disable_opt guard

兩者都在 `_process_block` 開頭對 `block.disable_opt` 做 early return，測試需明確用 `BasicBlockNode(..., disable_opt=True)` 驗證 skip 行為。

---

## 新增測試的快速指南

### GUI analyze params 測試

`tests/gui/adapter/test_analyze_params.py` 覆蓋 dataclass-based analyze params helper；`tests/gui/ui/test_analyze_form.py` 覆蓋 `AnalyzeFormWidget` 的 dataclass round-trip、hydrate 不 emit、使用者編輯 emit instance。新增 GUI adapter 測試時，analysis 參數應直接使用 adapter 回傳的 params dataclass instance，不要組 raw dict 或假設 `get_analyze_params()` 可迭代。

### measure-gui canonical result load 測試

load-result feature 的 targeted tests 分散在對應 ownership：
`tests/experiment/v2_gui/adapters/test_base_load.py` 鎖 adapter default load contract；
`tests/experiment/v2_gui/adapters/test_legacy_load.py` 鎖 adapter legacy single-file fallback；
`tests/gui/services/test_load.py` 鎖 state invalidation / version bump；
`tests/gui/ui/test_main_window_ui.py` 鎖 `Load Data...` button gate 與 file dialog；
`tests/gui/services/remote/` 鎖 `tab.load_data` dispatch、tool generation 與 MCP guard deps。
同目錄也覆蓋 measure MCP 的 operation handle 與 RPC timeout policy：bounded
GUI handler timeout 應回傳狀態，transport timeout 應被視為連線異常。

**新增整合測試**：在 `test_modules_integration.py` 加入新的 test class / method，  
用 `_make_prog(modules=[...])` 建構程式，斷言 `prog.binprog is not None`。

**新增單元測試**：在 `tests/program/v2/modules/test_<module>.py` 加入測試，  
使用 `mock_prog` fixture（MagicMock），驗證 cfg parsing / set_param / method call patterns。

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
