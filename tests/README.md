# AI Note for `tests/`

**Last updated:** 2026-06-23 — MCP taskboard coverage

> 註：`test_registry.py` 測的是 `program/v2/modules/registry.py` 的 `PulseRegistry`（pulse 定義 SHA256 去重）。

這份筆記整理測試套件的結構、fixture 架構與新增測試時的注意事項。

---

## 執行測試套件

**標準快速跑法**（約 15 s）：

```bash
.venv/bin/python -m pytest tests/ -n auto
```

`-n auto` 啟動 pytest-xdist 多進程平行化。`tests/conftest.py` 在每個 worker 進程啟動時
（偵測到 `PYTEST_XDIST_WORKER`）自動把 `OMP_NUM_THREADS / OPENBLAS_NUM_THREADS / MKL_NUM_THREADS`
設為 `"1"`，避免 8 worker × 多執行緒 BLAS 造成過訂（無此 pin 時約 70 s）。Serial 跑法不設
這些變數，BLAS 保持多執行緒（`pytest tests/` 約 109 s，適合 debug 或全覆蓋確認）。

**Qt GUI 子套件單獨跑**（約 22 s，不需 `-n auto`）：

```bash
.venv/bin/python -m pytest tests/gui tests/autofluxdep_gui tests/fluxdep_gui tests/dispersive_gui -q
```

### BackgroundRunner.quiesce() — 測試 teardown 必要模式

任何 fixture 若持有會透過 `BackgroundRunner`（或持有它的 `BackgroundService`）提交 pool 或
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

如果 fixture 持有的是 `Controller`（它包含 `BackgroundService`），改呼叫
`ctrl._background_svc.quiesce()`（見 `tests/gui/test_controller.py`）。

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
├── experiment/v2/runner/           # 高層 runner / task 狀態機測試
├── mcp/
│   └── taskboard/                  # taskboard MCP method spec/dispatch、path conflict、session identity、TTL/promotion 測試
├── notebook/analysis/fluxdep/      # Fluxonium 光譜分析模型測試
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

### 補充 — `make_mock_soc()`（`lib/zcu_tools/program/v2/mocksoc.py`）

若要測 `acquire()` / `poll_data()` 路徑（不只 compile），可用 `make_mock_soc()` 建立 `MockQickSoc`。  
它繼承 `QickConfig` 並提供 no-op 硬體控制方法，回傳 shape 正確的隨機資料，讓整段 acquire 流程可在無硬體下跑通。

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
