# `zcu_tools.experiment.v2.runner` — experiment run runtime

**Last updated:** 2026-07-02 — Schedule cfg-owned reps/rounds

這份筆記整理 `runner/` 的任務執行框架設計，說明各類別的職責、組合方式與執行流程。

---

## 架構總覽（一句話版）

`runner/` 目前同時支援兩條路徑：舊路徑以 **Composite 模式** 把測量任務組成 `AbsTask` 樹，結果以同構樹（`Result`）儲存；新路徑以 `SignalBuffer` / `Schedule` / `ProgramBuilder` 提供 Python-like 寫法，讓常規實驗用一般 Python 編排，leaf acquire 直接建立 program 並保留 partial update、progress、retry 與 stop checker 語意。`MultiMeasurementExecutor`（`multi_executor.py`）是 runner 層的 Executor 共用 scaffold（非樹節點）。

---

## 類別責任

### `TaskState`（`state.py`）

```
root_data  ─── 整棵結果樹（共享，不複製）
path       ─── (addr0, addr1, ...) 定位 root_data 中目前節點
cfg        ─── 當前層的設定（child() 呼叫時 deepcopy 隔離）
env        ─── 跨層共享的可變字典（repeat_idx 等 side-channel）
on_update  ─── 任何 set_value() 後觸發的回呼（節流後的 liveplot 更新）
_stop_flag ─── Optional[threading.Event]，跨層共享（child() 傳遞引用，不 deepcopy）
```

- `child(addr)` — 建立指向子節點的 `TaskState`，cfg 以 `deepcopy` 隔離；`_stop_flag` 共享引用。
- `child_with_cfg(addr, new_cfg)` — 建立子節點並替換 cfg；`_stop_flag` 同樣共享引用。
- `is_stop()` — 回傳 `_stop_flag` 是否已 set，可直接作為 stop_checker 傳給 `acquire()`。
- `set_stop()` — 從外部設置 stop flag（本地中斷用，一般用 `TaskHandle.cancel()` 更佳）。
- `set_value(value)` — 依目標型別 in-place 更新（dict → update、list → replace、ndarray → copyto），然後觸發 `on_update`。
- `_trigger_update_hook()` — 傳入一份 snapshot 給回呼（無 `on_update`），避免回呼自己再觸發回呼。也可被外部（如 `RepeatOverTime`）手動呼叫以強制刷新 liveplot。

`Result` 型別別名：遞迴聯合 `Sequence[Result] | Mapping[Any, Result] | NDArray`。

---

### `AbsTask`（`base.py`）

```python
class AbsTask(ABC, Generic[T_Result, T_RootResult, T_Cfg]):
    def init(self, dynamic_pbar: bool = False) -> None  # 可選覆寫，初始化 pbar / env；無 state 參數
    def run(self, state: TaskState[...]) -> None        # 必須覆寫，執行任務
    def cleanup(self) -> None                           # 可選覆寫，清理資源（pbar 等）
    def get_default_result(self) -> T_Result            # 必須覆寫，回傳初始化結果
```

**重要**：`init()` 不再接收 `state` 參數（只有 `dynamic_pbar`）。狀態初始化（例如設 `env["repeat_idx"]`）已移到 `run()` 內部。

**泛型參數**：

- `T_Result` — 此節點的結果型別（`NDArray`、`list[...]`、`dict[...]`）
- `T_RootResult` — 整棵樹根節點的結果型別（在所有子節點之間共享）
- `T_Cfg` — 此節點的設定型別（bound to `ExpCfgModel`）

快捷工廠方法：

- `task.scan(name, values, before_each)` → `Scan`
- `task.repeat(name, times, interval)` → `RepeatOverTime`
- `task.auto_retry(max_retries)` → `ReTryIfFail`

---

### `Task`（`task.py`）

葉節點，包裝單一測量函式。

```python
Task(
    measure_fn,      # (state, update_hook) → T_Raw
    raw2signal_fn,   # T_Raw → NDArray（預設 raw[0][0] · [1, 1j]）
    result_shape,    # 結果 ndarray 的 shape
    dtype,           # 預設 complex128
    pbar_n,          # 可選，進度條總長（rounds 數）
)
```

**執行流程**：

1. 呼叫 `measure_fn(state, update_hook)`（注意：不再讀 `state.cfg["dev"]`；設備設定改由實驗本身在 `measure_fn` 外負責）。
2. `update_hook(ir, raw)` 在每個 round 結束時被 `measure_fn` 呼叫，更新 pbar 並呼叫 `state.set_value()`（觸發 liveplot）。
3. 最終 `raw2signal_fn(final_raw)` → `state.set_value(signal)`。

**`dynamic_pbar`**：若為 `True`（被父節點的 `BatchTask`/`Scan` 等設定），pbar 在 `run()` 內創建並在退出時關閉（leave=False），否則在 `init()` 創建（leave=True）。

**`pbar_n`**：可透過 `task.set_pbar_n(n)` 動態更新（用於 rounds 數在 init 時未知的場景）。

**Generic typing**：`Task` 的 dtype generic default 屬於 PEP 696 契約，透過 `typing_extensions` 在 Python 3.12 / 3.13 維持相同型別介面；runner runtime API 不因 Python profile 改變。

---

### `BatchTask`（`batch.py`）

並列（sequential）執行一組命名子任務，結果是 `dict[key, child_result]`。

```python
BatchTask({
    "exp_a": task_a,
    "exp_b": task_b,
})
```

- `init()` 呼叫每個子任務的 `init(dynamic_pbar=True)`。
- `run()` 依序執行每個子任務 `task.run(state.child(name))`，顯示外層 pbar（N tasks）。每個子任務的執行委派給 `_run_child(task, state)` hook（預設直接 `task.run(state)`）。
- `cleanup()` 呼叫所有子任務的 `cleanup()`。

---

### `RetryBatchTask`（`batch.py`）

`BatchTask` 的變體：每個子任務各自重試失敗。

```python
RetryBatchTask({"t1": task_a, "t2": task_b}, retry_time=3)
```

- 繼承 `BatchTask` 不改 `run()`（因此 `state.is_stop()` 短路與外層 pbar 行為一致），只覆寫 `_run_child` hook：每個子任務改走 `run_with_retries(task, state, retry_time, dynamic_pbar=True)`，即失敗時 `cleanup()` + `init()` 重置後重試，最多 `retry_time` 次。
- `retry_time=0`（預設）等同 `BatchTask`（不重試）。
- 重試是 **per-child**：子任務 A 重試耗盡才向上拋出，不影響尚未開始的子任務 B 的重試預算。`autofluxdep` / `overnight` 的 Executor 用它對 flux/time sweep 下的每個量測加重試。

---

### `Scan`（`soft.py`）

在一組值上掃描子任務，結果是 `list[child_result]`。

```python
task.scan(
    name="gain",
    values=[100, 200, 300],          # 掃描值
    before_each=lambda i, state, v: state.cfg.modules.readout.set_param("gain", v),
)
```

- `init()` 只建立掃描層 pbar，並呼叫 `sub_task.init(dynamic_pbar=dynamic_pbar)`。
- `before_each` 是必要參數（無預設值，非 `Optional`）；`run()` 在每步無條件呼叫 `before_each(i, state, v)`，再執行 `sub_task.run(state.child(i))`。

---

### `RepeatOverTime`（`repeat.py`）

固定次數重複子任務，支援步間等待。

```python
task.repeat(name="T1", times=10, interval=60.0)  # 每 60 秒重複一次
```

內部建構子為 `RepeatOverTime(name, num_times, task, interval=0.0)`。

- `run()` 在每步開始前設 `state.env["repeat_idx"] = i`，子任務可讀取目前迭代索引。
- `run()` 每步等待直到 `time.time() - start_t >= interval`，再執行子任務。
- 若 interval > 0 且有剩餘等待時間，會呼叫 `state._trigger_update_hook()` 強制刷新 liveplot（透過 `MinIntervalFunc.force_execute()` context）。
- 額外有一個 `time_pbar`（Passing Time 進度條），顯示每步之間的等待進度。

---

### `ReTryIfFail`（`repeat.py`）

捕捉子任務拋出的例外並重試。

```python
task.auto_retry(max_retries=3)
```

失敗時呼叫 `cleanup()` + `init()` 重置子任務狀態後重試。若耗盡 max_retries 次仍失敗，向上重新拋出。

---

### `run_with_retries`（`repeat.py`）

獨立函式，提供與 `ReTryIfFail` 相同的重試邏輯，方便在不需包裝的場景直接呼叫，也是 `RetryBatchTask._run_child` 的底層。

---

### `SignalBuffer` / `Schedule` / `ProgramBuilder`（`schedule.py`）

`Schedule` 是新的 run scope：入口 deepcopy cfg、共享 env、持有 `StopSignal`，並提供 host-side `scan(...)` 與 leaf `prog_builder(...)`。`SignalBuffer` 是 result storage、liveplot update surface 與 update 節流 owner；對外保留 `buffer` / `signals_buffer` 名稱，因為它不是裸 numpy array。`ProgramBuilder` 是 direct acquire builder，負責建立 `ModularProgramV2`、注入 `round_hook` / stop checkers、執行 retry 並透過 buffer/slot 寫回結果；它不再把 acquire 包成 `measure_fn` / `Task`。

```python
signals_buffer = SignalBuffer(
    (len(freqs),),
    on_update=lambda data: viewer.update(freqs, signal2real(data)),
)
with Schedule(cfg, signals_buffer) as sched:
    freq_sweep = sched.cfg.sweep.freq
    sched.cfg.modules.readout.set_param("freq", sweep2param("freq", freq_sweep))

    _ = (
        sched.prog_builder(soc, soccfg)
        .add_reset(
            "reset",
            sched.cfg.modules.reset,
        )
        .add(
            PulseReadout("readout", sched.cfg.modules.readout),
        )
        .declare_sweep("freq", freq_sweep)
        .build_and_acquire()
    )
```

- `sched.scan(...)` / `sched.repeat(...)` 表示 host-side Python loop；`ProgramBuilder.declare_sweep(...)` 表示 program-side QICK loop，兩者不混用。
- `sched.repeat(name, times, interval)` 會在每步設定 `env["repeat_idx"]`，並在 interval 等待期間檢查 `StopSignal`；若等待期間需要強制刷新 liveplot，由 caller 對相關 `SignalBuffer` 呼叫 `trigger_update()`。
- `sched.batch({key: callable}, retry=N)` 執行 replayable child callable，retry 是 per-child；每個 child 收到獨立 deepcopy cfg 與共享 env，`KeyboardInterrupt` 視為 stop，不 retry 且不繼續後續 child。`batch()` 回傳 child return value dict，buffer 寫入仍由 child 透過 `SignalBuffer` / `SignalSlot` 顯式完成。
- `ProgramBuilder.build()` 回傳已建立的 program；`run_program(program)` 執行 caller 傳入的既有 integrated-acquire program 並更新 owner 對應的 buffer slot；`build_and_acquire()` 是先 build 再 run 的 convenience。
- Decimated trace 使用獨立方法：`run_program_decimated(program)` 與 `build_and_acquire_decimated()`。不要用字串或 flag 在同一個 acquire 方法裡切換 `acquire` / `acquire_decimated`。
- `Schedule.register_buffer(...)` 只用於 program build 後才知道 buffer shape 的案例（例如 decimated trace 需要 `program.get_time_axis(...)`）；它只把 caller 建好的 `SignalBuffer` 納入 default acquire target，不接管 `on_update`。
- host soft sweep 若需要 per-point program cache，由實驗在 `run()` 裡維護 dict：cache miss 時 `builder.build()`，每次 acquire 時 `builder.run_program(program)`。`ProgramBuilder` 不擁有 cache，也不判斷 cache key 的 cfg/module 等價性。
- `build_and_acquire(...)` / `run_program(...)` 不接受 `reps` / `rounds` 參數；builder 先從 owner cfg 建出 program，再由 `program.cfg_model.rounds` 驅動 pbar 與 acquire round 數。需要不同 reps/rounds 時，先在 `sched.cfg` 或 `step.cfg` 上改欄位再 build。
- `ProgramBuilder` 要求顯式 `.add(...)` / `.add_reset(...)` / `.add_pulse(...)` / `.add_readout(...)` 宣告 modules，並把 modules 傳給 `ModularProgramV2`；不再支援未宣告 modules 時由 `OneToneProgram` / `TwoToneProgram` / custom program 自行從 cfg 生成 modules。
- `sched.prog_builder(..., program_cls=...)` 是窄口徑 test seam；常規實驗不傳此參數，預設使用 `ModularProgramV2`。
- 預設只有一個 `SignalBuffer` 時，leaf acquire 會依目前 builder owner 的 path 寫入對應 slot；top-level builder 寫整個 buffer，step builder 寫該 step view，因此 public acquire API 不需要 `into=` 參數。
- 巢狀 host loop 的 `ScheduleStep.path` 已包含父層 index，所以內層 step 仍用 `signals_buffer[step]`；不要再額外寫成 `signals_buffer[parent.index, step]`。
- `SignalBuffer(on_update=...)` 是唯一 live update callback 入口；`SignalBuffer.set(...)` / `SignalSlot.set(...)` 寫入後自動觸發 update，也可用 `trigger_update()` 手動刷新。
- `SignalBuffer` 預設使用 complex128；常規 signal buffer 省略 `dtype=`，只有非預設 dtype 才顯式宣告。
- `SignalSlot.value` 回傳目前 writable view，供 `snr_checker(...)` 這類只需讀取目前 slot 的 utility 沿用，不代表 slot 擁有獨立結果生命週期。
- `Schedule` 不讀 Task runner 的 global state；需要跨 thread stop 時，把同一個 `StopSignal` 傳給 `Schedule(..., stop=stop)`，或在 worker closure 外層使用 `schedule_stop_scope(StopSignal(stop_event))` 讓 scope 內新建的 `Schedule` 共用 stop signal。scope 內也可呼叫 `sched.set_stop()`。尚未移除的 `run_task(...)` / `MeasureSession(...)` 在沒有顯式 `stop_flag` 時也會讀同一個 `schedule_stop_scope`，這只是 Task tree 遷移期的相容橋。
- `set_raw2signal_fn(...)` 或 acquire 方法的 `raw2signal_fn=...` 可覆寫 raw-to-signal；integrated acquire 未指定時使用 `default_raw2signal_fn`，decimated acquire 未指定時使用 `default_decimated_raw2signal_fn`。

---

### `MultiMeasurementExecutor`（`multi_executor.py`）

不是 task tree 節點，而是 **Executor 共用 scaffold**：把「同時跑多個量測 + 合併 live plot（可選錄 FFmpeg 動畫）」的共用機制收在一處，供 `autofluxdep` / `overnight` 的 Executor 繼承（這兩個子類別與各自的 `MeasurementTask` ABC 住在 app 模組，不在 runner）。

- 提供：`add_measurements`、`record_animation(path)`（FFMpeg facet，缺 `ffmpeg` 即 Fast-Fail）、`make_ax_layout` / `make_plotter`（依各量測 `num_axes()` 自動排版 subplot），以及 `_run_with_plotting(task, cfg, env_dict)`（在 plotter context 內呼叫 `run_task`，並在 `finally` 收尾動畫）。
- 子類別只負責各自的 `run()`（不同的外層 driver 與 cfg/env 前置），共用此處的版面 / plotter / 錄製機器。
- 對 task 的需求以結構性 `PlottableMeasurement` Protocol 表達（`num_axes` / `make_plotter` / `update_plotter`），讓基底不必把兩個 app 的 `MeasurementTask` ABC 強行合併。

---

### `MeasureSession` frontend（`session.py`）

`MeasureSession` 是既有 orchestration frontend，讓 caller 可以用一般 Python `for` loop 寫 scan / repeat / batch，同時保留 `Task + measure_fn` 葉節點 seam。一般 `experiment/v2` 實驗不再使用它；它保留為 runner 層的已測試舊 frontend 與低階 Task API 參考。

```python
with MeasureSession(cfg) as run:
    signals_buffer = run.buffer(
        (len(values),),
        on_update=lambda data: plot(values, data),
    )
    for step in run.scan("gain", values):
        step.cfg.modules.readout.set_param("gain", step.value)
        signals_buffer[step].measure(measure_fn, pbar_n=step.cfg.rounds)
```

責任邊界：

- `MeasureSession` 在入口 `deepcopy(init_cfg)`；`scan` / `repeat` / `batch` child 也各自 deepcopy parent cfg，讓 step mutation 不外洩。`scan` 接受一般 iterable；caller 不需要為 ndarray sweep values 預先 `.tolist()`。
- `env` 是整個 session 共用的 mutable dict；`repeat` 設定 `env["repeat_idx"]`。
- `MeasureBuffer` 管 ndarray result；`buffer.measure(...)` 量整個 buffer，`buffer[index_or_step].measure(...)` 量 view。傳入 `MeasureStep` 時，slot 使用該 step 的 isolated cfg；`.at(...)` 保留為等價的顯式方法。
- `MeasureBuffer(on_update=...)` 是常規 liveplot seam，callback 接收該 buffer 的 full ndarray。`MeasureSession(on_update=...)` 則是進階 snapshot hook，提供 full `root_data`、目前 cfg 與 shared env view；slot partial update 仍透過既有 `TaskState.set_value()` 寫入 ndarray view。
- `batch({name: callable}, retry=N)` 只接受 replayable child callable；retry 是 per-child，不能用不可重放的 `with` block 模式實作 retry。child return value 會進 `batch()` 的回傳 dict；若 child 也建立 buffer，`root_data[name]` 保留該 buffer subtree。
- `KeyboardInterrupt` 在 leaf / batch child 視為 early stop，不 retry，保留 partial data；一般 `Exception` 依 leaf 或 batch child retry budget 重試。
- 同一 session 只允許一個 unnamed root buffer；多個 root result 使用 named buffers 或 batch dict root。

Scope boundary：

- 一般 `experiment/v2` run orchestration 使用 `SignalBuffer` + `Schedule` + `ProgramBuilder`，raw-shot path 也使用 `Schedule` 做 cfg/stop/buffer orchestration。
- `autofluxdep` 與 `overnight` executors 仍由各自 executor + `MultiMeasurementExecutor` scaffold 擁有，並保留 Task tree。
- `MeasureStep.buffer(...)` 的 unnamed child buffer 用於 replayable batch child；scan / repeat loop 的常規寫法是先在 session root 預配置 buffer，再用 `buffer[step]` 寫入。

---

## 執行入口：`run_task`（`base.py`）

```python
result = run_task(
    task,            # AbsTask
    init_cfg,        # ExpCfgModel 實例（會 deepcopy）
    env_dict,        # 可選環境字典
    on_update,       # 可選更新回呼（會套 min_interval 節流）
    update_interval, # 節流間隔（秒），預設 0.1
    stop_flag,       # 可選 threading.Event，注入 state._stop_flag
)
```

**執行流程**：

1. `deepcopy(init_cfg)` → 建立 `TaskState`（path=()，即根節點），`stop_flag` 注入 `state._stop_flag`。
2. `task.init(dynamic_pbar=False)` — **不傳 state**。
3. `task.run(state)`。
4. `task.cleanup()`（在 `finally` 中確保執行）。
5. `KeyboardInterrupt` 只 log warning 不 raise；其他 Exception log + traceback 後 **會 raise**。
6. 回傳 `state.root_data`（中斷後回傳已完成 rounds 的部分資料，未完成部分保持 NaN）。

---

## 跨線程中斷

中斷機制由三層組成：

**1. `TaskHandle` / `run_task(stop_flag=...)`（Task tree 路徑）**

```python
import threading
from zcu_tools.experiment.v2.runner import run_task, TaskHandle

stop_event = threading.Event()
handle = TaskHandle(stop_event)

# worker thread 中執行
result = run_task(task, cfg, stop_flag=stop_event)

# main thread 任意時機呼叫
handle.cancel()          # 設置 stop_event
handle.is_cancelled()    # 查詢是否已取消
```

`TaskHandle` 是 `threading.Event` 的薄包裝。`stop_flag` 注入 `state._stop_flag`，透過 `child()` 傳遞到整棵任務樹。若 `run_task(...)` 沒有收到顯式 `stop_flag`，但 worker 已位於 `schedule_stop_scope(StopSignal(stop_event))` 內，runner 會使用該 `StopSignal` 的底層 event；這讓 GUI Stop 在 Task tree 移除前仍能覆蓋過渡路徑。

**2. `StopSignal` / `schedule_stop_scope(...)`（Schedule 路徑）**

一般 Experiment 的 `run()` 內部會自行建立 `Schedule`；GUI 這類外層呼叫端若要把 Stop 按鈕的 `threading.Event` 傳進所有 Schedule，可在 worker closure 外包一層 `schedule_stop_scope(...)`：

```python
import threading
from zcu_tools.experiment.v2.runner import StopSignal, schedule_stop_scope

stop_event = threading.Event()
with schedule_stop_scope(StopSignal(stop_event)):
    result = adapter.run(request, schema)

# 另一個 thread（如 GUI 按下停止鍵）：
stop_event.set()
```

scope 內新建的 `Schedule` 若沒有顯式 `stop=` 參數，會使用目前 context 的 `StopSignal`；scope 結束後 context 會恢復。若 caller 直接擁有 Schedule，也可用 `Schedule(cfg, buffer, stop=StopSignal(stop_event))` 明確注入。

---

**3. 容器節點短路**

Task tree 的 `BatchTask`、`Scan`、`RepeatOverTime` 在每個子任務開始前各自呼叫 `state.is_stop()`，若已設置則立即 `break`（不繼續剩餘子任務）。`RepeatOverTime` 的等待迴圈也加入 `is_stop()` 檢查，確保 interval 等待期間也能及時退出。

Schedule path 的 `sched.scan(...)`、`sched.repeat(...)`、`sched.batch(...)` 也在每個 host step 前檢查 `sched.is_stop()`；`repeat` 的 interval 等待期間同樣會檢查 stop。

中斷後，`run_task` 回傳 `state.root_data`（已完成的子節點有真實資料，未完成的保持 NaN）。

---

**4. `EarlyStopMixin.finish_round()`（program 層）**

Schedule 的 `ProgramBuilder` 會自動把 `sched.is_stop` 加入 acquire 的 `stop_checkers`。Task leaf 的 `measure_fn` 在建構 program 後、呼叫 `acquire()` 前，將 `ctx.is_stop` 加入 `stop_checkers`：

```python
def measure_fn(ctx, update_hook):
    prog = ModularProgramV2(...)
    return prog.acquire(
        soc,
        round_hook=update_hook,
        stop_checkers=[ctx.is_stop],   # ← 每個 round 結束後 EarlyStopMixin 會呼叫
    )
```

`EarlyStopMixin.finish_round()` 在每個 round 完成後（`RoundHookMixin` 寫入 signal 之後）依序呼叫 `stop_checkers`，任一回傳 `True` 即終止 `acquire()` 迴圈。中斷粒度為 **round-level**。

**SNR early stop 也走同一路徑**（`experiment/v2/utils/snr.py`）：

```python
from zcu_tools.experiment.v2.utils import snr_checker

stop_checkers=[ctx.is_stop, snr_checker(ctx, snr_threshold, signal2real_fn)]
```

`snr_checker` 從 `ctx.value` 讀最新 signal，達到 `snr_threshold` 後回傳 `True`。`snr_threshold=None` 時回傳 `lambda: False`，呼叫端無需 if-guard。

---

## 結果樹形狀範例

```
BatchTask({
    "rabi": Task(result_shape=(100,)),         → root_data["rabi"] = ndarray(100,)
    "ramsey": Task(result_shape=(200,)).repeat("t", 5)
})
→ root_data = {
    "rabi": ndarray(100,),
    "ramsey": [ndarray(200,), ndarray(200,), ..., ndarray(200,)]  # 5 次
}
```

子任務透過 `state.child("rabi")` 或 `state.child(0)` 取得子 `TaskState`，寫入時操作的是 `root_data["rabi"]` 或 `root_data["ramsey"][0]`（in-place）。

---

## `dynamic_pbar` 機制

目的：被 `BatchTask` / `Scan` / `RepeatOverTime` 包裝時，子任務的 pbar 應在每次 `run()` 呼叫後關閉（避免同時存在多條進度條），而最外層任務的 pbar 應保留到完成。

| 呼叫情境 | `dynamic_pbar` | pbar 創建位置 | `leave` |
|---------|----------------|--------------|---------|
| `run_task` 直接呼叫 | `False` | `init()` | `True` |
| 被 `BatchTask` 包裝 | `True` | `run()` 開頭，`run()` 結尾關閉 | `False` |

---

## 注意事項

- `env["repeat_idx"]` 由 `RepeatOverTime.run()` 在每步開始前設定，子任務可通過 `state.env` 讀取。
- 結果陣列以 `np.nan` 初始化，可在 liveplot 中安全顯示（未完成的點保持 NaN）。
- `run_task` 捕捉 `KeyboardInterrupt` 時只 log 不 re-raise（早停），但其他 Exception 會 re-raise，讓呼叫端可以感知失敗。
- `run_with_retries` 可獨立呼叫，也是 `ReTryIfFail` 的底層實作，共用相同的重試邏輯。
