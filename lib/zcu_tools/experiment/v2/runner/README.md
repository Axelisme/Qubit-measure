# `zcu_tools.experiment.v2.runner` — experiment runtime

**Last updated:** 2026-07-02 — executor ResultTree lifecycle

`runner/` 提供 experiment/v2 的 Python-like acquisition runtime。一般實驗用
`SignalBuffer` / `Schedule` / `ProgramBuilder` 編排 host-side loop 與 program
acquire；executor 類流程用 `ResultTree` 實作 executor-owned `BufferProtocol`，再交給同一個
`Schedule` 編排 outer loop，並用 `MultiMeasurementExecutor` 共用 result initialization、
per-measurement liveplot subscription、recording、retry、cleanup 與 `last_cfg` /
`last_result` lifecycle。舊 Task tree runtime 不再保留。

---

## 核心物件

### `SignalBuffer`

`SignalBuffer` 是 result storage、live update 入口與 update 節流 owner。

- `SignalBuffer(shape, dtype=np.complex128, on_update=..., update_interval=0.1)` 以
  NaN 初始化底層 ndarray。
- `buffer[step]` / `buffer.at(...)` 回傳 `SignalSlot`，slot 寫入的是 buffer 的
  writable view。
- `SignalBuffer.set(...)` / `SignalSlot.set(...)` 寫入後自動觸發 `on_update`。
- `trigger_update()` 用於沒有新資料寫入、但 caller 想強制刷新 liveplot 的等待期。

### `BufferProtocol`

`BufferProtocol` 是 `Schedule` 對 buffer 的唯一要求：buffer 提供 `data` 以及
`trigger_update(step, flush=False)`。`SignalBuffer` 實作這個 protocol；executor workflow
使用 `ResultTree` 作為 structured result buffer。`Schedule` 不直接宣告 result tree /
update callback；child-local `SignalBuffer` 寫入時同步更新 root buffer 的 `data` 中對應
path，並觸發該 buffer 的 update hook。

### `ResultTree`

`ResultTree` 是 executor-owned buffer，root data 形狀為
`list[dict[measurement_name, Result]]`。它提供：

- `ResultNode` typed handle：`tree.at(i).child("task").child("field")` 可定位 result
  subtree，`set(value, flush=True)` 可直接寫入並觸發 update。
- child-local buffer：executor leaf 以 `state.child("raw_signals", cfg=cfg).buffer(...)`
  建立 `SignalBuffer`，buffer 寫入時會回填 ResultTree leaf。
- per-measurement subscription：executor plotter 訂閱 `tree.measurement_node(name)`，
  callback 收 `ResultUpdateEvent`，包含 `measurement_name`、`outer_index`、
  `outer_value`、`env`、`node`、stacked `result` 與 `flush`。
- per-measurement cache：`measurement_result(name)` 只 merge 該 measurement；更新某個
  measurement 時只 invalidates 該 measurement cache。

一般 `SignalBuffer` experiment 不使用 ResultTree，public `on_update(data)` 行為不變。

### `Schedule`

`Schedule` 是一次 run 的 scope：入口 deepcopy cfg、typed env、持有 `StopSignal`，並
建立 host-side control flow。

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
        .add_reset("reset", sched.cfg.modules.reset)
        .add_readout("readout", sched.cfg.modules.readout)
        .declare_sweep("freq", freq_sweep)
        .build_and_acquire()
    )
```

- `Schedule` 的 cfg 是泛型，可以是 experiment cfg；`ProgramBuilder` 只會把 builder
  cfg 投影成 `ProgramV2Cfg` 給 program。若 cfg 已是 `ProgramV2Cfg` instance/subclass
  則保留原型別；若 mapping/object 沒有任何 `ProgramV2Cfg` 欄位，builder 會 fast-fail。
- `Schedule` 的 env 也是泛型，使用 `with Schedule(cfg, buffer, env=RunEnv(...))`
  可讓 `sched.env.xxx` / `step.env.xxx` 取得 linter 型別支援。env 只放穩定 run
  dependencies（例如 soc、soccfg、ModuleLibrary、predictor）；scan/repeat 的動態
  value/index 不寫入 env，而是由 `ScheduleStep.value` / `ScheduleStep.index` /
  `ScheduleStep.path` 表示。
- `sched.scan(...)` / `step.scan(...)` 表示 host-side Python loop。
- `sched.repeat(name, times, interval)` 表示 host-side repeat；等待期間會檢查
  `StopSignal`，需要刷新 liveplot 時由 caller 呼叫相關 buffer 的 `trigger_update()`。
- `sched.batch({key: callable}, retry=N)` 執行 replayable child callable；retry 是
  per-child，child 取得獨立 deepcopy cfg 與同一個 typed env。
- `ScheduleStep.path` 會累積巢狀 host loop index，所以預設 single-buffer acquire 可
  依 owner path 自動寫入對應 slot，不需要 `into=` 參數。
- `with Schedule(cfg, result_buffer) as sched` 可編排任何實作 `BufferProtocol` 的 result
  tree；
  `step.child("field", cfg=program_cfg).buffer(shape)` 會建立 child-local default
  `SignalBuffer`，buffer 寫入時同步回 `result_buffer.data` 並觸發 update。
- `ScheduleStep.value` 是 scan/repeat/batch 的 control-flow coordinate；result tree 由
  `step.data` 讀取、`step.set_data(..., flush=True)` 寫入，需要 ndarray slot 時用
  `step.array_data` 做型別邊界。`flush=True` 對 ResultTree 表示立即送出 node event；
  對一般 `SignalBuffer` 不改變 `on_update(data)` public shape。

### `ProgramBuilder`

`ProgramBuilder` 是 QICK program 的 builder 與 acquire adapter。

- `build()` 建立並回傳 program。
- `run_program(program)` 執行 caller 傳入的 integrated-acquire program。
- `build_and_acquire()` 是 `build()` 後立刻 `run_program(...)` 的 convenience。
- Decimated trace 使用獨立方法：`run_program_decimated(program)` 與
  `build_and_acquire_decimated()`。
- `build_and_acquire(...)` / `run_program(...)` 不接受 `reps` / `rounds`；builder
  建出的 program 讀取 cfg-owned `program.cfg_model.reps` / `rounds`。
- `prog_builder(..., cfg=...)` 可覆寫 program cfg；未指定時繼承 owner cfg。若 cfg 已是
  `ProgramV2Cfg` instance 則原樣 deepcopy；若是 mapping / object，builder 只抽取
  `ProgramV2Cfg` 定義的 runtime 欄位；沒有 runtime 欄位時會要求 caller 明確傳入
  `ProgramV2Cfg()` 或有效 program cfg。
- `set_raw2signal_fn(...)` 或 acquire 方法的 `raw2signal_fn=...` 可覆寫 raw
  conversion；integrated acquire 預設使用 `default_raw2signal_fn`，decimated acquire
  預設使用 `default_decimated_raw2signal_fn`。
- `program_cls=` 是測試 seam；常規實驗不傳，預設使用 `ModularProgramV2`。

---

## Executor Scaffold

`MultiMeasurementExecutor` 服務 `autofluxdep` / `overnight` 這類外層 workflow。
base executor 擁有 common run lifecycle：建立 default outer result、combined liveplot
layout、FFmpeg writer、`ResultTree` subscriptions、`Schedule` scope、measurement
init/cleanup、per-measurement retry、stop handling、writer finish、figure close，以及
`last_cfg` / `last_result`。concrete executor 只提供 cfg/env 建立與 outer-loop policy。

典型 executor 格式：

```python
def run_loop(root_sched: Schedule[FluxDepCfg, FluxDepEnv]) -> None:
    for i, (flux, flux_step) in enumerate(root_sched.scan("flux", flux_values)):
        update_flux_context(i, flux_step, flux)
        self._run_measurement_batch(flux_step, retry_time)
```

leaf measurement 取得 `ScheduleStep` 後，通常用
`raw_step = state.child("raw_signals", cfg=program_cfg)` 建立 program-owned cfg scope，再用
`raw_step.buffer(shape, dtype=...)` 建立與 result tree 綁定的 acquire buffer；接著直接
呼叫 `raw_step.prog_builder(...).build_and_acquire()`。`autofluxdep` / `overnight`
的 caller 入口以 explicit keyword deps 提供穩定依賴，executor 在 run 內組完整 typed
env 給 root `Schedule`。

executor leaf contract 由 `runner/task.py` 擁有：

- `Acquirer[T_Cfg, T_Env, T_Result]`：`init` / `run` / `cleanup` /
  `get_default_result`。
- `TaskPlotter[T_Env, T_Result, T_PlotDict]`：axes、plotter 建立與
  `ResultUpdateEvent` 更新。
- `TaskPersister[T_Result, T_SaveAxis]`：save/load mechanics 的 save 端 contract。
- `MeasurementBundle`：executor 接受的完整 leaf contract。
- `ComposedMeasurementBundle`：把 acquire / plot / persistence 三段 component 組成 leaf。
- `MeasurementTask`：單一 class 直接實作完整 bundle 的 convenience base。

---

## Stop 與 Retry

- `StopSignal` 是唯一 stop token；GUI worker 用 `schedule_stop_scope(StopSignal(event))`
  讓 scope 內所有 `Schedule` 共用同一個 stop event。
- `Schedule` 的 scan/repeat/batch 會在 host step 前檢查 stop；`ProgramBuilder` 會把
  `sched.is_stop` 注入 program acquire 的 `stop_checkers`，中斷粒度為 round-level。
- `Schedule.batch(..., retry=N)` 支援一般 `ProgramV2Cfg` schedule 的 replayable child
  retry。
- `autofluxdep` / `overnight` executor 的外層 retry 由
  `MultiMeasurementExecutor._run_measurement_with_retries(...)` 提供；retry 預算是
  per-measurement、per-flux/time-step。

---

## 測試

`tests/experiment/v2/runner/test_flow.py` 覆蓋 Schedule、typed env、SignalBuffer、
ProgramBuilder、host scan/repeat/batch、retry、stop 與 decimated acquire。
`test_result_tree.py` 覆蓋 ResultTree node set、child buffer、subscription、flush 與
ordinary SignalBuffer regression；`test_multi_executor.py` 覆蓋 executor template lifecycle、
retry/stop partial result 與 composed bundle delegation。單一實驗模組只保留 runtime
整合型檢查，不再為每個資料型 experiment 建獨立 runner 測試。
