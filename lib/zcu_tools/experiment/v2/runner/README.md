# `zcu_tools.experiment.v2.runner` — experiment runtime

**Last updated:** 2026-07-02 — Schedule-only runtime

`runner/` 提供 experiment/v2 的 Python-like acquisition runtime。一般實驗用
`SignalBuffer` / `Schedule` / `ProgramBuilder` 編排 host-side loop 與 program
acquire；executor 類流程用 `MultiMeasurementExecutor` / `MeasurementContext` 共用
liveplot、retry 與 stop scaffold。舊 Task tree runtime 不再保留。

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

### `Schedule`

`Schedule` 是一次 run 的 scope：入口 deepcopy cfg、共享 env、持有 `StopSignal`，並
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

- `sched.scan(...)` / `step.scan(...)` 表示 host-side Python loop。
- `sched.repeat(name, times, interval)` 表示 host-side repeat；等待期間會檢查
  `StopSignal`，需要刷新 liveplot 時由 caller 呼叫相關 buffer 的 `trigger_update()`。
- `sched.batch({key: callable}, retry=N)` 執行 replayable child callable；retry 是
  per-child，child 取得獨立 deepcopy cfg 與共享 env。
- `ScheduleStep.path` 會累積巢狀 host loop index，所以預設 single-buffer acquire 可
  依 owner path 自動寫入對應 slot，不需要 `into=` 參數。

### `ProgramBuilder`

`ProgramBuilder` 是 QICK program 的 builder 與 acquire adapter。

- `build()` 建立並回傳 program。
- `run_program(program)` 執行 caller 傳入的 integrated-acquire program。
- `build_and_acquire()` 是 `build()` 後立刻 `run_program(...)` 的 convenience。
- Decimated trace 使用獨立方法：`run_program_decimated(program)` 與
  `build_and_acquire_decimated()`。
- `build_and_acquire(...)` / `run_program(...)` 不接受 `reps` / `rounds`；builder
  建出的 program 讀取 cfg-owned `program.cfg_model.reps` / `rounds`。
- `set_raw2signal_fn(...)` 或 acquire 方法的 `raw2signal_fn=...` 可覆寫 raw
  conversion；integrated acquire 預設使用 `default_raw2signal_fn`，decimated acquire
  預設使用 `default_decimated_raw2signal_fn`。
- `program_cls=` 是測試 seam；常規實驗不傳，預設使用 `ModularProgramV2`。

---

## Executor Scaffold

`MultiMeasurementExecutor` 服務 `autofluxdep` / `overnight` 這類外層 workflow：
它負責 combined liveplot layout、FFmpeg animation、root `MeasurementContext` 建立、
measurement init/cleanup，以及 per-measurement retry helper。

`MeasurementContext` 是 executor-local context，語意接近 `Schedule`：

- `root_data` 指向整個結果樹。
- `path` 定位目前 measurement 的結果節點。
- `cfg` 在 child context 中 deepcopy 隔離。
- `env` 是 executor run 內共享 mutable dict。
- `stop` 是 `StopSignal`，與 leaf `Schedule(..., stop=state.stop)` 共用。
- `set_value(...)` in-place 更新目前結果節點並觸發 throttled plot update。

`context_signal_buffer(ctx, shape, ...)` 建立與 `MeasurementContext` 綁定的
`SignalBuffer`；buffer 寫入時會自動同步到 context 的目前結果節點。executor leaf
measurement 仍用 `Schedule` 做 program acquire，外層 flux/time loop 則保留直接 Python
loop，因為 executor cfg (`FluxDepCfg` / `OvernightCfg`) 不是 `ProgramV2Cfg`。

---

## Stop 與 Retry

- `StopSignal` 是唯一 stop token；GUI worker 用 `schedule_stop_scope(StopSignal(event))`
  讓 scope 內所有 `Schedule` 與 executor context 共用同一個 stop event。
- `Schedule` 的 scan/repeat/batch 會在 host step 前檢查 stop；`ProgramBuilder` 會把
  `sched.is_stop` 注入 program acquire 的 `stop_checkers`，中斷粒度為 round-level。
- `Schedule.batch(..., retry=N)` 支援一般 `ProgramV2Cfg` schedule 的 replayable child
  retry。
- `autofluxdep` / `overnight` executor 的外層 retry 由
  `MultiMeasurementExecutor._run_measurement_with_retries(...)` 提供；retry 預算是
  per-measurement、per-flux/time-step。

---

## 測試

`tests/experiment/v2/runner/test_flow.py` 覆蓋 Schedule、SignalBuffer、ProgramBuilder、
host scan/repeat/batch、retry、stop 與 decimated acquire。單一實驗模組只保留 runtime
整合型檢查，不再為每個資料型 experiment 建獨立 runner 測試。
