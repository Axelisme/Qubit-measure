# `zcu_tools.experiment.v2.runner` — experiment runtime

**Last updated:** 2026-07-02 — unified Schedule executor runtime

`runner/` 提供 experiment/v2 的 Python-like acquisition runtime。一般實驗用
`SignalBuffer` / `Schedule` / `ProgramBuilder` 編排 host-side loop 與 program
acquire；executor 類流程用 `ResultBuffer` 持有 root result tree，再交給同一個
`Schedule` 編排 outer loop，並用
`MultiMeasurementExecutor` 共用 liveplot、retry 與 measurement lifecycle。舊 Task tree
runtime 不再保留。

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

### `ResultBuffer`

`ResultBuffer(data, on_update=...)` 是 structured result tree 的 storage 與 update
owner，主要給 executor workflow 使用。`Schedule` 可以接收一個 `ResultBuffer`，但
不直接宣告 result tree / update callback；child-local `SignalBuffer` 寫入時同步更新
`ResultBuffer.data` 中對應 path，並觸發 `ResultBuffer` 的 update hook。

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

- `Schedule` 的 cfg 是泛型，可以是 experiment cfg；`ProgramBuilder` 只會把 builder
  cfg 投影成 `ProgramV2Cfg` 給 program。若 cfg 已是 `ProgramV2Cfg` instance/subclass
  則保留原型別；若 mapping/object 沒有任何 `ProgramV2Cfg` 欄位，builder 會 fast-fail。
- `sched.scan(...)` / `step.scan(...)` 表示 host-side Python loop。
- `sched.repeat(name, times, interval)` 表示 host-side repeat；等待期間會檢查
  `StopSignal`，需要刷新 liveplot 時由 caller 呼叫相關 buffer 的 `trigger_update()`。
- `sched.batch({key: callable}, retry=N)` 執行 replayable child callable；retry 是
  per-child，child 取得獨立 deepcopy cfg 與共享 env。
- `ScheduleStep.path` 會累積巢狀 host loop index，所以預設 single-buffer acquire 可
  依 owner path 自動寫入對應 slot，不需要 `into=` 參數。
- `with Schedule(cfg, result_buffer) as sched` 可編排 executor result tree；
  `step.child("field", cfg=program_cfg).buffer(shape)` 會建立 child-local default
  `SignalBuffer`，buffer 寫入時同步回 `result_buffer.data` 並觸發 update。
- `ScheduleStep.value` 是 scan/repeat/batch 的 control-flow coordinate；result tree 由
  `step.data` 讀取、`step.set_data(...)` 寫入，需要 ndarray slot 時用 `step.array_data`
  做型別邊界。

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

`MultiMeasurementExecutor` 服務 `autofluxdep` / `overnight` 這類外層 workflow：
它負責 combined liveplot layout、FFmpeg animation、measurement init/cleanup，以及
per-measurement retry helper；root result tree 與 update callback 由 `ResultBuffer`
持有，path、env、stop 與 control flow 由 `Schedule` 持有。

典型 executor 格式：

```python
def run_loop(root_sched: Schedule[FluxDepCfg]) -> None:
    for i, (flux, flux_step) in enumerate(root_sched.scan("flux", flux_values)):
        update_flux_context(i, flux_step, flux)
        flux_step.batch(
            {
                name: lambda child, measurement=measurement: (
                    self._run_measurement_with_retries(measurement, child, retry_time)
                )
                for name, measurement in self.measurements.items()
            }
        )
```

leaf measurement 取得 `ScheduleStep` 後，通常用
`raw_step = state.child("raw_signals", cfg=program_cfg)` 建立 program-owned cfg scope，再用
`raw_step.buffer(shape, dtype=...)` 建立與 result tree 綁定的 acquire buffer；接著直接
呼叫 `raw_step.prog_builder(...).build_and_acquire()`。

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

`tests/experiment/v2/runner/test_flow.py` 覆蓋 Schedule、SignalBuffer、ProgramBuilder、
host scan/repeat/batch、retry、stop 與 decimated acquire。單一實驗模組只保留 runtime
整合型檢查，不再為每個資料型 experiment 建獨立 runner 測試。
