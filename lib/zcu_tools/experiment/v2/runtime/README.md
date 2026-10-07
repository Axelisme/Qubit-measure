# `zcu_tools.experiment.v2.runtime` — experiment runtime

**Last updated:** 2026-10-08，移除舊 Autofluxdep 與 overnight 入口

`runtime/` 提供 experiment/v2 的 Python-like acquisition runtime。一般實驗用
`SignalBuffer` / `Schedule` / `ProgramBuilder` 編排 host-side loop 與 program
acquire。`ResultTree` 提供 structured `BufferProtocol`，可交給同一個
`Schedule` 編排 outer loop。

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
`trigger_update(step, flush=False)`。`SignalBuffer` 實作這個 protocol；`ResultTree`
提供 structured result buffer。`Schedule` 不直接宣告 result tree /
update callback；child-local `SignalBuffer` 寫入時同步更新 root buffer 的 `data` 中對應
path，並觸發該 buffer 的 update hook。

### `ResultTree`

`ResultTree` 是 structured buffer，root data 形狀為
`list[dict[measurement_name, Result]]`。它提供：

- `ResultNode` typed handle：`tree.at(i).child("task").child("field")` 可定位 result
  subtree，`set(value, flush=True)` 可直接寫入並觸發 update；若該 update 會送到
  subscriber，`ResultTree` 必須有 run env（caller 在建構時傳入），否則 fast-fail。
- child-local buffer：caller 以 `state.child("raw_signals", cfg=cfg).buffer(...)`
  建立 `SignalBuffer`，buffer 寫入時會回填 ResultTree leaf。
- per-measurement subscription：plotter 訂閱 `tree.measurement_node(name)`，
  callback 收 `ResultUpdateEvent`，包含 `measurement_name`、`outer_index`、
  `outer_value`、`env`、`node`、stacked `result` 與 `flush`。
- root-level `ResultTree.trigger_update(flush=True)` 會對所有已訂閱 measurement 廣播
  flush event；step-level update 只觸發 path 對應的 measurement。
- per-measurement cache：`measurement_result(name)` 只 merge 該 measurement；更新某個
  measurement 時只 invalidates 該 measurement cache。

一般 `SignalBuffer` experiment 不使用 ResultTree，public `on_update(data)` 行為不變。

### `Schedule`

`outcome.py` 擁有完成狀態值與 exception reason 格式化。`Schedule` 擁有狀態轉換、stop token 與 partial buffer 的生命週期。`RunStatus`、`ScheduleOutcome` 仍可從既有 runtime 與 schedule 入口匯入。

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
- `sched.batch({key: callable})` 執行 replayable child callable；child 取得獨立
  deepcopy cfg 與同一個 typed env。batch child 出現一般例外或 `KeyboardInterrupt`
  時會標記 `Schedule.outcome`、停止後續 child，並回傳已完成 child 的 partial
  result；batch 本身不做 per-child retry。
- `ScheduleStep.path` 會累積巢狀 host loop index，所以預設 single-buffer acquire 可
  依 owner path 自動寫入對應 slot，不需要 `into=` 參數。
- `with Schedule(cfg, result_buffer, stop=context.cancel_signal) as sched` 可編排任何實作 `BufferProtocol` 的 result
  tree；root result buffer 目前最多一個，多 root buffer 會 fast-fail。
  `step.child("field", cfg=program_cfg).buffer(shape)` 會建立 child-local default
  `SignalBuffer`，buffer 寫入時同步回 `result_buffer.data` 並觸發 update。
- `ScheduleStep.value` 是 scan/repeat/batch 的 control-flow coordinate；result tree 由
  `step.data` 讀取、`step.set_data(..., flush=True)` 寫入，需要 ndarray slot 時用
  `step.array_data` 做型別邊界。`flush=True` 對 ResultTree 表示立即送出 node event；
  對一般 `SignalBuffer` 不改變 `on_update(data)` public shape。
- `sched.trigger_update(flush=True)` 是 root-level update，會對 root buffer 傳
  `step=None`；step-level update 才帶 `ScheduleStep`。

### `ProgramBuilder`

`ProgramBuilder` 是 QICK program 的 builder 與 acquire adapter。`acquire_cancel.py` 擁有 acquire-local stop 與借用 run stop 的合成旗標，讓 data-driven early stop 不會改變 run outcome。

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
- `progress_label=` 可覆寫 acquire rounds progress bar label；`progress_leave=` 可覆寫
  root-vs-step 預設 close/retain policy。

---

## Stop 與 Retry

- `StopSignal` 是唯一 stop token，定義於 `experiment.stop_signal`。Caller 每次 run 建立一個，
  必須透過 `Schedule(..., stop=context.cancel_signal)` 顯式傳入。Runtime 不查 ambient
  scope，也不補建隱式 token。
- `Schedule` scope 接住一次 `KeyboardInterrupt`，包含 scan/repeat 的 caller loop body，將 shared signal 記為 `interrupted` 並保留 buffer，讓 with 後的 return 組出部分結果。其他例外不改變原有傳播。Scope 外中斷不受保護，Notebook 將正常返回的 interrupted 表示為 stopped；其他 caller 的 `raise_if_error()` 仍拋錯。
- `Schedule` 的 scan/repeat/batch 會在 host step 前檢查 stop；`ProgramBuilder` 會把
  acquire-local composite `cancel_flag` 傳給 program acquire：它讀取 `StopSignal` 的 external
  stop，但 `round_hook` 內的 data stop 只停止目前 program acquire，不會污染 `Schedule.outcome`。
  `StopSignal` 對外只暴露 external cancel flag contract：`is_set()` / `set()`；retry reset
  仍使用 `clear_stop()` 清除 stop flag 與 transient error cause。
- `ProgramBuilder` 的 data-driven early stop 走 `stop_condition=`：internal
  round hook 先把 completed round 寫入 buffer，再執行 condition；命中時呼叫
  acquire-local `cancel_flag.set()`，保留該 completed round 並停止下一 round；`Schedule.outcome`
  仍是 `completed`。
- `Schedule.outcome` 記錄 run 結果（資料保留不等於 operation 成功）：`completed`、`stopped`、`interrupted` 或
  `failed`。stop / `KeyboardInterrupt` / program build、acquire 或分析例外都會設定 `StopSignal`
  並保留目前已寫入 buffer / result tree 的 partial result；已寫入的 slot 保留，
  未完成的 slot 維持 NaN 初始化值。若 first round 尚未完成就 stop，program acquire
  以 stopped partial 例外結束，runner 將 outcome 標記為 `stopped` 並保留 NaN partial。
- `StopSignal` 也攜帶第一個 `failed` / `interrupted` cause。這不改
  `ProgramBuilder` 的 partial-result contract；它提供給持有同一個 RunContext 的 consumer（例如
  measure-gui run policy）在 experiment adapter 回傳後呼叫 `raise_if_error()`，把
  runner 內部錯誤轉成 operation failure。`stopped` 不記錄 error cause；retry reset 會同時
  清掉 stop flag 與 transient error cause。
- `ProgramBuilder.build_and_acquire(..., retry=N)` /
  `run_program(..., retry=N)` 是單次 program acquire retry；`Schedule.batch(...)` 不提供
  per-child retry。
- Program build/acquire 在每次 attempt 前與 failed attempt retry 前都重新檢查
  `StopSignal`；stop / interrupted outcome 不會被 retry，也不會被 retry reset 清掉。

---

## 測試

`tests/experiment/v2/runtime/test_flow.py` 覆蓋 typed env、SignalBuffer、ProgramBuilder、
host scan/repeat/batch、ProgramBuilder retry、`ScheduleOutcome` 與 decimated acquire。
`test_schedule_lifecycle.py` 覆蓋 context 使用邊界、shared stop token、host body 中斷的
partial/NaN 保存、其他 exception 傳播與 repeat stop。兩個檔案重用同一個 test cfg builder，
案例仍經公開 Schedule／buffer／StopSignal 接縫驗證，不直接測 outcome 的內部 helper。
`test_result_tree.py` 覆蓋 ResultTree node set、child buffer、subscription、flush 與
ordinary SignalBuffer regression。單一實驗模組只保留 runtime
整合型檢查，不再為每個資料型 experiment 建獨立 runner 測試。
