# `zcu_tools.experiment.v2.runtime` — experiment runtime

**Last updated:** 2026-10-25

**讀者：** framework-dev。Schedule、buffer、program acquire、停止與重試契約。

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
- `flush_pending_update()` 立即交付被節流、尚未呈現的更新。若 callback 已經完成，這個方法直接返回。顯式 `trigger_update(flush=True)` 則仍會強制刷新。

### `BufferProtocol`

`BufferProtocol` 是 `Schedule` 對 buffer 的唯一要求：buffer 提供 `data` 以及
`trigger_update(step, flush=False)`。`SignalBuffer` 實作這個 protocol；`ResultTree`
提供 structured result buffer。Result tree 與 update callback 由 buffer 宣告；child-local `SignalBuffer` 寫入時同步更新 root buffer 的 `data` 中對應
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

一般 `SignalBuffer` experiment 直接使用 public `on_update(data)` callback。

### `Schedule`

`outcome.py` 擁有完成狀態值與 exception reason 格式化。`Schedule` 擁有狀態轉換、stop token 與 partial buffer 的生命週期。`RunStatus`、`ScheduleOutcome` 仍可從既有 runtime 與 schedule 入口匯入。

`Schedule` 是 run 內的 scope：入口 deepcopy cfg、typed env、持有顯式 `Scope` child，並
建立 host-side control flow。

```python
signals_buffer = SignalBuffer(
    (len(freqs),),
    on_update=lambda data: viewer.update(freqs, signal2real(data)),
)
with Schedule(cfg, signals_buffer, scope=context.scope) as sched:
    freq_sweep = sched.cfg.freq
    sched.cfg.readout.set_param("freq", sweep2param("freq", freq_sweep))

    _ = (
        sched.prog_builder(
            soc, soccfg, reps=sched.cfg.reps, relax_delay=sched.cfg.relax_delay
        )
        .add_reset("reset", sched.cfg.reset)
        .add_readout("readout", sched.cfg.readout)
        .declare_sweep("freq", freq_sweep)
        .build_and_acquire(rounds=sched.cfg.rounds)
    )
```

- `Schedule` 在入口 deepcopy caller cfg。ProgramBuilder 不讀 cfg，作者明確傳入 program 控制參數。
  `sched.prog_builder(...)` 與 `step.prog_builder(...)` 都收 `reps=1`、`initial_delay=1.0`、
  `relax_delay=1.0`，delay 單位為 us。每次 build 隔離 modules 與 sweeps。
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
  result；per-child retry 由 caller 編排。
- `ScheduleStep.path` 會累積巢狀 host loop index，所以預設 single-buffer acquire 可
  依 owner path 自動寫入對應 slot。
- `with Schedule(cfg, result_buffer, scope=context.scope) as sched` 可編排任何實作 `BufferProtocol` 的 result
  tree；root result buffer 目前最多一個，多 root buffer 會 fast-fail。
  `step.child("field", cfg=program_cfg).buffer(shape)` 會建立 child-local default
  `SignalBuffer`，buffer 寫入時同步回 `result_buffer.data` 並觸發 update。
- `ScheduleStep.value` 是 scan/repeat/batch 的 control-flow coordinate；result tree 由
  `step.data` 讀取、`step.set_data(..., flush=True)` 寫入，需要 ndarray slot 時用
  `step.array_data` 做型別邊界。`flush=True` 對 ResultTree 表示立即送出 node event；
  對一般 `SignalBuffer` 不改變 `on_update(data)` public shape。
- `sched.trigger_update(flush=True)` 是 root-level update，會對 root buffer 傳
  `step=None`；step-level update 才帶 `ScheduleStep`。
- 正常離開 scope 時（包含 requested stop），`Schedule` 對 root `SignalBuffer` 呼叫 `flush_pending_update()`，交付最後尚未呈現的資料。已顯式 flush 或已經呈現的更新不重複發布。Body 拋出例外或 `KeyboardInterrupt` 時不做自動 flush；正常退出的 callback 錯誤會傳播，但 scope 仍關閉。
- 自動 flush 不擴充 `BufferProtocol`，也不對其它實作廣播 root update。`ResultTree` 維持由 caller 顯式 flush，以保留既有 env 與 path 事件語意。

### `ProgramBuilder`

`program_builder.py` 擁有完整的 QICK program 組裝與同步 acquire：`ProgramBuilder`、`ModuleFacade`、`AcquireOptions`、producer 型別契約與 raw conversion。`schedule.py` 保留 Schedule／Step、buffer／slot 與 run lifecycle。Builder 透過 Schedule 的公開合作方法選擇 slot、記錄 outcome 與處理 retry reset。Runtime package 入口仍匯出既有公開名稱。

`acquire_cancel.py` 合成 acquire child 的取消與作者的 data stop condition。Data stop 是正常完成，不設 scope event，也不標記 partial。

- `build()` 建立並回傳 program。
- `run_program(program)` 執行 caller 傳入的 integrated-acquire program。
- `build_and_acquire()` 是 `build()` 後立刻 `run_program(...)` 的 convenience。
- Decimated trace 使用獨立方法：`run_program_decimated(program)` 與
  `build_and_acquire_decimated()`。
- 四個 build/run acquire 方法都收具名 `rounds=1`，同時控制 producer 與 rounds progress。
  Retry 重用同一個 rounds 值。`reps` 與 `rounds` 必須為正整數，拒絕 bool。
  Builder 不投影或覆寫 cfg；`program_cls` factory 接 soccfg 與具名控制參數、modules、sweep、編譯選項。
- `set_raw2signal_fn(...)` 或 acquire 方法的 `raw2signal_fn=...` 可覆寫 raw
  conversion；integrated acquire 預設使用 `default_raw2signal_fn`，decimated acquire
  預設使用 `default_decimated_raw2signal_fn`。
- `program_cls=` 是測試 seam；常規實驗不傳，預設使用 `ModularProgramV2`。
- 四個 build/run acquire 方法共用 `options=AcquireOptions(...)`。省略時 retry=0、
  progress=False、progress_label="rounds"、progress_leave=None、stop_condition=None。
  `progress_label` 覆寫 rounds bar label；`progress_leave=None` 保留 root-vs-step 的
  close/retain policy。`progress` 控制 native acquisition progress，不控制 rounds bar。

---

## Stop 與 Retry

- `Scope` 擁有具名父子鏈與選擇器，`StopSignal` 擁有向下停止與完成事實。Caller 每次 run 建立根 Scope，
  透過 `Schedule(..., scope=context.scope)` 傳入。Schedule 在進入 context 時建立 schedule child，退出時 release。每次 acquire attempt 建立 acquire child，結束時 release。取消 child 不停止父 scope，後續 acquire 或 operation 可繼續。
- Child release 將第一個 error 與 partial 完成事實交到父 token，不設父 event。交付在 operation 層截止。GUI 的 run root 收到 Schedule error 後仍可用 `raise_if_error()` 轉成 failure。已完成 child 從 active tree 移除，不隨 run 長度累積。
- `Schedule` scope 接住一次 `KeyboardInterrupt`，包含 scan/repeat 的 caller loop body，將 schedule child 記為 `interrupted` 並保留 buffer，讓 with 後的 return 組出部分結果。其他例外不改變原有傳播。Scope 外中斷不受保護，Notebook 將正常返回的 interrupted 表示為 stopped；其他 caller 的 `raise_if_error()` 仍拋錯。
- `Schedule` 的 scan/repeat/batch 會在 host step 前檢查 stop；`ProgramBuilder` 會把
  composite `cancel_flag` 傳給 program acquire：它讀取 acquire child 的 external
  stop，但 `round_hook` 內的 data stop 只結束目前 program acquire，不改變 `Schedule.outcome` 或 partial 完成事實。
  `StopSignal` 對外只暴露 external cancel flag contract：`is_set()` / `set()`；retry reset
  仍使用 `clear_stop()` 清除 stop flag 與 transient error cause。
- `ProgramBuilder` 的 data-driven early stop 走 `AcquireOptions(stop_condition=...)`：internal
  round hook 先把 completed round 寫入 buffer，再執行 condition；命中時呼叫
  acquire-local `cancel_flag.set()`，保留該 completed round 並停止下一 round；`Schedule.outcome`
  仍是 `completed`。
- `Schedule.outcome.acquisitions` 按完成順序列出每次 program acquisition 的 `AcquisitionFacts`。
  `requested_rounds` 是傳給 producer 的 rounds，`completed_rounds` 是最後一次 attempt 的已完成 round hook 數，不累加失敗 retry 的資料。
  `attempts` 是實際 producer 呼叫次數，包含失敗；停止使 producer 未被呼叫時為零。
  `early_stopped` 只在 data condition 讓最後一次 attempt 在設定 rounds 前正常結束時為 True，external stop、failure 與最後一輪才命中都不是 early stop。
  `stop_value` 保留條件命中時的有限 scalar，單位與意義由 caller 定義。沒有提前停止、條件沒提供數值或數值非有限時為 None。
  布林 condition 仍可使用。需要保留數值時，condition 回 `StopDecision(met=True, value=...)`，型別由 `experiment.acquisition_facts` 匯入。
  SNR checker 回傳這個具名 decision，scalar trace 記 SNR，mux checker 記所有 active tones 的最小 SNR。
- Acquisition facts 隨 Schedule child release 交到父 token，在 operation 邊界截止。
  GUI、Notebook 與 workflow saver 將同一組 facts 放進 native `docs/metadata.acquisitions`。
  Workflow 的 `Completed.facts.acquisitions`、journal 與 point record 的 `operations[].acquisitions` 也保留它們。
  舊 native metadata 與 operation 記錄沒有此欄位時讀成空 tuple，不從 cfg 推測 rounds。
- 這些 facts 記 program acquisition，不聚合 host loop。Shaped length Rabi 每個 length × round 的 acquisition 是一筆 1/1，完整 host-round 的 SNR early stop 不會標成 program early stop。
  該路徑可以用 acquisition 筆數與每輪 length 點數計算完整 host rounds；中途停止時另有不足一輪的 acquisitions。
- `Schedule.outcome` 記錄 run 結果（資料保留不等於 operation 成功）：`completed`、`stopped`、`interrupted` 或
  `failed`。stop、`KeyboardInterrupt` 或 acquire 失敗會設定 `StopSignal`。
  Program build、raw conversion 與 update callback 的一般例外直接向 caller 傳播，不標記為 acquire 失敗。
  目前已寫入 buffer / result tree 的 partial result 保留；已寫入的 slot 保留，
  未完成的 slot 維持 NaN 初始化值。若 first round 尚未完成就 stop，program acquire
  以 stopped partial 例外結束，runner 將 outcome 標記為 `stopped` 並保留 NaN partial。
- `StopSignal` 也攜帶第一個 `failed` / `interrupted` cause。Schedule 結束時將 cause 交給 RunContext 的 token。Consumer（例如 measure-gui run policy）在 experiment adapter 回傳後呼叫 `raise_if_error()`，把 runner 內部錯誤轉成 operation failure。`stopped` 不記錄 error cause；retry reset 會同時
  清掉 stop flag 與 transient error cause。
- `ProgramBuilder.build_and_acquire(..., options=AcquireOptions(retry=N))` /
  `run_program(..., options=AcquireOptions(retry=N))` 是單次 program acquire retry；`Schedule.batch(...)` 不提供
  per-child retry。
- Program 只建構一次。Retry 只重跑同一 program 的 acquire，不重做 build 或失敗的 conversion callback。
  Build 前、每次 acquire attempt 前與 failed attempt retry 前都重新檢查 `StopSignal`；
  stop / interrupted outcome 不會被 retry，也不會被 retry reset 清掉。

---

## 測試

`tests/experiment/v2/runtime/test_program_builder.py` 覆蓋顯式 program 控制、Schedule cfg 隔離、
modules／sweep 隔離，以及兩種 acquire 的 rounds、retry 與 progress。
`test_program_acquire.py` 覆蓋 integrated／decimated acquire、program cache、progress 與 raw conversion；
`test_program_retry.py` 覆蓋 retry、stop checker、failed attempt 後 stop 不再 retry 與 `ScheduleOutcome`。
`test_schedule.py` 覆蓋 typed env 與 host scan/repeat/batch；`test_schedule_buffers.py` 覆蓋
root buffer 註冊、child buffer 同步與 shape validation。`test_signal_buffer_flush.py` 覆蓋
SignalBuffer 寫入通知、manual update 與 flush。
`test_schedule_lifecycle.py` 覆蓋 context 使用邊界、schedule child 的停止與錯誤交付、host body 中斷的
partial/NaN 保存、其他 exception 傳播與 repeat stop。Runtime 案例重用 `_flow_cfg.py` 的
cfg builder，共用 recording fake 放在 `_flow_fakes.py`，由最近的 conftest 還原每個案例的
mutable fake state。案例經公開 Scope／Schedule／buffer／StopSignal 接縫驗證。
`test_result_tree.py` 覆蓋 ResultTree node set、child buffer、subscription、flush 與
ordinary SignalBuffer regression。Runtime 整合行為由最小合成實驗驗證。
