# `zcu_tools.experiment.v2.runtime` — experiment runtime

**Last updated:** 2026-11-01，raw-shot acquisition 入口與共用 runner

**讀者：** framework-dev。Schedule、buffer、program acquire、停止與重試契約。

`runtime/` 提供 experiment/v2 的 Python-like acquisition runtime。一般實驗用
`SignalBuffer` 與 `Schedule` 編排 host-side loop 與 program acquire。`ResultTree` 提供 structured `BufferProtocol`，可交給同一個
`Schedule` 編排 outer loop。

---

## 核心物件

### `SignalBuffer`

`SignalBuffer` 是 result storage、live update 入口與 update 節流 owner。

- `SignalBuffer(shape, dtype=np.complex128, on_update=..., update_interval=0.1)` 以
  NaN 初始化底層 ndarray。
- `buffer[step]` / `buffer.at(...)` 回傳 `SignalSlot`，slot 寫入的是 buffer 的
  writable view。
- `SignalBuffer.set(...)` / `SignalSlot.set(...)` 只接受與目標完全相同的 shape，寫入後自動觸發 `on_update`。
  Shape 不符會在寫入前拋出 `ValueError`，訊息列出 target 與 value 的 shape，不發布 update。
  不做 NumPy broadcasting，純量或長度 1 陣列不能填滿多格 slot。純量 slot 的 shape 是 `()`，傳入 `np.array(value)` 即可寫入。
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
- child-local buffer：caller 以 `step.child("raw_signals").buffer(...)`
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

`Schedule[T_Env]` 是 run 內的 scope，持有 typed env 與顯式 `Scope` child，並建立 host-side control flow。
Schedule 與 step 都不接收或複製 cfg。作者直接讀取傳入 run 的 config。
若需修改設定，先自行 deepcopy，避免修改 caller 的 config。
以 `Schedule(buffer, scope=...)` 接收預建的 root buffer，或在
`with Schedule(scope=...) as sched` 內呼叫 `sched.buffer(shape, ...)` 建立 root `SignalBuffer`。
後者沿用 `SignalBuffer` 的 dtype、on_update 與 update_interval 參數；已有 root buffer 時 raise ValueError。

```python
from zcu_tools.experiment.v2.runtime import RunOptions, Schedule, SignalBuffer
from zcu_tools.program.v2 import Readout, TimelineBuilder

signals = SignalBuffer((len(freqs),), on_update=on_update)
builder = TimelineBuilder(
    context.soccfg, reps=config.reps, relax_delay=config.relax_delay,
)
freq = builder.add_sweep("freq", config.freq)
builder.emit(Readout("readout", config.readout, freq=freq))
prog = builder.build()

with Schedule(signals, scope=context.scope) as sched:
    sched.run_acquire(
        prog, context.soc, rounds=config.rounds,
        options=RunOptions(retry=config.acquire_retry),
    )
```

- `Schedule` 的 env 也是泛型，使用 `with Schedule(buffer, env=RunEnv(...), scope=context.scope)`
  可讓 `sched.env.xxx` / `step.env.xxx` 取得 linter 型別支援。env 只放穩定 run
  dependencies（例如 soc、soccfg、ModuleLibrary、predictor）；scan/repeat 的動態
  value/index 不寫入 env，而是由 `ScheduleStep.value` / `ScheduleStep.index` /
  `ScheduleStep.path` 表示。
- `sched.scan(...)` / `step.scan(...)` 表示 host-side Python loop。
- `sched.repeat(name, times, interval)` 表示 host-side repeat；等待期間會檢查
  `StopSignal`，需要刷新 liveplot 時由 caller 呼叫相關 buffer 的 `trigger_update()`。
- `ScheduleStep.path` 會累積巢狀 host loop index，所以預設 single-buffer acquire 可
  依 owner path 自動寫入對應 slot。
- `with Schedule(result_buffer, scope=context.scope) as sched` 可編排任何實作 `BufferProtocol` 的 result
  tree；root result buffer 目前最多一個，多 root buffer 會 fast-fail。
  `step.child("field").buffer(shape)` 會建立 child-local default
  `SignalBuffer`，buffer 寫入時同步回 `result_buffer.data` 並觸發 update。
- `ScheduleStep.value` 是 scan/repeat 的 control-flow coordinate；result tree 由
  `step.data` 讀取、`step.set_data(..., flush=True)` 寫入，需要 ndarray slot 時用
  `step.array_data` 做型別邊界。`flush=True` 對 ResultTree 表示立即送出 node event；
  對一般 `SignalBuffer` 不改變 `on_update(data)` public shape。
  `set_data` 與 `ResultNode.set` 寫入 ndarray target 時同樣要求 shape 完全相同。數字只可寫入 shape 為 `()` 的 target；mapping 與 list 的既有更新語意不變。
- `sched.trigger_update(flush=True)` 是 root-level update，會對 root buffer 傳
  `step=None`；step-level update 才帶 `ScheduleStep`。
- 正常離開 scope 時（包含 requested stop），`Schedule` 對 root `SignalBuffer` 呼叫 `flush_pending_update()`，交付最後尚未呈現的資料。已顯式 flush 或已經呈現的更新不重複發布。Body 拋出例外或 `KeyboardInterrupt` 時不做自動 flush；正常退出的 callback 錯誤會傳播，但 scope 仍關閉。
- 自動 flush 不擴充 `BufferProtocol`，也不對其它實作廣播 root update。`ResultTree` 維持由 caller 顯式 flush，以保留既有 env 與 path 事件語意。

### Schedule run acquisition

`Schedule` 與 scan step 提供 `run_acquire(program, soc, ...)`、
`run_acquire_summed(program, soc, processor=..., ...)` 與
`run_acquire_decimated(program, soc, ...)`，把轉換後的資料寫進 slot。
`run_acquire_shots(program, soc, ...)` 則回傳 `program.get_raw()` 的逐 shot int64 buffers，
不做轉換、不寫 slot，也不需要預建 buffer。四個方法都不建立 program。
Caller 可傳入手寫或 TimelineBuilder 產生的 program，只需提供對應的同步 acquire 方法。
Raw-shot 入口還要求 `get_raw()`；成功 acquire 後沒有 raw buffer 時拋出 `RuntimeError`。

`run_acquire_shots` 每次只執行一輪，`rounds` 省略或設為整數 1，其他值在入口拋出
`ValueError`。多輪 raw shots 由 caller 的 host loop 重複呼叫。
它接受 `RunOptions(retry=..., progress=..., on_stop=...)`，但拒絕非 None 的
`converter` 與 `stop_condition`，在 acquire 前拋出 `ValueError`。
它沿用同一個 runner 的 acquire child scope、cancel flag、retry 與 progress 清理。
外部取消時回傳 `None`，不讀取 raw buffer；Schedule 自己的 stop token 已設時記錄
`stopped` outcome。Acquire-only cancel 不停止 Schedule，後續 acquire 仍可執行。

三個 slot 入口會把轉換後的結果寫入目前 Schedule／step 的 slot，並觸發 live update。
它依 `RunOptions.retry` 自動重試，耗盡後向 caller 拋出原 producer 例外，不回傳 failed partial。
`RunOptions.stop_condition` 只停止目前 acquire。
`RunOptions[T_Raw]` 提供 retry、progress、stop_condition 與 converter。`rounds` 等 native 參數直接傳給 run 方法。
program 層無前綴的 `acquire*` 只取數並回傳，不承擔 slot/live update、自動 retry 或
Schedule stop_condition。
Integrated 的預設 converter 只接受單一 ADC、每 shot 單一 read，並將 IQ 轉成複數。
多 ADC 或多 read 時拋出 `ValueError`，訊息列出 ADC 數與各 ADC 的 read 數，要求提供
`RunOptions(converter=...)`。Summed 直接寫入累積總和。Decimated 的預設 converter
只接受單一 ADC，將 IQ 轉成複數並保留全部 repetition／read／sample 軸；多 ADC 時
拋出 `ValueError`，訊息列出 ADC 數並要求提供 converter。

`run_acquire.py` 的 mixin 集中 Schedule 與 Step 的四個入口，不持有 scope、slot
或 outcome。`acquisition.py` 的 `AcquisitionRunner` 集中 round hook、retry、stop、
partial-result、overrun 與 progress。
`AcquireOptions`、同步 producer keyword 型別與 raw converters 也由該模組提供。

`acquire_cancel.py` 合成 acquire child 的取消與作者的 data stop condition。
Data stop 是正常完成，不設 scope event，也不標記 partial。

## Stop 與 Retry

- `Scope` 擁有具名父子鏈與選擇器，`StopSignal` 只擁有向下停止請求與不解析的 reason。Caller 每次 run 建立根 Scope，
  透過 `Schedule(..., scope=context.scope)` 傳入。Schedule 在進入 context 時建立 schedule child，退出時 release。每次 acquire attempt 建立 acquire child，結束時 release。取消 child 不停止父 scope，後續 acquire 或 operation 可繼續。
- Child release 只解除父子關係，不上交停止、資料完整度或錯誤。已完成 child 從 active tree 移除，不隨 run 長度累積。
- `Schedule.__exit__` 與 AcquisitionRunner 都不接住 `KeyboardInterrupt`。Scan/repeat body、producer 與 converter 的中斷原樣傳出，不 retry，也不轉成 interrupted outcome。已寫入的 buffer 資料保留，但 caller 必須自己決定是否攔截例外並回傳。Notebook 的 SIGINT 合作停止與時限由 [Notebook adapter](../../../notebook/README.md#共用實驗入口) 擁有。
- `Schedule` 的 scan/repeat 會在 host step 前檢查 stop；`AcquisitionRunner` 會把
  composite `cancel_flag` 傳給 program acquire：它讀取 acquire child 的 external
  stop，但 `round_hook` 內的 data stop 只結束目前 program acquire，不改變 `Schedule.outcome`。
  `StopSignal` 對外提供 external cancel flag contract：`is_set()` / `set()`；
  `clear_stop()` 只清本 token 的 event，保留 reason，不清子孫。
- `AcquisitionRunner` 的 data-driven early stop 走 `RunOptions(stop_condition=...)`：internal
  round hook 先把 completed round 寫入 buffer，再執行 condition；命中時呼叫
  acquire-local `cancel_flag.set()`，保留該 completed round 並停止下一 round；`Schedule.outcome`
  仍是 `completed`。
- Runner 每次 retry 寫一行 warning，包含原因、下一次 attempt number 與原始錯誤。
  Shot 超量的原因是 `shot_overrun`，其他原因是 exception class name。
- Data condition 讓 acquire 在設定 rounds 前正常結束時，Runner 寫一行 info，包含 stop value 與 completed rounds。
  條件可回 bool 或 `experiment.acquisition_facts.StopDecision`。有限 scalar 才記入日誌，未提供或非有限時記 None。
  External stop、failure 與最後一輪才命中都不記 early-stop info。
- `Schedule.outcome` 記錄作者工具的 run 結果（資料保留不等於 operation 成功）：`completed`、`stopped` 或
  `failed`。Stop 或 acquire 失敗會設定 `StopSignal`，KeyboardInterrupt 只向外傳播。
  Program build、raw conversion 與 update callback 的一般例外直接向 caller 傳播，不標記為 acquire 失敗。
  目前已寫入 buffer / result tree 的 partial result 保留；已寫入的 slot 保留，
  未完成的 slot 維持 NaN 初始化值。若 first round 尚未完成就 stop，program acquire
  以 stopped partial 例外結束，runner 保留 NaN。只有 Schedule 自己的 token 已停止時，outcome 才記 `stopped`。
- `StopSignal.set(reason=None)` 保存並向下轉交不解析的停止文字，`reason` 屬性供作者讀取。`Scope.cancel(selector, reason=None)` 將文字傳給選中的 subtree。
- 一般 producer failure 直接傳播，不轉成 workflow 的 `Failed` 回傳值。錯誤原因只保留在作者工具的 Schedule outcome，不寫入 StopSignal。
  Notebook 不發布失敗當次的 record。GUI 將例外交給 operation failure 收尾。
- `AcquireOptions.on_stop` 與 `RunOptions.on_stop` 是選用的無參數作者 callback。單次 acquire 因外部停止而合作返回時呼叫一次，包含起跑前已停止與首輪前取消。
  Callback 在 acquire scope release、progress cleanup 後同步執行，早於 slot view 或 raw-shot 停止結果返回。純 data stop、producer／converter／update 例外與 KeyboardInterrupt 不呼叫。
  Callback 自己的例外原樣傳出，不 retry、不改 Schedule outcome。停止觀察只存於該次呼叫，不污染重用 options 的下一次 acquire，也不交給前端。
  Acquire-only cancel 不改 Schedule outcome 或父 stop；作者可用 callback 記自己的分類，不能用父 stop 判斷 acquire 是否被取消。
- `sched.run_acquire(program, soc, options=RunOptions(retry=N))` 設定單次 program acquire retry。
- 所有 producer failure（包含 shot 超量 `AcquisitionContractError`）的額外重試次數都是 `options.retry`。
  預設 `retry=0` 不重試。需要一次重試時明確傳入 `RunOptions(retry=1)`。
  耗盡後拋出最後一次 attempt 的同一例外。若失敗時 Schedule 本身也已停止，不再重試，但仍拋出 producer 例外，不當作正常 partial。
  每次從第一 round 重跑，producer 重建 raw buffers。超量 attempt 已發布的 slot 資料回復為本次 acquire 開始前的值，耗盡重試時也不保留該 attempt 的資料。
  Warning 保留 expected／received／chunk shots 與 round index。
- Program 只建構一次。Retry 只重跑同一 program 的 acquire，不重做 build 或失敗的 conversion callback。
  每次 acquire attempt 前與 failed attempt retry 前都重新檢查 `StopSignal`；
  合作停止的 `stopped` outcome 不會被 retry，也不會被 retry reset 清掉。`KeyboardInterrupt` 原樣向外傳播，不 retry，也不轉成 outcome。

---

## 測試

`test_run_acquire.py` 經 Schedule 公開入口覆蓋三種 acquisition 的 simulator 路徑、
slot/live update、converter、retry、data stop、取消、部分 round 與 callback 錯誤。
`test_run_acquire_shots.py` 經 Schedule／ScheduleStep 覆蓋 raw buffer 回傳、無 slot 取數、
native keyword 轉送、取消與 child release、stop observer 清理順序、retry、原始例外傳播，
以及拒絕 slot policy、多輪與缺少 raw buffer 的失敗契約。
`test_run_timeline.py` 以 deterministic transport 跑 emit program 的一般 QICK
acquisition pipeline。
`test_schedule.py` 覆蓋 typed env 與 host scan/repeat；`test_schedule_buffers.py` 覆蓋
root buffer 註冊、child buffer 同步與 shape validation。`test_signal_buffer_flush.py` 覆蓋
SignalBuffer 寫入通知、manual update 與 flush。
`test_schedule_lifecycle.py` 覆蓋 context 使用邊界、schedule child 的停止與錯誤交付、host body 中斷的
partial/NaN 保存、其他 exception 傳播與 repeat stop。Runtime 案例重用 `_flow_cfg.py` 的
cfg builder，共用 recording fake 放在 `_flow_fakes.py`，由最近的 conftest 還原每個案例的
mutable fake state。案例經公開 Scope／Schedule／buffer／StopSignal 接縫驗證。
`test_result_tree.py` 覆蓋 ResultTree node set、child buffer、subscription、flush 與
ordinary SignalBuffer regression。Runtime 整合行為由最小合成實驗驗證。
