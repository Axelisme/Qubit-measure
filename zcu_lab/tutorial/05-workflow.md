# 撰寫逐點 workflow

**讀者：** lab-author、agent。用顯式 Scope 串接量測，保存每點已收集的成果，並處理作者選定的例外與取消。

Workflow 在一個 async 函式內保留 history 與區域變數。`env` 是本次 workflow 的根 Scope。
`run_core(env, ...)` 明確把量測掛在根下。逐點量測則把 `iteration` 傳給 helper，讓取消只影響目前這一點。
宣告與 GUI 表單入口見 [workflow README](../workflows/README.md)。

## 用 scope 執行 workflow 取數

Workflow 自己寫同步 `acquire(cfg, *, context)`，在裡面組 program 並呼叫 runner。
以下範例只接受 `sampling="linear"`，依序播放 reset、π pulse、delay 與 readout，回傳 core 的 `T1Result`。
它使用預設 retry 0，不呼叫 core 的 run 或另一個實驗的取數 helper。
需要 retry 時，依 [workflow README](../workflows/README.md#撰寫-async-workflow) 在同步函式加入 retry 參數，
從 workflow settings 以 `functools.partial` 綁定。不要在 `T1Cfg` 加 retry 欄位。

```python
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.run_core import run_core
from zcu_tools.experiment.scope import Scope
from zcu_tools.experiment.v2.runtime import RunOptions, Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.round_zcu import sweep2array
from zcu_tools.experiment.v2.utils.snr import snr_checker
from zcu_tools.experiment.workflows import Completed
from zcu_tools.program.v2 import Pulse, Readout, TimelineBuilder
from zcu_tools.utils.process import rotate2real
from zcu_lab.v2.twotone.time_domain.t1.core import T1Cfg, T1Result


def acquire(cfg: T1Cfg, *, context: RunContext) -> T1Result:
    if cfg.sampling != "linear":
        raise ValueError("This example only supports linear T1 sampling.")
    times = sweep2array(cfg.delay, "time", {"soccfg": context.soccfg})
    signals = SignalBuffer((len(times),))
    with Schedule(signals, scope=context.scope) as sched:
        builder = TimelineBuilder(
            context.soccfg, reps=cfg.reps, relax_delay=cfg.relax_delay
        )
        delay = builder.add_sweep("delay", cfg.delay)
        if cfg.reset is not None:
            builder.emit(cfg.reset.build("reset"))
        builder.emit(Pulse("pi_pulse", cfg.pi_pulse))
        builder.timeline_inc(delay)
        builder.delay_by_timeline()
        builder.emit(Readout("readout", cfg.readout))
        program = builder.build()
        sched.run_acquire(
            program, context.soc, rounds=cfg.rounds,
            options=RunOptions(
                stop_condition=snr_checker(
                    signals.at(), cfg.earlystop_snr, lambda iq: rotate2real(iq).real
                ),
            ),
        )
    return T1Result(times=times, signals=signals.array)


async def run_t1(
    scope: Scope, cfg: T1Cfg
) -> Completed[RunRecord[T1Cfg, T1Result]]:
    return await run_core(scope, acquire, cfg, name="T1")
```

Async helper 收 `Scope` 與 typed cfg，不收另一個 WorkflowEnv。
`name` 由作者指定，框架不從 callable 猜名字。
`run_core` 複製 cfg，並透過 `Scope.call` 建立 operation child。
它在呼叫 acquire 前擷取 connected device snapshot，寫入 `RunRecord.metadata["device_snapshot"]`。
Acquire 正常回傳就交付 `Completed(value, facts)`，值在 `t1_record.value`，當次事實在 `t1_record.facts`。
保存沿用 `T1Exp.AXES_SPEC`，因此 GUI 仍能以單一 T1 實驗開啟 native 檔。
本例不建立 operation 即時圖。要顯示即時曲線，在 acquire 內用 `context.plots` 建圖。
更新函式以 `context.plots.on_plot_thread` 裝飾，再傳給 `SignalBuffer(on_update=...)`。
`facts.stopped` 只記錄 operation 自身在結束時的 stop bit。只取消其子孫不會把它標成 stopped。
這個位元不判斷資料是否完整或可用來擬合。
Acquisition、cfg 與程式例外原樣傳出。作者在能恢復的地方明確 catch，不把 framework run 分成 Completed／Failed 兩條路。

需要 workflow 圖的 helper 另外收 `plots: Plots`。`scope.identity` 提供根的歷史身分，
`scope.get_device_cfg(name)` 讀完整裝置設定。Child 繼承這些能力，不另建 provenance。

## 每點先建 record，再逐步填入

Record 用可變的 Pydantic model。未取得的欄位預設為 `None`，不能用零冒充資料。
以下片段放在已有 plan、cfg 與 saver 的 workflow 內。`save_native` 是 lab 的 exact-path native writer。
作者必須明填 `completion`，writer 不會從控制狀態推斷資料完整度。
本例選擇在 `facts.stopped` 為真時標 `stopped`，否則標 `complete`。只取消 operation 子孫時仍標 `complete`。
若實驗另有完整度判準，作者應依資料選擇 `complete`、`partial`、`stopped` 或 `unknown`。

```python
from typing import Literal
from pydantic import BaseModel
from zcu_tools.experiment.run_core import run_core
from zcu_lab.v2.twotone.time_domain.t1 import T1Exp
from zcu_lab.workflows.common.saving import save_native


class PointRecord(BaseModel):
    index: int
    first_file: str | None = None
    repeat_file: str | None = None
    reason: str | None = None
    status: Literal["collecting", "completed", "failed", "skipped"] = "collecting"


# Inside async def workflow_body(env, plan, tun):
for cfg, iteration in zip(plan.points, env.iterations(), strict=False):
    point_record = PointRecord(index=iteration.sequence - 1)
    try:
        try:
            t1_record = await run_t1(iteration, cfg)
        except OSError as exc:
            point_record.status = "failed"
            point_record.reason = str(exc)
            continue
        save_native(
            t1_record,
            iteration.dir / "t1.h5",
            completion="stopped" if t1_record.facts.stopped else "complete",
            spec=T1Exp.AXES_SPEC,
            snapshot=plan.snapshot,
            identity=iteration.identity,
        )
        point_record.first_file = "t1.h5"
        if iteration.cancelled:
            point_record.status = "skipped"
            continue

        try:
            repeat_record = await run_core(iteration, acquire, cfg, name="T1-repeat")
        except OSError as exc:
            point_record.status = "failed"
            point_record.reason = str(exc)
            continue
        save_native(
            repeat_record,
            iteration.dir / "t1-repeat.h5",
            completion="stopped" if repeat_record.facts.stopped else "complete",
            spec=T1Exp.AXES_SPEC,
            snapshot=plan.snapshot,
            identity=iteration.identity,
        )
        point_record.repeat_file = "t1-repeat.h5"
        point_record.status = "skipped" if iteration.cancelled else "completed"
    finally:
        iteration.commit(point_record)
```

範例只恢復作者選定的 acquisition `OSError`。保存不在 catch 內，writer 或其他程式錯誤繼續傳出。
實際裝置若使用其他例外型別，作者應在這兩個 await 周圍捕捉明確可恢復的型別。

`commit` 同步寫下當時的快照，不加 `await`。
後續修改 `point_record` 不會改已提交資料。同一 scope 第二次 commit 會 raise。
`finally` 只提交既有成果。它不重設 status，也不吞掉 `ScriptCancelled` 或真實錯誤。
根取消時，下一個 async operation 或 Ask 會丟普通 `ScriptCancelled`，作者的 finally 仍會執行。

`iteration.cancelled` 為真時，主迴圈用 `continue` 跳過目前點，不用 `return` 結束整個 workflow。
若量測在 helper 內，helper 才用 `return` 退出目前點。下一個 iteration 使用新的取消狀態。

## 目錄跟著 scope 走

`iteration.dir` 第一次使用才建立 `data_root/iteration-000001/`。
`with iteration.child("calibration") as scope:` 建立作者 child，其目錄是
`iteration-000001/calibration-000001/`。只有使用 `scope.dir` 或 commit 才建該目錄。
具名 operation 的目錄用 `operation-000001`，名稱仍留在路徑與 record 信封。
根 `env.commit(record)` 寫 data root 的 `record.json`，也只能提交一次。

`env.iterations()` 在 next 時釋放前一點，close 時釋放最後一點。
直接用 `zip(points, env.iterations())`，離開迴圈時 generator 會關閉。
若把 generator 存在變數中，就由作者在 finally 呼叫 `close()`。
Reader 走這棵目錄樹，依 commit 序號排序，不依目錄名排序。

## 作者決定如何恢復失敗

取數 helper 的例外不會自動變成 `Failed`。點與點獨立且例外可恢復時，把原因填進 point record，再 `continue`。
上面的 finally 會提交 failed 點。不能恢復的錯誤直接傳出，finally 仍會保存先前已填的欄位。
後續工作全部依賴這個結果時，填入原因並 `return Summary(reason=...)`，讓 workflow 結束。
這種 workflow 的 return annotation 要包含自己的 Summary model。

可重試時，由作者用有限的 `for attempt in range(...)` 包住相同的 await，並決定何時停止重試。
框架每個 operation 只呼叫一次，不替作者重做外部效果。

作者的分析 helper 仍可回 `Failed(reason)`，例如 fitter 明示不收斂。這是作者的分析結果，不是 `run_core` 的回傳分支。
Caller 對這種值選擇繼續、結束或自行重試。Writer 或程式例外仍應傳出，不把它們混成分析拒絕。
