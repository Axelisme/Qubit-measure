# 撰寫逐點 workflow

**讀者：** lab-author、agent。用顯式 Scope 串接量測，保存每點已收集的成果，並處理 Failed 與取消。

Workflow 在一個 async 函式內保留 history 與區域變數。`env` 是本次 workflow 的根 Scope。
`env.run(...)` 明確把量測掛在根下。逐點量測則把 `iteration` 傳給 helper，讓取消只影響目前這一點。
宣告與 GUI 表單入口見 [workflow README](../workflows/README.md)。

## 用 scope 執行 core

共用 helper 收 `Scope` 與既有的 typed cfg，不收另一個 WorkflowEnv。
`name` 由作者指定，框架不從 callable 猜名字。

```python
from zcu_tools.experiment.scope import Scope
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.workflows import Completed, Failed
from zcu_lab.v2.twotone.time_domain.t1.core import T1Cfg, T1Exp, T1Result


async def run_t1(
    scope: Scope, cfg: T1Cfg
) -> Completed[RunRecord[T1Cfg, T1Result]] | Failed:
    return await scope.run(T1Exp().run, cfg, name="T1")
```

`scope.run` 複製 cfg、擷取 connected device snapshot，並建立 operation child。
成功取得資料時回 `Completed`，值在 `t1_record.value`，當次事實在 `t1_record.facts`。
`partial_result=True` 表示 operation 或其下的工作曾被取消，不代表資料可用來擬合。
只有 acquisition 的可恢復失敗回 `Failed`。Cfg、writer 與程式錯誤仍會 raise。

需要 workflow 圖的 helper 另外收 `plots: Plots`。`scope.identity` 提供根的歷史身分，
`scope.get_device_cfg(name)` 讀完整裝置設定。Child 繼承這些能力，不另建 provenance。

## 每點先建 record，再逐步填入

Record 用可變的 Pydantic model。未取得的欄位預設為 `None`，不能用零冒充資料。
以下片段放在已有 plan、cfg 與 saver 的 workflow 內。`save_native` 是 lab 的 exact-path native writer。

```python
from typing import Literal
from pydantic import BaseModel
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
        t1_record = await run_t1(iteration, cfg)
        if isinstance(t1_record, Failed):
            point_record.status = "failed"
            point_record.reason = t1_record.reason
            continue
        save_native(
            t1_record,
            iteration.dir / "t1.h5",
            spec=T1Exp.AXES_SPEC,
            snapshot=plan.snapshot,
            identity=iteration.identity,
        )
        point_record.first_file = "t1.h5"
        if iteration.cancelled:
            point_record.status = "skipped"
            continue

        repeat_record = await iteration.run(T1Exp().run, cfg, name="T1-repeat")
        if isinstance(repeat_record, Failed):
            point_record.status = "failed"
            point_record.reason = repeat_record.reason
            continue
        save_native(
            repeat_record,
            iteration.dir / "t1-repeat.h5",
            spec=T1Exp.AXES_SPEC,
            snapshot=plan.snapshot,
            identity=iteration.identity,
        )
        point_record.repeat_file = "t1-repeat.h5"
        point_record.status = "skipped" if iteration.cancelled else "completed"
    finally:
        iteration.commit(point_record)
```

`commit` 同步寫下當時的快照，不加 `await`，也不等待 Pause 的 Resume。
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

## 拿到 Failed 之後

點與點獨立時，把原因填進 point record，再 `continue`。上面的 finally 會提交 failed 點。
後續工作全部依賴這個結果時，填入原因並 `return Summary(reason=...)`，讓 workflow 結束。
這種 workflow 的 return annotation 要包含自己的 Summary model。

可重試時，由作者用有限的 `for attempt in range(...)` 包住相同的 await，並決定何時停止重試。
框架不替作者重做外部效果。Pause 的 interrupted-attempt retry 是另一個機制，它只重跑原 capture。

不要對 `Failed` 再 raise。它是結果值，讓作者選擇繼續、結束或自行重試。
不要把真實的 writer 或程式例外轉成 `Failed`，否則讀者無法分辨未取得資料與保存出錯。
