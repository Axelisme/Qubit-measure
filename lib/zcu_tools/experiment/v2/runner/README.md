# `zcu_tools.experiment.v2.runner` — task runner

**Last updated:** 2026-07-01

這份筆記整理 `runner/` 的任務執行框架設計，說明各類別的職責、組合方式與執行流程。

---

## 架構總覽（一句話版）

`runner/` 以 **Composite 模式** 把測量任務組成樹狀結構：`AbsTask` 是抽象節點，`Task` 是葉節點（實際呼叫硬體），`BatchTask`/`RetryBatchTask`/`Scan`/`RepeatOverTime`/`ReTryIfFail` 是中間節點（組合或修飾子任務）。結果也以同構樹（`Result`）儲存，`TaskState` 攜帶當前在樹中的位置。`MultiMeasurementExecutor`（`multi_executor.py`）是 runner 層的 Executor 共用 scaffold（非樹節點）。

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

### `MultiMeasurementExecutor`（`multi_executor.py`）

不是 task tree 節點，而是 **Executor 共用 scaffold**：把「同時跑多個量測 + 合併 live plot（可選錄 FFmpeg 動畫）」的共用機制收在一處，供 `autofluxdep` / `overnight` 的 Executor 繼承（這兩個子類別與各自的 `MeasurementTask` ABC 住在 app 模組，不在 runner）。

- 提供：`add_measurements`、`record_animation(path)`（FFMpeg facet，缺 `ffmpeg` 即 Fast-Fail）、`make_ax_layout` / `make_plotter`（依各量測 `num_axes()` 自動排版 subplot），以及 `_run_with_plotting(task, cfg, env_dict)`（在 plotter context 內呼叫 `run_task`，並在 `finally` 收尾動畫）。
- 子類別只負責各自的 `run()`（不同的外層 driver 與 cfg/env 前置），共用此處的版面 / plotter / 錄製機器。
- 對 task 的需求以結構性 `PlottableMeasurement` Protocol 表達（`num_axes` / `make_plotter` / `update_plotter`），讓基底不必把兩個 app 的 `MeasurementTask` ABC 強行合併。

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

**1. `TaskHandle` / `ActiveTask`（runner 層）**

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

`TaskHandle` 是 `threading.Event` 的薄包裝。`stop_flag` 注入 `state._stop_flag`，透過 `child()` 傳遞到整棵任務樹。

**2. `ActiveTask` — 全域 stop_flag context manager**

實驗呼叫端如果無法直接傳 `stop_flag` 給 `run_task`（例如 GUI 把 `run_task` 包在另一個 helper 內），可使用 `ActiveTask`。stop_event 必須由外部建立並傳入，`ActiveTask` 不自動建立：

```python
import threading
from zcu_tools.experiment.v2.runner import ActiveTask

stop_event = threading.Event()
with ActiveTask(stop_event) as handle:
    result = run_task(task, cfg)   # 自動拿到 stop_event

# 另一個 thread（如 GUI 按下停止鍵）：
stop_event.set()   # 或 handle.cancel()
```

`run_task` 在 `stop_flag=None` 時自動讀取模組級 `_current_stop_flag`（即 `ActiveTask` 設置的 Event）。`ActiveTask` 不允許巢狀使用，第二次進入會拋出 `RuntimeError`。

---

**3. 容器節點短路**

`BatchTask`、`Scan`、`RepeatOverTime` 在每個子任務開始前各自呼叫 `state.is_stop()`，若已設置則立即 `break`（不繼續剩餘子任務）。`RepeatOverTime` 的等待迴圈也加入 `is_stop()` 檢查，確保 interval 等待期間也能及時退出。

中斷後，`run_task` 回傳 `state.root_data`（已完成的子節點有真實資料，未完成的保持 NaN）。

---

**4. `EarlyStopMixin.finish_round()`（program 層）**

`measure_fn` 在建構 program 後、呼叫 `acquire()` 前，將 `ctx.is_stop` 加入 `stop_checkers`：

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
