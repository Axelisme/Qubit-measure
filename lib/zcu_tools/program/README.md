# `zcu_tools.program` — QICK integration

**Last updated:** 2026-11-01，s14 runtime event time

**讀者：** framework-dev。QICK 整合、程式建構與 acquisition 依賴速查。

這份筆記整理 `lib/zcu_tools/program` 對 QICK 的實際依賴，目的是讓後續開發能快速定位「應該看哪個 QICK 類別/方法」，而不用每次從頭追。

## Program 層定位

`acquisition.py` 擴充 QICK 的 `AcquireMixin`，負責 acquisition 的型別封裝、追蹤、提前停止與 round hook。`v2/base.py` 定義 v2 program 基底 `MyProgramV2`；它使用這些 acquisition mixin，但負責的是 v2 program 結構，兩者不是同一種 base。`v2/timeline.py` 的 `TimelineProgramV2` 繼承 `MyProgramV2`，提供原生作者 hooks，提供可選的具名 timeline，所有指令共用既有 IR 管線與 acquire 擴充。`ProgramResourcesMixin` 提供 dmem 配置及 readout route 保護。

`v2.builder.TimelineBuilder` 記錄宣告、線性動作與巢狀 repeat／branch。`build()` 產生 `TimelineProgramV2`，不取數或依賴 experiment runtime。`SimulationSoc` acquisition 透過 compiled frontend 執行 TimelineProgramV2 的受支援 instruction effects。

## Native acquisition

`acquire(soc, rounds=1, ...)` returns physical IQ in ADC declaration order,
with final axis `[I,Q]`. It retains QICK's `remove_offset` policy and native
`threshold`/`angle` projection. Projection returns excited fraction in I and
zero in Q, using QICK's angle sign, offset and equality rules.
Callers supply native projection options explicitly in ADC declaration order.

Host classification uses `acquire_summed` with read axes, not an acquire option.
`centroid3_classifier(cfg, ro_ch=...)` returns detached g/e/f calibration
from native Pulse or Mux cfg by ADC number. Missing calibration, wrong type and
unknown ADC raise ValueError. It never reads the live library.

## Summed acquisition

`acquire_summed(soc, processor=..., rounds=1, step_rounds=False, **qick_options)`
passes every declared ADC to one `acquisition.ShotSumProcessor`. Each float64
array has shape `(points, reps, reads, 2)`: integer ADC integrals divided by the
rounded integration length, without IQ-offset subtraction. Non-reps loops are
flattened in C order. The processor returns float64 `(points, *axes)` per-point
sums. The framework snapshots each completed round and adds its sums without
averaging reps or rounds. Final data and round hooks have shape
`(*point_dims, *axes)`. Invalid dtype, point count, zero-length output axes or
changing output shape raises ValueError with expected and actual values.

Hardware and simulation share this processing and accumulation path. Cancellation
discards incomplete rounds and preserves completed sums; no completed round raises
`StoppedPartialAcquireError`. Hardware stepped acquisition returns None at
preparation, then `finish_acquire()` returns the summed ndarray. Simulation keeps
its existing stepped-acquisition rejection. Explicit `threshold`, `angle` and
`herald` options are rejected. Native `acquire` retains its IQ contract.

`v2.shot_sums(prog, axes)` resolves reads after compilation. Each `v2.Classify`
axis contributes a one-hot vector, while `v2.IQ` contributes `(1, I, Q)`. The
processor sums their ordered outer products over reps. Repeat a read axis to
obtain second moments. Classify uses a detached explicit classifier or the
readout cfg's classifier. `Classify` with `tone=None` selects a pulse readout or
its mux readout's only tone. Multi-tone mux requires a zero-based cfg tone position
and lists available positions on error. `IQ` still requires an explicit mux tone
position. Pulse axes require `tone=None`. Circle requires GE Post radius and emits
g/e/other; linear emits g/e; centroid3 emits g/e/f. Classification boundaries match
`v2.modules.readout.classification_processor`.

Timeline read labels use emit tags, falling back to module names for untagged
emissions. Missing or ambiguous labels
raise ValueError listing available labels. Explicit tags retain QICK's global
compile-time uniqueness check and RuntimeError, including collisions with native
timed macros. Any Repeat/Branch trigger on a selected ADC or a declared/actual
read-count mismatch is rejected. Other ADCs may still use control flow.
`v2.split_sums(sums)` splits a final IQ axis into counts and complex conditional
means. Zero counts stay zero, and unmeasured NaN counts stay NaN.
Both give NaN means without divide warnings.

Build first, then resolve labels and call
`sched.run_acquire_summed(program, soc, processor=..., rounds=...,
options=RunOptions(converter=...))`.
The optional converter receives both cumulative and final sums.
It owns any normalization and signal-slot shape. Without a converter, sums are written unchanged.
Retry, progress and stop conditions follow the other Schedule run methods.

## Shot time

`shot_time_s(rounds=..., reps=..., shot_lengths_us=..., relax_delay_us=...)`
從各 sweep 點的 shot 長度與共同 relax 計算完整 acquisition 的 shot 秒數。
輸入時間單位為 µs，rounds 與 reps 為正整數；不計 acquisition 固定開銷與 flux ramp。
`request_round_time_s(request)` 從整數 request 計算一輪 pacing 秒數。它以 Python 整數加總各點與 relax，再乘 reps，由 TimeBase 一次換成秒。整輪總和不套用單一 request 時間的 ns 精確界限。模擬 binding 只呼叫此入口，不累計 acquisition 的硬體總時間。既有 µs API 與硬體路徑不變。

## Conditional readout

Declare separate read labels for herald and terminal emissions. One readout
module may be emitted twice with distinct emit tags. Every herald class is kept:

```python
from zcu_tools.program.v2 import Classify, IQ, shot_sums, split_sums

processor = shot_sums(prog, [Classify("herald"), IQ("terminal")])
sums = prog.acquire_summed(soc, processor=processor, rounds=2, progress=False)
counts, means = split_sums(sums)
ground_counts = counts[..., 0]
conditional_iq = means[..., 0]
```

The last axis stores count, sum I and sum Q for each herald class. Classifier
and IQ axes remain in the requested order. Ground conditional means divide
the accumulated IQ sums by ground counts, so rounds with more ground shots
carry more weight. Zero accepted shots produce NaN means. No shot is removed
from the stored class totals. Raw integer transport sums remain available via
`get_raw()`, including every read.

Decimated acquisition keeps native traces and software-round averaging.
Nonempty `herald` raises ValueError because post-selection needs integrated
shots.

## 主要依賴面

- `ImproveAcquireMixin` 直接擴充 `qick.qick_asm.AcquireMixin`，是 `program` 層最核心的 QICK 接點。
- Accumulated acquire 在寫入 chunk 前拒絕超過預期的累計 shots，拋出 `program.acquisition.AcquisitionContractError`。訊息包含預期 shots、先前已收 shots、本次 shots、各 channel 的 IQ 列數與從 0 起算的 round index。錯誤 chunk 不寫入、不裁切，先前 raw chunks 保留，不發布未完成的 round。
- 主要加值分成四類：
  - 型別友善封裝（`TypedAcquireMixin`）：補強 `acquire()` / `acquire_decimated()` / `get_raw()` 等回傳型別。
  - 統計追蹤（`TrackerMixin`）：在 `finish_round()` 內注入 tracker 更新流程。
  - 提前停止（`EarlyStopMixin`）：透過單一 `cancel_flag` 在 `finish_round()` 內中止後續 rounds；accumulated acquisition的內部reps進度走共用`progress_bar.make_pbar` factory，GUI ambient factory與terminal fallback使用同一生命週期，正常或停止都會close並移除暫時進度項。
  - 每輪 callback（`RoundHookMixin`）與 joint-ADC shot sums（`ImproveAcquireMixin`）。`round_hook(round_count, raw, cancel_flag)` 只收到 completed round，可呼叫 `cancel_flag.set()` 表示不開始下一 round。

## `AcquireMixin` 行為對照（常見踩點）

- 生命週期：
  - `acquire()`/`acquire_decimated()` 先建立 `acquire_params`，再進入 `prepare_round()` + `finish_round()` 迴圈。
  - `finish_acquire()` 負責 rounds 聚合（accumulated 或 decimated）。
- 擴充掛點：
  - `TrackerMixin.finish_round()`：利用原生 `self.acc_buf`、`self.ro_chs`、`self.avg_level` 更新 tracker。
  - `RoundHookMixin.finish_round()`：每 completed round 呼叫 `round_hook`，輸入增量摘要資料與同一個 stop flag。
  - `ImproveAcquireMixin._process_accumulated()`：明傳單一 sum processor 時處理每點總和，其餘委派原生 IQ／threshold 路徑。
- 相容性關鍵：
  - 擴充層維持 `AcquireMixin` 的 `extra_args` 傳遞模式，避免破壞原生 acquire 參數流。
  - 若非 `accumulated` 或有 threshold 時，部分統計路徑會顯式 `NotImplementedError`，這是刻意防呆。

## soccfg 摘要顯示（`describe_soc`）

- `describe_soc(soccfg)`（`soc_summary.py`）是 `print(soccfg)`（QICK `QickConfig.description()`，很長）之外的精簡替代：generator / readout 各一張表，每列一個 channel，欄位為 type / 物理 port label / sample rate / max pulse 或 buffer length。
- port label（如 `0_230`）採 ZCU216 RF-DC 慣例 `block_(tile+228)`（DAC）/ `block_(tile+224)`（ADC），是 board-specific 假設，故 **只支援 ZCU216**，其他 board 直接 `raise NotImplementedError`（fast fail），不靜默產生誤導標籤。

## 快速查表（需求 -> 優先看哪裡）

- 想改「每回合後做什麼」：先看 `ImproveAcquireMixin` 各 `finish_round()`，再對照 `AcquireMixin.finish_round()`。
- 想改「accumulated 輸出格式」：看 `ImproveAcquireMixin._process_accumulated()` 與 `AcquireMixin._average_buf()`。
- 想改「時間軸或原始資料格式」：看 `AcquireMixin.get_time_axis()`、`get_raw()`。
- 想改「執行控制（early stop）」：看 `EarlyStopMixin.acquire()` 的 `cancel_flag` 參數與 `finish_round()`。

## Runtime event time

Timeline runtime 事件直接用 s14 保存輸出 offset，不另配事件時間 data register。
Pulse、ConfigReadout、Trigger 在 preprocess 後以 `t_regs['t'] = 's_out_time'` 交回 QICK expand。
同基準的下一事件只加差值。原生 macro／raw 寫 s14 或 branch 合流後會重新載入基準。
一般 Repeat 用 s14 作時間指針。TIME 更新獨立的 reference clock，不讀寫 s14。

## QICK ASM dict 注意事項

- 在 QICK asm v2，label definition 和 label reference 都可能使用 `LABEL` key，但需要靠 `CMD` 區分：
  - label definition 通常是只有 `LABEL`、沒有 `CMD` 的 dict。
  - `jump()` / `cond_jump()` 會產生 `{"CMD": "JUMP", "LABEL": ...}`。
  - `call()` 會產生 `{"CMD": "CALL", "LABEL": ...}`。
  - 大 program memory 模式可能先用 `REG_WR` + `SRC: "label"` / `LABEL` 寫入 `s15`，再用 `ADDR: "s15"` 跳轉。
- IR parser 以 `CMD` 區分 label 的用途。有 `CMD` 時保留 instruction 並把 `LABEL` 視為 reference；只有 label definition 才建立標籤。

## IR pass 維護注意事項

- `IRBranch` 只有 `cases: list[IRNode]`，不含 `insts`；有 `insts` 的是 `BasicBlockNode` / `BlockNode`。
- `UnrollLoopPass` 依 `IRLoop.n: Union[int, Register]` 決定是否可展開；`n` 為 `Register` 時無法靜態展開，不要從 QICK loop control 指令猜 iteration count，除非後續已建立可靠的 typed instruction model。
- `ZeroDelayDCEPass` 只刪除 lower-level `TIME C_OP=inc_ref LIT=#0` 這類 reference-time zero increment，不刪 tagged、register-driven、`set_ref` 或 `NOP`。
- `TimedMergePass` 將 literal `TIME inc_ref #N` 吸收入下游指令的 `@T` 欄位（`TimeOffset += pending_lit`）。`PortWriteInst`/`DportWriteInst`/`WmemWriteInst` 等帶 `@T` 的硬體寫入指令可吸收；不帶 `@T` 但不讀寫 `s14` 的同類指令則透明通過（不中斷累積）；只有讀寫 `s14` 或 `WaitInst` 才會強制 flush。

## 原始碼出處

### 本專案中的對應實作

- `lib/zcu_tools/program/acquisition.py`
- `lib/zcu_tools/program/v2/base.py`
- `lib/zcu_tools/program/__init__.py`

### QICK upstream（.venv 來源）

- `/.venv/lib/python3.13/site-packages/qick/qick_asm.py`
  - `class AcquireMixin`
  - `acquire()`, `acquire_decimated()`, `prepare_round()`, `finish_round()`, `finish_acquire()`
  - `_average_buf()`, `_apply_threshold()`, `get_raw()`, `get_time_axis()`
- `/.venv/lib/python3.13/site-packages/qick/__init__.py`
  - `QickConfig`, `AveragerProgram`/`RAveragerProgram`/`NDAveragerProgram` 匯出入口

## 維護建議

- 若 QICK 升版後 `AcquireMixin` 介面變動，優先比對：
  - `acquire_params` 欄位名稱
  - `finish_round()` 回傳語意（是否仍為「還有下一 round」）
  - `_average_buf()` 的輸入 shape 假設
- `EarlyStopMixin._finish_accumulated_round` 鏡射 QICK `AcquireMixin.finish_round` 的 accumulated branch，因為上游 loop 沒有提前退出的 seam。QICK 升版時，維護者須人工對照上游 `AcquireMixin.finish_round` 並移植相關變更。維護者也須透過現有公開介面的測試驗證 acquisition 與 early-stop 行為。
- mixin 疊加順序有語意。`tests/program/test_acquire_mro.py` 保留既有 `finish_round()` 解析鏈與 `MyProgramV2` 相對順序測試；新增 mixin 時，以 `tests/program/test_acquisition.py` 的公開 acquisition 案例驗證加權、round hook 與取消行為，不固定完整 MRO 排列。
