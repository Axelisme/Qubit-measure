# `zcu_tools.program` — QICK integration

**Last updated:** 2026-10-29，herald instance 與跨 acquire 分工

**讀者：** framework-dev。QICK 整合、程式建構與 acquisition 依賴速查。

這份筆記整理 `lib/zcu_tools/program` 對 QICK 的實際依賴，目的是讓後續開發能快速定位「應該看哪個 QICK 類別/方法」，而不用每次從頭追。

## Program 層定位

`acquisition.py` 擴充 QICK 的 `AcquireMixin`，負責 acquisition 的型別封裝、追蹤、提前停止與 round hook。`v2/base.py` 定義 v2 program 基底 `MyProgramV2`；它使用這些 acquisition mixin，但負責的是 v2 program 結構，兩者不是同一種 base。

## Explicit readout classification

`acquire(soc, classify=...)` accepts a `Mapping[int, ReadoutClassifier]`.
Keys are ADC channel numbers, not declaration positions, and must exactly match
the declared ADCs. Missing or extra keys fail before transport acquisition.
Omitting classify retains ordinary IQ acquisition. The old
`g_center/e_center/ge_radius` acquisition keywords are removed.

Authors call `program.v2.modules.readout.classification_parameters(cfg)` on
their readout cfg and pass the result explicitly. Acquisition does not discover
calibration through a program registry. Circle requires a Post radius and returns
ground/excited populations; Other is `1-g-e`. Linear returns two populations,
with excited defined by `real(IQ * exp(1j*angle)) > threshold`; equality is ground.
Angle is radians, opposite QICK's projection-angle sign. Both modes normalize
raw sums by each ADC's rounded integration length without subtracting IQ offset,
regardless of `remove_offset`. Simulation materializes raw shots for this same
classification pipeline instead of returning compact IQ means.
`classification_parameters(cfg, require_other=True)` rejects linear and reports
missing circle radius with a GE Post instruction.

## Shot time

`shot_time_s(rounds=..., reps=..., shot_lengths_us=..., relax_delay_us=...)`
從各 sweep 點的 shot 長度與共同 relax 計算完整 acquisition 的 shot 秒數。
輸入時間單位為 µs，rounds 與 reps 為正整數；不計 acquisition 固定開銷與 flux ramp。
模擬 binding 的 pacing 使用同一個公式，不累計 acquisition 的硬體總時間。

## Herald declarations

`PulseReadout.herald_spec(prog, occurrence=...)` derives one `program.herald.HeraldSpec`
from a compiled program. The required occurrence counts this instance's emissions
within each shot; the returned read_index counts all reads on its logical ADC.
One instance may be initialized once and emit both herald and terminal reads.
A separate instance using the same readout settings may emit the terminal read
instead. Sharing readout settings does not require sharing an instance.
Call herald_spec on the instance that emitted the herald; occurrence counts only
that instance's emissions. A later read on the same ADC is required.
Any Repeat/Branch trigger on that ADC, count mismatches and missing calibration
fail before acquisition.
The control-flow restriction includes non-herald reads because QICK's static
trigger counts cannot reliably determine dynamic read indices. Control flow on
other ADCs remains supported. Circle calibration requires the GE Post radius.
The method returns a classifier snapshot and does not register a herald.
Acquisition receives the spec explicitly and admits at most one herald per ADC.
`program.herald.herald_ground_mask` is the shared ground
acceptance helper. It takes complex IQ divided by ADC integration length, without
subtracting offsets. Circle uses the strict g circle and nearest-center region;
linear rotates by exp(+1j*angle) and accepts values at or below threshold.

## Herald post-selection

After compilation, pass the declaration explicitly:

```python
spec = readout.herald_spec(prog, occurrence=0)
data = prog.acquire(soc, herald=[spec], rounds=2, progress=False)
fractions = prog.get_herald_pass_fraction()
# After a completed round, fractions[spec.ro_ch] has the non-reps sweep shape.
```

`herald` accepts a sequence of `HeraldSpec`. Omission or an empty sequence keeps
ordinary acquisition. Each ADC uses its own herald mask. ADCs without a herald
keep all their shots and reads. The selected read must precede another read on
that ADC. Duplicate ADC specs, undeclared ADCs, invalid read indices and a
`reads_per_shot` mismatch raise `ValueError` before transport acquisition.
`trackers` with herald and `acquire_decimated(..., herald=[...])` also raise
`ValueError`.

Acquisition classifies the herald in raw sums divided by ADC sample length,
without subtracting IQ offset. It removes that read, selects accepted shots and
then processes the remaining reads. Ordinary IQ retains the `remove_offset`
policy. With typed `classify`, only accepted signal shots enter classification.
The result ends in g/e populations, and typed classification takes precedence
over QICK `threshold/angle`. Without typed classification, `threshold/angle`
uses QICK's projection sign, radians, offset and equality rules on accepted
shots. It returns excited fraction in I and zero in Q. Scalar and per-ADC-list
projection options retain QICK declaration-order semantics.

The return list stays in ADC declaration order. Each float64 array has shape
`(remaining_reads, *point_dims, 2)`, with only herald reads removed. Within one
acquire, acquisition adds accepted signal sums and counts across completed
rounds, then divides by accepted count. It does not average round means equally.
Zero-pass points have NaN in both components and acceptance fraction zero. Round hooks receive this
same weighted running result.

`get_herald_pass_fraction()` returns a detached dictionary keyed by herald ADC
numbers. Each float64 array contains accepted/attempted shots on the non-reps
point grid, without a read axis. An unswept program returns scalar arrays.
The accessor returns `None` before any herald round completes, after an ordinary
integrated or decimated acquire, or after cancellation before the first completed
round. A new admitted acquire clears previous fractions. Cancellation during a
round discards that round and retains completed-round data and fractions.
Finalization raises `StoppedPartialAcquireError` when no round completed.
A host scan that combines separate acquires owns that combination and weights
only delivered results by their accepted counts. If a builder skips acquire
because of a stop request, the cached program has no new acquisition summary;
its previous fractions must not be counted as another completed acquisition.

`HeraldAcquireMixin` owns only grid-sized sums/counts. It precedes
`RoundHookMixin` to replace the equal-round incremental summary. Existing tracker,
classification and Stop mixin order stays unchanged. Hardware and simulation use
the same shot-processing and weighted-summary methods. Herald forces simulation
to materialize its last-round `RawShots`, even without classification or threshold.
No new per-shot persistence or raw history is added. `get_raw()` retains the
existing transport contract, including the herald read.

## 主要依賴面

- `ImproveAcquireMixin` 直接擴充 `qick.qick_asm.AcquireMixin`，是 `program` 層最核心的 QICK 接點。
- Accumulated acquire 在寫入 chunk 前拒絕超過預期的累計 shots，拋出 `program.acquisition.AcquisitionContractError`。訊息包含預期 shots、先前已收 shots、本次 shots、各 channel 的 IQ 列數與從 0 起算的 round index。錯誤 chunk 不寫入、不裁切，先前 raw chunks 保留，不發布未完成的 round。
- 主要加值分成四類：
  - 型別友善封裝（`TypedAcquireMixin`）：補強 `acquire()` / `acquire_decimated()` / `get_raw()` 等回傳型別。
  - 統計追蹤（`TrackerMixin`）：在 `finish_round()` 內注入 tracker 更新流程。
  - 提前停止（`EarlyStopMixin`）：透過單一 `cancel_flag` 在 `finish_round()` 內中止後續 rounds；accumulated acquisition的內部reps進度走共用`progress_bar.make_pbar` factory，GUI ambient factory與terminal fallback使用同一生命週期，正常或停止都會close並移除暫時進度項。
  - 每輪 callback（`RoundHookMixin`）與 single-shot population/threshold（`SingleShotMixin`）。`round_hook(round_count, raw, cancel_flag)` 只收到 completed round，可呼叫 `cancel_flag.set()` 表示不開始下一 round。

## `AcquireMixin` 行為對照（常見踩點）

- 生命週期：
  - `acquire()`/`acquire_decimated()` 先建立 `acquire_params`，再進入 `prepare_round()` + `finish_round()` 迴圈。
  - `finish_acquire()` 負責 rounds 聚合（accumulated 或 decimated）。
- 擴充掛點：
  - `TrackerMixin.finish_round()`：利用原生 `self.acc_buf`、`self.ro_chs`、`self.avg_level` 更新 tracker。
  - `RoundHookMixin.finish_round()`：每 completed round 呼叫 `round_hook`，輸入增量摘要資料與同一個 stop flag。
  - `SingleShotMixin._process_accumulated()`：保留原生 IQ／threshold 路徑，明確傳入 classify 時回傳 g/e populations。
- 相容性關鍵：
  - 擴充層維持 `AcquireMixin` 的 `extra_args` 傳遞模式，避免破壞原生 acquire 參數流。
  - 若非 `accumulated` 或有 threshold 時，部分統計路徑會顯式 `NotImplementedError`，這是刻意防呆。

## soccfg 摘要顯示（`describe_soc`）

- `describe_soc(soccfg)`（`soc_summary.py`）是 `print(soccfg)`（QICK `QickConfig.description()`，很長）之外的精簡替代：generator / readout 各一張表，每列一個 channel，欄位為 type / 物理 port label / sample rate / max pulse 或 buffer length。
- port label（如 `0_230`）採 ZCU216 RF-DC 慣例 `block_(tile+228)`（DAC）/ `block_(tile+224)`（ADC），是 board-specific 假設，故 **只支援 ZCU216**，其他 board 直接 `raise NotImplementedError`（fast fail），不靜默產生誤導標籤。

## 快速查表（需求 -> 優先看哪裡）

- 想改「每回合後做什麼」：先看 `ImproveAcquireMixin` 各 `finish_round()`，再對照 `AcquireMixin.finish_round()`。
- 想改「accumulated 輸出格式」：看 `SingleShotMixin._process_accumulated()` 與 `AcquireMixin._average_buf()`。
- 想改「時間軸或原始資料格式」：看 `AcquireMixin.get_time_axis()`、`get_raw()`。
- 想改「執行控制（early stop）」：看 `EarlyStopMixin.acquire()` 的 `cancel_flag` 參數與 `finish_round()`。

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
