# QICK Note for `experiment/v2`

**Last updated:** 2026-06-25（singleshot population persistence）

這份筆記整理 `experiment/v2/` 的整體設計，說明 Experiment 層與 Task 層的分工、典型實驗的撰寫範本，以及各子模組的角色。`runner/` 的細節另見 `runner/README.md`。

---

## 兩層架構

`experiment/v2/` 有兩層抽象，分別解決不同問題：

- `Experiment` 層（`experiment/base.py`）
  - `AbsExperiment[T_Result, T_Config]`：最小基底，只提供 `last_result` 快取（不保留 `last_cfg`），`run` / `analyze` 由各子類別自行實作。
  - `PersistableExperiment[T_Result, T_Config]`：opt-in 持久化基底，宣告 class-level `AXES_SPEC`（`AxesSpec`）後即繼承共用的 `save()` / `load()`（見「持久化」一節）。
  - `ExperimentProtocol`：結構性合約（runtime_checkable），描述所有實驗共有的 `last_result` / `run` / `analyze` / `save` / `load` 表面，刻意開放讓實驗自行擴充方法。
- `Task` 層：`AbsTask` / `Task` / `BatchTask` / `RetryBatchTask` / `Scan` / `RepeatOverTime`（`runner/`）
  - 內部執行樹：驅動硬體、更新 liveplot、串接 sweep/repeat/retry

Experiment 是使用者（notebook）呼叫的入口；Task 是 Experiment 內部組裝出的執行樹。多數 `*Exp` 類別在 `run()` 裡組一棵 Task tree，然後交給 `run_task()` 執行。

---

## 目錄佈局

```text
experiment/v2/
├── __init__.py              # 暴露 onetone / twotone / singleshot / ... 子模組與 LookbackExp
├── lookback.py              # LookbackExp（decimated readout trace，用來校準 trig_offset）
├── runner/                  # 任務執行框架（見 runner/README.md）
├── utils/                   # 共用工具（merge_result_list, sweep2array, round_zcu_*, SNR）+ tracker/（KMeansTracker / MomentTracker）
├── onetone/                 # Readout 光譜：freq, power_dep, flux_dep, SA
├── twotone/                 # Qubit 光譜 + 時域（rabi, time_domain, reset, ro_optimize, rb, ...）
├── singleshot/              # 單次讀取相關（ge, check, len_rabi, ac_stark, mist, t1）
├── fastflux/                # 快速 flux 掃描（t1, mist, twotone, distortion）
├── mist/                    # MIST（measurement-induced state transition）flux/power dep
├── jpa/                     # JPA 校準（flux/freq/power/自動最佳化）
├── overnight/               # 隔夜穩定度量測（OvernightExecutor + singleshot/t1 task）
└── autofluxdep/             # FluxDepExecutor：多 task 自動 flux 掃描（搭配 FluxoniumPredictor）
```

---

## AbsExperiment 與 Result 合約

```python
class AbsExperiment(Generic[T_Result, T_Config]):
    def __init__(self) -> None:
        self.last_result: Optional[T_Result] = None
    # 最小基底：只有 last_result 快取；run / analyze 由各子類別實作。
    # 持久化（save / load）改由 PersistableExperiment 提供（見「持久化」一節）。
```

- **`T_Result`**：每個實驗各自定義的結果型別，為 `@dataclass(frozen=True)`。必須宣告 `cfg_snapshot: Optional[T_Config] = None` 欄位。
- **`run()`** 的呼叫慣例：`exp.run(soc, soccfg, cfg)`，回傳 `T_Result` 實例。
- **`last_result`**：`run()` 結束後寫入；`analyze` / `save` 可以省略 `result` 參數直接吃最後一次結果。`last_result` 內部攜帶 `cfg_snapshot` 屬性。不另外提供 `last_cfg` 屬性。
- **`last_result` 記帳由 decorator DRY**：`record_result`（套在 `run` / `load`）把回傳值寫入 `self.last_result`；`retrieve_result`（套在 `analyze` / `save`）在 `result` 參數為 `None` 時回退到 `self.last_result`（依參數名定位，與其在簽名中的位置無關）。
- **解包與存取**：所有對 Result 物件的存取均採用屬性存取（Property access，例如 `result.freqs`、`result.signals`），不可直接解包為 tuple。

---

## 持久化：PersistableExperiment + AxesSpec（ADR-0027）

實驗量測資料的存取走 **labber_io 原生 axes-list**，不再經 datasaver 的 dict 殼（`save_data` / `load_data` 已刪除）。

- **opt-in 基底**：要有持久化的實驗繼承 `PersistableExperiment[T_Result, T_Config]`（而非 `AbsExperiment`），並在類別層宣告 `AXES_SPEC`，即繼承共用的 `save()` / `load()`。未遷移的實驗留在最小的 `AbsExperiment` 上、各自保留不相容的 save/load 簽名。
- **宣告式 spec**（`experiment/axes_spec.py`）：
  - `AxesSpec(axes, z, result_type, cfg_type, tag)` — `axes` 是 `tuple[Axis, ...]`、`z` 是 `ZSpec`、`result_type` 是 frozen Result dataclass、`cfg_type` 是該實驗 Cfg、`tag` 是有層次的 on-disk tag（如 `"onetone/freq"`）。建構時 Fast-Fail：spec 引用的 `field_name` 必須是 Result 真實欄位，且 Result 必須有 `cfg_snapshot` 欄位。
  - `Axis(field_name, label, unit, scale=IDENTITY, dtype=np.float64)` — 把 Result 的某個軸欄位映到 on-disk channel；`scale` 帶 SI 單位轉換（`disk = memory * scale`），常數 `IDENTITY` / `MHZ_TO_HZ` / `US_TO_S`（頻率存 Hz、時間存 s，記憶體內仍是 MHz / us）。
  - `ZSpec(field_name, label, unit, dtype=np.complex128)` — log（z）channel。
- **inner-first 軸序慣例**：`axes` 以 inner-first 排列，`z.shape == tuple(len(ax) for ax in reversed(axes))`（inner 軸恆為 z 的最後一維）。**`load` 是 `save` 的恒等逆，兩邊都不做 caller-side transpose**。`load()` 只接受 canonical 檔案：axis count/name/unit、z channel name/unit 與 z shape 都必須符合 `AXES_SPEC`。
- **單位反轉與 cfg**：`save()` 對每個 axis 乘 `scale` 後寫盤；`load()` 除回 `scale` 並 cast 回 `dtype`，是 `save()` 的逐欄逆運算。cfg snapshot 透過 comment channel 走 `make_comment` / `parse_comment`（`load()` 以 `cfg_type.validate_or_warn` 還原），不佔 axes / z。`save()` 在 `cfg_snapshot` 為 `None` 時拋 `ValueError`。
- **grouped experiment dataset**：單一 Experiment Result 若含多個 peer Dataset Role，仍只產生一個 grouped `.hdf5` Experiment Data File。`CPMG_Exp` 使用 roles `lengths` / `signals`，axes 為 inner-first 的 `Time Index`、`Number of Pi`，盤上 `lengths` 單位為 seconds，記憶體內仍回復為 us。legacy `.npz` 不由 runtime `load()` 讀取，需用 `script/migrate_experiment_data.py --experiment twotone/cpmg` 轉成 grouped HDF5。
- **single-role 離散狀態軸**：bath reset freq-gain 把四點 pi/2 tomography phase 視為同一個 Result 的第三個 sweep axis；bath reset length 把 phase 視為第二個 axis，Result-native shape 為 `(Nlength, 4)`；`CKP_Exp` 把 ground/excited prepared state 視為 `initial_states` axis；`GE_Exp` 把 ground/excited prepared state 視為 `prepared_states` axis，Result-native shape 為 `(2, Nshot)`；singleshot `len_rabi` 與 MIST `power` / `freq` / `pre_freq` 把 `g/e` population components 視為 `population_states=[0, 1]` axis，canonical shape 為 `(Nsweep, 2)`。這類 homogeneous Result 存成單一 `.hdf5`，離散狀態不是 Dataset Role，也不再拆成多個 sidecar artifact；legacy artifact 只透過 `script/migrate_experiment_data.py` 轉換，舊 singleshot population HDF5 的 `(2, Nsweep)` z 只在 converter 邊界 transpose。

詳見 [[0027]]。

---

## 典型 `Exp.run()` 範本（以 `onetone/freq.py` 為例）

幾乎所有 `*Exp.run()` 都遵循這個樣板：

```python
@record_result                                                  # 自動把回傳值寫進 last_result
def run(self, soc, soccfg, cfg: FreqCfg) -> FreqResult:
    orig_cfg = deepcopy(cfg)                                     # 1. 執行前快照（給 cfg_snapshot）
    setup_devices(cfg, progress=True)
    modules = cfg.modules

    freqs = sweep2array(cfg.sweep.freq, "freq", {...})          # 2. 預測 sweep 點（已 round 到 ZCU 格點）

    def measure_fn(ctx, update_hook):                            # 3. 包裝單次測量
        cfg = ctx.cfg
        cfg.modules.readout.set_param("freq", sweep2param(...))
        return ModularProgramV2(
            soccfg, cfg,
            modules=[Reset(...), PulseReadout(...)],
            sweep=[("freq", freq_sweep)],
        ).acquire(
            soc,
            progress=False,
            round_hook=update_hook,
            stop_checkers=[ctx.is_stop],   # ← 外部中斷支援
        )

    with LivePlot1D("Frequency (MHz)", "Amplitude") as viewer:   # 4. 即時繪圖
        signals = run_task(
            task=Task(measure_fn=measure_fn, result_shape=(len(freqs),)),
            init_cfg=cfg,
            on_update=lambda ctx: viewer.update(freqs, signal2real(ctx.root_data)),
        )

    return FreqResult(                                          # 5. 回傳（cfg 走 cfg_snapshot）
        freqs=freqs, signals=signals, cfg_snapshot=orig_cfg
    )
```

關鍵元素：

1. **`orig_cfg = deepcopy(cfg)`**：在方法最開頭就拍下執行前快照，最後以 `cfg_snapshot=orig_cfg` 寫進 Result。不另存 `last_cfg`，cfg 一律由 Result 攜帶（見「AbsExperiment 與 Result 合約」）。新版 `run()` 的輸入 `cfg` 已是 `CfgModel` 型別（型別即驗證），不再做 `model_validate`。
2. **`sweep2array`**（`utils/round_zcu.py`）把 `SweepCfg` 展成實際會量到的點（已套 ZCU 的 freq/time/gain 量化），用來畫圖 / 存檔。
3. **`measure_fn(ctx, update_hook)`** 是傳給 `Task` 的單次測量函式；`update_hook` 是 round-level callback，每個 round 結束時被 QICK 呼叫以更新 pbar 與 liveplot。task runner（`run_task` 的 `init_cfg`）內部會 `deepcopy` 一份，`measure_fn`/`before_each` 裡的 mutation 都作用在那份副本（`ctx.cfg`）上，不會汙染 `orig_cfg`。
4. **LivePlot**：`LivePlot1D` / `LivePlot2D` 是 context manager，`on_update` 裡每次都以當前 `ctx.root_data` 重畫。
5. **回傳 Result**：`run()` 直接 `return XxxResult(...)`，由 `@record_result` decorator 寫入 `last_result`（給 `analyze` / `save` 使用），Result 內部攜帶 `cfg_snapshot`。

### sweep 參數 mutation 的歸屬：搬進 `measure_fn`

把 sweep 參數綁到 pulse（`modules.xxx.set_param("freq"/"gain"/"length", ...)`）一律在 `measure_fn` 內對 `ctx.cfg` 做，不要在 `run()` body 頂層直接改傳入的 `cfg`。`sweep2array` 只讀 `cfg.sweep.*`（sweep 定義）與通道，不依賴 pulse param，所以 `set_param` 不需要提前到外層。若 `measure_fn` 寫成 lambda 而需要塞 `set_param`，改寫成具名函式。

只有兩類「副本外操作」是有意保留的：

- **device setup**：`set_*_in_dev_cfg(cfg.dev, ...)` + `setup_devices(cfg, progress=True)` 需要在掃描前先把硬體帶進度地初始化到起點，留在 `run()` body；`progress=True` 本身即通知，不另加 warn。
- **singleshot 強制 reps/rounds**：singleshot 家族（`ge` / `check` / ...）的 `run()` 開頭以 `cfg = deepcopy(cfg)` 重綁本地副本後才改 `cfg.rounds = 1` / `cfg.reps = cfg.shots`，並在覆寫前 `warnings.warn(...)`。重綁後的 mutation 作用在本地副本，非副本外。

---

## Config 組合慣例

每個 Exp 使用 `ConfigBase`（`zcu_tools.cfg_model.ConfigBase`）定義設定，通常由三層組成：

```python
class FreqModuleCfg(ConfigBase):                   # 該實驗用到的 modules
    reset: Optional[ResetCfg] = None
    readout: PulseReadoutCfg

class FreqSweepCfg(ConfigBase):
    freq: SweepCfg

class FreqCfg(ProgramV2Cfg, ExpCfgModel):          # 主要 Cfg = program cfg + exp cfg base
    modules: FreqModuleCfg
    sweep: FreqSweepCfg
```

- `ConfigBase`（`zcu_tools/cfg_model.py`）是 `BaseModel` 的子類別，預設 `extra="forbid"`, `validate_assignment=True`，並提供 `with_updates()` 與 `to_dict()` 工具。所有模組/實驗 cfg 都應繼承 `ConfigBase` 而不是直接用 `BaseModel`。
- `ProgramV2Cfg`（來自 `program/v2`）定義 QICK 程式需要的欄位（`reps`、`rounds`、...）。
- `ExpCfgModel`（`experiment/cfg_model.py`）提供共用欄位（目前含 `dev`）與統一驗證行為。
- `SweepCfg` 已移至 `program/v2/sweep.py`（從 `zcu_tools.program.v2` import），繼承 `ConfigBase`，並帶有 `@model_validator` 驗證 `start/stop/step/expts` 一致性。

---

## Signal/Raw 處理慣例

- **`raw` 格式**：QICK 回傳的 raw 一般是 `list[ndarray]`，第一個元素 shape 為 `(nro, ..., 2)`（IQ 兩個實數）。
- **`default_raw2signal_fn`**（`runner/task.py`）：`raw[0][0].dot([1, 1j])` — 取第 0 個 RO channel、該 channel 的第 0 次 readout，再把 IQ 轉成 complex。絕大多數 1D 實驗用這個。
- **客製化 `raw2signal_fn`**：當 acquire 形狀不同（例如 `acquire_decimated`、多 RO、singleshot 用 `KMeansTracker` / `MomentTracker`）時，在 `Task(raw2signal_fn=...)` 傳入。
- **`signal2real` 函式**：每個 Exp 檔案會定義 local 的 `xxx_signal2real`（通常 `np.abs`），給 liveplot 用；analyze 階段可能換成 phase / real。
- **peak-picking smoothing**：ro-optimize 與 reset 這類以 SNR/map argmax 找最佳點的分析預設使用 `smooth_method="wavelet"`；`smooth_method="gaussian"` 保留為舊 Gaussian 對照。`smooth` 是通用強度：Gaussian 時是 sigma，wavelet 時是 threshold scale。

---

## Executor 模式（`autofluxdep` / `overnight`）

當要在外層再疊一層「sweep 多個子實驗」的場景（例如掃 flux × {freq, t1, t2echo, ...}），會用 Executor。

兩個 Executor 共用同一個基底 `MultiMeasurementExecutor`（`runner/multi_executor.py`，見 `runner/README.md`），由它提供版面排版（`make_ax_layout` / `make_plotter`）、`record_animation` 的 FFMpeg facet 與 `_run_with_plotting`（包 `run_task` + 動畫錄製）；子類別只各自實作 `run()`（不同的外層 driver 與 cfg/env 前置）。

- `FluxDepExecutor`（`autofluxdep/executor.py`）：註冊多個 `MeasurementTask`，組一棵 `RetryBatchTask(tasks).scan("flux", ...)`，並與 `FluxoniumPredictor` 協作（`before_each` 內依模型預測下一個 flux 點並設 flux device）。
- `OvernightExecutor`（`overnight/executor.py`）：類似概念，但改用 `RetryBatchTask(tasks).repeat("Iter", ...)` 在時間軸上重複。

兩者用的 `RetryBatchTask` 會對每個子 task 套上重試（`retry_time` 預算）；`record_animation(mp4_path)` 需要 `ffmpeg`。

實作新 executor task 時：繼承各 app 自己的 `MeasurementTask[T_Result, T_RootResult, T_PlotDict]`（兩個 app 因 cfg 泛型不同而各保留一份 ABC，共有的繪圖合約由 `PlottableMeasurement` Protocol 捕捉），實作 `num_axes()`（告訴 executor 需要幾個 ax）、`make_plotter()` / `update_plotter()`（建立並更新 `AbsLivePlot`）、`save()`（外層 sweep 版的存檔）。

---

## `experiment/utils/` 子模組重點

新增了 `comment.py`，提供兩個函式：

- **`make_comment(cfg, comment=None)`** — 把 `ConfigBase` 轉成 JSON 字串（含 cfg dump、可選文字說明、timestamp），用於實驗存檔時的 comment 欄位。
- **`parse_comment(comment)`** — 解析 `make_comment` 產生的 JSON，回傳 `(cfg_dict, comment_str, timestamp_str)`；若 JSON 解析失敗則全部回傳 `None`。

這兩個函式已透過 `experiment/utils/__init__.py` 直接 export。

## `utils/` 子模組重點

- **`sweep2array(sweep_cfg, name, {"soccfg", "gen_ch", "ro_ch"})`** — 展開 `SweepCfg` 為 numpy array，已套 ZCU 量化（`round_zcu_freq/time/gain/phase`）。Exp 幾乎都靠這個產生 x 軸。
- **`round_zcu_*`** — 單點版本的量化函式；`round_sweep_dict` 同時處理整個 sweep dict。時間的量化有特殊處理：預先減去 `0.5 * one_cycle` 以匹配 QICK sweep 用 `np.trunc` 的行為。多點 sweep 的 step 若在量化後變成 0，會 fast-fail 並要求放大 span 或減少 expts，避免 GUI/agent 看到低階 `SweepCfg` 一致性錯誤。
- **`merge_result_list(list_of_results)`** — 把 `list[dict[name, ndarray]]` 遞迴轉成 `dict[name, ndarray]`（外層 list 變成最外層維度）。Executor 取得 `Scan` 結果後呼叫它把 `list` 轉成 stacked array。
- **`estimate_snr` / `snr_as_signal` / `snr_checker`** — SNR 估計與 early-stop：當 SNR 達門檻時提前結束測量，節省時間。`snr_as_signal` 給 singleshot（吃 `MomentTracker` 的 raw）用。

---

## Lookback & Tracker

- **`LookbackExp`**（`lookback.py`）：用 `acquire_decimated` 拿 RO time trace，用來判斷 `trig_offset` 是否對齊 readout pulse；`analyze()` 從最大點往回找第一個低於 `ratio * max` 的點當作建議 offset。**`reps` 會被強制改成 1**（decimated 不支援多 rep 平均）。
- **`KMeansTracker`**（`utils/tracker/kmeans.py`）：線上維護 `(..., 2)` IQ 樣本的動態多群統計（每群 `cluster_mean` / `cluster_covariance` / `cluster_center` / `cluster_weight`），支援 leading dims 與 `share_axis` 共用群組。內部以增量統計維護每群矩，無需保留原始樣本；singleshot 家族可直接用其 cluster 統計估計 SNR。

---

## Twotone RB 測量策略

- RB 量測採「每個 seed 一個 program」的結構：在 program 內用 sweep index 遍歷 depth，而不是在 host 端對 `(seed, depth)` 重複建立程式。
- seed 對應的 random gate prefix 長度與 recovery gate id 以 `LoadValue` 從 dmem 查表；random 段用 register-driven `Repeat`，最後 recovery 用獨立 `ComputedPulse`。
- random 段 `Repeat` 固定使用 `n="rand_len"`；當 random 序列可為空時，交由 `LoadValue(values=[])` 的 no-op 行為吸收，不在 RB 層做 sentinel value 邏輯。
- rounds 交給 `acquire()` 原生流程處理，`Task` 層只掃 seed。

---

## 外部中斷支援（stop_checkers）

所有在 Task 框架（`measure_fn`）內呼叫的 `acquire()` 都應傳入 `stop_checkers=[ctx.is_stop]`，使其能感知來自 `ActiveTask` 或手動 `set_stop()` 的中斷信號。中斷以 round-level 粒度生效（每個 round 結束後檢查），不影響正在執行中的 round。

- 若使用 SNR early stop，再加入 `snr_checker(ctx, threshold, signal2real_fn)` 作為第二個 checker。
- `singleshot/ge.py`、`singleshot/check.py` 的裸 `prog.acquire(progress=True)` 不在 Task 框架內，不加 stop_checkers。
- `singleshot/t1/util.py` 的 `measure_with_sweep` 透過 `**acquire_kwargs` 轉發，並以 `setdefault` 確保 `stop_checkers` 預設為 `[ctx.is_stop]`（caller 不傳時仍套用預設）。

---

## 寫新 Experiment 時的檢查清單

1. 定義 `XxxModuleCfg` / `XxxSweepCfg`（通常繼承 `ConfigBase`）與 `XxxCfg = ProgramV2Cfg + ExpCfgModel + 自己欄位`。
2. 繼承 `PersistableExperiment[T_Result, XxxCfg]`（要有持久化）並宣告 class-level `AXES_SPEC`（`AxesSpec`）即繼承 `save` / `load`；只需自行實作 `run`（套 `@record_result`）/ `analyze`（套 `@retrieve_result`）。Result dataclass 須有 `cfg_snapshot` 欄位。
3. `run()` 模板：開頭 `orig_cfg = deepcopy(cfg)` → `sweep2array` → 組 `measure_fn` → `with LivePlot: run_task(Task(...))` → 回傳 `XxxResult(..., cfg_snapshot=orig_cfg)`（`@record_result` 自動寫入 `last_result`，不存 `last_cfg`）。
4. `measure_fn` 內的 `acquire()` 必須傳入 `stop_checkers=[ctx.is_stop]`（可加 SNR checker）。
5. 持久化由 `AXES_SPEC` 宣告：每個 `Axis` 帶 `scale`（頻率 `MHZ_TO_HZ`、時間 `US_TO_S`）讓盤上是 SI 單位、記憶體內維持習慣單位，`tag` 取有層次的名字（`"twotone/rabi/len"`），axes 以 inner-first 排列；繼承的 `save` / `load` 自動依 spec 做單位轉換與恒等逆 round-trip，無需自行寫 save/load。
6. 如果有多個 sweep 軸，考慮用 `Task.scan()` 或直接在 `ModularProgramV2(sweep=[...])` 裡放多個 sweep。
7. 如果要在樹的外層疊 flux / time sweep，改繼承 `AbsTask`（或 `MeasurementTask`）而不是 `AbsExperiment`，再由 Executor 組起來。

---

## 更新紀錄

| 日期 | Codebase commit | 說明 |
|------|-----------------|------|
| 2026-06-25 | — | singleshot `len_rabi` 與 MIST `power` / `freq` / `pre_freq` 持久化為 single-role canonical HDF5：`population_states=[0, 1]` 是 inner axis，sweep axis 是 outer axis，Result-native z shape 為 `(Nsweep, 2)`；legacy `(2, Nsweep)` population HDF5 只透過 `script/migrate_experiment_data.py` 轉換。 |
| 2026-06-25 | — | `GE_Exp` 與 bath reset `LengthExp` 持久化為 single-role canonical HDF5：GE 使用 `shot_indices` / `prepared_states` axes，bath length 使用 `phases` / `lengths` axes；runtime `load()` 嚴格驗證 canonical axis/z metadata 與 shape，legacy GE / bath length 檔只透過 migration script 轉換。 |
| 2026-06-24 | — | `CKP_Exp` 持久化為 single-role 3D HDF5，`initial_states=[0, 1]` 是第三個 axis；runtime 不再寫/讀 `_ground` / `_excited` sidecar，legacy sidecar 透過 `script/migrate_experiment_data.py --experiment twotone/ckp` 轉換。 |
| 2026-06-24 | — | bath reset `freq_gain` 持久化為 single-role 3D HDF5，phase 是同一 Result 的第三個 sweep axis；legacy phase sidecar artifact 不走 runtime save/load。 |
| 2026-06-24 | — | ADR-0027 Phase 002：`CPMG_Exp.save/load` 改為單一 grouped `.hdf5`，roles 固定為 `lengths` / `signals`；legacy `.npz` 只透過 `script/migrate_experiment_data.py --experiment twotone/cpmg` 轉換。 |
| 2026-06-24 | — | ADR-0027 持久化遷移：實驗資料改走 labber_io 原生 axes-list，刪除 datasaver dict 殼（`save_data` / `load_data`）；新增 `PersistableExperiment` opt-in 基底 + class-level `AXES_SPEC`（`AxesSpec` / `Axis` / `ZSpec`，inner-first、SI scale、load = save 恒等逆），實驗只留宣告式 spec 與 `record_result` / `retrieve_result` decorator，不再各自手寫 save/load；新增「持久化」一節並改寫合約／範本／Checklist。#10：`FluxDepExecutor` / `OvernightExecutor` 抽出共用基底 `MultiMeasurementExecutor`（版面 + `record_animation` FFMpeg facet + `_run_with_plotting`），兩者改用 `RetryBatchTask`，更新「Executor 模式」一節。 |
| 2026-06-05 | `cfc86975` | 移除所有非 executor 實驗的 `self.last_cfg` 中介屬性，cfg 統一由 Result 的 `cfg_snapshot` 攜帶；`run()` 改用開頭 `orig_cfg = deepcopy(cfg)` 快照；補上 `onetone/flux_dep`、`twotone/ckp`、`twotone/fluxdep` 缺失的 cfg_snapshot；修正 `len_rabi`/`t2echo`/`t2ramsey` 的 load() 把 cfg 誤存 last_cfg 導致 cfg_snapshot 永遠為 None 的 bug；改寫範本與 Checklist 第 3 項。再把各實驗 `run()` body 頂層的 `modules.xxx.set_param(...)` 副本外 mutation 一律搬進 `measure_fn`（作用於 `ctx.cfg`），只保留 device-setup 與 singleshot reps/rounds 兩類有意例外（見「sweep 參數 mutation 的歸屬」）。autofluxdep/overnight Executor 不在此變更範圍。 |
| 2026-05-25 | `fb0ffc22` | 為所有 Task 框架內的 acquire() 加入 stop_checkers=[ctx.is_stop]，支援外部中斷；新增「外部中斷支援」章節；Checklist 新增第 4 項。 |
| 2026-04-27 | `3f9bb55f` | 對齊近 30 天變更：`runner/QICK_NOTE.md` 改為 `runner/AI_NOTE.md`，`ConfigBase` 匯入路徑改為 `zcu_tools.cfg_model`，Checklist 改為繼承 `ConfigBase`。 |
| 2026-04-26 | `cd0bc869` | Config 慣例改用 `ConfigBase`；`SweepCfg` 移至 `program/v2`；新增 `experiment/utils/comment.py` 說明 |
