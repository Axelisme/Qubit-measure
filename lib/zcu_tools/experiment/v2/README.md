# `zcu_tools.experiment.v2` — experiment runtime

**Last updated:** 2026-07-15 — homophasal runtime phase reset

這份筆記整理 `experiment/v2/` 的整體設計，說明 Experiment 層與 runtime 層的分工、典型實驗的撰寫範本，以及各子模組的角色。`runner/` 的細節另見 `runner/README.md`。

---

## 兩層架構

`experiment/v2/` 有兩層抽象，分別解決不同問題：

- `Experiment` 層（`experiment/base.py`）
  - `AbsExperiment[T_Result, T_Config]`：最小基底，只提供 `last_result` 快取（不保留 `last_cfg`），`run` / `analyze` 由各子類別自行實作。
  - `PersistableExperiment[T_Result, T_Config]`：opt-in 持久化基底，宣告 class-level `AXES_SPEC`（`AxesSpec`）後即繼承共用的 `save()` / `load()`（見「持久化」一節）。
  - `ExperimentProtocol`：結構性合約（runtime_checkable），描述所有實驗共有的 `last_result` / `run` / `analyze` / `save` / `load` 表面，刻意開放讓實驗自行擴充方法。
- runtime 層（`runner/`）
  - 一般 Experiment：`SignalBuffer` / `Schedule` / `ProgramBuilder` 表達 Python-like acquire、host loop、batch、ProgramBuilder retry 與 stop。
  - executor workflow：`ResultTree` 實作 executor-owned `BufferProtocol`，持有 outer loop result tree、per-measurement subscription 與 stacked result cache；`Schedule` 編排 outer loop；`MultiMeasurementExecutor` 提供 combined liveplot、recording、retry、measurement init/cleanup、partial-result outcome、figure/writer cleanup 與 `last_cfg` / `last_result` lifecycle。

Experiment 是使用者（notebook）呼叫的入口；一般 `*Exp.run()` 以 `with Schedule(cfg, signals_buffer) as sched` 做單次 run orchestration，把 scan / buffer targeting 留在 Python control flow 中。`Schedule` 負責 cfg deepcopy、typed env 與 `StopSignal`；`SignalBuffer` 負責 update throttling 與 live update callback；`ProgramBuilder.build_and_acquire(...)` 直接建立 program、執行 acquire 並更新 buffer。

`SignalBuffer` + `Schedule` + `ProgramBuilder` 支援一般 program acquire、decimated trace、program-side sweep、host-side scan、repeat、replayable batch、single acquire retry、caller-owned program reuse、SNR early stop 與 custom raw conversion。需要取得 raw shots 的 singleshot `GE` / `Check` path 使用同一個 `Schedule` scope 做 cfg/stop/buffer orchestration，但 leaf 端改由 `ProgramBuilder.build()` 取得 program 後直接呼叫 `program.acquire(...)`，再把 `program.get_raw()` 轉入 `SignalBuffer`。`autofluxdep` / `overnight` 這類 executor-owned 多任務流程也使用同一個 `Schedule` runtime：外層用 executor-owned buffer 保存 result tree，再用 `scan` / `repeat` / `batch` 編排，leaf 用 `ScheduleStep.child(...).buffer(...)` 把 program acquire 寫回 result tree。

---

## 目錄佈局

```text
experiment/v2/
├── __init__.py              # 暴露 onetone / twotone / singleshot / ... 子模組與 LookbackExp
├── lookback.py              # LookbackExp（decimated readout trace，用來校準 trig_offset）
├── runner/                  # experiment runtime（見 runner/README.md）
├── utils/                   # 共用工具（merge_result_list, sweep2array, round_zcu_*, SNR）+ tracker/（KMeansTracker / MomentTracker）
├── onetone/                 # Readout 光譜：freq, power_dep, flux_dep, SA
├── twotone/                 # Qubit 光譜 + 時域（rabi, time_domain, reset, ro_optimize, rb, ...）
├── singleshot/              # 單次讀取相關（ge, check, len_rabi, ac_stark, mist, t1）
├── fastflux/                # 快速 flux 掃描（t1, mist, twotone, distortion）
├── mist/                    # MIST（measurement-induced state transition）flux/power dep
├── jpa/                     # JPA 校準（flux/freq/power/自動最佳化）
├── overnight/               # 隔夜穩定度量測（OvernightExecutor + measurement leaves）
└── autofluxdep/             # FluxDepExecutor：多 measurement 自動 flux 掃描（搭配 FluxoniumPredictor）
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

實驗量測資料的存取走 **labber_io 原生 axes-list**，而非 datasaver 的 dict 殼。

- **opt-in 基底**：要有持久化的實驗繼承 `PersistableExperiment[T_Result, T_Config]`（而非 `AbsExperiment`），並在類別層宣告 `AXES_SPEC`，即繼承共用的 `save()` / `load()`。未遷移的實驗留在最小的 `AbsExperiment` 上、各自保留不相容的 save/load 簽名。
- **宣告式 spec**（`experiment/axes_spec.py`）：
  - `AxesSpec(axes, z, result_type, cfg_type, tag)` — `axes` 是 `tuple[Axis, ...]`、`z` 是 `ZSpec`、`result_type` 是 frozen Result dataclass、`cfg_type` 是該實驗 Cfg、`tag` 是有層次的 on-disk tag（如 `"onetone/freq"`）。建構時 Fast-Fail：spec 引用的 `field_name` 必須是 Result 真實欄位，且 Result 必須有 `cfg_snapshot` 欄位。
  - `Axis(field_name, label, unit, scale=IDENTITY, dtype=np.float64)` — 把 Result 的某個軸欄位映到 on-disk channel；`scale` 帶 SI 單位轉換（`disk = memory * scale`），常數 `IDENTITY` / `MHZ_TO_HZ` / `US_TO_S`（頻率存 Hz、時間存 s，記憶體內仍是 MHz / us）。
  - `ZSpec(field_name, label, unit, dtype=np.complex128)` — log（z）channel。
- **inner-first 軸序慣例**：`axes` 以 inner-first 排列，`z.shape == tuple(len(ax) for ax in reversed(axes))`（inner 軸恆為 z 的最後一維）。**`load` 是 `save` 的恒等逆，兩邊都不做 caller-side transpose**。`load()` 只接受 canonical 檔案：axis count/name/unit、z channel name/unit 與 z shape 都必須符合 `AXES_SPEC`。legacy 單檔案的 label/unit 差異不放寬 runtime loader，而是在 `zcu_tools.experiment.legacy_migration` converter 邊界轉成 canonical HDF5；GUI adapter fallback 也只呼這個 converter 後再走同一個 strict `load()`。
- **單位反轉與 cfg**：`save()` 對每個 axis 乘 `scale` 後寫盤；`load()` 除回 `scale` 並 cast 回 `dtype`，是 `save()` 的逐欄逆運算。cfg snapshot 透過 comment channel 走 `make_comment` / `parse_comment`（`load()` 以 `cfg_type.validate_or_warn` 還原），不佔 axes / z。`save()` 在 `cfg_snapshot` 為 `None` 時拋 `ValueError`。
- **save path ownership**：`PersistableExperiment.save()` 寫入 caller 傳入的 final path；既有 path 由 datasaver writer fast-fail，不自動 suffix、不提供 overwrite 參數。GUI / runner / notebook 若需要 unique filename，必須在呼叫 `save()` 前用 `reserve_labber_filepath` 或自己的 orchestration policy 決定 final path。
- **grouped experiment dataset**：單一 Experiment Result 若含多個 peer Dataset Role，仍只產生一個 grouped `.hdf5` Experiment Data File。`GroupedAxesSpec` / `RoleSpec` 是 experiment 層的 semantic schema：每個 role 宣告 role name、inner-first axes、z/data field mapping、dtype、unit 與 scale；common helper 依 spec 組 `GroupedLabberData` payload、驗證 required roles / axis metadata / z shape、重建 comment/cfg snapshot，再交 typed builder 還原 Result。`RoleSpec` 只描述 mechanical mapping，不攜帶 arbitrary transform callback；需要把多個 role array 合成既有 Result 欄位（例如 auto-optimize 的 `params`）時，在 `GroupedAxesSpec` 的 typed builder 邊界完成。
- **grouped experiment roles**：`CPMG_Exp` 使用 roles `lengths` / `signals`，axes 為 inner-first 的 `Time Index`、`Number of Pi`，盤上 `lengths` 單位為 seconds，記憶體內仍回復為 us。RO auto-optimize 使用 roles `readout_freq` / `readout_gain` / `readout_length` / `snr`；JPA auto-optimize 使用 roles `jpa_flux` / `jpa_freq` / `jpa_power` / `jpa_phase` / `snr`。頻率與時間在 disk 上使用 SI units（Hz、s），typed loader 重建回 Result 記憶體單位（MHz、us）；JPA phase 是 integer index。這些 runtime `load()` 都只接受 complete grouped HDF5；legacy `.npz` 或 sidecar 只屬 migration input，需用 `script/migrate_experiment_data.py` 轉換。
- **legacy single-file converters**：第一批共用 converter 支援 `onetone/freq`、`onetone/flux_dep`、`twotone/freq`、`twotone/flux_dep`（含 `twotone/flux_dep/freq` alias），把舊 Labber HDF5 的 `Frequency` `MHz/Hz`、`Yoko` flux 軸與 `ADC unit` signal channel 重寫成當前 `AXES_SPEC`。`onetone/flux_dep` 的 canonical axes 是 `(freqs, values)`，對應 Result-native `signals.shape == (Nflux, Nfreq)`。
- **single-role 離散狀態軸**：bath reset freq-gain 把四點 pi/2 tomography phase 視為同一個 Result 的第三個 sweep axis；bath reset length 把 phase 視為第二個 axis，Result-native shape 為 `(Nlength, 4)`；`CKP_Exp` 把 ground/excited prepared state 視為 `initial_states` axis；`GE_Exp` 把 ground/excited prepared state 視為 `prepared_states` axis，Result-native shape 為 `(2, Nshot)`；singleshot `len_rabi`、MIST `power` / `freq` / `pre_freq` 把 `g/e` population components 視為 `population_states=[0, 1]` axis，canonical shape 為 `(Nsweep, 2)`；singleshot `ac_stark` 與 MIST `power_freq` 使用 `population_states` 加兩個 sweep axes，canonical shape 為 `(Ngain, Nfreq, 2)`；singleshot `t1` / `t1_with_tone` 使用 `population_states`、`initial_states` 與 `lengths`，canonical shape 為 `(Nt, 2, 2)`；`t1_with_tone_sweep` 使用 `population_states`、`lengths`、`initial_states` 與 generic `xs`/`Sweep Value` axis，canonical shape 為 `(Nx, 2, Nt, 2)`，只存 Result 的 g/e components，`other` 由 analysis 推導。這類 homogeneous Result 存成單一 `.hdf5`，離散狀態不是 Dataset Role，也不再拆成多個 sidecar artifact；legacy artifact 只透過 `script/migrate_experiment_data.py` 轉換，舊 singleshot population HDF5 的 `(2, Nsweep)` 或 multi-sidecar z 方向只在 converter 邊界重排。

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

    with LivePlot1D("Frequency (MHz)", "Amplitude") as viewer:   # 3. 即時繪圖
        signals_buffer = SignalBuffer(
            (len(freqs),),
            on_update=lambda data: viewer.update(freqs, signal2real(data)),
        )
        with Schedule(cfg, signals_buffer) as sched:
            freq_sweep = sched.cfg.sweep.freq
            sched.cfg.modules.readout.set_param("freq", sweep2param(...))
            _ = (
                sched.prog_builder(soc, soccfg)
                .add_reset(
                    "reset",
                    sched.cfg.modules.reset,
                )
                .add(
                    PulseReadout("readout", sched.cfg.modules.readout),
                )
                .declare_sweep("freq", freq_sweep)
                .build_and_acquire()
            )

        return FreqResult(                                      # 4. 回傳（cfg 走 cfg_snapshot）
            freqs=freqs, signals=signals_buffer.array, cfg_snapshot=orig_cfg
        )
```

關鍵元素：

1. **`orig_cfg = deepcopy(cfg)`**：在方法最開頭就拍下執行前快照，最後以 `cfg_snapshot=orig_cfg` 寫進 Result。不另存 `last_cfg`，cfg 一律由 Result 攜帶（見「AbsExperiment 與 Result 合約」）。`run()` 的輸入 `cfg` 是 `CfgModel` 型別（型別即驗證），不在方法內重做 `model_validate`。
2. **`sweep2array`**（`utils/round_zcu.py`）把 `SweepCfg` 展成實際會量到的點（已套 ZCU 的 freq/time/gain 量化），用來畫圖 / 存檔。
3. **`with Schedule(cfg, signals_buffer) as sched`** 是一般單次 run scope；`sched.cfg` 是 runner-owned deepcopy，mutation 不會汙染 `orig_cfg` 或 caller 傳入的 cfg。`Schedule` 可用 `env=RunEnv(...)` 接 typed dataclass 依賴；env 只放穩定 run context，不放 scan/repeat 動態 value/index，loop state 由 `ScheduleStep.value` / `index` / `path` 表示。`ProgramBuilder.build()` 回傳 program；`run_program(program)` 執行既有 integrated-acquire program；`build_and_acquire()` 直接建立 isolated cfg / program、執行 acquire 並寫回 buffer。`reps` / `rounds` 由 builder 建出的 `program.cfg_model` 讀取；builder owner cfg 可以是 experiment cfg，`ProgramBuilder` 只抽取 `ProgramV2Cfg` runtime 欄位，也可用 `prog_builder(..., cfg=program_cfg)` 明確覆寫；已是 `ProgramV2Cfg` 的 instance/subclass 會保留原型別，沒有任何 runtime 欄位的 cfg 會 fast-fail。Decimated trace 走 `run_program_decimated(...)` / `build_and_acquire_decimated(...)`，不用參數切換 acquire mode；若需先 build program 才能知道 buffer shape，使用 `sched.register_buffer(signals_buffer)` 註冊 caller 建好的 buffer。round-level `update_hook`、`cancel_flag`、pbar 與 raw2signal 由 Schedule runtime 持有，不再透過 `Task` 包裝。
4. **LivePlot**：`LivePlot1D` / `LivePlot2D` 是 context manager，常規寫法是在 `SignalBuffer(on_update=...)` 裡每次以當前完整 buffer ndarray 重畫；`SignalBuffer.set(...)` / `SignalSlot.set(...)` 寫入後自動觸發 update，`trigger_update()` 可手動刷新。
5. **回傳 Result**：`run()` 直接 `return XxxResult(...)`，由 `@record_result` decorator 寫入 `last_result`（給 `analyze` / `save` 使用），Result 內部攜帶 `cfg_snapshot`。

目前所有 Experiment 均走新 runtime 或直接 `SignalBuffer` path，包含 `lookback.py`、`fake.py`、`onetone/*`、`twotone/*`、`singleshot/*`、`jpa/*`、`fastflux/*`、`mist/*`、`autofluxdep/*` 與 `overnight/*`。

`onetone/freq` 的 frequency sampling 有兩種 mode：`linear` 沿用 program-side
`SweepCfg` sweep；`homophasal` 保留同一個 `sweep.freq` 使用者介面，但由已擬合的 resonator
circle 參數產生非等距 frequency array，再把 gen/readout raw frequency words 放進
`LoadWord` tables，由單一 program-side sweep 透過 `PulseReadout.freq_val` /
`ro_freq_val` 逐點更新 runtime wave registers。homophasal 的 generator 與 readout
runtime templates 在每個點重置 DDS phase accumulator，避免非等距頻率順序成為額外相位；
`linear` 保持一般連續相位行為。這類非等距模式仍先把點 round 到硬體格點，並在相鄰點
collapse 時 fast-fail。

### sweep 參數 mutation 的歸屬：搬進 runner-owned cfg

把 sweep 參數綁到 pulse（`modules.xxx.set_param("freq"/"gain"/"length", ...)`）一律在 runner-owned cfg 上做：一般 Schedule 寫法改 `sched.cfg` 或 `step.cfg`；executor leaf 用 `program_step = state.child("raw_signals", cfg=program_cfg)` 建立 program cfg scope 後再改 `program_step.cfg`。不要在 `run()` 頂層直接改 caller 傳入的 `cfg`。`sweep2array` 只讀 `cfg.sweep.*`（sweep 定義）與通道，不依賴 pulse param，所以 `set_param` 不需要提前到外層。

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
- 每個 concrete experiment 直接宣告自己的 local module cfg，並組合
  `ProgramV2Cfg` / `ExpCfgModel`；不透過 one-tone/two-tone cfg base 隱藏欄位。
- `ExpCfgModel`（`experiment/cfg_model.py`）提供共用欄位（目前含 `dev`）與統一驗證行為。
- `SweepCfg` 已移至 `program/v2/sweep.py`（從 `zcu_tools.program.v2` import），繼承 `ConfigBase`，並帶有 `@model_validator` 驗證 `start/stop/step/expts` 一致性。
- **run-time cfg materialization** 由 `zcu_tools.experiment.cfg_assembler` 擁有，而不是 `ModuleLibrary` store 擁有。核心 `assemble_experiment_cfg(raw_cfg, cfg_model, *, ml, device_snapshot, overrides=None)` 是 stateless function：caller 每次傳入 current `ml` 與當下 device snapshot；它負責套 overrides、注入 `dev` snapshot、lower `modules`、format single sweep、最後 `cfg_model.model_validate()`。`make_cfg(...)` 是薄 wrapper，預設在呼叫當下讀 `GlobalDeviceManager.get_all_info()`；`ModuleLibrary.make_cfg(...)` 只作過渡 forwarding wrapper，caller migration 後刪除。

---

## Signal/Raw 處理慣例

- **`raw` 格式**：QICK 回傳的 raw 一般是 `list[ndarray]`，第一個元素 shape 為 `(nro, ..., 2)`（IQ 兩個實數）。
- **`default_raw2signal_fn`**：Schedule 路徑使用 `runner/schedule.py` 的預設轉換：`raw[0][0].dot([1, 1j])`，取第 0 個 RO channel、該 channel 的第 0 次 readout，再把 IQ 轉成 complex。
- **客製化 `raw2signal_fn`**：integrated acquire 用 `build_and_acquire(raw2signal_fn=...)` / `run_program(raw2signal_fn=...)`；decimated trace 用 `build_and_acquire_decimated(raw2signal_fn=...)` / `run_program_decimated(raw2signal_fn=...)`。`ProgramBuilder.set_raw2signal_fn(...)` 可設定同一個 builder 的預設轉換。
- **`signal2real` 函式**：每個 Exp 檔案會定義 local 的 `xxx_signal2real`（通常 `np.abs`），給 liveplot 用；analyze 階段可能換成 phase / real。
- **scalar/array 邊界**：座標轉換工具（例如 value↔flux）可接受 scalar 或 ndarray；若後續 plotting/analysis 需要 indexing、min/max 或與另一個 sweep array 對齊，呼叫端在邊界用 `np.asarray(..., dtype=...)` 正規化成 ndarray，而不是用型別宣告假設回傳一定是 array。
- **peak-picking smoothing**：ro-optimize 與 reset 這類以 SNR/map argmax 找最佳點的分析預設使用 `smooth_method="wavelet"`；`smooth_method="gaussian"` 保留為舊 Gaussian 對照。`smooth` 是通用強度：Gaussian 時是 sigma，wavelet 時是 threshold scale。ro-optimize length 的 `t0` 是 `SNR/sqrt(length + t0)` 的 duration-normalization term；`t0 > 0` 啟用短 readout bias，且較小的正值 bias 較強。`None` 與 `0.0` 都是純平滑 SNR argmax。

---

## Executor 模式（`autofluxdep` / `overnight`）

當要在外層再疊一層「sweep 多個子實驗」的場景（例如掃 flux × {freq, t1, t2echo, ...}），會用 Executor。

兩個 Executor 共用同一個基底 `MultiMeasurementExecutor`（`runner/multi_executor.py`，見 `runner/README.md`），由它提供版面排版（`make_ax_layout` / `make_plotter`）、`record_animation` 的 FFMpeg facet、`ResultTree` per-measurement plot update、measurement init/cleanup、per-measurement retry、error/stop partial result、figure/writer `try/finally` cleanup 與 `last_cfg` / `last_result` / `last_run_outcome`。子類別各自只實作 `run()` 的 cfg/env 前置與 `Schedule` outer loop。

- `FluxDepExecutor`（`autofluxdep/executor.py`）：註冊多個 runner-owned `MeasurementBundle` / `MeasurementTask`，caller 以 explicit keyword deps 提供 `soc`、`soccfg`、`ml`、`predictor`；executor 在 run 內組 `FluxDepEnv`，用 root `Schedule.scan("flux", ...)` 掃 flux，並與 `FluxoniumPredictor` 協作，於每個 flux step 更新 typed `FluxDepInfoTracker`、設定 flux device，再交由 base executor 的 batch helper 執行 measurement。
- `OvernightExecutor`（`overnight/executor.py`）：caller 以 explicit keyword deps 提供 `soc`、`soccfg`；executor 在 run 內組 `OvernightEnv`，用 root `Schedule.repeat("Iter", ...)` 在時間軸上重複 measurement batch，並以 `trigger_update(flush=True)` 強制送出 per-measurement liveplot event。

兩者的 `retry_time` 是 per-measurement、per-flux/time-step 預算；`record_animation(mp4_path)` 需要 `ffmpeg`。

executor leaf contract 由 `runner/task.py` 擁有：`Acquirer`、`TaskPlotter`、`TaskPersister`、`MeasurementBundle`、`ComposedMeasurementBundle` 與 direct-implementation `MeasurementTask`。app-local duplicated ABC 不保留；每個 leaf 取得 `ScheduleStep` 後建立 child-local buffer，再用該 step 的 `ProgramBuilder` 直接執行 QICK acquire。

---

## `experiment/utils/` 子模組重點

`comment.py` 提供兩個函式：

- **`make_comment(cfg, comment=None)`** — 把 `ConfigBase` 轉成 JSON 字串（含 cfg dump、可選文字說明、timestamp），用於實驗存檔時的 comment 欄位。
- **`parse_comment(comment)`** — 解析 `make_comment` 產生的 JSON，回傳 `(cfg_dict, comment_str, timestamp_str)`；若 JSON 解析失敗則全部回傳 `None`。

這兩個函式透過 `experiment/utils/__init__.py` 直接 export。

## `utils/` 子模組重點

- **`sweep2array(sweep_cfg, name, {"soccfg", "gen_ch", "ro_ch"})`** — 展開 `SweepCfg` 為 numpy array，已套 ZCU 量化（`round_zcu_freq/time/gain/phase`）。Exp 幾乎都靠這個產生 x 軸。
- **`round_zcu_*`** — 單點版本的量化函式；`round_sweep_dict` 同時處理整個 sweep dict。時間的量化有特殊處理：預先減去 `0.5 * one_cycle` 以匹配 QICK sweep 用 `np.trunc` 的行為。多點 sweep 的 step 若在量化後變成 0，會 fast-fail 並要求放大 span 或減少 expts，避免 GUI/agent 看到低階 `SweepCfg` 一致性錯誤。
- **`merge_result_list(list_of_results)`** — 把 `list[dict[name, ndarray]]` 遞迴轉成 `dict[name, ndarray]`（外層 list 變成最外層維度）。Executor `ResultTree.measurement_result(name)` 只對單一 measurement 呼叫它，並快取 stacked view；更新某個 measurement 時不重算 unrelated measurement。
- **`estimate_snr` / `snr_as_signal` / `snr_checker`** — SNR 估計與 early-stop：`estimate_snr` 搭配 `snr_checker` 用於曲線 early-stop；`snr_as_signal(raw, ge_axis, skew_penalty=0.0)` 從 `MomentTracker` 的 g/e IQ moments 計算 pooled-sigma separation SNR。`skew_penalty=0.0` 是純 SNR；提高 `skew_penalty` 會以連續 rational penalty 降低 projected skew 與 g/e shape mismatch 較大的候選點，供 readout optimization 與 JPA optimization 共用。

---

## Lookback & Tracker

- **`LookbackExp`**（`lookback.py`）：用 `Schedule` + `run_program_decimated(...)` 拿 RO time trace，用來判斷 `trig_offset` 是否對齊 readout pulse；因 time axis 來自 built program，先 build program 後用 `sched.register_buffer(...)` 註冊 buffer。`analyze()` 從最大點往回找第一個低於 `ratio * max` 的點當作建議 offset。**`reps` 會在 runner-owned cfg 內強制改成 1**（decimated 不支援多 rep 平均）。
- **`KMeansTracker`**（`utils/tracker/kmeans.py`）：線上維護 `(..., 2)` IQ 樣本的動態多群統計（每群 `cluster_mean` / `cluster_covariance` / `cluster_center` / `cluster_weight`），支援 leading dims 與 `share_axis` 共用群組。內部以增量統計維護每群矩，無需保留原始樣本；singleshot 家族可直接用其 cluster 統計估計 SNR。

---

## Twotone RB 測量策略

- RB 量測採「每個 seed 一個 program」的結構：在 program 內用 sweep index 遍歷 depth，而不是在 host 端對 `(seed, depth)` 重複建立程式。
- recovery gate 是累積 Clifford 的**完整 group inverse**（24 種），不是只把態送回 +Z 的 state-restoring gate：module load 時從 `CLIFFORD_GROUP` 的 6-state permutation（對 24 元 quotient group faithful）程式化生成 `CAYLEY` / `INVERSE_INDEX` 查表，順序約定 `CAYLEY[i][j]` = 先作用 C_j 再 C_i、累積寫 `acc = CAYLEY[next][acc]`，兩處註解互相錨定，改其中一邊必須同步。
- seed 對應的 random gate prefix 長度與 recovery gate id 以 `LoadValue` 從 dmem 查表；random 段用 register-driven `Repeat`；recovery 因 inverse decomposition 最多含 2 個 physical pulse，使用兩個獨立 `ComputedPulse` slot（`recovery_gate_0/1`），不足處以 `BasicGate.Id`（gain=0）補位。兩個 slot 必須共用完整 `gate_pulses` candidate list，使 recovery 段時長與 depth 無關。
- random 段 `Repeat` 固定使用 `n="rand_len"`；當 random 序列可為空時，交由 `LoadValue(values=[])` 的 no-op 行為吸收，不在 RB 層做 sentinel value 邏輯。
- rounds 交給 `acquire()` 原生流程處理，host 層只掃 seed。

---

## 外部中斷支援（cancel_flag）

新 Schedule 寫法由 `ProgramBuilder.build_and_acquire()` / `run_program(...)` 自動把 acquire-local composite `cancel_flag` 傳給 program acquire；它會觀察 `Schedule.stop` 的 external stop，但 data-driven early stop 只停止目前 program acquire，不會把 `Schedule.outcome` 改成 `stopped`。direct ProgramBuilder path 在 external stop / `KeyboardInterrupt` / acquire error 時會保留目前 buffer partial result，並把狀態寫入 `Schedule.outcome`（`completed` / `stopped` / `interrupted` / `failed`）。executor leaf 使用傳入的 `ScheduleStep`，因此 external stop 與 outer loop 共用同一個 `StopSignal`；executor retry 耗盡或中斷時回傳目前累積的 partial result，並寫入 `last_run_outcome`。外部 stop 若在 current round 未完成時被 acquire loop 觀察到，會丟棄該 partial round、保留先前 completed rounds；first round 尚未完成就 stop 時，runner 保留 NaN partial 並標記 `stopped`，不對空 rounds 平均。`failed` / `interrupted` 仍保留 partial result，但 ambient `StopSignal` 會攜帶第一個非取消錯誤 cause，讓 GUI operation policy 可在 experiment adapter 回傳後轉成 failed outcome；retry 成功會清除 transient failure cause。

- 若使用 SNR early stop，將 `snr_checker(signals_buffer[step], threshold, signal2real_fn)` 傳給 `ProgramBuilder(...).build_and_acquire(stop_condition=...)`；runner 只在 completed round 寫入 buffer 後檢查，命中時呼叫 acquire-local `cancel_flag.set()`，保留目前 round 並讓 `Schedule.outcome` 維持 `completed`。
- `singleshot/ge.py`、`singleshot/check.py` 的 raw-shot acquire path 不經 `ProgramBuilder.run_program(...)`，因此在實驗邊界直接傳入 `cancel_flag=sched.stop` 或 `cancel_flag=step.stop`，並把 first-round no-data stop 視為 stopped partial。

---

## 寫新 Experiment 時的檢查清單

1. 定義 `XxxModuleCfg` / `XxxSweepCfg`（通常繼承 `ConfigBase`）與 `XxxCfg = ProgramV2Cfg + ExpCfgModel + 自己欄位`。
2. 繼承 `PersistableExperiment[T_Result, XxxCfg]`（要有持久化）並宣告 class-level `AXES_SPEC`（`AxesSpec`）即繼承 `save` / `load`；只需自行實作 `run`（套 `@record_result`）/ `analyze`（套 `@retrieve_result`）。Result dataclass 須有 `cfg_snapshot` 欄位。
3. `run()` 模板：開頭 `orig_cfg = deepcopy(cfg)` → `sweep2array` → `with LivePlot` → 宣告 `signals_buffer = SignalBuffer(..., on_update=...)`（預設 complex dtype 省略 `dtype=`）→ `with Schedule(cfg, signals_buffer) as sched` → 用 `_ = (sched.prog_builder(...).declare_sweep(...).build_and_acquire())` 或 host `sched.scan(...)` 量測 → 回傳 `XxxResult(..., cfg_snapshot=orig_cfg)`（`@record_result` 自動寫入 `last_result`，不存 `last_cfg`）。
4. `ProgramBuilder.build_and_acquire()` / `run_program(...)` 自動注入 Schedule `cancel_flag`；若直接呼叫 `program.acquire(...)`，必須明確傳入 `cancel_flag=sched.stop` 或 `cancel_flag=step.stop`。SNR early stop 走 builder 的 `stop_condition=snr_checker(...)`。
5. 持久化由 `AXES_SPEC` 宣告：每個 `Axis` 帶 `scale`（頻率 `MHZ_TO_HZ`、時間 `US_TO_S`）讓盤上是 SI 單位、記憶體內維持習慣單位，`AXES_SPEC.tag` 取有層次的 on-disk 名字（`"twotone/rabi/len"`），axes 以 inner-first 排列；繼承的 `save` / `load` 自動依 spec 做單位轉換與恒等逆 round-trip，無需自行寫 save/load。
6. 如果有多個 sweep 軸，先區分 host loop 與 program loop：host loop 用 `sched.scan(...)` / `sched.repeat(...)` / `sched.batch(...)`，program loop 用 `ProgramBuilder.declare_sweep(...)`；batch child 必須是 replayable callable，buffer 寫入由 child 明確指定，batch 本身不做 per-child retry。host soft sweep 若需要重用每個點的 program，在 `run()` 裡維護 dict：cache miss 時 `builder.build()`，每次量測時 `builder.run_program(program)`。
7. 如果要在 Experiment 外層疊 flux / time sweep，優先讓 Executor 使用 `ResultTree` 作為 `BufferProtocol` result buffer 並傳入 root `Schedule`，外層用 `scan` / `repeat` / `batch`，leaf 用 `state.child(..., cfg=program_cfg).buffer(...)` 建立 result slot 與 program cfg scope；plot update 透過 `ResultUpdateEvent`，不要解析 `ScheduleStep.path`。
