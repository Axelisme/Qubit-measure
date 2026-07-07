# `zcu_tools.experiment.v2_gui` — measure-gui adapters

**Last updated:** 2026-07-07 — Readout module writeback roles

`experiment/v2_gui/` 是 measure-gui 的**實驗領域層**：把 `experiment/v2/` 的每個 `*Exp`
包成一個 GUI adapter，供框架層 `gui/app/main/` 驅動。依賴方向 `experiment/v2_gui/` →
`gui/app/main/`（框架契約 `ExpAdapterProtocol`），反向不成立。框架層不含任何實驗領域知識。

---

## 目錄佈局

```text
experiment/v2_gui/
├── registry.py          — register_all / register_all_roles（啟動時把 adapter 與 role 填進框架 catalog）
└── adapters/
    ├── base.py          — BaseAdapter[T_Cfg, T_Result, T_AnalyzeResult, T_AnalyzeParams]（共用實作）
    ├── shared/          — CfgBuilder / build_exp_spec / make_*_module_spec / md_* sugar / ROLE_TABLE role defaults
    ├── lookback / onetone / twotone / fake
    └── twotone/reset/   — reset 校準實驗群（single_tone / dual_tone / bath / check）
```

每個 adapter 實作 `cfg_spec()`（純結構 spec，classmethod）、`make_default_value(ctx)`（讀
md/ml 算預設值）、`run` / `analyze` / `get_writeback_items`，並在同一 class 以
`guide_text: ClassVar[AdapterGuide]` 宣告 operator-facing prose；framework-facing `guide()`
入口由 `BaseAdapter` 統一提供。需要 SoC-dependent
但可預測的 run-time cfg 檢查時，覆寫 `validate_run_request(req, raw_cfg)` 做純 preflight
（例如 `len_rabi` 先確認 length sweep 在 ZCU 時間格點上不會量化成 zero-step）。詳細框架契約見
`gui/app/main/README.md`。

`BaseAdapter.build_exp_cfg` 是 GUI run path 的 cfg materialization seam：adapter 先用
`schema.to_raw_dict(req.md, req.ml)` 在 GUI adapter 層完成 EvalValue / md lowering，再把
concrete raw cfg 交給 `zcu_tools.experiment.cfg_assembler.make_cfg` / `assemble_experiment_cfg`。
assembler 每次呼叫接收 request 當下的 current `ml` 與 device snapshot；不要把 active
`ml/md` 綁進長壽 service object，也不要讓 `ModuleLibrary` store 擁有 live device snapshot。

Adapter default value 由 `CfgBuilder` 組裝。`.role(path, role)` 預設採 `Init.ADOPT`：優先引用
ModuleLibrary / WaveformLibrary 的 calibrated entry，缺項時退回 inline blank；`.role(..., Init.INLINE)`
明確要求 inline blank，不 adopt library；`.role(..., Init.DISABLED)` 只用於 optional ref，library miss
時產生 `None`（disabled）。spec 的 `optional=True` 是結構能力，`Init.DISABLED` 是 adapter default
初始化選擇，兩者需同時成立。

Adapter defaults 以 notebook bring-up seed 與目前 MetaDict 校準值共同定義：有可信 md 時保持
md-linked expression，缺校準時退回保守 notebook seed；guide prose 需描述這個 operator-facing
fallback policy，而不是臆測固定硬體值。

qubit 類 Pulse role 的 channel seed 依 setup alias fallback `qub_ch → qub_1_4_ch → qub_4_5_ch`
解析，最後才退回 `0`。Rabi 專用 drive pulse 預設採 ModuleLibrary `pi_len → pi_amp`，讓
`len_rabi` / `amp_rabi` 優先沿用已校準的 pi pulse，再由各 adapter 的 spec lock 覆寫 sweep-owned
欄位（`len_rabi` 覆寫 waveform length，`amp_rabi` 覆寫 gain）；library 缺項時才使用 blank pulse
與 notebook fallback seed。一般 state-prep pi pulse role 維持 `pi_amp → pi_len`。Pi/2 pulse role
只採 `pi2_amp → pi2_len`，不降級到 pi pulse role。

`bath_reset` role 的 `cavity_tone_cfg`、`qubit_tone_cfg`、`pi2_cfg` 都是 nested pulse refs：
前兩者預設為 custom pulse 並由 bath reset seed/md 填值，`pi2_cfg` 預設採 pi/2 pulse role
（`pi2_amp → pi2_len`）讓 reset tomography pulse 可直接引用既有校準 module。

`twotone/freq` 的 qubit-drive module 是 spectroscopy probe，不是 state-prep init pulse；
UI label 使用 `Probe Pulse`，欄位 key 仍維持 runtime/notebook contract 的 `qub_pulse`。

`build_exp_spec(extra=...)` 表示 adapter-defined top-level knobs，放在 `sweep` 與 `reps`
之間；它們可以是正式 ExpCfg 欄位，也可以是 run-only adapter 欄位。正式欄位正常 lower 到
ExpCfg；run-only 欄位由 adapter 在 `build_exp_cfg()` 或 custom `run()` 內讀取後 pop 掉。
`onetone/freq` 的 `sampling_mode` 是正式 `FreqCfg` 欄位，GUI 維持既有 `sweep.freq`
結構，選 `homophasal` 時 adapter 從 md 的 `r_f` / `rf_w` / `theta0` 注入 fit params。

跨 session 狀態的少數 default 走 `CfgBuilder.value_ref(path, key, type_name?, default?)`，透過
`ctx.values` 立即讀取 registered value source，成功後寫入普通 direct value，不把 lazy ref 放進
cfg tree。role default table 的同類入口是 `Source(key, type_name?)` seed；只用於來源多變、
強型別 helper 反而會拉寬依賴的情境。device source 只發布具名 device 資訊，不推論 flux
語義；`onetone/flux_dep` 與 `twotone/flux_dep` 的 `dev.flux_dev` 以 `device.flux.name`
為預設（也就是名為 `flux` 的 registered device），無可用來源時 fallback 到 `flux_yoko`。

Role default characterization golden 跟隨 `ROLE_TABLE` 與 `make_default_value` 產出的完整 value tree；
調整 seed 或 default-value policy 時，更新 `tests/experiment/v2_gui/adapters/shared/_role_default_golden.json`
才是契約變更的紀錄，不在 runtime builder 內保留舊 fixture 相容邏輯。

`BaseAdapter.load` 是 GUI load path 的 canonical result seam：預設建構 `exp_cls()` 並呼
`exp.load(filepath=...)`，與 `BaseAdapter.save` 的 canonical persistence 對稱。需要 constructor
參數、非 canonical/manual save、grouped data，或需要額外 metadata 才能安全分析/writeback 的
adapter 必須 override `load()` 或讓預設路徑以明確 `NotImplementedError` fast-fail。legacy
單檔案資料相容只在 adapter migration 邊界提供，adapter 只能透過 `legacy_migration_experiment` 指向白名單 converter；
流程是 canonical load 先失敗，才把原檔 read-only 轉成 `/tmp` canonical HDF5 後再呼同一個
`exp.load()`。這是 adapter 邊界，不是 experiment runtime compatibility。load
不把 `result.cfg_snapshot` 反填回 Config tab；`cfg_snapshot is None` 時 module writeback 維持
graceful skip。

`BaseAdapter` 在 class definition/import 時驗證 `AdapterCapabilities` 與 lifecycle method 是否
一致。`analysis=FIT` 必須實作 `analyze()` 且不得實作 interactive setup；`analysis=INTERACTIVE`
必須實作 `setup_interactive_analysis()` 且不得實作 `analyze()`；`analysis=NONE` 不得實作
analyze hooks。`get_analyze_params()` 只在 analyze-params **無法全 default 建構**（有欄位無 default）
時才必須覆寫；params 每欄位都有 default（含 `NoAnalyzeParams`、及把常數折進欄位 default 的 adapter）時
沿用 base default（回 `params_cls()`）。`post_analysis=True` 僅允許搭配 primary FIT analyze，並必須實作
`get_post_analyze_params()` / `post_analyze()`；post-analysis 是第二層 CPU-only 探索/比較視圖，
不更新 writeback draft，writeback 仍只由 primary `analyze()` result 經 `get_writeback_items()`
產生。

Adapter guide 是 prose，不是 machine contract。Guide prose 放在各 adapter 檔案內，避免
新增或刪除實驗時跨檔同步；adapter 以 local `guide_text` class var 提供內容，
GUI / MCP 只呼叫 `guide()`。

ro-optimize 與 reset 的 peak-picking adapter 以 `smooth_method` 提供 `wavelet` / `gaussian`
選擇，預設 `wavelet`；`smooth` 在 GUI 標為 smoothing strength，不再只代表 Gaussian sigma。
twotone `ro_optimize/length` 的 GUI analyze param 對外命名為 `duration_t0`，表示
`SNR/sqrt(length + t0)` 的 duration-normalization term，不是 penalty strength；
`None` / `0.0` 是純 SNR peak，較小的正值比大正值更偏好短 readout。

twotone `ro_optimize` adapters 的 readout spec 一律只接受 pulse readout；writeback
產生兩層 readout writeback：`best_ro_freq` / `best_ro_gain` / `best_ro_length`
仍是 MetaDict scalar；當 run result 帶有 `cfg_snapshot` 且三個 best 值可由本次
analyze result 加上 current MetaDict 補齊且皆為 finite number 時，adapter 同時提出
ModuleLibrary `readout_dpm`，並以 writeback `role_id="readout_dpm"` 標示這個
readout role proposal；缺值或 non-finite 值只略過 module writeback。
`readout_dpm` 以 `cfg_snapshot.modules.readout` 作為 template，將 readout
pulse/readout 頻率設為 `best_ro_freq`、pulse gain 設為 `best_ro_gain`、pulse
waveform length 設為 `best_ro_length + READOUT_DPM_PULSE_TAIL_US`（0.1 us tail）、
ADC readout length 設為 `best_ro_length`。

`onetone/freq` analyze 仍寫回 MetaDict `r_f` / `rf_w` / `theta0`；當 run result 帶有
`cfg_snapshot` 且 `cfg_snapshot.modules.readout` 是 pulse readout 時，adapter 也提出
ModuleLibrary `readout_rf`，並以 writeback `role_id="readout"` 標示它是 Pulse
readout role 的 proposal；`readout_rf` 是 target name，不是新的 role id。
`readout_rf` 以該 snapshot readout 作為 template，只用 fitted `r_f` 覆寫
`pulse_cfg.freq` / `ro_cfg.ro_freq`，gain、waveform length/style、channels、ADC
readout length、trigger timing 等欄位都沿用 snapshot。`cfg_snapshot is None` 或
readout 不是 pulse readout 時，module writeback graceful skip，只保留 MetaDict items。

---

## Operator workflow guide 約束

`run-measure-gui` skill 與各 adapter guide 描述 operator 流程時，把 mock / real
量測都當黑盒：不假設已知 flux 位置，不用 simulator truth、FakeDevice 內部真值或
predictor 輸出取代掃描與看圖判讀。

標準 bring-up 順序是：`lookback` → `onetone/freq` → `onetone/flux_dep` 找
`flx_int` / period（視 2D map 判讀）→ 將 flux device 移到 `flx_int` →
在該 flux 重跑 `onetone/freq` → `twotone/freq` 寬掃 → `twotone/freq` 窄掃 →
agent / user 審核 figure 與 writeback preview → 後續 Rabi / T1 / T2。

`onetone/freq` 第一次通常用 `linear` sampling 建立 `r_f` / `rf_w` / `theta0`，並在有
pulse-readout snapshot 時提出 `readout_rf`；這些 writeback 齊全後，可切到 `homophasal`
sampling，讓同一個 start/stop/expts 掃描在 resonator circle phase 上等距。

`twotone/flux_dep` 是 readout 與 `q_f` 可信後的後續 qubit model mapping，不是早期找
flux 的工具；readout/qubit 參數還沒處理好時通常看不到可用 arc。Writeback 責任留給
agent / human 判讀，guide 不暗示用自動 fidelity gate 代替判斷。

---

## Reset 校準實驗群（`adapters/twotone/reset/`）

對應 notebook `single_qubit.md` 的三種 reset 校準流程，每種一條多步 adapter 鏈。
每個校準實驗都可能提供最終 reset module；使用者各參數實驗不照順序、會重複跑。

| 流程 | 步驟 adapter | 最終 reset module |
| --- | --- | --- |
| single-tone（sideband） | `freq` → `length` | `reset_10` |
| dual-tone | `freq` → `power` → `length` | `reset_120` |
| bath（cavity-assisted） | `freq_gain` → `length` → `phase` | `reset_bath` / `reset_bath_e` |
| 共用驗證 | `check`（RabiCheck，三型共用，`analysis=NONE`） | — |

### cfg → writeback 的兩種產出

- **校準純量**：校準掃描的純量結果經 `MetaDictWriteback` 寫回 md（如 `reset_f`、
  `reset_f1/2`、`reset_gain1/2`、`bathreset_freq/gain`、`bathreset_max/min_phase`）。
- **reset module**：除了既有 md writeback，每個校準實驗在校準 md 齊時額外 emit
  `ModuleWriteback`，把校準好的 reset module 註冊回 ModuleLibrary（撞名覆寫，沿用
  `register_module` 語意）。

> 領域層只產 `MetaDictWriteback` / `ModuleWriteback` 兩種 item，**不產**
> `WaveformWriteback`（`adapters/` 零使用——備查）。波形寫回沒有校準 adapter 需要。

### reset module 組裝流程（gated per-experiment）

機制由 `adapters/shared/writeback_helpers.py` 的 `reset_module_writeback_items` 統一：
每個校準 adapter 在 `get_writeback_items` 內呼它，傳入該 reset 型別的 `field_md_map`
（dotted 欄位 path ↔ md key）與 `target`/`desc`。

- **gate**：該 module 需要的 md key **全部齊**（`md_has_key`）且 `cfg_snapshot` 非 None
  才提供；否則回 `[]`（只剩既有 md item）。
- **組裝**：齊了就以**這次** `run_result.cfg_snapshot.modules.tested_reset` 為模板，經
  `module_cfg_to_value` 建成 `(spec, value)`，把每個校準欄位**從 md 覆寫**
  （`value.with_field(path, float(md[key]))`），包成
  `ModuleWriteback(edit_schema=CfgSchema(spec, value))`。

各型 `field_md_map`：

- **single-tone**（`freq` / `length`）→ `reset_10`：`pulse_cfg.freq ← reset_f`。
- **dual-tone**（`freq` / `power` / `length`）→ `reset_120`：`pulse1/2_cfg.freq ←
  reset_f1/2`、`pulse1/2_cfg.gain ← reset_gain1/2`（共用常數
  `dual_tone/_shared.py::RESET_120_FIELD_MD_MAP`）。
- **bath**（`freq_gain` / `length` / `phase`）→ **兩個 variant**（共用
  `bath/_shared.py::bath_reset_writeback_items`）：`cavity_tone_cfg.freq/gain ←
  bathreset_freq/gain` + `pi2_cfg.phase ← bathreset_max_phase`（`reset_bath`，reset
  到 ground）/ `bathreset_min_phase`（`reset_bath_e`，reset 到 excited）；兩 variant
  各自獨立 gate。
- **`check`** 不提供（驗證步，`tested_reset` 是 ref 非校準型）。

被掃描的欄位（如 single freq 的 `pulse_cfg.freq`、dual power 的 gain）在 cfg 是
`lock_literal 0.0`，snapshot 取到 scalar，md 覆寫即可——所以順序無關、可重跑，不依賴
「最後一步」或中間步驟的 md-link 攜帶。

### 設計決策

- **gated per-experiment**：reset module 提供條件是「校準 md 齊」而非「跑到最後一步」，
  配合使用者亂序、重複跑各校準實驗的工作流；同一校準齊時不論跑哪個實驗都提供。
- **D3**：bath `freq_gain` 透過單路徑 GUI save pipeline 寫入 single-role 3D HDF5；
  tomography phase 是同一 Result 的內部 sweep axis，不再拆成四個 phase-resolved sidecar
  檔，也不再 `save` fast-fail。
- **D5**：length / 部分掃描是「看曲線」型，analyze 只渲圖、不抽純量 → 無 md writeback。
- **graceful without snapshot**：`cfg_snapshot is None`（如從檔載入）時，module
  writeback 全略過，只剩既有 md item。
