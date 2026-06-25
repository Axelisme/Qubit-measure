# QICK Note for `experiment/v2_gui`

**Last updated:** 2026-06-25（cfg materialization boundary）

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
    ├── shared/          — CfgBuilder / build_exp_spec / make_*_module_spec / md_* sugar / role defaults
    ├── lookback / onetone / twotone / fake
    └── twotone/reset/   — reset 校準實驗群（single_tone / dual_tone / bath / check）
```

每個 adapter 實作 `cfg_spec()`（純結構 spec，classmethod）、`make_default_value(ctx)`（讀
md/ml 算預設值）、`run` / `analyze` / `get_writeback_items` / `guide`。需要 SoC-dependent
但可預測的 run-time cfg 檢查時，覆寫 `validate_run_request(req, raw_cfg)` 做純 preflight
（例如 `len_rabi` 先確認 length sweep 在 ZCU 時間格點上不會量化成 zero-step）。詳細框架契約見
`gui/app/main/README.md`。

`BaseAdapter.build_exp_cfg` 是 GUI run path 的 cfg materialization seam：adapter 先用
`schema.to_raw_dict(req.md, req.ml)` 在 GUI adapter 層完成 EvalValue / md lowering，再把
concrete raw cfg 交給 `zcu_tools.experiment.cfg_assembler.make_cfg` / `assemble_experiment_cfg`。
assembler 每次呼叫接收 request 當下的 current `ml` 與 device snapshot；不要把 active
`ml/md` 綁進長壽 service object，也不要讓 `ModuleLibrary` store 擁有 live device snapshot。

ro-optimize 與 reset 的 peak-picking adapter 以 `smooth_method` 提供 `wavelet` / `gaussian`
選擇，預設 `wavelet`；`smooth` 在 GUI 標為 smoothing strength，不再只代表 Gaussian sigma。

---

## Operator workflow guide 約束

`run-measure-gui` skill 與各 adapter guide 描述 operator 流程時，把 mock / real
量測都當黑盒：不假設已知 flux 位置，不用 simulator truth、FakeDevice 內部真值或
predictor 輸出取代掃描與看圖判讀。

標準 bring-up 順序是：`lookback` → `onetone/freq` → `onetone/flux_dep` 找
`flx_int` / period（視 2D map 判讀）→ 將 flux device 移到 `flx_int` →
在該 flux 重跑 `onetone/freq` → `twotone/freq` 寬掃 → `twotone/freq` 窄掃 →
agent / user 審核 figure 與 writeback preview → 後續 Rabi / T1 / T2。

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
