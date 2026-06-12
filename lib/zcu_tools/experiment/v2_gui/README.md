# QICK Note for `experiment/v2_gui`

**Last updated:** 2026-06-12 (reset adapter group)

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
md/ml 算預設值）、`run` / `analyze` / `get_writeback_items` / `guide`。詳細框架契約見
`gui/app/main/README.md`。

---

## Reset 校準實驗群（`adapters/twotone/reset/`）

對應 notebook `single_qubit.md` 的三種 reset 校準流程，每種一條多步 adapter 鏈，
最後一步把校準好的 reset module 註冊回 ModuleLibrary。

| 流程 | 步驟 adapter | 最終 reset module |
| --- | --- | --- |
| single-tone（sideband） | `freq` → `length` | `reset_10` |
| dual-tone | `freq` → `power` → `length` | `reset_120` |
| bath（cavity-assisted） | `freq_gain` → `length` → `phase` | `reset_bath` / `reset_bath_e` |
| 共用驗證 | `check`（RabiCheck，三型共用，`analysis=NONE`） | — |

### cfg → writeback 的兩種產出

- **中間步驟**：校準掃描的純量結果經 `MetaDictWriteback` 寫回 md（如 `reset_f`、
  `reset_f1/2`、`reset_gain1/2`、`bathreset_freq/gain`、`bathreset_max/min_phase`）。
- **最後一步**：除了既有 md writeback，額外 emit `ModuleWriteback`，把整顆校準好的
  `tested_reset` 註冊成最終 reset module（撞名覆寫，沿用 `register_module` 語意）。

### reset module 組裝流程（D2(a)）

最後一步 adapter 從 `run_result.cfg_snapshot.modules.tested_reset` 取出校準好的 reset
cfg，經 `module_cfg_to_value`（`gui/app/main/cfg_schemas.py`）建成 `(spec, value)`，包成
`ModuleWriteback(edit_schema=CfgSchema(spec, value))`：

- **single-tone `length`** → `reset_10`：`tested_reset` 已 md-link 校準 sideband freq；
  使用者在 writeback dialog 調最終 length。
- **dual-tone `length`** → `reset_120`：`tested_reset` 已 md-link 校準 freq1/2 + gain1/2。
- **bath `phase`** → 兩個 module：同一 `tested_reset`（cavity freq/gain 已 md-link）僅以
  `value.with_field("pi2_cfg.phase", …)` 覆寫單一欄位 —— `reset_bath` 用 max_phase（reset
  到 ground）、`reset_bath_e` 用 min_phase（reset 到 excited）。

校準參數靠中間步驟的 md-link（`make_default_value` 把 `tested_reset` 的固定欄位接到對應 md
key），所以 snapshot 天生攜帶校準值，最後一步只需整顆註冊（必要時覆寫單欄位）。

### 設計決策

- **D2(a)**：最後一步從 `cfg_snapshot` 整顆 `tested_reset` 註冊（而非逐欄位重建），
  與 rabi `amp_rabi` / `len_rabi` 的 `ModuleWriteback` 機制一致。
- **D3**：bath `freq_gain` 是 2D（四個 phase-resolved 檔），單路徑 GUI save pipeline
  無法表示，故 `save` fast-fail（`NotImplementedError`）。
- **D5**：length / 部分掃描是「看曲線」型，analyze 只渲圖、不抽純量 → 無 md writeback。
- **graceful without snapshot**：`cfg_snapshot is None`（如從檔載入）時，module
  writeback 全略過，只剩既有 md item。
