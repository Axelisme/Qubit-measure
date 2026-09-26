# `zcu_tools.resources` — experiment work resources

**Last updated:** 2026-09-27 — resource owner split

`resources/` 組織實驗工作資源的命名、讀寫與交換。它不是通用 storage framework，也不包含 datafile。

- [`context/`](context/README.md) 管理具名工作 scope 內的 MetaDict 與 ModuleLibrary。
- [`qubit_params.py`](qubit_params.py) 管理 result scope 的 typed `params.json` handoff。
- [`sample_table/`](sample_table/README.md) 管理樣品量測 CSV 與 opt-in v2 欄位協定。
- [`waveform_assets.py`](waveform_assets.py) 管理 qubit-scoped `.npz` arbitrary waveform asset。
- [`syncfile.py`](syncfile.py) 提供持久化物件共用的 mtime 同步機制。

---

## `ArbWaveformDatabase`（`waveform_assets.py`）

以單一 `.npz` 檔儲存任意波形資料的類別方法資料庫（Class-level singleton path）。GUI 與 notebook 共用這個 repository，避免 formula recipe、import、preview 與 `ArbWaveform` 使用端各自長出不同規則。

```python
ArbWaveformDatabase.init("data/arb_waveforms/")
ArbWaveformDatabase.create_from_formula(
    "my_pulse",
    {
        "segments": [{"duration": 1.0, "formula": "sin(2*pi*t)"}],
        "normalize": "peak",
    },
    overwrite=False,
)
data = ArbWaveformDatabase.load("my_pulse")
```

**檔案結構**：

```
<database_path>/
    my_pulse.npz   # idata, qdata, time, optional recipe_json
```

**主要方法**：

| 方法 | 說明 |
|------|------|
| `list()` | 只列出 sorted data keys，不開 `.npz` |
| `list_entries()` | 列出 data key 加檔案 `mtime` / `file_size` |
| `inspect(data_key)` | 載入單筆 asset 並即時計算 duration、sample count、peak Abs、recipe summary |
| `load(data_key)` / `get(data_key)` | 取得 `ArbWaveformData` 或舊 notebook 常用的 `(idata, qdata, time)` |
| `save(data_key, idata, time, qdata=None, recipe=None)` | 寫入 raw data；`recipe` 可省略 |
| `create_from_formula(...)` / `update_formula(...)` | 用 formula recipe 重新渲染並覆寫資料 |
| `import_file(...)` / `import_data(...)` | 只接受 `.npz` 或已在記憶體中的三條 1D array |
| `delete(...)` / `rename(...)` | 只操作 asset 檔案，不掃描 `ModuleLibrary` references |

**約束**：

- `.npz` 必須只有 `idata`、`qdata`、`time`，以及可選的 `recipe_json`；folder layout、`.npy` 與 `.csv` 不屬於支援格式。
- `.npz` 寫入路徑固定用明確 keyword (`idata`, `qdata`, `time`, optional `recipe_json`) 呼叫 `np.savez`；不要用 dynamic payload `**dict` 讓 `allow_pickle` overload 判斷變模糊。
- `idata`、`qdata`、`time` 必須是一維、同長度、finite array；`time[0] == 0` 且嚴格遞增，單位固定為 us。
- `Abs = hypot(I, Q)` 必須落在 `[0, 1]`；I/Q 可為負值。
- formula recipe 是可選資料；用 recipe 生成等於完全覆寫原本 data，並把 recipe 一起寫入同一個 `.npz`。
- 被 `ArbWaveform`（`modules/waveform.py`）在建立波形時 lazy load；若 requested sample count 小於 asset data 長度，使用端只截斷並寫 logger warning，不做縮放。

**Preview helper（ADR-0034）**：`prepare_preview_series(data, normalize: bool) -> ArbWaveformPreview` 是純 numpy domain 函式，統一計算 peak-normalize（可選）+ I/Q/Abs 三條 series。GUI dialog 與 agent PNG service 共用此 helper，而非各自重寫 normalization 算式。

---

## `SyncFile`（`syncfile.py`）

所有持久化物件的基礎類別，實作 **mtime-based 雙向同步**。

```
_path         ─── 對應的磁碟路徑（None 表示純記憶體模式）
_modify_time  ─── 上次讀/寫時的 mtime（nanoseconds）
_dirty        ─── 記憶體資料有未寫回磁碟的修改
_readonly     ─── 禁止任何寫回操作
```

`has_persistence` 是判斷物件是否綁定磁碟路徑的 public API；跨模組呼叫者不應讀取 `_path`。

**`sync()` 邏輯**（每次讀/寫操作前自動觸發）：

```
if _path is None → return（純記憶體，無需同步）
if file exists:
    if _dirty and not _readonly → dump()（寫回，因為記憶體更新）
    elif mtime >= _modify_time  → load()（重新載入，因為磁碟更新）
else:
    if not _readonly → dump()（檔案不存在則建立）
```

**注意**：sync 採「記憶體優先」策略——若 `_dirty=True`，即使磁碟檔案同時被更新也會用記憶體版本覆蓋。

**`auto_sync` 裝飾器**（`syncfile.py`）：

```python
@auto_sync("read")   # 進入方法前 sync()
@auto_sync("write")  # 進入前 sync()，退出後再 sync()（確保寫回）
```

裝飾器只接受 `SyncFile` instance method；若第一個參數不是 `SyncFile`，會直接 `TypeError`，避免錯誤 receiver 被 warning 後繼續執行到不明確的 attribute failure。

## 跨模組依賴

```
ContextManager
    ├── ModuleLibrary  (module_cfg.yaml)
    │       └── ModuleCfg / WaveformCfg  (from program/v2/modules)
    └── MetaDict       (meta_info.json)

Experiment cfg materialization
    ├── zcu_tools.experiment.cfg_assembler.make_cfg / assemble_experiment_cfg
    ├── ModuleLibrary  (current ml；module lookup / lowering context)
    └── GlobalDeviceManager.get_all_info()  (thin wrapper 呼叫當下的 default snapshot provider)

QubitParams          (params.json typed owner；fluxdep/dispersive/t1_curve/predictor caller 共用)
ArbWaveformDatabase  (獨立，被 modules/waveform.py:ArbWaveform 使用)
SampleTable          (獨立，供 notebook 記錄量測結果用；v2 協定由 `sample_table/schema.py` 提供，不改變 SampleTable 本身)
```

## 注意事項

- `SyncFile.sync()` 採 mtime 比較，且目前沒有 file lock。若兩個 process 同時寫同一個檔案，會 warning 衝突並以本地 dirty 內容覆蓋磁碟版本。
- `MetaDict` 的 `_dirty` + `sync()` 表示每次單一 `__setattr__` 都會立即寫回磁碟；多 key 寫入用 `update()` 共用同一次 write transaction。
- `ModuleLibrary.get_waveform()` / `get_module()` 回傳 deepcopy，修改回傳值不影響 library 內部狀態（需透過 `register_*` / `update_*` 才能持久化）。
- experiment cfg materialization 每次呼叫都使用 caller 傳入的 current `ml` 與 device snapshot；active context 切換不應被長壽 object 綁住。
- `QubitParams.set_dispersive_fit()` 會要求 `params.json` 已有 `fluxdep_fit`；dispersive export 不能建立沒有 fluxdep handoff 的半成品檔案。
- `QubitParams.set_t1_curve_fit()` 會要求 `params.json` 已有 `fluxdep_fit`；T1 curve fit section 是 downstream handoff，不建立沒有 fluxonium fit 的半成品。
- `t1_curve_fit.params` 以 white-list 表達 active noise channels：`Temp` 必填，`Q_cap` / `x_qp` / `Q_ind` 可省略；`fixed`、`free`、`bounds`、`init` 與 `stderr` 只能提到 active params。
- `QubitParams.set_fluxdep_fit()` 不會刪除 `dispersive` 或 `t1_curve_fit`；這些 section 以各自的 `timestamp` 表示最後修改時間。
