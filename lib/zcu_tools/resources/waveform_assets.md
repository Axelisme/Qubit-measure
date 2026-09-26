# `waveform_assets` 參考

**Last updated:** 2026-09-27 — moved from resources README

本檔是 [`waveform_assets.py`](waveform_assets.py) 的 owner 參考，記錄 `ArbWaveformDatabase` 的檔案格式、方法與約束。家族導覽見 [resources README](README.md)。

## `ArbWaveformDatabase`

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
- `ArbWaveform`（`program/v2/modules/waveform.py`）在建立波形時 lazy load asset，以完整 asset 時間軸（`time[-1]`）作為播放長度，依 generator 的 sample rate 用 `np.interp` 重採樣 I/Q；不截斷、不重設時間軸。

**Preview helper（ADR-0034）**：`prepare_preview_series(data, normalize: bool) -> ArbWaveformPreview` 是純 numpy domain 函式，統一計算 peak-normalize（可選）+ I/Q/Abs 三條 series。GUI dialog 與 agent PNG service 共用此 helper，而非各自重寫 normalization 算式。
