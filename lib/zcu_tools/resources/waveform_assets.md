# `waveform_assets` 參考

**Last updated:** 2026-10-01 — formula writes separate from inspection

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
| `load(data_key)` / `get(name)` | `load` 取得 `ArbWaveformData`；`get` 給 program 層使用，回傳 `(idata, qdata, time)`，Q channel 全為 0 時 `qdata` 為 `None` |
| `save(name, idata, time, qdata=None)` | Notebook 用的 raw 寫入；一律覆寫同名 asset，不寫 recipe（`qdata` 省略時存成全 0）。需要 collision 檢查時改用 `import_data(..., overwrite=False)` |
| `create_from_formula(data_key, recipe, *, overwrite=False)` / `update_formula(data_key, recipe)` | 渲染 formula recipe，並把 arrays 與 recipe 寫進同一個 `.npz`。`create_from_formula` 在 key 已存在且 `overwrite=False` 時拋 `data_key_exists`；`update_formula` 只更新已存在的 asset，不存在時拋 not found。建立新 recipe 或依 recipe 重新渲染，只能經這兩個方法；`import_file` 例外，會原樣保存來源 `.npz` 已有的 recipe |
| `import_file(data_key, source_path, *, overwrite=False)` / `import_data(data_key, *, idata, qdata, time, overwrite=False)` | `import_file` 只接受 `.npz`，原檔若含 recipe 會一併保留；`import_data` 接受記憶體中的三條 1D array，不寫 recipe。兩者在 key 已存在且 `overwrite=False` 時拋 `data_key_exists` |
| `delete(...)` / `rename(...)` | 只操作 asset 檔案，不掃描 `ModuleLibrary` references |

Formula 寫入方法 `create_from_formula`／`update_formula` 成功返回 `None`，不在寫檔後
重新載入資產或產生預覽。需要 metadata 時，caller 另呼叫 `inspect`；該讀取失敗不
撤銷已成功的寫入。`import_data`／`import_file` 的既有回傳契約不在這次變更範圍。

**約束**：

- `.npz` 必須只有 `idata`、`qdata`、`time`，以及可選的 `recipe_json`；folder layout、`.npy` 與 `.csv` 不屬於支援格式。
- `.npz` 寫入路徑固定用明確 keyword (`idata`, `qdata`, `time`, optional `recipe_json`) 呼叫 `np.savez`；不要用 dynamic payload `**dict` 讓 `allow_pickle` overload 判斷變模糊。
- `idata`、`qdata`、`time` 必須是一維、同長度、finite array；`time[0] == 0` 且嚴格遞增，單位固定為 us。
- `Abs = hypot(I, Q)` 必須落在 `[0, 1]`；I/Q 可為負值。
- formula recipe 是可選資料；用 recipe 生成等於完全覆寫原本 data，並把 recipe 一起寫入同一個 `.npz`。
- `import_data`、`import_file` 與 `create_from_formula` 預設在 key collision 時拒絕；caller 明示 `overwrite=True` 才可覆寫。`rename` 永不覆蓋新 key；`update_formula` 覆寫既有 playback arrays 與 recipe。`save` 是固定覆寫的 raw asset 便利入口。
- `rename`／`delete` 後，舊 key 的 ModuleLibrary 引用可能失效；repository 不修改引用，使用端在載入缺失 asset 時報錯。
- `ArbWaveform`（`program/v2/modules/waveform.py`）在建立波形時 lazy load asset，以完整 asset 時間軸（`time[-1]`）作為播放長度，依 generator 的 sample rate 用 `np.interp` 重採樣 I/Q；不截斷、不重設時間軸。

**Preview helper**：`prepare_preview_series(data, normalize: bool) -> ArbWaveformPreview` 是純 numpy domain 函式，統一計算 peak-normalize（可選）+ I/Q/Abs 三條 series。GUI dialog 與 agent PNG service 共用此 helper，而非各自重寫 normalization 算式。

## 資產與 recipe 約束

Data key 符合 `^[A-Za-z][A-Za-z0-9_]*$`，單一檔案為 `<data_key>.npz`。
載入拒絕未知 key；若有 `recipe_json` 卻無法解析則直接失敗，不退回 raw asset。
三條 playback array 同長且長度介於 2 與 `MAX_ARB_WAVEFORM_SAMPLES = 1_000_001`；
匯入 raw `.npz` 的 time 軸可不等距，但必須從 0 開始並嚴格遞增。
`inspect` 即時計算衍生摘要，不另存 duration、sample count 或 `peak_abs`。

Formula recipe 只保存 `segments` 與 `normalize`。每段有正的 `duration` 和非空的
`formula`；解析時去除前後空白，保存時保留原字串。每段一條 SymPy-compatible expression，
實數輸出進 I，複數輸出分配到 I/Q。`t` 是段內時間，`T` 是全長時間，單位都是 us。
總長是各段 duration 的和，增刪段不會重新分配其他段的長度。除最後一段含終點外，
每段使用半開區間，內部交界點交給下一段。條件式 `Piecewise` 不屬於支援的 v1 expression。

渲染以 `round(total_duration * 1000) + 1` 個點覆蓋完整時間軸；超過樣本上限會拒絕。
`normalize` 必填，只接受 `none`、`peak`。`none` 對超出單位幅度的輸出 fast-fail；
`peak` 以全長 `max(hypot(I, Q))` 同時縮放 I/Q，正 peak 即使小於 1 也放大至 1，
零波形保持零。播放讀取保存的 arrays，不依 recipe 在播放時重新渲染。
資產 key 可被 ModuleLibrary 引用；rename/delete 不掃描也不改寫這些引用，失效 key 由使用端處理。

GUI 建立新 recipe 的隱藏預設是 `peak`，編輯已有 recipe 時保留原 `normalize`。
GUI validation 失敗時禁用 Save；agent-facing validation 提供具名 reason。
這些是操作端的呈現政策，repository 仍負責共同驗證與渲染。
