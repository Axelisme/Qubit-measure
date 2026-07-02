# `zcu_tools.meta_tool` — persistent experiment metadata

**Last updated:** 2026-07-02 — QubitParams t1_curve_fit white-list channels

這份筆記整理 `meta_tool/` 的設計，說明各類別的職責、同步機制與使用模式。

---

## 架構總覽（一句話版）

`meta_tool/` 提供實驗的**持久化設定管理**：`SyncFile` 是自動讀寫同步的抽象基礎，`MetaDict` 以 JSON 儲存任意實驗參數、`ModuleLibrary` 以 YAML 管理波形與模組設定，`ExperimentManager` 把兩者綁定到以 flux 值命名的資料夾上下文，`QubitParams` 擁有每個 result scope 的 `params.json` typed handoff file，`SampleTable` 儲存樣品量測紀錄，`ArbWaveformDatabase` 管理 qubit-scoped arbitrary waveform `.npz` asset。

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

---

## `MetaDict`（`metadict.py`）

以 JSON 儲存實驗參數的 dict-like 物件，所有欄位可透過屬性語法存取。

```python
md = MetaDict("experiment/meta_info.json")
md.qubit_freq = 5.0e9     # 自動 sync + 寫回
print(md.qubit_freq)      # 自動 sync + 讀取
```

**型別特殊處理**：

- 寫入時呼叫 `format_obj()`（utils），將 complex 等非 JSON 原生型別轉成字串。
- 載入時呼叫 `_restore_complex()`，將字串還原為 `complex`（如 `"(1+2j)"` → `1+2j`）。

**受保護的屬性**（不進入 `_data`）：`_` 開頭的名稱，以及 `["dump", "load", "sync", "update_modify_time", "clone", "items", "keys", "get"]`。

**`clone(dst_path, readonly)`**：複製整個 MetaDict 到新路徑（要求目標不存在）。

---

## `QubitParams`（`params.py`）

`QubitParams` 是 `result/<chip>/<qub>/params.json` 的唯一 typed 讀寫 module。它不是任意 key/value store；caller 透過語意方法讀寫 project identity、`fluxdep_fit`、`dispersive`、`t1_curve_fit` 與 predictor 所需的 fluxonium model。

**主要責任**：

| 方法 | 說明 |
|------|------|
| `ensure_project(project)` | 建立或更新 canonical `project.{chip_name, qubit_name, resonator_name}`，並同步 legacy `name` |
| `migrate_project_from_path(result_root=...)` | 將 v0 identity 原地升級；canonical `project` 優先，缺 project 時才由 `result/` 路徑推導 |
| `set_fluxdep_fit(fit)` | 寫入 fluxdep fit；保留獨立的 `dispersive` section，並更新 `fluxdep_fit.timestamp` |
| `require_dispersive_inputs(default_bare_rf=...)` | 讀出 dispersive GUI 的硬輸入，並集中 `bare_rf` seed 優先序 |
| `set_dispersive_fit(fit)` | 寫入 dispersive fit；要求檔案已存在且已有 `fluxdep_fit`，並更新 `dispersive.timestamp` |
| `set_t1_curve_fit(fit)` | 寫入 T1 curve noise fit handoff；要求檔案已存在且已有 `fluxdep_fit`，並更新 `t1_curve_fit.timestamp` |
| `require_t1_curve_fit()` | 讀出 T1 curve fit 的 noise params、stderr 與 fit metadata |
| `require_fluxonium_model(flux_bias=...)` | 讀出 `FluxoniumPredictor` / sim 需要的 `(EJ, EC, EL, flux_half, flux_period, flux_bias)` |

**獨立 section 與 timestamp**：`fluxdep_fit`、`dispersive` 與 `t1_curve_fit` 是獨立 module section；寫入其中一個不會刪除另一個。每次 typed 寫入會更新該 section 的 `timestamp`，供 caller 判斷最後修改時間。`t1_curve_fit` 只保存後續模擬需要的 fit params 與 metadata；sample arrays 和 dense model curves 不放進 `params.json`。`t1_curve_fit.params` 中 `Temp` 必填，`Q_cap` / `x_qp` / `Q_ind` 只保存 active noise channel；省略某個 noise key 表示該 channel 未納入 all-in-one fit。

**未知 section preservation**：typed 寫入只更新自己的 section，其它未知 section 會保留。`to_raw()` / `replace_raw()` / `update_raw()` 只供 `notebook.persistance` 舊 helper 過渡使用；新 caller 應使用 typed 方法。

---

## `ModuleLibrary`（`library.py`）

以 YAML 儲存波形（`WaveformCfg`）與模組（`ModuleCfg`）設定的管理器。

```yaml
# module_cfg.yaml
waveforms:
  pi_pulse:
    style: gauss
    length: 0.05
    sigma: 0.01
modules:
  readout:
    type: readout/pulse
    pulse_cfg: ...
    ro_cfg: ...
```

**主要方法**：

| 方法 | 說明 |
|------|------|
| `get_waveform(name, override_cfg, type)` | 取得波形設定（deepcopy），可 override 特定欄位 |
| `get_module(name, override_cfg, type)` | 取得模組設定（deepcopy），可 override 特定欄位 |
| `register_waveform(**wav_kwargs)` | 新增/覆蓋波形設定並寫回 |
| `register_module(**mod_kwargs)` | 新增/覆蓋模組設定並寫回 |
| `update_module(name, override_cfg)` | 部分更新既有模組設定 |
| `make_cfg(exp_cfg, cfg_model, **kwargs)` | thin wrapper；轉呼 `zcu_tools.experiment.cfg_assembler.make_cfg(..., ml=self, ...)` |

**Experiment cfg materialization 邊界**：

`ModuleLibrary` 是 YAML-backed store：它擁有 waveform/module 的持久化、lookup、register/update/delete 與 mtime sync，不擁有 live device snapshot，也不擁有 experiment cfg materialization 流程。

把 concrete raw experiment cfg 轉成 typed `ExpCfgModel` 的核心邏輯位於 `zcu_tools.experiment.cfg_assembler`：

1. `assemble_experiment_cfg(raw_cfg, cfg_model, *, ml, device_snapshot, overrides=None)` 是 stateless materializer。
2. `raw_cfg` 已是 concrete dict；GUI 的 `CfgSchema` / `EvalValue` / md lowering 在 adapter 層完成，不進 assembler。
3. caller 在每次 run / notebook call 當下傳入 current `ml` 與 `device_snapshot`。核心不持有 active context，也不直接讀 `GlobalDeviceManager`。
4. `make_cfg(raw_cfg, cfg_model, *, ml, overrides=None, device_snapshot=None)` 是薄 wrapper；若未傳 `device_snapshot`，在呼叫當下讀 `GlobalDeviceManager.get_all_info()`，再轉呼 `assemble_experiment_cfg`。
5. `ModuleLibrary.make_cfg(...)` 是 forwarding wrapper，避免形成第二套 materialization implementation；新呼叫點優先使用 `zcu_tools.experiment.cfg_assembler.make_cfg(...)`。

**Cfg 解析 API**（統一走 Factory wrapper）：

```python
# library.py 內 store 解析點（_load / register_*）統一使用：
WaveformCfgFactory.from_raw(raw, ml=self)
ModuleCfgFactory.from_raw(raw, ml=self)
```

兩個 Factory 都是薄封裝：`from_raw(raw, ml=...)` 內部直接呼叫 `TypeAdapter(...).validate_python(..., context={"ml": ml})`。`ModuleCfg` / `WaveformCfg` 的分派表由各自 TypeAlias 的 `Union[...]` 決定，不再使用 runtime registry。

**為什麼叫 `from_raw` 而不是 `validate`？** Pydantic v2 的 `BaseModel` 已有 deprecated 的 `validate()` classmethod；如果我們在 cfg 上覆寫同名方法會被 pyright 標記為 deprecated。`from_raw` 既避免衝突，語意也更清楚（從「raw 任意輸入」轉成具體 typed cfg）。

**`ModuleDumper`**（自訂 YAML Dumper）：

- 縮排層級減少時插入空白行（增加可讀性）。
- dict 的 value 依型別排序：str > int > float > bool > other > list > dict（避免 nested dict 佔用視覺空間）。

---

## `ExperimentManager`（`manager.py`）

將 `ModuleLibrary` + `MetaDict` 綁定到 **flux 上下文**（以電流/電壓命名的資料夾）。

```
exp_dir/
    0113_10_1.234mA/
        module_cfg.yaml
        meta_info.json
    0113_11_5.678mA/
        module_cfg.yaml
        meta_info.json
```

**主要方法**：

| 方法 | 說明 |
|------|------|
| `list_contexts()` | 回傳所有已存在的 flux 上下文標籤（排序後）|
| `new_flux(value, clone_from, label, unit)` | 建立新 flux 上下文（資料夾不能已存在）|
| `use_flux(label, readonly)` | 載入既有 flux 上下文 |
| `label` | 目前啟用的上下文標籤（property）|
| `flux_dir` | 目前上下文資料夾的 Path |

**`new_flux()` 的 `clone_from` 參數**：

- `str` — 從同 exp_dir 下的另一個 flux 標籤複製設定。
- `(ModuleLibrary, MetaDict)` tuple — 直接從記憶體中的物件複製。
- `None` — 建立空的設定（新量測點從零開始）。

**`new_flux()` 的 `unit` 參數**：`Literal["A", "V", "K", "none"]`，預設 `"none"`。

**自動標籤（`_auto_label()`）**：

- 格式：`MMDDH[_<value>]`（例如 `0113_10_1.234mA`，無 value 時僅日期時段）。
- `unit="none"` 時 value 以 fixed-point + SI 前綴格式化（例如 `1.230m`），使用 u/m/空/k/M/G/T 前綴，不附加物理單位字串。
- 若標籤已存在，自動附加 `_2`、`_3`...。
- 單位換算：超過閾值時自動選擇合適的前綴（mA/A、mV/V、mK/K）。

---

## `SampleTable`（`table.py`）

以 CSV 儲存量測樣品紀錄（pandas DataFrame 包裝）。

```python
st = SampleTable("samples.csv")
st.add_sample(qubit="Q1", flux=1.23e-3, T1=50e-6)
df = st.get_samples()
```

**方法**：`add_sample(**kwargs)`、`extend_samples(**kwargs)`（批次）、`update_sample(idx, **kwargs)`、`get_samples()`。

---

## `ArbWaveformDatabase`（`arb_waveform.py`）

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

## 使用模式（典型 notebook 開頭）

```python
from zcu_tools.meta_tool import ExperimentManager

em = ExperimentManager("Database/chipA/qubitQ1")

# 新測量點
ml, md = em.new_flux(value=1.23e-3, unit="A")
md.qubit_freq = 5.0e9
ml.register_module(readout={...})

# 或載入既有測量點
ml, md = em.use_flux("0113_10_1.234mA")
from zcu_tools.experiment.cfg_assembler import make_cfg

cfg = make_cfg(
    {"reps": 1000, "rounds": 10},
    SomeExpCfgModel,  # SomeExpCfgModel = 你的 ExpCfgModel 子類
    ml=ml,
)

from zcu_tools.meta_tool import FluxDepFit, ParamsProject, QubitParams

params = QubitParams.for_result_dir("result/chipA/qubitQ1")
params.ensure_project(ParamsProject("chipA", "qubitQ1"))
params.set_fluxdep_fit(
    FluxDepFit(
        EJ=4.0,
        EC=1.0,
        EL=0.5,
        flux_half=0.5,
        flux_int=1.0,
        flux_period=2.0,
    )
)
```

---

## 跨模組依賴

```
ExperimentManager
    ├── ModuleLibrary  (module_cfg.yaml)
    │       └── ModuleCfg / WaveformCfg  (from program/v2/modules)
    └── MetaDict       (meta_info.json)

Experiment cfg materialization
    ├── zcu_tools.experiment.cfg_assembler.make_cfg / assemble_experiment_cfg
    ├── ModuleLibrary  (current ml；module lookup / lowering context)
    └── GlobalDeviceManager.get_all_info()  (thin wrapper 呼叫當下的 default snapshot provider)

QubitParams          (params.json typed owner；fluxdep/dispersive/t1_curve/predictor caller 共用)
ArbWaveformDatabase  (獨立，被 modules/waveform.py:ArbWaveform 使用)
SampleTable          (獨立，供 notebook 記錄量測結果用)
```

---

## 注意事項

- `SyncFile.sync()` 採 mtime 比較，且目前沒有 file lock。若兩個 process 同時寫同一個檔案，會 warning 衝突並以本地 dirty 內容覆蓋磁碟版本。
- `MetaDict` 的 `_dirty` + `sync()` 表示每次 `__setattr__` 都會立即寫回磁碟（兩次 `sync()`：設定前確保最新，設定後立即寫回）。
- `ModuleLibrary.get_waveform()` / `get_module()` 回傳 deepcopy，修改回傳值不影響 library 內部狀態（需透過 `register_*` / `update_*` 才能持久化）。
- experiment cfg materialization 每次呼叫都使用 caller 傳入的 current `ml` 與 device snapshot；active context 切換不應被長壽 object 綁住。
- `QubitParams.set_dispersive_fit()` 會要求 `params.json` 已有 `fluxdep_fit`；dispersive export 不能建立沒有 fluxdep handoff 的半成品檔案。
- `QubitParams.set_t1_curve_fit()` 會要求 `params.json` 已有 `fluxdep_fit`；T1 curve fit section 是 downstream handoff，不建立沒有 fluxonium fit 的半成品。
- `t1_curve_fit.params` 以 white-list 表達 active noise channels：`Temp` 必填，`Q_cap` / `x_qp` / `Q_ind` 可省略；`fixed`、`free`、`bounds`、`init` 與 `stderr` 只能提到 active params。
- `QubitParams.set_fluxdep_fit()` 不會刪除 `dispersive` 或 `t1_curve_fit`；這些 section 以各自的 `timestamp` 表示最後修改時間。
