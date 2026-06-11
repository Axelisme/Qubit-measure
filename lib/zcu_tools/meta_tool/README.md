# QICK Note for `meta_tool`

**Last updated:** 2026-06-08

這份筆記整理 `meta_tool/` 的設計，說明各類別的職責、同步機制與使用模式。

---

## 架構總覽（一句話版）

`meta_tool/` 提供實驗的**持久化設定管理**：`SyncFile` 是自動讀寫同步的抽象基礎，`MetaDict` 以 JSON 儲存任意實驗參數、`ModuleLibrary` 以 YAML 管理波形與模組設定，`ExperimentManager` 把兩者綁定到以 flux 值命名的資料夾上下文，`SampleTable` 儲存樣品量測紀錄，`ArbWaveformDatabase` 管理任意波形 npz 資料。

---

## `SyncFile`（`syncfile.py`）

所有持久化物件的基礎類別，實作 **mtime-based 雙向同步**。

```
_path         ─── 對應的磁碟路徑（None 表示純記憶體模式）
_modify_time  ─── 上次讀/寫時的 mtime（nanoseconds）
_dirty        ─── 記憶體資料已修改、尚未寫回磁碟
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
| `make_cfg(exp_cfg, cfg_model, **kwargs)` | 合併 exp_cfg + kwargs + GlobalDeviceManager 設備資訊，並解析 modules 欄位 |

**`make_cfg()` 詳細流程**（簽名：`make_cfg(exp_cfg, cfg_model, **kwargs) -> T_ExpCfg`）：

1. `deepcopy(exp_cfg)` 後用 kwargs 強制覆蓋。
2. 讀取 `GlobalDeviceManager.get_all_info()` 取得當前設備資訊，以 exp_cfg 的 `dev` 欄位 override。
3. 將 `modules` 欄位中的字串/dict 解析為 `ModuleCfg` 物件：字串 → `self.get_module(name)`；dict → `ModuleCfgFactory.from_raw(d, ml=self)`（discriminated union 自動 dispatch 到具體 leaf）。
4. 若 `cfg_model` 有唯一 sweep 子欄位，則自動呼叫 `format_sweep1D()` 將 `sweep` 欄位規格化（縮寫格式 → `{name: SweepCfg}` 格式）。
5. 以 `cfg_model.model_validate(exp_cfg)` 建立並回傳型別安全的 config 物件（`T_ExpCfg`，bound to `ExpCfgModel`）。

**Cfg 解析 API**（統一走 Factory wrapper）：

```python
# library.py 內所有解析點（_load / register_* / make_cfg）統一使用：
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

以 npz 格式儲存任意波形資料的類別方法資料庫（Class-level singleton path）。

```python
ArbWaveformDatabase.init("data/arb_waveforms/")
ArbWaveformDatabase.save("my_pulse", idata, time, qdata=None)
idata, qdata, time = ArbWaveformDatabase.get("my_pulse")
```

**資料夾結構**：

```
<database_path>/
    <waveform_name>/
        data.npz   # 包含 idata, qdata, time
        example.png
```

**約束**：

- `idata`/`qdata` 的最大絕對值必須 ≤ 1（正規化振幅）。
- `time` 必須單調遞增（微秒單位）。
- 被 `ArbWaveform`（`modules/waveform.py`）在建立波形時 lazy load（避免循環 import）。

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
cfg = ml.make_cfg({"reps": 1000, "rounds": 10}, SomeExpCfgModel)  # SomeExpCfgModel = 你的 ExpCfgModel 子類
```

---

## 跨模組依賴

```
ExperimentManager
    ├── ModuleLibrary  (module_cfg.yaml)
    │       └── ModuleCfg / WaveformCfg  (from program/v2/modules)
    │               └── GlobalDeviceManager.get_all_info()  (make_cfg 時注入)
    └── MetaDict       (meta_info.json)

ArbWaveformDatabase  (獨立，被 modules/waveform.py:ArbWaveform 使用)
SampleTable          (獨立，供 notebook 記錄量測結果用)
```

---

## 注意事項

- `SyncFile.sync()` 採 mtime 比較，且目前沒有 file lock。若兩個 process 同時寫同一個檔案，會 warning 衝突並以本地 dirty 內容覆蓋磁碟版本。
- `MetaDict` 的 `_dirty` + `sync()` 表示每次 `__setattr__` 都會立即寫回磁碟（兩次 `sync()`：設定前確保最新，設定後立即寫回）。
- `ModuleLibrary.get_waveform()` / `get_module()` 回傳 deepcopy，修改回傳值不影響 library 內部狀態（需透過 `register_*` / `update_*` 才能持久化）。
- `make_cfg()` 中 `GlobalDeviceManager.get_all_info()` 會讀取當前所有設備的 info（含 addr、當前電流/電壓值等），注入到 `cfg["dev"]`，確保每次實驗設定包含設備快照。

---

## 更新紀錄

| 日期 | Codebase commit | 說明 |
|------|-----------------|------|
| 2026-05-21 | `d957bc8c` | `new_flux`/`_auto_label` 新增 `unit="none"` 支援：無物理單位的純 flux 數值以 fixed-point + SI 前綴格式化（u/m/空/k/M/G/T），不附加物理單位字串；`unit` 預設值從 `"A"` 改為 `"none"`。 |
| 2026-04-29 | `f2f30ae1` | 對齊現況：`ModuleCfgFactory`/`WaveformCfgFactory` 改為 TypeAdapter 薄封裝，移除「registry + register」敘述；補充 `SyncFile` 無 file lock 的衝突語意。 |
| 2026-04-26 | `cd0bc869` | `make_cfg()` 新增 sweep auto-format 步驟；回傳型別改為 `T_ExpCfg`（bound to `ExpCfgModel`） |
| 2026-04-26 | `254bd29c` | `library.py` 移除 `UnionModuleCfg` / `UnionWaveformCfg` import 與模組級 `TypeAdapter`，5 個解析點改用 `ModuleCfg.from_raw(raw, ml=self)` / `WaveformCfg.from_raw(raw, ml=self)` |
| 2026-04-27 | `5e09cf1c` | 把 `ModuleCfg` / `WaveformCfg` 的 registry / `from_raw` 搬到新的 `ModuleCfgFactory` / `WaveformCfgFactory`（顯式註冊取代 `__pydantic_init_subclass__` 自動收集）。`library.py` 5 個解析點改用 `*Factory.from_raw(...)` |
