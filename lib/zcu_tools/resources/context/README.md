# `zcu_tools.resources.context` — named experiment work contexts

**Last updated:** 2026-09-27 — resource owner split

Context 是綁定特定參數資源的具名實驗工作 scope；同一 context 預期具有相同的環境與實驗語意。儀器值只是命名便利，名稱可以是任意合法字串。Context 不會自動監控實驗環境。

---

## `MetaDict`（`metadict.py`）

以 JSON 儲存實驗參數的 dict-like 物件，所有欄位可透過屬性語法存取。

```python
md = MetaDict("experiment/meta_info.json")
md.qubit_freq = 5.0e9     # 自動 sync + 寫回
print(md.qubit_freq)      # 自動 sync + 讀取
```

**型別特殊處理**：

- 寫入時先呼叫 `format_obj()`（utils）把 model / array / scalar 轉成 JSON-friendly Python value，再把 `complex` 寫成標記物件 `{"__complex__": [real, imag]}`。
- 載入時只把標記物件還原為 `complex`；舊檔案中無標記且形如 complex 的字串會以 deprecation warning 還原，供舊 `meta_info.json` 過渡。
- 使用者字串維持字串語意；若字串本身形如舊 complex literal，dump 時會加上 `{"__metadict_string__": value}` escape marker，避免下一次載入被 legacy parser 誤判。

**受保護的屬性**（不進入 `_data`）：`_` 開頭的名稱，以及 `MetaDict` / `SyncFile` class 或 MRO 上已存在的名稱，例如 `dump`、`load`、`sync`、`has_persistence`、`clone`、`items`、`keys`、`get`、`update`。對 protected name 寫入或載入 protected key 會 fail-fast，避免 `_data` 內存在被 class attribute 遮蔽、永遠讀不到的 shadow key。

**批次寫入**：`update(values, **kwargs)` 在一次 auto-sync write transaction 中更新多個 key；單一屬性賦值仍會立即同步，批次修改應優先使用 `update()`。

**`clone(dst_path, readonly)`**：複製整個 MetaDict 到新路徑（要求目標不存在）。

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
# library.py 內 store 解析點（_load / register_* / update_*）統一使用：
WaveformCfgFactory.from_raw(raw, ml=self)
ModuleCfgFactory.from_raw(raw, ml=self)
```

兩個 Factory 都是薄封裝：`from_raw(raw, ml=...)` 內部直接呼叫 `TypeAdapter(...).validate_python(..., context={"ml": ml})`。`ModuleCfg` / `WaveformCfg` 的分派表由各自 TypeAlias 的 `Union[...]` 決定，不再使用 runtime registry。

**為什麼叫 `from_raw` 而不是 `validate`？** Pydantic v2 的 `BaseModel` 已有 deprecated 的 `validate()` classmethod；如果我們在 cfg 上覆寫同名方法會被 pyright 標記為 deprecated。`from_raw` 既避免衝突，語意也更清楚（從「raw 任意輸入」轉成具體 typed cfg）。

**`ModuleDumper`**（自訂 YAML Dumper）：

- 縮排層級減少時插入空白行（增加可讀性）。
- dict 的 value 依型別排序：str > int > float > bool > other > list > dict（避免 nested dict 佔用視覺空間）。

## `ContextManager`（`manager.py`）

將 `ModuleLibrary` + `MetaDict` 綁定到 **具名 context**（常以電流/電壓命名的資料夾）。

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

## 使用模式（典型 notebook 開頭）

```python
from zcu_tools.resources import ContextManager

em = ContextManager("Database/chipA/qubitQ1")

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

from zcu_tools.resources import FluxDepFit, ParamsProject, QubitParams

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
