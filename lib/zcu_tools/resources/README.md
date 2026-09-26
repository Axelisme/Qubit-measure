# `zcu_tools.resources` — experiment work resources

**Last updated:** 2026-09-27 — navigation-only family README

`resources/` 組織實驗工作資源的命名、讀寫與交換。它不是通用 storage framework，也不包含 datafile。

- [`context/`](context/README.md) 管理具名工作 scope 內的 MetaDict 與 ModuleLibrary。
- [`qubit_params.py`](qubit_params.py) 管理 result scope 的 typed `params.json` handoff；方法與 section 規則見 [qubit_params.md](qubit_params.md)。
- [`sample_table/`](sample_table/README.md) 管理樣品量測 CSV 與 opt-in v2 欄位協定。
- [`waveform_assets.py`](waveform_assets.py) 管理 qubit-scoped `.npz` arbitrary waveform asset；格式、方法與約束見 [waveform_assets.md](waveform_assets.md)。
- [`syncfile.py`](syncfile.py) 提供持久化物件共用的 mtime 同步機制；同步規則見其 module docstring。

## Scope 差別

各 owner 的 scope、schema、commit 與失敗政策分開，不共用 ResourceManager：

- Context（`context/`）綁定一個具名實驗工作 scope 的 MetaDict 與 ModuleLibrary。
- `QubitParams` 擁有 result scope 層級的 `params.json`，跨多個 context 共用。
- `SampleTable` 是 notebook 記錄量測結果的 CSV；v2 欄位協定是 opt-in。
- `ArbWaveformDatabase` 是 qubit-scoped 的 asset repository，與 context 分開；rename／delete 不改寫 ModuleLibrary 的 references。

`zcu_tools.resources` root 不 re-export 任何名稱；caller 從各 owner 的 module 或子 package import。

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

- Experiment cfg materialization 每次呼叫都使用 caller 傳入的 current `ml` 與 device snapshot；active context 切換不應被長壽 object 綁住。
- 各 owner 的使用限制在自己的文件：MetaDict 與 ModuleLibrary 見 [context README](context/README.md)，QubitParams 見 [qubit_params.md](qubit_params.md)，SyncFile 的同步與 file lock 限制見 `syncfile.py` 的 module docstring。
