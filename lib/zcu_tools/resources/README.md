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
- `ArbWaveformDatabase` 是 qubit-scoped 的 asset repository，與 context 分開。

`zcu_tools.resources` root 不 re-export 任何名稱；caller 從各 owner 的 module 或子 package import。

## 閱讀入口

各 owner 的行為、限制與跨模組接縫只在自己的文件：

- MetaDict、ModuleLibrary、ContextManager，以及 experiment cfg materialization 與 context 的接縫：[context README](context/README.md)。
- QubitParams：[qubit_params.md](qubit_params.md)。
- SampleTable 與 v2 欄位協定：[sample_table README](sample_table/README.md)。
- ArbWaveformDatabase：[waveform_assets.md](waveform_assets.md)。
- SyncFile 的同步規則與 file lock 限制：`syncfile.py` 的 module docstring。
