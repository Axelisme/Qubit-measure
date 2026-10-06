# `zcu_tools.resources` experiment work resources

**Last updated:** 2026-10-06，讀檔轉換、參數容器與離線遷移

`resources/` 組織實驗工作資源的命名、讀寫與交換。它不是通用 storage framework，也不包含 datafile。

- [`context/`](context/README.md) 管理具名工作 scope 內的 MetaDict 與 ModuleLibrary。
- [`qubit_params.py`](qubit_params.py) 管理 result scope 的 typed `params.json` handoff；方法與 section 規則見 [qubit_params.md](qubit_params.md)。
- [`sample_table/`](sample_table/README.md) 管理樣品量測 CSV 與 opt-in v2 欄位協定。
- [`waveform_assets.py`](waveform_assets.py) 管理 qubit-scoped `.npz` arbitrary waveform asset；格式、方法與約束見 [waveform_assets.md](waveform_assets.md)。
- [`syncfile.py`](syncfile.py) 提供現行持久化物件共用的 mtime 同步機制；同步規則見其 module docstring。
- [`document_store.py`](document_store.py) 提供新格式 typed YAML 的記憶體快照、單檔樂觀交易與跨程序鎖，目前未接線到現行 runtime。
- [`entry/`](entry/README.md) 組合條目身分、setup 範本與獨立完整的 point。每個視圖綁自己的 DocumentStore，建立 point 時複製 seed。
- [`storage_migration/`](storage_migration/README.md) 組合離線 legacy 轉換、evidence 與可續跑 publication。Caller 注入具體 mapping 與 native 驗證，不接正常 runtime。

## Scope 差別

各 owner 的 scope、schema、commit 與失敗政策分開，不共用 ResourceManager：

- Context（`context/`）綁定一個具名實驗工作 scope 的 MetaDict 與 ModuleLibrary。
- `QubitParams` 擁有 result scope 層級的 `params.json`，跨多個 context 共用。
- `SampleTable` 是 notebook 記錄量測結果的 CSV；v2 欄位協定是 opt-in。
- `ArbWaveformDatabase` 是 qubit-scoped 的 asset repository，與 context 分開。

`DocumentStore` 擁有單檔交易，不換算數值。Caller 提供文件 model 與整份文件的 validation。參數容器的檔案與視圖使用相同工作單位；值與來源表的 stderr 都不經 SI 投影。不同 store 不共用快照；任一欄位衝突會拒絕整筆交易。通知失敗獨立回報，不撤銷已完成的提交。它不提供跨檔掉電保證，也不自動監看檔案。

DocumentStore 在自己的單檔 edit 內合併、驗證、replace 並發布快照，不開放多檔 prepare／restore／publish 接縫。非空 commit 同時寫回 model 的讀檔轉換，以 normalized tree 比較衝突，再把差異套回 raw YAML，保留未改節點。讀取與空 edit 不 normalize 寫檔。Entry 的 setup 與 point 不共用條目鎖或交易。

`zcu_tools.resources` root 不 re-export 任何名稱；caller 從各 owner 的 module 或子 package import。

## 閱讀入口

各 owner 的行為、限制與跨模組接縫只在自己的文件：

- MetaDict、ModuleLibrary、ContextManager，以及 experiment cfg materialization 與 context 的接縫：[context README](context/README.md)。
- QubitParams：[qubit_params.md](qubit_params.md)。
- SampleTable 與 v2 欄位協定：[sample_table README](sample_table/README.md)。
- ArbWaveformDatabase：[waveform_assets.md](waveform_assets.md)。
- SyncFile 的同步規則與 file lock 限制：`syncfile.py` 的 module docstring。
