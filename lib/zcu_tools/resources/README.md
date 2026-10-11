# `zcu_tools.resources` experiment work resources

**Last updated:** 2026-10-11，舊 context 與同步基底退休，離線遷移移至 migrations

`resources/` 組織實驗工作資源的命名、讀寫與交換。它不是通用 storage framework，也不包含 datafile。

- [`context/`](context/README.md) 管理 point-local typed ModuleLibrary。
- [`qubit_params.py`](qubit_params.py) 直接管理 result scope 的 typed `params.json` handoff。公開讀寫先重新讀取 JSON，寫入採原子替換；方法與 section 規則見 [qubit_params.md](qubit_params.md)。
- [`sample_table/`](sample_table/README.md) 管理條目的只追加 JSONL 樣本歷史、CSV 匯出與 opt-in v2 欄位協定。
- [`waveform_assets.py`](waveform_assets.py) 是 qubit-scoped arbitrary waveform asset 的公開入口，持有 Database 與 scope binding。core.waveform_format 負責 NPZ samples codec；內部 data、formula 與 NPZ 模組分別負責 payload／preview、formula 語意／rendering 與實體 I/O；格式、方法與約束見 [waveform_assets.md](waveform_assets.md)。
- [`document_store.py`](document_store.py) 提供 typed YAML 的記憶體快照、單檔樂觀交易與跨程序鎖。Entry、point-local ModuleLibrary 與 GUI／MCP 參數操作使用它。
- [`entry/`](entry/README.md) 組合條目身分、setup 範本與獨立完整的 point。每個視圖綁自己的 DocumentStore，建立 point 時複製 seed。

SyncFile 與舊 CSV SampleTable 已退休。現行樣本歷史使用 JSONL，舊 CSV 的離線讀取與轉換由 root [`migrations/`](../../../migrations/README.md) 提供。

離線 legacy 轉換、evidence 與可續跑 publication 由 root [`migrations/storage_migration/`](../../../migrations/storage_migration/README.md) 擁有。Caller 注入具體 mapping 與 native 驗證。`lib` 不 import `migrations`。

## Scope 差別

各 owner 的 scope、schema、commit 與失敗政策分開，不共用 ResourceManager：

- `ResultEntry` 綁定條目與兩個 roots。Setup 是範本，point 是獨立的完整參數文件；ModuleLibrary 與 point 同目錄。
- `ModuleLibrary` 綁定一個 point 的 module 與 waveform 文件。參數與 point 身分由 `entry/` 管理。
- `QubitParams` 擁有 result scope 層級的 `params.json`，跨多個 point 共用。
- `ResultEntry.samples(component)` 綁定元件 JSONL 樣本歷史。v2 座標協定保持 opt-in。
- `ArbWaveformDatabase` 是 qubit-scoped 的 asset repository，與 context 分開。

`DocumentStore` 擁有單檔交易，不換算數值。Caller 提供文件 model 與整份文件的 validation。參數容器的檔案與視圖使用相同工作單位；值與來源表的 stderr 都不經 SI 投影。不同 store 不共用快照；任一欄位衝突會拒絕整筆交易。通知失敗獨立回報，不撤銷已完成的提交。它不提供跨檔掉電保證，也不自動監看檔案。

DocumentStore 在自己的單檔 edit 內合併、驗證、replace 並發布快照，不開放多檔 prepare／restore／publish 接縫。非空 commit 同時寫回 model 的讀檔轉換，以 normalized tree 比較衝突，再把差異套回 raw YAML，保留未改節點。讀取與空 edit 不 normalize 寫檔。Entry 的 setup 與 point 不共用條目鎖或交易。

`zcu_tools.resources` root 不 re-export 任何名稱；caller 從各 owner 的 module 或子 package import。

## 閱讀入口

各 owner 的行為、限制與跨模組接縫只在自己的文件：

- 條目、setup、point、角色與來源：[entry README](entry/README.md)。
- ModuleLibrary 與 experiment cfg materialization 的接縫：[context README](context/README.md)。
- QubitParams：[qubit_params.md](qubit_params.md)。
- JSONL 樣本與 v2 欄位協定：[sample_table README](sample_table/README.md)。
- ArbWaveformDatabase：[waveform_assets.md](waveform_assets.md)。
