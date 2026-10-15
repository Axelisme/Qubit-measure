# `zcu_tools.resources` experiment work resources

**Last updated:** 2026-10-15，library 讀取清理例外

`resources/` 組織實驗工作資源的命名、讀寫與交換。它不是通用 storage framework，也不包含 datafile。

- [`context/`](context/README.md) 管理 point-local typed ModuleLibrary。
- [`qubit_params.py`](qubit_params.py) 直接管理 result scope 的 typed `params.json` handoff。公開讀寫先重新讀取 JSON，寫入採原子替換；方法與 section 規則見 [qubit_params.md](qubit_params.md)。
- [`sample_table/`](sample_table/README.md) 管理條目的只追加 JSONL 樣本歷史、CSV 匯出與 opt-in v2 欄位協定。
- [`waveform_assets.py`](waveform_assets.py) 是 qubit-scoped arbitrary waveform asset 的公開入口。每個 Database instance 持有固定 root，capture capability 固定原目錄但每次讀現在 bytes；沒有全域 repository 選擇器。core.waveform_format 負責 NPZ samples codec；內部 data、formula 與 NPZ 模組分別負責 payload／preview、formula 語意／rendering 與實體 I/O；格式、方法與約束見 [waveform_assets.md](waveform_assets.md)。
- [`document_store.py`](document_store.py) 提供 typed YAML 的記憶體快照、單檔樂觀交易與跨程序鎖。Entry、point-local ModuleLibrary 與 GUI／MCP 參數操作使用它。
- [`parameters.py`](parameters.py) 的 `open_parameters(path, MyValues)` 開啟 caller 自定值 model 的獨立參數文件。它重用 DocumentStore，不要求 entry、kind、registry、雙 roots、general 或 ledger。
- [`entry/`](entry/README.md) 組合條目身分、setup 範本與獨立完整的 point。每個視圖綁自己的 DocumentStore，建立 point 時複製 seed。

SyncFile 與舊 CSV SampleTable 已退休。現行樣本歷史使用 JSONL，舊 CSV 的離線讀取與轉換由 root [`migrations/`](../../../migrations/README.md) 提供。

離線 legacy 轉換、evidence 與可續跑 publication 由 root [`migrations/storage_migration/`](../../../migrations/storage_migration/README.md) 擁有。Caller 注入具體 mapping 與 native 驗證。`lib` 不 import `migrations`。

## 獨立參數

Caller 從 `zcu_tools.resources.parameters` import `open_parameters`，提供既有文件路徑與 Pydantic value model。文件使用 `format: zcu.parameter-container` 和 `format_version: '2.0'`。`components` 保存 caller model 的直接值，`metadata` 是 logical dotted path 到 opaque YAML mapping 的旁表。入口接受同 major 的 forward minor，拒絕舊 major 或任何含 `provenance` 旁表的文件。錯誤包含文件路徑、讀到版本與期望版本 2.0。入口不建立缺檔、不轉換舊檔，也不修改其他成果格式。

`params.qubit.t1` 直接回數字，巢狀 model 和 mapping 使用跟隨快照的 FieldView。`params.snapshot()` 回隔離的 caller value model，不讀檔。`params.meta('qubit.t1')` 回隔離 mapping，缺項回 None，明確空 mapping 回 `{}`。若值欄名與公開 method 相同，以 item access 讀寫，例如 `params['snapshot']`。

`with params.edit() as draft` 把數筆值與 metadata 放同一筆 DocumentStore 提交。Draft 可用屬性或 `set(path, value)` 改值，用 `set_meta(path, mapping)` 整份替換該 path 的 metadata。空 mapping 清空內容。只改值保留 metadata。框架不依 stderr、source、comment 等 key 驗證領域意義，不查 ledger，不自動補時間或來源。

`refresh()` 明確重讀文件。不同 mapping leaf 沿既有規則合併，同 leaf 衝突拒絕整筆提交。驗證、body、lock 或提交前 I/O 失敗不發布候選。提交後 observer 失敗只記錄錯誤，不撤銷已提交文件。這是單檔保證，不提供跨檔原子性、掉電恢復或自動 retry。

`parameter_fields.py` 集中 FieldView 與完整 caller-model candidate 驗證。Entry 的 setup／point 視圖重用這個 FieldView、stage_model 與 metadata 讀寫，並使用同一 2.0 header 契約。`parameter_metadata.py` 擁有 opaque mapping 讀寫與 2.x header 拒絕契約。這些機制不解釋 metadata 的科學意義。

## Scope 差別

各 owner 的 scope、schema、commit 與失敗政策分開，不共用 ResourceManager：

- `ResultEntry` 綁定條目與兩個 roots。Setup 是範本，point 是獨立的完整參數文件；ModuleLibrary 與 point 同目錄。
- `ModuleLibrary` 綁定一個 point 的 module 與 waveform 文件。參數與 point 身分由 `entry/` 管理。
- `QubitParams` 擁有 result scope 層級的 `params.json`，跨多個 point 共用。
- `ResultEntry.samples(component)` 綁定元件 JSONL 樣本歷史。v2 座標協定保持 opt-in。
- `ArbWaveformDatabase` 是 qubit-scoped 的 asset repository，與 context 分開。

`DocumentStore` 擁有單檔交易，不換算數值。Caller 提供文件 model 與整份文件的 validation。參數容器的檔案與視圖使用相同工作單位；值與 metadata 都不經 SI 投影。不同 store 不共用快照；任一欄位衝突會拒絕整筆交易。通知失敗獨立回報，不撤銷已完成的提交。它不提供跨檔掉電保證，也不自動監看檔案。

DocumentStore 在自己的單檔 edit 內合併、驗證、replace 並發布快照，不開放多檔 prepare／restore／publish 接縫。非空 commit 同時寫回 model 的讀檔轉換，以 normalized tree 比較衝突，再把差異套回 raw YAML，保留未改節點。一般讀取與空 edit 不 normalize 寫檔。只有明確啟用政策的 ModuleLibrary 遇壞條目時，會在同鎖內先完整備份再清理主檔，見 [context owner](context/README.md) 與 ADR-0063。Entry 的 setup 與 point 不共用條目鎖或交易。

`zcu_tools.resources` root 不 re-export 任何名稱；caller 從各 owner 的 module 或子 package import。

## 閱讀入口

各 owner 的行為、限制與跨模組接縫只在自己的文件：

- 條目、setup、point、角色與 metadata：[entry README](entry/README.md)。
- ModuleLibrary 與 experiment cfg materialization 的接縫：[context README](context/README.md)。
- QubitParams：[qubit_params.md](qubit_params.md)。
- JSONL 樣本與 v2 欄位協定：[sample_table README](sample_table/README.md)。
- ArbWaveformDatabase：[waveform_assets.md](waveform_assets.md)。
