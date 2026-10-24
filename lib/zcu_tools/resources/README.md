# `zcu_tools.resources` experiment work resources

**Last updated:** 2026-10-24

**讀者：** framework-dev。實驗資源的 scope、讀寫與持久化 owner 導覽。

`resources/` 組織實驗工作資源的命名、讀寫與交換。資料檔格式與讀寫由 [datafile](../datafile/README.md) 擁有。

- [`context/`](context/README.md) 管理 point-local typed ModuleLibrary。
- `predictor_file.py` 擁有 `zcu.predictor` v1 JSON 的具名讀寫。它依模型名稱使用 simulate 的 nullable 參數型別，驗證欄名、有限數值與來源 metadata。`validate_predictor_file(PredictorFile)` 提供不做 I/O 的候選驗證，與 load/save 共用表示規則，供 dry-run 使用。完整性與物理約束由 `params.build_predictor()` 驗證。它不讀 component、alias 或舊 `params.json`。
- `simulation_replay.py` owns opt-in private FTF and single-fluxonium input JSON, content integrity,
  atomic file I/O and acquire/round reference resolution. It retains detached
  waveform samples and settings, not measured I/Q. Template arrays use explicit
  JSON records of dtype, shape and flat values, with typed float64, int64 and bool
  reconstruction. Numeric manifests stay opaque and are verified by their numeric
  owner. Normal waveform assets and data.h5
  policies do not change. This module does not import program or simulate.
- [`six_fluxonium/`](six_fluxonium/README.md) reads six independent circuit materials with six local DC sources, one shared coil and fixed mux/pulse routes.
- [`single_fluxonium/`](single_fluxonium/README.md) reads immutable single-qubit YAML with microwave, fast-flux and single-cavity routes. `simulation_schema.py` and `yaml.py` share neutral physical records and safe YAML decoding with FTF.
- [`ftf/`](ftf/README.md) 讀取 immutable FTF YAML，擁有具名欄位、units、limits、接線與 noise PSD 的 field-path 診斷，不依賴 simulate、device 或 program。
- [`qubit_params.py`](qubit_params.py) 直接管理 result scope 的 typed `params.json` handoff。公開讀寫先重新讀取 JSON，寫入採原子替換；方法與 section 規則見 [qubit_params.md](qubit_params.md)。
- [`sample_table/`](sample_table/README.md) 管理條目的只追加 JSONL 樣本歷史、CSV 匯出與 opt-in v2 欄位協定。
- [`waveform_assets.py`](waveform_assets.py) 是 qubit-scoped arbitrary waveform asset 的公開入口。每個 Database instance 持有固定 root，capture capability 固定原目錄但每次讀現在 bytes；沒有全域 repository 選擇器。core.waveform_format 負責 NPZ samples codec；內部 data、formula 與 NPZ 模組分別負責 payload／preview、formula 語意／rendering 與實體 I/O；格式、方法與約束見 [waveform_assets.md](waveform_assets.md)。
- [`document_store.py`](document_store.py) 提供 typed YAML 的記憶體快照、單檔樂觀交易與跨程序鎖。公開 `DocumentRevision(path, revision)` 記錄絕對路徑與已安裝原始 bytes 的 SHA-256。`DocumentStore` 與 `ModuleLibrary` 的 `document_revision` 屬性，以及 entry 的 `document_revision(view)` 函式都不重新讀檔。提交或 refresh 成功才更新版本，失敗保留舊版本。Entry、point-local ModuleLibrary 與 GUI／MCP 參數操作使用它。
- [`entry/`](entry/README.md) 組合條目身分、setup 範本與獨立完整的 point。每個視圖綁自己的 DocumentStore，建立 point 時複製 seed。

樣本歷史使用 JSONL，舊 CSV 的離線讀取與轉換由 root [`migrations/`](../../../migrations/README.md) 提供。

離線 legacy 轉換、evidence 與可續跑 publication 由 root [`migrations/storage_migration/`](../../../migrations/storage_migration/README.md) 擁有。Caller 注入具體 mapping 與 native 驗證。`lib` 不 import `migrations`。

## Entry 分析格式檔

`dispersive_file.py` 擁有 `zcu.dispersive` v1，公開 `DispersiveFile`、`DispersiveParams`、`PlotTransitions` 與 `load_dispersive_file`／`save_dispersive_file`。Resources root 不 re-export 這些名稱。

JSON 必須含 `format`、嚴格整數 `version: 1`、`params` 與 `source`。Envelope 與 source 不接受額外欄位。`analysis_source.AnalysisSource` 驗證來源。`source.run_id` 與 `source.created_at` 必填，未知寫 null。Run ID 是非空字串。Created_at 只接受 `YYYY-MM-DDTHH:MM[:SS[.ffffff]]` 加 `Z` 或 `±HH:MM`。日期須有效，小數秒可有一至六位。讀寫保留原文，不改成 UTC，也不補目前時間。

`params.g` 與 `params.bare_rf` 是 GHz，可省略或 null。`params.plot_transitions.r_f` 是 GHz 的 resonator seed，可省略或 null。其餘 transition keys 保留原始 JsonValue，不解釋躍遷名稱。

Load 讀 caller 指定的 exact path，缺檔與格式錯誤直接拋出。Save 重新驗證具名 model，保留 explicit null、未提供的欄位與來源字串。Parent directory 必須存在。Writer 先寫 sibling temporary，再 atomic replace，失敗清 temporary。這不是多檔交易、multiwriter lock 或掉電持久保證。ResultEntry 只提供 `dispersive_path` 目錄，caller 選擇檔名。

離線 T1 分析格式與 codec 由 `migrations/storage_migration/t1_curve_file.py` 擁有，runtime 不讀寫。

## Point 欄位視圖

`parameter_fields.py` 集中 FieldView 與完整 caller-model candidate 驗證。Entry 的 setup／point 視圖重用 FieldView、stage_model 與 metadata 讀寫，並使用 2.0 header 契約。`parameter_metadata.py` 擁有 opaque mapping 讀寫與 2.x header 拒絕契約。這些機制不解釋 metadata 的科學意義。

## Scope 差別

各 owner 各自管理 scope、schema、commit 與失敗政策：

- `ResultEntry` 只以 Result 構成條目，Database root 只設定輸出位置。Create／open 不檢查或建立 Database，首次 Save 才建立所需目錄。Rename 只移動 Result，舊 Database 輸出不動。Setup 是範本，point 是獨立的完整參數文件；ModuleLibrary 與 point 同目錄。
- `ModuleLibrary` 綁定一個 point 的 module 與 waveform 文件。參數與 point 身分由 `entry/` 管理。
- `QubitParams` 擁有 result scope 層級的 `params.json`，跨多個 point 共用。
- `ResultEntry.samples(component)` 綁定元件 JSONL 樣本歷史。v2 座標協定保持 opt-in。
- `ArbWaveformDatabase` 是 qubit-scoped 的 asset repository，與 context 分開。

`DocumentStore` 擁有單檔交易，不換算數值。Caller 提供文件 model 與整份文件的 validation。參數容器的檔案與視圖使用相同工作單位；值與 metadata 都不經 SI 投影。不同 store 不共用快照；任一欄位衝突會拒絕整筆交易。通知失敗獨立回報，不撤銷已完成的提交。它不提供跨檔掉電保證，也不自動監看檔案。

DocumentStore 在自己的單檔 edit 內合併、驗證、replace 並發布快照，不開放多檔 prepare／restore／publish 接縫。非空 commit 同時寫回 model 的讀檔轉換，以 normalized tree 比較衝突，再把差異套回 raw YAML，保留未改節點。一般讀取與空 edit 不 normalize 寫檔。只有明確啟用政策的 ModuleLibrary 遇壞條目時，會在同鎖內先完整備份再清理主檔，見 [context owner](context/README.md) 與 ADR-0063。Entry 的 setup 與 point 不共用條目鎖或交易。

Caller 從各 owner 的 module 或子 package import。

## 閱讀入口

各 owner 的行為、限制與跨模組接縫只在自己的文件：

- 條目、setup、point、角色與 metadata：[entry README](entry/README.md)。
- ModuleLibrary 與 experiment cfg materialization 的接縫：[context README](context/README.md)。
- QubitParams：[qubit_params.md](qubit_params.md)。
- JSONL 樣本與 v2 欄位協定：[sample_table README](sample_table/README.md)。
- ArbWaveformDatabase：[waveform_assets.md](waveform_assets.md)。
