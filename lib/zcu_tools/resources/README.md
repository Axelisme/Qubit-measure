# `zcu_tools.resources` experiment work resources

**Last updated:** 2026-10-28（external upgrade backups）

**讀者：** framework-dev。實驗資源的 scope、讀寫與持久化 owner 導覽。

`resources/` 組織實驗工作資源的命名、讀寫與交換。資料檔格式與讀寫由 [datafile](../datafile/README.md) 擁有。

- [`context/`](context/README.md) 管理 point-local typed ModuleLibrary。
- `predictor_file.py` 擁有 `zcu.predictor` v2 JSON 的具名讀寫。`PredictorFile.params` 只含 simulate 的 nullable 能量／asymmetry，`PredictorFile.flux` 是 `PredictorFlux(offset, period, unit)`。`PredictorFlux.from_mapping` 解碼與檔案相同的 offset／period／unit 格式，`to_dict()` 回傳 detached typed JSON 欄位。RPC 與檔案共用此 decoder，不另建驗證。Flux 透過 `to_frame()` 複用 `analysis.flux.FluxFrame` 與 `DevicePeriod` 的驗證，不在 resources 重寫仿射數學。此 resources→analysis 依賴沿用 C8 分層與 C9 無循環 contract。`encode_predictor_document` 驗證並回傳 detached `PredictorDocument`，供 save、validate 與離線 migration baseline 共用 v2 表示，不讓 caller 序列化 frozen mappings。`validate_predictor_file` 與 save／upgrade 保留嚴格完整規則，不建數值模型。Load 忽略 envelope、params、flux 與 source 的未知欄位，只將已知參數交給 simulate。`PredictorFile.source` 缺失或無法解析時為 None，無時區時間也投影為 None；可解析的來源保留原文。Writer 要求明確的合法 source。完整性與物理約束由 `params.build_predictor()` 驗證。V1 只透過純函式 `upgrade_predictor_document` 轉為 v2，runtime 拒讀並列出升級命令。`predictor_upgrade.py` 擁有明確升級的逐檔預覽、備份及替換。格式與還原見 [Predictor v2](../../../docs/storage-format-history/predictor-v2.md)。它不讀 component、alias 或舊 `params.json`。
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
- [`record_table/`](record_table/README.md) 管理條目的只追加 JSONL 樣本歷史、CSV 匯出與 opt-in v2 欄位協定。它也擁有 explicit record upgrade 的逐列分類、原 bytes 備份與原子替換，CLI 只呼叫該公開接縫。
- [`waveform_assets.py`](waveform_assets.py) 是 qubit-scoped arbitrary waveform asset 的公開入口。每個 Database instance 持有固定 root，capture capability 固定原目錄但每次讀現在 bytes；沒有全域 repository 選擇器。core.waveform_format 負責 NPZ samples codec；內部 data、formula 與 NPZ 模組分別負責 payload／preview、formula 語意／rendering 與實體 I/O；格式、方法與約束見 [waveform_assets.md](waveform_assets.md)。
- [`document_store.py`](document_store.py) 提供 typed YAML 的記憶體快照、單檔樂觀交易與跨程序鎖。公開 `DocumentRevision(path, revision)` 記錄絕對路徑與已安裝原始 bytes 的 SHA-256。`DocumentStore` 與 `ModuleLibrary` 的 `document_revision` 屬性，以及 entry 的 `document_revision(view)` 函式都不重新讀檔。提交或 refresh 成功才更新版本，失敗保留舊版本。Entry、point-local ModuleLibrary 與 GUI／MCP 參數操作使用它。
- [`entry/`](entry/README.md) 組合條目身分、setup 範本與獨立完整的 point。每個視圖綁自己的 DocumentStore，建立 point 時複製 seed。
- `upgrade_backup.py` 集中 parameter、predictor、record 與 spectrum 明確升級的備份定位。來源最近的 `result`／`Database` 所屬專案決定 `<專案根>-backup/`，鏡像相對路徑並加 UUID。CLI `--backup-root` 與 runner `backup_root` 可覆寫。Standalone 來源必須覆寫；資料樹內目的地與跳出鏡像根的 symlink 在任何寫入前拒絕。它不依賴 migrations；操作規則見 [備份位置](../../../docs/parameter-upgrades.md#備份位置)。
- `parameter_schema.py` 擁有 caller 注入的 `ParameterSchemaPolicy`。`parameter_upgrade.py` 提供具名文件、線性純函式升級鏈與 `main(argv, *, plan)`，預設 dry-run。Component CLI 接受檔案或 entry 目錄，目錄的 component 文件與 predictor JSON 由 entry owner 列舉。Predictor 檔案走獨立的 v1→v2 owner，不使用 component schema plan。Policy 的版本錯誤列出 Preview 與含 `--apply` 的 Apply 指令，目標由 caller 傳入，預設是來源檔。Lib 只讀版本、驗證連續性與管理單檔備份／replace，不知道 lab 欄位。完整舊格式分類、逐版本驗證與步驟由 caller 提供，未標版只在明確 upgrade 且分類成功後視為 v1。Runtime 讀取從不改檔。`UpgradePlan.notes` 列出整體操作警告；每份 `ParameterDocument.notes` 隨轉換傳到 `UpgradeResult.notes`，CLI 在該檔結果下呈現。Notes 不寫入 YAML，runner 不解析 caller 的欄位或診斷內容。

樣本歷史使用 JSONL，舊 CSV 的離線讀取與轉換由 root [`migrations/`](../../../migrations/README.md) 提供。

離線 legacy 轉換、evidence 與可續跑 publication 由 root [`migrations/storage_migration/`](../../../migrations/storage_migration/README.md) 擁有。Caller 注入具體 mapping 與 native 驗證。`lib` 不 import `migrations`。

## 檔案鎖

`file_lock.lock_path_for(data_path)` 統一 DocumentStore、ledger 與明確升級工具的鎖路徑。
它對 `Path.resolve(strict=False)` 的絕對路徑取 SHA-256，以 hex digest 加 `.lock` 作檔名。
相對路徑、symlink parent 與 `..` 指向同一檔案時使用同一把鎖。

`XDG_RUNTIME_DIR` 有值且指向目錄時，鎖放在 `$XDG_RUNTIME_DIR/qubit-measure/locks`。
否則放在 `tempfile.gettempdir()/qubit-measure-<user>/locks`。
選擇鎖路徑時按需建立 root，建立模式為 `0o700`。取得鎖後保留鎖檔，不放在 result 或 Database 資料目錄。
DocumentStore 的 explicit `lock_path` 仍可指定其他鎖檔。

鎖只協調同一台機器、同一使用者且使用相同 runtime root 的程序，不提供跨機器互斥。
舊版使用資料檔旁的 sidecar，新版使用 runtime lock，因此新舊版本不可同時寫同一份資料。
既有 `*.lock` 可能來自舊版本或其他機器，掃描與傳輸仍忽略它們。不自動刪除既有鎖檔。

## Entry 分析格式檔

`dispersive_file.py` 擁有 `zcu.dispersive` v1，公開 `DispersiveFile`、`DispersiveParams`、`PlotTransitions` 與 `load_dispersive_file`／`save_dispersive_file`。Resources root 不 re-export 這些名稱。

Reader 回傳 `DispersiveFileView`，要求 `format`、嚴格整數 `version: 1` 與 `params`，忽略未知欄位。`source` 缺失或無法解析時為 None，無時區時間也投影為 None；可解析來源保留原文。`DispersiveParamsView` 驗 bare_rf／r_f，g 保持 opaque JSON，使用者呼叫 `require_g()` 才驗證有限 GHz 數值。Writer 仍使用嚴格的 `DispersiveFile`／`DispersiveParams` 與 `AnalysisSource`。Source 的 run_id 與 created_at 必填，未知寫 null；時間須含時區，不補目前時間。

`params.g` 與 `params.bare_rf` 是 GHz，可省略或 null。`params.plot_transitions.r_f` 是 GHz 的 resonator seed，可省略或 null。其餘 transition keys 保留原始 JsonValue，不解釋躍遷名稱。

Load 讀 caller 指定的 exact path，缺檔與格式錯誤直接拋出。Save 以嚴格 writer model 重新驗證，保留 explicit null、未提供的欄位與來源字串；讀取 view 若缺合法 source 或 g 非法則拒絕寫入。Parent directory 必須存在。Writer 先寫 sibling temporary，再 atomic replace，失敗清 temporary。這不是多檔交易、multiwriter lock 或掉電持久保證。ResultEntry 只提供 `dispersive_path` 目錄，caller 選擇檔名。

離線 T1 分析格式與 codec 由 `migrations/storage_migration/t1_curve_file.py` 擁有，runtime 不讀寫。

## Point 欄位視圖

`parameter_fields.py` 集中 FieldView 與完整 caller-model candidate 驗證。Entry 的 setup／point 視圖重用 FieldView、stage_model 與 metadata 讀寫，並使用 2.0 header 契約。`parameter_metadata.py` 擁有 opaque mapping 讀寫與 2.x header 拒絕契約。這些機制不解釋 metadata 的科學意義。

## Scope 差別

各 owner 各自管理 scope、schema、commit 與失敗政策：

- `ResultEntry` 只以 Result 構成條目，Database root 只設定輸出位置。Create／open 不檢查或建立 Database，首次 Save 才建立所需目錄。Rename 只移動 Result，舊 Database 輸出不動。Setup 是範本，point 是獨立的完整參數文件；ModuleLibrary 與 point 同目錄。
- `ModuleLibrary` 綁定一個 point 的 module 與 waveform 文件。參數與 point 身分由 `entry/` 管理。
- `QubitParams` 擁有 result scope 層級的 `params.json`，跨多個 point 共用。
- `ResultEntry.records(component)` 綁定元件 JSONL 樣本歷史。v2 座標協定保持 opt-in。
- `ArbWaveformDatabase` 是 qubit-scoped 的 asset repository，與 context 分開。

`DocumentStore` 擁有單檔交易，不換算數值。Caller 提供文件 model 與整份文件的 validation。參數容器的檔案與視圖使用相同工作單位；值與 metadata 都不經 SI 投影。不同 store 不共用快照；任一欄位衝突會拒絕整筆交易。通知失敗獨立回報，不撤銷已完成的提交。它不提供跨檔掉電保證，也不自動監看檔案。

DocumentStore 在自己的單檔 edit 內合併、驗證、replace 並發布快照，不開放多檔 prepare／restore／publish 接縫。非空 commit 同時寫回 model 的讀檔轉換，以 normalized tree 比較衝突，再把差異套回 raw YAML，保留未改節點。一般讀取與空 edit 不 normalize 寫檔。只有明確啟用政策的 ModuleLibrary 遇壞條目時，會在同鎖內先完整備份再清理主檔，見 [context owner](context/README.md) 與 ADR-0063。Entry 的 setup 與 point 不共用條目鎖或交易。

Caller 從各 owner 的 module 或子 package import。

## 閱讀入口

各 owner 的行為、限制與跨模組接縫只在自己的文件：

- 條目、setup、point、角色與 metadata：[entry README](entry/README.md)。
- ModuleLibrary 與 experiment cfg materialization 的接縫：[context README](context/README.md)。
- QubitParams：[qubit_params.md](qubit_params.md)。
- JSONL 樣本與 v2 欄位協定：[record_table README](record_table/README.md)。
- ArbWaveformDatabase：[waveform_assets.md](waveform_assets.md)。
