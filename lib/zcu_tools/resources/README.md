# `zcu_tools.resources` experiment work resources

**Last updated:** 2026-10-23，Single／six-fluxonium settings 與共用 simulation YAML/schema

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
- [`document_store.py`](document_store.py) 提供 typed YAML 的記憶體快照、單檔樂觀交易與跨程序鎖。公開 `DocumentRevision(path, revision)` 記錄絕對路徑與已安裝原始 bytes 的 SHA-256。`DocumentStore`、`ParameterView` 與 `ModuleLibrary` 的 `document_revision` 屬性，以及 entry 的 `document_revision(view)` 函式都不重新讀檔。提交或 refresh 成功才更新版本，失敗保留舊版本。Entry、point-local ModuleLibrary 與 GUI／MCP 參數操作使用它。
- [`parameters.py`](parameters.py) 的 `open_parameters(path, MyValues)` 開啟 caller 自定值 model 的獨立參數文件。它重用 DocumentStore，不要求 entry、kind、registry、雙 roots、general 或 ledger。
- [`entry/`](entry/README.md) 組合條目身分、setup 範本與獨立完整的 point。每個視圖綁自己的 DocumentStore，建立 point 時複製 seed。

樣本歷史使用 JSONL，舊 CSV 的離線讀取與轉換由 root [`migrations/`](../../../migrations/README.md) 提供。

離線 legacy 轉換、evidence 與可續跑 publication 由 root [`migrations/storage_migration/`](../../../migrations/storage_migration/README.md) 擁有。Caller 注入具體 mapping 與 native 驗證。`lib` 不 import `migrations`。

## Entry 分析格式檔

`dispersive_file.py` 擁有 `zcu.dispersive` v1，公開 `DispersiveFile`、`DispersiveParams`、`PlotTransitions` 與 `load_dispersive_file`／`save_dispersive_file`。`t1_curve_file.py` 擁有 `zcu.t1-curve` v1，公開 `T1CurveFile`、`T1CurveParams`、`T1NoiseParams` 與 `load_t1_curve_file`／`save_t1_curve_file`。兩者共用 `analysis_source.AnalysisSource`。Resources root 不 re-export 這些名稱。

兩種 JSON 都必須含 `format`、嚴格整數 `version: 1`、`params` 與 `source`。Envelope 與 source 不接受額外欄位。`source.run_id` 與 `source.created_at` 必填，未知寫 null。Run ID 是非空字串。Created_at 只接受 `YYYY-MM-DDTHH:MM[:SS[.ffffff]]` 加 `Z` 或 `±HH:MM`。日期須為有效的 extended calendar date，分隔字元須為大寫 `T`，小數秒可有一至六位。Basic、week-date、ordinal、其他分隔字元、無時區與無效日期時間都拒絕。讀寫共用驗證並保留原文，不改成 UTC，也不補目前時間。

Dispersive 的 `params.g` 與 `params.bare_rf` 是 GHz，可省略或 null。`params.plot_transitions.r_f` 是 GHz 的 resonator seed，可省略或 null。其餘 transition keys 保留原始 JsonValue，不解釋躍遷名稱。T1 的 `params.params` 保存 `Q_cap`、`x_qp`、`Q_ind` 與 `Temp`。前三者無量綱，Temp 是 K。整個 noise params 可省略或 null；個別欄位也可省略或 null。已知數值須有限，拒絕 bool 和數字字串。

T1 noise params 中存在的 key 是 active parameter，值為 null 表示 active 但數值未知。`fixed`、`free`、`bounds`、`stderr` 與 `init` 只能引用 active 名稱。Init 的 active noise channel 集合須與 params 一致。Bounds 保存兩個有限數值或 null 端點。`cost`、`reduced_chi2`、`success`、`message`、`residual_mode`、`loss`、`max_nfev` 保存 optimizer metadata，不代表實驗完成狀態。Storage 不要求正值、完整 Temp 或至少一個 noise channel。分析 owner 負責物理完整性，成功 load 不表示資料可直接做分析。

Load 讀 caller 指定的 exact path，缺檔與格式錯誤直接拋出，不作 legacy fallback。Save 重新驗證具名 model，保留 explicit null、未提供的內容欄位與來源字串。Parent directory 必須存在。Writer 先寫 sibling temporary，再 atomic replace，失敗清 temporary。這不是多檔交易、multiwriter lock 或掉電持久保證。ResultEntry 只提供 `dispersive_path` 與 `t1_curve_path` 目錄，caller 選擇檔名，預設以 component 名命名。

離線 migration 把 `params.json.dispersive` 與 `fluxdep_fit.plot_transitions` 合寫 `dispersive/<component>.json`，把 `t1_curve_fit` 寫 `t1_curve/<component>.json`。沒有 dispersive fit 時仍可只保存 transition seed，不補 g 或 bare_rf。遷移的 run_id 為 null，created_at 只用對應 section 的含時區 timestamp。未知欄位與原 timestamp 原文仍留在 component.ext.legacy_params，無法驗證的 section 只列 pending，不產生不合格式的檔案。

## 獨立參數

Caller 從 `zcu_tools.resources.parameters` import `open_parameters`，提供既有文件路徑與 Pydantic value model。文件使用 `format: zcu.parameter-container` 和 `format_version: '2.0'`。`components` 保存 caller model 的直接值，`metadata` 是 logical dotted path 到 opaque YAML mapping 的旁表。入口接受同 major 的 forward minor，拒絕舊 major 或任何含 `provenance` 旁表的文件。錯誤包含文件路徑、讀到版本與期望版本 2.0。入口不建立缺檔、不轉換舊檔，也不修改其他成果格式。

`params.qubit.t1` 直接回數字，巢狀 model 和 mapping 使用跟隨快照的 FieldView。`params.snapshot()` 回隔離的 caller value model，不讀檔。`params.meta('qubit.t1')` 回隔離 mapping，缺項回 None，明確空 mapping 回 `{}`。若值欄名與公開 method 相同，以 item access 讀寫，例如 `params['snapshot']`。

`with params.edit() as draft` 把數筆值與 metadata 放同一筆 DocumentStore 提交。Draft 可用屬性或 `set(path, value)` 改值，用 `set_meta(path, mapping)` 整份替換該 path 的 metadata。空 mapping 清空內容。只改值保留 metadata。框架不依 stderr、source、comment 等 key 驗證領域意義，不查 ledger，不自動補時間或來源。

`refresh()` 明確重讀文件。不同 mapping leaf 沿既有規則合併，同 leaf 衝突拒絕整筆提交。驗證、body、lock 或提交前 I/O 失敗不發布候選。提交後 observer 失敗只記錄錯誤，不撤銷已提交文件。這是單檔保證，不提供跨檔原子性、掉電恢復或整筆交易的自動 retry。原子替換透過 utils.atomic_replace，只對 Windows 短暫 sharing／lock violation 重試約 0.5 秒。

`parameter_fields.py` 集中 FieldView 與完整 caller-model candidate 驗證。Entry 的 setup／point 視圖重用這個 FieldView、stage_model 與 metadata 讀寫，並使用同一 2.0 header 契約。`parameter_metadata.py` 擁有 opaque mapping 讀寫與 2.x header 拒絕契約。這些機制不解釋 metadata 的科學意義。

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
