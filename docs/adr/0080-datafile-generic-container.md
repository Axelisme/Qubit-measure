---
status: accepted
date: 2026-10-22
related: [0063, 0072]
---

# datafile：通用容器格式與獨立介面

**Last updated:** 2026-10-27

## 脈絡

`datafile` 原本同時擁有通用的變數／軸模型與一整組實驗語義：native `data.h5` 2.1 的根屬性是 run_id、experiment tag、時間與 completion，`/context` 放 entry、point、roles，`/device` 放 device snapshot，`/provenance` 放軟體版本；`RunMetadata`、`RunSnapshot`、`SoftwareProvenance` 都定義在 datafile 內。分析結果則走另一個獨立的 JSON 檔（`zcu.analysis-result` 1.0），因為 `data.h5` 的槽位是為量測 run 設計的，cfg 對分析結果沒有意義。另有 `fluxdep_database.py` 這種實驗專用 reader 放在 datafile。

使用者 2026-10-22 裁定：「我們應該約定好一個足夠通用，足夠穩定，容許使用者根據自己的需要擴充的datafile格式與介面，這樣其他人有不同的repo也能只import datafile而不需要知道其他模塊的用途」；「datafile應該專心處理數據與json就好，符合KISS原則，而使用方可以自己用adapter來將json轉成需要的強型別」；對「cfg、metadata、ext 以意義分槽」的異議：「對於分析結果之類的，cfg是沒什麼意義的，metadata與ext定位也模糊」。本 ADR 把容器改成以表示法分槽，實驗語義全部退回 experiment 層。使用者 2026-10-22 對草稿回「可以」。

## 決策

### 容器只認三種東西

一個 `.h5` 檔是一個容器，根下固定三個命名空間，datafile 只知道它們的表示法，不知道意義：

| 命名空間 | 內容 | datafile 的責任 |
|---|---|---|
| `vars/<name>` | 一個陣列加有序的具名軸；軸有 name、unit、dtype，陣列任意維度，軸以 HDF5 dimension scale 附在對應維度 | 驗 schema（名稱合法、軸長與形狀一致、dtype 為數值）、讀寫 |
| `docs/<name>` | 一份有限 JSON 文件（UTF-8 scalar dataset） | 驗「是有限 JSON」、原樣讀寫，不解釋任何鍵 |
| `groups/<name>` | 生產者自己的 HDF5 子樹 | 完全不碰：讀時原樣保留、寫回時原樣帶著 |

根屬性：`format="zcu.container"`、`format_version`（major.minor）、`producer`（生產者自報的字串，例如 `zcu_tools.experiment 3.0`），以及可選的 `kind`（生產者的檔案種類名，例如 `experiment-run`、`analysis-result`，供工具列檔與分派，datafile 不驗證其值）。

三個命名空間都可以為空：只有 docs 的檔（純分析結果）、只有 vars 的檔都合法。名稱規則共用：非空的合法路徑段，不含 `/`、NUL、`.`、`..`，不重複。

`cfg`、`metadata`、`analysis`、`source` 這些是生產者的命名慣例，不是格式的一部分。experiment 層的量測 run 用 `docs/cfg`（含 `cfg_type`、`schema_version` 兩個鍵）與 `docs/metadata`（run_id、experiment、started_at、finished_at、completion、snapshot、provenance、device_snapshot，具名型別由 experiment 擁有）；分析結果用 `docs/analysis` 與 `docs/source`（指回來源檔的相對路徑與 run_id），曲線放 `vars`。Labber 的 comment、tags、project、user 是 Labber 轉出時的參數，不進容器；需要留痕時由生產者放進自己的 doc。

### 版本與相容規則

- `format_version` 只有兩個等級：minor 只能新增命名空間內的可選內容，reader 必須接受同 major 的任何 minor；major 改變就是另一個 reader。
- 未知內容一律保真：讀到的 `docs`／`groups`／`vars` 不管認不認得，寫回時完整帶著；`vars` 之外的 HDF5 節點不改動。這取代今天的整檔 `NativeExtensions.file_image`。
- Reader 只接受 zcu.container major 3。舊 `zcu.experiment-data` 2.x 不在支援範圍，直接拒絕。
- 生產者的文件 schema 版本放在各自的 doc 內（如 `docs/cfg.schema_version`），容器不管。

### 介面

公開面收斂為一個模組層級的 API，型別只用 datafile 自己的：

- `save(path, container: Container, *, replace: bool = False)`：寫到呼叫者的精確路徑，同目錄 temp 加 atomic replace。讀寫共用 `Container(vars, docs, groups, attrs)`，caller 可以直接修改 load 的結果再寫回，不另行拆出 header 參數。
- `load(path) -> Container`：`groups` 是 detached opaque handle，只供寫回，原檔關閉或刪除後仍可使用。`attrs` 是 frozen `ContainerAttrs(format_version: str, producer: str, kind: str | None, extra: Mapping[str, JsonScalar])`。新檔由 caller 指定 3.0 與 producer，kind 可為 None。讀回的同 major 來源 minor 與未知根屬性 extra 在 save 保留，不降版。`format` 固定為 zcu.container，不接受 extra 覆寫 header keys。extra 只接受有限 JSON scalar，null 以 HDF5 null-dataspace attribute 表示。
- `read_doc(path, name) -> JsonObject`：只讀指定的有限 JSON object，不讀 vars、其他 docs 或 opaque groups。只接受與 load 相同的 zcu.container 3.x。缺文件 raise 指名來源檔與文件名的 KeyError；格式或文件非法 raise 指名來源檔的 FormatError。壞 vars 不阻礙合法文件讀取。
- `inspect(path) -> ContainerInfo`：只讀根屬性與 var schema、doc/group 名稱，不解碼 docs，也不載陣列。
- `Variable(values, axes: tuple[Axis, ...], unit)`、`Axis(name, unit, values)`：axes 按 NumPy 維度順序排列。`VariableSchema` 回報 inspect 的形狀、軸、dtype 與單位。`ContainerInfo` 的 axes 不載座標值。另提供 `JsonObject`、`JsonScalar` 與 `FormatError`。

Labber 轉出保留為 datafile 內的獨立模組。`write_labber` 消費 Container.vars，不解釋 docs，LabberMetadata 由 caller 明傳。`read_labber` 回傳 `(Container, LabberMetadata)`，原始子樹與 timestamps 放在 opaque groups/labber。既有低層 readers 仍可直接取得 timestamps。Labber independent 的多 Log_N 版面由內部 helper 直接寫，不經中介檔。

### 獨立性

datafile 不 import `zcu_tools` 的其他 package。今天僅有的三個依賴（`core.numeric_scalar.normalize_numpy_scalar`、`format_version`、`utils.atomic_replace`）搬進 datafile 自有的小模組或改用標準函式庫。`.importlinter` 的 `datafile-standalone` contract 禁止 `zcu_tools.datafile` import 其他 framework packages。Contract 明列目前所有其他 package 與 format_version module，不新增 ignore_imports。外部依賴只有 NumPy、h5py、pydantic。

Fluxdep database reader 由 `gui.app.fluxdep.database` 擁有。`analysis_document.py` 的 JSON 信封是 `docs/analysis` 的內容約定，owner 是 experiment.artifacts，`AnalysisDocument.to_doc()` 回傳有限 JSON 物件。現有 artifact 仍寫 analysis/<pane>.json，SaveLayout 與 ledger 不因 owner 搬遷改變。

### 其他 package 的對應

- `experiment`：擁有 `RunMetadata`（pydantic）與 cfg 的具名型別，save 時 `model_dump(mode="json")` 成 doc、load 時 `TypeAdapter` 驗回（[[0063]] 的 `axes_spec` mapping 不變，只是輸出目標從 `ExperimentPayload` 改為 `vars`）。`RunRecord.metadata` 承載它。`native_persistence.read_run_metadata(path)` 用 read_doc 只讀 metadata；`metadata_from_container(container)` 驗證已載入的 Container。兩者共用同一個 RunMetadata 驗證函式，宿主不重做驗證，也不為取得已載 metadata 再開檔。
- GUI、notebook、workflow 的宿主：只傳 `RunRecord`，不再另傳 metadata；載入對話框等需要 run_id、point、experiment 的地方讀 experiment 層的型別。Results browser 的唯讀歷史展示依 [[0067]] 直接讀保存文件，不套用當前實驗 schema；真正 Load 仍走 experiment 驗證。
- 遷移工具：`migrations/` 讀舊 Labber，輸出新容器。歷史 Labber comment 與 Instrument config 解碼只在 migrations。

### 衍生輸出的身分與命名

2026-10-27 的保存裁決由 [[0063]] 擁有。GUI／Notebook run 保存與 workflow Labber export 在 comment 的 cfg 旁加入 `snapshot: {point, roles}`。Datafile 的 encoder 只驗有限 JSON，不解析目標身分。既有 reader 不要求 snapshot，未知 keys 不改變 cfg 的解讀。這不改 Container 3.0、Labber 結構或 remote wire version。

存檔 callback 接收 SaveContext 的名稱、run_id、point、roles、cfg values 與 saved_at，回傳 stem。SaveLayout 使用 `runs/<stem>/` 與日期樹內的 `<stem>.hdf5`，不另加 run_id。只有 Labber 初始目的已存在時才用既有 suffix helper。Native 保留明確 replace=True 的同路徑覆寫。完整 run_id 仍保存在 metadata，不從檔名猜測。

使用者於 2026-10-27 裁決：「Q-L3我的想法是adapter決定怎麼生成名字，可以有預設實現。」並要求：「run_id也是由adapter行數補上，請你遵守框架不做多餘行為的設計理念，將可能需要的資訊作為參數傳給save_stem。」預設以 dotted adapter／tag／workflow name、roles 與完整 run_id 組名，自訂 callback 可以完全替換。歷史形狀與辨識方式見 [存檔歷程](../storage-format-history/2026-10-27-labber-identity.md)。

## 後果

- 其他 repo 可以只安裝 datafile 讀寫同一種容器；他們的 docs 與 groups 名稱自己定，不需要知道 experiment、resources 或 GUI。
- 分析結果與量測 run 用同一個容器，差別只在 docs 的名字與 vars 的內容。
- 新寫入使用 3.0。[[0063]] 的保存權威不變，通用格式與版本規則由本篇擁有。測試從 save／load／inspect／append 與 Labber 的公開接縫驗證。
- 失去的：`/context`、`/device` 逐欄驗證從 datafile 移到 experiment 層；datafile 不再能單獨判斷一個檔「是不是合法的量測 run」，只能判斷「是不是合法的容器」。

## 實作時的固定選擇

1. `docs/cfg` 的 `cfg_type`、`schema_version` 是文件內的鍵，不是 HDF5 屬性；容器對 doc 只有名稱與內容。
2. `groups` 寫回時原樣帶著；若同一次寫入改變了某個 `vars` 的形狀，`groups` 內指向該 var 的 HDF5 引用不保證仍有效，與今天整檔影像的限制相同，寫在 `save` 的 docstring。
3. 容器只保留整份 save／load，不提供逐列 append writer。
4. 使用者於 2026-10-24 同意刪除未用的 append API 與 native 2.x reader。不支援的舊資料 Load 失敗即可，因此不保留 native 2.x 批次轉換工具。
