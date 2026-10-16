---
status: superseded
---

# 量測成果 Artifact、設定快照與讀回

**狀態：** 已結案，由 [[0063]] 的保存權威、[[0069]] 的 native restore 與 [[0067]] 的 GUI Artifact／Writeback 職責接替。原先等待 measure-form 與 cfgschema-retire 的排程不再是依賴。以下保留提案與當時的問題，現行契約以正式 ADR 為準。
**關聯：** [[0063]] 的保存權威、[[0069]] 的核心與前端分工、[[0067]] 的 adapter interface、[[0072]] 的分層、[[0062]] 的 workflow 自行保存、[Measure 表單推導](../0073-measure-form-derivation.md) 的 cfg schema 升版。推導過程見 [討論紀錄](../../discussions/2026-10-14-artifact-persistence.md)。

## 脈絡

一次 Run 的設定快照（cfg snapshot）今天有三個寫入處，來源都是同一個 `RunRecord.cfg`：

| 載體 | 寫入者 | 讀者 |
| --- | --- | --- |
| ledger `acquired` 事件的 `cfg_summary` | GUI `RunService._append_acquired`、Notebook `SavedRun.save` | 沒有 |
| `data.h5` 的 `/cfg`（`CfgSnapshot`：values、cfg_type、schema_version） | `experiment.native_persistence.prepare_run_data` | GUI Load |
| Labber root comment 的 cfg | `datafile.write_labber` | Notebook `load`、Log Browser |

GUI 只讀 `data.h5`，Notebook 的 `load` 只讀 Labber，兩個前端讀不同的檔。

保存的單位是 `SaveArtifact`（`lib/zcu_tools/experiment/run_saving.py`）。一個 artifact 含多個 member，member 種類由 `ArtifactKey.member: Literal["data", "figure", "analysis"]`（`lib/zcu_tools/resources/entry/save_layout.py`）封閉列舉。種類的顯示、格式與讀回散在 GUI services、`run_saving.py` 與 adapter hooks。

讀回時框架會窺探 adapter 的結果：`LoadService` 以 `getattr(result, "cfg", None)` 取 cfg（`lib/zcu_tools/gui/app/measure/services/load.py`），`RunService` 以 `isinstance(result, RunRecord)` 取 cfg（`services/run.py`）。`load_run_with_metadata` 遇到 cfg 驗證失敗或 schema major 不符就拒絕整個檔案，資料也讀不到。

## 決策

### 設定快照的正本只在資料檔

設定快照的正本是 `data.h5` 的 `/cfg`，內容為 `CfgSnapshot`（values、cfg_type、schema_version）。Labber 照抄一份 cfg 給 Log Browser。ledger `acquired` 事件刪除 `cfg_summary` 欄位，因為它沒有讀者；`acquired` 事件本身保留。讀取既有 ledger 時忽略不支援的欄位，舊行裡的 `cfg_summary` 照樣可讀。

使用者 2026-10-14 在快照存放位置一題選 A：正本只放資料檔，ledger 不再存 cfg。

### 兩個前端都讀 data.h5

Notebook 與 GUI 都從 `data.h5` 讀回 run。只有 Log Browser 讀 Labber 檔。

使用者 2026-10-14：「Notebook與GUI應該都讀data.h5，只有log browser讀Labber檔。」

### Artifact 與 Writeback 各是一個子模組

Artifact 與 Writeback 用同一個形狀：框架定義基底與固定的子類型，並擁有每個子類型的顯示與處理；adapter 決定交出哪些子類型，並填入內容。要支援新種類時，在框架內新增子類型，不由 adapter 自訂。Writeback 現有的 `WritebackItem` 與四個子類型（`lib/zcu_tools/gui/app/measure/adapter/types.py`）就是這個形狀。

一個 Artifact 是一次寫入單位，例如 T1 的一次 run。它包含多個不同種類的項目（item）。可擴充的是項目種類，今天有量測資料、圖與分析三種。

使用者 2026-10-14：「我的想法，這應該與Writeback類似，顯示邏輯需要耦合框架，但同時要保留一定的擴充」；「擴充空間，那這兩個功能的定位應該是gui框架下的一個子模塊，對外解耦，對內可擴充。」；並修正子類型所在的層級：「Artifact目前，一個Artifact是一個寫入單位，但內部支援多個不同的項目。」

### 分兩層：experiment 的共用核心與 GUI 子模組

Artifact 的存檔與讀回也被 Notebook（`save_run`）與 workflows（`zcu_lab/workflows/common/saving.py`）使用，所以拆兩層：

| 層 | 預定位置 | 內容 | 使用者 |
| --- | --- | --- | --- |
| 共用核心 | `lib/zcu_tools/experiment/artifacts/` | 項目種類的定義、每種項目寫到 `data.h5`／Labber／PNG／JSON 的方式、讀回 | Notebook、GUI、workflows |
| GUI 子模組 | `lib/zcu_tools/gui/app/measure/artifacts/` | 存檔中心的顯示、保存狀態追蹤、adapter 交出 artifact 的介面 | measure GUI |

Writeback 採同樣結構：`lib/zcu_tools/gui/app/measure/writeback/` 收攏子類型、服務與顯示；寫入參數容器的 executor 留在 `resources.entry`（`lib/zcu_tools/resources/entry/writeback.py`）。

各格式的位元組層讀寫仍由 `datafile` 擁有，路徑仍由 `SaveLayout` 計算（[[0063]]、[[0072]]）。共用核心決定每種項目產生哪些輸出、呼叫哪個 writer，取代今天 `run_saving.py` 的角色。這些依賴方向是 gui → experiment → datafile／resources，由既有 C8 分層與 C9 無循環 contract 涵蓋，本篇不新增 contract。

使用者 2026-10-14 選擇「3A」。

### 格式由框架決定，adapter 只交內容

每種項目的檔案格式由共用核心決定。adapter 只交出內容，例如 typed RunRecord、Figure 或 AnalysisDocument，不寫檔、不選格式。

使用者 2026-10-14：「好，同意做法一」。

### 讀回：框架讀檔，核心實驗轉換

讀回分三步，每步的 owner 不同：

1. 框架以 `datafile.load_run_data` 把檔案讀成 `StoredRun`（`lib/zcu_tools/datafile/native_models.py`），並比對 experiment tag。這一步不需要實驗程式碼。
2. 核心實驗提供公開的 `from_stored(stored) -> RunRecord`，把 `StoredRun` 轉成 typed record。這是唯一的轉換，Notebook 與 GUI 共用。
3. GUI adapter 把 `StoredRun` 交給核心轉換，回傳的結果對框架不透明。Load 只把結果載進分頁供畫圖、分析與 writeback，不把設定填回表單；要看當時的設定，從 Inspect 看數據檔的 cfg。

框架不碰 typed cfg（例如 `T1Cfg`）。`LoadService` 與 `RunService` 刪除 `getattr(result, "cfg")` 與 `isinstance(result, RunRecord)` 的窺探。

Notebook 用 `NotebookAdapter(...)(T1Exp()).load(path)` 讀回。`load` 只收路徑，內部讀 `data.h5` 後呼叫 `from_stored`；不收 `Path | StoredRun` 的聯集。

使用者 2026-10-14 對「框架讀檔、把 `StoredRun` 交給 adapter」回答：「看起來沒問題」。`from_stored` 的位置與只收路徑的 Notebook `load` 由 advisor 提出，使用者隨整份摘要一併同意。

### 資料優先，設定可缺

讀回時資料與設定分開處理：

- 變數、軸或單位與實驗宣告不符：報錯，不讀回。
- cfg 驗證失敗，包含 cfg_type 或 schema major 不符：`RunRecord.cfg` 為 `None`，並發出警告寫明原因。
- `StoredRun` 一律保留原始 cfg JSON。
- GUI 照常載入資料，不回填表單；原始 cfg 由 Inspect 檢視。
- 不相容的 cfg 型別變更要升 schema major。只有必須還原 typed cfg 時，才另寫離線遷移。

[[0063]] 的「框架不提供版本遷移鏈或相容 loader」維持不變：讀回時不轉換舊 cfg，只把它標為不可用。

使用者 2026-10-14：「我認為可以」。

## 被拒絕方案

- **框架給路徑，adapter 自己寫檔。** 資料只有寫它的 adapter 讀得懂；Notebook 無法統一讀回；每個 adapter 都要自己產生 Labber；失敗隔離、原子替換與版本管理要在約 52 個 adapter 各做一次。
- **cfg 驗證失敗就拒絕整個檔案（今天的行為）。** 一次性的 `migrations/` 只轉換能通過 typed 驗證的 cfg，其餘留待處理；之後 cfg schema 一演進，舊檔的資料就跟著讀不到。
- **ledger 保留 `cfg_summary` 作為第二份快照。** 沒有讀者，且與 `data.h5` 在不同時間序列化同一物件，兩份可能不一致。

## 後果

- 一次 Run 的 cfg 只有一份正本。ledger 只記事實，不記參數。
- Notebook 改讀 `data.h5`，與 GUI 得到同一個 `RunRecord`。
- 新增項目種類只改共用核心與 GUI 子模組，不改 adapter 介面。
- 舊檔在 cfg schema 升版後仍能讀回資料；需要 cfg 的分析在各自入口拒絕。
- 代價：`AcquiredPayload` 刪欄位，GUI 與 Notebook 的 acquired 寫入與相關測試要改；`load_run_with_metadata` 改為寬容 cfg；GUI services、adapter base 與 `run_saving.py` 的程式碼搬進兩個子模組。

## 留下的問題

- workflow 的 `GroupedAxesSpec` builder 會讀 cfg snapshot。cfg 缺少時它能容忍到什麼程度，尚未決定。
- Writeback 子模組的確切切分，例如哪些服務、remote handler 與 widget 搬進去。
- Notebook 單檔 Labber `save()` 的去留。
- 轉正時要修訂的既有段落：[[0063]] 的「Run 輸出位置與 writer 權責」與 GUI Native Load 段；[[0069]] 的「Result 只表達資料，不攜帶 cfg_snapshot」、核心操作形狀的 `load`，以及「RunRecord.cfg 允許 None」一段（目前 native load 不會產生 `None`）；[[0067]] 的「Adapter capability 與事件」段，補上 Artifact 與 Writeback 子模組是框架擁有的 adapter interface。

## 補充裁定（2026-10-15）

- Load 不回填表單。2026-10-13 已裁定「算了，設計妥協，Load不再填入表單，相關功能與工具都可以刪除」；本篇原寫的「adapter 回填表單」是 advisor 討論時遺漏此裁定，使用者 2026-10-15 確認維持：「1可以」。
- 既有 ledger 行含 `cfg_summary`：讀取時忽略不支援的欄位，不寫遷移腳本。使用者：「2採用忽略不支援欄位的方式」。
- 只有 Labber 檔、沒有 `data.h5` 的舊資料，Notebook 改版後讀不回，需先用 `migrations/` 轉換。使用者：「3同意」。
- 範圍：子模組搬遷與 Writeback 一起做，不採延後的便宜版。使用者：「4一起做吧」。
