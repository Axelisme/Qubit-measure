---
date: 2026-10-14
status: active
superseded_by: null
related:
  - docs/adr/draft/artifact-persistence.md
  - docs/adr/0063-persistence-ownership.md
  - docs/adr/0069-experiment-core-and-frontends.md
  - docs/adr/0067-gui-application.md
  - docs/adr/0072-module-boundaries.md
basis:
  - main@30838a9b6（現況調查時）
  - main@32e6ec295（撰寫時）
---

# 量測成果 Artifact、設定快照與讀回的推導

## 結論

決策見 [Artifact persistence ADR draft](../adr/draft/artifact-persistence.md)。實作排在 measure-form 與 cfgschema-retire 之後。本篇記推導過程、使用者的修正與被否決的路。

- 設定快照的正本只在 `data.h5` 的 `/cfg`，Labber 照抄給 Log Browser；ledger `acquired` 事件刪除 `cfg_summary`。
- Notebook 與 GUI 都讀 `data.h5`。
- Artifact 與 Writeback 各是一個子模組，形狀相同：框架定義固定子類型並擁有其顯示與處理，adapter 選子類型並填內容。
- Artifact 拆兩層：`experiment` 下的共用核心，與 `gui/app/measure` 下的 GUI 子模組。
- 格式由框架核心依項目種類決定，adapter 不寫檔。
- 讀回由框架讀成 `StoredRun`，核心實驗以 `from_stored` 轉成 `RunRecord`，GUI adapter 再解讀 cfg。
- 讀回採「資料優先、設定可缺」。

## 問題

討論從「cfg snapshot 存在哪裡」開始。現況調查發現同一個 `RunRecord.cfg` 有三個寫入處：ledger `acquired.cfg_summary`、`data.h5` 的 `/cfg`、Labber comment。其中 `cfg_summary` 沒有任何讀者；它的名稱是 summary，內容卻是完整的 `model_dump`。GUI 只讀 `data.h5`，Notebook 只讀 Labber。

調查也發現框架用兩種方式窺探 adapter 的結果：Load 用 `getattr(result, "cfg", None)`，Run 用 `isinstance(result, RunRecord)`。ADR-0069 寫「RunRecord.cfg 允許 None」，但 native load 遇到 cfg 驗證失敗會拒絕整個檔案，實際不會產生 `None`。

## 考慮過的選項與比較

### 快照存哪裡

使用者在這一題選 A：正本只放資料檔，ledger 拿掉沒有讀者的 `cfg_summary`。使用者同時定下讀檔分工：「Notebook與GUI應該都讀data.h5，只有log browser讀Labber檔。」這與 Notebook 現行只讀 Labber 不符，成為後續工作。

### 框架對 Artifact 的認知

使用者提出新方向：「框架應該要有Artifact的認知，內部可能是泛型？或是子類型？然後怎麼存與有什麼內容由adapter決定」。接著拿 Writeback 作比照：「我的想法，這應該與Writeback類似，顯示邏輯需要耦合框架，但同時要保留一定的擴充」。我理解為：框架定義基底與固定子類型（如 `WritebackItem`），子類型的顯示、格式與讀回歸框架，adapter 決定交哪些與內容。

使用者再把兩個功能定位為子模組：「擴充空間，那這兩個功能的定位應該是gui框架下的一個子模塊，對外解耦，對內可擴充。」

我最初把子類型放在 Artifact 層。使用者修正：「Artifact目前，一個Artifact是一個寫入單位，但內部支援多個不同的項目。」子類型因此改放在項目層。一個 Artifact 是一次寫入，例如 T1 的 run；可擴充的是項目種類，今天有量測資料、圖與分析。

### 子模組放在哪一層

使用者說的是「gui框架下的一個子模塊」，但 Artifact 的存檔與讀回也被 Notebook 的 `save_run` 與 workflows 使用。我提了兩層的做法：共用核心放 `experiment`，管項目種類、寫法與讀回；GUI 子模組放 `gui/app/measure`，管存檔中心顯示、追蹤與 adapter 交出介面。使用者選「3A」。Writeback 比照：GUI 子模組收攏子類型、服務與顯示，寫入 executor 留在 `resources.entry`。

### 格式由誰決定

使用者最初的說法是「怎麼存與有什麼內容由adapter決定」。我比較了兩種做法：

| | 做法一：框架核心依項目種類決定格式 | 做法二：框架給路徑，adapter 自己寫 |
| --- | --- | --- |
| Notebook 讀回 | 統一讀 `data.h5` | 只有寫檔的 adapter 讀得懂 |
| Labber | 框架產生一次 | 每個 adapter 各自產生 |
| 失敗隔離、原子替換、版本 | 框架做一次 | 約 52 個 adapter 各做一次 |

使用者：「好，同意做法一」。adapter 只決定交出哪些項目與內容。

### 讀回由誰解讀

我提議框架讀檔、把 `StoredRun` 交給 adapter。使用者回「看起來沒問題」，並問能否用 `NotebookAdapter(exp)` 載入，接受 `StoredRun` 或路徑。我提議：核心實驗提供公開的 `from_stored(stored) -> RunRecord`，作為 Notebook 與 GUI 共用的唯一轉換；Notebook 的 `load` 只收路徑，內部讀 `data.h5` 再呼叫 `from_stored`；GUI adapter 拿到 `RunRecord` 後自己做表單回填。這項沒有逐字裁決，使用者在最後的摘要中一併同意。

### cfg 讀不出來時

我原本建議嚴格：cfg 驗證失敗就拒絕。使用者問了遷移，以及 cfg 無法解析時 `data.h5` 是否就失效。查證後，一次性的 `migrations/` 只轉換能通過 typed 驗證的 cfg，其餘留待處理；風險在於之後的 schema 演進會讓舊檔整個讀不到。我因此改建議「資料優先、設定可缺」：變數、軸、單位不符仍報錯；cfg 驗證失敗回 `cfg=None` 加警告與原因；原始 JSON 留在 `StoredRun`；GUI 不回填、唯讀顯示原始設定；不相容的型別變更升 schema major，必要時才另寫離線遷移。使用者：「我認為可以」。ADR-0063 的「不提供版本遷移鏈」維持。

### 記錄與實作時程

我提議先寫 ADR draft 與本篇，實作等 measure-form 與 cfgschema-retire 落地後再做，並先提實作計畫。使用者：「1同意 2可以」。

## 被否決的選項與理由

- 框架給路徑、adapter 自己寫檔（做法二）：資料只有寫它的 adapter 讀得懂，Notebook 無法統一讀回，每個 adapter 都要產生 Labber，失敗隔離、原子替換與版本管理在約 52 個 adapter 重複。
- 子類型放在 Artifact 層：使用者修正，Artifact 是寫入單位，子類型在項目層。
- cfg 驗證失敗就拒絕整個檔案：schema 演進後舊檔的資料跟著失效。
- Notebook `load` 接受 `Path | StoredRun` 聯集：兩種輸入走同一個 `from_stored`，只收路徑即可。
- 框架在 Load 與 Run 窺探 `result.cfg`：框架不碰 typed cfg，改由核心實驗轉換、adapter 解讀。

## 留下的問題

- workflow 的 `GroupedAxesSpec` builder 讀 cfg snapshot；cfg 缺少時的容忍程度另議。
- Writeback 子模組的確切切分。
- Notebook 單檔 Labber `save()` 的去留。
- ADR-0069 與現況不符的兩處：「Result 只表達資料，不攜帶 cfg_snapshot。」的敘述，以及「RunRecord.cfg 允許 None」與 native Load 實際行為。兩處在 draft 轉正時一併修訂。

## 依據

- 現況程式：`lib/zcu_tools/experiment/run_saving.py`（`SaveArtifact`、`save_artifacts`）、`lib/zcu_tools/resources/entry/save_layout.py`（`ArtifactKey.member`）、`lib/zcu_tools/experiment/native_persistence.py`（`prepare_run_data`、`load_run_with_metadata`）、`lib/zcu_tools/datafile/native_models.py`（`StoredRun`、`CfgSnapshot`）、`lib/zcu_tools/resources/entry/ledger_models.py`（`AcquiredPayload`）、`lib/zcu_tools/gui/app/measure/services/load.py` 與 `services/run.py`、`lib/zcu_tools/gui/app/measure/adapter/types.py`（`WritebackItem`）、`lib/zcu_tools/notebook/adapter.py` 與 `notebook/saving.py`。
- 相關 ADR：[0063](../adr/0063-persistence-ownership.md)、[0069](../adr/0069-experiment-core-and-frontends.md)、[0067](../adr/0067-gui-application.md)、[0072](../adr/0072-module-boundaries.md)。

## 後記（2026-10-15）

本篇「讀回由誰解讀」一節寫 GUI adapter 把 cfg 轉成表單回填，這是錯的。2026-10-13 已裁定 Load 不再填回表單，討論時遺漏了這個前提。使用者 2026-10-15 確認維持該裁定：Load 只載入結果，設定由 Inspect 檢視。修正後的決策見 ADR draft 的「補充裁定」段。

## 轉正後記（2026-10-15）

正式決策由 [ADR-0063](../adr/0063-persistence-ownership.md)、[ADR-0069](../adr/0069-experiment-core-and-frontends.md) 與 [ADR-0067](../adr/0067-gui-application.md) 接替，原 draft 已結案。本篇保留當時的推導，不再作為待實作清單。
