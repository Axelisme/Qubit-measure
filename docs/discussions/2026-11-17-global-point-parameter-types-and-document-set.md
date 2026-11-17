---
date: 2026-11-17
status: active
superseded_by: null
related:
  - 2026-11-16-range-find-writeback-upsert.md
  - 2026-11-17-alias-file-cached-observation.md
  - ../adr/0063-persistence-ownership.md
  - ../adr/0067-gui-application.md
  - ../adr/0081-point-data-disk-sync.md
  - ../../lib/zcu_tools/resources/entry/README.md
  - ../../lib/zcu_tools/gui/session/README.md
basis:
  repo: e49630b95d4e2468a4519bfd76eaf933dc87c80e（型別調查）、de8245e3ff4ff958a05663771bf15dc30bbfeaff（文件組調查）、0908a9d7b3a62ae04bdd884758186e3118e807e3（落地候選）
  external: none
---

# Global／point 參數型別收斂與文件組

## 結論

使用者 2026-11-16 裁定：entry 根的那組文件叫 global，「setup」不再當它的名詞或屬性。
Global 與 point 用同一套型別：`ParameterDocument`、`ParameterGeneral`、`ParameterView`。
選哪一份一律用 `point: str | None`，None（預設）是 global。
Focus 由管理層持有：GUI 是 ContextService，Notebook 是 caller。資料物件不帶 global／point 身分。

同日使用者也採用「文件組」：一個 scope 的 `parameters.yaml`、`library.yaml`、`aliases.yaml` 一起切換、一起捕捉，合稱文件組。
Session 的 `DocumentSet` 代表文件組，ContextService 以 `global_set`、`active_set` 持有兩組。
Resources 每種檔一個開檔入口，形狀相同：`entry.parameters(point=None)`、`entry.library(point=None, *, assets=None)`、`entry.aliases(point=None)`。每個入口只讀自己的檔。
Run／Load 的固定捕捉改名 `DocumentSetCapture`，位置值是 `DocumentSetRef(entry_name, point)`。
磁碟格式、header、檔名與路徑不變。

這也回答 [range_find writeback 快照](2026-11-16-range-find-writeback-upsert.md)〈留下的問題〉的「兩種 View 要不要合併」：
`SetupView` 與 `PointView` 已合併成 `ParameterView`，`commit_parameter_updates(view, updates)` 只收這一種。

現行規則由 [ADR-0063〈同一種資料不因 global 或 point 分兩套〉](../adr/0063-persistence-ownership.md#同一種資料不因-global-或-point-分兩套) 與 [ADR-0067〈Session、狀態與協作〉](../adr/0067-gui-application.md) 擁有。

## 問題

參數檔原本有兩組成對型別：`SetupDocument`／`PointDocument`、`SetupGeneral`／`PointGeneral`、`SetupView`／`PointView`。
跟著成對的還有 capture、writeback target、session target（`SavedEntry`／`SavedPoint`）、`ParameterContext.from_entry`／`from_point`，以及 ContextService 的 `setup: bool` 旗標。
使用者：「使用一個中性的View來取代SetupView/PointView，類似的物件也比照辦理，讓setup/point的概念只是管理層類似focus的指標，底層資料結構是一樣的，我會預期這有比較好的實作可讀性與精簡程度。」

調查（基準 `e49630b95`）查到的事實：

- 兩組型別由 agent 在 2026-10-04 為 point 疊在 setup 上的 layering 而分。同日 `de173d3b3` 移除 layering，型別沒有合回。沒有使用者裁決要求兩種型別。
- 兩份檔都叫 `parameters.yaml`，header 都是 `zcu.parameter-container` 2.0，繼承同一個 schema 基底。兩種 General 的欄位與驗證相同，所以收斂不改序列化、不需遷移。
- 兩種 View 的公開成員除了建構參數與 `PointView.resolve` 之外相同。Point 多帶的 roots 存在私有 binding，用意是不占合法元件名：View 以屬性存取元件，新增公開成員會縮小合法元件名。
- ContextService 已經用「target 有沒有 point」決定開哪一份，型別分流在別處重做了一次：`clone_from` 的 isinstance、migration manifest 依 View 型別定文件種類。
- 同一組 entry 層文件有 setup、entry、Global、Entry template 四個名字。GUI 的 Destination 選單在 global 模式下列兩項，指向同一份檔。
- 影響面：六個型別名 372 處、52 檔（16 產品、36 測試）。

使用者追加：「目前的使用需求與功能是，這三個檔是一起切換的，捕捉也是一起捕捉的，值得給這三個的合併一個名字，而這三個的相關方法也要一致」。
調查（基準 `de8245e3f`）查到：

- ContextService 用六個平行欄位存兩組文件，「有 point 取 point，否則取 setup」的挑選寫在多處。
- ResultEntry 有 7 個開檔入口：`setup`、`use_point(label)`、`module_library(label, assets=)` 與 4 個 aliases 函式。Refresh 的回傳與 revision 的取法三種檔各不相同。
- 逐檔開檔、互不依賴是刻意的：開 library 不讀 parameters，aliases 缺檔不擋其他檔，migration 單獨讀寫 library，Notebook 只用參數 view。
- 開檔呼叫的候選池有 142 檔（23 非測試、119 測試）。

## 考慮過的選項與比較

型別問題：

- **Keep**：維持兩組型別，每處同時處理兩者的地方寫聯集。
- **窄 Protocol**：只為 `commit_parameter_updates` 加一個兩種 View 都滿足的 Protocol。
- **Q**：只合併 View，Document 與 General 維持兩組。
- **提案 A（採用）**：Document、General、View 各收成一個；capture、writeback target、session target、`ParameterContext` 的 factory 也各收成一個；管理層旗標改成 `point: str | None`；「Setup」視窗改名「Entry」。
  `ParameterView` 的公開成員是兩種 View 的聯集，不新增成員。身分檢查分三處，各比完整來源，不以 label 取代。
- **View 帶公開 `point` 屬性**：讓物件自己知道是 global 還是 point。
- **P′**：把 `resolve` 移出 View。
- **名稱用 Entry**：把 global 那組叫 entry。
- **只做 resources 層**：GUI 與 session 維持現狀。

文件組問題：

- **Keep**：只做型別收斂，不給三個檔合名。
- **N**：統一名字與開檔入口，ContextService 的六個欄位不動。
- **B′（採用）**：session 的 `DocumentSet` 代表已安裝的一組，ContextService 換成 `global_set`／`active_set`；resources 每種檔一個形狀相同的入口；revision 一律 `document_revision(x)`，refresh 一律 `refresh(*, strict=False) -> bool`。
- **S-eager**：resources 加 `entry.document_set(point)`，一次開三檔，移除逐檔入口。
- **S-lazy**：resources 加延遲開檔的文件組，與逐檔入口並存。
- 名稱另比過 DocumentGroup、ActiveDocuments、WorkingSet、Scope、Workspace、Environment、Context；位置值另比過 `DocumentSetTarget`、`ContextTarget`。

實作時的後續裁定（designer 在使用者授權下於 2026-11-17 代裁，下列第 1、2 點待使用者追認）：

1. **Focus 切換的簽名**：`use_point(point: str | None = None)` 取代 `use_point(PointUseRequest)` 與 `use_global()`，刪除只有一欄的 `PointUseRequest`。另一選項是保留 `PointUseRequest | None`。`point.use` wire 不變。
2. **Entry 視窗的程式識別**：可見名稱已是 Entry，內部的 `SetupDialog`、`SetupControlPort`、`get_setup_preferences`、`DialogName.SETUP` 一併改成 Entry 系列與 `DialogPrefill`；遠端 `open_dialogs` 值由 `"setup"` 改 `"entry"`，wire 版本由 139 升到 140。
   另兩個選項是整組保留並記成獨立概念，或只改 Python 名、保留 wire 值。
   語義邊界：setup 只留給「啟動時準備 session」（`StartupSetup`、`ctx.setup`、`device_setup`）、一般英文與持久格式 token。
3. **Migration 的持久 token**：manifest 與 `migration-initialization.json`（格式 1.0）用 `"setup"` 表示 global 參數文件。這個值與讀寫它的 literal 保留，定義處加註解；說明文字、區域名稱與 README 改 global。要改 token 得另開格式升版。
4. **歷史討論快照**：舊名清除不含 `docs/discussions/`。舊快照不改正文、不加後記，由本篇以 `related` 回答它留下的問題。

先例與立場：

- ADR-0063（entry 與 point 各存完整獨立文件、inherit／empty／clone、point 身分）：遵循。它約束行為與資料身分，沒有要求兩種型別。
- ADR-0067（ContextService 持有選擇、整組切換、不逐名稱回退）、ADR-0081（同一套 revision 與 watcher）、ADR-0008（Apply 寫當下作用中文件組）、ADR-0037（固定 capture、缺值報錯）：遵循。
- ADR-0072 與 repo 規則〈抽象要有現存的理由〉：遵循。型別收斂只刪類別；`DocumentSet` 集中隱藏六個平行欄位與重複的挑選邏輯。
- ADR-0075（MCP 不追隨 GUI dialog focus）：區別。這裡的 focus 是文件選擇，不是 dialog 焦點。
- entry README「來源放私有 binding、不占合法元件名」：遵循，`ParameterView` 因此不新增公開成員。
- ModuleLibrary 與 aliases 原本就是一種型別加 point 選檔：遵循，參數檔照此收斂。
- Resources 逐檔開檔、互不依賴：遵循，否決 S-eager 的理由。
- 2026-10-04 agent 為 layering 分出的兩組型別：推翻。理由隨 layering 移除消失，依它寫的型別分流交回 owner。

## 被否決的選項與理由

- Keep：與使用者要求相反，每處仍寫聯集。
- 窄 Protocol：使用者 2026-11-16 已拒絕在 light change 只為一個函式加局部 Protocol。
- Q：Document 與 General 本來就相同，留著重複沒有理由。
- View 帶公開 `point`：縮小合法元件名（元件可以叫 `point`），磁碟零遷移不再無條件成立。
- P′：要改 lab 教學與 Notebook 用法，沒有對應好處；`resolve` 已是保留名。
- 名稱用 Entry：entry 同時指含所有 point 的 Result entry，會混。
- 只做 resources 層：使用者選整份。
- 文件組 Keep 與 N：六個平行欄位與挑選邏輯留著，不滿足「給這三個的合併一個名字」。
- S-eager：單檔操作要多讀另兩檔，壞的 aliases 會擋住 library 操作；`assets` 注入沒有位置；Notebook 與 migration 不需要整組。
- S-lazy：resources 多一個除了 session 沒有使用者的類別，整組的准入與 watcher 政策仍在 session。Notebook 或第二個整組使用者出現時再提。
- 名稱 DocumentGroup（library 的 group 另指分區）、ActiveDocuments（active 指 focus）、WorkingSet（與 working point 混）、Scope／Workspace／Environment／Context（已有別義）。
- 位置值 `DocumentSetTarget`／`ContextTarget`：repo 的「Target」指會寫檔的物件，「Context」語意太寬。
- 保留 `PointUseRequest`：wrapper 沒有驗證、第二欄或第二種語意，選檔形狀也與 `point: str | None` 不同。
- 保留 Setup 系列識別：這些名字指的就是 Entry 視窗與 entry／point 管理，讀者得在兩個名字之間對照。
- 把 migration token 改成 `"global"`：改了磁碟格式，舊 manifest 續跑會失效。

## 留下的問題

- Aliases 存檔沒有樂觀鎖，library 有。改成相同是行為改動，另案處理。
- Resources 層的整組物件（S-lazy）等第二個整組使用者出現再提。
- Migration 格式 1.0 的 `"setup"` token 要改，需另開格式升版與舊 manifest 處理。
- 上列實作裁定第 1、2 點待使用者追認；推翻時改 status 並指向新紀錄。

## 依據

- 型別調查基準 `e49630b95`，文件組調查基準 `de8245e3f`（兩者皆在 main 歷史中）。調查工作檔未入庫，結論與數字寫在本文。
- 現行行為：[resources.entry README](../../lib/zcu_tools/resources/entry/README.md)、[gui.session README](../../lib/zcu_tools/gui/session/README.md)、[migration README](../../migrations/storage_migration/README.md)、[measure remote README](../../lib/zcu_tools/gui/app/measure/remote/README.md)。
- 相關 ADR：[ADR-0063](../adr/0063-persistence-ownership.md)、[ADR-0067](../adr/0067-gui-application.md)、[ADR-0081](../adr/0081-point-data-disk-sync.md)。
