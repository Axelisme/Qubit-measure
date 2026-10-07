# `zcu_tools.resources.entry` result entry composition

**Last updated:** 2026-10-07，logical document projection、point storage capture 與 workflow roots

`ResultEntry` 組合明確傳入的 Result 與 Database 根目錄。條目名稱與 point label 是安全的單一路徑段，不代表物理量。`setup.yaml` 的 UUID entry_id 是不可變身分。載入驗證 UUID 與 UTC 建立時間，既有 handle 的 refresh 不接受另一個身分。

`create` 建立 setup、points、records 與同名 Database 條目目錄。任一目的地已存在就拒絕。失敗只清理本次建立的條目，不刪 caller 的根目錄或既有資料。`open` 要求完整的新格式條目，不猜測 legacy 格式。

## 文件與 seed

Setup 是元件範本。Point 自帶 kind、元件、來源與 general。`new_point` 複製建立當下 setup 的 components 與 provenance，使用新的 created_at，並建立只有標頭的 module_cfg。General 不從 setup 繼承。New point 的 description=None 保留 seed／clone 描述，字串明確設定描述，包括空字串。設定在本次建立與失敗清理範圍內完成。Setup 與 point 的修改、引用與快照互不影響。

`SetupView` 與 `PointView` 各綁一個 DocumentStore。屬性讀取只看記憶體快照。`edit()`、單行賦值與 `refresh()` 只讀寫自己的文件。修改多個 point 需逐筆交易，不承諾整批 all-or-nothing。

Package functions `snapshot(view)`、`source_path(view)` 與 `document_value(document, path)` 提供隔離的 typed 讀取、綁定文件位置與 logical YAML 值，不讀檔或執行 model hooks。Snapshot 也接受 EditView，複製尚未提交的 draft。Document_value 只讀實際元件或 general 路徑，不展開角色，區分 explicit null 與缺值。`document_meta(document, path)` 投影本文件的來源。`resolve_path(document, path, roles=..., reserved_names=...)` 只展開已捕捉角色的首段，不檢查後續 leaf 或 library key。Role、component、general 欄位和 expression 保留名稱碰撞時明確報錯，general 欄位必須使用 general namespace。這些能力不加成 view method，保留 snapshot、path 等合法元件名稱。

`document_fields(document)` 列舉 logical data 的隔離 DocumentField。Models／mapping 展開為葉列，list、null 與空 mapping 是單列。Segments 保留 literal keys；有空字串或點號的 row 沒有 editable_path，不以另一條 dotted path 冒充它。Rows 保留完整 provenance 和 cloned_from，不執行 serializers/model hooks、不讀檔或顯示 opaque forward-minor 資料。

`manual_update(document, path, value, stderr=...)` 複製完整記憶體 point，重用 view 的欄位驗證與 manual 來源規則，不建立 store、讀檔或查 ledger。`accept_source(edit, path, provenance=...)` 在已準備的 draft 值上附加本地來源，處理 container leaves 並清除 clone 標記，不重跑值的 validator 或 model hook。來源驗證失敗不改 draft。兩個能力都以 package function 提供，不保留新的元件名稱。

PointView 的公開 constructor 必須提供 result_root 與 database_root，加上 source、entry_id，在建構時一次捕捉完整來源身分。直接建構的 view 與 ResultEntry 回傳的 view 使用相同 clone 規則，不依靠事後補登記。

`new_point(clone_from=...)` 接受同條目 label、`<entry>/<point>` 或相同 Result／Database roots 的 PointView。Clone 重讀已發布的 point 與 module_cfg，不用 view 的舊快照，也不讀目的 setup。目的 point 保留來源的其他 general 資料，更新 created_at。每筆來源記錄標示本次直接來源工作點。成功重新接受欄位會清除該路徑與子欄位的 cloned_from，即使值相同。

`module_library(label, assets=...)` 開啟該 point 的 typed library，綁定本條目 ledger 與 asset capability。它不讀 point、不選 active point，也不建立缺失文件。Library handle 與 clone 共用 library owner 的文件及來源驗證。

跨條目 clone 取 point 與 library 的來源聯集，把每個不同的非 manual source 匯入為一筆本地 import 事件。Library-only 事件也匯入，共用事件只產生一筆 import。兩份來源表共用本地 ID mapping，並更新直接 cloned_from。事件保存原行文字，附檔複製到目的 records，來源表只改 source 與 cloned_from。Manual 不產生 import；同條目 clone 不改 source。來源條目移走後，本地引用仍可查回。Clone 不複製 data 或圖片，不承諾 point 與 module_cfg 的跨檔 snapshot。失敗清理本次 point，已追加的 import 可以保留。

## 元件與定義

框架不保存具體 kind 或內建角色清單。`component_registry` 是空的共用 ComponentRegistry，`role_registry` 指向它的 roles。Caller 顯式註冊 model、RoleSpec、shorthand 與預設焦點的 kind patterns。Lib 不 import 定義模組，依賴方向由 ADR-0063 與 import contracts 固定。Import 本身不註冊，重複 bootstrap 報錯。

Registry 只做 kind 查表與 ComponentSchema 子類檢查。重複 kind 報 ValueError，非法 model 報 TypeError，未知 kind 報 UnknownKindError 並附相近名稱。失敗不占用名稱，unregister 後可明確替換。`kinds()` 回傳隔離的註冊順序 tuple，不驗證 model 或執行 factory。

ComponentSchema 只要求字串 kind。原 model 決定必填欄位、型別、extra 策略、defaults、field validators、model validators 與 post-init hooks。Ext、wiring 與 module 都不是框架特例。Forbid model 的錯字由 Pydantic 拒絕，allow model 的額外欄位可讀寫及 set／meta。沒有值且 default 為 None 的欄位讀取報 AttributeError。

新增元件保留已產生的 defaults，避免 reload 或 seed 重生 factory 值。Seed 也複製當次 template validation 產生、原 YAML 尚未保存的 defaults。框架不檢查 canonical 或冪等，不替定義者修復 validator。

## 視圖與來源

所有 dict 與巢狀 BaseModel 欄位回傳 FieldView。子 view 保留完整路徑，每次讀取都跟隨 owning view 的最新快照。Scalar、null 與 list 回傳獨立 YAML 值，沒有 list index 子 view。Item access 使用單一 literal key，set／meta 使用點分路徑。空 key 或含點號的 key 無法以點分路徑定址，框架不拒絕這些 key，也不提供 escaping。

每次寫入複製完整父元件 candidate，保留 dict 內的 model 型別，再由原完整 model 驗證。Schema 或 body 失敗不污染 draft、來源或磁碟。EditView 的元件、general、子 view 與 set 共用同一份 draft。元件名稱必須是 public identifier，不得遮蔽 view 方法。

值與 provenance 在同一文件、同一交易提交。普通賦值、draft set、加入元件與子 view 寫入記錄 manual 來源和 UTC 時間。重新接受同值也更新來源並清除 clone 標記。`meta` 沒有值或來源時回 None，否則回傳獨立 Provenance。Stderr 保存 caller 提供的工作單位數字，不換算。

`edit_view.set(..., provenance=...)` 透過 RecordsLedger.get 驗證來源的本地事件與附檔。非 manual source 必須是本條目同 ID、同 entry_id 的合法事件。未知 ID 轉成 ValueError 並保留 KeyError cause；格式與 I/O 錯誤傳播。Manual 不讀 ledger。

## Records ledger

`ResultEntry.ledger` 綁定條目的 UUID 與 records 目錄。RecordsLedger 擁有 typed 事件、JSON 附檔、完整 schema 驗證與 append-only 儲存。它不推導 run，也不產生 acquire、save 或 accept 領域事件。Caller 提供事件 ID、UTC 時間與 operation origin；MCP 另需 call_id。所有公開型別從 entry package 匯出。

每次讀取與追加都持同一 resolved path 的 sidecar lock，不使用 cache。新事件只寫 1.0；較新 1.x 行可讀，原行與未知欄位保持不變。帶 record 的 append 先發布新 JSON 附檔，再追加事件，不修改 caller model 或既有附檔。讀取不跳過損壞行。失敗可以留下孤兒附檔或不完整尾行，不提供 crash recovery。

Accepted 不要求前序事件或 run。Producer 先 append accepted，再提交值與來源。Point commit 失敗時 accepted 可以保留，值與來源仍由 DocumentStore 單檔交易控制。Ledger 鎖不延伸到 point；兩者不是跨檔交易。

## 輸出位置

`ResultEntry.result_path` 與 `database_path` 是唯讀的條目目錄，不是上層 roots。`SaveLayout` 接收這兩個 exact paths、opaque run_id、可選 point 與帶 timezone 的 saved_at，集中計算輸出位置。它不建立目錄或檔案。共用 saver 使用這些輸出，entry 不編排保存。`capture_point_storage(view)` 從 bound point 的 constructor origin 與 cached snapshot 捕捉 entry 身分、名稱、point、description、exact paths 與 scoped ledger，不 refresh 或重開 ResultEntry。

`ArtifactKey` 用 section、artifact name 與 member kind 定位輸出。Figure 與 analysis 另需 member_name，保留同 artifact 的多成員身分。名稱不承載物理語意，安全路徑段規則共用 entry owner。Figure 檔名先編碼 percent，再編碼 underscore，避免 token 與分隔符混淆。Analysis 使用安全的 member_name 作檔名，producer 傳入 pane 名稱；section／artifact name 不參與 JSON 路徑。同 pane 可以跨 artifact 共用目的地，保存依選集順序覆寫，不做批次內容去重或路徑衝突預檢。Symlink 不能讓目的地離開 entry root。

Data 回傳 run 的 data.h5 與當地日期樹內的 Labber 新檔名。Figure 回傳 run PNG 與 point 或全域 figures 副本。Analysis 回傳 `runs/<run_id>/analysis/<member_name>.json`。`Output` 提供 writer format 與絕對 exact path。Labber suffix 重用 datafile helper，只避讓既有檔案，不原子保留。其他 paths 固定，上層保存流程擁有 replacement 與失敗隔離。

`workflow_roots(result_path=..., database_path=..., run_id=...)` 使用同一組 exact entry paths，回傳 frozen `WorkflowRoots`。Result 是小型 manifest/journal 的 `workflows/<run_id>/`，database 是大量 acquisition data 的對稱目錄。它沿既有安全單段與 symlink escape 規則，只計算 absolute exact paths，不建立目錄或檔案，也不需要 point 或 saved_at。Workflow engine 擁有目錄建立與保存，不借 single acquisition 的 outputs 推導 workflow roots。

`new_run_id` 將帶 timezone 的時間轉為 UTC 時間戳，加上六個小寫 UUID4 hex 字元。預設取 UTC 現在，不從名稱解析時間，也不保證跨程序唯一性。

## 保存與版本

DocumentStore 擁有單檔衝突檢查與 atomic replace。檔案、快照與視圖使用相同工作單位。不同 leaf 可合併，同 leaf 衝突拒絕整筆交易。Entry 不提供條目鎖或多檔交易。

Model 的 before validator 可把舊欄位改成新欄位。Open、refresh 與空 edit 不寫檔。非空 commit 在最新 normalized tree 合併與完整驗證，再把序列化差異套回 raw YAML，保留未改節點的引號、註解與順序。保存消費 actual model_dump 輸出，不用原欄位值繞過 serializer。格式轉換不生成 manual provenance。

Forward-minor 先讓 validator 處理完整 input，再保存剩餘未知欄位。Forbid／ignore 的 typed snapshot 隱藏 temporary extras，allow model 保留它們。補回 temporary extras 的 model 位置若不再是 mapping，或 key 已被 serializer 佔用，直接 raise，不猜重塑後的位置。當前 minor 沿用原 model 的 extra 策略，ignore 丟棄的欄位在非空 commit 時移除。

List 是整欄衝突與通知單位。等長編輯按位置保留未改資料。Resize 保留 typed base／draft 的最長相等前綴，替換後綴不沿用舊位置的 future keys。框架不推測元素身分。

## 角色與失敗

`PointView.resolve` 只使用 bound point 的快照。Sequence 取 role registry 的 RoleSpec，Mapping 可覆寫 kind pattern 與 via。未傳角色表時使用 caller 設定的有序 shorthand，沒有設定就報錯。

解析依序使用 explicit、匹配焦點、已解析元件的角色同名欄位或 via。Via 首段是已解析角色，其餘是元件內路徑。路徑的字串值視為目標元件名稱，缺少目標在 resolve 報錯，普通載入不預檢整圖引用。焦點只填一個角色；RoleRegistry.default_focus 集中選取符合 focus kind patterns 的唯一候選，roles resolver 與 session focus 共用它。

未知角色、多餘 explicit、缺元件、kind 不符、缺引用或多個候選以 RoleResolutionError 回報。RoleView 固定解析當下的名稱映射，元件值仍跟隨 PointView，回傳映射不可修改。

`rename_entry` 移動兩個目錄，不改檔案內容。第二次移動失敗時復原第一次。復原也失敗時，RenameRecoveryError 回報兩個路徑與原始 I/O 原因，不承諾跨檔掉電保證。

Entry／point 提供隔離的 publication 與 read binding。ResultEntry.list_names 只列含 setup.yaml 的直屬目錄，不解析文件或尋找 legacy 參數。ContextService、Notebook runtime、完整 GUI／MCP 參數操作、Ledger producer 與 accept／writeback 仍由後續切片接線。
