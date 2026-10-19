# `zcu_tools.resources.entry` result entry composition

**Last updated:** 2026-10-20，entry 分析格式檔目錄

`ResultEntry` 組合明確傳入的 Result 與 Database 根目錄。條目名稱與 point label 是安全的單一路徑段，不代表物理量。`setup.yaml` 的 UUID entry_id 是不可變身分。載入驗證 UUID 與 UTC 建立時間，既有 handle 的 refresh 不接受另一個身分。

`create` 建立 setup、points、records 與同名 Database 條目目錄。任一目的地已存在就拒絕。失敗只清理本次建立的條目，不刪 caller 的根目錄或既有資料。`open` 要求完整的新格式條目，不猜測 legacy 格式。

## 條目 alias 表

`ResultEntry.load_aliases` 與 `save_aliases` 使用具名 `EntryAliases`，保存 entry 的 authored 有序表。`aliases.yaml` 使用 `format: zcu.aliases`，寫固定 `format_version: "1.0"`，讀接受 major 1 的 forward minor。Mapping 保留空字串、原始 `$` link 與順序。空 mapping 是明確空表，只有檔案不存在才回 `None`。格式、結構與非字串值錯誤直接回報，不退回 preset。

Resources 只保存與驗證資料表示。Session 的 AliasTable 擁有語法、backward-link 與使用時解析。儲存先寫 sibling temporary，再原子 replace；失敗保舊檔並清 temporary。這不是 point、setup、ledger 或 memento，也不提供跨檔交易、multiwriter merge 或掉電恢復。

## 可攜 entry 範本

`load_entry_template(path)` 讀取 `format: zcu.entry-template`、`format_version: "1.0"`，接受 major 1 的 forward minor。它回傳具名 `EntryTemplate`，包含有序的 `components` model mapping 與 `EntryAliases`。Caller 必須先註冊 component models。Reader 不建立 entry，不產生 UUID 或時間，不載入 module library。

範本必須有 `components` mapping 與 `aliases`。Components 沿用 point 的名稱、kind 與原 model 欄位驗證。`aliases` 是完整嵌套的 `zcu.aliases` 1.x document，內含 `format`、`format_version` 與 `aliases` mapping。Reader 在讀檔時拒絕錯誤 header、kind、欄位與非字串 alias bindings。Alias 語法、backward links 與解析仍由 GUI Session 的 AliasTable 擁有。`scripts/create_entry.py` 組合 reader、ResultEntry 與 alias 保存能力，產生 GUI 可 Open 的普通 entry。

## 文件與 seed

Setup 是元件範本。Point 自帶 kind、元件、metadata 與 general。General 的固定欄位只有 entry_id、created_at、description 與 ext。Setup 的 entry_id 是條目身分；point 可保留 seed 的 entry_id，但它不代表 point 身分。Caller 可在 general.ext.flux_value／general.ext.flux_unit 記錄 global flux，ext 沿既有 YAML 值契約保存原值，不提供 flux 專用型別、驗證或單位換算。`new_point` 整份複製建立當下的來源 YAML，再更新 created_at。不傳 clone_from 時來源是 setup.yaml，包含 general.ext、metadata 與可接受的 forward-minor 未知欄位，並建立只有標頭的 module_cfg。傳 clone_from 時來源是已發布的 point.yaml，module_cfg 沿既有 library clone 契約處理。New point 的 description=None 保留 seed／clone 描述，字串明確設定描述，包括空字串。設定在本次建立與失敗清理範圍內完成。Setup 與 point 的修改、引用與快照互不影響。

`SetupView` 與 `PointView` 各綁一個 DocumentStore。屬性讀取只看記憶體快照。`edit()`、單行賦值與 `refresh()` 只讀寫自己的文件。修改多個 point 需逐筆交易，不承諾整批 all-or-nothing。

Package functions `snapshot(view)`、`source_path(view)` 與 `document_value(document, path)` 提供隔離的 typed 讀取、綁定文件位置與 logical YAML 值，不讀檔或執行 model hooks。Snapshot 也接受 EditView，複製尚未提交的 draft。Document_value 只讀實際元件或 general 路徑，不展開角色，區分 explicit null 與缺值。`document_meta(document, path)` 回傳本文件該路徑的隔離 opaque metadata，不查值是否存在。`document_unit(document, path)` 只讀固定欄位既有的靜態 schema unit，包含尚未 set 的 optional leaf。直接指向已宣告欄位時，list、mapping 或 null 值不影響 label。Ext、mapping/list 內的子路徑、缺失路徑或無 metadata 的欄位沒有單位。查詢不讀 leaf 值、不執行 schema callback、不推導可變單位，也不換算值。`resolve_path(document, path, roles=..., reserved_names=...)` 只展開已捕捉角色的首段，不檢查後續 leaf 或 library key。Role、component、general 欄位和 expression 保留名稱碰撞時明確報錯，general 欄位必須使用 general namespace。這些能力不加成 view method，保留 snapshot、path 等合法元件名稱。

`document_fields(document)` 列舉 logical data 的隔離 DocumentField。Models／mapping 展開為葉列，list、null 與空 mapping 是單列。Segments 保留 literal keys；有空字串或點號的 row 沒有 editable_path，不以另一條 dotted path 冒充它。Row 的 metadata 是該 editable_path 的隔離 mapping，無法定址的列為 None。列舉不執行 serializers/model hooks、不讀檔或顯示 opaque forward-minor 資料。

PointView 的公開 constructor 必須提供 result_root 與 database_root，加上 source、entry_id，在建構時一次捕捉完整 clone 身分。SetupView 只需 store 與 source。直接建構的 view 與 ResultEntry 回傳的 view 使用相同 clone 規則，不依靠事後補登記。

`new_point(clone_from=...)` 接受同條目 label、`<entry>/<point>` 或相同 Result／Database roots 的 PointView。Clone 重讀已發布的 point 與 module_cfg，不用 view 的舊快照，也不讀目的 setup。目的 point 保留來源的其他 general 資料，更新 created_at。參數值與 metadata 原樣複製，不查 ledger、不匯入事件、不改寫 source，也不加 cloned_from。

`module_library(label, assets=...)` 開啟該 point 的 typed library，綁定本條目 ledger 與 asset capability。它不讀 point、不選 active point，也不建立缺失文件。Library handle 與 clone 共用 library owner 的文件、來源驗證及讀取清理政策。Clone 讀取來源 library 時，可能先逐 byte 備份原檔，再從來源主檔移除壞條目與附屬來源。Clone 只複製保留的條目，並只匯入保留的來源。

跨條目 clone 只依 library 來源表，把每個不同的非 manual source 匯入為一筆本地 import 事件，共用事件只產生一筆 import。Library 來源表改寫為本地 ID，並更新直接 cloned_from。事件保存原行文字，附檔複製到目的 records。Manual 不產生 import；同條目 clone 不改 source。來源條目移走後，本地引用仍可查回。Clone 不複製 data 或圖片，不承諾 point 與 module_cfg 的跨檔 snapshot。失敗清理本次 point，已追加的 import 可以保留。後續 clone 失敗不回滾已完成的來源 library 清理，也不刪除其備份。

## 元件與定義

框架不保存具體 kind 或內建角色清單。`component_registry` 是空的共用 ComponentRegistry，`role_registry` 指向它的 roles。Caller 顯式註冊 model、RoleSpec、shorthand 與預設焦點的 kind patterns。Lib 不 import 定義模組，依賴方向由 ADR-0063 與 import contracts 固定。Import 本身不註冊，重複 bootstrap 報錯。

Registry 只做 kind 查表與 ComponentSchema 子類檢查。重複 kind 報 ValueError，非法 model 報 TypeError，未知 kind 報 UnknownKindError 並附相近名稱。失敗不占用名稱，unregister 後可明確替換。`kinds()` 回傳隔離的註冊順序 tuple，不驗證 model 或執行 factory。

ComponentSchema 只要求字串 kind。原 model 決定必填欄位、型別、extra 策略、defaults、field validators、model validators 與 post-init hooks。Ext、wiring 與 module 都不是框架特例。Forbid model 的錯字由 Pydantic 拒絕，allow model 的額外欄位可讀寫及 set／meta。沒有值且 default 為 None 的欄位讀取報 AttributeError。

`component_field_specs(kind)` 依宣告順序深度優先投影 kind 的 pydantic 欄位，供編輯器使用。每個 FieldSpec 帶 segments、value_kind、required、choices、unit 與 description；巢狀 model 先列 group 再列子欄位，`kind` 不列。投影只讀 class 宣告，不讀值、不建 defaults、不跑 validator 或 I/O。未知 kind 報 UnknownKindError。頂層 text 欄位的 key 若等於沒有 via 的已註冊角色名稱，`reference_role` 填該角色；`role_candidates(document, role)` 依文件順序列出 kind 符合角色 pattern 的元件，供引用欄位的下拉選單使用。這沿用角色解析的既有慣例，不是引用標記，也不檢查目標是否存在。

新增元件保留已產生的 defaults，避免 reload 或 seed 重生 factory 值。Seed 也複製當次 template validation 產生、原 YAML 尚未保存的 defaults。框架不檢查 canonical 或冪等，不替定義者修復 validator。

## 視圖與 metadata

SetupView、PointView 與 EditView 共用的 FieldView 將 `{"__complex__": [re, im]}` 葉值讀成 Python complex，RoleView 也沿同一讀取路徑。其他 dict 與巢狀 BaseModel 欄位回傳 FieldView。子 view 保留完整路徑，每次讀取都跟隨 owning view 的最新快照。Scalar、null 與 list 回傳獨立 YAML 值，沒有 list index 子 view，list 內保留原始 YAML 元素。Malformed tag 由唯一的 core.complex_codec 拒絕，不當作一般子 view。Item access 使用單一 literal key，set／meta 使用點分路徑。空 key 或含點號的 key 無法以點分路徑定址，框架不拒絕這些 key，也不提供 escaping。

手動 scalar 賦值與 EditView.set 將 native complex 委派 core.complex_codec 編回既有 tag，再走原本的 schema 驗證。Container assignments 仍使用既有 YAML/model 契約。Snapshot、document_value、document_fields 與 ParameterContext 等 non-view 路徑保留 raw tag，文件格式不變。每次寫入複製完整父元件 candidate，保留 dict 內的 model 型別，再由 resources 共用的 stage_model 以原完整 model 驗證。Schema 或 body 失敗不污染 draft、metadata 或磁碟。EditView 的元件、general、子 view 與 set 共用同一份 draft。元件名稱必須是 public identifier，不以底線開頭，也不是 Python keyword。Schema 只檢查這些 identifier 規則。View 綁定與 add_component 另拒絕會遮蔽 SetupView、PointView 或 EditView 公開成員的名稱，錯誤指名衝突成員。公開成員聯集由三個 class 自動取得，不維護保留字表。元件的欄位名不受框架保留，因此 wiring、ext、kind、components、entry_id 與 created_at 都可作元件名。

ResultEntry 管理的 store 在 use、refresh 與 edit 時也檢查遮蔽，失敗不發布快照、值或 metadata。直接建構 SetupView／PointView 時，constructor 檢查初次綁定的快照。Caller 自建 store 時，後續 refresh／mutation 的檢查由 caller 的 store validate callback 負責。沒有 callback 的 store 可能發布遮蔽名稱，該元件無法以 view 屬性存取，需用 snapshot 與 document_value 讀取。

Setup 與 point 文件使用 `format: zcu.parameter-container`、`format_version: '2.0'`，值在 components，`metadata` 是 dotted path 到 opaque YAML mapping 的旁表，與 [獨立參數](../README.md#獨立參數) 共用 header、metadata 讀寫與 value staging。舊 major 或含 `provenance` 旁表的文件在 open／use_point 直接拒絕，錯誤含路徑、讀到版本與期望版本；不轉換、不雙讀。

值與 metadata 在同一文件、同一交易提交。普通賦值、draft set、加入元件與子 view 寫入都保留既有 metadata，不產生來源、時間或 clone 標記。`edit().set_meta(path, mapping)` 整份替換該 path 的 metadata，`{}` 清空；不查值、不 deep merge、不解釋 stderr、source 或 comment 等 key。`meta` 回隔離 mapping，缺項回 None。Metadata 的領域意義與更新政策由 caller 擁有，Entry 不查 ledger。

## 樣本歷史

`ResultEntry.samples(component)` 以安全單一路徑段綁定 `samples/<component>.jsonl`，不讀元件 schema、不解析角色，也不建立檔案。`JsonlSampleTable` 擁有只追加的行、每個 run 的最新整行讀取與明示 CSV 匯出；欄位、來源與失敗契約見 [sample table owner](../sample_table/README.md)。

## Records ledger

`ResultEntry.ledger` 綁定條目的 UUID 與 records 目錄。RecordsLedger 擁有 typed 事件、JSON 附檔、完整 schema 驗證與 append-only 儲存。它不推導 run，也不產生 acquire、save 或 accept 領域事件。Caller 提供事件 ID、UTC 時間與 frontend origin。Origin 是 gui、notebook 或 mcp。所有公開型別從 entry package 匯出。

Acquired 只保存 run 身分、tab、experiment、point、roles 與事件時間，不另存 cfg 摘要。Cfg 正本隨 native run data 保存。

每次讀取與追加都持同一 resolved path 的 sidecar lock，不使用 cache。新事件只寫 1.0；讀取 1.0 與較新 1.x 時，envelope 與 kind-specific payload 都忽略不支援欄位，包括舊 cfg_summary。`get` 的 raw event_json 保留原行，讀取不改寫行或附檔。已知欄位錯誤、未知 major／kind 與損壞行仍拒絕，append 仍驗證 caller 的已知欄位。帶 record 的 append 先發布新 JSON 附檔，再追加事件，不修改 caller model 或既有附檔。失敗可以留下孤兒附檔或不完整尾行，不提供 crash recovery。

`ResultEntry.record_analysis(analysis_kind, *, run_ids, summary, record=None)` 記錄跨 run 分析，回傳可供 `ledger.get(id)` 查詢的事件 UUID。事件保留 caller 的分析種類、輸入 runs 順序與 JSON 摘要，origin 為 notebook；不查 run 資料，也不要求工作點。非 None 的 record 交給 ledger 保存與引用，內容格式由 caller 決定。入口只記錄分析，不寫回值；使用者以一般 point 或 setup 編輯寫入值，需要時自行寫 metadata，Entry 不自動連回分析事件。

Accepted 不要求前序事件或 run。Producer 先 append accepted，再提交值與 metadata。Point commit 失敗時 accepted 可以保留，值與 metadata 仍由 DocumentStore 單檔交易控制。Ledger 鎖不延伸到 point；兩者不是跨檔交易。

`writeback` 提供共用 parameter／module／waveform／sample 項目與 `apply_writeback`。SampleWriteback 接受具體元件名與 producer JSON 欄位，補上 captured run_id、point、accepted 時間與事件 source；dev 欄位選填，原值與 caller 單位不轉換。Caller 傳入固定 SavedPoint 的 write target、已解析的具體路徑，以及具名 evidence，包含實驗名稱、run_id、origin 與可選 source。ParameterWriteback 的 metadata 為 None 時保留該 path 既有 metadata，mapping 則在同筆 point commit 整份替換；executor 不加入事件、來源或時間。Module／waveform 項目照舊帶 accepted Provenance 交給 library owner。流程追加一筆 accepted 後沿項目順序各自提交，首錯停止。項目失敗拋 `WritebackApplyError(event_id, applied)`，原始錯誤為 cause，前項與事件保留。Ledger append 失敗原樣傳播，不寫值。Library 在首次執行 library 項目時才開啟，key、family 與持久值驗證仍由 library owner 負責。流程不 import frontend 或 experiment，不建立跨項交易或回復框架。Notebook 與 GUI 共用 executor。GUI context owner 綁定公開 setters 並發布成功前綴；GUI service 標記該前綴 applied 後原樣拋 cause，保持既有 wire 錯誤分類。

## 輸出位置

`ResultEntry.dispersive_path` 與 `t1_curve_path` 分別回傳 Result 條目下的 `dispersive/` 與 `t1_curve/` 目錄。屬性不建立目錄、不讀 component 或 ledger，並拒絕逃出條目的 symlink。Entry 不提供這些檔案的 codec，也不把它們複製到 point。Caller 以 component 名或自選名稱組成 JSON 檔名，再經 [resources 分析格式 owner](../README.md#entry-分析格式檔) 的具名 load/save 讀寫。

`ResultEntry.result_path` 與 `database_path` 是唯讀的條目目錄，不是上層 roots。`SaveLayout` 接收這兩個 exact paths、前端啟動實驗的名稱、opaque run_id、可選 point 與帶 timezone 的 saved_at，集中計算輸出位置。它不建立目錄或檔案。共用 saver 使用這些輸出，entry 不編排保存。`capture_point_storage(view, roles=...)` 從 bound point 的 constructor origin 與 cached snapshot 捕捉 entry 身分、名稱、point、description、exact paths 與 scoped ledger，不 refresh 或重開 ResultEntry。可選的 RoleView 必須綁同一 UUID、point source 與 Result／Database roots，另一個合法 handle 可以使用。錯誤來源即使 roles 為空也拒絕。回傳的 roles 是解析當下名稱的隔離唯讀 mapping，不重新解角色或查目前焦點；省略 roles 時回傳空 mapping。

`ArtifactKey` 用 section、artifact name 與 member kind 定位輸出。Figure 與 analysis 另需 member_name，保留同 artifact 的多成員身分。名稱不承載物理語意，安全路徑段規則共用 entry owner。Figure 檔名先編碼 percent，再編碼 underscore，避免 token 與分隔符混淆。Analysis 使用安全的 member_name 作檔名，producer 傳入 pane 名稱；section／artifact name 不參與 JSON 路徑。同 pane 可以跨 artifact 共用目的地，保存依選集順序覆寫，不做批次內容去重或路徑衝突預檢。Symlink 不能讓目的地離開 entry root。

Data 回傳 run 的 data.h5 與當地日期樹內的 Labber 新檔名 `<experiment>_<id6>_<序號>.hdf5`。實驗名中的 `/` 轉 `_`，其餘沿用安全路徑段驗證。Id6 取 run_id 後六碼，短於六碼時保留全部。序號從 1 起避讓既有檔案。Figure 回傳 run PNG 與 point 或全域 figures 副本。Analysis 回傳 `runs/<run_id>/analysis/<member_name>.json`。`Output` 提供 writer format 與絕對 exact path。Labber suffix 重用 datafile helper，只避讓既有檔案，不原子保留。其他 paths 固定，上層保存流程擁有 replacement 與失敗隔離。

`workflow_roots(result_path=..., database_path=..., run_id=...)` 使用同一組 exact entry paths，回傳 frozen `WorkflowRoots`。Result 是小型 manifest/journal 的 `workflows/<run_id>/`，database 是大量 acquisition data 的對稱目錄。它沿既有安全單段與 symlink escape 規則，只計算 absolute exact paths，不建立目錄或檔案，也不需要 point 或 saved_at。Workflow engine 擁有目錄建立與保存，不借 single acquisition 的 outputs 推導 workflow roots。

`new_run_id` 將帶 timezone 的時間轉為 UTC 時間戳，加上六個小寫 UUID4 hex 字元。預設取 UTC 現在，不從名稱解析時間，也不保證跨程序唯一性。

## 保存與版本

DocumentStore 擁有單檔衝突檢查與 atomic replace。檔案、快照與視圖使用相同工作單位。不同 leaf 可合併，同 leaf 衝突拒絕整筆交易。Entry 不提供條目鎖或多檔交易。

Model 的 before validator 可把舊欄位改成新欄位。Setup／point 的 open、refresh 與空 edit 不寫檔。Library 的讀取清理是例外，依前述政策先備份再改寫。非空 commit 在最新 normalized tree 合併與完整驗證，再把序列化差異套回 raw YAML，保留未改節點的引號、註解與順序。保存消費 actual model_dump 輸出，不用原欄位值繞過 serializer。格式轉換不生成 metadata。

Format 與 format_version 只描述容器結構。檔頭不符或 major 不同時拒絕載入。較新 2.x minor 保留容器層的未知欄位與 metadata 內容，不覆寫元件或 nested model 的 extra 策略。任何 minor 都由原 model 驗證完整元件 input：forbid 拒絕未知欄位，ignore 丟棄它們，allow 保留並公開它們。Ignore 丟棄的欄位在非空 commit 時移除。元件的 defaults、validators 與舊寫法演進由定義者決定，不綁容器版本號。非法元件在 load、refresh 與 commit 驗證時失敗，不發布新的值、metadata 或快照。

容器層剩餘未知欄位的 serialization 位置若不再是 mapping，或 key 已被 serializer 佔用，直接 raise，不猜重塑後的位置。框架不提供版本遷移鏈或相容 loader；離線資料轉換由 caller 明確執行獨立工具。

List 是整欄衝突與通知單位。等長編輯按位置保留未改資料。Resize 保留 typed base／draft 的最長相等前綴，替換後綴不沿用舊位置的 future keys。框架不推測元素身分。

## 角色與失敗

以下是 Notebook `PointView.resolve` 的角色契約。GUI 使用 ContextService 的
同一張元件／device alias 表，Dialog Apply 才保存 entry，普通 set 只改 memory。
公開 `read_context` 不要求 GUI consumer roles 或 kind 檢查。Tab 不固定角色或環境，
Run 記錄當次固定 names。Notebook 不讀 aliases.yaml，MCP temporary overrides 不寫它。

`PointView.resolve` 只使用 bound point 的快照。Sequence 取 role registry 的 RoleSpec，Mapping 可覆寫 kind pattern 與 via。未傳角色表時使用 caller 設定的有序 shorthand，沒有設定就報錯。

解析依序使用 explicit、匹配焦點、已解析元件的角色同名欄位或 via。Via 首段是已解析角色，其餘是元件內路徑。路徑的字串值視為目標元件名稱，缺少目標在 resolve 報錯，普通載入不預檢整圖引用。焦點只填一個角色；RoleRegistry.default_focus 集中選取符合 focus kind patterns 的唯一候選，Notebook roles resolver 與 GUI primary alias 的首次初始化共用唯一候選判定，但 GUI 不使用這套 focus／via 推論角色。

Explicit 的 None 明確不綁該角色，不改用焦點或 via；讀取時指名 RoleResolutionError。省略 explicit key 才允許自動推論。自動推論缺引用或多個候選時省略該角色；via 的來源角色未綁時也省略衍生角色。未知角色、多餘 explicit、明確指定不存在的元件、實際 dangling 引用或 kind 不符仍以 RoleResolutionError 回報。RoleView 保留同次宣告，讀取已宣告未綁角色（包括 hasattr）時指名 RoleResolutionError；未知 attribute 仍為 AttributeError。RoleResolutionError 是 ValueError，工具不能把未綁角色視為不存在的 attribute。它的訊息指名角色與原因，只在有要求 kind 或使用焦點時列出它們；raise 端有更合適的措辭時（例如 session alias 表）以 message 提供完整訊息，`reason` 仍是可比對的短分類。RoleView 固定解析當下的名稱映射與 owning point 來源，元件值仍跟隨 PointView，回傳映射不可修改。來源由 PointView.resolve 捕捉；存檔 caller 只交給 capture_point_storage 驗證，不比較 callback 或讀 private binding。

`rename_entry` 移動兩個目錄，不改檔案內容。第二次移動失敗時復原第一次。復原也失敗時，RenameRecoveryError 回報兩個路徑與原始 I/O 原因，不承諾跨檔掉電保證。

Entry／point 提供隔離的 publication 與 read binding。ResultEntry.list_names 只列含 setup.yaml 的直屬目錄，不解析文件或尋找 legacy 參數。ContextService 與 Notebook 沿公開 capability 讀寫 entry；GUI／MCP accept 由 context owner 接共用 writeback，不私取 point handles。
