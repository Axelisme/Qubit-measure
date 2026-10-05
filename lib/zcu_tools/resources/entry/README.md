# `zcu_tools.resources.entry` result entry composition

**Last updated:** 2026-10-06，元件定義注入、遞迴視圖與讀檔轉換

`ResultEntry` 組合明確傳入的 Result 與 Database 根目錄。條目名稱與 point label 是安全的單一路徑段，不代表物理量。`setup.yaml` 的 UUID entry_id 是不可變身分。載入驗證 UUID 與 UTC 建立時間，既有 handle 的 refresh 不接受另一個身分。

`create` 建立 setup、points、records 與同名 Database 條目目錄。任一目的地已存在就拒絕。失敗只清理本次建立的條目，不刪 caller 的根目錄或既有資料。`open` 要求完整的新格式條目，不猜測 legacy 格式。

## 文件與 seed

Setup 是元件範本。Point 自帶 kind、元件、來源與 general。`new_point` 複製建立當下 setup 的 components 與 provenance，使用新的 created_at，並建立只有標頭的 module_cfg。General 不從 setup 繼承。Setup 與 point 的修改、引用與快照互不影響。

`SetupView` 與 `PointView` 各綁一個 DocumentStore。屬性讀取只看記憶體快照。`edit()`、單行賦值與 `refresh()` 只讀寫自己的文件。修改多個 point 需逐筆交易，不承諾整批 all-or-nothing。

`new_point(clone_from=...)` 只複製同條目的來源 point 與 module_cfg，不讀 setup，也不複製 records、data 或圖片。目的 point 保留來源的其他 general 資料，更新 created_at。每筆來源記錄標示本次直接來源工作點。成功重新接受欄位會清除該路徑與子欄位的 cloned_from，即使值相同。跨條目 clone 明確拒絕。

## 元件與定義

框架不保存具體 kind 或內建角色清單。`component_registry` 是空的共用 ComponentRegistry，`role_registry` 指向它的 roles。Caller 顯式註冊 model、RoleSpec、shorthand 與預設焦點的 kind patterns。Lib 不 import 定義模組，依賴方向由 ADR-0063 與 import contracts 固定。Import 本身不註冊，重複 bootstrap 報錯。

Registry 只做 kind 查表與 ComponentSchema 子類檢查。重複 kind 報 ValueError，非法 model 報 TypeError，未知 kind 報 UnknownKindError 並附相近名稱。失敗不占用名稱，unregister 後可明確替換。

ComponentSchema 只要求字串 kind。原 model 決定必填欄位、型別、extra 策略、defaults、field validators、model validators 與 post-init hooks。Ext、wiring 與 module 都不是框架特例。Forbid model 的錯字由 Pydantic 拒絕，allow model 的額外欄位可讀寫及 set／meta。沒有值且 default 為 None 的欄位讀取報 AttributeError。

新增元件保留已產生的 defaults，避免 reload 或 seed 重生 factory 值。Seed 也複製當次 template validation 產生、原 YAML 尚未保存的 defaults。框架不檢查 canonical 或冪等，不替定義者修復 validator。

## 視圖與來源

所有 dict 與巢狀 BaseModel 欄位回傳 FieldView。子 view 保留完整路徑，每次讀取都跟隨 owning view 的最新快照。Scalar、null 與 list 回傳獨立 YAML 值，沒有 list index 子 view。Item access 使用單一 literal key，set／meta 使用點分路徑。空 key 或含點號的 key 無法以點分路徑定址，框架不拒絕這些 key，也不提供 escaping。

每次寫入複製完整父元件 candidate，保留 dict 內的 model 型別，再由原完整 model 驗證。Schema 或 body 失敗不污染 draft、來源或磁碟。EditView 的元件、general、子 view 與 set 共用同一份 draft。元件名稱必須是 public identifier，不得遮蔽 view 方法。

值與 provenance 在同一文件、同一交易提交。普通賦值、draft set、加入元件與子 view 寫入記錄 manual 來源和 UTC 時間。重新接受同值也更新來源並清除 clone 標記。`meta` 沒有值或來源時回 None，否則回傳獨立 Provenance。Stderr 保存 caller 提供的工作單位數字，不換算。

`edit_view.set(..., provenance=...)` 先驗證來源的本地引用。非 manual source 必須是本條目 records/ledger.jsonl 中同 ID、同 entry_id 的事件。被引用事件必須有 zcu.ledger 1.x 標頭。較新 minor 可讀且原 bytes 不變。Entry 不寫 ledger，不驗證完整事件 schema，也不從其他條目補來源。

## 保存與版本

DocumentStore 擁有單檔衝突檢查與 atomic replace。檔案、快照與視圖使用相同工作單位。不同 leaf 可合併，同 leaf 衝突拒絕整筆交易。Entry 不提供條目鎖或多檔交易。

Model 的 before validator 可把舊欄位改成新欄位。Open、refresh 與空 edit 不寫檔。非空 commit 在最新 normalized tree 合併與完整驗證，再把序列化差異套回 raw YAML，保留未改節點的引號、註解與順序。保存消費 actual model_dump 輸出，不用原欄位值繞過 serializer。格式轉換不生成 manual provenance。

Forward-minor 先讓 validator 處理完整 input，再保存剩餘未知欄位。Forbid／ignore 的 typed snapshot 隱藏 temporary extras，allow model 保留它們。補回 temporary extras 的 model 位置若不再是 mapping，或 key 已被 serializer 佔用，直接 raise，不猜重塑後的位置。當前 minor 沿用原 model 的 extra 策略，ignore 丟棄的欄位在非空 commit 時移除。

List 是整欄衝突與通知單位。等長編輯按位置保留未改資料。Resize 保留 typed base／draft 的最長相等前綴，替換後綴不沿用舊位置的 future keys。框架不推測元素身分。

## 角色與失敗

`PointView.resolve` 只使用 bound point 的快照。Sequence 取 role registry 的 RoleSpec，Mapping 可覆寫 kind pattern 與 via。未傳角色表時使用 caller 設定的有序 shorthand，沒有設定就報錯。

解析依序使用 explicit、匹配焦點、已解析元件的角色同名欄位或 via。Via 首段是已解析角色，其餘是元件內路徑。路徑的字串值視為目標元件名稱，缺少目標在 resolve 報錯，普通載入不預檢整圖引用。焦點只填一個角色；只有一個元件符合 caller 的 focus kind patterns 時才作預設焦點。

未知角色、多餘 explicit、缺元件、kind 不符、缺引用或多個候選以 RoleResolutionError 回報。RoleView 固定解析當下的名稱映射，元件值仍跟隨 PointView，回傳映射不可修改。

`rename_entry` 移動兩個目錄，不改檔案內容。第二次移動失敗時復原第一次。復原也失敗時，RenameRecoveryError 回報兩個路徑與原始 I/O 原因，不承諾跨檔掉電保證。

本模組尚未接線到 ContextService、notebook runtime、GUI 或 MCP 的參數操作。Ledger producer 與 accept／writeback 服務由後續切片提供。
