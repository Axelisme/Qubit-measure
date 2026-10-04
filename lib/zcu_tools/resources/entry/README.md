# `zcu_tools.resources.entry` — result entry composition

**Last updated:** 2026-10-04，角色解析與 ledger 引用的版本邊界

`ResultEntry` 組合明確傳入的 Result 與 Database 根目錄。條目名稱與 point label 是安全的單一路徑段，不代表物理量或身分。`setup.yaml` 的 UUID `entry_id` 是條目身分，建立後不可變。載入驗證 UUID 與 UTC 建立時間；既有 handle 的 refresh 不接受另一個身分。

`create` 建立 setup、points、records 與同名 Database 條目目錄。任一目的地已存在就拒絕。失敗只清理本次建立的條目，不刪 caller 的根目錄或既有資料。`open` 要求完整的新格式條目，不猜測 legacy 格式。

## 文件與 seed

Setup 是元件範本。Point 是自帶 kind、元件、來源與 general 的完整文件。`new_point` 複製建立當下 setup 的 components 與 provenance，使用新的 created_at，並建立只有標頭的 module_cfg。之後 setup 與 point 的修改、引用與快照互不影響。

`SetupView` 與 `PointView` 各綁一個 DocumentStore。屬性讀取只看記憶體快照。`edit()`、單行賦值與 `refresh()` 只讀寫自己的文件，不重驗其他 point。要修改多個 point，caller 對 `list_points()` 寫迴圈，每個 point 各一筆交易，不承諾整批 all-or-nothing。

`new_point(clone_from=...)` 只複製同條目的來源 point 與 module_cfg，不讀 setup，也不複製 records、data 或圖片。目的 point 保留來源的其他 general 資料，更新 created_at。每筆來源記錄標示本次直接來源工作點。成功重新接受欄位會清除該路徑與子欄位的 cloned_from，即使值相同。失敗的寫入不改 draft 或來源。跨條目 clone 明確拒絕。

DocumentStore 擁有單檔衝突檢查、SI／工作單位換算與 atomic replace。獨立 handles 的不同 leaf 可以合併；同 leaf 衝突會拒絕整筆交易。Entry 不提供多檔交易或條目鎖。

## 元件與驗證

元件名稱是未保留的 public identifier，不允許點號或遮蔽視圖。Setup 與 Point 都提供 `add_component`，以原 registered model 完整驗證必填欄位、型別與 field validators。引用只在同一文件內解析。缺少目標時回報來源檔案、元件、欄位與目標名稱，不發布無效快照。

Registry 在註冊前驗證 ComponentSchema 型別、extra=forbid、數值欄位的 UnitSpec 與引用路徑。物理數值必須宣告單位，無因次量也宣告 1／1。直接或 nullable 巢狀 model 遵守同一規則。Wiring 的 channel 是離散 index，不換算，也不要求物理單位。失敗不占用 kind，重複註冊報錯，unregister 後可明確替換。

Defaults、default_factory 與 field validators 使用 Pydantic 原生行為。加入元件時保留已產生的 default 值，避免重新載入或 seed 重生 factory 值。Seed 也複製當次 template validation 產生、原 YAML 尚未保存的 defaults。已知欄位拒絕拼字錯誤。沒有值且沒有非 null default 的欄位讀取明確報錯；有值的 native default 可以讀取。

D114 讓 register 拒絕所有 model-level validator 與自訂 model_post_init，包含繼承與支援的巢狀宣告。跨欄位檢查本批不支援。不建立 partial model，不跳過原 field validators，也不包裝 caller 的例外。

D104 的 canonical 檢查由 entry 負責。提交前與 open、refresh、交易進入的讀取都檢查驗證前後的工作單位值。改值超出等價界線時，以 ValidationError 回報欄位、前後值與來源檔案，保留原檔與既有快照。D105 只對宣告 UnitSpec、兩邊都是有限 float 的 leaf 使用 math.isclose，rel_tol=1e-12、abs_tol=0.0。其他值嚴格比較。通過後發布驗證後的值，不發布 SI 換算的末位誤差值。DocumentStore 不負責 canonical 等價規則。

內建 kinds 包含 resonator、fluxonium、transmon、JPA 與 current source。內建物理欄位可缺值，單位涵蓋頻率、能量、電流、時間與無因次量。引用保存元件名稱，不展開目標物件。

## 值與來源

值與 provenance 在同一文件、同一交易提交。普通賦值、draft set、加入元件與容器內的寫入都記錄 manual 來源和 UTC 時間。重新接受同值也更新來源，清除該欄位及子欄位的 clone 標記。失敗不發布值或來源。

`meta` 只讀取快照，沒有值或來源時回傳 None。回傳的 Provenance 包含固定來源欄位；clone 記錄是獨立副本。物理 stderr 使用值的 UnitSpec，落盤為 SI，視圖為工作單位。Ext 不換算。

`edit_view.set(..., provenance=...)` 先驗證來源的本地引用。非 manual source 必須是本條目 records/ledger.jsonl 中同 ID、同 entry_id 的事件。被引用事件必須有 zcu.ledger 1.x 標頭，錯誤 format／版本指向 ledger 路徑；較新 minor 可讀且原 bytes 不變。Entry 不寫 ledger，不驗證完整事件 schema，也不從其他條目補來源。這個切片不提供 producer、accept／writeback 服務或 status／stale。

## 角色與焦點

`PointView.resolve` 只使用 bound point 的快照，不查 setup、其他 point 或 GUI session。Caller 宣告有序角色表，名稱必須經 role registry 註冊。Sequence 取 registry 的 RoleSpec，Mapping 可以覆寫 kind pattern 與 via。未傳角色表時，notebook shorthand 只宣告 qubit 與 readout。

解析依序使用 explicit、匹配焦點、已解析元件的同名引用或 via。焦點只填一個角色。Point 只有一個 qubit 時，以它作預設焦點，多 qubit 不猜測。Via 的首段是已解析角色，其餘是該元件註冊的引用路徑，含巢狀欄位。一般字串欄位不當作引用。Notebook 可以註冊 pair 與 coupler，不增加內建 kind。

多餘 explicit、未知角色、缺元件、kind 不符、缺引用或多個候選都以 RoleResolutionError 回報角色、焦點、要求的 kind 與原因。RoleView 固定解析當下的名稱映射，元件值仍經原 PointView 讀写與 refresh。回傳映射不可修改。

## 視圖、版本與失敗

EditView 的元件、wiring、ext、general 與點分 set 共用同一份 draft。欄位驗證失敗不修改 draft；身分、引用或 canonical 驗證失敗使整筆交易不提交。description 未設定時為 None。Ext 保留任意 YAML 值，不換算單位；非屬性形式的 key 可用 item access。巢狀 model 讀取投影成工作單位 YAML mapping，點分 set 可更新已存在的容器。

同 major 的較新 minor 保留未知欄位與原版本。Typed 視圖只投影已知欄位。未知欄位留在 YAML tree，不換算，也不開放 typed API 讀寫。當前 minor 的未知正式欄位仍報錯。

`rename_entry` 移動兩個目錄，不改檔案內容。第二次移動失敗時復原第一次。復原也失敗時，RenameRecoveryError 回報已移動的 Result、預定 Database 目的地、無法復原的 Result 原路徑與兩個原始 I/O 原因。這不是跨檔掉電保證。

本模組尚未接線到 ContextService、notebook caller、GUI 或 MCP。Ledger producer 與 accept／writeback 服務由後續切片提供，不改現行 context 的責任。
