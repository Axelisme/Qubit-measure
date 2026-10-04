# `zcu_tools.resources.entry` — result entry composition

**Last updated:** 2026-10-04 — forward-minor typed projection

`ResultEntry` 組合兩個明確傳入的根目錄。名稱是安全的單一路徑段，不代表物理量或身分。`setup.yaml` 的 UUID `entry_id` 是身分，建立後不可變。載入驗證 UUID 與 UTC 建立時間；既有 handle 不接受 refresh 帶入另一個身分。

`ResultEntry.create` 建立新格式 setup、points 與 records 目錄，以及同名 Database 目錄。任一目的地已存在就拒絕。建立失敗只清本次建立的條目目錄，不刪 caller 的根目錄或既有資料。`open` 要求完整的新格式條目，不猜測 legacy 格式。

`SetupView` 與元件、wiring、ext 視圖讀取 DocumentStore 的記憶體快照。單行 description 與元件欄位寫入都使用型別化交易。DocumentStore 擁有單檔衝突檢查、版本、SI／工作單位邊界與 `.entry.lock`；ResultEntry 擁有身分的提交前驗證。

元件名稱是未保留的 public identifier，不能拆成點分路徑或遮蔽視圖。Registry 在註冊前驗證 ComponentSchema 型別、extra=forbid、數值欄位的單位與引用路徑，再保存整份宣告。失敗不占用 kind；重複註冊報錯，unregister 後可明確替換。已知欄位與 wiring 拒絕拼字錯誤，缺物理值的讀取明確報錯。UnitSpec 與比例換算沿用 DocumentStore，不從欄位名稱猜單位。Ext 使用獨立的任意 YAML mapping，沒有換算；非屬性形式的 key 可用 item access。

`rename_entry` 移動兩個目錄，不改檔案內容。第二次移動失敗時復原第一次；復原也失敗則以 `PartialCommitError` 回報已完成、待完成與復原失敗的路徑，並保留兩個原因。這不是跨檔掉電保證。

Setup 允許省略 notebook model 的必填欄位與直接巢狀 model 的必填欄位。未填值不落盤，讀取明確報錯；有提供的值仍經型別驗證。Registry 保留原始完整 model，必填完整性由後續疊合視圖檢查。

內建 kinds 包含 resonator、fluxonium、transmon、JPA 與 current source。物理值可缺省，單位宣告涵蓋頻率、能量、電流、時間與無因次量。引用保存元件名稱，不展開目標物件。載入、refresh 與提交驗證有值的引用，包含 notebook 宣告的巢狀路徑；缺少目標時回報檔案、元件、欄位與目標名稱。失敗不發布無效快照。

EditView 的元件、wiring、ext 與 general 屬性更新同一份 draft。點分 set 也更新這份 draft，不另開交易。型別或欄位驗證失敗不修改 draft；身分或引用驗證失敗使整筆交易不提交。General description 未設定時回傳 None，general.ext 保留任意 YAML 值。Notebook 的巢狀 model 讀取投影成工作單位的 YAML mapping；點分 set 可更新其已存在的容器。

同 major 的較新 minor 文件保留未知欄位與原版本。Typed 視圖只投影已知欄位；頻率與 wiring 時間仍使用宣告的工作單位。未知欄位留在 DocumentStore 的 YAML tree，不換算，也不開放 typed API 讀寫。已知值、kind 與引用仍驗證；當前 minor 的未知正式欄位仍報錯。

本模組尚未接線到 ContextService、notebook caller、GUI 或 MCP。Stderr 接縫與其他 notebook model 形狀仍在實作中。工作點、來源、set 的 provenance 參數和角色解析由後續切片提供，不改現行 context 的責任。
