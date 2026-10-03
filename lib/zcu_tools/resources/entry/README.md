# `zcu_tools.resources.entry` — result entry composition

**Last updated:** 2026-10-04 — registry 宣告與部分 model 切片

`ResultEntry` 組合兩個明確傳入的根目錄。名稱是安全的單一路徑段，不代表物理量或身分。`setup.yaml` 的 UUID `entry_id` 是身分，建立後不可變。載入驗證 UUID 與 UTC 建立時間；既有 handle 不接受 refresh 帶入另一個身分。

`ResultEntry.create` 建立新格式 setup、points 與 records 目錄，以及同名 Database 目錄。任一目的地已存在就拒絕。建立失敗只清本次建立的條目目錄，不刪 caller 的根目錄或既有資料。`open` 要求完整的新格式條目，不猜測 legacy 格式。

`SetupView` 與元件、wiring、ext 視圖讀取 DocumentStore 的記憶體快照。單行 description 與元件欄位寫入都使用型別化交易。DocumentStore 擁有單檔衝突檢查、版本、SI／工作單位邊界與 `.entry.lock`；ResultEntry 擁有身分的提交前驗證。

元件名稱是未保留的 public identifier，不能拆成點分路徑或遮蔽視圖。Registry 在註冊前驗證 ComponentSchema 型別、extra=forbid、數值欄位的單位與引用路徑，再保存整份宣告。失敗不占用 kind；重複註冊報錯，unregister 後可明確替換。已知欄位與 wiring 拒絕拼字錯誤，缺物理值的讀取明確報錯。UnitSpec 與比例換算沿用 DocumentStore，不從欄位名稱猜單位。Ext 使用獨立的任意 YAML mapping，沒有換算；非屬性形式的 key 可用 item access。

`rename_entry` 移動兩個目錄，不改檔案內容。第二次移動失敗時復原第一次；復原也失敗則以 `PartialCommitError` 回報已完成、待完成與復原失敗的路徑，並保留兩個原因。這不是跨檔掉電保證。

Setup 允許省略 notebook model 的必填欄位與直接巢狀 model 的必填欄位。未填值不落盤，讀取明確報錯；有提供的值仍經型別驗證。Registry 保留原始完整 model，必填完整性由後續疊合視圖檢查。

本模組尚未接線到 ContextService、notebook caller、GUI 或 MCP。目前元件切片包含 resonator、typed wiring 與 ext。完整內建 kinds、元件引用驗證與完整 draft 視圖仍在實作中。工作點、來源和角色解析由後續切片提供，不改現行 context 的責任。
