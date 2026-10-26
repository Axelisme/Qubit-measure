# Record device 升級與改名改接線

**讀者：** operator、lab-author、agent。宣告舊 record 值的 device 歸屬，預覽與套用具名轉換，並處理之後的改名或改接線。

## 範圍與 operator 宣告

資料狀態與其他工具見 [遷移對照表](../migrations/README.md)。Native component 的四類分類、手動補完與還原見 [flux 升級指南](flux-migration.md)，舊 repo 匯入見 [storage migration](storage-migration.md)。這些操作不推測歷史 frame。

Record 歷史位於 `result/<entry>/records/components/<component>.jsonl`。同一檔案可以混有新舊形狀，必須逐列判讀。JSONL 沒有獨立 header 或 row version。

舊 `dev_value`／`dev_unit` 沒有記錄 device 名稱。工具不從檔名、component、setup、單位或目前接線推測名稱。Operator 必須先確認該批舊值來自哪個 device，再用 `--device-name` 宣告。若不同歷史列來自不同 device，不能把整檔都宣告為同一名稱。先保留原檔，按可靠的量測紀錄人工釐清歸屬。

工具只把完整的舊 pair 移到 `devices.<name>.value`／`value_unit`。它不換算值或單位，不合成 `flux_frame` 或 `component`，也不改 component 文件的 `schema_version`。未知欄位、舊 flat frame、列順序與重複 `run_id` 都保留。已具名且不需改動的列連原 bytes 都保留。

Device 名稱不得含 ASCII `.`，不得是空字串或全空白。合法名稱原樣保留，不 trim。`" Bias "` 與 `"Bias"` 是不同名稱。命令報告會印出 `operator-declared device` 與宣告的原字串。

## 預覽、套用與重跑

使用同一長期入口，不另裝一次性 record CLI。給定明確 JSONL 路徑，先執行 dry-run：

```bash
uv run --directory /absolute/checkout --no-sync -- python scripts/upgrade_parameters.py \
  --scope records --device-name yoko_q1 \
  /data/result/sample/records/components/Q1.jsonl
```

多個路徑可以放在同一命令。只有在所有來源檔的舊值都屬於同一 device 時，才使用同一個名稱批次處理。省略 `--scope` 仍是 [component YAML 升級](parameter-upgrades.md)，record 範圍不使用 component 版本步驟。

Dry-run 不寫來源、不建鎖、不備份。報告依檔案列出狀態，再以從 1 起算的實體行號列出 `old`、`new`、`partial` 或 `corrupt`。沒有舊 pair 的合法檔案是 `skipped`。有可轉舊列的合法檔案是 `dry-run`。

先停止會追加這些 histories 的 GUI、MCP 或 notebook 工作階段。核對名稱與所有行的診斷後才套用：

```bash
uv run --directory /absolute/checkout --no-sync -- python scripts/upgrade_parameters.py \
  --scope records --device-name yoko_q1 \
  /data/result/sample/records/components/Q1.jsonl --apply
```

Apply 取得來源的 `.lock`，最多等待 5 秒，並在鎖內重讀整份檔案。合法候選先保存原 bytes，備份檔名為 `<原檔名>.<唯一識別碼>.bak`。完成備份後，工具才用同目錄 temporary 檔原子替換來源。報告列出備份路徑與 `applied`。若替換失敗，報告仍列出已完成的備份。

這把鎖協調 upgrade writers。一般 JSONL append 不參與這把鎖，所以不能一邊量測一邊 apply。這也不是多檔交易或掉電恢復機制。

任一 `partial` 或 `corrupt` 列都使整份檔案 `rejected`，不備份、不替換。其他檔案照常處理。有任何檔案被拒絕，命令回傳 1；全部成功或跳過回傳 0。名稱或命令參數無效時回傳 2，不開始處理檔案。

套用後再跑同一命令，合法的已具名列會是 `new`，整檔顯示 `skipped`。工具不再改寫或備份。檢查時不要用 `read()` 的 row count 當歷史列數，因為它只回傳每個 `run_id` 的最新整列；升級保留的是全部歷史行。

## 四類列形狀與修復

以下以 `--device-name yoko_q1` 為例。分類只檢查舊 pair 與所宣告名稱的目標 pair。每列仍須是 JSON object，且 `run_id` 是非空字串，行末須有 LF。

### Old，完整舊 pair

```json
{"run_id":"run-a","dev_value":0.25,"dev_unit":"mA","flux_int":0.1,"flux_period":0.8,"t1_us":20}
```

`dev_value` 必須是有限、非 boolean 的實數。`dev_unit` 必須是非空且非全空白字串。單位不限制為 A、V 或 native；`mA` 不會自動除以 1000。轉換結果是：

```json
{"run_id":"run-a","devices":{"yoko_q1":{"value":0.25,"value_unit":"mA"}},"flux_int":0.1,"flux_period":0.8,"t1_us":20}
```

若目標 entry 只有 metadata，例如 `{"label":"local line"}`，工具保留 metadata 並合入 pair。其他 device entry 一律不動。它們即使只有一欄或含非有限數值，也不影響這次分類。

### New，沒有舊欄位

```json
{"run_id":"run-b","devices":{"yoko_q1":{"value":0.3,"value_unit":"V"}},"note":"already named"}
```

目標 pair 完整且合法時保留。沒有 device 資料的 opaque 列、只有其他 device 的列、目標只有 metadata 的列，也原樣保留。不要求每列都有 device 值。

### Partial，只有一欄或目標衝突

```json
{"run_id":"run-c","dev_value":0.2}
{"run_id":"run-d","devices":{"yoko_q1":{"value_unit":"V"}}}
{"run_id":"run-e","dev_value":0.2,"dev_unit":"V","devices":{"yoko_q1":{"value":0.2,"value_unit":"V"}}}
```

缺少 pair 的任一欄屬 `partial`。只要舊 pair 的任一欄與目標 `value` 或 `value_unit` 共存，就屬 `partial/conflict`，即使數字相同也不覆寫。

先依原始量測證據補足缺失欄位，或人工確認衝突的哪一份才是權威值。不要猜單位。若決定保留具名 pair，人工移除舊 pair 前先另存原檔。修正後重跑 dry-run，整檔沒有拒絕列才 apply。

### Corrupt，內容不合法

```json
{"run_id":"run-f","dev_value":true,"dev_unit":"V"}
{"run_id":"run-g","devices":{"yoko_q1":{"value":1,"value_unit":" "}}}
{"run_id":"","dev_value":1,"dev_unit":"V"}
```

兩欄都有但值不是有限非 boolean 實數，或單位不是 nonblank 字串，屬 `corrupt`。無法解析的 JSON、非 object 列、無效 `run_id`、未完成尾行，以及非 object 的 `devices` 或目標 entry，也會被拒絕。診斷指出實體行號與未通過的檢查。

保留原 bytes 與診斷。從來源紀錄修復 JSON、值或單位後再 dry-run。不要用隨意數字、空字串或 null 讓檔案勉強通過。

## 備份還原

停止所有讀寫這份歷史的工作階段。先另存目前檔案，再把報告中的 `.bak` 複製回原 JSONL 路徑。不要把備份當另一份 history 一起匯入。

```bash
cp /data/result/sample/records/components/Q1.jsonl /data/Q1.before-restore.jsonl
cp /data/result/sample/records/components/Q1.jsonl.<唯一識別碼>.bak \
  /data/result/sample/records/components/Q1.jsonl
```

備份是升級前的完整 bytes。還原也會還原舊 pair，分析端不會因此自動認得 device 歸屬。需要升級時重新宣告名稱並 dry-run。

## 舊 flat frame 的 explicit fallback

值改成具名不代表補回了歷史 frame。新的 `flux_from_frame` 依序選 explicit `flux`、新的 `flux_frame.<device>.integer/period`、caller fallback。它不自動讀舊 flat `flux_int`／`flux_period`。

若舊列沒有 explicit flux，caller 必須確認當時的 frame 與單位，然後明示 fallback。例如已完成 device 歸屬轉換，而且這批資料的值與 frame 都用 V：

```python
from zcu_tools.resources.record_table import (
    JsonlRecordTable,
    RecordFluxFrame,
    flux_from_frame,
)

records = JsonlRecordTable("/data/result/sample/records/components/Q1.jsonl").read()
flux = flux_from_frame(
    records,
    device_name="yoko_q1",
    fallback_frame=RecordFluxFrame(dev_unit="V", flux_int=0.1, flux_period=0.8),
)
```

Fallback 只適用於沒有新 row frame 的列，且 device 單位必須匹配。不同歷史時段的 frame 不同時，caller 應分開選取資料並傳入各自確認的 fallback，不能套用目前 component 校準取代歷史。工具保留舊 flat 欄位，但不把它們包裝成新 frame。單 device `RecordFluxFrame` 的單位是 A、V 或 native；若原始值用 mA，caller 必須先明確做分析副本的單位換算，不能只改單位標籤。

## 改名與改接線

以下操作適用於已使用具名 `flux` 的 component。它們不屬於 record upgrade，也沒有批次改名工具。

Device 改名時，將 component 的 `flux.<old-name>` key 改成 `flux.<new-name>`，保留該 entry 的校準值。若 qubit 的 `flux_source` 仍指向舊名，同步改為新名，因為 form 用它選預設 device。不要用整份資料的文字搜尋取代去改歷史。

改接線會改變 device 到 qubit 的關係。刪除該 component 的 `flux.<device>` entry，再重新執行對應的 fluxdep 校準。不要把舊校準當成新接線的校準。只刪受影響的 entry，保留其他 device 的校準。

手動修改目前 session 使用的文件後，透過 MCP `context.reload` 重新載入並驗證。Reload 失敗時先修正來源，不繼續用舊畫面的值量測。

Flux 使用處依各自持有的 device 資料檢查，不要求完整 setup 名單。Record 與 writeback 換算使用 Run snapshot，frame 用到但 snapshot 缺值的 device 會指名報錯。Form 對照既有 device 選項，`flux_from_frame` 對照該列 `devices`，缺少使用到的 device 時指出名稱。

歷史 records 保留當時的名稱與 frame，不跟著改名、不批次改寫。分析可以保留不同歷史名稱，按每列當時的 frame 換算；沒有新 frame 的舊列仍由 caller 明示 fallback。

## 相關工具與格式

- [同一長期 CLI 的 component 文件範圍](parameter-upgrades.md) 處理 YAML 版本鏈，不把 record 形狀放進 component 步驟。
- [舊儲存庫匯入](storage-migration.md) 處理舊 result／Database；[migrations 工具](../migrations/README.md) 另有舊樣本 CSV 的匯入範圍。
- [Record device 格式歷程](storage-format-history/record-devices.md) 記錄辨識與保留範圍。[Record owner](../lib/zcu_tools/resources/record_table/README.md) 說明框架公開接縫。
