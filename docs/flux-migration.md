# Component flux 升級至 v3

**讀者：** operator、lab-author、agent。預覽 v1→v2→v3 轉換，修復被拒文件，核對缺名診斷與重新校準。

## 先確認版本與資料範圍

長期入口是 `scripts/upgrade_parameters.py`，處理 native `setup.yaml`、`points/<label>/point.yaml` 與 entry template，預設 dry-run，不操作儀器。
Runtime 與升級目標都是 `schema_version: 3`。Runtime 不讀舊 component 形狀，不自動升級。
先在副本核對轉換，不要手改版本標頭避開檢查。

先看 [資料狀態對照表](../migrations/README.md)。舊 result／Database 走 [storage migration](storage-migration.md)。
舊 record `dev_value/dev_unit` 走同一 CLI 的 [record 範圍](record-device-upgrades.md)，由 operator 宣告 device 名稱。
Component 升級不補寫舊 record 的 `flux_frame` 或 `component`。

## 現行模型

```yaml
schema_version: 3
components:
  Q1:
    kind: qubit/fluxonium
    flux_dev: local_bias
    flux:
      offset: -0.125
      period: {local_bias: 0.8, global_bias: -0.4}
```

這是相關欄位節錄，不是完整可載入文件。
公式為 `φ = offset + Σ V_d／period_d`。Offset 是所有儀器輸出 0 時的 flux，單位 Φ0，不做 mod 1。
Period 使用該 device 原生單位／Φ0，可帶負號。當下沒有值的儀器不貢獻，period 保留。
未校準形狀為 `flux: {offset: null, period: {}}`。
Qubit、mux、JPA 的 `flux_dev` 是主要／預設控制儀器，不參與公式，也不必有校準。

## 保留 v1→v2→v3 升級鏈

### V1 歷史輸入與 v2 中間形狀

V1 qubit 使用 `flux_half`、`flux_int`、`flux_period`、`flux_unit`，來源為 `flux_source`。
Mux 另有 `coil_flux.<qubit>.half/int/period`。
自己的線保留現行數值轉換，在 v2 成為 `flux.<device>.{half, integer, period}`。
Coil 只保留 signed period，在 v2 寫 integer 0、half = period／2，原 half／int 列在報告。
下一步不會把已捨棄的 coil offset 貢獻加回來。
工具不換單位、不推測方向，原 spacing 檢查 `|integer − half| = |period| / 2` 不變。

Native 單檔依下列順序選 v2 主要儀器：

1. Qubit 自己的線。
2. V1 `flux_source` 指名的 coil。
3. 唯一 coil。
4. 多 coil 未指名時不填來源。

最後一種使 v3 offset=null，但全部 signed periods 保留。
報告列出每個來源與原校準的候選 offset，提醒重新校準。
Flux jump 可能使候選 offset 不一致，不因此判腐敗。候選相同也不代表可任選來源。
缺名 local 用 `unnamed_device`，缺名 coil 用 `unnamed_coil`，兩筆可同時保留。
Native 自己的線用 `flux_source: unnamed_device` 指名，最終改為 `flux_dev`。

### Main v2 輸入到 v3

V2 保留每台儀器 `{half, integer, period}` 與 `flux_source`。
其參考條件是其他儀器停在各自 integer，見 [歷史 v2](storage-format-history/component-flux.md)。
V2→v3 以 `flux_source` 為主要儀器，有對應 entry 時：

```text
offset = 0.5 - half_main / period_main - sum_other(integer[o] / period[o])
```

公式保留主要儀器存下的 half 分支，不做 mod 1。
例如：

```yaml
# 歷史 v2 節錄
schema_version: 2
flux_source: Q1_bias
flux:
  Q1_bias: {half: 0.0, integer: -0.001, period: 0.002}
  coil_bias: {half: 0.005, integer: 0.0, period: 0.01}
```

對應 v3：

```yaml
schema_version: 3
flux_dev: Q1_bias
flux:
  offset: 0.5
  period: {Q1_bias: 0.002, coil_bias: 0.01}
```

沒有 `flux_source` 或主要儀器缺 entry 時 offset=null，仍照抄全部 periods，包含正負號。
`flux: {}` 轉為 `{offset: null, period: {}}`。
此步才將 qubit、mux、JPA 的 `flux_source` 改名為 `flux_dev`。
歷史步驟不刪除，也不把 v2 重新定義成 v3。

## Storage 匯入的來源與覆寫

舊 repo 的 `migrate_storage` 共用轉換，但分批匯入有獨立的來源規則。
每批與最終輸出的 `flux_dev` 都只來自明示來源，沒有時留空，form device 留空必填。
明示來源包含完整 overlay 中有效的 setup／mapping 繼承來源，不要求 point 本批重複宣告。
不從單位、校準 entry、live device 或佔位名稱猜來源。
原資料明示的 `unnamed_*` 名稱仍照明示處理，不按字串全域排除。

每批 offset 按完整 overlay 的 v1 結構角色重算。
Qubit 自身欄位是自己的線，mux `coil_flux` 是 coil。順序為自己的線、唯一 coil、否則 null。
若經 v2 再升 v3，主線身分仍取 v1 結構角色，不靠自動佔位 `flux_source`。
無明示來源的 local／coil periods 分別保留在 `unnamed_device`／`unnamed_coil`。
不新增 finalizer，也不讓自動佔位成為後續批次的來源。

Point 這批有完整、有效且明示來源的校準時，取代繼承 offset 與該儀器 period，其他 periods 保留。
同批 local／mux 撞鍵、partial、非法名稱、數值與 spacing 仍拒絕，不改為最後寫入者勝出。

## 核對分類與診斷

未標版文件只有通過完整 v1 分類後才視為 v1。
Current v3 仍須完整驗證，成功顯示 `skipped`，不改寫或再次備份。

Partial 包含混合版本、只搬一半欄位或校準缺任一數字。
工具不選權威值、不用零湊齊。先保存原檔，優先還原完整 v1 或 v2 備份後重跑。
沒有備份時，依原始量測紀錄修復整份文件與版本，再 dry-run。
無法確認的校準可由 operator 決定放棄，留下 v3 未校準形狀，再重新量測。

Corrupt 包含 NaN、infinity、boolean、零 period、非法 device 名稱、spacing 錯誤、mux membership 錯誤或同 device 撞鍵。
缺來源名稱本身不再拒絕，使用佔位並提醒改名。
Upgrade CLI 與 `migrate_storage` 都逐實際目的位置報告 `unnamed_device`／`unnamed_coil`，不只列一次樣板路徑。

離線只驗 device 名稱語法，拒絕 ASCII `.`、空字串與全空白，不 trim 合法名稱。
升級不查 device catalog，成功不代表 device 已存在或接線。
改名前若沒有當下值，該佔位 period 不貢獻。依接線證據改名，不從單位推測。

## Dry-run、備份與 apply

先複製整個 entry 到離線目錄，停止會寫這些文件的工作階段。
範例先指向副本：

```bash
uv run --directory /absolute/checkout --no-sync -- python scripts/upgrade_parameters.py \
  /offline-copy/setup.yaml /offline-copy/points/*/point.yaml
```

Dry-run 不建檔、不建鎖、不備份。
報告逐文件列 `dry-run`、`skipped` 或 `rejected`，再列欄位前後值與 diagnostics。
核對各目的位置、signed periods、offset 分支、dropped coil 值、未知欄位與重校提醒。
任何拒絕回傳 1，全成功或跳過回傳 0，其餘文件不因單一拒絕停止。

確認授權及範圍後才加 `--apply`：

```bash
uv run --directory /absolute/checkout --no-sync -- python scripts/upgrade_parameters.py \
  /offline-copy/setup.yaml /offline-copy/points/*/point.yaml --apply
```

每份合法候選在同名 `.lock` 下重讀，先建原 bytes 的唯一 `.bak`，再以同目錄 temporary 檔原子替換。
報告列出備份。Partial／corrupt 不備份、不替換。備份後替換失敗仍保留備份。
這不是跨文件交易，也不保證掉電時整批一致。重跑時合法 v3 全部顯示 `skipped`。

## 驗證、重新校準與還原

重新開啟副本或明確 `context.reload`，打開使用該 point 的表單，確認 device 選項與必填空白。
舊資料沒有其他線當時的 bias，轉換不是重建量測事實。
Coil 的舊 half／int 已捨棄，正式量測前以 fluxdep 重新校準。

量測 writeback 可獨立接受 signed period 與預設勾選的 offset。
Offset 扣除其他有 period 且有 Run snapshot 值的儀器貢獻，note 也列出值為 0 者。
被掃儀器以擬合 half 覆蓋 snapshot 舊值一次。缺席儀器不貢獻、不列 note。
獨立 app Export params 只更新 EJ／EC／EL 與選定 period，一律保留 component offset。
兩者不是同一個校準入口，receipt 仍為 written／skipped。

還原前停止 writers，另存目前文件與診斷。
將報告所列 `<原檔名>.<唯一識別碼>.bak` 複製回 YAML 原路徑，核對 bytes，再 reopen 或 reload。
備份仍是原版本，runtime 3 可能要求升級。還原只恢復文件，不代表完成升級。
Device 改名與改接線見 [record 指南](record-device-upgrades.md#改名與改接線)，歷史 records 不批次改名。
版本、鎖與失敗語意見 [通用升級指南](parameter-upgrades.md)。
