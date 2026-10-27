# Component flux v1→v2 升級

**讀者：** operator、lab-author、agent。判讀舊 flux 校準，預覽升級、修復被拒文件，並從備份還原。

## 先確認版本與資料範圍

長期入口是 `scripts/upgrade_parameters.py`。它處理 native `setup.yaml`、`points/<label>/point.yaml` 與 entry template，預設 dry-run。它不操作儀器。

目前 runtime 與離線升級目標都是 `schema_version: 2`。Runtime 不再讀舊 component flux 欄位。先在副本核對轉換，再安排正式資料升級。不要手改版本標頭避開檢查。

不同資料使用不同入口，先看 [資料狀態對照表](../migrations/README.md)。舊 result／Database 先走 [storage migration](storage-migration.md)。舊 record 的 `dev_value/dev_unit` 走同一 CLI 的 [record 範圍](record-device-upgrades.md)，由 operator 明示 device 名稱。Component 升級不補寫舊 record 的 `flux_frame` 或 `component`。

## 欄位怎麼變

| 舊欄位 | 新欄位 |
| --- | --- |
| Qubit 的 `flux_half`、`flux_int`、`flux_period` | `flux.<qubit.flux_dev>.half`、`integer`、`period` |
| Mux group 的 `coil_flux.<qubit>.half/int/period` | 該 qubit 的 `flux.<mux_group.flux_dev>.half/integer/period` |
| Qubit 的 `flux_unit` | 移除，使用時取 device info 的原生單位 |

v1 的 `flux_source` 改名為 v2 的 `flux_dev`，值原樣保留。Qubit、mux 與 JPA 都使用這個新欄位。它指定主要（預設）控制儀器，不參與 φ 計算，也不必有 flux 校準。Key 是實際 device 名稱，不固定為 `coil`。Mux 的來源若叫 `global_bias`，目標就是 `flux.global_bias`。數字不換單位、不推測方向，負 `period` 也原樣保留。

舊校準沒有保存其他線當時的 bias。轉換把其他 contributing devices 位於各自 integer 當作參考點，這是**假設**，不是重建出的量測事實。報告會提醒重新校準。正式量測前，用 fluxdep 在已知其他線 bias 的條件下更新 frame。

離線只驗 device 名稱語法，不含 ASCII `.`，也不能是空字串或全空白。合法名稱不 trim。文件沒有 device catalog，因此升級成功不代表 device 存在或已接線。未知名稱要到 GUI 打開 point 的表單等使用處才會發現。核對選項、接線與 Run snapshot，再量測。

## 四類形狀

以下是合成 point，數字使用各 device 的原生單位。範例沒有來自真實量測的資料。

### Legacy，完整舊格式

```yaml
format: zcu.parameter-container
format_version: "2.0"
schema_version: 1
general:
  created_at: "2025-01-01T00:00:00Z"
components:
  Q1:
    kind: qubit/fluxonium
    EJ: 4.0
    EC: 1.0
    EL: 0.8
    resonator: R1
    flux_source: local_bias
    flux_unit: V
    flux_half: 0.5
    flux_int: 0.1
    flux_period: 0.8
  R1:
    kind: resonator
    wiring: {ch: 0, ro_ch: 0}
  MG:
    kind: group/mux
    wiring: {}
    resonators: [R1]
    qubits: [Q1]
    flux_source: global_bias
    coil_flux:
      Q1: {half: -0.2, int: 0.0, period: -0.4}
```

工具檢查整份容器、aliases、註冊的 component 與校準。未標 `schema_version` 的文件只有通過完整舊格式分類後才視為 v1，不需要 operator 先補標頭。沒有任何校準值的舊 qubit 可以轉成 `flux: {}`，不憑空補數字。

上例 dry-run 列出 local 與 mux 的欄位搬移、舊欄位刪除與版本變更。`setup.yaml` 還需要合法的 `general.entry_id`。Entry template 使用自己的 `zcu.entry-template` header 與完整 aliases 文件，不加 point general。

### Current，完整新格式

上例成功轉換後，保留 R1、其他 qubit 欄位與 mux 綁定，flux 部分變成：

```yaml
schema_version: 2
components:
  Q1:
    kind: qubit/fluxonium
    flux_dev: local_bias
    flux:
      local_bias: {half: 0.5, integer: 0.1, period: 0.8}
      global_bias: {half: -0.2, integer: 0.0, period: -0.4}
  MG:
    kind: group/mux
    flux_dev: global_bias
    # resonators、qubits、wiring 保留，coil_flux 已移除。
```

這是變更欄位的節錄，不是可獨立載入的完整文件。Current 文件仍要通過完整驗證，成功後顯示 `skipped`，不重寫、不再備份。

### Partial，移到一半

例如在完整 legacy 文件的 Q1 加入以下區塊，但仍保留 `flux_half/int/period` 或 mux `coil_flux`：

```yaml
flux:
  local_bias: {half: 0.5, integer: 0.1, period: 0.8}
```

整份文件混有舊校準與新 entries，會被拒絕。`schema_version: 2` 仍帶舊欄位、舊文件只有新 entries，或一筆校準缺少任一數字，也屬 partial。工具不選權威值、不覆蓋已有 entry。

先另存原檔。若有完整 v1 備份，建議還原後重跑工具。沒有備份時，逐顆 qubit、逐個 mux 查原始校準紀錄，選擇下列一條路：

1. 還原完整 v1 形狀，移除已核對的重複新 entries，保留所有舊校準，再跑 dry-run。
2. 人工完成整份 v2。把每筆已確認的 local 與 mux 校準放入實際 device key，將 `int` 改為 `integer`，移除所有被取代的欄位。整份核對後才標 2，再用同一命令驗證，應顯示 `skipped`。

未知校準不能填零湊齊。確認沒有可保留的校準時，可以明確刪除該筆未完成校準，留下空 `flux` 後重新量測。這會放棄那筆舊校準，須由 operator 決定。

### Corrupt，檢查不通過或沒有可用來源

把 legacy 範例的 `flux_int` 改成 `0.2`，會違反 `|integer − half| = |period| / 2`。以下情形也會被拒絕：

- 數字是 NaN、infinity、boolean，或 period 為零。
- v1 的 `flux_source: bad.name`，或有校準卻沒有可用的明示來源。
- `coil_flux.Q2` 出現，但 mux 的 `qubits` 沒有 Q2，或 Q2 不是文件內的 qubit。
- 多個 local／mux 校準要寫入同一 qubit 的同一 device entry。
- 缺少必要容器、kind、aliases 或 component 欄位。

診斷列出文件路徑與未通過的檢查。Point 路徑含 label，使用 API 也可明填 `UpgradeSource.point`。先按原始紀錄補正來源名稱、mux membership 或校準數字，不從單位或檔名猜 device。Mux 來源缺失時，工具不猜 `coil`，整份文件不寫入。修正後重跑 dry-run，確認拒絕原因消失。

## Dry-run、備份與 apply

先複製整個 entry 到明確的離線工作目錄。停止會寫入這些文件的工作階段。下列範例路徑指向副本，確認結果後才對正式文件執行：

```bash
uv run --directory /absolute/checkout --no-sync -- python scripts/upgrade_parameters.py \
  /offline-copy/setup.yaml /offline-copy/points/*/point.yaml
```

預設不建檔、不建鎖、不備份。報告按文件列出 `dry-run`、`skipped` 或 `rejected`，再列出 literal field paths 的前後值。Partial／corrupt 出現在 `failed checks` 中，其餘文件繼續。任何拒絕使命令回傳 1，全成功或跳過回傳 0。

逐份核對來源名稱、正負 period、未知欄位與未涉資料是否保留。確認授權及範圍後才加 `--apply`：

```bash
uv run --directory /absolute/checkout --no-sync -- python scripts/upgrade_parameters.py \
  /offline-copy/setup.yaml /offline-copy/points/*/point.yaml --apply
```

每份合法候選在同名 `.lock` 下重讀，先建原 bytes 的唯一 `.bak`，再用同目錄 temporary 檔原子替換。報告列出備份路徑。Partial／corrupt 不備份、不替換。若備份後替換失敗，已完成的備份仍保留。這不是跨文件交易，也不保證掉電時整批一致。

再次執行同一命令，合法 v2 文件應全部顯示 `skipped`。若尚有拒絕，保留輸出，修復那些文件再重跑，已完成的文件不再改寫。

## 驗證與還原

核對所有 qubit 的 `flux` keys、三欄一致性、mux 對應與保留欄位。升級不驗 device 存在性。等 runtime 為 2 後，重新開啟副本或明確 `context.reload`，在 GUI 打開使用該 point 的表單，確認 device 選項及缺名診斷。重新校準前不要把假設的多 device 參考點當成實測值。

還原時先停止 writers，另外保存目前文件與診斷。把報告所列 `<原檔名>.<唯一識別碼>.bak` 複製回原 YAML 路徑，核對 bytes，再 reopen 或 reload。備份是原版本，runtime 2 可能仍要求升級。還原只恢復文件，不代表完成升級。

Device 改名時改 component 的 `flux_dev` 與 `flux.<device>` key。改接線時刪除受影響 entry，再做 fluxdep。歷史 records 保留當時名稱與 frame，不批次改名。完整流程見 [改名與改接線](record-device-upgrades.md#改名與改接線)。

版本、鎖與失敗語意見 [通用升級指南](parameter-upgrades.md)。格式辨識見 [component flux 格式歷程](storage-format-history/component-flux.md)。
