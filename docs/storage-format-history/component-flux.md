# Component flux frame v1→v2

**讀者：** operator、lab-author、framework-dev、agent。辨識 component flux 格式，查找轉換範圍與參考點假設。

## 改前與改後

`schema_version: 1` 的 qubit 保存 `flux_half`、`flux_int`、`flux_period` 與 `flux_unit`。Mux group 另存 `coil_flux.<qubit>.half/int/period`。

`schema_version: 2` 將兩種校準合併在 qubit 的 `flux`，以實際 device 名稱為 key：

```yaml
flux_dev: local_bias
flux:
  local_bias: {half: 0.5, integer: 0.1, period: 0.8}
  global_bias: {half: -0.2, integer: 0.0, period: -0.4}
```

Local 的 key 來自 v1 qubit 的 `flux_source`。Global 的 key 來自 v1 mux group 明示的 `flux_source`，不固定叫 `coil`。轉換移除 qubit 的四個舊欄位與 mux 的 `coil_flux`。Qubit、mux 與 JPA 的 v1 `flux_source` 都改名為 v2 `flux_dev`，值原樣保留。各數字保留原工作單位，使用時單位取 device info。YAML 容器的 `format_version` 不變。

每個 entry 的 `half`、`integer`、`period` 都是有限數字，period 非零。三欄須滿足 `|integer − half| = |period| / 2`。負 period 表示增加 bias 使 flux 減少。

## 原因與假設

多個 DC device 可以共同影響一顆 qubit。新的校準以其他 contributing devices 都位於 integer 時的 bias 為參考點。

舊文件沒有記錄其他線當時的值，轉換只能假設上述參考條件。報告提醒 operator 重新校準，不能將轉換結果當成重建出的量測事實。工具不換單位、不補未知校準、不推測缺少的來源名稱。

## 升級與版本切換

長期入口是 `scripts/upgrade_parameters.py`，由 resources runner 執行 lab 的純 v1→v2 步驟。未標版文件只有通過完整舊格式分類後才視為 v1。Partial／corrupt 不寫入，其餘文件繼續。預設 dry-run，apply 前保存原 bytes 備份，合法 v2 重跑跳過。

Runtime 與升級目標都是 2。Runtime 升 2 與移除舊欄位、更新模擬範本在同一個 commit 交付，避免形狀與版本錯配。讀取端不自動升級，也不增加 v1 相容讀取。

舊 repo 的 `storage_mapping` 共用相同純轉換，完整 seed／value batch 先轉成 v2，才交目前 schema 驗證與 publication，不先發布 v1 再升級。匯入使用實際的 v2 component schema 驗證。長期升級鏈不 import `migrations/`，刪除一次性匯入工具不會刪掉歷史步驟。

離線文件沒有 device catalog。工具只驗名稱語法，不驗 device 存在性。未知 device 在 GUI 打開 point 等使用處檢查並報名。

## 不變的歷史資料

Component 升級不改 JSONL 歷史，不補 `flux_frame` 或 `component`。Record 的舊 `dev_value/dev_unit` 由同一長期 CLI 的 record 範圍按 operator 宣告改為具名 device 值，不推測歷史 frame。改名或改接線仍保留 record 當時的名稱與 frame。

- [Flux 升級指南](../flux-migration.md) 提供四類形狀、手動修復、dry-run、備份與還原。
- [通用版本指南](../parameter-upgrades.md) 定義 runner 與失敗語意。
- [Record device 升級](../record-device-upgrades.md) 說明具名值與改名改接線。
- [資料狀態對照表](../../migrations/README.md) 區分舊 repo、native YAML 與 CSV 工具。
