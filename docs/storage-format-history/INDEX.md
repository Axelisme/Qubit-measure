# 存檔格式變更歷程

**讀者：** operator、lab-author、framework-dev、agent。依格式辨識方式查找變更原因與遷移範圍。

| 格式變更 | 辨識方式 | 舊資料處理 |
| --- | --- | --- |
| [Entry／point 文件檔名](document-file-names.md) | 兩層使用 `parameters.yaml`、`library.yaml`、`aliases.yaml` | 先用原 CLI 的 `--scope filenames` 備份改名，再按需升級內容；runtime 不讀舊名 |
| [Fluxdep spectrums v2](fluxdep-spectrums-v2.md) | HDF5 根 attrs `format=zcu.fluxdep-spectrums`、`version=2` | `scripts/upgrade_spectrums.py` 預設預覽，`--apply` 先備份再換算 |
| [Record component 信封](record-component.md) | JSONL 列含 `component` | 舊列照常讀取，不補寫 |
| [Record device 值歸屬](record-devices.md) | 逐列看 flat pair 或 `devices.<name>.value/value_unit` | 同一長期 upgrade CLI 由 operator 宣告名稱，不補 frame/component |
| [Record flux frame](flux-frame.md) | 現行列含 `flux_frame.offset` 與 `flux_frame.period.<device>`，另列歷史 flat／per-device 形狀 | 舊 records 不重寫；作者確認校準後明確轉換分析副本或提供 fallback |
| [Component flux v1→v2，歷史](component-flux.md) | YAML `schema_version: 2`，`flux.<device>.{half, integer, period}` 與 `flux_source` | 保留 main v2 參考點語義，由現行升級鏈轉為 v3 |
| [Component flux v3](component-flux-v3.md) | YAML `schema_version: 3`，`flux: {offset, period}` 與 `flux_dev` | V1→v2→v3 保留主儀器 half 分支、不 mod 1，coil 舊 half／int 列報告並提醒重校 |
| [Predictor v2](predictor-v2.md) | JSON `format: zcu.predictor`、`version: 2`，能量與 `flux: {offset, period, unit}` 分開 | Entry upgrade 預覽／備份替換，runtime 拒讀 v1 |
| [Component schema 版本](parameter-schema-version.md) | YAML 含整份文件的 `schema_version` | 明確執行長期 upgrade；runtime 不猜版本或改檔 |

## 其他格式變更

- 2026-11-01，[Readout power QND](2026-11-01-readout-power-qnd.md)。新 tag 與 grouped paired-IQ sums 取代 SNR-only power，舊 tag 明確拒讀，無法由 SNR 重建 QND moments。

- 2026-11-01，[MIST drive-frequency native 軸順序](2026-11-01-mist-drive-frequency-axes.md)。新檔維度為 frequency／gain；舊方陣檔的磁碟 gain 軸在前，Load 明確拒絕。目前無轉換工具，原檔不改不刪。

- 2026-10-28，[Entry 名稱與量測參數快照](2026-10-28-entry-names-and-run-parameters.md)。Entry 移除 UUID，Run 保存參數環境，Load 不讀 live point，Save 依目前 roots 與記錄名稱寫入。

- 2026-10-28，[移除模組來源](2026-10-28-remove-module-provenance.md)。Library 2.0 忽略舊 provenance，讀取不改檔，clone 不匯入事件。

- 2026-10-27，[Labber 目標身分與 caller 命名](2026-10-27-labber-identity.md)。Labber comment 加入 point／roles snapshot，Labber 與 native 路徑改用 caller 的 save_stem，既有檔案不改名。
- 2026-10-26，[移除 acquisition facts](2026-10-26-remove-acquisition-facts.md)。Run metadata、workflow operation／RPC 與 summary 不再保存 acquisition observations，舊 acquisitions key 只讀取忽略。
- 2026-10-11，[舊 result／Database 與 sample CSV 遷入 native entry](2026-10-11-legacy-storage-and-sample-csv.md)。整理 chip／qubit 來源到 entry／native 的對照，以及 CSV 到 record JSONL 的 v2 欄位與離線轉換工具。
