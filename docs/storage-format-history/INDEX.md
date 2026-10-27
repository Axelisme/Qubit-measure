# 存檔格式變更歷程

**讀者：** operator、lab-author、framework-dev、agent。依格式辨識方式查找變更原因與遷移範圍。

| 格式變更 | 辨識方式 | 舊資料處理 |
| --- | --- | --- |
| [Record component 信封](record-component.md) | JSONL 列含 `component` | 舊列照常讀取，不補寫 |
| [Record device 值歸屬](record-devices.md) | 逐列看 flat pair 或 `devices.<name>.value/value_unit` | 同一長期 upgrade CLI 由 operator 宣告名稱，不補 frame/component |
| [Record flux frame](flux-frame.md) | 每列含 `flux_frame.<device>.integer/period` | 沒有 explicit flux 或新 frame 時，作者明確提供 fallback |
| [Component flux v1→v2](component-flux.md) | YAML `schema_version: 2`，qubit 有 `flux.<device>` | 長期 upgrade 完整分類，假設參考點需重校準，runtime 與輸出均為 v2 |
| [Component schema 版本](parameter-schema-version.md) | YAML 含整份文件的 `schema_version` | 明確執行長期 upgrade；runtime 不猜版本或改檔 |

## 其他格式變更

- 2026-10-27，[Labber 目標身分與 caller 命名](2026-10-27-labber-identity.md)。Labber comment 加入 point／roles snapshot，Labber 與 native 路徑改用 caller 的 save_stem，既有檔案不改名。
- 2026-10-26，[移除 acquisition facts](2026-10-26-remove-acquisition-facts.md)。Run metadata、workflow operation／RPC 與 summary 不再保存 acquisition observations，舊 acquisitions key 只讀取忽略。
- 2026-10-11，[舊 result／Database 與 sample CSV 遷入 native entry](2026-10-11-legacy-storage-and-sample-csv.md)。整理 chip／qubit 來源到 entry／native 的對照，以及 CSV 到 record JSONL 的 v2 欄位與離線轉換工具。
