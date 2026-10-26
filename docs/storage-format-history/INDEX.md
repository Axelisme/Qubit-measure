# 存檔格式變更歷程

**讀者：** operator、lab-author、framework-dev、agent。依格式辨識方式查找變更原因與遷移範圍。

| 格式變更 | 辨識方式 | 舊資料處理 |
| --- | --- | --- |
| [Record component 信封](record-component.md) | JSONL 列含 `component` | 舊列照常讀取，不補寫 |
| [Record device 值歸屬](record-devices.md) | 逐列看 flat pair 或 `devices.<name>.value/value_unit` | 同一長期 upgrade CLI 由 operator 宣告名稱，不補 frame/component |
| [Record flux frame](flux-frame.md) | 每列含 `flux_frame.<device>.integer/period` | 沒有 explicit flux 或新 frame 時，作者明確提供 fallback |
| [Component schema 版本](parameter-schema-version.md) | YAML 含整份文件的 `schema_version` | 明確執行長期 upgrade；runtime 不猜版本或改檔 |
