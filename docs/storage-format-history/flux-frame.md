# Record flux frame

**讀者：** operator、lab-author、framework-dev、agent。辨識多 device record frame 與分析時的 fallback 規則。

## 改前與改後

單 device 分析曾使用 flat row-frame：

```json
{"run_id":"run-a","devices":{"bias":{"value":0.5,"value_unit":"V"}},"flux_int":0.1,"flux_period":0.8}
```

新 row-frame 以 device 名稱為鍵。儀器值與校準使用各 device 的原生單位：

```json
{"run_id":"run-b","devices":{"bias":{"value":0.5,"value_unit":"V"},"coil":{"value":0.1,"value_unit":"A"}},"flux_frame":{"bias":{"integer":0.1,"half":0.5,"period":0.8},"coil":{"integer":0.0,"half":-0.2,"period":-0.4}}}
```

CSV 攤平後的校準欄位是 `flux_frame.bias.integer`、`flux_frame.bias.period` 與其他 device 的對應欄位。
儀器值在 `devices.bias.value` 與 `devices.coil.value`。
新例子的 flux 是 0.25 Phi0，包含 bias 的 0.5 與 coil 的 -0.25。

## 原因與辨識

多條儀器線可能同時影響一顆 qubit。
Frame 記錄每台儀器自己的 integer 參考點與 signed period。
所有 device 都停在各自 integer 時，flux 是零。
分析以 `Σ(value - integer) / period` 計算總 flux，負 period 表示反向貢獻。

Record 沒有獨立的版本標頭。
每列是否有新 row-frame，以 `flux_frame` 判斷。
Nested 輸入以 frame keys 判斷 device membership。
Dotted 輸入以該列該 device 任一 frame 欄位非 null 判斷。
全 null 代表該列沒有該 device。
攤平無法區分原本沒有 device 與原本有 key 但欄位全 null，檢查後者須保留 nested 輸入。

## 舊資料與讀取規則

`flux_from_frame` 每列依序使用 explicit flux、新 row-frame、caller fallback。
有效 explicit flux 不檢查未用 frame。
使用 row-frame 時，缺 contributing device 的值、integer 或 period 都報錯並指出列與欄位。
它不忽略部分 frame，也不退回 fallback。

`flux_from_frame` 不再解讀舊 flat `flux_int/flux_period` row-frame。
只帶舊 flat frame 的列若沒有 explicit flux，作者須把該 frame 作 caller fallback 傳入：

```python
from zcu_tools.resources.record_table import RecordFluxFrame, flux_from_frame

flux = flux_from_frame(
    table,
    device_name="bias",
    fallback_frame=RecordFluxFrame(dev_unit="V", flux_int=0.1, flux_period=0.8),
)
```

多 device fallback 使用 `analysis.flux.FluxFrame`。
每台儀器的 record `value_unit` 必須與 caller 宣告的單位相同。
工具不換單位，不自動讀 `params.json`，也不推斷舊欄位別名。

舊 records 不補寫或自動遷移。
獨立 `resolve_record_flux`、`validate_record_table_v2` 與 v2 遷移入口仍保留其 flat 座標契約。
此處只變更 `flux_from_frame` 的 row-frame 讀取格式。

## 相關契約

- [Record table owner](../../lib/zcu_tools/resources/record_table/README.md#作者-flux-換算) 定義 nested、dotted 與 fallback 行為。
- [Analysis owner](../../lib/zcu_tools/analysis/README.md#多-device-flux-座標) 定義純數學與具名型別。
