# Record flux frame

**讀者：** operator、lab-author、framework-dev、agent。辨識歷史與現行 record frame，選擇分析 fallback 並處理缺值。

## 歷史形狀

早期單儀器 flat row-frame：

```json
{"run_id":"run-a","devices":{"bias":{"value":0.5,"value_unit":"V"}},"flux_int":0.1,"flux_period":0.8}
```

Main v2 的 component 模型產生每台儀器自己的參考點：

```json
{"run_id":"run-b","component":"Q1","devices":{"bias":{"value":0.5,"value_unit":"V"},"coil":{"value":0.1,"value_unit":"A"}},"flux_frame":{"bias":{"integer":0.1,"half":0.5,"period":0.8},"coil":{"integer":0.0,"half":-0.2,"period":-0.4}}}
```

歷史 v2 公式為 `Σ(value - integer) / period`，所有儀器停在各自 integer 時 flux 為零。
上例為 0.25 Φ0。這是歷史形狀，不是目前 reader 支援的 row frame。
Component 升級不重寫這些 records。

## 現行 v3 frame

Record 沒有獨立版本標頭，現行 frame 複製 component 的 `{offset, period}`：

```json
{"run_id":"run-c","component":"Q1","devices":{"bias":{"value":0.5,"value_unit":"V"},"coil":{"value":0.1,"value_unit":"A"}},"flux_frame":{"offset":-0.125,"period":{"bias":0.8,"coil":-0.4}}}
```

公式為 `offset + Σ value / period`，上例仍為 0.25 Φ0。
CSV 校準欄位是 `flux_frame.offset`、`flux_frame.period.bias`、`flux_frame.period.coil`。
儀器值在 `devices.bias.value` 與 `devices.coil.value`，各用原生單位。
未校準 frame 為 `{offset: null, period: {}}`。

Lab adapters 使用同次 Run 的 parameter capture，devices 來自 Run-start snapshot。
Workflow 使用該次 Plan 捕捉的 frame。Notebook 作者量測前捕捉後經 accept 明傳。
Append owner 加 [component 信封](record-component.md)。接受時不重讀目前 point，也不以新校準覆蓋原 frame。
Fluxdep record 另存被掃 device、原始 half、integer 與正的 period，這些是擬合點，不是 component 欄位。
Point writeback 才分開提 signed period 與 offset。

## 逐列分析與缺值

`flux_from_frame` 依序使用 explicit flux、現行 row frame、caller fallback。
有效 explicit flux 不檢查未用 frame。使用 row frame 時，缺 offset 報錯並指出列與欄位，不退回 fallback。
有 period 但缺 device value 的儀器不貢獻，period 保留。
Nested 可區分缺 key 與明寫 NaN，後者以及 inf、bool 都報錯。
Dotted device value／period 的 NaN cell 視為缺席，inf、bool 仍報錯。
攤平無法區分缺 key 與明寫 NaN，只有 nested 保證拒絕明寫 NaN，不新增 presence 欄位。

## 舊資料與 fallback

Reader 不自動解讀歷史 flat 或 v2 per-device frame，也不自動寫回升級。
沒有現行 row frame 的列可由作者確認歷史校準與單位後明傳 fallback：

```python
from zcu_tools.analysis.flux import DevicePeriod, FluxFrame
from zcu_tools.resources.record_table import flux_from_frame

frame = FluxFrame(offset=-0.125, devices={"bias": DevicePeriod(0.8, unit="V")})
flux = flux_from_frame(table, fallback_frame=frame)
```

Fallback 統一用新 `FluxFrame`，單儀器與多儀器同型別。有值儀器的 record 單位必須與 caller 宣告相同。
已有但不合法的 frame 不會被 fallback 遮蓋。歷史 per-device frame 應先在分析副本明確轉換，不能直接加 fallback 繞過。
不換單位、不讀目前 component 猜歷史校準，也不推斷舊欄位別名。
Standalone `resolve_record_flux`／`validate_record_table_v2` 保留既有 flat row schema，fallback 改用恰有一台儀器的新 `FluxFrame`。
0 台或多台均報錯並說明台數。這個 standalone v2 與 component schema v3 是不同版本線。

## 相關契約

- [Record table owner](../../lib/zcu_tools/resources/record_table/README.md#作者-flux-換算) 定義 nested、dotted 與 fallback 行為。
- [ADR-0082](../adr/0082-component-flux-frame.md) 定義 writeback 與版本邊界。
- [Analysis owner](../../lib/zcu_tools/analysis/README.md#多-device-flux-座標) 定義純數學與具名型別。
- [Component v3 歷程](component-flux-v3.md) 定義升級公式與 half 分支。
