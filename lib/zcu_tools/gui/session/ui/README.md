# Shared session UI

**Last updated:** 2026-10-28

**讀者：** framework-dev、agent。查找共用 dialog 的 owner，以及 UI 命令交給 session 的接縫。

## Dialog owners

| 元件 | 操作與責任 |
| --- | --- |
| `SetupDialog`、`SocDialog` | 分別透過 SetupControlPort 開啟 entry／切換 point，以及透過 SocControlPort 連線與瀏覽 soccfg。 |
| `InspectDialog`、`AliasPanel`、`ComponentTreeWidget` | 顯示與編輯 context、library 與 alias，將提交交給 context owner。 |
| `DeviceDialog` | 透過 DeviceControlPort 操作裝置，使用已登錄 adapter 的專屬 panel。 |
| `ArbWaveformDialog` | 透過 arb waveform port 讀寫 entry-scoped waveform 資產。 |
| `PredictorDialog` | 維護獨立的參數與 flux trial，透過 PredictorControlPort 查詢與校準。 |
| `ProgressStack` | 將 owner 提供的 progress 投影為 Qt 控件。 |

本目錄只依賴所需的窄 control facets，不反向 import app。
App 決定 dialog 的呈現與生命週期，session services 擁有狀態提交與資源操作。
Widget 不直接寫入 point 文件或管理 driver 連線。
各 facet 的完整責任見 [session README](../README.md)。

## Predictor 與生命週期

`PredictorParameterEditor` 持有模型與參數控件，`PredictorCurveTabs` 持有曲線與跨圖互動。
Dialog 保留 cache、device 訂閱、calibration 與 tracked table，不把整個 dialog 交給 child。
Trial 不安裝到量測 session，也不寫回 component。明確的 Save as 才寫指定 predictor 檔案。

UI 回呼、訂閱與 timer 經 [LifetimeScope](../../README.md#callback-lifetime-lifetimepy) 登記。
Owner 在拆除時關閉 scope，先停用回呼再釋放資源。
共用與 app-local 邊界、predictor 分工見 [ADR-0067](../../../../../docs/adr/0067-gui-application.md)。
