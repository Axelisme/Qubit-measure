# Shared session UI

**Last updated:** 2026-10-29，Context／Predictor 窄 control ports

**讀者：** framework-dev、agent。查找共用 dialog 的 owner，以及 UI 命令交給 session 的接縫。

## Dialog owners

| 元件 | 操作與責任 |
| --- | --- |
| `SetupDialog`、`SocDialog` | Setup 透過 SetupControlPort 開啟 entry、切換 Global／point，並以 inherit（預設）／empty／clone 建立 point。SoC 透過 SocControlPort 連線與瀏覽 soccfg。 |
| `InspectDialog`、`AliasPanel`、`ComponentTreeWidget` | 顯示與編輯 context、library 與 alias，將提交交給 context owner。 |
| `DeviceDialog` | 透過 DeviceControlPort 操作裝置，使用已登錄 adapter 的專屬 panel。 |
| `ArbWaveformDialog` | 透過 arb waveform port 讀寫 entry-scoped waveform 資產。 |
| `PredictorDialog` | 呈現獨立的 PredictorTrial，協調查詢、校準結果、cache 與另存 I/O。 |
| `ProgressStack` | 將 owner 提供的 progress 投影為 Qt 控件。 |

本目錄只依賴所需的窄 control ports，不反向 import app。
ContextControlPort 與 PredictorControlPort 直接由 owner service 實作，UI 不因移除純轉發層而取得額外能力。
App 決定 dialog 的呈現與生命週期，session services 擁有狀態提交與資源操作。
Widget 不直接寫入 point 文件或管理 driver 連線。
各 port 與組合 facet 的完整責任見 [session README](../README.md)。

## Predictor 與生命週期

Qt-free 的 `PredictorTrial` 擁有 model、能量草稿、flux frame、offset、各儀器值與所選軸，集中請求組裝、欄位驗證與校準規則。它提供 snapshot／restore 與 PredictorFile 匯出，不持有 widget。
`PredictorParameterEditor` 與 `PredictorFluxEditor` 必須接收同一份 trial，負責控件呈現與草稿提交。`PredictorCurveTabs` 持有曲線與跨圖互動。
Dialog 保留版面、tracked table、live mode、timer、status、檔案對話框、marker 與生命週期。它只在載入 frame 時讀 device cache，沒有 device 訂閱；live 值由宿主供應。
Dialog 將保存、還原與匯出交給 trial，再協調 caretaker、控件呈現與檔案 I/O。Programmatic 預填只通知一次；user edit 保留清結果、同步狀態、保存與 debounce 順序。校準先呈現 offset，再發布成功 status。
Snapshot 與 codec 位於 `gui/session/predictor_cache.py`，predictor.json v2 格式不變，非法草稿不覆寫上次 cache。
Trial 不安裝到量測 session，也不寫回 component。明確的 Save as 才寫指定 predictor 檔案。

UI 回呼、訂閱與 timer 經 [LifetimeScope](../../README.md#callback-lifetime-lifetimepy) 登記。
Owner 在拆除時關閉 scope，先停用回呼再釋放資源。
共用與 app-local 邊界、predictor 分工見 [ADR-0067](../../../../../docs/adr/0067-gui-application.md)。
