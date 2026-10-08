# workflows/gui — Workflow form attachments

**Last updated:** 2026-10-08，native parameter capture

這個 package 擁有具體 workflow 的表單、明寫的 to_cfg 與 Qt panel attachment，不實作通用 editor 或 Engine。Native workflow definitions 不依賴這些 GUI attachments。

T1TunablesForm 與 nested CalibrationTunablesForm 定義使用者可編輯的普通值、Field 約束與呈現 metadata。Defaults 只讀 host 捕捉的 ParameterContext。general.ext.guess_us 存在時保留 expression 輸入，沒有 capture 或缺 leaf 時使用 native 暫定預設。明確 null 與無效引用值產生欄位錯誤，不以 fallback 蓋掉。

T1TunablesPanel 用 host 提供的 env 建立 state，嵌入通用 FormEditor，並公開 state 與 editor。Host 提交前檢查 editor.local_parse_errors。Panel 不提供 native preview、Apply／Discard／Start 或 live source capture。

FormState 處理 draft、驗證與原文差異。to_cfg 只把合法普通值轉成 native T1Tunables。Host 決定何時提交，以及如何把 FormChange.original 交給 Engine；表單不更新 owner revision，也不操作設備。
