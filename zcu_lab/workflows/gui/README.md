# workflows/gui — Workflow form attachments

**Last updated:** 2026-10-08，T1 tunables 垂直 caller

這個 package 擁有具體 workflow 的表單與明寫的 to_cfg，不擁有 Engine 或 Qt editor。Native workflow definitions 不依賴這些 GUI attachments。

T1TunablesForm 與 nested CalibrationTunablesForm 定義使用者可編輯的普通值、Field 約束與呈現 metadata。Defaults 只讀 host 捕捉的 MetaDict。guess_us 存在時保留 expression 輸入，缺 key 時使用 native 暫定預設，不以 fallback 蓋掉無效的引用值。

FormState 處理 draft、驗證與原文差異。to_cfg 只把合法普通值轉成 native T1Tunables。Host 決定何時提交，以及如何把 FormChange.original 交給 Engine；表單不更新 owner revision，也不操作設備。
