# workflows/gui — Workflow form attachments

**Last updated:** 2026-10-08，T1 tuning host caller seed

這個 package 擁有具體 workflow 的表單、明寫的 to_cfg 與 Qt panel attachment，不實作通用 editor 或 Engine。Native workflow definitions 不依賴這些 GUI attachments。

T1PlanForm 編輯 absolute flux range/list、delay points 與共用 drive/readout ModuleInput。to_cfg 以 captured library 的公開 factory 解析 ref/inline，轉成 native modules，展開 setpoints，接 host 的歷史 RunSnapshot。缺 root、錯 family 或 materialization 失敗回報 consuming-field error。目前拒絕 arb waveform，包括 flat_top 的 arb ramp，避免遺失固定 asset binding 後讀取 global scope。非 arb flat_top 的 length 沿 native 總長語意原值直傳。Plan 不含完整 cfg、template 或 acquisition grid。T1PlanPanel preview 先阻擋 local parse errors，再產生 native T1Plan，不 Start 或捕捉來源。

T1TunablesForm 與 nested CalibrationTunablesForm 定義使用者可編輯的普通值、Field 約束與呈現 metadata。Defaults 只讀 host 捕捉的 ParameterContext。general.ext.guess_us 存在時保留 expression 輸入，沒有 capture 或缺 leaf 時使用 native 暫定預設。明確 null 與無效引用值產生欄位錯誤，不以 fallback 蓋掉。

T1TunablesPanel 用 host 提供的 env 建立 state，嵌入通用 FormEditor，並公開 state 與 editor。Host 提交前檢查 editor.local_parse_errors。Panel 不提供 native preview、Apply／Discard／Start 或 live source capture。

FormState 處理 draft、驗證與 input 差異。to_cfg 只把合法普通值轉成 native T1Tunables。Expression 留在 FormInput，Engine journal 只記普通值。Host 決定何時提交；表單不更新 owner revision，也不操作設備。

T1TuningPanel 為已接受的 native T1 run 嵌入 T1TunablesPanel。它的 GUI Apply 與外部 JSON edit 共用 gui/app/workflows 的 TunablesHost。Host 保存已接受的expression input並提供Apply／Discard。GUI與外部edit提交當下完整普通值，不持GUI draft revision守衛。Engine保存數值revision，過渡期不檢查holder。這個 caller 不 Start、capture、安排 worker 或保存 input 檔案。
