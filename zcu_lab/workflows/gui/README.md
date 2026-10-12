# `workflows.gui` Workflow form attachments

**Last updated:** 2026-10-12，表單保留、宿主搬至 lib

這個 package 擁有具體 workflow 的表單、to_cfg 與校準 module 轉換，不實作通用 editor、Window 或 Engine。
Native science 不依賴 GUI。宿主、catalog、泛型 attachment、表單協調、presentation、input 保存、RPC 與 MCP 組裝由 lib 擁有，見 [host README](../../../lib/zcu_tools/gui/app/workflows/README.md)。

## Catalog 與 Coherence forms

`zcu_lab.workflows.registry.build_registry` 依序配對 `coherence_fluxdep`、`mist_fluxdep`、`overnight`，Coherence 是預設。
每個 `WorkflowAttachment` 保留 native step、plan/knobs model、converter 與 scalar seed 的同一版本配對。
Builder 在 bounded reload 中 import current declarations；WorkflowRegistry 只驗 declaration，不提供 lookup。
GUI Start 用選定附件。RPC／MCP workflow.start 的 name 省略時選預設，未知名稱明確拒絕，不猜舊名或提供 alias。

`CoherencePlanForm` 編輯 absolute flux range/list、必填 flux_unit、delay points、初始 pi_length_us 與共用 drive/readout ModuleInput。
`measure_t1`、`measure_t2ramsey`、`measure_t2echo` 是固定 plan bool，預設 true，至少開啟一項。
All false 回 consuming-field error，不提交 acquisition。Disabled lifetime 不量、不保存、不產生 record 元素。
Plan 透過通用 WorkflowPlanPanel preview；local parse errors 先阻擋 preview，再呼附件 converter。

`CoherenceTunablesForm` 含共用 `cal` 與獨立 `t1`、`t2ramsey`、`t2echo`，不把 plan flags 放進 paused tunables。
`T1TunablesForm` 是 t1 子表單，不再含重複 calibration。Echo／Ramsey 子表單各有自己的 guess、averaging 與 timing。
Ramsey detune_ratio 預設 0.05 fringes/step，不論開關都驗有限值。Plan/tunables 仍分開驗證，不新增聯合 admission。
T1 guess 初次輸入保留 captured general.ext.guess_us expression，位於 t1.guess_us；沒有 capture／leaf 時用 native ordinary default。
Null 或 invalid referenced value 產生欄位錯誤，不 fallback。Expression 留在 FormInput，Engine journal 只存普通值。

`MISTPlanForm` 另要求固定 gain start/stop/points 與 disturbance pulse，reset 可明確省略；gain 不放 tunables。
Overnight forms 保留固定 pair、interval 與沒有 feedback 的原契約。
三個附件共用通用 WorkflowFormPanel、WorkflowPlanPanel 與既有 tuning view，不建立第二個 Window 或 Controller。
附件只負責 Start/forms/native run，沒有 export/sample binding。
Saved run conversion 由 coherence_export 與 mist_export 獨立工具負責，GUI/RPC/MCP 不提供匯出入口。

## 固定來源與 modules

Prepare/Start 用 read_context() 固定完整 alias 表與 sources，不宣告角色或檢查 kind。
Global qubit alias 必須可讀；EMPTY 可建 app，但來源依 base 契約拒 Prepare/Start。
Defaults 在同一 capture 解析 $qubit.module.x180.waveform.length，保留具體路徑 expression，缺值或不可讀就留 pi_length_us 未填。
它是 required Rabi-range estimate，供 host seed 使用，不是 fitted pi，也不放進 native CoherencePlan。

`calibration_modules.resolve_calibration_modules` 集中 Coherence/MIST 的 drive 與 pulse-backed readout 規則。
它經 captured core/native factories materialize ref/inline modules與 assets。
Pure arb drive 不能改 Length-Rabi duration，回 qub_pulse consuming-field error；flat_top arb ramp 與 arb readout 保留。
這裡不提供 gain-Rabi、time-scaling、live repository lookup 或 guessed modules。

Coherence 的新 flux_dev defaults 在固定 capture 解析 $flux_dev，成功就用 direct literal device name。
既有 direct draft 不隨 alias 修改，使用者仍可切 eval。合法 saved input 與 MIST defaults 不改。
Missing capture/unbound alias 保留 $flux_dev expression error input，公開欄位顯示 Invalid 並指名 flux_dev，不阻斷 panel 或補造 device。
Fake caller 明填 native；YOKO 明填 A 或 V。表單不推測單位、不轉換值、不預驗 device membership。
Native snapshot 沿當次 capture，後續 alias 修改、Prepare 或 Save 不重解析既有 run。

## 單一 run 與 remote

Window 將 fixed context/storage、typed cached device mapping、software provenance 與 native plan 交給唯一 Controller。
Start 只建一個 operation，pending token 不等於 Engine 已接受。Worker 持 RUN lease，透過 native env effects acquisition。
Pause 保留 operation、lease、plan、accepted input、capture 與 Plots。Resume 用同 token、新 tunables revision 重做未提交點，不在 Resume 讀硬體。
Device actual snapshot 只在每次 env.run callback 前讀取；native 保存使用該次 evidence。
Stop/close 等 producer 返回與 terminal delivery 後才 finish，replacement Start/close 再 release presentation。

GUI、RPC、MCP 呼同一 Window/Controller，不另存 run 或重做 target 核對。
Partial Start 的 plan/tunables 都是 path-array/value batch。空 batch 用當次 capture defaults，不用 GUI draft。
Host 用 FormState.write 一次套用 call-local input，invalid 留原 panels/base，成功才 reset 至實際提交 input。
GUI 無 batch 時仍讀自身 panels/local text。Transport stop 不代表 native cancellation。

workflow.form 讀同一 schema、input、ordinary leaves、path errors 與 revision。
Run 附上後需明填 run_id，讀 retained Controller binding 的唯一 knobs state/env；pending acceptance 拒絕。
Optional batch 只做 call-local preview，不提交或改 draft/base/accepted input/local text/Engine。
Complex tag 與 whole tuple 的 JSON array 規則沿共用 codec。

Tuning view 直接呈現 Host 的唯一 FormState，GUI Apply 與 JSON edit 共用 Controller.apply_tunables。
兩者整批提交當下普通值，不保存另一個 draft revision guard。
只有成功才更新 accepted input/base 並清 local text；invalid input 留在 shared draft，RPC 保留 field path/message。
Paused Apply 後 Resume 保留 commits，新 revision 只用於重做未提交點。

Records RPC 直接投影 Engine committed records；point 現在含獨立 stage 與 enabled measurements。
Live/summary 只建立 enabled lifetime 圖。具名圖/PNG 沿 retained Plots 與 owner-thread renderer，status-only 不渲染。
宿主不輪詢 committed records 來畫 overnight 趨勢。Workflow 使用 `env.plots` 自行決定圖的資料與更新時機，不自動保存或重開 run 圖。

## Input 保存與 source reload

App 用 SingleFileCaretaker 保存 parsed FormInput 原文，包括 expression，不保存 local unparsed text、journal 或硬體狀態。
開窗 restore，terminal close 後 flush；--clean 只略過開窗讀檔，關窗仍保存。
FormState.write_compatible 丟棄不相容 path，不做 saved schema migration 或舊 workflow 名稱對映。
已附 run 的 knobs 取 Controller retained binding，不另複製 input。

WorkflowForms 保留各名稱的 RAM parsed drafts。切換先建好兩個新 panels，再發布 selection。
Prepare 用新 capture 建 defaults；source reload 保留 current capture，替換 current declarations/forms，不重建 session/services/controller/hardware。
Reload 只允許 idle/terminal；pending/running/paused 在 preflight 前拒絕。
Compatible input 保留；removed/renamed selection 改選新 catalog 第一個附件，並顯示原名與新名。
Import/panel replacement 失敗保留舊 panels/RAM，但阻擋 Start 直到 retry；fixed source 改動要求 restart，沒有舊 callable fallback。

整個 zcu_lab.workflows，包括註冊清單與表單，是 bounded reload scope；lib framework、components 與 native experiment sources 固定。
Scripts 注入的 SourceCatalogLoader 每次重新 import 當前 registry，宿主不保存舊 builder。
Window 的同 run、capture、取消與保存邊界見 ADR-0062、ADR-0063、ADR-0067、ADR-0068。
