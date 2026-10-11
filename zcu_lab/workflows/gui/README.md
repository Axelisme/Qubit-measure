# workflows/gui — Workflow form attachments

**Last updated:** 2026-10-11，共用 alias preset 與字串 expression

這個 package 擁有具體 workflow 的表單、明寫的 to_cfg 與 Qt panel attachment，不實作通用 editor 或 Engine。Native workflow definitions 不依賴這些 GUI attachments。

T1PlanForm 編輯 absolute flux range/list、必填 flux_unit、delay points 與共用 drive/readout ModuleInput。Fake caller 明填 native，YOKO caller 明填 A 或 V；表單不推導單位或提供 default。to_cfg 以 captured library 的公開 factory 解析 ref/inline，轉成 native modules，展開 setpoints，接 host 的歷史 RunSnapshot。缺 root、錯 family 或 materialization 失敗回報 consuming-field error。Native ModuleCfgFactory.from_core 接同一 materialization context，arb 與 flat_top arb ramp 的播放路徑固定在 captured asset directory，不讀 global repository。T1 drive 若是純 arb，在 qub_pulse 回 consuming-field error，因為 Length-Rabi 需要改 length 而 arb duration 固定。Flat_top 的 arb ramp、direct arb readout 與 flat_top arb readout ramp 繼續支援；不加入 gain-Rabi 或 time-scaling。非 arb flat_top 的 length 沿 native 總長語意原值直傳。Plan 不含完整 cfg、template 或 acquisition grid。T1PlanPanel preview 先阻擋 local parse errors，再產生 native T1Plan，不 Start 或捕捉來源。

初始 pi_length_us 是必填的 Rabi-range estimate。Window 用 `read_context()` 固定當次完整 alias 表與來源，不宣告角色或檢查元件 kind。Prepare 要求可讀的 global qubit alias，缺列或未綁定時原樣傳播指名 alias 的錯誤。EMPTY 可建立 app，但 Prepare／Start 依 base 契約拒絕；來源狀態顯示隔離這些失敗。Defaults 在同一 capture 解析 `$qubit.module.x180.waveform.length`，用具體元件路徑預填可改的 expression，缺值或不可讀就留空，不使用 simulator 常數或其他模組。

T1WorkflowWindow 的 Start 從當次固定 capture 驗證輸入，以 validated pi_length_us 與 native module scalars 建 Calibrated seed。Native T1Plan 不新增此欄位。Window 將固定 context/storage、表單用 cached native device mapping 與 software provenance 交 WorkflowController，在 worker 持 lease 啟動 native run；pending token 不代表 Engine 已接受。Stop 與關閉視窗使用同一 pending controller，producer 返回後才完成 plots。Window 使用固定 Controller，每次 Start 將 ports 與 pending Host 交給它，不另存 run form capture 或 accepted input。

Pause/Resume 的公開 controls 與按鈕使用同一 native run。Pause 保留 operation、RUN lease、plan/input capture 與 Plots lifetime，Resume 不讀裝置，以同 token 提交 worker；Start 表單直接使用公開 typed cached native mapping，不做 live read。Device 實際值只由每次 env.run 在 worker 捕捉，SessionDevices 已接 live typed cfg、完整 selected-model setup 與 native capture helper。每次 env.run 在保留的 RUN lease 下、callback 前讀取，Save 使用固定 evidence。整個 run 終止才解除 plan/input 鎖定。

T1WorkflowWindow.create_remote_service以同一Window、session bus與owner組裝T1RemoteControlAdapter。RPC提供partial-input Start、native status及明填run_id的Pause／Resume／Stop，不另存run或重做target核對。Start的plan／tunables各送path-array／value batch，只送與defaults不同的欄位，空batch用當次capture的defaults。Host用既有FormState.write一次套用，不寫GUI draft或做tree overlay。Invalid回既有FormErrors path/message且不改panels；成功用既有reset(base)顯示實際提交input。GUI不傳batch時仍讀自己的panels及local parse errors。Pending token不代表worker接受，transport stop不取消native run。

Records／具名圖 RPC 讀同一 Window 的 native run。Records 直接投影 Engine 已提交的 records，保留 commit cursor 與 captured revision，不從圖反推數值。圖名與 PNG 由 Window 持有的 Plots 提供，PNG 沿 shared renderer 在 Qt owner 產生；status-only 不渲染。Pending acceptance 明確失敗，terminal 圖保留至新 Start 或 close；close 成功釋放後不再提供舊圖。MCP forwarding 已接同一 GUI RPC。

獨立 GUI app 已接線，B4 12 的共用 alias preset 已接入。App 直接重用 `zcu_lab.ALIAS_PRESET`，交給 shared `build_session_services(alias_rows=...)`。Preset 保留 qubit 空列、resonator 的 `$qubit.resonator` 軟連結與 flux_dev 空列，不複製 workflow-local preset。FormState 已透過共用 evaluator 解碼所有非 discriminator 的字串 expression。T1 flux_dev 預設為 `{"expr":"$flux_dev"}`，Preview 顯示固定 capture 的解析名稱，missing／unbound alias 在該欄位指名 Invalid。普通字串仍是 literal，flux_unit 仍必填。表單不預驗 device membership，未知名稱留到實際使用時失敗。RunSnapshot 與 native 保存沿當次 capture，後續 alias 修改或 Save 不重新解析。S4 source reload 與 U7 input 保存／還原未做。Base 09 終驗、land review 前完整 integration gates 與使用者 CP-B 驗收仍待完成。

App 組裝一份 SessionServices，Window constructor 為 `(state, services, owner, *, on_close=None, parent=None)`。Window 的 runner 與 progress 取自 services，GUI 與 MCP 操作同一 Window／Controller。parent=None 建立 top-level Window，指定 QWidget 時嵌入該 Qt parent。關窗先等待 run 終止，再呼叫 on_close；callback 失敗拒絕關窗，保留資源供重試。這個介面遷移不代表 CP-B 驗收。

workflow.form 讀同一 Window 的 plan／knobs schema、input、普通求值 leaves、path errors 與 native revision。未 Start 讀 constructor captures；已附 run 必須明填 run_id，改用 Controller binding 的固定 env 與唯一 knobs state，pending acceptance 明確拒絕。Optional batch 只在 call-local FormState 套用目前 parsed input 加修改，不提交、不改 GUI draft/base、accepted input、local text 或 Engine。Read 包含 local parse errors；batch 預覽不把尚未 parse 的文字當成候選。Complex 沿既有 codec tag，whole tuple 在 JSON 為 array；values 不代表可提交。Panel 的 env 是 constructor capture，不是 source reload。

Start 同步提交失敗時沒有 producer。Window 先結束本次 Plots operation，再釋放呈現並清除 reference，保留原提交錯誤與固定 Controller。GUI 與 RPC 使用相同清理路徑，失敗後視窗仍可關閉。

T1TunablesForm 與 nested CalibrationTunablesForm 定義使用者可編輯的普通值、Field 約束與呈現 metadata。Defaults 只讀 host 捕捉的 ParameterContext。general.ext.guess_us 存在時保留 expression 輸入，沒有 capture 或缺 leaf 時使用 native 暫定預設。明確 null 與無效引用值產生欄位錯誤，不以 fallback 蓋掉。

T1TunablesPanel 只呈現必填的 host-owned FormState 與固定 env，嵌入通用 FormEditor，並公開同一 state 與 editor。Window 建立 Start draft，Tuning 使用 TunablesHost 的 state；panel 不建立 defaults 或複製 draft。Host 提交前檢查 editor.local_parse_errors。Panel 不提供 native preview、Apply／Discard／Start 或 live source capture。

FormState 處理 draft、驗證與 input 差異。to_cfg 只把合法普通值轉成 native T1Tunables。Expression 留在 FormInput，Engine journal 只記普通值。Host 決定何時提交；表單不更新 owner revision，也不操作設備。

T1TuningPanel 接收已有 native acceptance 的 Controller，直接呈現其 Host 的唯一 FormState，不重建 draft 或 form capture。GUI Apply 與外部 JSON edit 都呼 Controller.apply_tunables。Host 保存已接受的 expression input 並提供 Apply／Discard。GUI與外部edit提交當下完整普通值，不持GUI draft revision守衛。Engine保存數值revision，過渡期不檢查holder。Window.apply_tunables 先用 Controller.require_run_id 核對明填的 native target，再交同一 T1TuningPanel.apply_external。Pending startup 尚未接受時明確失敗，不碰 draft。workflow.tunables RPC 接 run_id 與 path/value batch，actor 固定為 agent/rpc。空 batch 提交當前完整 parsed input，不採 Start 的 defaults-only 語意。Model-invalid input 留在 shared draft，RPC 保留 FormErrors 的 path/message。只有成功才更新 accepted input/base 並清 local text。Engine 拒絕 lifecycle 或 journal failure 時不重試，也不假回成功。Paused Apply 後 Resume 保留已提交 records，未提交步驟用新 revision 重做。這個 caller 不 Start、capture、安排 worker 或保存 input 檔案。
