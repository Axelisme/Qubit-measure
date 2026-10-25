# Workflow form attachments

**讀者：** lab-author。維護 workflow 表單、cfg 轉換與宿主註冊接線。

**Last updated:** 2026-10-22，async scripts 與 live tunables

這個 package 擁有具體 workflow 的表單、to_cfg 與校準 module 轉換，不實作通用 editor、Window 或 Engine。
Native science 不依賴 GUI。宿主、泛型 attachment、表單草稿、presentation、RPC 與 MCP 組裝由 lib 擁有，見 [measure host README](../../lib/zcu_tools/gui/app/measure/README.md#workflow-hosting)。

## Catalog 與 Coherence forms

`zcu_lab.workflows.registry.build_registry` 依序配對 `coherence_fluxdep`、`mist_fluxdep`、`overnight`、`overnight_mux`。
每個 `WorkflowAttachment` 保留 native async script、plan/knobs model、converter 與 scalar seed 的同一版本配對，import 來自 `gui.app.measure.services.workflow_attachment`。
Caller 把清單交給 measure headless facet。`WorkflowRegistry` 只驗 declaration，不提供 lookup。
MCP 以 `workflow_open(name)` 明填名稱，再以回傳的 tab_id 呼叫 `workflow_start`。RPC 使用 `workflow.open` 與 `workflow.start`，未知名稱拒絕，不預選 Coherence 或提供舊名 alias。

`CoherencePlanForm` 編輯 absolute flux range/list、必填 flux_unit、delay points、初始 pi_length_us 與共用 drive/readout ModuleInput。
`measure_t1`、`measure_t2ramsey`、`measure_t2echo` 是固定 plan bool，預設 true，至少開啟一項。
`measure_singleshot` 也是固定 plan bool，預設 false。開啟後在校準之後、lifetimes 之前保存合格點的 GE 原始資料，不擬合。
All false 回 consuming-field error，不提交 acquisition。Disabled lifetime 不量、不保存、不產生 record 元素。
共用 FormEditor 顯示 owner 的 Plan draft 與 Form preview。Start 經 headless owner 驗 Form，再呼附件 converter；remote preview 不執行 native lowering。

`CalibrationTunablesForm` 的 `rabi_mode` 預設 `len_then_amp`，可改 `amp_only` 或 `len_only`。MIST 與 Coherence 共用這份子表單。
`amp_gain_max` 預設 1，控制 amp-only 上限；`amp_gain_half_width` 預設 0.2，控制後續 gain 追蹤半寬。
`amp_min_gain` 預設 0.0001，`amp_max_window_fraction` 預設 0.9，控制 gain 接受範圍。Amp 與 Len 共用 Rabi 誤差、RMS residual 與 MAE ratio knobs。
`target_pi_length_us` 是 Len feedback 的目標，也是 amp-only 的固定長度；空白時取 min(captured host seed π, rabi_min_stop_us / rabi_stop_factor)。
Readout 優化的 averages（ro_reps）預設 100、ro_rounds 預設 1。其餘量測 stage 的 averages（reps）預設 1000、rounds 預設 10，包括 MIST。Overnight 與 overnight_mux 的 t1_earlystop_snr、t2_earlystop_snr 預設 20.0，其餘 stage 的 Early-stop SNR 預設值不變。Early-stop SNR 可留空（None）關閉。Readout 優化與 MIST 沒有 early stop。Singleshot shots 維持原值。
Amp 使用同一份 Rabi averages、rounds、acquire_retry、earlystop_snr 與 relaxation knobs。已開始的 acquisition 不受後續表單修改影響。

`CoherenceTunablesForm` 含共用 `cal` 與獨立 `singleshot`、`t1`、`t2ramsey`、`t2echo`，不把 plan flags 放進 tunables。
`SingleshotTunablesForm` 提供 `min_spacing`、`shots`。間距是 plan flux 單位，從上次實際 GE 擷取起算；shots 是每種 probe-off/on preparation 的數量。
預設間距 0、shots 100000 是暫定模擬值。GE 要求本點 Rabi 已接受，不另加 residual 或對比門檻。它們可在 run 中更新，已開始的 GE operation 保留當次 cfg。
`T1TunablesForm` 是 t1 子表單，不再含重複 calibration。Echo／Ramsey 子表單各有自己的 guess、averaging 與 timing。
Ramsey detune_ratio 預設 0.05 fringes/step，不論開關都驗有限值。Plan/tunables 仍分開驗證，不新增聯合 admission。
`defaults(env)` 不接收 previous，只宣告新表單初值。Module 切換使用 `ModuleInputField.converter(env)` 的共用繼承政策。T1 guess 以 captured general.ext.guess_us 的 expression 宣告，開表單時算成值，位於 t1.guess_us；沒有 capture／leaf 時 defaults 明確給 20 µs。
`guess_us` 沒有 Pydantic 靜態預設，`t1` 也因此沒有靜態預設，只由 `CoherenceTunablesForm.defaults` 提供。
Null、非數字或 ≤0 的 referenced value 讓 guess_us 留空，result 只報 `t1.guess_us` 的 Field required，不落回 20。使用者寫的 expression 留在 FormInput，Engine journal 只存普通值。

`MISTPlanForm` 另要求固定 gain start/stop/points 與 disturbance pulse，reset 可明確省略；gain 不放 tunables。
measure_mode 預設 excited，可選 ground 或 both。to_cfg 從固定 capture 的 $qubit.t1 取微秒 T1 存入 plan.t1_us，無值時為 None。MIST relax 有 T1 用 5*T1，沒有則用 tunables 的 relax_factor*guess_us。
Overnight forms 保留固定 pair 與 interval。
`relax_factor` 乘 T1 估計，不乘 Echo guess。T1 member 用 `t1_guess_us`；Echo 優先用本輪剛接受的 T1，未接受時用 `t1_guess_us`。
兩個 member 的等待時間都不低於 `relax_min_us`（預設 1 µs，允許 0），表單以 µs 顯示並傳入 tunables。Fit 不改 delay grid、pulse 或 readout。
`OvernightTunablesForm.uniform` 的標籤是 Uniform (linear) sweep，預設 true。設 false 只改 T1 在同一視窗內的 delay 分布，不改 Echo。
`OvernightMuxPlanForm` 使用手寫 FormModel 與 ModuleInputField，提供 mux readout、nullable reset、完整 tone model list 與 participants 原始索引。
每個 tone 分別選已校準的 x180 drive 與 x90 pi2_pulse，不從半 gain 推導。
Fresh defaults 從固定 capture 的 `$mux.qubits` 建立完整名稱順序；`$participants` 缺少或未綁定時選所有 qubit，已有選擇則保留其順序。
已選 tone 的 x180/x90 與 mux readout/reset 只引用現存 module slot。缺 pulse/readout 留待使用者填入，不建立替代 module；reset 可保留 None。
表單沒有 flux 欄位。`OvernightMuxTunablesForm` 沿用 overnight 的欄位與意義，包含預設 true 的 `uniform`。關閉時只改 T1 的共用 delay 分布，Echo 不變。
兩個 guess 分別填最長 qubit 的 T1 與 Echo，供所有 participants 共用各 member 的 delay grid。
Mux Echo relax 使用本輪已接受 participant T1 的最大值；全部未接受時才用 `t1_guess_us`。`relax_min_us` 與單 qubit 表單相同。
四個附件共用 measure headless form owner 與 FormEditor，不建立獨立 workflows Window 或第二份 session。
附件只負責 Start/forms/native run，沒有 export/sample binding。
Saved run conversion 由 coherence_fluxdep.export 與 mist_fluxdep.export 獨立工具負責，GUI/RPC/MCP 不提供匯出入口。

## 固定來源與 modules

Owner 的 prepare_forms/Start 用 read_context() 固定完整 alias 表與 sources，不宣告角色或檢查 kind。
Global qubit alias 必須可讀；EMPTY 可開 workflow tab，但來源依 base 契約拒 prepare_forms/Start。
Defaults 在同一 capture 解析 $qubit.module.x180.waveform.length，以具體路徑 expression 宣告，開表單時算成值；缺值、不可讀或不合法就留 pi_length_us 未填。
它是 required Rabi-range estimate，供 host seed 使用，不是 fitted pi，也不放進 native CoherencePlan。

`common.calibration_modules.resolve_calibration_modules` 集中 Coherence/MIST 的 pi drive、獨立 spectroscopy probe 與 pulse-backed readout 規則。
兩張 plan 表單的 `probe_pulse` 與 `qub_pulse` 並列，使用同一個 drive 家族 ModuleInputField。
新表單先引用固定 capture 的 `$qubit.pulses.qub_probe`。沒有時，複製 captured `$qubit.module.x180` 成 inline，設 freq 0 MHz、gain 0.1、waveform.length 5 µs，不改 library 或 pi drive。
缺 x180 時 probe 留空必填，不借 Template 或使用者稍後填入的 `qub_pulse`。既有 draft 不重新種值。
Frequency stage 保留 probe waveform/長度，頻率由 tracker 掃描設定，線寬自適應 gain 從 probe 的 gain 起跑。Rabi 與下游仍用 `qub_pulse`。
每次 attempt 的實際 probe gain 照舊存入 `frequency.probe_gains`。
三個 plan 表單用 `gui.form_adapter.ModuleInputField` 生成 tagged module 欄位，在 `converters(env)` 與 `options(env)` 綁同一份固定 capture，經公開 `to_spec` 取 key 或持久 Spec，再交 captured core/native factories materialize modules 與 assets。Inline 保留全部 module 家族，ref choices 仍依各欄位的原篩選，nullable reset 保 None。科學檢查仍由 workflows 負責。
Inline module 子表單與 `input_schema` 不含 `desc`，`to_spec` 使用 Spec 預設值。把 library key 複製成 Inline 時，description 捨棄 library 原值、改用預設。Pulse 不保存 `ro_ch`。Flat `PulseReadoutCfg.ro_ch` 是必填 ADC 通道，表單照常提供編輯。
Pure arb drive 不能改 Length-Rabi duration，回 qub_pulse consuming-field error；flat_top arb ramp 與 arb readout 保留。
這裡不提供 gain-Rabi、time-scaling、live repository lookup 或 guessed modules。

Coherence、MIST 與 Overnight 的新 flux_dev defaults 從固定 capture 的 $qubit.flux_source 取得 direct literal device name。
既有 direct draft 不隨 alias 修改，使用者仍可切 eval。合法 saved input 不改。
缺 capture、未綁 qubit 或缺 flux_source 時，flux_dev 留空並顯示必填錯誤，不補造 device。
Fake caller 明填 native；YOKO 明填 A 或 V。表單不推測單位、不轉換值、不預驗 device membership。
Native snapshot 沿當次 capture，後續 alias 修改、Prepare 或 Save 不重解析既有 run。

## 單一 run 與 remote

Headless tab owner 將 fixed context/storage、typed cached device mapping、software provenance 與 native plan 交給該 run 的 controller。
Start 只建一個 operation，pending token 不等於 Engine 已接受。Worker 持 RUN lease，透過 `await env.run(core.run, cfg)` acquisition。
Pause 保留 operation、lease、plan、accepted input、capture 與 Plots。Resume 保留同一 workflow coroutine，重試被 Pause 中斷的 operation，不在 Resume 讀硬體。
Device actual snapshot 只在每次 env.run callback 前讀取；native 保存使用該次 Completed.facts。
Stop 等 producer 返回與 terminal delivery 後才 finish。Busy tab 拒絕 close；replacement Start 或成功 close 才 release presentation，不 disconnect 共用硬體。

GUI、RPC、MCP 呼同一 WorkflowControlFacet，不另存 run。tab_id 識別 opened draft owner，run_id 識別 native run。
Start 的 plan/tunables 可用 path-array/value batch。省略/null batch 使用 shared parsed draft；明填 batch，包括 []，使用當次 fresh captured defaults 加 edits。
Host 用 FormState.write 一次套用 call-local input，invalid 留原 draft/base，成功才 reset 至實際提交 input。
GUI 可另傳 renderer-local errors 擋 Start。Transport 斷線不代表 native cancellation。

workflow.form 以 tab_id/form 讀 authored input、ordinary leaves 與 path errors，不附 schema。
workflow.describe_form 以 typed path 取得同一 candidate 的單層 metadata，[] 是 root。
Optional batch 只做 call-local preview，不提交或改 draft/base/accepted input/local text/Engine。
Tunables 使用 retained binding 的唯一 state/env。Complex tag 與 whole tuple 的 JSON array 規則沿共用 codec。

Tuning view 直接呈現 Host 的唯一 FormState，GUI Apply 與 RPC workflow.tunables 共用 facet.apply_tunables。
兩者整批提交當下普通值，不保存另一個 draft revision guard。
只有成功才更新 accepted input/base 並清 local text；invalid input 留在 shared draft，RPC 保留 field path/message。
Running 或 paused 都可 Apply。作者下次讀 `tun.current` 時取得新 revision；已捕捉的 operation request 不變。

Records RPC 直接投影 Engine committed records；point 現在含獨立 stage 與 enabled measurements。
Live/summary 只建立 enabled lifetime 圖。具名圖/PNG 沿 retained Plots 與 owner-thread renderer，status-only 不渲染。
宿主不輪詢 committed records 來畫 overnight 趨勢。Workflow 使用 `env.plots` 自行決定圖的資料與更新時機，不自動保存或重開 run 圖。

## Draft 與來源生命週期

每個 opened workflow tab 保留自己的 RAM parsed drafts。同名多開不共用 input。
Headless owner 的 prepare_forms 在 idle 時重新 capture sources，保留 authored input，
不把它重設為 defaults。Busy 時拒絕。Core FormState 的 restore_input 等共用能力
仍由 Form owner 提供，不表示宿主自動讀寫草稿檔。

舊 workflows app 的 SingleFileCaretaker 與 panel bounded reload 隨獨立 app 移除。
Measure 不自動保存或還原 workflow drafts，也沒有舊 app 的 --clean 草稿檔行為。
Measure 的一般啟動腳本、window layout 保存與 experiment reload 仍依其既有契約。
替換 workflow registrations 前要關閉全部 workflow owners，facet 不在已有 run 的
binding 上套用新附件。Run、capture、取消與保存邊界見 ADR-0062、ADR-0063、
ADR-0067、ADR-0068。
