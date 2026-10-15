**Last updated:** 2026-10-15，單層欄位查詢、lean cfg 與 Library／writeback Form

# `zcu_tools/mcp/measure/`

Measure MCP 透過 GUI 的 loopback remote socket 操作同一份 GUI 狀態。GUI core 不 import MCP。GUI `RemoteMethodEntry` 擁有 wire schema、exposure、readonly 與 operation policy。MCP 每次連線讀取 live `rpc.catalog`，不複製 policy 表，也不依 catalog 動態建立 tools。

## 固定工具與 RPC

`mcp_servers/measure.py` 把使用者的 recipe definitions 注入 session 與 `assembly.py`。Registry 驗證名稱、手寫 schema 與 callback，工具表拒絕不一致的注入及重名。Framework 不反向 import 使用者 recipes，也不從 GUI catalog 產生 tools。共用工具為 `connect`、`status`、`wait`、`cancel`、`finish_early`、`rpc_list`、`rpc_describe`、`rpc_call`、`tab_analyze`、`tab_interact`、`tab_close`、`answer` 與 `apply_writeback`，以及 `simulation_initialize`、`device_setup`、`recipe_guide`。

## Setup 工具

`recipe_guide(recipe)` 使用 registry 的 adapter mapping，回傳 recipe、adapter 與 `guide: {markdown, resource_dir}`，不從 recipe 名稱猜 adapter。Markdown 保留全文；resource_dir 是 GUI host 的絕對目錄字串，未宣告 guide 時是 null。Guide 中的圖片使用相對路徑，agent 需要時自行從該目錄讀取。工具不讀圖、不附圖；跨機 agent 可能無法存取這個路徑。`simulation_initialize` 接現行 GUI coordinator，重用有效 simulation，切換時可能斷開真實裝置。它不套用 project，也不連真實硬體。Caller 仍須取得切換授權。Ready 只核對 MockSoc 與 connected fake_flux，不依賴 predictor。全域 status 不讀或投影 installed predictor。

`device_setup(name, cfg)` 接受 partial Info patch。工具先讀 before snapshot，合併 before.info 與 caller patch，再送一次完整 `device.setup(name, cfg)` RPC。YOKO patch 含 value 時，caller 必須明填 value_unit，不能借 before 的 unit。只改 output／rampstep 不需要 unit。欄位必填、extra、mode/unit 與 limits 由 RPC 的同一 Info model validator 拒絕，connected admission 由 GUI DeviceService 擁有。非 YOKO 沿各自的 Info 契約，不加 flux 規則或單位換算。

Raw RPC 是非同步完整 cfg 入口，不接受 partial updates、不補 cached 舊值。Caller 先讀 `device.snapshot`，修改完整 Info cfg，呼叫 `device.setup`，用 `wait(op)` 等待，再讀 snapshot。全部 model-required 欄位必須提供，default 欄位沿 Info constructor。YOKO mode 必填；raw RPC 可省略 value_unit，由 Info 依 mode 產生。這不取代固定工具對 value patch 的明填要求。

`device_setup` 沒有 timeout 參數，固定最多等待 300 秒。逾時回 running 與已知 op，不取消；caller 用 `wait(op)` 續等。需要短等時使用 raw RPC 與 caller 的 wait。Client deadline 必須大於 300 秒並預留 transport/reply overhead。`simulation_initialize` 保留原可調 timeout 與 verification.ready。

兩個 mutation 工具保留 pre_read、start、wait、post_read、requested、before/after 與 native operation。Device requested 是 `{name, cfg}`，cfg 在 before-read 與合併完成前為 null，之後是完整候選，不保證 model 接受。Before.info 為 null 時不能合併，pre_read failed。Before 有 Info 則送一次 RPC，由 service 拒絕未連線裝置，回 start failed、完整 candidate、before facts 與 null after。Device 不做 verification；snapshot 有目標值不代表 operation finished。Native finished／failed／cancelled、driver error 與 unknown 原樣保留。Step completed 只表示該步成功。工具不重連、重送或回滾。Caller 用已知 op 的 wait 和完整 snapshot 接手。

## 日常與細部操作

日常量測優先 recipe，既有資料分析使用共用分析工具。細部 setup、cfg、保存、writeback 及排查由 RPC 承接。日常／排查是使用指引，不是權限模式。`rpc_call` 接受 catalog 中全部 `rpc` 與 `tool` methods；tool 名稱只提示高層入口。GUI 仍驗證每次操作。公開 method 的參數、回覆及前置條件由 `rpc_describe` 提供。

Predictor 透過 `rpc_call(method="predictor.predict"|"predictor.calibrate"|"predictor.info", params=...)` 查詢。每次恰填一個來源：`qubit="Q2"` 讀目前選定工作點的具體元件，或 `params={"model":"transmon","EJmax":20,"EC":0.25,"d":0.1,"flux_half":0,"flux_period":1,"flux_bias":0}`（SQUID transmon，d 為 junction asymmetry），或 `params={"model":"fluxonium","EJ":5,"EC":1,"EL":0.5,"flux_half":0,"flux_period":1,"flux_bias":0}` 做獨立試算。qubit 不解析 alias 或 path。直接參數不需要工作點，不讀 dialog/cache。能量用 GHz；flux_half、flux_period、flux_bias 與 device_value 用來源的 native device units，不是 flux quanta；frequency_mhz 與預測頻率用 MHz。

`predictor.info` 回 `{model, params, missing}`，省略或 null 欄位保持未知。Predict 另填 device_value，Calibrate 另填 device_value 與 frequency_mhz。兩者需要完整有效參數，可填 from_level/to_level，預設 0/1。Calibrate 回 `{flux_bias_before, flux_bias_after, frequency_mhz, clamped}`，不寫元件、dialog 或檔案。兩種模型都選最小絕對 bias 的根，不選最接近目前 bias 的根。frequency_mhz 是實際達到的 MHz。不可達目標回全域最近頻率並標 clamped=true，同距時選最小絕對 bias。可達目標標 clamped=false。無效輸入仍拒絕，數值求解失敗不以 fallback 掩蓋。

來源皆未填或同時填回 `invalid_params/invalid_predictor_source`；無效具體元件名稱回 `invalid_predictor_qubit`；直接參數格式或物理值無效回 `invalid_predictor_params`。計算缺欄位回 `missing_predictor_params`，message 與 data.missing 列出缺項。工具沿 GUI 的 live catalog 描述與錯誤傳遞，不另建參數或寫回邏輯。沒有 target、load/set/clear 或 installed model。

Raw RPC 不提供高層工具的結果聚合或 PNG 解碼。例如 `device.connect` 回傳 handle，caller 用 `wait(op)` 等待後再讀 `device.snapshot`。`tab.save_data` 與 `tab.save_artifacts` 同步返回當次完整 Save report，不回 Save handle。Report 的 run_id 沿用 Run 時產生的同一個 ID，即使全部輸出失敗也不為 null。Caller 用各路徑的 status 與 error 判斷保存成功，不以 run_id 是否存在判斷。Report 保留全部 members，不用 await 或再次讀 snapshot。Readonly snapshot 的 confirmed run_id 仍只表示已有持久輸出成功。

## Recipe 與 execution

`RecipeTab.use_library` 選實際 library key，不建立舊 ref tree，也不複製內容。Slot 與 nested key 保存 tagged ref 的 kind/name，inline 保存 kind/module。持久 Spec 的 child key 仍是字串。`use_library` 只寫清單中 value 恰為該 ref 的 Option，以一次整個欄位 write 選 key。未列出的 key 在 RPC 前拒絕，訊息指名欄位並列可用 keys。Locked Slot 的固定 inline tag 也在 RPC 前拒絕 key selection；沒有 captured library 的普通欄位不因此變成 locked。

`read_field` 每次只送一次 `tab.describe_cfg`，從同一 owner turn 取得目前的單層描述與局部觀察，再回傳具名 RecipeFormField。不保留上次 observation，也不把兩次 RPC 拼成一份讀取。input 保留 Form JSON，input_present 區分未填與 null。value 是 ordinary leaf 或 subtree，不能作為有效性的證明。mode 是 key、inline 或 None，key 是實際 library key 或 None。fields 只列選中 Option 的可編輯子欄位，不含 tag；ref 沒有 fields。Core 決定目前選項，MCP 不匹配選項或推導 union。mode 只讀公開 S2 input kind。locked 使用公開 selection_fixed 或 fixed metadata，不因缺 key 選項而鎖定。單一普通 model wrapper 攤平後，fields 直接列其中子欄位，實際寫入路徑仍保留 module。errors 包含絕對 path/message。DTO 不含 CfgSchema kind 或 ref 子欄位 readonly 投影。Tag 葉不是 read_field 路徑。Inline 路徑包含 module，例如 `modules.readout.module.pulse_cfg.module.freq`。Range 使用 start、stop、n，set_sweep 的公開 expts 寫入 n。局部 errors 只含 Form preview 的 root 與該欄位／後代錯誤。Lowered cfg 錯誤請讀 get_cfg 或交給 Run admission 檢查。未知、inactive 或 whole-leaf 子路徑由 owner 回 invalid_params／form_path，data.form_path 保留請求路徑。

`RecipeTab.read_module(field)` 回傳具名 RecipeModuleValues，含 detached value、key／inline mode 與 source。Key mode 讀 Reset 的固定 library capture，source 是 library:<key>。Inline mode 只求該欄位的值，source 是 dotted core module path。其他欄位 invalid 不擋讀取，該 module 自身 invalid 才拒絕。值不含 extras lowering、整份 cfg 驗證或 native to_cfg。

`use_inline(field)` 走既有 tab.use_inline RPC。Server 先檢查 holder、busy、lifetime 與通知重入 guard，再以公開 read_module_kind 讀取 core-module 欄位的 input tag，不依賴 draft 驗證。Ref 模式執行同一個 to_inline 選項指令，從開表單／Reset 的固定 library 複製快照，以一次原子 write 發布。Inline 模式即使必填值未填或 expression 無效，也不求值來判斷模式、不改 input、不發布，但仍檢查 guards。回覆沿用lean Form observation，保留 draft 的 errors。停用或來源無效的欄位拒絕轉換，不補 fallback；要從範本／型別預設開始，明確選 Inline 選項。Caller 之後用 set 寫入 module 葉，例如 tab.use_inline("modules.qub_pulse") 後呼叫 tab.set("modules.qub_pulse.module.freq", 125.0)，不自己組 envelope。

`retain_module_source(field, summary_field=..., value_path=...)` 記錄通用 scalar projection 及 module source。summary_field 使用 dotted inline 葉路徑；key mode 也用相同身分，即使 input 沒有該葉。value_path 是 module-relative path，不含 inline envelope。Run 讀目前 scalar 值，保留數字而非 library key 或整個 module。無效 projection 不記成功前綴。後續 set 的來源可取代 label，既有 Run capture 不變。Frequency／linewidth policy 仍由 caller 決定。

Recipe 將既有 GUI 操作串成有界實驗流程，不在 MCP 複製實驗核心或 cfg defaults。參數來源、缺參數與分析分支由個別 recipe 宣告。首次呼叫等待最多 300 秒；缺參數、失敗或互動需求會提早交付。仍執行時回傳 execution，背景接續不因等待逾時而停止。

`onetone_spectrum_over_flux` 必填 `flux_device`、`flux_unit`、`flux_range` 與 `flux_points`，都不能為 null。Range 保留兩個 native endpoints，不換算數字。Unit 必須與 cached device snapshot 相同，native 也必須明填。Helper 成功後寫 cfg 的 flux_dev 與独立 flux_unit，不把 unit 附在 sweep 上。其他不符情境在 Run 前拒絕，不由 recipe 連裝置或修改共用安全規則。

`twotone_spectrum` 的 center_mhz／span_mhz／points 必填。三個 resonator spectrum 可省略 frequency range，domain helper 讀目前 frequency／linewidth 後寫數字 endpoints。正式 freq／power／flux 的 full-span 倍率為 3／3／2，省略 count 保留 GUI 值。四個 spectrum 摘要只保 sweep，不另計 center／span。Frequency／flux／unit policy 由 zcu_lab.recipes.helpers 擁有。

`RecipeTab` 提供通用 read_value、read_library_root、read_field、read_device、set、set_fields 與 set_sweep。Read_device 只讀 cached snapshot，不連線或換算。Set_fields 先驗證，再以一次 edit_cfg 寫入；None 保留值並標 gui_default。Retain_source 只記 caller 選好的 scalar/sweep label，不發 RPC，也不改 Form input。Framework 不解讀 calibration、角色或 linewidth。

Client deadline 必須超過 300 秒並留傳輸與回覆開銷。Stdio server 同步處理請求，首次等待期間不保證同連線的另一控制請求立即處理。Client timeout 不等於取消，不可因此自動重跑。

全域 `status` 的 `entry` 是 null 或 GUI `entry.info` 的 UUID、name、result_path、database_path，不投影 chip／qubit／resonator。它只讀 metadata，不建立 context observation。未 opened entry 時 `ready.missing` 包含 `entry`。

全域 `status` 列出 GUI operations 與目前 MCP session 的非終態 executions，終態只給數量及按 ID 查詢提示。`status(execution)` 讀指定 execution 的本地摘要，不重新連線。`wait(op)` 只觀察 GUI operation；`wait(execution)` 包含後續結果讀取、同步保存及當次 image content 交付。已接受的分析失敗以 outcome data 回報，和查詢失敗分開。Execution ID 不跨 MCP server session，也不是持久恢復機制。

`finish_early` 對 recipe 停止採集，有可用結果就先保存 raw，再繼續分析及保存。`cancel` 優先，停止後續分析與保存；已啟動且不可取消的保存仍等真實結果。Registered analysis 的 cancel 也不再啟動新的結果讀取。GUI cancellation 回覆獨立放在 `gui_cancel`，不能拿它覆寫 execution 的既有 terminal outcome。未註冊的 `cancel(op)` 沿用直接 GUI hook。

Recipe 不自動挑選重用 tab，也不自動清理。`RecipeSession.open_tab` 的 adapter、aliases、reuse 簽名不變。它先建立 tab 或驗證 reuse，再以一次 `tab.reset_cfg(aliases=...)` 捕捉來源，之後才設定參數與 Run。省略 aliases、None 與 {} 使用 Reset 當時的全域列。非空 mapping 只覆寫這次 Reset，不寫 alias 表，也不保存到 memento。Target 對到 Reset 的 qubit alias。

Parameter/library reads、cfg、Edit 與 Run 使用同一固定來源。`tab.parameter_read` 不收 aliases，`tab.get_cfg` 不作臨時 preview。`tab.run_start` 在 focus 與 operation 效果前拒絕非空 aliases。None 與 {} 使用畫面上的值。全域 source 變更不刷新既有 Form，取得最新值要明確 Reset。關閉由 `tab_close` 指定。

Load 只載既有結果及分析來源，不覆寫 Form input。RPC/MCP 的 Load 回覆保留結果與分析準備 facts，不含回填狀態。要看數據檔當時的 native cfg，使用結果 Inspect。

Author Run 的 cfg、來源與 completion 由 `recipe.py` 的 opaque handles 擁有。Framework 讀 detached snapshots，不讀 handle 的 private backing。`raw_save.py` 捕捉原 Run 的當次同步 Save report，保留全部輸出與該 Run 的 run_id，全部輸出失敗亦保留這個 ID。Author handle 使用這份保存契約。沒有 SaveReport 時 steps.raw_save 為 null，不推論 not_started、unknown 或 Save 是否已送出。save_raw 的傳輸與 admission 失敗照常拋出，recipe 回覆的原因只在 error；不加 requested 旗標。

Author analysis 綁定原 Run 與同一 Run 的 Primary。它重用 `AnalysisExecution`，在 recipe 的 worker 完成分析、圖像與候選 capture。Recipe-local close 和 session close 都能停止接續。寫回提問只篩選已捕捉的 stable names，不重讀 GUI 或寫入。`recipe_execution.py` 擁有 generator、單一 recipe admission、一次性取消、captured question 與 typed snapshot。它以同一 worker 完成 author handles；analysis 共用 driver 的本地 condition 交付進度。Session 與 tool table 使用必填的同一份注入 definitions。Guide 與 analysis-only scalar estimates 也讀 owning registry。單一 recipe 包含 interactive 與 awaiting_answer；terminal 之後才放行下一個呼叫。

## 分析、互動與寫回

Recipe 以 yield 提出 completed capture 的寫回候選。`answer(recipe, decision)` 的 recipe 是 execution ID，decision 為 accepted 或 skipped；它只回答，不寫入也不授權。普通 `tab.accept(items)` 寫 GUI 當前 draft。摘要的 `question_items`／`question_preview` 是 active question，`writeback.receipts` 則列出實際寫入與失敗前綴；兩者不能互相冒充。Query 只讀 local capture。Session close 喚醒並收束 generator finally。

`tab_analyze` 在既有資料上啟動 Primary 或 Post 分析。Execution 捕捉結果、實際參數、失效內容、完整 analysis artifact 的 Save report、當次 image content 及 writeback 候選。互動分析立即交接 tab、op、狀態、可用命令與圖像。

Analysis completion 等原 GUI operation 完成，同步保存完整 analysis artifact，再以原 opaque operation 讀一次 writeback preview。沒有圖像也讀候選。Interactive handoff 不 capture，`done` 加入原 completion，完成後才 capture 定案 draft。屬於 recipe 的 done 等待該 recipe 的下一個結果／提問，不回另一個 analysis execution；standalone 保持原路徑。Recipe 分析由自己的 worker 完成同一份 capture，不另建等待 worker，也不另讀候選。Full 的 writeback 為 null 表示尚未取得；has_draft=false／items=[] 才是已確認空 draft。Read 失敗以 writeback_read error 回報，保留已知 result、當次 Save report 及可用 image content。來源 superseded 不改讀新 pane，不 retry，也不重新授權。Recipe wait 在 done 後等待本地 completion 的交接，不再提早交付舊 interactive handoff。

GUI 的分析投影把非有限 summary 數字換成 null，以 `invalid` 記錄欄位路徑與原因。
Execution 保存同一份投影，recipe、`status(execution)` 與 `wait(execution)` 不重新推導原因。
不可估誤差不刪除有限 fit value、warning 或已確認的保存路徑，也不觸發自動寫回。
這項表示轉換不改 operation 的 failed outcome 或 generic context 的拒絕規則。

Framework 原樣傳遞 native analysis summary 與 invalid 路徑，不特別解讀 fit_quality、不重算品質或設定寫回門檻。有擬合的 summary 附 fit_quality 是實驗定義者的約定，見 zcu_lab README。

Analysis-only 沒有 recipe 身分。摘要使用注入 definitions 已宣告的 scalar value_key，完全相同的 name/error_key/unit 宣告可合併。Present key 的宣告衝突直接 ValueError，不猜實驗身分，不新增 tab.snapshot 讀取。每個 pane 保留完整原生 summary 與 invalid；scalar estimates 不抽掉 summary 的欄位，framework 不投影 quality。

`tab_interact` 省略 payload 時讀 committed state、commands、info、preview_active 與 figure，不改焦點。帶 payload 時 GUI 驗證命令並跟隨 Analysis pane。`done` 接住原 analysis operation，然後加入其 execution 的完成讀取與保存。此 method 的純讀與 command 都要求 holder，已接受的 owner-loop commit 依序生效。沒有來源鎖或自動重試。

Recipe、`tab_analyze`、`wait(execution)` 與 `tab_interact(done)` 使用同一 execution 摘要。摘要列出 Run 前捕捉的 resolved 條件與來源、Primary/Post scalar estimates、原生 summary 與 invalid、全部候選及 destination 身分。Run 或分析已送出而 receipt 未確認時保留 unknown，不從缺少 handle 推斷未啟動。`run_id` 取已捕捉 Save report 的原 Run 身分，全部輸出失敗也保留同一 ID。沒有捕捉到 report 時才為 null。各 output 的 status/error 表示保存結果。這與 readonly tab snapshot 的 confirmed run_id 不同，後者在尚無持久輸出成功時仍為 null。Execution ID 仍是 session-local。

Recipe 的 actual 是平面 mapping，key 沿 summary_parameters.name，每項保 value／source／unit。Scalar source=explicit 或 sweep 三個 component 全為 explicit 時省略，mixed sweep 整項保留；全部明填時回 {}。Recipe/helper 把 agent 明填值標 explicit，通用 API 仍接受 custom label。Summary 與 `status(execution, detail="full")` 使用同一份固定 actual。Run 內部只保宣告摘要 facts，不保存 publication、input 或 source_basis 副本。Cfg commands 不使用 ref，actual 與 candidate 也不顯示它。Full 仍保 captured analysis results 與 writeback proposals。查詢不新增 RPC 或授權。Artifact 沿用 producer 的 section、名稱與完整 members，每個 member 保留全部路徑、format、status 與 error。Reserved 不代表已保存。Partial 保留當次成功與失敗輸出；resave 以當次 report 取代選中 artifact 的輸出，不累積跨次成功路徑。

當次工具回覆可附記憶體中的 MCP image content，不落地 session 暫存 PNG。之後要看圖，用 `tab.get_figure` 取得 GUI 當下圖像，不保證是先前回覆的圖。`status` 不附圖片，也不重新擷取圖像。

`artifacts` 只記錄持久輸出的當次保存結果，不帶 `lifetime` 欄位。Execution ID 只在 MCP session 有效。持久路徑是本機已確認的輸出。同一 run 再 Save 可覆寫 run 成員，Labber 每次另產新檔。保存紀錄不保證檔案仍存在，查詢不自動 stat 或補存。

`apply_writeback(tab, items=None)` 按 preview 的穩定 `target_name` 選擇當前候選。省略 items 或傳 null 寫全部，空陣列寫零項。它不看 GUI 勾選，也不回答 recipe。工具先讀 Primary 與既有 Post 的當前 preview，再檢查未知、重複及跨 stage 同名；全部檢查通過後才用當前 session ID 寫入。寫入順序是 Primary 再 Post，首錯停止，不 retry 或 rollback。收據只列 GUI-confirmed completed，並保留 skipped、failed stage、not_started 及失敗 stage 的不確定性。

摘要列出每個候選的 target、proposal 與 current，full 保留勾選旗標。Destination 摘要只列 context/project 身分，native context readiness 保留在 full。Preview 與摘要不授權寫入。Caller 寫入前核對提案與當前目的地，GUI 仍在 dispatch 檢查 holder。候選內容的調整使用 `tab.writeback_*` RPC。Module edits 使用與 tab Form 相同的 path 陣列，回 input／values／errors，只有有效 Spec 可提交。Library 使用 `context.ml_edit`，child key 保持 key，不用 editor handle。`editor.*` 與舊 eval／value_ref wire 已退休，`value.list`／`value.read` 保留。

MCP 以 `context.aliases_set` 整表套用 authored rows，以 `context.set_parameter` 寫目前 point 的具體路徑。兩者直接走 owner，不開 Inspect、不切頁或焦點，也不改既有 Form／Run 的固定 capture。Aliases 在 Inspect 第三頁，`dialog.screenshot` 只拍目前頁，不接受 tab selector。

## 連線與觀察

`connect` 可連既有 GUI 或依 launch 參數啟動，這一步不連硬體。首次 label 預設 `agent`，mode 預設 `owner`；`readonly` 永不取得控制權。Owner 在未鎖定 Local 自動取得。GUI 鎖定或別的在線 holder 不阻止連線，回覆的 `control` 顯示 `{holder, locked, online, me}`。身份初始化在認證與 wire 檢查之後、catalog 之前。省略 label/mode 的重連保留原身份，顯式換身份會重開 socket。

`rpc_call` 提供 `control.status`、`control.acquire`、`control.release`。重複 connect 同一身份只查詢現在的控制狀態，不自動再 acquire。四種控制拒絕 `not_holder`、`control_locked`、`control_held`、`readonly_connection` 在 MCP `isError` 的文字 content 中保留 JSON `error`，包含原生 code、message、reason、data。GUI 負責每次 dispatch 的權限判定，MCP 不複製 holder 規則。

Control token 留在 session 供重新握手；顯式切換 port 不沿用前一 GUI token，tool log 遮蔽 token。認證失敗與 wire 不相容分別回報。Bridge 不會把仍活著的自有 GUI 當成可在另一空 port 再啟動的程序。

Session 將 GUI operation ID 映射為 opaque integer handle。明確重連或 EOF 後清除 catalog 與舊 handle 對應。GUI 可重用 wire ID，MCP session 不重用已外露 handle。GUI-origin operation 經 `status` 使用同一映射。每次 wire 操作前先確認連線，再解析 handle；不向新 GUI 重送舊 ID。

GUI 在 owner dispatch 檢查目前 holder，不維護每連線 seen baseline。MCP 不保存第二份 authority 或版本表，也不以隱藏讀取解鎖 mutation。只讀 method 可由 observer 使用，`tab.interact` 的純讀與 command 都要求 holder。控制身份與 domain readiness、busy、source-bound operation 及 authentication 分別檢查，見 [ADR-0070](../../../../docs/adr/0070-control-identity.md)。

接手既有或重啟後的 GUI 時，按操作需要讀 tab、SoC、context 與 device snapshots，核對實際狀態，不是建立寫入許可。Context 可能包含大型敏感資料，不可序列化時完整讀取失敗。摘要與 resource counters 不代替內容。

Cfg 使用 `tab.get_cfg` 的 Form input/values/errors。欄位能力用 `tab.describe_cfg` 逐層讀取，不下載整棵 schema。`tab.edit_cfg`、`tab.reset_cfg` 與 `tab.run_start` 不收 ref。Edit 作用於目前 input，Reset 重新捕捉來源及 defaults。Run 只接受目前有效值，並隔離執行值與 source provenance。後續編輯不改已接受 Run，轉移或斷線也不取消它。新 holder 可取消，observer 可 wait。

Admission 拒絕、斷線或 timeout 都不自動重送。傳輸失敗不證明 mutation 沒有副作用；先讀現況，再由 caller 決定下一步。

## 傳輸與關閉

Shared SocketTransport 以 shared framing 限制每個 frame 為 8 MiB UTF-8 bytes，不含換行，不分批。超大 request 在送出前拒絕，連線仍可用。超大 response 關閉連線並回 `message_too_large`，不能假定 mutation 未執行。重新連線會初始化身份並重載 catalog，舊 handle 不跨連線使用。

`tab_close` 與 RPC `app.shutdown` 都由 GUI 在同次 owner dispatch 檢查 active operations 及 unsaved artifacts。`discard_unsaved` 不能略過 busy。Shutdown 回覆的 `shutting_down` 與 PID 表示已接受正常關閉要求，不保證程序已退出；RPC 不等 process exit、不強制終止。

MCP server 結束時斷線並加入其 workers。它不關閉 GUI，也不保證停止硬體。MCP 不訂閱或佇列 GUI events；完成情況透過 snapshots、operation wait 與 execution 讀取。

## 驗證

`tests/mcp/measure/` 經 public tools/session 與 recording transport 驗證 recipe、分析、catalog、連線、holder admission 與 operation。GUI remote/service 測試驗證 socket 與 GUI-origin 路徑。離線選集只用 fake/mock，不啟動真儀器；fixture 與測試歸屬見 `tests/README.md`。

Entry new/open 只建立或開啟 entry，不自選 point，也沒有參數環境。Agent 先明確呼叫 point.use 或 point.new，才能 Run、WB 或目前 point 的 Inspect writes；未選取時沿既有 failed precondition 拒絕。WB 使用候選捕捉的 SavedPoint，選取其他同 entry point 不更換目的地。
