---
status: accepted
---

# ADR-0062 — 實驗執行與 workflow 編排

**Last updated:** 2026-10-27，operation 事實與 acquisition 日誌分工

關聯 [[0066]]，operation 與取消。保存權威見 [[0063]]，表單與 cfg 的使用邊界見 [[0065]]。

## 問題與決策

具名 workflow 是 `@workflow(name, description)` 宣告的單一 `async def (env, plan, tun: Live[Tunables]) -> Summary`。Plan、Tunables 與 Summary 從簽名推導，作者用區域變數保存 history，以 await 等待 operation、Ask 與 commit，以 return 結束。沒有 init、State 或 Next／Done／Aborted 步驟模型。非 None 的 BaseModel Summary 只在 done 時寫 summary.json，不算 commit，也不新增 GUI／MCP result 投影。

Pause 立即中斷當前 operation，保留作者等待中的 await。Resume 後只重跑同一 request 的 stale partial，不重跑 workflow 或已自然完成的結果。Running 與 paused 都可 Apply tunables；新值在下一次 `tun.current` 讀取生效，不追改已捕捉的 cfg。回到過去某點要開新 run、切 plan 範圍，並由作者用 saved_run reader 初始化 history。使用者於 2026-10-22 裁定：「好吧，你說服我了，採用你的建議，可以記錄進adr/discussion中」。選項比較見 [單一腳本討論](../discussions/2026-10-22-workflow-single-generator.md)。

Workflow 擁有 cfg 組裝、分析、校準接受度與可恢復 Failed 的政策。Engine 擁有控制、operation 事實與 commit 保存。其他例外停止 run，不改標為 Pause。

## Capability 與 driver limits

Workflow 作者可在步驟開頭檢查所需的 soc、devices 或 context 並 fast fail。Engine 不維護 requires 宣告，也不保證啟動前完成 capability admission；未自行檢查的缺少 port 在使用時失敗。使用者於 2026-10-08 裁定：「workflow定義者可以開頭自己檢查是否有需要的而fast fail，這不由框架保證」，並補充「我認為可以信任workflow定義者，只要做best-effort防呆就好」。這不放寬各入口既有的 holder、RUN lease、driver safety 或當次硬體授權。

裝置 limits 留在儀器 driver，不另外建立 workflow 宿主 limits policy。這讓各前端使用同一份領域檢查，而不是只有 workflow 被保護。Driver 失敗向外傳播，不 retry 或 clamp。使用者於 2026-10-08 說明：「儀器driver hardcode檢查，這樣最有保證且符合領域認知集中」，並確認既有 device 不需改動：「目前的device就已經可以了，為什麼要改device?」。

`env.call` 是唯一 operation 原語，`env.run`、`env.setup_device` 與 `env.wait_until` 是它的 typed 糖。作者直接 `await env.ask(question)`，Future 留在共用 async host 內部。Engine 使用 `experiment.script_host.ScriptHost`，控制命令、request_id 與 scope 樹依 [[0075]] 的「Async 協作與控制命令」，不另建協定。使用者於 2026-10-22 對「一個原語 env.call，env.run 是 Experiment 的 typed 糖」回覆「可以」。這取代舊 Effect 與 generator 協定。

序列相同時直接用 core.run，例如 `await env.run(T1Exp().run, cfg)`。自訂序列寫與 core 相同的 `(cfg, *, context: RunContext) -> Result` 同步 helper，cfg 是位置參數，其參數名稱不限定。作者也可用 async helper 包裝 env.run。分析政策與接受門檻仍由 workflow 擁有，不經 adapter Form 或 adapter analysis。使用者於 2026-10-22 裁定：「我會預期這個run_t1的接口與core的run一致，這樣我就可以按需直接用core的run，有客製化需求也可以獨立寫一個一樣接口的helper，而env. run的工作就是提供run的接口，最好是能復用Run類」「補充，我說的run_t1是你剛剛指的_acquire_t1」，並以「OK」確認採用計畫。

`env.run` 每個 attempt 建立新的 RunContext，借用 soc、soccfg 與 connected drivers，使用獨立的 operation Plots 與 operation child token。框架在 worker 呼叫前捕捉 device snapshot，再以 cfg deepcopy 呼叫 fn。它回 `Completed[RunRecord[Cfg, Result]] | Failed`。Completed.value 是 RunRecord，Completed.facts 是 OperationFacts，與框架寫入 journal／該 commit 的 record.json 信封使用同一事實結構。Schedule acquire error 回 Failed，build、conversion、callback 與其他例外 raise。不符合此 typed run 簽名的操作用 env.call，不另造 workflow Run、buffer 或 schedule 糖。

框架按每次實際 operation attempt 記 kind、iteration、attempt、時間、snapshot、outcome、reason、stopped 與 name，不新增 Acquired 或 operation run_id。
Name 保留作者傳入的 operation label，舊記錄缺少時讀成 None。
Completed.facts、journal 與 point record 的 operations 信封共用這些事實，包含 Failed 或 discarded stage。

使用者於 2026-10-26 裁決：「是作者的自由，框架只提供工具。」及「我想把AcquisitionFacts機制刪除或取代」。
框架不要求 run 使用 ProgramBuilder，也不收集僅覆蓋該路徑的 acquisition facts。
ProgramBuilder 每次 acquire retry 寫一行 warning，包含原因與下一次 attempt number。
Data-driven early stop 寫一行 info，包含 stop value 與完成 rounds。
Operation、native RunMetadata 與 RPC 不攜帶這些 program observations，workflow 摘要也不聚合其計數。
舊文件與 journal 的 `acquisitions` key 在載入時忽略，不轉換或寫回；缺少此 key 不影響載入。

Software provenance 按 workflow run 記錄，manifest 與既有 journal run_started.identity 副本保留，不逐 operation 複製。`env.identity` 提供 detached startup RunIdentity。Lab saver 顯式用 facts、identity、spec.tag 與 snapshot 組裝既有 RunMetadata，不自行捕捉 AcquisitionEvidence。RunRecord.metadata 與 native Container 的保存邊界依 [[0080]]。使用者另修正：「框架不應該知道run裡面是跑acquire或是其他」「workflow有需要run_id嗎？他沒有一次run的概念吧？只有commit的概念」。Measure headless owner 與 Qt pane 的分工仍依 [[0067]]。

## 表單 input 與 Engine 普通值

FormInput 保存 expression 原文；宿主捕捉求值環境，決定 Start、Apply、Discard 與 input 保存時機。Start 與 Apply 使用同一個宣告的 Form.to_cfg() 轉接。Apply 將完整 native tunables model 交給 Engine，不攤平表單路徑。Engine 依 workflow 宣告型別重新驗證完整 model，整體替換並增加一次 revision，不比對差異。相同值也建立新 revision。Engine 不求值或保存 expression。每次 Apply 的 journal 記一行 actor、revision_before／after 與完整新 tunables JSON，不記逐葉 old／new。Journal append 成功後才發布新值；stale revision、model error 或 append failure 不發布候選。`tun.current` 每次讀取最新完整 model。已捕捉的 cfg 不隨新 revision 改變，Resume 重試仍使用原 capture；要套用新 cfg，由作者在下一次讀取後建立。

持久 snapshot 的 plan 與 knobs 各存固定 FormInput data／selections JSON。啟動、source reload 與 RAM switching 共用 Core 的依名與形狀遞迴還原。新欄位用 defaults，無法還原的欄位或 selection 列 dropped 並顯示。還原不呼叫 converter 或 options hook。壞 JSON 或格式不符時顯錯並以 defaults 開啟，不備份。共用契約見 [[0079]]、[[0063]]。

Workflow host 只有在 Engine 成功 publication 後才接受 input／base。失敗保留可修正的 draft，不以輸入已編輯推定 Engine 已接受。這避免 Engine 維護另一份 input 狀態，代價是 Engine journal 無法重建當時的 expression。Form／editor 的共用與宿主提交邊界見 [[0065]]，具體接線見 [measure workflow host](../../lib/zcu_tools/gui/app/measure/README.md#workflow-hosting)。使用者於 2026-10-16 以「都採用建議」確認將表單草稿的這些穩定邊界併入現行 ADR。

使用者於 2026-10-18 裁定：「基本正確，但我認為不同比較不同，直接用就行了，也不用記錄。」選擇日誌格式時確認「甲乙都可以」。採完整快照保留更新者與 revision，不增加遞迴比對。List 隨完整 model 提交，不需要 native index path。Commit record.json 記本輪實際讀過的 tunables revision，去重並保留讀取順序，不只記最大 revision。不增加舊 journal reader 或格式遷移。Form 的 typed path 編輯入口仍由宿主擁有，見 [[0079]]。依賴方向沿用 C8、C9、C18 與 form-core-no-ui-policy。

## 一個 run 的 operation

一個 run 從 Start 到終局使用同一個 OperationRunner operation 與 handle。Start 只 begin 一次。Controller 是 run 的單一 owner，GUI 與 RPC 使用同一個 target 核對與提交入口。Run binding 在通知前提交，pending target 不表示 native startup 已接受。未接受的 run 不能更新 tunables，也不能借用上一個 run 的 draft。

宿主在一個 worker 呼叫一次 Engine.execute，整個 coroutine、Pause 與 Ask 都留在該 worker。Pause set 當前 operation child、增加 epoch 並關 gate，不停止 iteration。Stale partial 留 discarded 事實，Resume 開 gate 後 env 重跑同 request 與原 cfg；自然完成的完整結果只等 gate，不重跑。Plan、form capture、accepted input、progress、handle 與本程序 RUN lease 都保留，不 settle。Resume 與 Answer 不另排 worker segment。

Complete、Stop 或 failure 只有一個終局入口。Stop 使進行中及之後的 operation 合作回 Completed(partial_result=True)，Ask 回 None，不向作者注入 CancelledError。宿主等待原 worker 的 finally、executor drain 與 owner delivery；同步 startup failure 也由同一 owner 收尾。清理完成前不能 replacement Start。終局完成才 settle handle、清理 progress 與釋放 lease，清理失敗遵守 [[0066]] 的 best-effort 規則，不承諾跨 State、channel、檔案與硬體的原子交易。

本程序 RUN lease 在 paused 期間仍保留。它不仲裁其他 GUI process 的儀器操作。Measure-gui 在另一 process 執行實驗，不代表 workflows 的 cache 已刷新，也不保證儀器保持某個值。

## 裝置讀回與 Resume

Start／Resume 不再讀取所有已連線裝置，也不在 journal 記錄 device snapshot。Start 表單直接使用 DeviceService 的公開 typed cached native mapping。Resume 只接受 run_id；健康 paused run 的提交失敗不建立新 operation，不清掉原 handle／lease。

供 acquisition provenance 使用的 device 實際值由每次 env.run 在同一 RUN lease 下、experiment callback 前捕捉。DevicePort.get_connected_devices 回 detached driver mapping，env.run 用它建 RunContext，再以 capture_device_snapshot 讀取全部 connected drivers。OperationFacts 保存這份固定 JSON，框架寫入 journal／commit 信封，lab saver 用它組裝既有 native metadata。讀取失敗使 run failed，不呼叫 experiment，不使用 cache fallback。缺 devices capability 不等於零 device，真正零 device 的 host 明確提供回傳 {} 的 port。SessionDevices 在 worker 借公開 driver mapping，getter 回 detached 的實際 Info 子類。完整 setup 用 selected driver.info_model 重新驗證 dump 後呼叫 driver.setup，傳同一 StopSignal 的 event。Journal 記錄 name 與完整 cfg；框架不認識 flux、unit 或 device 種類。使用者 set_flux 工具重用已交付的 check_flux_unit。Host capture 重用 native acquisition helper；這個 run 前 capture 不更新 GUI State。Device 接線與 Window 的 SessionServices 組裝已完成，不能據此宣稱完整 CP-B 或 integration 全綠。

Run 結束後，host 另在背景讀取一次已連線儀器，由 State owner 更新 cached DeviceInfo，供 dialog 觀察。這筆讀回不改寫 run 前固定 snapshot 或已保存 provenance，不讓 cache 成為 Run 的控制命令，也不改變 acquisition capture 的失敗與保存契約。活動快照與讀回的分工見 [[0076]]。

共用 alias preset、裸 alias 求值與 FormState 的顯式字串 expression 已接線。普通字串仍是 literal，結構 selector 仍使用普通 typed tag。Coherence、MIST 與 Overnight 的新 flux_dev default 從固定 capture 的 $qubit.flux_source 讀取 direct literal device name。缺 capture、未綁 qubit 或缺來源時留空，由公開 form 指名 Invalid，不補造 device 或阻斷 panel；其他解析錯誤照常拋出。既有 direct draft 不隨 alias 修改，使用者仍能切 eval，合法 saved input 不改。Source reload 與 input 保存由具體 GUI app 擁有，不授權硬體操作。

各裝置的讀值不是跨欄位、跨裝置或跨 process 的原子 snapshot。同步 I/O 可能阻塞，沒有固定 query 或 settling 時間保證。Snapshot 只記錄讀回值，不要求硬體自動復原；workflow 的後續 setup_device 仍由 driver 設定值與驗證 limits。

## 獨立 workflow 轉換工具

使用者在 2026-10-12 裁定：「我希望B，現有實作邏輯能直接轉成匯出工具嗎？」及「A還有這個export_run應該是綁在每個不同的workflow下的小工具，因為每種workflow的輸出格式可以不同」。各 workflow 的輸出格式不同，因此轉換工具由各 workflow 擁有，不交給 live GUI run 或通用 dispatcher。

轉換 saved run 不需要 live GUI run。各 workflow 提供自己的小工具，GUI／RPC／MCP 不再提供匯出入口。Coherence 工具支援 Labber、samples；MIST 工具只支援 Labber 並拒絕 samples。Overnight 沒有本次要求的輸出，不新增空工具。每個工具提供可 import 的函式與 humble main(argv)，不建立通用 export_run dispatcher。

工具在 zcu_lab 讀固定 manifest 與每輪 record.json，只消費結構完整的 committed point，再讀 point 的具名 native reference。缺摘要列出每輪路徑，malformed／identity mismatch／缺檔指名 source。工具不讀 journal、不從檔名掃描或 leftover 檔建立提交點。這不是 crash resume，也不新增 framework reader 或保存保證。Shared typed envelope/Labber writer helpers 集中 Coherence 與 MIST 兩個使用者的輸入與失敗隔離，不做 registry 或 live Engine adapter。

使用者在 2026-10-25 裁定：「我好像沒有用merge的需求」。Workflow 轉換工具移除覆寫 `data_root/merged.h5` 的 merged 輸出，保留 Labber 與 Coherence samples。RunMetadata 的 run_id 契約不為 workflow root 放寬，單次 acquisition 的 native 保存不變。

輸出由目的地決定，不另設 outputs 選單或格式旗標。Coherence 收到 --result 就追加 samples，收到 --database 就寫 Labber，兩者都有就都寫。MIST 只收 --database 寫 Labber。未提供任何目的地時，函式與 CLI 都在入口拒絕。共同資料在任何寫入前先驗。Labber 使用既有 independent layout，在 SaveLayout 的新日期樹路徑保存。各 allocation/writer failure 分別回報，未配置 Labber path 為 None，已配置的失敗保留 path。不寫 ledger 或自動重試。

Coherence 的每個 enabled/measured kind 各有 Signal/Time Flux×Index 與 Flux-only lifetime，保留各自 actual delay，不 union、rounding 或插值。Rejected fit 保留 raw 與 NaN lifetime，disabled 不產 channel。MIST 保留 actual gain inner-first Signal 與 Flux-only Success。匯出以保存的 actual gains 為真相，不與請求的 plan grid 比較；各點 actual gain grid 必須完全相同，不同時指名兩點與值並拒絕。Samples 每點最多一行，有 accepted frequency 或任一 lifetime 就寫，只含接受值的 MHz/us 欄位。座標來自該點最後 Completed acquisition 的固定 native snapshot，按 frequency→Rabi→readout→T1→Ramsey→Echo，包含 calibration 與 rejected fit。全部 rows 先驗，owner 的 prefix/count/cause 與 latest-wins 不改。

科學型別留在 zcu_lab，framework export-only carriers 與 frontend binding/helpers 退役。依賴方向由 C8、C12、C16、C18 及 workflow-export-science-no-frontends contract 保護。該 contract 包含兩個工具、shared input/Labber helpers 與 samples producer。Datafile 擁有檔案格式，sample table 擁有追加與 latest-wins，工具不重做 HDF5 編碼。保存分工沿 [[0063]]。

## Iteration 目錄與保存責任

使用者在 2026-10-12 裁定：「我以為一直都是框架一開始就給這個iter的儲存路徑，workflow內部自己決定在哪裡存什麼」與「你沒辦法阻止workflow 內部亂搞，也不該試圖保證不亂搞，這些機制都是過度工程」。本裁定將量測與保存分開，不維持舊暫存路徑及檔案發布介面。

`env.new_commit_dir()` 每次配置並建立新的 absolute `data_root/commits/<六位配置序號>/`。`env.commit_dirs()` 是惰性無限 generator，可用 `zip(plan.fluxes, env.commit_dirs())`，有限序列放前面。作者用 `await env.commit(record, d)` 提交到已配置的 d，省略 d 時自配。同 d 二次 commit raise。配置序號與 commit 序號分開，信封同時記錄，reader 按 commit 序號排序。配置目錄不換 iteration token，成功 commit 才換；未提交目錄與作者檔案保留，但沒有 record.json。

`env.run` 只負責呼叫 core 同簽名 callable，保留 cfg deepcopy、device capture、live plots、StopSignal 與失敗來源。它不解讀 callable 內部的序列。`Completed` 只表示 operation 返回，partial_result 表示合作中止，不代表已保存。Workflow 自行呼叫 native saver，決定存什麼、檔名及是否保存，並在自己的 point record 記錄檔名與用途。校準保存仍由既有 native writer 處理。Frequency、Rabi、readout 各自回傳 frozen 具名結果，包含自己的 status、reason、accepted scalars 與 immutable 檔名引用。Frequency widening 保留全部 Completed attempts，fit 拒絕也保留 raw 檔名；Schedule Failed／skipped 不捏造檔名。Caller 將各 stage 結果放進 point record，不用整體校準成功收據丟掉前面已接受的值。框架不提供檔案清單，不預建量測或圖形子目錄，不追蹤、掃描或保證 workflow 檔案。

Workflow saver 例外停止 run，不改標為量測 `Failed`。Pause、Stop 或程序中斷可能留下未提交或寫一半的檔案，框架不清理、不承諾 rollback。Point record 的檔名也不保證檔案存在或完整，由讀取者處理缺資料。

Engine 仍擁有 manifest、journal 與 best-effort `record.json`。結果 journal append 與 flush 成功後，Engine 才用同目錄 temporary 加 replace 保存摘要。摘要 I/O 失敗只記 log，不影響提交或下一步。Manifest 的 atomic replace 也保留。這兩種框架文件的替換，不代表 workflow 量測檔案有發布保證。依賴方向仍由 C18 與 framework-no-user contract 保護。

## Coherence 與部分校準結果

具體 workflow 的 stage 隔離不改每個 point 一次 commit 的邊界。Frequency 接受即更新自己的 history；Rabi 需要本點 frequency，readout 與 enabled lifetime 需要本點 frequency/Rabi。Readout 非接受時沿合法 seed／prior accepted 設定繼續量 lifetime，不用 prior frequency 或 pi 取代缺少的本點依賴。Carry 與 history 由作者 coroutine 的區域變數保存。Pause 只重試被打斷的 operation，不丟棄整點；Stop 讓作者處理 partial 並收尾，不增加 stage checkpoint。

Coherence 固定 plan flags，按 T1、Ramsey、Echo 量 enabled 項目。三份 lifetime history 獨立，只有各自接受的值更新 delay grid。Point 以 t1、t2ramsey、t2echo 三個欄位保存具名結果；未量測或未到達的欄位是 None。T1Result 是獨立型別，不繼承共同 lifetime result。各個 completed raw 都有明示用途的 relative native reference，讓轉換工具取得部分證據，而不從 acquisition 次序或目錄猜資料。

2026-10-25 使用者選擇 point record 的獨立欄位形狀。目前沒有真實量測資料，不保留舊 measurements 格式的相容讀取；generic saved_run 只依 caller 指定的 model 讀取。

同日使用者同意保留校準複合 stage，函式體依序呼叫單實驗 helper。Coherence 與 MIST 共用的 calibrate_point 依序呼叫 frequency、Rabi（含所選 AmpRabi）與 readout，再回傳 PointCalibrationResult，由主迴圈填 point。Expected failed/rejected 正常回傳；unexpected exception 不回傳 aggregate，point 只保留主迴圈已填的欄位，已存 native raw 保留在磁碟。例如 frequency 成功後 Rabi 內部 bug，point 沒有 frequency 結果，但 frequency h5 仍在。

校準與具體 lifetime policy 留在 zcu_lab.workflows。共用 CalibrationPulses／StageEnv 服務 Coherence 與 MIST 的三個 native stage，readout_due 讓兩個 caller 共用零基 flux index cadence。這些 module 不依賴 frontend，方向由 workflow-science-no-frontends 與 C16 保護。GUI forms/capture、單一 Controller 與 operation 邊界不改；不增加通用 pipeline、registry、export bridge 或 saved input migration。

## Overnight live 趨勢與保存責任

Framework 只提供 `env.plots` live plot 工具，不知道趨勢圖、具體 context 型別或 workflow 名稱。Workflow 自己決定圖的資料、畫法與更新時機。Overnight 的 scalar 投影與繪圖是科學程式，留在使用者 workflow；宿主不輪詢 committed records 來呼叫它，也不在註冊項目加入呈現 callback。框架不保證趨勢只使用 journal 已提交的紀錄。

使用者在 2026-10-12 更正：「1workflow框架不可以有calibrated認知或是overnight判斷，沒不用畫趨勢圖」與「我說的趨勢圖是workflow不知道趨勢圖，他只是準備可以liveplot的工具，讓workflow決定要畫什麼以及什麼時候更新」。這不是取消趨勢圖，而是將 live 繪圖的資料與更新責任交給 workflow。

依 2026-10-22 operation 繪圖邊界裁定，measure 的 headless workflow owner 管理 native Plots 生命週期，Qt pane 顯示圖像，remote 使用共用 PNG renderer。它們不讀 records 來重畫科學趨勢，也不持有繪圖 cursor。Workflow 層 env.plots 跨 Pause 與 iteration 保留。每次 env.run attempt 以同 host 的獨立 operation Plots 組新的 RunContext，返回、丟棄與失敗先 finish。新 attempt release 並整組替換舊 operation，terminal 保留最新一組。Pane 與 MCP 讀相同兩群投影，operation 名稱為 operation/<name>。Replacement Start 與 close 釋放兩組 presentation，不銷毀 caller 保留的 Figures。名稱與 lifetime 邊界見 [[0069]]。

框架不自動保存 overnight 趨勢圖，不從 run 目錄重開互動式圖，也不新增 env.save_figure、Engine 提交通知或 committed journal reader。Workflow 實作者若要保存 commit 目錄內的圖，在自己的 workflow 程式內處理。使用者可自行從 record.json 或 native 資料重畫。

Journal writer 與 Engine 提交順序不變。Iteration 檔案保存依上節由 workflow 擁有。現有具名圖 RPC、PNG renderer 與 Plots 工具也不變。依賴方向由 C18 與 trend-science-no-frontends contract 保護，科學投影不依賴 frontend admission。

## 取捨與 owner

一 run 一 handle 讓 Pause、Resume、tunables failure 與 worker failure 使用同一終局政策。代價是 paused run 保留本地 RUN lease，同程序的其他硬體 mutation 仍受 lease 阻擋。這不是從 artifact 還原的跨程序 resume，也不提供 hardware rollback。

通用迭代與 journal 契約見 [workflows engine](../../lib/zcu_tools/experiment/workflows/README.md)。GUI run owner 與 tunables 提交見 [measure workflow host](../../lib/zcu_tools/gui/app/measure/README.md#workflow-hosting)。具體定義與 T1 attachments 見 [使用者 workflows](../../zcu_lab/workflows/README.md)及 [GUI attachments](../../zcu_lab/workflows/forms.md)。
