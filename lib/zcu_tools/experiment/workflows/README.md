# `experiment.workflows` — 具名 workflow 核心

**Last updated:** 2026-10-22

**讀者：** framework-dev。具名 workflow 的執行、控制與保存契約。

此 package 擁有具名 workflow 的通用執行契約。具體 flux 清單、校準、分析與 record 語意屬於使用者 workflow，不屬於核心。架構分工見 [ADR-0062](../../../../docs/adr/0062-experiment-workflow.md)。

## 能力與資料

宿主以 `EnginePorts` 注入 clock、device、`Plots`、progress 與 detached context。繪圖沿用既有 `PlotHost`，artists 留在 host owner thread；核心不持 raw Axes，也不捕捉 Figure snapshot。宿主持有硬體連線與 worker，核心不建立執行緒、不 connect／disconnect，不依賴 Qt 或 GUI。Progress factory 沿用現有 runtime bar 契約，不能把會再次查 ContextVar 的 make_pbar 當成 concrete factory。

`Next` 配對步驟摘要與下一步 state。`Done` 不新增點 record，`Aborted` 表達明確的業務終止。`Completed` 配對 experiment 返回時的 Run.cfg 副本與 Result，workflow 自行決定保存與單次檔案格式。`Failed` 表達可由 workflow 處理的 Schedule 失敗，只攜帶失敗資訊。

`workflow(name, *, plan, tunables, init)` 將具名宣告附在原函式；init 由 execute 執行，catalog 由宿主建立。State／record 型別由 init 與 step 簽名推導，state 只需支援 deepcopy，不限定 dataclass。`WorkflowRegistry` 由宿主顯式填入，每次重載建立新 instance。Tunables 宣告檢查 JSON schema。Engine.update_tunables 接收完整的宣告型別 model、expected_revision 與 actor，重新驗證全部欄位後整體替換，不比對差異。每次成功都增加一次 revision，相同值也一樣。Package 內的 TunableValues 集中完整 model 驗證、deepcopy capture 與 revision。TunablesChanged 記完整新 tunables JSON 與 actor、revision_before／after，journal append 成功後才發布新值。更新以完整 model 提交，日誌使用現行格式。Engine 擁有 run_id 與 lifecycle 政策。當前 step 的 capture 不受新 revision 影響。

`InitEnv` 只提供原始 UTC start 與 detached context。`WorkflowEnv` 的 effect helpers 使用 `yield from`，把 request 與回應的型別交接留在核心。同步 `get_device_cfg` 是例外，它在 worker 直接讀取 live cfg。環境共用 injected `Plots` 與 engine-owned progress bars。WorkflowEnv 的 soc／devices 與 context 存取在缺 port 時立即 raise，供定義者在步驟開頭自檢。Engine 不做 capability admission，也不記錄 requires。硬體操作應經 effect 保留 journal 與取消；框架信任定義者，不強制阻擋直接操作。

`Run` 為 experiment 複製 dataclass 或 Pydantic model instance cfg，將同一個 StopSignal 與明確 env 接入既有 Schedule。型別本身與無結構輸入在入口失敗。cfg.dev 只放角色到 device 名稱，不觸發 automatic setup。同步 `env.get_device_cfg(name)` 回傳 detached 的實際 `BaseDeviceInfo` 子類。`yield from env.setup_device(name, cfg)` 傳遞完整 model 與同一個 StopSignal，journal 記錄 name 與全部 cfg 欄位。Host 以 selected driver.info_model 重新驗證 dump，拒絕未驗證的 model_copy 更新，再把 signal.event 交給 native setup。框架不認識 flux、unit 或 device 種類。Unknown／disconnected names 在使用時指名報錯，driver limits 與 I/O cause 原樣傳播，不 retry 或 clamp。使用者套件的 set_flux 工具重用 `check_flux_unit`，單位規則不在核心重做。Start／Resume 不讀裝置，也不記錄 device provenance；RunIdentity 只保存 run／entry identity 與 software provenance。Schedule 的局部 cfg 不倒灌回 Run，後一個 completed outcome 不覆蓋前一個非 completed。Run 保留 live callback 的原始 failure cause，Engine 依來源區分可恢復的 Schedule Failed、宿主 Stop 與核心失敗。即使 experiment 重拋本次 signal.error，來源也不改變。

每次 `env.run` 在宿主保留的 RUN lease 下、callback 前，經 DevicePort 捕捉全部 connected drivers。讀取失敗使 run failed，不執行 callback。缺少 devices 能力不是零 device；真正零 device 仍注入明確回傳 `{}` 的 port。`Run` 要求 caller 提供固定、detached 的 device JSON，直接 caller 先捕捉，確知無 device 時明填 `{}`。Native callback 與 saver 只傳遞這份資料，後續 acquisition／Save 不重新讀取。同步 getter 只讀選定 device，與每次 env.run 的全 device capture 用途不同。SessionDevices 在同一 RUN lease 下重用公開 connected mapping 與 native acquisition helper，不更新 GUI State。

Record encoder 將摘要投影成 detached JSON。非有限數字變成 null，曲線陣列與過大 record 會警告。只有摘要序列化可降級為 repr；deepcopy、必要 journal 欄位與檔案 I/O 不能借用此降級規則。

Engine 在每次 step 呼叫前建立 `data_root/iter/<至少六位 call_seq>/`。唯讀 `env.iteration_dir` 是已存在的 absolute Path，指向該 iteration 的根目錄。屬性不可重綁，workflow 仍可自行操作檔案系統。Engine 不預建量測或圖形子目錄，也不掃描、驗證、追蹤或清理 workflow 檔案。

`env.run(experiment, cfg, live=...)` 只執行量測。`Completed` 不代表已保存；workflow 自行呼叫 native saver，決定檔名及是否保存。需要讓工具找到的檔名與用途寫進自己的 point record。Saver 例外沿 workflow 的非正常例外路徑停止 run，不變成量測 `Failed`。Pause／Stop 可能留下未提交或不完整的檔案，框架不承諾 rollback。`ExperimentFailed.run_no` 只識別 invocation 內從 1 計數的 effect，與保存或檔名無關。

Package 內的保存 seam 只負責 typed journal、manifest、iteration 路徑與摘要。Journal 使用 Engine 的同一把 control lock，逐行追加並 flush，不宣稱 fsync 或崩潰接續。Manifest 只摘要生命週期，不保存每點 record 或 state，沿同目錄 temporary 加 replace。

每個已結束 invocation 在 `iter/<至少六位 call_seq>/record.json` 保存輸入與結果摘要。格式為 `zcu-workflow-iteration`，版本為 1，包含 identity、workflow、完整 plan、本次 capture 的完整普通 tunables、started 與 result。Started 保存 call_seq、commit_seq、revision、kind。Result 沿用 journal 的 StepCommitted／Finished／Discarded／Failed 欄位。具體檔名屬於 encoded point record，框架不解讀。摘要不保存 state，也不複製 native 檔案的 cfg 或量測結果；EncodedRecord 沿用 json／repr／null 規則。

Engine 在結果 journal append 與 flush 成功後，以同目錄 temporary 加 replace 原子保存摘要。執行中的 invocation 沒有 record.json，不先建 pending 檔。摘要序列化或 I/O 失敗只記 log，不改變提交、state publication、後續 step 或 run 終態。Journal 仍是 commit 唯一權威；record.json 不是 checkpoint，不支援 crash resume。核心不提供 committed journal reader。Workflow 透過 `env.plots` 自行決定繪圖資料與更新時機。框架不輪詢 records，也不保證圖只呈現 journal 已提交的紀錄。Workflow 實作者自行處理 iter 內的圖形保存，不新增框架存圖 interface。責任分工見 ADR-0062。

RunIdentity 持有宿主傳入的 native SoftwareProvenance。Manifest 與 run_started journal 在 provenance 底下保存同一份資訊，unknown 保留 None，不以預設 False 或字串代替。

Engine.start 只驗證並保存啟動資料。第一次 execute 才執行 init；每次 execute 在呼叫執行緒同步跑到 paused 或 terminal。宿主安排 worker 與 lease。核心以同一把 control lock 序列化控制、revision 與不可中斷的提交，不在 effect 執行期間持有這把鎖。Pause／Stop 不等待 effect。一般 Pause 或 effect 中 Stop 會丟棄未提交 invocation，以 driver.close 執行 finally。Next 已返回時仍先提交；Resume 使用已提交 state 與最新 tunables 重做未提交步驟。內部 typed run 保留宣告的泛型配對，Engine 只持有 detached control view。Terminal run 只有在 execute 真正返回後才能被新 run 取代。

`yield from env.ask(question)` 暫停並保留當前 invocation，不關閉 generator、不提交或丟棄 iteration。Engine 使用共用 `experiment.script_driver.ScriptDriver` 推進所有 step；既有 effects 包成 `Wait`，operation delivery 與 typed outcome 由 workflow host dispatch 填入。`pending_question(run_id)` 回傳 detached 的 authored `Question[Answer]`，只有等待答案且 worker 已返回時有值。沒有 timeout 或預設答案。等待 Ask 時 `resume(run_id)` 拒絕；一般 Pause 仍丟棄步驟並由 Resume 重做。

`answer(run_id, answer) -> RunStatus` 與 `execute` 都是同步 worker segment。宿主在同一 RUN lease、同一 operation 下排 worker，不在 UI owner thread 執行 answer。Engine 持 control lock 先經 driver 驗證具體 Answer model，再記 Resumed 並 reserve；錯型 raise TypeError，保持 paused 且不寫事件。驗證成功後釋放 lock，由 worker 送入答案並跑到下一次 paused 或 terminal，不需要另呼叫 Resume 或 Execute。當前 invocation 保留原 tunables capture；新 revision 只供後續 step 使用。`stop(run_id) -> RunStatus` 只設控制狀態與 StopSignal，不推進 generator。等待 Ask 時 Stop 回 stopping，宿主必須在同 operation／RUN lease 排 `drain(run_id) -> RunStatus` worker segment。Drain 以 None resume 作者，執行收尾後丟棄 iteration，回 stopped；收尾拋錯回 failed。Terminal owner delivery 之前不得釋放 lease 或接受 replacement。沒有 retained driver 的 paused run 可在 Stop 當下結束。

`WorkflowEnv.ask(question) -> ScriptGenerator[A | None]` 在取消時回 None，作者必須明確分支。取消之後的 Ask 也立即回 None。新的 `env.run` 不啟動 acquisition，回傳 workflow owner 的 `Cancelled(run_id, reason="cancelled")`，run_id 是 Engine 的 workflow run ID，不是 experiment ID。Run 的結果型別為 `Completed | Failed | Cancelled`。新的 device／wait helpers 也不啟動工作，直接回 None。這些 delivery 由 effect owner 填入，共用 driver 只負責推進與驗證 cancelled delivery。

`WorkflowEnv.plots` 直接提供注入的 `Plots`，跨步驟與 retry 保留同一個繪圖 lifetime。`live` 接 `LiveUpdate` 回呼，收到 buffer 維度順序的 float64 axes tuple 與 complex128 samples，包含 partial NaN。回呼同步執行，不修改或保留借用的資料；workflow 用 typed handles 投影實數，或以 draw 任意繪圖、save 保存。Run 只綁定一個 buffer，保留回呼錯誤來源，不驗證繪圖策略。宿主在 producer 停止後 finish／release，Pause 不結束 Plots。回呼簽名目前是顧問代裁，待使用者審查。`assemble_rows`／`assemble_scalars` 把稀疏結果組成 detached 顯示陣列，保留缺值與 filled mask，不插值。

獨立 GUI app 由 `zcu_tools.gui.app.workflows` 提供。`zcu_tools.gui.app.workflows` 擁有泛型 Window、App 組裝、forms 協調、source reload 與 input 保存。Window 使用同一份 SessionServices 的 runner／progress，組合根注入註冊清單、source loader 與 alias preset。核心與宿主不依賴具體 workflow 或 `Calibrated`，不反向 import `zcu_lab`。各 workflow 保留自己的 forms 與科學流程。這些組裝不改變核心的 device、run 或保存契約。

`saved_run` 提供 typed manifest／iteration reader，由 caller 傳入 workflow 名稱與 point TypeAdapter。它只消費 structured committed summaries，指名缺失與 malformed source，不讀 journal 或從 leftover native 檔創造提交點。`export` 以 caller 的 native declaration 讀具名 trace，並對已準備好的 grouped payload 分別寫 merged native 與 Labber。Writer failure 保留各自 path／error，不操作 ledger 或修復 run identity。

各 workflow 的獨立工具留在 zcu_lab，擁有科學 payload、channel、單位與樣本政策。框架不提供通用 dispatcher，不解讀具體實驗或 device snapshot。GUI/Controller 不保留 export binding。缺 record 不代表無量測。

## 驗證 owner

`tests/experiment/workflows/` 從公開契約驗證核心結果。既有 `tests/experiment/v2/runtime/` 仍擁有 Schedule、ProgramBuilder 與 buffer 行為，不在此複製底層 runtime 測試。
