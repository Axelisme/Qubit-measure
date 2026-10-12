# `experiment.workflows` — 具名 workflow 核心

**Last updated:** 2026-10-12，iteration 保存與獨立 workflow conversion 分工

此 package 擁有具名 workflow 的通用執行契約。具體 flux 清單、校準、分析與 record 語意屬於使用者 workflow，不屬於核心。架構分工見 [workflows ADR 草稿](../../../../docs/adr/draft/workflows-engine.md)。

## 能力與資料

宿主以 `EnginePorts` 注入 clock、device、`Plots`、progress 與 detached context。繪圖沿用既有 `PlotHost`，artists 留在 host owner thread；核心不持 raw Axes，也不捕捉 Figure snapshot。宿主持有硬體連線與 worker，核心不建立執行緒、不 connect／disconnect，不依賴 Qt 或 GUI。Progress factory 沿用現有 runtime bar 契約，不能把會再次查 ContextVar 的 make_pbar 當成 concrete factory。

`Next` 配對步驟摘要與下一步 state。`Done` 不新增點 record，`Aborted` 表達明確的業務終止。`Completed` 配對 experiment 返回時的 Run.cfg 副本與 Result，workflow 自行決定保存與單次檔案格式。`Failed` 表達可由 workflow 處理的 Schedule 失敗，不包含部分 Result。

`workflow(name, *, plan, tunables, init)` 將具名宣告附在原函式，不執行 init，也不建立全域 catalog。State／record 型別由 init 與 step 簽名推導，state 只需支援 deepcopy，不限定 dataclass。`WorkflowRegistry` 由宿主顯式填入，每次重載建立新 instance。Tunables 宣告檢查 JSON schema，dot path 的 leaf／model 判定也由宣告 owner 提供。Package 內的 TunableValues 集中完整 model 驗證、deepcopy capture 與 revision，整批 journal append 成功後才發布新值。Engine 擁有 run_id 與 lifecycle 政策。當前 step 的 capture 不受新 revision 影響。

`InitEnv` 只提供原始 UTC start 與 detached context。`WorkflowEnv` 的 effect helpers 使用 `yield from`，把 request 與回應的型別交接留在核心。同步 `get_device_cfg` 是例外，它在 worker 直接讀取 live cfg。環境共用 injected `Plots` 與 engine-owned progress bars。WorkflowEnv 的 soc／devices 與 context 存取在缺 port 時立即 raise，供定義者在步驟開頭自檢。Engine 不做 capability admission，也不記錄 requires。硬體操作應經 effect 保留 journal 與取消；框架信任定義者，不強制阻擋直接操作。

`Run` 為 experiment 複製 dataclass 或 Pydantic model instance cfg，將同一個 StopSignal 與明確 env 接入既有 Schedule。型別本身與無結構輸入在入口失敗。cfg.dev 只放角色到 device 名稱，不觸發 automatic setup。同步 `env.get_device_cfg(name)` 回傳 detached 的實際 `BaseDeviceInfo` 子類。`yield from env.setup_device(name, cfg)` 傳遞完整 model 與同一個 StopSignal，journal 記錄 name 與全部 cfg 欄位。Host 以 selected driver.info_model 重新驗證 dump，拒絕未驗證的 model_copy 更新，再把 signal.event 交給 native setup。框架不認識 flux、unit 或 device 種類。Unknown／disconnected names 在使用時指名報錯，driver limits 與 I/O cause 原樣傳播，不 retry 或 clamp。使用者套件的 set_flux 工具重用已交付的 check_flux_unit，單位規則不在核心重做。Start／Resume 不讀裝置，也不記錄 device provenance；RunIdentity 只保存 run／entry identity 與 software provenance。Schedule 的局部 cfg 不倒灌回 Run，後一個 completed outcome 不覆蓋前一個非 completed。Run 保留 live callback 的原始 failure cause，Engine 依來源區分可恢復的 Schedule Failed、宿主 Stop 與核心失敗。即使 experiment 重拋本次 signal.error，來源也不改變。

每次 `env.run` 在宿主保留的 RUN lease 下、callback 前，經 DevicePort 捕捉全部 connected drivers。讀取失敗使 run failed，不執行 callback。缺少 devices 能力不是零 device；真正零 device 仍注入明確回傳 `{}` 的 port。`Run` 要求 caller 提供固定、detached 的 device JSON，直接 caller 先捕捉，確知無 device 時明填 `{}`。Native callback 與 saver 只傳遞這份資料，後續 acquisition／Save 不重新讀取。同步 getter 只讀選定 device，與每次 env.run 的全 device capture 用途不同。SessionDevices 在同一 RUN lease 下重用公開 connected mapping 與 native acquisition helper，不更新 GUI State。

Record encoder 將摘要投影成 detached JSON。非有限數字變成 null，曲線陣列與過大 record 會警告。只有摘要序列化可降級為 repr；deepcopy、必要 journal 欄位與檔案 I/O 不能借用此降級規則。

Engine 在每次 step 呼叫前建立 `data_root/iter/<至少六位 call_seq>/`。唯讀 `env.iteration_dir` 是已存在的 absolute Path，指向該 iteration 的根目錄。屬性不可重綁，workflow 仍可自行操作檔案系統。Engine 不預建量測或圖形子目錄，也不掃描、驗證、追蹤或清理 workflow 檔案。

`env.run(experiment, cfg, live=...)` 只執行量測。`Completed` 不代表已保存；workflow 自行呼叫 native saver，決定檔名及是否保存。需要讓工具找到的檔名與用途寫進自己的 point record。Saver 例外沿 workflow 的非正常例外路徑停止 run，不變成量測 `Failed`。Pause／Stop 可能留下未提交或不完整的檔案，框架不承諾 rollback。`ExperimentFailed.run_no` 只識別 invocation 內從 1 計數的 effect，與保存或檔名無關。

Package 內的保存 seam 只負責 typed journal、manifest、iteration 路徑與摘要。Journal 使用 Engine 的同一把 control lock，逐行追加並 flush，不宣稱 fsync 或崩潰接續。Manifest 只摘要生命週期，不保存每點 record 或 state，沿同目錄 temporary 加 replace。

每個已結束 invocation 在 `iter/<至少六位 call_seq>/record.json` 保存輸入與結果摘要。格式為 `zcu-workflow-iteration`，版本為 1，包含 identity、workflow、完整 plan、本次 capture 的完整普通 tunables、started 與 result。Started 保存 call_seq、commit_seq、revision、kind。Result 沿用 journal 的 StepCommitted／Finished／Discarded／Failed 欄位。具體檔名屬於 encoded point record，框架不解讀。摘要不保存 state，也不複製 native 檔案的 cfg 或量測結果；EncodedRecord 沿用 json／repr／null 規則。

Engine 在結果 journal append 與 flush 成功後，以同目錄 temporary 加 replace 原子保存摘要。執行中的 invocation 沒有 record.json，不先建 pending 檔。摘要序列化或 I/O 失敗只記 log，不改變提交、state publication、後續 step 或 run 終態。Journal 仍是 commit 唯一權威；record.json 不是 checkpoint，不支援 crash resume。核心不提供 committed journal reader。Overnight 的 live 趨勢只取 Engine.records，不保存或重開趨勢圖。Workflow 實作者自行處理 iter 內的圖形保存，不新增框架存圖 interface。責任分工見 ADR-0062。

RunIdentity 持有宿主傳入的 native SoftwareProvenance。Manifest 與 run_started journal 在 provenance 底下保存同一份資訊，unknown 保留 None，不以預設 False 或字串代替。

Engine.start 只驗證並保存啟動資料。第一次 execute 才執行 init；每次 execute 在呼叫執行緒同步跑到 paused 或 terminal。宿主安排 worker 與 lease。核心以同一把 control lock 序列化控制、revision 與不可中斷的提交，不在 effect 執行期間持有這把鎖。Pause／Stop 不等待 effect，取消時關閉並丟棄 step generator。Next 已返回時仍先提交；Resume 使用已提交 state 與最新 tunables 重做未提交步驟。內部 typed run 保留宣告的泛型配對，Engine 只持有 detached control view。Terminal run 只有在 execute 真正返回後才能被新 run 取代。

`WorkflowEnv.plots` 直接提供注入的 `Plots`，跨步驟與 retry 保留同一個繪圖 lifetime。`live` 接 `LiveUpdate` 回呼，收到 buffer 維度順序的 float64 axes tuple 與 complex128 samples，包含 partial NaN。回呼同步執行，不修改或保留借用的資料；workflow 用 typed handles 投影實數，或以 draw 任意繪圖、save 保存。Run 只綁定一個 buffer，保留回呼錯誤來源，不驗證繪圖策略。宿主在 producer 停止後 finish／release，Pause 不結束 Plots。回呼簽名目前是顧問代裁，待使用者審查。`assemble_rows`／`assemble_scalars` 把稀疏結果組成 detached 顯示陣列，保留缺值與 filled mask，不插值。

獨立 GUI app 已接線。Window 使用同一份 SessionServices 的 runner／progress，app 重用 zcu_lab 的單一 ALIAS_PRESET。這些組裝屬於宿主，不改變核心的 device、run 或保存契約。具體 forms、source reload 與 input 保存由 zcu_lab.workflows.gui 擁有。

各 workflow 的獨立工具在 zcu_lab 讀 manifest、iteration record.json 與 point 具名 native reference。框架不提供 reader、export-only carrier 或通用 dispatcher，不解讀具體實驗、device snapshot 或單位。GUI/Controller 不保留 export binding。工具的輸入驗證、科學 payload 與 writers 分工見 zcu_lab.workflows。缺 record 不代表無量測，不能從 leftover native 檔創造提交點。

## 驗證 owner

`tests/experiment/workflows/` 從公開契約驗證核心結果。既有 `tests/experiment/v2/runtime/` 仍擁有 Schedule、ProgramBuilder 與 buffer 行為，不在此複製底層 runtime 測試。
