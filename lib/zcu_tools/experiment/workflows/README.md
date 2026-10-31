# `experiment.workflows` 具名 async workflow

**Last updated:** 2026-11-01（operation 能力與 run_core）

**讀者：** framework-dev、lab-author。查閱 workflow 宣告、Scope、控制、operation 事實與 record 保存。

此 package 擁有 workflow 宣告、控制與 framework JSON。Lab 擁有量測順序、分析、檔名與 record 欄位。
框架不匯入 `zcu_lab`，也不解讀原始量測檔案。跨模組分工見 [ADR-0062](../../../../docs/adr/0062-experiment-workflow.md)。

## 宣告與 live tunables

`@workflow(name, description)` 宣告一個 `async def (env, plan, tun)`。
簽名以 `WorkflowEnv[C]`、Plan model、`Live[Tunables]` 與 `BaseModel | None` return annotation 配對。
Plan 必須 frozen、extra-forbid。Tunables 必須 extra-forbid，且符合 JSON schema 限制。
宿主顯式建立 WorkflowRegistry，每次 catalog 重載使用新 instance。

作者 return 才結束 workflow。非 None 的 BaseModel return 只在 done 時寫 metadata root 的 `summary.json`，不算 commit。
`RunEnded.summary` 記是否寫了 summary。`tun.current` 每次回最新 revision 的完整 detached model。
作者可以固定一份 capture，也可以在 operation 之間重讀。
框架依讀取順序保存去重 revisions，不取最高值。每次 commit 保存自上次提交後的讀取紀錄。

`Engine.update_tunables` 接完整 model、expected_revision 與 actor。
它在 journal append 成功後發布新值。相同值仍增加 revision。
驗證或 revision 衝突不改 run。Journal I/O failure 停止 run，保留原始 cause，等待作者與 executor 清理。

## 顯式 Scope 與生命週期

`WorkflowEnv` 是同一個 experiment Scope 的 workflow 根，`env.host.root is env`。
Host 不預建 iteration，也不在 commit 時切換 token。
`with env.scope("calibration") as scope` 建作者 child。
`for cfg, iteration in zip(points, env.iterations())` 每點取得一個 iteration。
Generator 在 next 時釋放前一點，在 close 時釋放最後一點。保留 generator 引用的 caller 必須自行 close。
Scope 的 with 即使遇到例外也會 release，release 後不能再建立 child。

每個 child 都有 `call`、`setup_device`、`wait_until`、`dir`、同步 `commit`、`identity` 與 `get_device_cfg`。
Child 繼承根的 host capabilities，缺失能力在使用時 raise MissingCapability。
`scope.identity` 回 detached startup RunIdentity，不新增 provenance。
需要 workflow 圖的 helper 顯式接 `plots: Plots`，不靠另一個 env 查目前 scope。

路徑使用 `workflow:<catalog名>/iteration#1/operation:T1#1`。
`call(fn, name="fit")` 與 `run_core(scope, fn, cfg, name="T1")` 的 name 由 caller 指定，不從 callable 反射。
`active_path` 在第一個 fork 停下，`live_paths: list[str]` 列出所有存活葉的完整路徑。
選擇器搜整棵存活子樹，ambiguous 時回 candidates，不任選最新 child。

## Operation 與控制

EnginePorts 注入 clock、devices、WorkflowPlots、progress factory 與 context。
宿主持有連線、worker 與 RUN lease。核心不 connect、disconnect，也不依賴 Qt。
`Engine.start` 驗證並保存啟動資料，不執行作者。宿主在一個 worker 呼叫一次 `Engine.execute`。
共用 ScriptHost 執行作者 coroutine、finally 與 executor drain，全部結束後才允許 replacement Start。

`await scope.call(fn)` 在 executor 執行 `fn(CallContext)`，回 Completed[T]。
CallContext 提供本次 operation 的 scope、soc、soccfg、connected devices 與獨立 plots。
未供應的資源為 None，devices={} 代表已供應但沒有連線裝置。Raw ScriptHost 只供應 scope。
框架每次建立 operation plots，callback 結束後 finish，下一次 operation release 舊呈現。
`Completed.value` 是原始值。`Completed.facts` 是同時寫入 journal 與 commit 的 immutable OperationFacts。
facts.stopped 只記 operation 自己的 stop bit，不含只取消子孫的情況。
任意程式例外直接傳回作者，不降級成 Failed。

`from zcu_tools.experiment.run_core import run_core` 引入 core 轉接 helper。
`await run_core(scope, fn, cfg, name="T1")` 接 core 的 `(cfg, *, context: RunContext) -> Result`，包括 bound Experiment.run。
Helper 驗證 cfg 是 ConfigBase，深複製後以 CallContext 組 RunContext。
必要資源為 None 時 raise MissingCapability，不把缺 devices 當成空 mapping。
呼叫 core 前，helper 經 capture_device_snapshot 讀全部 connected drivers。
任一讀取失敗就不執行 core。回傳 `Completed[RunRecord[Cfg, Result]]` 的 record.metadata["device_snapshot"] 保留呼叫前讀值。
Helper 不保存 raw 或圖。一般 blocking 工作直接用 call。

`scope.get_device_cfg(name)` live-read detached 完整設定。
`await scope.setup_device(name, cfg)` 把完整 capture 與 operation stop bit 交給 device owner。
`await scope.wait_until(target)` 接 aware 絕對 deadline，呼叫 clock 一次。
這兩種 operation 不要求 device snapshot。Driver 的驗證、限制與 I/O 錯誤照原 cause 傳播。

每個 operation 只呼叫一次。框架不丟棄合作停止的回傳值，也不自動重跑 callable。

`Engine.cancel(run_id, selector, expected=None)` 回 CancelResult，不等待 cleanup。
Root cancel 在 Host lock 內設 bit 並記第一次 typed ack，loop wake 只排程，不等待。
重複 root cancel 回同一 ack，active_path 固定為根路徑。Engine 不另加一輪 loop acknowledgement。
這個 ack 只代表接受與 token 已 set，終態 future 才代表作者 finally 與 executor 已 drain。
Acquire 只停一次取得，schedule 停該段，iteration 跳過目前點，run 是 operation 的別名。
這些子層取消後，core 回傳就交付 Completed，core 拋錯就原樣傳給作者。作者在量測間檢查 `iteration.cancelled`。
根 Stop 或 workflow cancel 則在下一個 async operation 或 Ask raise 普通 ScriptCancelled。
進行中的 call 先 drain 並保存 completion facts，再 raise。真實錯誤優先，不被取消覆蓋。
終態為 cancelled，ScriptCancelled 不記 error。作者 finally 可以同步 commit 已有成果。

未帶序號或 expected 的 workflow 根取消先由 Stop owner lock 記錄 StopRequested／manifest 並 latch，
再呼叫 Host 設根 token。持久化不進 loop control callback。
Host 已 drain、owner 尚未 finalization 時，Stop 仍可被 owner 接受並使終態 cancelled。
帶序號或 expected 的根及其他 scope 只由 Host 核對，不寫 Stop 意圖。
Terminal owner 對合法請求回 scope-not-active refusal，錯誤語法仍 raise ValueError。
GUI Stop 與 Engine.stop 共用此 owner。清理例外或 control I/O failure 保留 failed 與原始 cause。

## Record 與目錄

`scope.dir` 第一次使用才建立目錄。根是 data_root，iteration 是 `iteration-000001/`。
作者 child 用 `<label>-<seq:06d>`，內建 child 用 `<kind>-<seq:06d>`。
例如 operation:T1#1 的目錄是 `operation-000001`。Name 留在 path 與信封，不進目錄名。

`scope.commit(record)` 同步提交一個 BaseModel 快照。
每個 scope 只能提交一次，第二次 raise ValueError。Root commit 寫 data_root/record.json。
作者一開始建可變 record，未取得欄位為 None，再逐步填結果。
每點用 try/finally，只在 finally 提交一次。`continue`、根取消與真實錯誤都經過這個位置。
Commit 後改 record 不會改已提交資料。完整範例見 [workflow 教學](../../../../zcu_lab/tutorial/05-workflow.md)。

成功 commit 先追加並 flush record_committed journal，再發布 in-memory records。
Record.json 以同目錄 temporary 加 replace 寫入。此副本寫入失敗只記 log，已 flush 的 journal 仍是權威。
信封包含 scope_path、name、commit_seq、目錄、UTC 時間、型別、encoded record、revisions 與 operations。
Manifest 為每種 model 保存一次資訊用 JSON schema，不作 runtime validator。
Encoder 將非有限值轉 null。只有作者 record/summary 序列化可降級 repr，必要 metadata 與 I/O failure 不降級。

作者自行保存 RunRecord、圖與其他原始檔，並在 record 記相對檔名。
Saver 錯誤沿作者例外路徑傳播。Stop 不回滾檔案，框架不追蹤或刪除作者檔案。
Lab saver 將 Completed、歷史 snapshot、identity 與 spec.tag 交給 record_with_metadata。
Caller 必須明填 completion。此函式以 RunMetadata 組內容，合併進 RunRecord.metadata，保留 helper 的 device_snapshot 與作者欄位。保存入口只接 record，native 輸出通用 Container。
Native run_id 是 `<identity.run_id>-<facts.iteration:06d>-<facts.attempt>`。
Lab 以 facts.stopped 選 stopped 或 complete，並將 stopped 資料分類為 partial。只取消 operation 的子孫時，facts.stopped 仍為 false，存檔標 complete。

獨立轉換工具共用 `export.load_trace` 讀具名 native reference，`export.write_outputs` 只寫 Labber。
`write_outputs(..., save_stem=...)` 把 workflow name、完整 run_id、point、roles、cfg values 與 saved_at 放入共用 SaveContext。
預設 callback 產生 dotted workflow name、hyphen-joined roles 與完整 run_id 的 stem。Caller 可完全替換 callback，框架不補名稱。
Labber 寫日期樹內的 `<stem>.hdf5`，只有初始目的已存在時才沿用 suffix helper。Comment 在 cfg 旁帶歷史 snapshot.point 與 snapshot.roles。
Workflow 自己的雙側 root 與 raw operation 檔名不變。
Coherence 的 samples 由 sample table owner 追加。Lab 工具依呼叫者提供的目的地決定輸出，未提供目的地時在寫入前拒絕。
Merged 輸出依使用者 2026-10-25 裁定移除，見 [ADR-0062](../../../../docs/adr/0062-experiment-workflow.md#獨立-workflow-轉換工具)。

## Facts 與讀取

每次 executor attempt 追加 operation_finished event。
OperationFacts 保存 kind、iteration、attempt、started_at、finished_at、outcome、reason、stopped 與 name。
Name 是作者傳給 call 的 operation label，未命名或舊記錄為 None。
舊 journal 與 record 的 `acquisitions` key 在載入時忽略，不轉換或寫回。
ProgramBuilder 的 acquire retry 與 data-driven early stop 由日誌記錄，不由 operation 推斷。
Kind 是 call、setup_device 或 wait_until，run 僅保留供歷史讀取。Iteration 取自最近 iteration 物件的 birth sequence，沒有 iteration 時為零。
Attempt 在該 iteration 中從一開始，每個 operation 增加一次，不隨 commit 重置。
Stopped 是完成時 operation 自己的 bit，不推斷子孫的控制狀態或資料完整度。
目前 Outcome 是 completed 或 raised。discarded 僅保留供讀取歷史 Pause 記錄，不是現行停止行為。
歷史記錄的 Attempt 也計入當時的 Pause 重試。Device snapshot 只在 run_core 的 RunRecord metadata，不進 operation journal。

Commit 的 operations 保存該 scope 與其 children 的 attempts。未 commit 的 attempt 仍留 journal。
`saved_run.read_records(run, Model | TypeAdapter(Model))` 走 nested 目錄樹，按 commit_seq 回 SavedRecord tuple。
異質資料用 read_records_raw。型別不符、壞欄位、重複 commit_seq 或信封位置錯誤會指名 source raise。
Reader 不讀 journal，不從 leftover native files 補造遺失 record。
`read_summary(run)` 讀 metadata root 的 summary.json，缺檔回 None。

## Ask 與顯示

`await env.ask(question)` 等一題一個 request_id 的 Answer，不回 Optional。
等答覆時 lifecycle 是 waiting；Answer 讓同一作者 coroutine 繼續一次。
根 Stop 可中斷等待。Journal 保留成功接受的 QuestionAnswered 事實。
Engine.parse_answer 先比對 ID 再驗 schema。Engine.answer 通知同一 worker，不另起 execution segment。
Wrong model、錯誤欄位、stale 或重複 ID 不消耗新問題。QuestionMismatch 帶目前 pending_request_id。
Accepted answer 的記憶體交接留在 control callback，journal I/O 留在 callback 之外。

`env.plots` 保留 workflow 趨勢圖。每次 scope.call 使用相同 presentation host 的獨立 operation Plots。
返回或失敗都先 finish，下一次 operation release 舊 operation。Core 與 finish 同時失敗時保留 core cause 並附 cleanup note。
WorkflowPlots.figures 將 workflow 圖與目前 operation/<name> 圖投影給 pane/MCP。
Workflow 圖不得使用 operation/ 前綴。保存圖時從 Completed.value.result 重畫，不依賴 live 圖位置。
Progress bars 由 env 保留，execution owner 在 producer drain 後逐一嘗試 close，再發布終態。

測試 owner 是 `tests/experiment/workflows/`，只用合成 workflow。
Schedule 與 ProgramBuilder 見 [runtime](../v2/runtime/README.md)，GUI 組裝見 [measure README](../../gui/app/measure/README.md#workflow-hosting)。
