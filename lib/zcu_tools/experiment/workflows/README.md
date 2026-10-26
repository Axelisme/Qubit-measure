# `experiment.workflows` 具名 async workflow

**Last updated:** 2026-10-27

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

每個 child 都有 `run`、`call`、`setup_device`、`wait_until`、`dir`、同步 `commit`、`identity` 與 `get_device_cfg`。
Child 繼承根的 host capabilities，缺失能力在使用時 raise MissingCapability。
`scope.identity` 回 detached startup RunIdentity，不新增 provenance。
需要 workflow 圖的 helper 顯式接 `plots: Plots`，不靠另一個 env 查目前 scope。

路徑使用 `workflow:<catalog名>/iteration#1/operation:T1#1`。
`run(fn, cfg, name="T1")` 與 `call(fn, name="fit")` 的 name 由 caller 指定，不從 callable 反射。
`active_path` 在第一個 fork 停下，`live_paths: list[str]` 列出所有存活葉的完整路徑。
選擇器搜整棵存活子樹，ambiguous 時回 candidates，不任選最新 child。

## Operation 與控制

EnginePorts 注入 clock、devices、WorkflowPlots、progress factory 與 context。
宿主持有連線、worker 與 RUN lease。核心不 connect、disconnect，也不依賴 Qt。
`Engine.start` 驗證並保存啟動資料，不執行作者。宿主在一個 worker 呼叫一次 `Engine.execute`。
共用 ScriptHost 執行作者 coroutine、finally 與 executor drain，全部結束後才允許 replacement Start。

`await scope.call(fn)` 在 executor 執行 `fn(CallContext)`，回 Completed[T] | Failed。
CallContext 提供本次 operation 的 Scope 與起跑 epoch。其他能力由 closure 顯式傳入。
`Completed.value` 是原始值，`partial_result` 是 operation 或已完成 descendant 的取消事實。
`Completed.facts` 是同時寫入 journal 與 commit 的 immutable OperationFacts。
任意程式例外直接傳回作者，不降級成 Failed。

`await scope.run(fn, cfg)` 接 core 的 `(cfg, *, context: RunContext) -> Result`，包括 bound Experiment.run。
Cfg 必須是 `zcu_tools.cfg_model.ConfigBase`。每次 attempt 深複製原 capture，建 RunContext 與獨立 operation Plots，借用 soc、soccfg 與 connected drivers。
呼叫前經 capture_device_snapshot 讀全部 connected drivers。任一讀取失敗就不執行 core。
回傳是 `Completed[RunRecord[Cfg, Result]] | Failed`，不自動保存 raw 或圖。
非 ConfigBase 的工作用 call。

`scope.get_device_cfg(name)` live-read detached 完整設定。
`await scope.setup_device(name, cfg)` 把完整 capture 與 operation stop bit 交給 device owner。
`await scope.wait_until(target)` 接 aware 絕對 deadline，Pause 重試同一 deadline。
這兩種 operation 不要求 device snapshot。Driver 的驗證、限制與 I/O 錯誤照原 cause 傳播。

Pause 停止目前 operation、增加 epoch 並關 gate，不停止 iteration。
被打斷且 stale 的 attempt 留 discarded 事實。Resume 重跑同 callable 與原 capture。
自然完成的結果只等待 gate，不重跑。框架不重新執行整個 workflow。

`Engine.cancel(run_id, selector, expected=None)` 回 CancelResult，不等待 cleanup。
Root cancel 在 Host lock 內設 bit 並記第一次 typed ack，loop wake 只排程，不等待。
重複 root cancel 回同一 ack，active_path 固定為根路徑。Engine 不另加一輪 loop acknowledgement。
這個 ack 只代表接受與 token 已 set，終態 future 才代表作者 finally 與 executor 已 drain。
Acquire 只停一次取得，schedule 停該段，iteration 跳過目前點，run 是 operation 的別名。
這些子層取消回 partial 或 Failed 值。作者在量測間檢查 `iteration.cancelled`。
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

`scope.commit(record)` 同步提交一個 BaseModel 快照，不經 pause gate。
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
Saver 錯誤沿作者例外路徑傳播。Pause 或 Stop 不回滾檔案，框架不追蹤或刪除作者檔案。
Lab saver 將 Completed、歷史 snapshot、identity 與 spec.tag 交給 record_with_metadata。
此函式將 RunMetadata 裝入 RunRecord.metadata，保存入口只接 record，native 輸出通用 Container。
Native run_id 是 `<identity.run_id>-<facts.iteration:06d>-<facts.attempt>`。
Completed.partial_result 決定 complete 或 stopped，不用 facts.stopped 代替。

獨立轉換工具共用 `export.load_trace` 讀具名 native reference，`export.write_outputs` 只寫 Labber。
Coherence 的 samples 由 sample table owner 追加。Lab 工具依呼叫者提供的目的地決定輸出，未提供目的地時在寫入前拒絕。
Merged 輸出依使用者 2026-10-25 裁定移除，見 [ADR-0062](../../../../docs/adr/0062-experiment-workflow.md#獨立-workflow-轉換工具)。

## Facts 與讀取

每次 executor attempt 追加 operation_finished event。
OperationFacts 保存 kind、iteration、attempt、started_at、finished_at、device_snapshot、outcome、reason、stopped 與 name。
Name 是作者傳給 run／call 的 operation label，未命名或舊記錄為 None。
舊 journal 與 record 的 `acquisitions` key 在載入時忽略，不轉換或寫回。
ProgramBuilder 的 acquire retry 與 data-driven early stop 由日誌記錄，不由 operation 推斷。
Kind 是 run、call、setup_device 或 wait_until。Iteration 取自最近 iteration 物件的 birth sequence，沒有 iteration 時為零。
Attempt 在該 iteration 中從一開始，包含 Pause retries，不隨 commit 重置。
Stopped 是完成時 operation 自己的 bit，partial_result 另含 descendant 取消。
Outcome 是 completed、partial、discarded、failed 或 raised。只有 run 有 pre-call device snapshot。

Commit 的 operations 保存該 scope 與其 children 的 attempts。未 commit 的 attempt 仍留 journal。
`saved_run.read_records(run, Model | TypeAdapter(Model))` 走 nested 目錄樹，按 commit_seq 回 SavedRecord tuple。
異質資料用 read_records_raw。型別不符、壞欄位、重複 commit_seq 或信封位置錯誤會指名 source raise。
Reader 不讀 journal，不從 leftover native files 補造遺失 record。
`read_summary(run)` 讀 metadata root 的 summary.json，缺檔回 None。

## Ask 與顯示

`await env.ask(question)` 等一題一個 request_id 的 Answer，不回 Optional。
Engine.parse_answer 先比對 ID 再驗 schema。Engine.answer 通知同一 worker，不另起 execution segment。
Wrong model、錯誤欄位、stale 或重複 ID 不消耗新問題。QuestionMismatch 帶目前 pending_request_id。
Accepted answer 的記憶體交接留在 control callback，journal I/O 留在 callback 之外。

`env.plots` 保留 workflow 趨勢圖。每次 scope.run 使用相同 presentation host 的獨立 operation Plots。
返回、丟棄或失敗都先 finish，下一 attempt release 舊 operation。Core 與 finish 同時失敗時保留 core cause 並附 cleanup note。
WorkflowPlots.figures 將 workflow 圖與目前 operation/<name> 圖投影給 pane/MCP。
Workflow 圖不得使用 operation/ 前綴。保存圖時從 Completed.value.result 重畫，不依賴 live 圖位置。
Progress bars 由 env 保留，execution owner 在 producer drain 後逐一嘗試 close，再發布終態。

測試 owner 是 `tests/experiment/workflows/`，只用合成 workflow。
Schedule 與 ProgramBuilder 見 [runtime](../v2/runtime/README.md)，GUI 組裝見 [measure README](../../gui/app/measure/README.md#workflow-hosting)。
