# `experiment.workflows` — 具名 async workflow

**Last updated:** 2026-10-22

**讀者：** framework-dev、lab-author。查閱 workflow 宣告、執行、控制與 record 保存的分工。

此 package 擁有 workflow 的宣告、控制與 framework JSON。使用者 workflow 擁有量測順序、分析、檔名與 record 的領域欄位。框架不匯入 `zcu_lab`，也不解讀原始量測檔案。既有架構分工見 [ADR-0062](../../../../docs/adr/0062-experiment-workflow.md)。

## 宣告與 live tunables

`@workflow(name, description)` 附在一個 `async def (env, plan, tun)` 上，不包裝或執行它。簽名以 `WorkflowEnv[C]`、Plan model、`Live[Tunables]` 與 `BaseModel | None` 回傳註記配對型別。Plan 必須 frozen、extra-forbid，Tunables 必須 extra-forbid 且符合 JSON schema 限制。宿主顯式建立 `WorkflowRegistry`，每次 catalog 重載使用新 instance。

作者在同一個 coroutine 保留區域變數。`await env.commit(record)` 提交 BaseModel record；同一 run 可以提交不同的 model。作者 return 才結束 workflow。非 None 的 BaseModel return 只在 done 時寫 metadata run root 的 `summary.json`，不算一次 commit。`RunEnded.summary` 記是否寫了 summary。

`tun.current` 每次回最新 revision 的完整 detached model。作者可以保留其中一份作固定 capture，也可以在 iteration 中重新讀取。框架把本輪實際讀過的 revision 去重，依讀取順序保存為清單，不取最高值。

`Engine.update_tunables` 接完整 model、expected_revision 與 actor。它重新驗證全部欄位，在 journal append 成功後整體發布。相同值仍增加 revision。`TunablesChanged` 保存完整新值與前後 revision。驗證或 revision 衝突不改 run；journal I/O failure 則停止 run，保留原始 cause，等待作者與 executor 清理。

## Host 與操作

宿主以 `EnginePorts` 注入 clock、devices、`WorkflowPlots`、concrete progress factory 與 context。宿主持有連線、worker 與 RUN lease。核心不 connect、disconnect，也不依賴 Qt。

`Engine.start` 驗證與保存啟動資料，不執行作者。宿主在一個 worker 呼叫一次 `Engine.execute`。這個呼叫透過共用 `experiment.script_host.ScriptHost` 推進整個 coroutine，包含 Pause 與 Ask 的等待。控制命令只通知同一個 worker，不另排 execute、answer 或 drain segment。新 run 必須等原 worker 的 finally 與 executor 都結束。

`await env.call(fn)` 在 executor 執行 `fn(CallContext)`，回 `Completed[T] | Failed`。CallContext 只有 operation stop_signal 與 epoch。其他能力由 closure 使用 env。`Completed.value` 是 fn 的原始值，`partial_result` 表示 operation 或已完成 descendant 在完成時曾被取消。必填的 `Completed.facts: OperationFacts` 是同時寫入 journal 與 commit 信封的當次事實。任意程式例外直接傳回作者，不變成可恢復的 Failed。

`await env.run(fn, cfg)` 接 core 的 `(cfg, *, context: RunContext) -> Result` callable，可以直接傳 bound Experiment.run。cfg 是位置參數，名稱不限定；context 必須是具名 keyword-only 參數。cfg 必須是 ExpCfgModel。每個 attempt 建立新的 RunContext，借用 soc、soccfg 與 connected drivers，使用當次 operation token 及獨立的 operation Plots。框架在呼叫前經 `capture_device_snapshot` 讀全部 connected drivers，再以新的 cfg deepcopy 執行 fn。回傳為 `Completed[RunRecord[Cfg, Result]] | Failed`，saver 接收 Completed，以 value 取原始 record，以 facts 取框架捕捉的事實。非 ExpCfgModel scan 使用 env.call。

`DevicePort.get_connected_devices` 回 detached mapping，driver 生命週期仍屬宿主。DeviceMapping 集中異質 driver 的型別邊界。缺 devices capability 會 raise MissingCapability；確定沒有 device 時，宿主提供回空 mapping 的 port。Snapshot 讀取失敗阻止 core fn 執行。框架不依 cfg.dev 自動 setup 或預先解析角色。

`env.get_device_cfg(name)` 同步讀取一個 detached 實際 Info model。`await env.setup_device(name, cfg)` 把完整設定與 operation token 交給 driver owner 重新驗證及操作。`await env.wait_until(target)` 接 aware deadline。這兩種操作也經 env.call，不因記錄 attempt 而要求 devices snapshot。

Schedule acquire failure 回 Failed，作者決定重試、繼續或 return。新 operation 不繼承前次已交付的 acquire error。Build、conversion 與 callback 錯誤照原 cause raise；runtime retry 只重試同一個已建立 program 的 acquire。

Pause set 當前 operation token、推進 epoch 並關閉 gate，不 set iteration token。被打斷且 stale 的 partial attempt 記為 discarded。Resume 後重跑同一個 callable 與原 capture，作者不會提前收到該 partial。已自然完成的結果等 Resume 後交付，不重跑。

`Engine.cancel(run_id, scope, expected=None)` 沿共用 host 核對 label 或 label#N，回 CancelResult，不等待 cleanup。Acquire 只停一次取得，Schedule 停該段後續取得，iteration 停本輪且 commit 後下一輪正常，run 是當前 operation 的別名。Status 回 active_path。例如 `engine.cancel("run-id", "acquire", expected=3)` 拒絕不是第 3 次的 acquire，不停止父 scope。
GUI Stop 與 `Engine.stop` set script 根，往下傳到 iteration 與 operation。作者收到 `Completed(partial_result=True)`，後續 operation 仍啟動並看到已 set 的 token，後續 Ask 回 None。作者必須處理空值並完成清理。框架不向作者 Task 注入 CancelledError。明確 Stop 的終態為 cancelled；清理例外或 control I/O failure 為 failed。

## 作者範例

序列相同時直接重用 core，或用 typed async helper 包裝它：

```python
async def run_t1(
    env: WorkflowEnv[Calibrated], cfg: T1Cfg
) -> Completed[RunRecord[T1Cfg, T1Result]] | Failed:
    return await env.run(T1Exp().run, cfg)

# Inside the authored workflow coroutine:
for flux, d in zip(plan.fluxes, env.commit_dirs()):
    outcome = await env.run(T1Exp().run, cfg)
    # Equivalent when a helper adds lab-specific policy:
    # outcome = await run_t1(env, cfg)
    if isinstance(outcome, Failed):
        continue
    save_t1(outcome, d / "t1.h5", snapshot=plan.snapshot, identity=env.identity)
    await env.commit(Point(flux=flux), d)
```

範例中的 cfg、Point 與 saver 由 lab 定義。不同序列可以寫同簽名的同步 helper，再傳給 env.run。作者 async helper 仍只 await env／tab／session 或可靜態解析的 lab helper，不自行建立 asyncio task。Guardrail 的範圍見 [lab 教學](../../../../zcu_lab/workflows/README.md#async-await-guardrail)。

要回到過去某點，開新 run、切 plan 範圍，用 saved_run reader 從上一個 run 初始化 history。Engine 不提供 cursor 回退或自動回灌。

## Commit 目錄與保存

`env.new_commit_dir()` 每次建立新的 absolute `data_root/commits/<六位配置序號>/`。配置順序與 commit 順序獨立。`env.commit_dirs()` 每次 next 才配置，使用 `zip(values, env.commit_dirs())`，把有限 values 放前面，避免多配置一次。

`await env.commit(record, d)` 只接受本 run 已配置且未提交的 d。省略 d 時自配新目錄。同目錄二次提交或 foreign path 會 raise，不發布 record。配置目錄不換 iteration token，成功 commit 才換新 iteration scope。未提交目錄可保留作者檔案，但沒有 framework record.json。

作者自行保存 RunRecord、圖形與其他原始檔，並在自己的 record 記檔名。Completed 不代表已保存。Saver 例外沿作者例外路徑傳播，不變成 acquire Failed。Pause 或 Stop 不回滾檔案，框架也不掃描、追蹤或刪除它們。

成功 commit 先追加並 flush `record_committed` journal，再發布 in-memory records。`record.json` 以同目錄 temporary 加 replace 寫入；這份副本寫入失敗只記 log，已 flush 的 journal 仍是 commit 權威。它不是 checkpoint，不支援 crash resume 或 fsync 承諾。

`env.identity` 每次回 detached 的既有 startup RunIdentity，作者修改回傳值不影響 run。Software provenance 在 workflow 啟動時保存，manifest 與 run_started.identity 各留既有副本。每次 operation 不另建 provenance 或 run_id。

Lab 的 `save_native(completed, destination, *, spec, snapshot, identity)` 顯式接收 env.identity。它從 Completed.value 取 RunRecord，以 facts、identity、spec.tag 與 snapshot 組裝既有 RunMetadata。H5 run_id 是 `<identity.run_id>-<facts.iteration:06d>-<facts.attempt>`。Completion 由 Completed.partial_result 推導，True 為 stopped，False 為 complete。它不使用 facts.stopped 判定 completion，避免漏掉 descendant 取消。labber_path 為 None。這是既有 writer 的接線，不表示 RunRecord.metadata 或 datafile 通用容器已完成。

Record 信封保存 commit_seq、配置目錄、UTC 時間、record 型別名、encoded record、本輪 revisions 與 operations。Manifest 為每種出現的 record model 保存一次 JSON schema。這些 schema 只提供資訊，不作 runtime validator。Record encoder 把非有限值改成 null，只有 authored record/summary 序列化可降級成 repr；必要 metadata 與 I/O failure 不使用這項降級。

## Operation 事實與讀取

每次實際 executor attempt 都追加 flattened `operation_finished` JSONL event。它包含 call_seq 與 operation；seq、time、run_id 是既有 journal 欄位。`OperationFacts` 欄位為 kind、iteration、attempt、started_at、finished_at、device_snapshot、outcome、reason 與 stopped。

Kind 是 run、call、setup_device 或 wait_until。Iteration 從 1 起，成功 commit 才推進，配置目錄不推進。Attempt 在 iteration 中從 1 起，成功 commit 後重置。Stopped 是完成時 operation 自己的 stop bit，不從 descendant 取消或後來的 Stop、Pause 狀態推導。Partial_result 獨立記錄 operation 或已完成 descendant 的取消，因此 partial_result=True 不代表 stopped=True。只有 run 有呼叫前的 device snapshot，其餘是 None。Outcome 是 completed、partial、discarded、failed 或 raised。Pause discard 與重試各留一筆，例外也留事實，不改原始例外分類。

Commit 的 operations 按 attempt 順序列出同一結構。未 commit 的操作只留 journal。Snapshot 固定在操作前，後續量測、driver mutation 或 Save 不重新讀取或覆寫它。

`saved_run.read_records(run, Model | TypeAdapter(Model))` 回按 commit_seq 排序的 `SavedRecord` tuple。型別名不符、欄位驗證失敗或 malformed envelope 會指名 record.json source raise。`read_records_raw(run)` 回保留 JSON 的 `SavedRecord[EncodedRecord]`，供異質 record 使用。Reader 不用 manifest schema 驗證，也不從 leftover native files 推導提交。

`saved_run.read_summary(run)` 讀 metadata root 的 summary.json。缺檔回 None，run 目錄不存在或格式錯誤則 raise。GUI/MCP 不新增 summary result 欄位。

## Ask 與顯示

`await env.ask(question)` 等待一題一個 host 配發的 request_id。Engine 的 pending question 保存原題與 declared answer type。`Engine.parse_answer(run_id, fields, request_id=...)` 先比對題 ID，再驗 schema，不消耗問題。`Engine.answer(run_id, answer, request_id=...)` 驗證並通知原 worker，不執行新 segment。

Stale 或無 pending 的 ID 會 raise QuestionMismatch，攜帶目前 pending_request_id 或 None。拒絕不改 lifecycle，也不記 question_answered。成功 host acknowledgement 後，workflow owner 才保存原 ID、題目與已接受答案。Host control callback 只捕捉記憶體資料，持久化在 callback 之外。

GUI facet 與 MCP status 讀相同 pending envelope。GUI Answer 使用畫面原題的 ID，MCP workflow_answer 必填 request_id。Wrong model 或 schema 不消耗問題。Stop 在同一 worker 交 None 並執行 finally。只有 terminal owner delivery 才釋放 operation、progress 與 RUN lease。

`env.plots` 是原注入的 `WorkflowPlots`，跨 commit、Pause 與 retry 保留 workflow 趨勢圖。每次 `env.run` attempt 經 `start_operation()` 建立同一 presentation host 上的獨立 Plots。返回、Pause 丟棄與失敗都先 finish；下一個 attempt 開始前 release 舊 operation。Core 與 finish 同時失敗時保留 core 例外並附 cleanup note；只有 finish 失敗則直接 raise，不交 Completed。

`WorkflowPlots.figures()` 回 workflow `<name>` 與目前 `operation/<name>` 的同一投影，供 pane 與 MCP 使用。Workflow 圖名登記時拒絕 `operation/` 前綴，通用 Plots 與 operation 圖不受此限制。新 attempt 整組替換 operation，terminal 保留最新一組；宿主在 workflow producer 結束後 finish，tab replacement 或 close release 兩組 presentation，不銷毀原生 Figures。

作者用 typed plot handles 或 draw 更新 workflow 圖。保存圖片時從 `completed.value.result` 重畫，不依賴 operation live 圖的保存位置。`assemble_rows` 與 `assemble_scalars` 將稀疏結果組成 detached 顯示陣列，不插值。

GUI 組裝見 [measure README](../../gui/app/measure/README.md#workflow-hosting)，agent 工具見 [measure MCP](../../mcp/measure/README.md#workflow-tools)。框架不依賴具體 workflow、校準資料或 GUI 選中狀態。

## 驗證 owner

`tests/experiment/workflows/` 從公開 Engine、env 與 saved reader 驗證作者交接、控制、commit 和 I/O failure。`tests/experiment/v2/runtime/` 擁有 Schedule、ProgramBuilder 與 buffer 的行為。GUI/MCP 的註冊、wire 投影與元件操作由各自 owner tests 驗證。
