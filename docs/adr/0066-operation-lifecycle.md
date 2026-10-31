---
status: accepted
---

# ADR-0066 — Operation 執行、取消與關閉邊界

## 問題與決策

GUI、agent 與不同 app 都能啟動長時工作。若把請求前置條件、硬體互斥、結果追蹤和 worker 綁成一個不可分的物件，互動分析就得假借 hardware lease；若把取消當作終局，仍在執行的量測和未清理的連線會被誤報為已停止。本篇分配跨 owner 的執行與生命週期責任；各 operation 的能力和終局政策仍由其 domain owner 決定。

Domain guard 檢查請求所需的前置條件與 tab busy。Measure 的 `GuardService.admit(tab_id, action) -> AdmissionDecision` 回傳是否放行、第一個拒絕原因與共用訊息。UI 的按鈕與 tooltip、application facet 和 remote 預檢讀取同一份判斷。`check_load/analyze/writeback` 是 admit 的薄包裝，成功回傳 None，失敗以 decision 的 reason/message raise `GuardError`。Qt 與 remote controls 在同次 owner-thread command 先 check，再把 tab_id 交給 service。Admission 不取得硬體 lease，hardware exclusion 仍由 gate 在啟動時裁決。

2026-10-28 使用者以「選甲」選擇合併靜態前提與 tab busy 的 domain admission，hardware exclusion 留在 gate；再以「同意TaskC」確認此責任分工。

Save 的前置條件由 `SaveService` 單一檢查，無 Run result 時保留 `no_run_result` 與「No run result available to save.」，缺歷史 metadata 或 captured storage 時回 `storage_unavailable`。Run 由 `capture_run_inputs` 先呼叫一次 admit，拒絕一律以 decision 的 reason/message raise `GuardError`，包括 invalid cfg。放行後只呼叫一次 `accept()`，其執行時 resource／reentrancy 拒絕保留 `CfgPreconditionError`。Capture 一次捕捉 `RunInputs` 的 cfg、device observations、parameters、snapshot 與 storage。角色取同次 parameters，entry／point 取原 storage source。RunService 使用這些輸入，不回讀 live State 取代它們。RunInputs 是資料，不是不可偽造 proof，也不需釋放。四個只含 tab_id 的 Permit 已刪除，不留相容別名。Tab busy 歸 guard，startup、其他 tab Run 與 hardware conflict 在啟動時仍由 owning service 與 Exclusion gate 檢查，hardware lease 在終局釋放。Handle 只追蹤 operation，不是持有硬體的證明。不同工作按需求組合 exclusion、handle、progress 和 cancel。互動分析可以等待使用者而不佔硬體 lease；GUI data Save 與 artifact batch Save 在主執行緒同步完成，不取得 handle 或硬體 lease。mapping、PNG 編碼、文件寫入與 ledger 追加在同一次呼叫完成，tab.save_* 直接回傳該次 SaveReport 的完整輸出投影。MCP 不 await Save 或再讀 snapshot。Save 只需數秒，短暫停頓比 worker、跨執行緒回傳與報告來源追蹤簡單。auto-align 仍只有背景執行策略。Measure 的 guard 與 app-specific exclusion 政策見 [measure owner](../../lib/zcu_tools/gui/app/measure/README.md)。

`OperationRunner` 集中啟動檢查、handle 建立、可選 exclusion／progress 配置與終局清理。Async worker、lexical 同步 body 與跨 callback 的互動式 lease 共用這套生命週期，async 入口另負責背景提交。Domain owner 決定執行內容、State commit 與 finished／failed／cancelled 的解讀，不再自行組合資源開啟及清理序列。Core 返回資料的 partial 語意由作者負責。Runner 不讀寫 State，也不決定實驗停止、device rollback 或 figure policy。

同步入口把資源存續期與 ambient event-origin scope 分開。Explicit lease 可以跨 callback 保留，直到 domain owner 決定 outcome 並 settle；inline 則把同一機制包成 lexical context manager。Event origin 只在同步 body 或各次 callback 執行期間進入，離開時還原，不跨 event-loop turn 留在 ContextVar。2026-10-28，使用者以「強烈建議的都採用」採用同步 lease 回到 OperationRunner 的方案，讓 Device、SoC 連線與互動分析共用失敗隔離及資源釋放順序。

Async operation 提供 work 與 terminal policy。`BackgroundExecutor` 僅執行 work 並將 terminal callback 交回 owner。Work 邊界的 BackgroundError 只含 Exception 或 KeyboardInterrupt，executor 交付原錯誤物件，不包裝、不吞掉，也不擴大為攔截所有 BaseException。KeyboardInterrupt 不自動分類為 domain cancellation。這讓新向外傳播的中斷也能走 owner 的 failed 與資源清理路徑，不留下 busy 或 lease。2026-10-31，使用者確定 core 錯誤直接傳出，並以「Ba Cb , 因為還沒用過 Da」選擇 Stop 後 core 拋錯仍 failed 的 B(a)；作者與前端的責任見 [[0069]]。

需要的 figure、progress 和 stop scope 由 operation 的 work 組裝，而非由通用 executor 猜測 domain。互動式工作可由使用者動作推進 handle，並非每個 handle 都對應 worker thread。終局 policy 在必要的 domain commit 後才 settle handle；progress discard、settle 與 lease release 分別作 best-effort 清理，失敗會記錄，不能推導跨 State、channel 和硬體的原子交易。

純分析 operation 不需要注入硬體 gate。Runner 在開 handle 前拒絕缺少的必要 facet。Optional on_opened hook 在 progress notification 與 submit 前交出 token，domain owner 可先保留 admission，並在 startup failure 後發布同一 token 的 terminal activity。Fluxdep search 由 app owner capture detached input 與來源版本，worker 成功後只提交仍有效的結果。GUI 與 remote caller 共用這份 owner policy，細節見 [fluxdep owner](../../lib/zcu_tools/gui/app/fluxdep/README.md)。

State 的語意寫入由單一 owner loop 序列執行。Qt queued delivery 和 headless owner scheduler 是不同 adapter；worker 不直接提交 State。能阻塞的 await 位於 off-owner caller，owner loop 用通知或非阻塞 poll 推進，不能阻塞等待需要 owner callback 的完成。Await 的期限到達只表示這一次等待結束，不改變 operation outcome。Operation Handle 回報 finished／failed／cancelled 等終局，不承載完整實驗結果；結果由 domain owner 保存，progress 由獨立 read surface 提供。RPC holder 檢查與 agent 的 handle 呈現屬 [GUI／Remote 契約](0068-remote-transport.md)，不等同於 lease。控制權轉移或斷線不取消已接受工作，新 holder 可取消，observer 可 wait，見 [[0070]]。

## 取消與互動

Cancel 是請求，不是停止、rollback 或 cleanup 完成。`OperationHandles` 的 cancel hook 把請求接到各 operation 的停止能力；`OperationGate` 只管互斥，二者不互相冒充。Measure 的 `OperationControl.cancel_operation` 先確認 handle 有可取消的 domain hook；沒有取消能力時回 `not_cancellable`，未知或淘汰 token 在 remote／agent 入口報錯，不假稱已完成。run 和 device setup 以協作 stop signal 推進，沒有 cancellation point 的 connect 不因發出請求而停。Domain terminal policy 保留真正的 worker failure，不以 stop flag 一律改成 cancelled。GUI 單次 Run 的 core 正常返回就發布結果並標 finished，不論停止對象是 root 或子孫。GUI Stop、MCP 不帶 scope 的 cancel 與 cancel_scope("run") 都遵守此規則。提早結束由 stop_requested 與存檔 completion stopped 表達。Core 拋錯就交付原錯誤、標 failed、不發布 record，即使此前已送 Stop。Failed Run 的 source identity 與 stop_requested 仍記該次請求事實，不代表有可保存結果。已完成的量測、device setup 恢復及互動 view 的 teardown 各由其 owner 決定。不能僅因 handle 有 cancel hook 就推定所有按 id 取消入口均涵蓋必要的 domain cleanup。

2026-11-01，使用者以「6b 其餘採用建議」選擇 core 正常返回一律 finished，不依取消的 scope 分類 GUI terminal。Workflow root 仍是 [[0062]] 的緊急停止，drain 並記 facts 後 raise ScriptCancelled，不能套用 GUI 單次 Run 的 finished 規則。Core 正常回傳後才接受的停止請求仍可更新 stop_requested，但不回寫已固定的 acquisition completion。

每個 operation 的 `OperationChannel` 只傳遞有序的 `Settled` 與 `Stop(reason)`。Stop 先入列，再呼叫可用的 cancel hook。Awaiter 限時消費，terminal outcome 仍由實際工作決定。多次 Stop 的 reason 依到達順序以 newline 合併，只在 cancelled outcome 的 feedback 欄位交付。這項順序只解決 channel 事件的折疊，不保證 hook、副作用、重入或多個 awaiter 在整個系統都沒有 race。`operation.await` 只回 completed 或 timeout。Measure GUI 保留一般 Stop 按鈕與 dialog，由 [app owner](../../lib/zcu_tools/gui/app/measure/README.md) 控制。Measure MCP 的 `wait(op)` 經 `operation.await` 取終局及 Stop reason，不訂閱 EventBus 的 operation feedback，也不把它當成下一次 tool reply 的 event piggyback。GUI 的 EventBus push 仍供其他 consumer 使用。Wire 與 MCP tool 的投影由 [Remote owner](0068-remote-transport.md) 維護。

## Shutdown 與 device 資源

Measure 在 begin_shutdown 時阻止新的 experiment entry，`ShutdownCoordinator` 對當時的 live handles 發出 cancel request，以非阻塞 poll 回報 WAITING／SETTLED／TIMED_OUT。同步 Save 不在 Handle 清單中，也沒有要等待的 Save worker。清單仍不能代表 auto-align 等全部背景工作。timeout 僅代表等待期限已到，**不是** cleanup 完成或可安全退出的證明。共用 Qt shutdown driver 目前在 timeout 仍執行 close callback，且未涵蓋所有無 handle 工作。共用 timeout 不自動強關是尚未落實的目標，不是這個 driver 的現行保證。未完成邊界及轉正條件見 [operation draft](draft/operation-lifecycle-boundaries.md)。

Measure 的 caller 已另有背景 drain。`Controller.close(timeout_ms=5000)` 在 owner thread 停止接受新工作，再等待 worker 與 owner delivery；逾時 raise TimeoutError，保留已關閉的 admission 供重試。成功後重複 close 無害。MainWindow 在保存狀態、停止 remote 與釋放 view 前呼叫 close；逾時保留視窗，讓使用者選「再等一下」或明示「強制關閉」。強制關閉接受晚到 callback 的風險，不能當作 cleanup 完成。使用者於 2026-10-15 以「A+」選擇背景 drain 逾時留窗、再等或明示強關的方案，避免未停止的 worker 回呼已銷毀的 view。

這個 measure-specific drain 不取代 begin_shutdown 的取消協調，不處理硬體斷線或 RF 安全，也不表示 shared shutdown driver、其他 app 或 GUI device/factory teardown 已全部完成。Autofluxdep 的 running-run 強制關閉是 app 明示選項，不是 generic runner 的預設。

2026-10-21 註：autofluxdep 已移除，上一段的 Autofluxdep 選項是歷史行為。Measure 的現行關閉與資源清理見 [measure owner](../../lib/zcu_tools/gui/app/measure/README.md)；這不表示它承接 Autofluxdep 的強制關閉選項。

已註冊 device 的 manager close API 由 registry 協調 disconnect；registry claim 防止同一 driver identity 被兩個 manager close 同時處理，driver 自己仍須守住 mutating operation 與單次 I/O 的同步。`drop_device` 只移除登記，不是 disconnect。Notebook 由建立 ResourceManager 的 owner 在所管理的 device sessions 成功 disconnect 後才關閉 factory；manager close 的部分失敗保留未完成資源與錯誤供明確處置，不把嘗試過 close 當成成功，也不保證重試一定成功。Notebook 的已實作順序與 registry failure contract 見 [device owner](../../lib/zcu_tools/device/README.md)；GUI DeviceService 目前仍直接 close/drop，不能宣稱已全部走 manager close API 或已具備對應 factory teardown，核准目標與轉正條件見 [operation draft](draft/operation-lifecycle-boundaries.md)。GUI State 擁有對外可觀察的 device cache，registry 保存 driver identity，cache 不是即時硬體狀態的證明。

Disconnect 只處理 connection/session，不隱含 RF off、歸零、reset 或自動 retry。它也不保證設備在斷線後保持原狀。硬體安全狀態需要 app／experiment 明示的操作與證據，不能從 settled handle、closed session 或 shutdown timeout 推導。

## 取捨與相鄰責任

共用 runner／channel 減少各 operation 重複生命週期與跨線程事件組合，代價是各 domain 仍須清楚定義終局和清理。將長時工作一律包在 hardware gate 或讓 executor 辨認 experiment stop scope 都會混淆 owner；把 Stop reason 與取消拆成兩個通道會失去共同順序。本篇不新增硬體關閉政策、全域交易或強制退出承諾。局部 port、registry 同名 replacement、in-flight claim、driver lock 與 app 操作矩陣見 [session owner](../../lib/zcu_tools/gui/session/README.md)、[device owner](../../lib/zcu_tools/device/README.md) 和各 app README。Workflow segment／Pause／Stop 另見 [[0062]]，GUI process remote adapter 啟停另見 [[0064]]。
