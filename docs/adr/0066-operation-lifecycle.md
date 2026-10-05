---
status: accepted
---

# ADR-0066 — Operation 執行、取消與關閉邊界

## 問題與決策

GUI、agent 與不同 app 都能啟動長時工作。若把請求前置條件、硬體互斥、結果追蹤和 worker 綁成一個不可分的物件，互動分析就得假借 hardware lease；若把取消當作終局，仍在執行的量測和未清理的連線會被誤報為已停止。本篇分配跨 owner 的執行與生命週期責任；各 operation 的能力和終局政策仍由其 domain owner 決定。

Domain guard 檢查請求所需的前置條件。measure 的 `GuardService` 向 run、save、analyze、writeback 等受保護入口發 typed Permit；Permit 可攜帶已檢查的請求資料，但型別本身不能保證取得之後的 live state 不變。會變動的 busy／hardware conflict 在啟動時由 Exclusion gate 檢查並以 lease 保護。Permit 不需釋放，lease 必須在終局釋放；Handle 只追蹤 operation，不是持有硬體的證明。不同工作按需求組合 exclusion、handle、progress 和 cancel。互動分析可以等待使用者而不佔硬體 lease；data save 與 artifact batch save 使用非取消的 handle 等待真正存檔完成，卻不持硬體 lease。單項同步 image export 不因此取得 handle，auto-align 仍只有背景執行策略。Measure 的 guard 與 app-specific exclusion 政策見 [measure owner](../../lib/zcu_tools/gui/app/measure/README.md)。

`OperationRunner` 集中啟動檢查、handle 建立、可選 lease／progress 配置、背景提交與終局清理。各 operation 提供 work 與 terminal policy，決定 State 寫入、partial result、失敗與 cancelled 的解讀。Runner 不讀寫 State，也不決定實驗停止、device rollback 或 figure policy。`BackgroundExecutor` 僅執行 work 並將 terminal callback 交回 owner；需要的 figure、progress 和 stop scope 由 operation 的 work 組裝，而非由通用 executor 猜測 domain。互動式工作可由使用者動作推進 handle，並非每個 handle 都對應 worker thread。終局 policy 在必要的 domain commit 後才 settle handle；progress discard、settle 與 lease release 分別作 best-effort 清理，失敗會記錄，不能推導跨 State、channel 和硬體的原子交易。

State 的語意寫入由單一 owner loop 序列執行。Qt queued delivery 和 headless owner scheduler 是不同 adapter；worker 不直接提交 State。能阻塞的 await 位於 off-owner caller，owner loop 用通知或非阻塞 poll 推進，不能阻塞等待需要 owner callback 的完成。Await 的期限到達只表示這一次等待結束，不改變 operation outcome。Operation Handle 回報 finished／failed／cancelled 等終局，不承載完整實驗結果；結果由 domain owner 保存，progress 由獨立 read surface 提供。Resource version guard 與 agent 的 handle 呈現屬 [GUI／Remote 契約](0068-remote-transport.md)，不等同於 lease。

## 取消與互動

Cancel 是請求，不是停止、rollback 或 cleanup 完成。`OperationHandles` 的 cancel hook 把請求接到各 operation 的停止能力；`OperationGate` 只管互斥，二者不互相冒充。Measure 的 `OperationControl.cancel_operation` 先確認 handle 有可取消的 domain hook；沒有取消能力時回 `not_cancellable`，未知或淘汰 token 在 remote／agent 入口報錯，不假稱已完成。run 和 device setup 以協作 stop signal 推進，沒有 cancellation point 的 connect 不因發出請求而停。Domain terminal policy 應保留真正的 worker failure，不以 stop flag 一律改成 cancelled；已完成的量測、device setup 恢復及互動 view 的 teardown 各由其 owner 決定。不能僅因 handle 有 cancel hook 就推定所有按 id 取消入口均涵蓋必要的 domain cleanup。

每個 operation 的 `OperationChannel` 只傳遞有序的 `Settled` 與 `Stop(reason)`。Stop 先入列，再呼叫可用的 cancel hook。Awaiter 限時消費，terminal outcome 仍由實際工作決定。多次 Stop 的 reason 依到達順序以 newline 合併，只在 cancelled outcome 的 feedback 欄位交付。這項順序只解決 channel 事件的折疊，不保證 hook、副作用、重入或多個 awaiter 在整個系統都沒有 race。`operation.await` 只回 completed 或 timeout。Agent 主動詢問使用者的 prompt 使用另一條 `NotifyChannel`，以 reply／dismiss／timeout 為事件，不混入 operation outcome。Measure GUI 保留一般 Stop 按鈕與 dialog，由 [app owner](../../lib/zcu_tools/gui/app/measure/README.md) 控制。Measure MCP 的 `wait(op)` 經 `operation.await` 取終局及 Stop reason，不訂閱 EventBus 的 operation feedback，也不把它當成下一次 tool reply 的 event piggyback。GUI 的 EventBus push 仍供其他 consumer 使用。Wire 與 MCP tool 的投影由 [Remote owner](0068-remote-transport.md) 維護。

## Shutdown 與 device 資源

Measure 在 begin_shutdown 時阻止新的 experiment entry，`ShutdownCoordinator` 對當時的 live handles 發出 cancel request，以非阻塞 poll 回報 WAITING／SETTLED／TIMED_OUT。Handle 清單包含 data save 與 artifact batch；仍不能代表 auto-align 等全部背景工作。timeout 僅代表等待期限已到，**不是** cleanup 完成或可安全退出的證明。核准的共用預設是不自動強關，目前 Qt shutdown driver 在 timeout 仍執行 close callback，且未涵蓋所有無 handle 工作。等待 executor cleanup、未完成狀態回報與關閉順序的轉正條件見 [operation draft](draft/operation-lifecycle-boundaries.md)。Autofluxdep 的 running-run 強制關閉是 app 明示選項，不是 generic runner 的預設。

已註冊 device 的 manager close API 由 registry 協調 disconnect；registry claim 防止同一 driver identity 被兩個 manager close 同時處理，driver 自己仍須守住 mutating operation 與單次 I/O 的同步。`drop_device` 只移除登記，不是 disconnect。Notebook 由建立 ResourceManager 的 owner 在所管理的 device sessions 成功 disconnect 後才關閉 factory；manager close 的部分失敗保留未完成資源與錯誤供明確處置，不把嘗試過 close 當成成功，也不保證重試一定成功。Notebook 的已實作順序與 registry failure contract 見 [device owner](../../lib/zcu_tools/device/README.md)；GUI DeviceService 目前仍直接 close/drop，不能宣稱已全部走 manager close API 或已具備對應 factory teardown，核准目標與轉正條件見 [operation draft](draft/operation-lifecycle-boundaries.md)。GUI State 擁有對外可觀察的 device cache，registry 保存 driver identity，cache 不是即時硬體狀態的證明。

Disconnect 只處理 connection/session，不隱含 RF off、歸零、reset 或自動 retry。它也不保證設備在斷線後保持原狀。硬體安全狀態需要 app／experiment 明示的操作與證據，不能從 settled handle、closed session 或 shutdown timeout 推導。

## 取捨與相鄰責任

共用 runner／channel 減少各 operation 重複生命週期與跨線程事件組合，代價是各 domain 仍須清楚定義終局和清理。將長時工作一律包在 hardware gate 或讓 executor 辨認 experiment stop scope 都會混淆 owner；把 Stop reason 與取消拆成兩個通道會失去共同順序。本篇不新增硬體關閉政策、全域交易或強制退出承諾。局部 port、registry 同名 replacement、in-flight claim、driver lock 與 app 操作矩陣見 [session owner](../../lib/zcu_tools/gui/session/README.md)、[device owner](../../lib/zcu_tools/device/README.md) 和各 app README。Workflow segment／Pause／Stop 另見 [[0062]]，GUI process remote adapter 啟停另見 [[0064]]。
