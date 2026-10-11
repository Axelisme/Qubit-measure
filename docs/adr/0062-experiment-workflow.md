---
status: accepted
---

# ADR-0062 — 實驗執行與 workflow 編排

**Last updated:** 2026-10-11，字串 expression 與固定保存接線

關聯 [[0066]]，operation 與取消。保存權威見 [[0063]]，表單與 cfg 的使用邊界見 [[0065]]。

## 問題與決策

一個 workflow run 可以執行多個步驟，也可以在暫停後重做未提交的步驟。Worker 返回不代表 run 終局。若每次返回都終局 operation，宿主就必須另外保管 progress、目標身分與 Resume admission，並協調多處狀態。

具名 workflow 宣告 plan、tunables 與 init，步驟函式透過 env effects 執行實驗、操作裝置、畫圖與保存。Engine 迭代步驟並記錄結果，不決定具體實驗的校準接受度。Workflow 擁有 cfg 組裝、分析與正常步驟失敗的政策。非正常失敗的例外停止 run，不改標為 Pause。

## 一個 run 的 operation

一個 run 從 Start 到終局使用同一個 OperationRunner operation 與 handle。Start 只 begin 一次。Controller 是 run 的單一 owner，GUI 與 RPC 使用同一個 target 核對與提交入口。Run binding 在通知前提交，pending target 不表示 native startup 已接受。未接受的 run 不能更新 tunables，也不能借用上一個 run 的 draft。

Pause 丟棄當前未提交步驟，worker 返回。已提交 records、cursor、plan、form capture、accepted input、progress、handle 與本程序 RUN lease 都保留，不 settle。Paused 時可以修改 tunables。Resume 重用同一 handle 提交 worker，從 cursor 以最新 tunables 重做未提交步驟，不重做 init。

Complete、Stop 或 failure 只有一個終局入口。有 producer 時，必須等它返回並完成 owner delivery；無 producer 的 paused run 或同步 startup failure，由同一 owner 收尾。清理完成前不能 replacement Start。終局完成才 settle handle、清理 progress 與釋放 lease，清理失敗遵守 [[0066]] 的 best-effort 規則，不承諾跨 State、channel、檔案與硬體的原子交易。

本程序 RUN lease 在 paused 期間仍保留。它不仲裁其他 GUI process 的儀器操作。Measure-gui 在另一 process 執行實驗，不代表 workflows 的 cache 已刷新，也不保證儀器保持某個值。

## 裝置讀回與 Resume

Start／Resume 不再讀取所有已連線裝置，也不在 journal 記錄 device snapshot。Start 表單直接使用 DeviceService 的公開 typed cached native mapping。Resume 只接受 run_id；健康 paused run 的提交失敗不建立新 operation，不清掉原 handle／lease。

Device 實際值只由每次 env.run 在同一 RUN lease 下、experiment callback 前捕捉。DevicePort.capture_snapshot 讀取所有 connected drivers，將固定 JSON 交給 Run，再由 native acquisition evidence 與 metadata 保存。讀取失敗使 run failed，不呼叫 experiment，不使用 cache fallback。缺 devices capability 不等於零 device，真正零 device 的 host 明確提供回傳 {} 的 port。SessionDevices 在 worker 借公開 driver mapping，getter 回 detached 的實際 Info 子類。完整 setup 用 selected driver.info_model 重新驗證 dump 後呼叫 driver.setup，傳同一 StopSignal 的 event。Journal 記錄 name 與完整 cfg；框架不認識 flux、unit 或 device 種類。使用者 set_flux 工具重用已交付的 check_flux_unit。Host capture 重用 native acquisition helper，不更新 GUI State。Device 接線與 Window 的 SessionServices 組裝已完成，不能據此宣稱完整 CP-B 或 integration 全綠。

共用 alias preset、裸 alias 求值與 FormState 的顯式字串 expression 已接線。普通字串仍是 literal，結構 selector 仍使用普通 typed tag。T1 flux_dev 的表單 default 是顯式 $flux_dev expression，與數值欄位使用同一份固定 alias capture。Focused GUI／MCP 表單 seam、固定 RunSnapshot 與 native 再次 Save 已通過，後續 alias 修改與 Save 不重新解析該 snapshot。這些證據不涵蓋 scenario 14 或完整 integration。S4 source reload 與 U7 input 保存還原尚未實作。Base 09 終驗與使用者 CP-B 驗收仍須分別完成，不授權硬體操作。

各裝置的讀值不是跨欄位、跨裝置或跨 process 的原子 snapshot。同步 I/O 可能阻塞，沒有固定 query 或 settling 時間保證。Snapshot 只記錄讀回值，不要求硬體自動復原；workflow 的後續 device effect 仍由 driver 設定值與驗證 limits。

## 取捨與 owner

一 run 一 handle 讓 Pause、Resume、tunables failure 與 worker failure 使用同一終局政策。代價是 paused run 保留本地 RUN lease，同程序的其他硬體 mutation 仍受 lease 阻擋。這不是從 artifact 還原的跨程序 resume，也不提供 hardware rollback。

通用迭代與 journal 契約見 [workflows engine](../../lib/zcu_tools/experiment/workflows/README.md)。GUI run owner 與 tunables 提交見 [workflows host](../../lib/zcu_tools/gui/app/workflows/README.md)。具體定義與 T1 attachments 見 [使用者 workflows](../../zcu_lab/workflows/README.md)及 [GUI attachments](../../zcu_lab/workflows/gui/README.md)。
