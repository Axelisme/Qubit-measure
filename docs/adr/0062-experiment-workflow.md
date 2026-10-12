---
status: accepted
---

# ADR-0062 — 實驗執行與 workflow 編排

**Last updated:** 2026-10-12，獨立校準結果與 Coherence 提交

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

## Terminal fluxdep 匯出

Engine.records 提供 committed evidence，record.json 不作提交判定。匯出是明確 host 命令，不新增 engine finalize hook 或 Done 自動保存。純資料型別在 experiment.workflows.exporting，科學 payload 在 zcu_lab.workflows.export。Host 的 export_fluxdep_run 接收一次呼叫的 frozen FluxdepExportRequest，使用注入的科學 callback、SaveLayout 與 scoped ledger。Controller 用同一 run binding 固定 Start plan、identity、provenance 與 entry paths，terminal delivery 只補 host UTC finished_at。GUI 動作、RPC 與 MCP 呼同一命令，不讀當前 draft 或硬體。

Native 原子覆寫 workflow data_root/merged.h5。Labber 使用相同 payload 與明填 independent layout，在 SaveLayout 選出的新日期樹路徑保存。Datafile 擁有既有異質 layout 與完整寫入的 finalization。Signal/Time 2D 與 T1 Flux-only 1D 保持原 rank，不 union 或 broadcast。Native/Labber 的 allocation/writer failure 分別回報，未分配 Labber path 為 None。Saved ledger 事件只含 known-path attempts，附檔保留完整報告；ledger 失敗不抹掉已成功路徑。依賴方向由 C8、C12、C16、C18 及 workflow-export-science-no-frontends contract 保護，不讓 framework 反向 import 使用者 adapter。格式與批次保存分工沿 [[0063]]。

## Iteration 目錄與保存責任

使用者在 2026-10-12 裁定：「我以為一直都是框架一開始就給這個iter的儲存路徑，workflow內部自己決定在哪裡存什麼」與「你沒辦法阻止workflow 內部亂搞，也不該試圖保證不亂搞，這些機制都是過度工程」。本裁定將量測與保存分開，不維持舊暫存路徑及檔案發布介面。

Engine 在每次 invocation 開始前建立 `data_root/iter/<至少六位 call_seq>/`。唯讀 `env.iteration_dir` 指向當輪已存在的 absolute 根目錄。屬性不可重綁，不限制 workflow 自行寫檔。Resume 重做未提交步驟時分配新的 call_seq 與目錄。

`env.run` 只負責量測，保留 cfg deepcopy、device capture、Schedule、live callback、StopSignal 與失敗來源。`Completed` 只表示量測成功。Workflow 自行呼叫 native saver，決定存什麼、檔名及是否保存，並在自己的 point record 記錄檔名與用途。校準保存仍由既有 native writer 處理。Frequency、Rabi、readout 各自回傳 frozen 具名結果，包含自己的 status、reason、accepted scalars 與 immutable 檔名引用。Frequency widening 保留全部 Completed attempts，fit 拒絕也保留 raw 檔名；Schedule Failed／skipped 不捏造檔名。Caller 將各 stage 結果放進 point record，不用整體校準成功收據丟掉前面已接受的值。框架不提供檔案清單，不預建量測或圖形子目錄，不追蹤、掃描或保證 workflow 檔案。

Workflow saver 例外停止 run，不改標為量測 `Failed`。Pause、Stop 或程序中斷可能留下未提交或寫一半的檔案，框架不清理、不承諾 rollback。Point record 的檔名也不保證檔案存在或完整，由讀取者處理缺資料。

Engine 仍擁有 manifest、journal 與 best-effort `record.json`。結果 journal append 與 flush 成功後，Engine 才用同目錄 temporary 加 replace 保存摘要。摘要 I/O 失敗只記 log，不影響提交或下一步。Manifest 的 atomic replace 也保留。這兩種框架文件的替換，不代表 workflow 量測檔案有發布保證。依賴方向仍由 C18 與 framework-no-user contract 保護。

## Coherence 與部分校準結果

具體 workflow 的 stage 隔離不改 Engine 的 whole-step 提交。Frequency 接受即更新自己的 history；Rabi 需要本點 frequency，readout 與 enabled lifetime 需要本點 frequency/Rabi。Readout 非接受時沿合法 seed／prior accepted 設定繼續量 lifetime，不用 prior frequency 或 pi 取代缺少的本點依賴。Carry 只在 step 副本中更新，由 Next 發布，Pause／Stop 仍丟棄整點，不增加 stage checkpoint。

Coherence 固定 plan flags，按 T1、Ramsey、Echo 量 enabled 項目。三份 lifetime history 獨立，只有各自接受的值更新 delay grid。Point 的 measurements tuple 只含 enabled 結果；disabled 不用 null placeholder。各個 completed raw 都有明示用途的 relative native reference，讓轉換工具取得部分證據，而不從 acquisition 次序或目錄猜資料。

校準與具體 lifetime policy 留在 zcu_lab.workflows。共用 CalibrationInputs 服務 Coherence 與 MIST 的三個 native stage，readout_due 讓兩個 caller 共用零基 flux index cadence。這些 module 不依賴 frontend，方向由 workflow-science-no-frontends 與 C16 保護。GUI forms/capture、單一 Controller 與 operation 邊界不改；不增加通用 pipeline、registry、export bridge 或 saved input migration。

## Overnight live 趨勢與保存責任

Live 趨勢圖只用 Engine.records 的 committed evidence。具體 scalar 投影與繪圖在 zcu_lab.workflows.trend，不在框架重 fit。GUI presentation owner 在 zcu_lab.workflows.gui.figures，與 Window 固定。它只持有 native Plots 和繪圖 cursor，沒有 Engine、run target 或 operation。Window 沿既有約 50ms poll 更新 live 圖，最後提交刷新後才 finish。Pause 保留 Plots，replacement Start 與 close 釋放 presentation，不銷毀 caller 保留的 Figures。

使用者在 2026-10-12 裁定：「趨勢圖也不用，後來用戶自己從數據重畫最自由，但可以workflow實作者存每個iter希望存的圖」與「讓workflow實作者自己在內部存，框架保持簡單」。本裁定取消趨勢圖保存與重開能力，保留現有 live 顯示。框架不提供 committed journal reader，不自動保存 overnight 趨勢圖，不從 run 目錄重開互動式圖，也不新增 env.save_figure 或 Engine 提交通知。使用者可自行從 record.json 或 native 資料重畫；workflow 實作者若要保存 iter 內的圖，在自己的 workflow 程式內處理。

Journal writer 與 Engine 提交順序不變。Iteration 檔案保存依上節由 workflow 擁有。現有具名圖 RPC、PNG renderer 與 Plots 工具也不變。依賴方向由 C18 與 trend-science-no-frontends contract 保護，科學投影不依賴 frontend admission。

## 取捨與 owner

一 run 一 handle 讓 Pause、Resume、tunables failure 與 worker failure 使用同一終局政策。代價是 paused run 保留本地 RUN lease，同程序的其他硬體 mutation 仍受 lease 阻擋。這不是從 artifact 還原的跨程序 resume，也不提供 hardware rollback。

通用迭代與 journal 契約見 [workflows engine](../../lib/zcu_tools/experiment/workflows/README.md)。GUI run owner 與 tunables 提交見 [workflows host](../../lib/zcu_tools/gui/app/workflows/README.md)。具體定義與 T1 attachments 見 [使用者 workflows](../../zcu_lab/workflows/README.md)及 [GUI attachments](../../zcu_lab/workflows/gui/README.md)。
