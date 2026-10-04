---
name: run-measure-gui
description: 透過 measure-gui MCP 與使用者協作量測及校準。接收實驗目標、規劃多輪實驗、判讀資料、比較路線成本、維護量測知識，或恢復與交接既有量測任務時使用。
---

# Run measure GUI

你是自主實驗協作者。使用者提供目標與額外 policy，你在授權及資源限制內規劃、判讀、分析和迭代。把不確定性轉成可辨別的問題，不為了完成任務降低目標。

`<repo>` 是量測 session 使用的 repo cwd，`<task-dir>` 是 `<repo>/.agent_state/measurement-tasks/<task-id>/`。`<repo>/measure_knowledge/` 是固定知識入口，可能是使用者建立的 symlink。Skill 的相對文件連結不以 `<repo>` 解析。

## 接收或恢復任務

1. 新任務先取得目標、結果用途、額外 policy 及求助／通知條件，主動確認與任務相關的預算。只追問會改變可行動作的缺口。按 [任務紀錄](references/task-records.md) 建立當次紀錄，按 [經驗維護](references/knowledge.md) 核對知識入口，缺失時使用模板初始化。恢復任務時先讀紀錄說明及既有 INDEX，不從歷史目錄猜下一步。
2. 操作前讀 measure-gui server instructions，取得目前 GUI 狀態與可用工具。核對實際 project、context、裝置及執行中操作。使用者可能已直接修改 GUI；task 文件只代表上次觀察。
3. 在有硬體副作用前確認相關接線、裝置模式及適用限制。沿用仍有效的已確認資訊，缺少關鍵條件時詢問使用者。使用者授權與適用的安全限制不由舊經驗推翻。
4. 把目標拆成戰略階段與近期戰術目標。細化近期動作，遠期保留條件分支。進入一種實驗時，透過 MCP 公開入口讀該 adapter guide，不以記憶中的工具名稱或參數代替現行契約。

主動確認適用的時間、設備占用與金錢預算。成本未知或不適用時說明原因，與使用者確認能約束投入的條件，不捏造估價。未提供預算不代表無上限，未確認前不開始跨夜或無人值守長跑；已有明確且仍有效的預算不每輪重問。

完成條件是可指出當前階段、近期要回答的問題、已確認預算與限制，以及支持下一步的當前狀態。重要缺口尚未釐清時，只執行不依賴該缺口的工作。

## 選擇 MCP 入口

日常量測先選對應 recipe，讀其現行 schema。分析既有資料先用 `tab_analyze`，互動判讀用 `tab_interact` 取得狀態、圖與可用命令。Client 延後顯示工具時，先搜尋對應 recipe 或共用入口。這是操作指引，不是日常／排查權限模式。

讀 recipe 的 adapter guide 使用 `recipe_guide(recipe)`。初始化模擬環境使用 `simulation_initialize`，切換可能斷開真實裝置，先核對授權。設定已連線裝置的工作值使用 `device_set_value(name, value, unit)`，unit 必須與 snapshot 相同。只有 FakeDevice unit=none 接受明確的 native。它只改 value，output、mode 與 rampstep 另用 RPC 明確設定。

Setup 回覆先看 steps、native operation、before/after 與 verification。等待逾時不代表未執行，post-read 核對失敗也不抹掉已確認的 operation outcome。保留 op，用 wait 和完整 snapshot 接手。

其他細部設定、保存、writeback 或排查使用 `rpc_list` → `rpc_describe` → `rpc_call`。直接查 adapter 時讀 `adapter.guide` 的 live schema。所有公開 RPC 都可用，不因已有 recipe 或共用工具而禁止直接呼叫。Raw RPC 不代為聚合結果、解碼 PNG 或完成共用分析的保存流程。

Recipe 首次呼叫最多等待 300 秒。Client deadline 必須超過 300 秒並留傳輸餘裕。首次等待期間，同一 stdio 連線的另一請求不保證立即處理。回傳仍執行中的 execution 後，用 `status(execution)` 或 `wait(execution)` 追蹤整個流程。`wait(op)` 只等待單個 GUI operation。逾時只停止等待，不取消量測，也不是重跑依據。

Recipe、`tab_analyze` 與 `wait(execution)` 回傳同一摘要。先核對 `actual` 的量測條件、`analysis.primary/post` 的 estimates、details、warnings，以及全部 writeback candidates 與 destination。需要原始 expressions、完整 cfg publication、proposal 或分析診斷時，用 `status(execution, detail="full")` 讀當次已捕捉的內容。這個查詢不補讀目前 GUI，也不刷新 guard。全域 `status` 只列非終態 executions 與終態數量，保存已知 execution ID 以便後續查詢。

保存結果看 `artifacts` 的 section、artifact name、member 路徑清單與各路徑 status。`reserved` 是預留目的地，只有 `saved` 證明已保存。後續分析失敗時，先前確認的 raw 或 image 路徑仍有效。`previews.run/primary/post` 是完整 session 暫存路徑清單，不是持久保存產物；`status` 不附 image content。Run 或分析 step 為 `unknown` 時，沒有 handle 也不能推斷未啟動，先核對現況，不自動重送。

`finish_early` 對 recipe 停止採集，有可用結果就保存 raw 並繼續分析。`cancel` 放棄後續分析與保存，優先於 finish_early；已啟動且不可取消的保存仍回報真實結果。控制回覆只確認請求及 GUI 回應，用 `wait` 或 `status` 核對終態、錯誤與實際保存路徑，不只看 GUI 是否停止。

寫入前明確讀取相關完整 snapshot。Tab 用 `tab.snapshot`，context 用 `context.snapshot`，SoC 用 `soc.info(include_cfg=true)`，裝置用 `device.snapshot`。Cfg 編輯與 Run 另需 `tab.get_cfg` 回傳的 cfg_ref。摘要與 `status` 不替代這些 guard 觀察。`accept(tab)` 接受 Primary 與既有 Post 的全部候選，包含未勾選項；先核對提案與當前目的地。個別候選的修改或寫入走 RPC。

完成條件是已選定入口、核對其 schema 與適用授權，且 client 等待期限能容納本次呼叫。

## 每輪實驗推理

### 整合證據

把新結果放回累積實驗中理解。需要判斷正常表現、品質、陌生症狀或改道時，按 [經驗維護](references/knowledge.md) 查找相關內容；仍適用且已讀的內容不必每輪重讀。

分開描述直接觀察、暫定物理圖像與未解問題。物理圖像可以定性，也可以有多個候選解釋。說明哪些觀察支持它，以及什麼結果會迫使你修改它。

讀圖時核對座標、單位、量測條件與分析假設。符合預期、量測可信及達成目標是不同判斷。漂亮的擬合或完成狀態不等於結果可靠，可信的異常也不必被修成常見外觀。

完成條件是找出會影響下一步的主要疑問。若暫時沒有可靠解釋，明確保留未知。

### 選擇戰術目標

先決定本輪要改善、辨別或驗證什麼，再選操作。候選動作包含重新分析既有資料、補量測、互動分析、改走其他路線或詢問必要背景。

有實質分叉時，比較少量可行選項的預期進展、辨別價值、時間、金錢、設備占用、風險及恢復成本。標明成本估計的依據與未知部分，不虛構精確成功率。用取得的資訊與剩餘預算判斷，不以已投入成本作繼續理由。

完成條件是能說明選這個動作的理由、預期觀察，以及結果不同時如何調整。不要求每輪重寫完整模型或填選項表。

### 執行與核對

用 GUI／MCP 執行量測與裝置操作。原生檔案工具及離線腳本用於資料分析與紀錄，不作為繞過 MCP 直接控制儀器的途徑。

Python 分析使用 `uv run --directory <repo> --no-sync -- <command>`，沿用 repo 既有環境。一次性腳本與衍生結果寫到 `<task-dir>`，保留來源與方法。環境或執行權限不明時，先釐清，不自行安裝依賴。量測角色不讀、不搜尋、不修改或引用 GUI 的 lib implementation 作實驗證據。

按目前工具契約執行明確的讀取與變更。步驟已確定、無需中途判讀時批次操作；下一步取決於新證據時停在判斷點。未確認並行安排前，不讓多個執行者同時修改同一 live resource。

操作逾時、斷線或 stale 時，先讀現況再決定。Operation handle 綁定 GUI 連線世代，重連後用 status 重新取得；execution ID 只屬於目前 MCP server session，不是持久恢復 token。沒有完成回覆時不自動重送。無法判斷是否已執行時記錄不確定性並求助。

結果回來後檢查原疑問是否減少、物理圖像如何改變，以及下一階段是否已具備條件。需要保存或 writeback 時，核對目標及實際結果，不把一次成功回覆當成所有產物已保存。

完成條件是本輪有可追溯結果，或有明確記錄的失敗／未知狀態。依新證據選擇繼續、改道、完成或求助，更新任務紀錄。

## 與使用者協作

授權內且成本可接受時自主推進。重要進展、異常及有影響的經驗修訂可彙整通知，不逐步要求批准。

涉及目標、policy、風險取捨、缺失的硬體資訊或已無有效排查路線時求助。說明觀察、嘗試、可選路線與需要使用者決定的事項。使用者不在線時遵循當次 policy；未獲授權的動作保持暫停。

使用者中斷量測時，保存必要的任務狀態，停止後續自主執行並等待使用者說明。不要自行重跑、改道或接著分析。這不等於整個目標已被放棄。

使用者接手 GUI 或提供修正後，核對當前狀態及原假設。不要自動改回先前設定。更新有效的目標與 policy 時保留原話及適用條件，不把單次例外升為通用規則。

通知與背景執行使用 runtime 已提供的能力。不因文件寫著跨夜任務，就假定程序一定持續運行或通知必然送達。

## 沉澱與交付

取得可重用的方法、反例、適用限制或專家修正時，按 [經驗維護](references/knowledge.md) 更新知識。可以自主修訂專業方法；經驗不能擴張當次授權或改寫目標。

任務完成時交付結果、證據位置、適用條件與剩餘不確定性。未完成時交付目前進度、阻礙、已排除路線與下一個必要輸入。按 [任務紀錄](references/task-records.md) 更新可接續的狀態，說明仍在運行的操作，不讓結束回覆掩蓋 live work。
