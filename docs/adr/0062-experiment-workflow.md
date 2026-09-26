---
status: accepted
---

# ADR-0062 — 實驗執行與 workflow 編排

關聯 [[0026]]（operation 與取消）、[[0027]]（單一實驗資料）、[[0029]]（物理 prediction 能力）、[[0040]]（autofluxdep artifact）、[[0045]]／[[0046]]（通用 cfg 模型與 lowering）。兩套 workflow 的具體節點介面不同，不因共用本篇而相互轉換。

## 問題與決策

一般 `Schedule` 的單一結果 buffer、executor 的多量測結果樹、autofluxdep GUI 的依賴式 Node，以及 GUI 的 Run 操作有不同的生命週期。若讓通用 runtime 理解 measurement identity、fit 結果或 GUI operation，新增實驗就得修改不擁有該領域政策的層。

`experiment.v2.runtime` 提供 host loop、acquire、buffer update、stop、retry 與 cleanup 機制。具體實驗擁有 cfg 生成、fit、Result 與輸出政策。一般 `Schedule` 只要求 `BufferProtocol` 的 `data` 和 `trigger_update`；executor 擁有 `ResultTree`，在 buffer 邊界把路徑轉成 per-measurement `ResultUpdateEvent`。consumer 不靠解析 `ScheduleStep.path` 辨認 measurement。`flush=True` 讓結果更新立即送給 subscriber，**不是**資料落盤或 artifact commit。

`MultiMeasurementExecutor` 擁有多量測的共同 run lifecycle 與 `MeasurementBundle` leaf contract；具體 executor 擁有 flux／iteration 外圈政策。autofluxdep GUI 的 Builder／Node 不實作 `MeasurementBundle`。兩套介面不在本決策中統一，也不決定它們的永久去留。

## Autofluxdep 的執行邊界

GUI app 的 orchestrator 按使用者 placement 的相對順序執行，不依 dependency 隱式拓撲重排。它依宣告解析當點的輸入，建立 `RunEnv`、建立短命 Node、驗證並合併對外的 `Patch`；Builder 是無 run state 的定義與工廠。每次 run 的 Result、Tools、feedback 與 InfoStore 留在各自的 run owner。純計算 provider 也使用依賴與輸出契約，不必偽裝為硬體量測。acquire、fit 接受度、缺值後的實驗選擇及 predictor 校正不歸通用 orchestrator 判斷。

Resolver 支援 `Need.NOW`（只接受當點）及 `Need.LATEST`（允許前點），module dependency 還可宣告 library fallback。Required input 無合格來源時記錄 skip 原因；Node 執行失敗不等於 dependency skip。Run setup 目前另行前置 `PredictorBuilder`；dependency 的共同預設與未使用 predictor 的預載不符合已核准的逐項明示目標，現況與轉正條件見 [dependency draft](draft/autofluxdep-explicit-dependencies.md)。不得把現有預設推廣成新 Node 的政策。

## 設定與回饋

Run 開始時，app 以 run-local `ModuleLibrary` 為 enabled Node 建立 `RunCfgSnapshot`，保存 lowered `base_cfg`、generation `knobs` 與 Builder 宣告的 `OverridePlan`。每點由 base 產生 cfg，僅套用 plan 允許的 patch。`all_points` 每點必填；`after_first_point` 第一點沿用 base、後續必填；`fallback` 有 patch 才覆寫。未宣告 path、不存在的 target、違反 mode 或 whole-module replacement 都由 runtime 拒絕。GUI decoration 只向使用者說明範圍，不代替 runtime enforcement。通用 cfg model／renderer 不理解實驗 generation 政策。

App 的 workflow editable cfg、run-start base／plan 與實際 point cfg 是不同的觀測對象。artifact 保存 editable cfg 與 run-start base／plan，不因此保證每點最終 cfg 的完整 provenance，也不聲稱 md、ml、device 在同一原子時刻擷取。資料 commit 與 finalize 規則歸 [[0040]]。

Feedback 是 run-lived 且 placement-scoped 的 capability，不讓同一 Builder 的多個 placement 共用可變 state。其 estimator／controller state 不作為 `Patch` dependency；提供下游消費的實驗結果仍經正式 `provides`／`Patch`。Generic feedback 返回 sample、query age 與 freshness；disabled 或沒有 observation 返回 `None`，未宣告 slot 則失敗。它不決定 fit gate、clamp、fallback target 或 stop。Node 決定如何將 sample 與 domain prior 組合。

`qubit_freq` 在 run 中保留 raw/base predictor 不變；可信的 physical recovery 如有啟用，只安裝 run／placement-local overlay。Residual correction 由 feedback slot 保存，不修改共用 predictor。這取代了舊篇讓 Node 直接校準 raw predictor 的 hard-bias 描述；物理模型的通用能力仍歸 [[0029]]。Estimator decay 與 node 的混合公式留在 [autofluxdep README](../../lib/zcu_tools/gui/app/autofluxdep/README.md)，不是另一項跨模組決策。

## Run 與 operation

App 的 `RunSession` 可跨多個同程序 execution segment。每個 Run／Continue segment 使用自己的 `OperationRunner` operation 與 hardware lease。Pause 在 flux boundary 結束 segment：已完成的 point 先 commit，artifact 記錄 non-terminal pause；lease 釋放，InfoStore、feedback、結果和 writer 保留供同一 session Continue。Continue 開新 operation，從保存的 cursor 恢復。Stop 是 terminal cancellation／finalize，Restart 建立新 session，不沿用舊 feedback。Failure 不會改標為 Pause。App 在 running／paused 時守住 workflow input mutation；釋放 lease 不表示硬體狀態不變。這不是從 artifact 還原的跨程序 resume，也不保證硬體自動復原。

取得 raw data、fit 成功與提供合格 Patch 是三個不同事件。資料可保留，而 operation 仍維持 failure outcome。Data-driven early stop 只結束當次 acquire；使用者取消走 run stop。Runtime 的 program-acquire retry 和 executor 的 per-measurement retry 各有自己的 attempt 邊界，不推廣成 GUI Node 自動 retry。具體 API 與 liveplot、retry 使用方法見 [runtime README](../../lib/zcu_tools/experiment/v2/runtime/README.md)；app 的 run 操作與 artifact 見 [autofluxdep README](../../lib/zcu_tools/gui/app/autofluxdep/README.md)。遠端目前只觀測 app-owned state，不授權遠端 Run、Stop 或 cfg mutation。

## 取捨

保留使用者排序與兩套 workflow 接口，避免為了統一型別而讓純計算 Node 偽裝量測，或讓 `Schedule` 了解 GUI workflow。代價是兩套外層 policy 仍各有 owner；後續若合併，需要另行核實接口、結果生命週期與 caller，不在此篇承諾 recovery 演算法或跨程序恢復。
