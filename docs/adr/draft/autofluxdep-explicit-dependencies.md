# Autofluxdep 逐項宣告依賴與 predictor 載入

狀態：已確認待實作。這是未編號的未來跨模組設計，不代表目前行為。源自已核准的 Experiment Runtime／Autofluxdep workflow P2，以及部分仍有效的 ADR-0018。現行跨模組責任見 [ADR-0062](../0062-experiment-workflow.md)。

## 問題與決定

當點缺少 frequency 或 module 時，無條件沿用前點會把過期輸入帶入新的實驗設定。每項 dependency 應明示是否接受當點、先前產出或 library fallback，及缺值如何處理。Required input 找不到合格來源時 skip 並附原因；型別或宣告錯誤、執行失敗不能偽裝成 skip。Patch 未提供某 key 不能單獨授權沿用舊值。具體型別、fallback 優先順序、alias 與 node 設定須在 resolver／node contract 的實作中核實，不在本 draft 固定所有 node 的依賴表。

另一個問題是 framework 不應預設每個 workflow 都需要 predictor。純計算 provider 仍可用同一需求／產出契約，但只有需要它的 workflow 才載入。Orchestrator 不理解 prediction 的實驗語意。

## 現況與落差

`gui/app/autofluxdep/nodes/spec.py` 的 `Dependency.need` 和 `ModuleDep.need` 都以 `Need.LATEST` 為預設。`orchestrator.py` 的 resolver 會先讀當點，缺值時回退上一點。ModuleDep 另有明示的 `ModuleFallback`，但它的預設是 `LIBRARY`。這尚未實作「每項依賴明示 freshness／fallback，不以 latest-available 作共同預設」的目標。

`services/run_setup.py` 的 `build_run_providers` 目前無條件將 `PredictorBuilder()` 放在 enabled user nodes 前，即使沒有 node 宣告需要其 output。這尚未實作「不因框架預設載入未使用 predictor」的目標。現況不可在 ADR-0062 中宣告成已核准的無預載設計。

## 轉正條件

- 所有相關 dependency 的來源與缺值政策由 node 明示；resolver 不再將 latest-available 與 library fallback 當作未宣告時的共同預設。確認舊 node 的選擇及下游缺值行為，再驗證 skip 和錯誤邊界。
- Run setup 僅在 workflow 需要 predictor 時載入；核實 pure provider 的使用與排序，不讓通用 orchestrator 處理 concrete prediction。
- 實作、caller 文件與驗收完成後，依目標分支分配正式 ADR 編號。未完成時本 draft 保留，不覆寫現行 ADR。

排除拓撲重排、固定的全 node dependency 表、硬體操作及本 draft 直接授權修改程式。
