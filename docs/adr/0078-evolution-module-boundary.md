---
status: accepted
---

# ADR-0078 — Evolution 黑盒邊界與算法內聚

## 背景

演化引擎需要同時比較不同算法的適用條件、準備成本與資料重用。
數值 certificate 又依賴各算法的 eigensystem、週期表或其他中間結果。
若為了隔離內部模組而隱藏這些資訊，分類會重算資料，或只能靠不完整的宣告選算法。

[[0077]] 的實驗 constant planner、shadow graph、profiles 與 checked plan replay 已移除。
它不再是執行契約。現行引擎由 dispatch 決定候選順序、builder、frame 與 chain，
各算法提供數值 build、apply 和 certificate。

## 決策

### 黑盒邊界是 simulate.evolution

使用者於 2026-10-18 定案：「Q1A，是的，演化引擎黑盒子就是指simulate.evolution。」

外部透過 `simulate.evolution` package root 匯出的名稱呼叫引擎及使用資料型別。
Kernel、dispatch、scheduler 與 cache 是內部實作，不是第二套整合入口。
呼叫端提供 numeric model source、timeline、初態與誤差預算，不逐段選算法。
引擎回傳 readout 起點的量子態與 family 診斷。
Environment 顯式組合 population 轉換與 ADC；readout 的樣本加總仍由 `simulate.adc` 擁有。

### 內部允許實質耦合，dispatch 擁有決策知識

使用者於 2026-10-18 裁決：「Q2只要是有助於分類的資訊，planner都可以知道。」

這項裁決適用於現行 dispatch。Dispatch 可以直接讀取各算法的資料、中間結果與數值工作估算。
它擁有適用條件的組合、成本比較、嘗試順序、guard 拒絕後的候選，以及 family 內的 builder 或 frame 選擇。
不為了解耦而新增 Protocol、registry 或通用 declaration 來遮住分類需要的資訊。

Kernel 擁有需要數值中間結果的 certificate，並向 dispatch 回報 acceptance 和可重用產品。
Dispatch 合併每個 family 的唯讀 `RowPlan`。
Scheduler 原樣傳遞 plan，以 `ApplyContext` 提供給 apply，不理解 plan 的數值內容。
Apply 消費已完成的 admission 與 frame／chain 決策，不依當次 apply 子集重新分類。
這些 plans 只活在一次 evolve，不進 model-only prepare LRU，也不留在 kernel state。

### 算法內聚，共用數值有單一 owner

使用者於 2026-10-18 提出整理方向：「之後我的規劃是讓你先整理目前模擬引擎，使其對外是黑盒子，對內是實質偶合，決策分類有全部算法的知識，算法部分內聚，以不影響優化與可讀性為主要指標，移除已經沒用的邏輯，將研究成果以及嘗試紀錄到repo文檔中供以後參考。然後就可以開始planner的重構，使各優化算法的實作張力能以自然的表達」。

一個算法的數值實作集中在自己的模組及專用 helpers。
多個算法共用的數值運算由一個模組擁有，例如 rates 到 Q／decay 的代數、period playback 與 fractional tail。
Kernel 不呼叫另一個 kernel class 的方法來取得共用運算。
排程共用接縫保留 `Kernel`、`Prepared`、`RowPlan` 與 `ApplyContext`，不把它們提升為對外擴充框架。
`PathDiagnostics` 仍按 family 名稱計數，builder 與 frame variant 是內部資訊。

跨 package 依賴方向沿用 [[0072]]，由 `.importlinter` 的 C8 分層、C9 無循環與
C12 穩定工具不依賴有狀態模組檢查。本決策不新增依賴方向，也不放寬 ignore_imports。
Package root 的對外入口規則由介面與 caller review 維持；上述 contracts 不檢查內部子模組的匯入路徑。

## 取捨

新增或改動算法時，dispatch 也可能需要一起修改。
這是刻意保留的耦合，分類可以使用完整資訊，不必維護另一套通用能力描述。
內部接縫只隔離排程與數值表示，不承諾每個 kernel 都能獨立替換而不動 dispatch。

Guard 拒絕可以換候選，prepare 或 apply 的數值失敗則向外傳遞。
事後 accuracy exceedance 只記錄並警告，不修正 density、不重算，也不重新分類。
未使用的實驗 planner 機制不保留相容層；未來 planner 的資料生命週期與重算消除需要另行設計。
本決策不宣稱所有中間結果都已消除重算。

## 現行契約與研究依據

- [Evolution README](../../lib/zcu_tools/simulate/evolution/README.md) 擁有現行 API、決策圖、數值 owner 與失敗語意。
- [模擬環境草稿](draft/simulation-environments.md) 記錄 environment、model、evolution 與 ADC 的組合責任。
- [Planner evidence 與固定成本討論](../discussions/2026-10-18-planner-evidence-and-overhead.md) 保留已移除 planner 的實驗結果，不作現行 API 契約。
