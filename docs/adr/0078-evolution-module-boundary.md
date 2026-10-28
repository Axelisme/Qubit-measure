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
這些 plans 只活在一次 evolve，也不留在 kernel state。

### 算法內聚，共用數值有單一 owner

使用者於 2026-10-18 提出整理方向：「之後我的規劃是讓你先整理目前模擬引擎，使其對外是黑盒子，對內是實質偶合，決策分類有全部算法的知識，算法部分內聚，以不影響優化與可讀性為主要指標，移除已經沒用的邏輯，將研究成果以及嘗試紀錄到repo文檔中供以後參考。然後就可以開始planner的重構，使各優化算法的實作張力能以自然的表達」。

一個算法的數值實作集中在自己的模組及專用 helpers。
多個算法共用的數值運算由一個模組擁有，例如 rates 到 Q／decay 的代數、period playback 與 fractional tail。
Kernel 不呼叫另一個 kernel class 的方法來取得共用運算。

使用者於 2026-10-21 裁決：「前兩個採用建議」。腔場 flat second moments 的
exact-result memo 由 `simulate.adc.CavityAverageCache` 擁有，集中完整 lossless key、
有界容量與回傳陣列 ownership。`experiment.ftf` 在一次 acquire 的 span composition
建立至多 64 MiB owner，供 g/e 各點串行共用，並在 return 或 exception 時清理。
`simulate.ftf.CavityDrive` 只轉交 owner，不組 ADC key。Owner 不留在 Engine、global
或長駐 source；這讓重複 flat 積分可重用，又不延長數值產品的 acquire lifetime。
Shaped/callable 路徑、末態場與主分支選擇不進 memo。
排程共用接縫保留 `Kernel`、`Prepared`、`RowPlan` 與 `ApplyContext`，不把它們提升為對外擴充框架。
`PathDiagnostics` 仍按 family 名稱計數，builder 與 frame variant 是內部資訊。

跨 package 依賴方向沿用 [[0072]]，由 `.importlinter` 的 C8 分層、C9 無循環與
C12 穩定工具不依賴有狀態模組檢查。本決策不新增依賴方向，也不放寬 ignore_imports。
Package root 的對外入口規則由介面與 caller review 維持；上述 contracts 不檢查內部子模組的匯入路徑。

### 損耗漂移下的結構與 preparation lifetime

2026-10-27（UTC），advisor 的 before-accepted ruling：
「保留 single 的 _models／_prepared 與 FTF acquire-local 的 CavityAverageCache；拆 _facts 為結構與 rates。」

Single、six 與 FTF 三個模擬環境都納入損耗漂移。每次 acquisition 開始取一次有效損耗，整次 acquisition 固定；漂移預設關閉。
快取層的去留以同一次 acquisition 內的實際命中判斷；只在 acquisition 之間命中不構成保留理由。
使用者於 2026-10-27 裁決：「我預期都會加入漂移機制」、「沒有不漂移的時候，我考慮C，應該是說，用不到的就可以移除」。

Single CPMG、six 與 FTF 的代表性 acquisition 在指定三層都沒有同次命中。
這不是所有 workload 的普遍命中率宣稱。刪除 Engine 的 `prepare_cache`、
`result_memo` 與 source 的 `readout_cache`。Engine 不保留跨 evolve 的 prepare
payload 或 exact result，source 不保留跨 preparation 的 ADC solver。
未漂移的 FTF 重複 acquisition 若需要重算，接受這個代價。

Single 的完整 ModelSource models 與 caller-owned preparation 仍可重用。
`simulate.fluxonium.SingleFluxoniumModel` 擁有損耗無關的 Hamiltonian 投影、
eigensystem 與 dressed readout frequencies。Experiment 只組合 owner 的結果。
損耗改變時，source 保留結構，關閉舊 ModelSource 並清除完整 preparation，
再依本次 acquisition 的固定 losses 建立 rates。溫度、DC、truncation 或其他
結構依賴改變時，source 丟棄結構。Dissipative propagator、density 與完整
preparation 不得跨不同 losses 重用。

Engine 的 routing-facts cache 只驗證 caller-owned preparation 的當前 route。
它保留 inventory 與 pricing facts，不保留數值傳播結果。
FTF 的 complete preparation LRU、envelope fingerprints 與上述 acquire-local
`CavityAverageCache` 不屬於已刪三層。抽樣 RNG、raw 與 replay 的契約不變。
依賴方向仍由 C8、C9、C12、C19 保護，沒有新增 package edge 或例外。

FTF 與 single 不同，不拆損耗無關結構。漂移開啟時，每次 acquisition 用 effective settings 重建完整 `FTFModelFacts`，包含 eigensystem，並清除舊的 complete preparations；
整次 points／rounds 共用這次的 losses，不跨不同 losses 重用 rates 或完整 preparation。漂移關閉的既有行為不變。
這是 advisor 於 2026-10-27 依使用者授權「B你可以自己開task，不用我看過」訂定的重建策略。數值、設定與 truth 的 owner 分工見 [[0072]]。

### Integrated raw 由環境生成，介面由 program 擁有

使用者於 2026-10-20 裁決：「1-1」「基本認同丙，但我質疑有需要進core嗎？」。
介面改放 program 後，使用者答「同意」。

`program.v2.simulation.RawShots` 定義延遲產生完整單發 ADC sums 的介面。
環境以 `IntegratedRound` 一起交付純統計資料和 raw source。
Program 只在單發 processing 或 getter 需要資料時呼叫 source，不重跑 backend。
`core.simulation.RoundStatistics` 只有資料，不定義 materializer、Protocol 或 callback。
Core 的 no-callback 邊界不變。

ADC 和 white_noise 各自生成完整 raw，包含物理訊號、receiver noise、nearest-even int64 轉換和排列。
兩者以結構型別符合 RawShots，不 import program。
`experiment.simulation` 和 `experiment.ftf` 是現有組合根。
`simulate.rounding` 擁有兩種環境和 decimated transport 共用的數值取整及 shot-axis 排列。
這些函式處理數值 ADC sums，不負責 QICK averaging 或硬體操作。
依賴方向由 C8、C9、C12 和 C19 保護，不增加 ignore_imports。

單 carrier raw 直接抽 CTMC waiting times 和 directional transitions。
零跳直接取 noiseless sums，有跳以 compiled exact 腔記憶傳播計算離散 sample sums。
所有多跳都保留，不設跳數上限，不建 shot LUT，也不改成先抽跳數的 sampler。
已知 flat 輸入用閉式傳播，其他 envelope 保留 noiseless baseline 的 solver tolerances。
每次跳躍沿用原有腔場，不重設 vacuum。
直接平均維持 conditional moments 和既有 RNG stream，不做逐 shot 工作。
Raw 和直接平均只要求期望一致，不平移 raw 以匹配有限 shots 的平均 realization。
Caller 保留的完整 preparation 只存 noiseless inputs，不存 noisy shots 或 round identity。

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
