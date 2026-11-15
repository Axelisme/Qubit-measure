---
status: accepted
---

# 0078 — Evolution 黑盒邊界與算法內聚

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
Kernel、dispatch、scheduler 與 cache 維持內部實作。呼叫端經 `Engine.inspect_dispatch` 查詢選路，不直接呼叫內部 dispatcher。

2026-11-16，使用者核可公開 dispatch 查詢的實作計畫：「1核可」；對 Floquet guard 選「B」，保留一週期 propagator 計算。
`Engine.inspect_dispatch(source, batches, budget=...)` 與 evolve 共用 classifier 和 guard，不演化 density。
查詢逐 split row 回傳 family、guard 判定與相對排程工作量 `scheduling_cost`，或 deferred、short-circuit、needs-prefix 原因。
查詢以投影的完整 inventory 使用次數計價，不修改 `RoutingHistory` 的 usage、content membership 或 generation，也不註冊 routing reuse facts。
因此查詢的已決定路由等於相同初始 history 下 evolve 的路由，且不改變後續 evolve 的路由或 density。
Guard preparation 在查詢返回時釋放，不留在 Engine；source model cache 與 shift registration 仍可被填入。
呼叫端提供 numeric model source、timeline、初態與誤差預算，不逐段選算法。
引擎回傳 readout 起點的量子態與 family 診斷。
Environment 顯式組合 population 轉換與 ADC；readout 的樣本加總仍由 `simulate.adc` 擁有。

2026-11-12，使用者對具名歷史與整次 acquire 的 family 白名單裁定：「可以」。
量測層可持有 `RoutingHistory`，經環境 builder 的 `history` 借給 model source。未提供時由 source 自有。Source 關閉只清自己的歷史，借用歷史由 caller 管理。
`RoutingHistory` 擁有選路用的使用記錄。`clear()` 清空記錄並使先前的 preparation 失效，不重設模型快取、初態、RNG 或時鐘。歷史與 reuse 判定由 evolution 擁有。
Builder 的 `families` 是整次 acquire 的候選白名單，不是逐段指定算法。每段仍由 dispatch 選路並驗證。固定候選可以略過成本捷徑，不能略過輸入前置條件或數值 guard。無可用候選時回報 domain 與受影響的點、段。
這兩項控制不是物理 YAML、`AcquireRequest`、實驗 cfg 或 program binding 的欄位。不提供逐段 family 對照、callback、偏好順序或公開 kernel registry。詳細生命週期與失敗契約見 [evolution README](../../lib/zcu_tools/simulate/evolution/README.md#accuracy-failures-and-lifecycle)。

### 凍結數值輸入與參考解交接

`simulate.evolution` 公開通用數值 codec 與 solver-free reference cache I/O。
Codec 保存 batch、initial density、各 scheduled level 與 DC 的 `SecularModel`、使用中的 flux expansion 和 slow-envelope 設定。
讀回的是有限數值模型，不是 Python builder、裝置佈局或 mutable routing history。
這些數值輸入不等於演化引擎私有的 preparation cache，後者仍不提供第二套入口。
`FrozenRecord` 與 manifest 讀寫同樣歸 `simulate`，容器擁有 version、唯一 name、兩個 hash 與可重產 gzip。
Caller 的 JSON metadata 對 codec 不透明，場景 reason、families、routes、companions 與 provenance 都由 tools 解讀。
容器不放 solver 欄位。Tests 側入口由呼叫端明示 manifest-to-solver，只讀 name、physical，不讀 metadata 或場景名稱清單。

`physical_input`、`canonical_json` 與 `digest` 擁有 canonical physical hash。
完整 hash 包含 model identity，oracle-sharing hash 只排除這個 namespace。
`reference_key` 再加入 coarse/fine tolerances、version 1 與 convergence limit 2e-6。
沿用既有公式與 NPZ 的 `density`、`convergence_error` 欄位，不因搬移 owner 改 key，也不加入 solver identity。
`load_reference` 永遠 cache-only，miss 就失敗。
`write_reference` 只原子保存 producer 交來的數值，不求解或替它認證收斂。
獨立 solver 與 reference producer 留在 tests 側，經數值檔交接。失敗不換 solver。

場景 producer 與數值 codec 的分工見 [[0072]]。
取捨與歷史 cache 缺資料時的驗證方式見[凍結輸入討論](../discussions/2026-11-16-frozen-simulator-inputs.md)。

### 歷史控制與精度驗收

2026-11-12，使用者裁定：「B，只要為了優化速度可以損失一點一致性，只要誤差在容許範圍」，並要求「應該已正確的物理演化為誤差參考，而不是當前實作，否則誤差會累計」。
Dispatch 可以依量測歷史選擇不同 kernel，量測層用公開參數控制空歷史、沿用歷史與允許的 family，不以 monkeypatch 操作內部狀態。
精度驗收以獨立的完整 Hamiltonian 參考解為準，不以當前或重構前輸出作真值，也不要求重構前後逐值一致。Density 最大絕對差與 population 等效 expected I/Q 誤差的目標均為 `2e-3`。
可用的候選須逐一對參考解驗證。現行有四個 guard 邊緣案例超標，以 strict xfail 保留，不放寬誤差門檻。案例與限制見 [evolution README 的 kernel 表](../../lib/zcu_tools/simulate/evolution/README.md#kernel-summary)，環境 expected I/Q 驗收尚未交付。
算法數值集中在自己的 owner，新增算法或優化不要求修改不相關模組。實際擴充位置見 [evolution README 的 Adding a family](../../lib/zcu_tools/simulate/evolution/README.md#adding-a-family)。

### Family 去留：合格只比耗時

以每點跑完整條 pulse sequence 後，最終 density 對獨立參考解的最大絕對差判定精度。
最終誤差 ≤ `2e-3` 的 family 視為一樣準，彼此只比耗時。
另一個合格的 family 快超過 10% 才算支配它。
某種輸入只有它能算的 family 不參加比較，包含獨占一個適用域的 family，以及各適用域的最後一道候選。

當時 Gaussian gate 序列的 S 表中，`shaped_sambe` 的誤差與耗時為 `1.45e-4`、0.27 秒，
`period_table` 為 `2.96e-5`、9.0 秒，`gate_cache` 為 `1.10e-5`、54 秒。
三者都合格，後兩者被 `shaped_sambe` 支配，即使它們準一個數量級。
這組數字是 2026-11-13 討論時的範例，不替代後續量表的實測記錄。

適用邊界：

- 單一 kernel 的誤差或每段 `budget` 不算合格證據，只有量到的完整序列最終誤差算。
- 要因此刪掉較準但較慢的 family，須先有支配它的 family 在同型態長序列上仍 ≤ `2e-3` 的實測，gate 數須遠多於 24。沒有這份證據就不刪。
- 刪除須由使用者看過數據後點頭。

2026-11-13，使用者說：「我通常是要求在给定最终精度容许范围内尽可能加速，所以通常这种精度差距极小的，我会优先选速度。」
使用者並澄清：「你要注意哦，这个精度是最终精度，也就是整个 post sequence 模拟最后的精度，而不是单个 kernel 的精度哦。」
對整理後的規則答「確認」。
出處與選項比較見 [family 支配定義的討論](../discussions/2026-11-13-family-dominance-within-tolerance.md)。
量表判定機制見 [tools README](../../tools/README.md#engine-family-pareto-table)。

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

### t6 只保留 direct

`t6` 保留 constant 單 tone 的適用域、guard、prepare-key 身分與失敗就 raise 的語意。
每列 density 都經共享的 `dop853.advance_endpoints` 在 H0 旋轉座標積分，
使用完整 secular bath 與同一套 budget-scaled 局部容差。
移除 reusable channel 的建置、Fourier playback、價格比較與完整 acquire key inventory。
其他 family 的 period channel 不受這項決定影響。

2026-11-15，使用者對「保留並補價格」與「移除 channel、保留 direct」兩案答「2B」，
選擇後者。Channel 算得準，但四個代表性場景與 benchmark 都沒有選到它。
真實表格的相位幾乎每列不同，同一 key 要累積約 1.5 萬個載波週期才開始划算。
因此不為目前未使用的高重用情境保留這條路徑。
若日後出現這類工作負載，再依新證據評估，不保留相容性分支。
比較、先例與被否決選項見 [移除 t6 channel 的討論](../discussions/2026-11-15-t6-channel-removal.md)。

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

2026-10-28，使用者以「A我認為可以完整版。」授權 single、six、FTF 三環境的中途 projective readout。這句授權決定支援範圍，lifetime 沿用本節既有規則。
分支 continuation density 僅在一次 preparation 的 span 演化中存在，不跨 acquisition 保留或作下次初態。
環境在 projective read 起點投影並續演化，Engine 仍不保留跨 evolve 的數值 preparation。
同輸入的 immutable 完整 preparation 是否重用，仍依本節既有設定、losses 與 route 規則。
其中的 outcome weights、各 event populations 與 conditional sums 是可重用的數值產品，不是上一個 shot、round 或 acquisition 的量子態或已抽樣 herald 結果。
已交付 raw source 保留自己的不可變抽樣輸入，不因此承接前次量子態。

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
