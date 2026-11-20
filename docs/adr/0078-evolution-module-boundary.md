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
查詢不記錄 history，因為記錄會讓觀察介面改變之後其他輸入的選路，呼叫端還得記得這個副作用。`scheduling_cost` 是 chunk interleaving 的相對工作量，不是秒，也不是排序用的冷攤提估價。2026-11-16，使用者對「查詢不改 routing history」與「欄位只回報既有排程工作量」兩項答「追認」。
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

白名單或 pin 不略過結構檢查。`multitone_gate` 沒有建構後的精度檢查，這是它本身的規則，不是 pin 略過 guard。

### `Engine` 的 GPU backend

2026-11-18，使用者對讓 `dop853` 提供 OpenCL 執行後端的方案答「A」。平台範圍與無可用 GPU 的行為遵循[〈跨平台，不綁作業系統〉](#跨平台不綁作業系統)與[〈GPU 是可選功能〉](#gpu-是可選功能)。

`Engine(backend="auto"|"cpu"|"opencl")` 選的是 `dop853` 的執行裝置，不增加 family，也不讓呼叫端逐段選算法。預設 `"auto"` 在第一個合格 row 才偵測裝置，按成本選 CPU 或 GPU。測試、配對量測與逐位比對可用 `"cpu"` 固定 CPU。GPU 執行中出錯不靜默退回 CPU。安裝、適用範圍及失敗契約見 [GPU backend](../../lib/zcu_tools/simulate/evolution/README.md#gpu-backend)。

### 凍結數值輸入與參考解交接

`simulate.evolution` 公開通用數值 codec 與 solver-free reference cache I/O。
Codec 保存 batch、initial density、各 scheduled level 與 DC 的 `SecularModel`、使用中的 flux expansion 和 slow-envelope 設定。
讀回的是有限數值模型，不是 Python builder、裝置佈局或 mutable routing history。
這些數值輸入不等於演化引擎私有的 preparation cache，後者仍不提供第二套入口。
`FrozenRecord` 與 manifest 讀寫同樣歸 `simulate`，容器擁有 version、唯一 name、兩個 hash 與可重產 gzip。
Caller 的 JSON metadata 對 codec 不透明，場景 reason、families、routes、companions 與 provenance 都由 tools 解讀。
容器不放 solver 欄位。Tests 側入口由呼叫端明示 manifest-to-solver，只讀 name、physical，不讀 metadata 或場景名稱清單。

2026-11-16，使用者以「1核可」核可分層實作計畫，凍結數值格式與 key 公式因此歸 `simulate`，參考解 solver 留在 tests 側。凍結數值不入庫，存放與讀寫分工見 [[0072]] 的開發工具段；`simulate` 只收顯式路徑，不解析 Git 位置。

`FROZEN_PHYSICAL_VERSION` 擁有 manifest 與 standalone input 的 physical 解碼契約。
欄位、單位、dtype、預設或還原 Engine 輸入的方式改變時 bump，由 producer 處理遷移。
`SCENE_CACHE_VERSION` 擁有 index JSON 與 cache 目錄版次，tools 由它推導路徑。
兩者目前都是 1。`SEGMENT_NUMERIC_VERSION` 只記演算法數值語義，不是輸入相容性 gate。
成對工具比較兩個格式版本與每筆完整 record 身份，不因數值優化的版本變更而拒絕候選。
跨 process 計時規則歸 [[0072]]，不進 codec 或 Engine。

`physical_input`、`canonical_json` 與 `digest` 擁有 canonical physical hash。
完整 hash 包含 model identity，oracle-sharing hash 只排除這個 namespace。
`reference_key` 再加入 coarse/fine tolerances、version 1 與 convergence limit 2e-6。
沿用既有公式與 NPZ 的 `density`、`convergence_error` 欄位，不因搬移 owner 改 key，也不加入 solver identity。
`load_reference` 永遠 cache-only，miss 就失敗。
`write_reference` 只原子保存 producer 交來的數值，不求解或替它認證收斂。
獨立 solver 與 reference producer 留在 tests 側，經數值檔交接。失敗不換 solver。
Tests reference producer 的 `--scenario NAME` 要求在呼叫端明示的清單中恰好一列同名，選取後才讀 payload 或呼叫 solver。
零筆列合法名稱，多筆列來源並失敗。其他列不讀不解，不帶 scenario 時仍整批準備。
Solver 由清單與 solver 的顯式對應決定，不讀 tools metadata，也不 import tools。
單列準備的取捨見[共用清單與 Pareto 選集討論](../discussions/2026-11-17-pareto-selection-and-scene-append.md)。

場景 producer 與數值 codec 的分工見 [[0072]]。
取捨與歷史 cache 缺資料時的驗證方式見[凍結輸入討論](../discussions/2026-11-16-frozen-simulator-inputs.md)。

### 模擬器的測試與量測分層，不做 e2e

模擬器的測試與 kernel／family 比較只依賴模擬器，按層分開驗證：
- Kernel：一段輸入，`Engine.evolve` 以 `families=` 固定候選，對獨立參考解量誤差與耗時。
- Dispatch：rows → 選中的 family、guard 判定與排程工作量，不演化 density。Guard 走與 evolve 相同的路徑，包括它自己需要的 ODE，所以測試用小模型。入口是 `Engine.inspect_dispatch`。
- 序列：手寫多段 `TimelineBatch`，在 `Engine.evolve` 上量最終 density 誤差與總耗時。「合格只比耗時」用的最終誤差、整體沒變慢與 family 去留都在這一層量。序列場景照約化軸（gate 種類 × 個數）宣告，不逐實驗加。
- 前端：program → 編譯後的訊號，不演化。
- Environment：給定 density，驗讀出。

不做 program → 編譯 → 演化 → I/Q 一次跑完的 e2e 測試。演化精度仍在序列層驗，kernel／family 比較的計時只包 `Engine.evolve`。
例如，成對計時在序列層比兩個 checkout 的 `Engine.evolve`，不經 acquire。FTF environment 測試用不演化的 Engine 替身驗讀出交接。

整實驗牆鐘只有 [`tools.experiment_wall_time`](../../tools/README.md#實驗-run-牆鐘) 這支工具可以入庫執行完整實驗。它用固定模擬輸入，在 `time_scale=0` 下記錄正常參數的 Run 耗時與分項。它只計送出 Run 到終態，不呼叫 Analyze 或 Save。它不作測試、gate 或精度證據。2026-11-20，使用者問：「為什麼要用不入庫的 scratch 腳本量?而不是入庫的？」2026-11-21 對以整個 Run 牆鐘作優化指標、工具入庫但不當 gate 的建議答「都採用建議」。取捨見[實驗 Run 牆鐘工具](../discussions/2026-11-21-experiment-run-wall-time.md)。

除錯例外不變。使用者回報的錯在各層都重現不出來時，可以跑一次 e2e 定位。腳本不入庫，也不當驗收門檻。

2026-11-15，使用者說：「我會預期模擬器測試腳本只會依賴模擬器模塊，後端與前端分開測，kernel與dispatch也分開測，不該有e2e的測試需求」。對「整條序列的最終誤差放哪一層」答「A」，即加序列層，不只留 kernel 與 dispatch，也不保留 acquire e2e 只記耗時；對除錯邊界答「我選B」，即除錯時可以跑 e2e，腳本不入庫、不當門檻。
選項比較見[模擬器測試與量測的分層](../discussions/2026-11-15-simulator-test-measurement-layering.md)。

### 每個 kernel 的量測場景只有一段

優化或校準某個 kernel 時，量測場景是一段由該 kernel 處理的輸入。
測重複就是同一段接 N 次；一次只改一個約化軸；誤差與耗時在同一組座標上量。
多段工作負載放在序列層，用來確認整體沒變慢、選路正確與 family 去留。

例：`dop853` 的多 tone 取樣場景是兩個 tone 同起同止、各帶一條包絡的一段，以 `families=("dop853",)` 固定，重複 1、2、20、100 次。
不用「兩個 Gaussian 錯開 8 ns」這種會被切成好幾段的寫法。
邊界：要量的就是跨段的東西時（使用者等待的總時間、選路、段間銜接），用序列層手寫的多段 `TimelineBatch`。

2026-11-15，使用者說：「每一个 kernel 的场景，应该都应该只有一段。如果你想要测试重复的，那就是好几个重复段，不会出现两个高选然后交错在一起，然后实际上有五段这样子，这这没什么用。」（「高选」讀作「高斯」）
這條原則出自 STIRAP 多段 workload 的量測檢討，原始討論未入庫；分層本身見上一段。

### 場景覆蓋靠約化的參數空間

模擬器的測試與驗收不為個別實驗加專屬場景。
先把輸入約化成 kernel 與 guard 實際看到的物理量：每個 tone 的耦合（gain × 該 flux 下的矩陣元）、到最近躍遷的失諧、時長、包絡形狀與 tone 間距。
在這個空間上事先宣告格點，系統地覆蓋；每個實驗只核它的座標落在已覆蓋的區域。

例：兩個常數 tone 的輸入（AC Stark、FTF mux 等）用三條軸的一張格點覆蓋，不各量一組：共振轉角（耦合 × 時長）、離共振比（耦合 ÷ 失諧）、累積 Stark 相位（耦合² × 時長 ÷ 失諧）。
序列層的 RB 型場景照 gate 種類 × 個數宣告，並用隨機相位，因為固定的 X、Y、−X、−Y 循環會抵消部分 RWA 誤差（[RB 型相位序列作為固定量表輸入](../discussions/2026-11-14-random-gate-sequence-workloads.md)）。
邊界：約化丟掉的物理會改變結果時（例如讀出投影、連續 flux、量子 resonator），把它加成約化空間的新一軸，不加實驗場景。

2026-11-14，使用者主動提出：「实验场景非常多，要完整覆盖是相当困难的，无脑扩充测试场景只会拖慢开发速度，你应该约化不同场景，系统性的覆盖可能情况」。對整理後的原則答：「符合，我預期不會加專屬實驗場景」。
原始討論（`multitone_rwa` 的量測範圍）未入庫。

### 歷史控制與精度驗收

2026-11-12，使用者裁定：「B，只要為了優化速度可以損失一點一致性，只要誤差在容許範圍」，並要求「應該已正確的物理演化為誤差參考，而不是當前實作，否則誤差會累計」。
Dispatch 可以依量測歷史選擇不同 kernel，量測層用公開參數控制空歷史、沿用歷史與允許的 family，不以 monkeypatch 操作內部狀態。
精度驗收以獨立的完整 Hamiltonian 參考解為準，不以當前或重構前輸出作真值，也不要求重構前後逐值一致。Density 最大絕對差與 population 等效 expected I/Q 誤差的目標均為 `2e-3`。
可用的候選須逐一對參考解驗證。現行有兩個 guard 邊緣案例超標，以 strict xfail 保留，不放寬誤差門檻。案例與限制見 [evolution README 的 kernel 表](../../lib/zcu_tools/simulate/evolution/README.md#kernel-summary)，環境 expected I/Q 驗收尚未交付。
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

### 先保證正確，再優化瓶頸

Guard 在整個參數空間不放行最終誤差超過 `2e-3` 的結果。
速度優化只針對瓶頸，也就是實際工作負載中耗時最多、使用者在等的場景。
瓶頸從已有的耗時紀錄找，例如 kernel 表的實測證據、量表摘要與討論紀錄；找不到紀錄才問使用者。
不要求快速算法涵蓋它能算對的全部區域，安全但較窄的 guard 可以接受。

例：`t6` 的 direct 做法從四個有耗時紀錄的慢場景下手，掃長度的 first-evolve wall 從 736 s 降到 6.3 s，四個場景的最終誤差都 ≤ `2e-3`（[t6 的 direct 加速策略](../discussions/2026-11-14-t6-direct-strategy.md)）。
邊界：修法已在做、很快落地時，過渡期維持現狀；修法還要研究、時程未定時，先收窄 admission，改走慢但對的候選，修好再放回。
例：sampled multitone 在修法研究期間暫停 admission，由其他候選計算（[Sampled multitone 的過渡期拒收](../discussions/2026-11-14-sampled-multitone-interim.md)）。

2026-11-14，使用者問「模拟器不应该是覆盖所有可能范围吗？」，並說：「我预期就是首先先确保能全覆盖，然后再来是对于瓶颈场景做优化。」對過渡期的邊界選項答「A」。
這條原則出自 `multitone_rwa` 適用範圍的討論，原始討論未入庫。

例外：`multitone_gate` 由比價選中後就建構、就使用，不在建構後另做精度檢查，也不因精度退回 `dop853`；數值失敗照常直接回報。其他 family 的 guard、sampled `multitone_rwa` 的抽查與動態維度的 pilot 驗算不適用這條例外。

2026-11-19，使用者說：「我应该说过，不要因为听度而回退。比价的时候决定好就不要再改了，就算结果超标也是能接受。我不想白付时间去尝试错误的路径」（「听度」即精度），並把範圍限定在 `multitone_gate`。

### Evolution 輸出合法 density

`Engine.evolve` 回傳的每個 readout density 都滿足 continuation 輸入的條件：Hermitian、trace 為 1、最小特徵值 ≥ −1e−8。
積分器的輸出不合法時，engine 取它的 Hermitian 部分，把負特徵值截為 0，再重新正規化。
投影前後的最大元素絕對差不超過該次呼叫的 budget 才採用，否則以 RuntimeError 指出 point。
Continuation 檢查的容差與各 kernel 的選路都不變。

例：FTF heralded 把 density 投影成純態，接著一段 50 ns 的高 gain 雙 tone 段走 CPU DP5(4)，輸出的最小特徵值是 −2.4e−6。這個誤差符合 `2e-3` 的精度目標，但下一段的 continuation 檢查拒收。改為投影後，兩個重現案例都通過，修正耗時約占 evolve 總時間的 0.05%–0.09%。
邊界：修正量超過 budget 表示積分器本身不準，照數值失敗回報，不放行。

2026-11-19，使用者看過診斷與四個選項後選 A，回覆「其余采用建议」。討論見 [Evolution 輸出合法 density](../discussions/2026-11-19-evolution-valid-density.md)。

### 固定預算，找適合的場景

評估模擬算法時，精度預算固定為使用者對該場景宣告的需求，預設是完整 density 對參考解 ≤ `2e-3`。
不為了讓某個算法過關而放寬預算，而是在固定預算下找出它划算的場景（控制形狀、掃描點數、時長、初態），寫清適用範圍；不適用的場景由 guard 或選路退回。

例：能量最低 24 態截斷在單 tone FTF 掃描誤差 ≤ `1e-3`、每點快約 2 倍，是適合的場景；兩個 qubit 同時打 X90 的 depth 500 序列每個 gate 只分到極小額度，截斷不划算，結論是場景不適合，不是放寬 RB 的預算（[FTF 模擬的子空間約化](../discussions/2026-11-18-ftf-subspace-reduction.md)）。
邊界：使用者自己按場景宣告不同預算是改需求，不是替算法放寬。例如長序列的預算按 gate 數放寬為 max(`2e-3`, `3.9e-6` × gate 數)，depth 500 約 `1e-2`。

2026-11-18，使用者說：「我認為不同場景的預算分配確實不同，但模擬算法的衡量不應該改變預算，而是固定預算並找適合的場景」，並說：「還有對於depth 500這種超長序列，我能接受更多的誤差。」對長序列預算的選項答「A」。
這條原則出自子空間約化在單 tone 與雙 qubit 序列上的比較。

### 跨平台，不綁作業系統

模擬器的算法與執行機制在 Windows 與 Linux 上都成立，不依賴單一作業系統的設定或行為。

例：GPU dispatch 不按 Linux xe driver 的 `preempt_timeout_us=640000` 設定長度。程式量每批耗時，把每次送出控制在約 0.1 s。這個做法不需要修改 Linux driver 或 Windows 的 TDR 設定（預設 2 s）。

邊界：某台機器沒有可用的 GPU 驅動時，可以改用 CPU。結果仍正確，只是較慢。macOS 不在支援範圍。

2026-11-18，使用者說：「由于我会跨电脑平台使用，所以不要采用局限于特定 OS 系统的算法」，並確認：「只有 Windows 和 Linux，其餘採用建議」。
這條原則出自 [`dop853` 的 OpenCL GPU backend](../discussions/2026-11-18-ftf-gpu-dop853-backend.md)。

### GPU 是可選功能

沒裝 GPU 套件或沒有可用 GPU 時，模擬器的行為與結果與沒有 GPU 功能時完全相同。

例：Windows 筆電沒裝 `gpu` extra，執行同一份 FTF 腳本仍走 CPU，結果與加入 GPU 功能前逐位相同。

邊界：只有使用者明示要求 GPU（`backend="opencl"`）時，找不到可用 GPU 才報錯。

2026-11-18，使用者說：「也有可能在没有 GPU 的系统上使用，所以 GPU 功能确实是属于可选开启」，並說：「OpenCL 应该是没问题的」。
這條原則出自 [`dop853` 的 OpenCL GPU backend](../discussions/2026-11-18-ftf-gpu-dop853-backend.md)。

### 快速算法先修好，不輕易放棄

快速算法有誤差問題時，先查原因並試著修正，不先刪除或整個收窄。
修不好的部分才收 guard，瓶頸範圍留在快速算法（〈先保證正確，再優化瓶頸〉）。
提出刪除或收窄之前，先寫清誤差或變慢的來源，以及能否修正。

例：`floquet` 最初只量到拒絕或單點較慢的例子。使用者選擇再研究一輪並設停損條件，結果在 committed 批次上比 `dop853` 快 1.65–1.81 倍，因此保留並重校價格（[Floquet 的保留與價格重校](../discussions/2026-11-16-floquet-kept-price.md)）。
邊界：這條處理算不準的算法。已經算得準、卻沒有服務任何現有瓶頸的做法可以移除，例如 `t6` 的 channel 做法（[t6 只保留 direct](../discussions/2026-11-15-t6-channel-removal.md)）。
找不到任何能算對的範圍時才收窄或刪除；刪除 family 仍照〈Family 去留〉由使用者看過數據點頭。

2026-11-14，使用者說：「我不太想放弃快速的算法，目前的优化需求还蛮大的」。
這條原則出自 `single_tone_rwa` 去留與 `multitone_rwa` 超標兩題，原始討論未入庫。

### 支援範圍不看使用頻率

模擬器要支援、要算對、價格要合理的輸入範圍，由物理與介面決定，並在約化參數空間上覆蓋（〈場景覆蓋靠約化的參數空間〉）。
使用者常不常用不決定這個範圍。

例：價格的 prepare 攤提把重用次數當輸入軸覆蓋，同一段接 N = 1、20、100，不問同一段常不常重複。
Sampled multitone 過渡期的速度代價照樣用 benchmark 判斷，不以「這類序列較少使用」取代（[Sampled multitone 的過渡期拒收](../discussions/2026-11-14-sampled-multitone-interim.md)）。
邊界：使用頻率只在已有耗時紀錄時用來排優化的先後，不決定支不支援、算不算對。
例：`t6` 移除只在高重用才快的 channel 做法後，那類輸入照樣算對，只是變慢（[t6 只保留 direct](../discussions/2026-11-15-t6-channel-removal.md)）。

2026-11-16，使用者問「你為什麼老是問我常不常用？我常用與你模擬器要不要支援是兩回事」，對邊界答「A」。
原始討論（`multitone_rwa` 的校價需求題）未入庫。

### 模擬器不替使用者擋實驗參數

實驗參數設得不好時，模擬器照常給結果。
它算不準的部分以警告說明是模擬器的限制，不因參數不當而報錯。

例：projective read 改成 Markov chain 時，51 點、21 能階是合法參數，沒有被記憶體預算擋下（[Projective read 的輸出改成 Markov chain](../discussions/2026-11-17-projection-markov-chain-readout.md)）。
例：FTF 的 cavity drive 使相鄰態能量順序反轉時，模擬照常用原色散模型計算，不重排狀態。一次 acquisition 對每組反轉的態發一次警告，列出首個受影響點及段、總點數與段數，並在固定 gain 比例與觀察到的腔場歷程下，給出 cavity drive gain 的共同縮放上限。超過上限的結果不在色散模型有效範圍內。見 [FTF cavity drives](../../lib/zcu_tools/simulators/ftf/environment/README.md#cavity-drives)。2026-11-19，使用者在「改成警告」與「保留報錯、補上限」兩案中選前者：「1cA」；比較見 [FTF cavity Stark 反轉：從報錯改成警告](../discussions/2026-11-20-ftf-stark-inversion-warning.md)。
邊界：契約外的輸入（型別、範圍、維度錯）照舊在進入點 fast fail。
數值近似的 guard 照〈先保證正確，再優化瓶頸〉不放行超標結果，因為那是算法算錯，不是參數造成的。

2026-11-15，使用者覆核 sideband reset 的殘餘光子檢查時說：「4我覺得有需要嗎？實驗參數設置不當就會發生的問題為什麼模擬器要擋？」對選項答「A」，即算不準也照常給結果並警告。
原始討論未入庫。

### 模擬器不從程式內容推斷實驗類型或硬體架構

規則：選路、精度與維度不依「程式內容被辨認成哪種實驗或架構」切換。需要的精度由使用者宣告，沒宣告就用預設 budget。直接比較計算結果或讀數值指標的判準，依據是數值，可以決定選路與維度，例如動態維度的 pilot 與 `drop(k)`、sampled `multitone_rwa` 的抽查、各 family 的 guard 指標。

例：有 flux 脈衝的程式照樣用預設 budget；CZ 也可能從 charge line 打。

邊界：使用者宣告的精度需求與數值指標都可以改變選路；實驗名稱、脈衝種類與架構不行。

2026-11-18，使用者說：「这不合理，CZ gate 可能也是从 charge line 打，而且模拟器理应不知道现在的架构是 FTF 还是只有单个 F 还是什么的。」這條原則出自[FTF 動態維度](../discussions/2026-11-18-ftf-dynamic-dimension.md)。

### 模擬器帶通用調試介面

每個模擬器提供同一組調試介面。定位與驗證工具只依賴這組介面，不看是哪個模擬器，也不靠模擬器另外實作功能。
共通的部分有型別；各模擬器值得記錄的額外資訊放在一個 dict 屬性裡，不要求強型別。

例：FTF、single、six 每次 acquire 用同一個 writer 寫一行分層耗時（總 wall、編譯、演化、讀出與各 family 聚合），模擬器特有的值放在 `extras`。
設定 `numerics.debug_capture: true` 時，三者用同一個 `CaptureBuffer` 錄下每次 `Engine.evolve` 的凍結輸入與實際選路，`tools/inspect_simulator_capture.py` 對三者用法相同（[共用 acquire telemetry writer](../discussions/2026-11-16-shared-acquire-telemetry.md)、[Engine capture 的逐輪選路事件](../discussions/2026-11-17-engine-capture-events.md)）。
邊界：介面只涵蓋定位與驗證共通需要的資料，模擬器特有的診斷由各自決定放不放進 dict。
共通部分可以帶中性的模擬知識；場景特化的邏輯（例如 FTF 的腔記憶、six 的 scatter）不變成共同流程上的 hook。
調試者就是模擬器維護者，介面可以暴露內部資訊，不另外為外部使用者設計穩定 API。
直接用 `ModelSource` 呼叫 `Engine.evolve` 的使用方有錄製與體檢，沒有分層耗時，因為沒有前端與讀出。

2026-11-16，使用者在優化流程的討論中說：「優化手段要看模擬器是否實作功能，應該是要規範模擬器應該帶有用於調試的通用接口，類似基類或protocol的概念」。
對資料形狀答：「可以，但我認為這個資料不一定要完全強型別，可以用多餘資訊的接口，比如dict屬性，放值得記錄的資訊」。
對誰使用這組介面答：「可以，調試者應該就是模擬器維護者，我沒有意向去解耦合兩者責任」。
對共通部分能帶什麼答：「中性的模擬知識是沒有問題的。場景特化的才不建議」。
原始討論未入庫。

### 優化優先於周邊約定；除錯靠 dump，不靠重算

規則：模擬器優化的做法、門檻與效能，以使用者看得到的結果正確性為準，也就是完整 density 在誤差預算內。存檔格式、版本號、相容性與重現流程都配合優化調整。除錯需要的資訊寫進 debug capture manifest，缺資訊就擴充 manifest。

例：數值 kernel 改動後，完整 density 通過精度預算就能落地。除錯時讀 manifest 裡的 Engine 輸入、逐列 routes、seed、每輪 DC、model identity 與有效設定。

邊界：使用者看得到的結果正確性、GUI 與設定檔的使用介面、量測結果的語意維持原樣。它們與優化衝突時，由使用者裁決。

2026-11-18，使用者說：「关于 replay 啊之类的其他成本，拜托你以优化的效率为优先，而不是这些其他的细枝末节的仪式、一些储存格式。你如果迁就它的话，优化很难做啊。」並說：「你需要的资讯应该本身就在 dump 的结果里了。如果你会需要 replace 多余的资讯、多余的算法，那就是你 dump 的资料不够多嘛。」對原則行答「好」。
這條原則出自[移除 private replay](../discussions/2026-11-18-remove-private-replay.md)。

### 用快場景的小幅變慢，換慢場景的大幅加速

規則：預設選路可以讓快的場景多付一筆有上限的固定成本，例如抽查、估價、pilot 驗算，換取慢的場景大幅變快。這類先算一點再決定的步驟最後沒被採用時，端到端時間超過 main 的 1.2 倍就回報使用者。成本不必先估價才付。

例：sampled 雙 tone 列的抽查沒過時，場景比 main 慢 1.168 倍；抽查通過時，24 點掃描比 main 快 3.149 倍。

邊界：多付的成本有上限，不隨工作量變大；超過 1.2 倍時由使用者決定。

2026-11-18，使用者說：「以比较快的场景稍微变慢一点为代价，让慢场景快很多是可以接受的」。這條原則出自[sampled multitone_rwa 的實測放行](../discussions/2026-11-18-sampled-multitone-rwa-admission.md)。

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
其中可重用的數值產品是 FTF 的 outcome weights 與各 event populations、single／six 的 `ProjectionChain` 初始分佈與轉移表，以及兩者的 conditional sums。
它們不是上一個 shot、round 或 acquisition 的量子態或已抽樣 herald 結果。
已交付 raw source 保留自己的不可變抽樣輸入，不因此承接前次量子態。

Engine 的 routing-facts cache 只驗證 caller-owned preparation 的當前 route。
它保留 inventory 與 pricing facts，不保留數值傳播結果。
FTF 的 complete preparation LRU、envelope fingerprints 與上述 acquire-local
`CavityAverageCache` 不屬於已刪三層。抽樣 RNG、raw 與 replay 的契約不變。
依賴方向仍由 C8、C9、C12、C19 保護，沒有新增 package edge 或例外。

FTF 與 single 不同，不拆損耗無關結構。漂移開啟時，每次 acquisition 用 effective settings 重建完整 `FTFModelFacts`，包含 eigensystem，並清除舊的 complete preparations；
整次 points／rounds 共用這次的 losses，不跨不同 losses 重用 rates 或完整 preparation。漂移關閉的既有行為不變。
這是 advisor 於 2026-10-27 依使用者授權「B你可以自己開task，不用我看過」訂定的重建策略。數值、設定與 truth 的 owner 分工見 [[0072]]。

### Projective read 的 Markov chain 表示

2026-11-16，使用者要求「模擬器不該有 OOM」，看過選項後授權「OOM你就改designer決定」。
Designer 於 2026-11-17 依此代裁 single／six 的 projective read 表示。
2026-11-17，使用者覆核這份代裁（輸出改成 Markov chain、floor 改成逐點判斷的狀態層級語意、同 seed 結果改變、category 軸為人工 bins、移除 2 GiB 預算），答「兩筆追認」。

DC rank-1 投影之後，下一段演化只依投影狀態，所以一串 projective read 是以投影狀態為狀態的 Markov chain。
Single runner 輸出 `simulate.adc.ProjectionChain`：每點第一個 read 的 populations，加上每個後續 read 依前一個投影狀態的轉移表。
Six 對每個 cavity 使用同一個 runner。
Runner 不列舉 outcome history，也不設 projection 記憶體預算；記憶體隨點數線性成長，raw 另隨 reps 成長。
每一點、每個轉移各自判斷：該點近似機率不低於 `projection_prune_floor` 的 source state 採用 `Engine.evolve` 算出的轉移列，其餘狀態在該點設 identity。
一點的近似不依同次 acquire 的其他點；floor 為零時是精確 chain。
`simulate.adc` 擁有 chain 的 raw／trace 取樣、integrated 的逐態 flow counts 與累積邊界 bins 編碼；six 合併 integrated 時呼叫同一個編碼器。
FTF 的 history 帶腔記憶，DC 末態不足以決定後續，FTF 維持 `ProjectionReadout`。
沒有 projective read 的 single／six 與 FTF 輸出逐位元不變。
有 projective read 時同 seed 的抽樣值改變；single private replay 升版，舊 archive 明確拒絕。
選項比較、誤差界與先例立場見[討論快照](../discussions/2026-11-17-projection-markov-chain-readout.md)。

### Integrated raw 由環境生成，介面由 program 擁有

使用者於 2026-10-20 裁決：「1-1」「基本認同丙，但我質疑有需要進core嗎？」。
介面改放 program 後，使用者答「同意」。

`program.v2.simulation.RawShots` 定義延遲產生完整單發 ADC sums 的介面。
環境以 `IntegratedRound` 一起交付純統計資料和 raw source。
Program 只在單發 processing 或 getter 需要資料時呼叫 source，不重跑 backend。
`simulate.boundary.RoundStatistics` 只有資料，不定義 materializer、Protocol 或 callback。
統計資料的 no-callback 邊界不變。

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
