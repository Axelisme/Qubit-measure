---
status: accepted
---

# 0078 — Evolution 黑盒邊界與算法內聚

## 背景

演化引擎要比較各算法的適用條件、準備成本與資料重用。數值 certificate 依賴各算法的 eigensystem、週期表等中間結果。若把這些結果藏在算法後面，分類只能重算，或靠不完整的宣告選路。

## 決策

### 黑盒邊界是 simulate.evolution

**規則。** 外部只透過 `simulate.evolution` 的 package root 呼叫引擎及使用資料型別。呼叫端提供 numeric model source、timeline、初態與誤差預算。引擎回傳 readout 起點的量子態與 family 診斷。Environment 明示組合 population 轉換與 ADC，讀出樣本的加總歸 `simulate.adc`。

**例子。** 要查一列選到哪個 family，用 `Engine.inspect_dispatch`，不直接呼叫 dispatcher。查詢與 evolve 共用分類、guard 和完整 inventory 的計價，不演化呼叫端的 density。已決定的路由與相同初始 history 下的 evolve 一致。`scheduling_cost` 是交錯排程用的相對工作量，不是秒數或候選排序用的冷攤提估價。逐列回報、未決原因及查詢的快取行為見 [Dispatch inspection](../../lib/zcu_tools/simulate/evolution/README.md#dispatch-inspection)。

**邊界。** Kernel、dispatch、scheduler 與 cache 都是內部實作。查詢不寫 routing history 或登記重用事實，也不改變下一次 evolve 的路由或 density。Guard preparation 在查詢返回時釋放，source 仍可填入模型快取與 shift registration。量測層可借出 `RoutingHistory`，也可指定整次 acquire 的候選白名單；逐段仍由 dispatch 選路及驗證。這兩個控制不進物理 YAML、`AcquireRequest`、實驗 cfg 或 program binding。沒有逐段 family 對照、callback、偏好順序或公開 kernel registry。固定單一候選可以略過成本捷徑。白名單或 pin 都不能略過前置條件或 guard；`multitone_gate` 的建構後例外見[正確性規則](#先保證正確再優化瓶頸)。借用、清除及無可用候選時的錯誤契約見 [Accuracy, failures and lifecycle](../../lib/zcu_tools/simulate/evolution/README.md#accuracy-failures-and-lifecycle)。

**依據。**
- 2026-10-18，使用者定案：「Q1A，是的，演化引擎黑盒子就是指simulate.evolution。」原裁決未附入庫討論，見 Git `ee1eeaa8d`。
- 2026-11-12，使用者對具名歷史與整次 acquire 的 family 白名單答「可以」。原裁決未附入庫討論，見 Git `a95315c32`。
- 2026-11-16，使用者核可公開 dispatch 查詢的實作計畫：「1核可」。Floquet guard 保留一週期 propagator 計算，使用者答「B」。對「查詢不改 routing history」與「欄位只回報既有排程工作量」答「追認」。原裁決未附入庫討論，核可與追認分別見 Git `2c59fcdec`、`0a81af3c7`。測試與量測的背景見[模擬器測試與量測的分層](../discussions/2026-11-15-simulator-test-measurement-layering.md)。

### `Engine` 的 GPU backend

**規則。** `Engine` 選的是 `dop853` 的執行裝置，不新增 family，也不讓呼叫端逐段選算法。預設等到第一列符合條件才偵測裝置，按成本選 CPU 或 OpenCL。

**例子。** 測試、配對量測或逐位比對要固定 CPU 時，明示 CPU backend。參數、安裝與啟用條件見 [GPU backend](../../lib/zcu_tools/simulate/evolution/README.md#gpu-backend)。

**邊界。** GPU 執行中出錯不靜默退回 CPU。平台與無可用裝置時的行為分別見[跨平台規則](#跨平台不綁作業系統)與[可選 GPU 規則](#gpu-是可選功能)。

**依據。** 2026-11-18，使用者對 `dop853` 提供 OpenCL 後端答「A」。見 [`dop853` 的 OpenCL GPU backend](../discussions/2026-11-18-ftf-gpu-dop853-backend.md)。

### 凍結數值輸入與參考解交接

**規則。** `simulate.evolution` 公開凍結數值的 codec、容器、identity 與 solver-free reference I/O。Tools 擁有場景與 metadata 的解讀，tests 擁有獨立 solver 與參考解 producer。凍結數值不入庫，codec 只接顯式路徑，不解析 Git 位置。存放、場景 producer 與跨 process 計時的分工見 [[0072]]。

**例子。** 成對計時只讀已備妥的參考解。即使 cache miss 也不就地求解，由 tests 側 producer 按顯式的 manifest-to-solver 對應補出。只補一個場景時，producer 須先找到唯一同名列才讀 payload 或求解。沒有同名列就列出合法名稱；多筆同名則列出來源並報錯。容器欄位、hash 與 key、cache I/O 和單列選取的錯誤契約見 [Frozen numeric inputs and references](../../lib/zcu_tools/simulate/evolution/README.md#frozen-numeric-inputs-and-references)。

**邊界。** 讀回的是有限的數值模型，不是 Python builder、裝置佈局、routing history 或私有 preparation。容器不解讀 caller metadata，也沒有 solver 欄位。Producer 不 import tools，不從 metadata 或場景名稱猜 solver，失敗不換 solver。Physical 解碼契約變更由 producer 遷移；成對工具核對 physical、scene cache 兩個格式版本與完整 record 身份，不把算法數值版次當作相容性門檻。搬移 owner 不改既有 reference key。

**依據。** 2026-11-16，使用者以「1核可」核可分層實作計畫。格式與缺資料時的驗證方式見[凍結輸入討論](../discussions/2026-11-16-frozen-simulator-inputs.md)，單列準備見[共用清單與 Pareto 選集討論](../discussions/2026-11-17-pareto-selection-and-scene-append.md)。

### 模擬器的測試與量測分層，不做 e2e

**規則。** 模擬器測試與 kernel／family 比較只依賴模擬器，不把 program、編譯、演化和 I/Q 串成一項測試。Kernel／family 比較只計 `Engine.evolve` 的時間。各層各驗一件事：

- Kernel：用 `Engine.evolve` 固定候選，以一段輸入對獨立參考解量誤差與耗時。
- Dispatch：用 `Engine.inspect_dispatch` 查逐列 family、guard 與排程工作量，不演化呼叫端的 density。Guard 仍走 evolve 的路徑，包含自身需要的 ODE，故用小模型。
- 序列：用手寫多段 `TimelineBatch` 在 `Engine.evolve` 量最終 density 誤差及總耗時。整體有沒有變慢、family 是否保留都在這層判斷。場景按 gate 種類與個數宣告，不逐實驗增加。
- 前端：驗 program 編譯出的訊號，不演化。
- Environment：給定 density，驗讀出。FTF 的測試用不演化的 Engine 替身驗交接。

**例子。** 比較兩個 checkout 的序列總耗時時，成對計時直接呼叫 `Engine.evolve`，不經 acquire。精度案例與指標見[測試文件](../../tests/README.md#evolution-精度矩陣)。

**邊界。** 各層都重現不了使用者回報的錯時，可以跑一次 e2e 定位。腳本不入庫，不當驗收門檻。整實驗只允許 [`tools.experiment_wall_time`](../../tools/README.md#實驗-run-牆鐘) 入庫記錄正常參數的 Run 牆鐘與分項，不當測試、gate 或精度證據。該工具用固定模擬輸入，在 `time_scale=0` 下只計送出 Run 到終態，不呼叫 Analyze 或 Save。這個具體範圍由 designer 代裁，使用者未覆核。

**依據。**
- 2026-11-15，使用者：「我會預期模擬器測試腳本只會依賴模擬器模塊，後端與前端分開測，kernel與dispatch也分開測，不該有e2e的測試需求」。對「整條序列的最終誤差放哪一層」答「A」，即加序列層；同意只為定位而跑一次 e2e，答「我選B」。見[模擬器測試與量測的分層](../discussions/2026-11-15-simulator-test-measurement-layering.md)。
- 2026-11-20，使用者問：「為什麼要用不入庫的 scratch 腳本量?而不是入庫的？」2026-11-21，對用 Run 牆鐘作優化指標、工具入庫而不當 gate 答「都採用建議」。同日 designer 代裁上述工具的具體範圍，使用者未覆核。見[實驗 Run 牆鐘工具](../discussions/2026-11-21-experiment-run-wall-time.md)。

### 每個 kernel 的量測場景只有一段

**規則。** 優化或校準單一 kernel 時，場景只用該 kernel 處理的一段。要量重複就把同一段接 N 次。一次改一個約化軸，誤差與耗時在同一組座標量。

**例子。** 固定 `dop853` 測 sampled 多 tone。兩個 tone 同起同止，各帶一條包絡；以同一段重複 1、2、20、100 次。「兩個 Gaussian 錯開 8 ns」會被切成多段，不用來量單一 kernel。

**邊界。** 要量使用者等待的總時間、選路、段間銜接或 family 去留時，改用序列層手寫的多段 `TimelineBatch`。

**依據。** 2026-11-15，使用者：「每一个 kernel 的场景，应该都应该只有一段。如果你想要测试重复的，那就是好几个重复段，不会出现两个高选然后交错在一起，然后实际上有五段这样子，这这没什么用。」其中「高选」讀作「高斯」。原始 STIRAP 多段 workload 的量測檢討未入庫；分層見[上一節](#模擬器的測試與量測分層不做-e2e)。

### 場景覆蓋靠約化的參數空間

**規則。** 測試與驗收不為個別實驗加場景。先把輸入約化成 kernel 與 guard 看得到的物理量：每個 tone 的耦合，也就是 gain 乘該 flux 下的矩陣元、到最近躍遷的失諧、時長、包絡形狀與 tone 間距。先在約化空間宣告覆蓋格點，再檢查實驗的座標是否落在覆蓋內。

**例子。** AC Stark 與 FTF mux 的兩個常數 tone 共用一張三軸格點：共振轉角（耦合 × 時長）、離共振比（耦合 ÷ 失諧）、累積 Stark 相位（耦合² × 時長 ÷ 失諧）。

**邊界。** 約化丟掉的物理若會改變結果，就加一個物理軸，例如讀出投影、連續 flux 或量子 resonator，而不是新增實驗專屬場景。序列層按 gate 種類與個數宣告。RB 型場景用隨機相位，避免固定 X、Y、−X、−Y 循環抵消部分 RWA 誤差。

**依據。** 2026-11-14，使用者：「实验场景非常多，要完整覆盖是相当困难的，无脑扩充测试场景只会拖慢开发速度，你应该约化不同场景，系统性的覆盖可能情况」。對整理後的原則答「符合，我預期不會加專屬實驗場景」。`multitone_rwa` 的原始量測範圍討論未入庫；序列取樣見 [RB 型相位序列作為固定量表輸入](../discussions/2026-11-14-random-gate-sequence-workloads.md)。

### 歷史控制與精度驗收

**規則。** Dispatch 可以依量測歷史選不同 kernel。精度以獨立的完整 Hamiltonian 參考解驗收，不拿目前或重構前的輸出當真值，也不要求前後逐值相同。每個可用候選都要驗。Density 最大絕對差與 population 等效 expected I/Q 誤差的目標均為 `2e-3`。

**例子。** 比較空歷史與沿用歷史時，量測層用公開的 history 與 family 控制，不用 monkeypatch 改引擎狀態。兩種選路各自對參考解驗收。

**邊界。** 已知的兩個 guard 邊緣超標以 strict xfail 保留，不放寬門檻。環境 expected I/Q 驗收尚未交付。見 [Kernel summary](../../lib/zcu_tools/simulate/evolution/README.md#kernel-summary) 與[精度矩陣](../../tests/README.md#evolution-精度矩陣)。新增或優化算法不要求修改不相關模組，擴充位置見 [Adding a family](../../lib/zcu_tools/simulate/evolution/README.md#adding-a-family)。

**依據。** 2026-11-12，使用者：「B，只要為了優化速度可以損失一點一致性，只要誤差在容許範圍」，並要求「應該已正確的物理演化為誤差參考，而不是當前實作，否則誤差會累計」。原裁決未附討論紀錄。

### Family 去留：合格只比耗時

**規則。** 每點跑完整條 pulse sequence 後，最終 density 對獨立參考解的最大絕對差 ≤ `2e-3` 即合格。合格的 family 視為一樣準，只比耗時。另一個合格 family 快超過 10% 才算支配。

**例子。** 2026-11-13 討論的 Gaussian gate 序列 S 表中，`shaped_sambe` 的誤差是 `1.45e-4`、耗時 0.27 秒；`period_table` 是 `2.96e-5`、9.0 秒；`gate_cache` 是 `1.10e-5`、54 秒。三者都合格，`shaped_sambe` 支配準一個數量級但更慢的兩個。這組範例不替代後續量表的實測。

**邊界。** 獨占一個適用域的 family，或該域最後一道候選，不參加比較。單 kernel 的誤差與逐段 `budget` 都不是合格證據。要刪掉被支配的 family，先量到支配者在同型態長序列上仍 ≤ `2e-3`，gate 數須遠多於 24，再請使用者看數據決定。量表機制見 [Engine family Pareto table](../../tools/README.md#engine-family-pareto-table)。

**依據。** 2026-11-13，使用者：「我通常是要求在给定最终精度容许范围内尽可能加速，所以通常这种精度差距极小的，我会优先选速度。」並澄清：「你要注意哦，这个精度是最终精度，也就是整个 post sequence 模拟最后的精度，而不是单个 kernel 的精度哦。」對整理後的規則答「確認」。見 [family 支配定義的討論](../discussions/2026-11-13-family-dominance-within-tolerance.md)。

### 先保證正確，再優化瓶頸

**規則。** Guard 在整個參數空間不放行最終誤差超過 `2e-3` 的結果。速度優化只針對有耗時紀錄、使用者在等的瓶頸，也就是實際工作負載中耗時最多的場景。從 kernel 表、量表摘要或討論找紀錄，找不到才問使用者。快速算法的 guard 可以比它算得對的區域窄。

**例子。** `t6` 的 direct 優化針對四個有耗時紀錄的慢場景。掃長度的 first-evolve wall 從 736 s 降到 6.3 s，四個場景的最終誤差都 ≤ `2e-3`。見 [t6 的 direct 加速策略](../discussions/2026-11-14-t6-direct-strategy.md)。

**邊界。** 修法很快會落地時，過渡期維持現狀。修法仍在研究、時程未定時，先收窄 admission，改走慢但對的候選，修好再放回。Sampled multitone 的[過渡期拒收](../discussions/2026-11-14-sampled-multitone-interim.md)用的就是後者。另有一項例外：`multitone_gate` 比價選中就建構及使用，不在建構後另做精度檢查、不因精度退回 `dop853`，數值失敗仍直接回報。其他 family guard、sampled `multitone_rwa` 抽查及動態維度 pilot 不適用這項例外。

**依據。**
- 2026-11-14，使用者問「模拟器不应该是覆盖所有可能范围吗？」，並說：「我预期就是首先先确保能全覆盖，然后再来是对于瓶颈场景做优化。」對時程未定時先收窄 admission 答「A」。`multitone_rwa` 適用範圍的原始討論未入庫。
- 2026-11-19，使用者：「我应该说过，不要因为听度而回退。比价的时候决定好就不要再改了，就算结果超标也是能接受。我不想白付时间去尝试错误的路径」。其中「听度」即精度。使用者把範圍限定在 `multitone_gate`。原裁決未附入庫討論，見 Git `7043877d5`。現行路徑見 [Kernel summary](../../lib/zcu_tools/simulate/evolution/README.md#kernel-summary)。

### Evolution 輸出合法 density

**規則。** `Engine.evolve` 在回傳點保證 readout density 符合 continuation 的 Hermitian、trace 與最小特徵值條件。不合法的輸出投影回合法 density，但最大元素絕對改動不得超過本次呼叫的 budget。

**例子。** FTF heralded 的純態經 50 ns 高 gain 雙 tone 段走 CPU DP5(4)，最小特徵值為 −2.4e−6。這次輸出的誤差符合 `2e-3` 的參考解精度目標，卻過不了 continuation 的合法性檢查。投影後兩個重現案例都通過；修正約占 evolve 總耗時的 0.05%–0.09%。

**邊界。** 修正超出 budget 就回報數值失敗，不放行、不改 kernel 選路，也不放寬 continuation 容差。投影步驟、容差與指出 point 的錯誤見 [Accuracy, failures and lifecycle](../../lib/zcu_tools/simulate/evolution/README.md#accuracy-failures-and-lifecycle)。

**依據。** 2026-11-19，使用者選定回傳前投影，修正量超過 budget 就回報失敗，並回覆「其余采用建议」。見 [Evolution 輸出合法 density](../discussions/2026-11-19-evolution-valid-density.md)。

### 固定預算，找適合的場景

**規則。** 按使用者宣告的預算評估算法，預設為完整 density 對參考解 ≤ `2e-3`。在預算不變的前提下，找出划算的控制形狀、掃描點數、時長和初態。不適用的場景由 guard 或選路退回。

**例子。** 截到能量最低 24 態時，單 tone FTF 掃描誤差 ≤ `1e-3`、每點約快 2 倍。兩 qubit 同時打 X90 的 depth 500 序列分給每個 gate 的額度很小，截斷不划算；不為此放寬 RB 的預算。

**邊界。** 使用者可以按場景另訂需求，不為讓某個算法過關而放寬預算。以 FTF RB 長序列為例，使用者接受按 gate 數放寬到 max(`2e-3`, `3.9e-6` × gate 數)。例子中的 depth 500 序列有 2,581 個 gate，預算約 `1e-2`。

**依據。** 2026-11-18，使用者：「我認為不同場景的預算分配確實不同，但模擬算法的衡量不應該改變預算，而是固定預算並找適合的場景」，並說：「還有對於depth 500這種超長序列，我能接受更多的誤差。」對 FTF RB 長序列的預算答「A」。見 [FTF 模擬的子空間約化](../discussions/2026-11-18-ftf-subspace-reduction.md)。

### 跨平台，不綁作業系統

**規則。** 模擬器的算法與執行機制同時適用 Windows 和 Linux，不依賴其中一個系統的設定或行為。

**例子。** GPU dispatch 量每批耗時，把每次送出控制在約 0.1 s，不依 Linux xe driver 的 `preempt_timeout_us=640000`，也不要求改 Windows 的 TDR 設定（預設 2 s）。

**邊界。** 沒有可用 GPU 驅動時可用 CPU，結果仍正確，只是較慢。macOS 不在支援範圍。

**依據。** 2026-11-18，使用者：「由于我会跨电脑平台使用，所以不要采用局限于特定 OS 系统的算法」，並確認：「只有 Windows 和 Linux，其餘採用建議」。見 [`dop853` 的 OpenCL GPU backend](../discussions/2026-11-18-ftf-gpu-dop853-backend.md)。

### GPU 是可選功能

**規則。** 沒裝 GPU 套件或沒有可用 GPU 時，預設走原 CPU 路徑，行為與結果逐位不變。

**例子。** Windows 筆電沒裝 `gpu` extra，同一份 FTF 腳本照樣走 CPU。

**邊界。** 只有明示要求 GPU 時，找不到可用裝置才報錯。選項見 [GPU backend](../../lib/zcu_tools/simulate/evolution/README.md#gpu-backend)。

**依據。** 2026-11-18，使用者：「也有可能在没有 GPU 的系统上使用，所以 GPU 功能确实是属于可选开启」，並說：「OpenCL 应该是没问题的」。見 [`dop853` 的 OpenCL GPU backend](../discussions/2026-11-18-ftf-gpu-dop853-backend.md)。

### 快速算法先修好，不輕易放棄

**規則。** 快速算法有誤差問題時，先查原因並試修。提出刪除或收窄前，寫清誤差或變慢的來源、能不能修。修不好的範圍才收 guard，快速算法在算得對的瓶頸繼續使用。

**例子。** `floquet` 起初只量到拒絕或單點較慢。使用者選擇再研究一輪並設停損條件，在 committed 批次上比 `dop853` 快 1.65–1.81 倍，因而保留並重校價格。見 [Floquet 的保留與價格重校](../discussions/2026-11-16-floquet-kept-price.md)。

**邊界。** 查過與試修後仍找不到能算對的範圍，才收窄或刪除。過渡期另依[正確性規則](#先保證正確再優化瓶頸)。已算得準、卻不服務現有瓶頸的路徑可以移除，例如 [`t6` 的 channel](#t6-只保留-direct)。刪除 family 仍按[去留規則](#family-去留合格只比耗時)讓使用者看數據決定。

**依據。** 2026-11-14，使用者：「我不太想放弃快速的算法，目前的优化需求还蛮大的」。`single_tone_rwa` 去留與 `multitone_rwa` 超標的原始討論未入庫。

### 支援範圍不看使用頻率

**規則。** 物理與介面決定哪些輸入要支援、算對與合理計價；在[約化參數空間](#場景覆蓋靠約化的參數空間)上驗覆蓋，不看使用者常不常用。

**例子。** 量 prepare 價格如何隨重用攤提時，把同一段分別接 1、20、100 次。高重用的輸入即使用得少，也在量測範圍內。

**邊界。** 只有在已有耗時紀錄時，使用頻率才用來排優化先後。過渡期的速度代價照樣量，不以「這類序列較少使用」替代 benchmark。不為省時間而放棄算對高重用的輸入。

**依據。** 2026-11-16，使用者：「你為什麼老是問我常不常用？我常用與你模擬器要不要支援是兩回事」，對使用頻率只用來排優化先後答「A」。`multitone_rwa` 校價需求的原始討論未入庫。過渡期量測及高重用案例見 [Sampled multitone 的過渡期拒收](../discussions/2026-11-14-sampled-multitone-interim.md)及 [t6 只保留 direct](../discussions/2026-11-15-t6-channel-removal.md)。

### 模擬器不替使用者擋實驗參數

**規則。** 就實驗參數而言，模擬器只拒絕實機也會拒絕的程式。程式合法但參數不合理時，照算物理上預期的結果。模型算不準的部分用警告說明，不因參數不當報錯。

**例子。** FTF 規劃 ADC window 時，若整個 template 沒有 readout pulse，event 只有 receiver offset 與雜訊，不報「沒有 generator 候選」。下一個 pulse 在 window 內抵達且已配到這個 event 時，兩個 pulse 相干相加。若抵達的 pulse 漏配，規劃就報錯，指出 ADC、window 和 pulse，不回傳缺訊號的 I/Q。見 [FTF readout and sampling](../../lib/zcu_tools/simulators/ftf/environment/README.md#readout-and-sampling)。

**邊界。** 型別、範圍、維度等契約外輸入仍在進入點 fast fail。物理上無法判定的輸入也拒絕，例如 Six 的 raw ADC 字到兩個 cavity 的同餘距離平手。模擬器未實作的情況不能回傳缺訊號的結果；FTF 仍按發射時間配對 ADC trigger，在途漏配由規劃守門攔下。算法數值近似仍受[精度 guard](#先保證正確再優化瓶頸)約束。

FTF cavity drive 使相鄰態能量反轉時，原色散模型仍能算，不重排狀態。警告指出受影響範圍與 gain 共同縮放上限；超限結果不在色散模型有效範圍內。計數方式與上限假設見 [FTF cavity drives](../../lib/zcu_tools/simulators/ftf/environment/README.md#cavity-drives)。

**依據。**
- 2026-11-15，使用者覆核 sideband reset 的殘餘光子檢查：「4我覺得有需要嗎？實驗參數設置不當就會發生的問題為什麼模擬器要擋？」對照算並警告答「A」。原始討論未入庫。
- 2026-11-19，使用者同意 cavity Stark 反轉照算並警告，答「1cA」。2026-11-20，designer 依授權代裁共同縮放上限的表示，使用者未覆核。見 [FTF cavity Stark 反轉：從報錯改成警告](../discussions/2026-11-20-ftf-stark-inversion-warning.md)。
- 2026-11-21，使用者定下「只拒絕實機也會拒絕的程式」這條規則，原話是「模擬器要給出合法的預期結果，只要合法，不因為不合理就抱錯」。FTF 的 receiver-only event、相干相加與在途漏配拒絕是 designer 代裁，使用者未覆核。Six 的平手拒絕也是 designer 代裁，隨 raw 字的同餘機制在使用者覆核後保留。使用者覆核的是 raw 字的同餘機制：「那我选，K 吧，感觉有些少频率的，比如说像是 one tone frequency，如果不是等间距的话，要先计算 table，然后存起来。感觉就会出现模拟上需要解读弱智的问题。」這句沒有單獨裁定平手或 FTF 時序。見 [FTF 合法讀出時序](../discussions/2026-11-21-ftf-legal-readout-timing.md)。
- 51 點、21 能階的合法輸入不受 projection 記憶體預算阻擋。案例與驗收見 [Projective read 的輸出改成 Markov chain](../discussions/2026-11-17-projection-markov-chain-readout.md)。

### 模擬器不從程式內容推斷實驗類型或硬體架構

**規則。** 選路、精度與維度不依「程式內容被辨認成哪種實驗或架構」切換。使用者宣告精度，沒宣告就用預設 budget。

**例子。** 有 flux 脈衝的程式照用預設 budget；CZ 也可能從 charge line 打。

**邊界。** 使用者宣告的精度和直接量到的數值可以改變選路。動態維度的 pilot 與 `drop(k)`、sampled `multitone_rwa` 抽查及各 family guard 都可以靠直接比對計算結果或讀數值指標來決定選路或維度，不看實驗名稱、脈衝種類或架構。

**依據。** 2026-11-18，使用者：「这不合理，CZ gate 可能也是从 charge line 打，而且模拟器理应不知道现在的架构是 FTF 还是只有单个 F 还是什么的。」見 [FTF 動態維度](../discussions/2026-11-18-ftf-dynamic-dimension.md)。

### 模擬器帶通用調試介面

**規則。** 每個模擬器提供同一組調試介面，定位與驗證工具只依賴這組介面。共同欄位有型別；模擬器自行決定哪些額外值放進 dict。

**例子。** FTF、single 與 six 每次 acquire 以共用 writer 寫一行總 wall、編譯、演化、讀出與各 family 的分項，模擬器特有的欄位放在 `extras`。見[共用 telemetry](../../lib/zcu_tools/simulators/README.md#acquire-telemetry)。

**邊界。** 共用介面可以帶中性的模擬知識，不把 FTF 的腔記憶或 six 的 scatter 放成共用 hook。調試者就是模擬器維護者，介面可暴露內部資訊，不另為外部使用者承諾穩定 API。[模擬器的 debug capture](../../lib/zcu_tools/simulators/README.md#debug-capture)共用錄製與體檢介面，保存凍結輸入與實際選路。直接用 `ModelSource` 呼叫引擎仍能錄製與體檢，但沒有前端和讀出的耗時。

**依據。** 2026-11-16，使用者：「優化手段要看模擬器是否實作功能，應該是要規範模擬器應該帶有用於調試的通用接口，類似基類或protocol的概念」。對資料形狀答：「可以，但我認為這個資料不一定要完全強型別，可以用多餘資訊的接口，比如dict屬性，放值得記錄的資訊」。對使用者答：「可以，調試者應該就是模擬器維護者，我沒有意向去解耦合兩者責任」。對共通部分答：「中性的模擬知識是沒有問題的。場景特化的才不建議」。原始討論未入庫；實例見[共用 acquire telemetry writer](../discussions/2026-11-16-shared-acquire-telemetry.md)與 [Engine capture 的逐輪選路事件](../discussions/2026-11-17-engine-capture-events.md)。

### 優化優先於周邊約定；除錯靠 dump，不靠重算

**規則。** 以使用者看得到的結果正確性評估優化，完整 density 必須在誤差預算內。存檔格式、版本號、相容性與重現流程配合優化調整。除錯讀 debug capture manifest，缺欄位就補 manifest，不靠重算。

**例子。** 數值 kernel 改動通過完整 density 精度預算後，不必為 replay 保留舊數值路徑。要診斷某次演化，從 manifest 讀 Engine 輸入、逐列 routes、seed、每輪 DC、model identity 和有效設定。用法見[模擬器的 debug capture](../../lib/zcu_tools/simulators/README.md#debug-capture)。

**邊界。** 使用者看得到的結果正確性、GUI 與設定檔介面、量測結果語意維持原樣。若與優化衝突，由使用者裁決。

**依據。** 2026-11-18，使用者：「关于 replay 啊之类的其他成本，拜托你以优化的效率为优先，而不是这些其他的细枝末节的仪式、一些储存格式。你如果迁就它的话，优化很难做啊。」又說：「你需要的资讯应该本身就在 dump 的结果里了。如果你会需要 replace 多余的资讯、多余的算法，那就是你 dump 的资料不够多嘛。」對原則行答「好」。見[移除 private replay](../discussions/2026-11-18-remove-private-replay.md)。

### 用快場景的小幅變慢，換慢場景的大幅加速

**規則。** 預設選路可以先付一筆有上限的固定成本來抽查、估價或做 pilot，讓慢場景大幅加速。這筆成本不必先估價才付。

**例子。** Sampled 雙 tone 列抽查沒過時，場景耗時是 main 的 1.168 倍；抽查通過時，24 點掃描比 main 快 3.149 倍。

**邊界。** 多付的成本不隨工作量變大。先算的路徑最後沒被採用，且端到端時間超過 main 的 1.2 倍時，回報使用者決定。

**依據。** 2026-11-18，使用者：「以比较快的场景稍微变慢一点为代价，让慢场景快很多是可以接受的」。見 [sampled multitone_rwa 的實測放行](../discussions/2026-11-18-sampled-multitone-rwa-admission.md)。

### 內部允許實質耦合，dispatch 擁有決策知識

**規則。** Dispatch 直接讀各算法的資料、中間結果與數值工作估算，決定適用條件、價格、候選順序、拒絕後的下一候選及 family 內的 builder、frame 與 chain。Kernel 擁有需要數值中間結果的 certificate，回報 acceptance 與可重用產品。

**例子。** Dispatch 合併各 family 的唯讀 `RowPlan`。Scheduler 不解讀數值內容，原樣交給 apply。Apply 用已完成的 admission 與 frame／chain 決策，不針對當次子集重新分類。型別與傳遞方式見 [Row plans and the kernel seam](../../lib/zcu_tools/simulate/evolution/README.md#row-plans-and-the-kernel-seam)。

**邊界。** Plan 只活在一次 evolve，不留在 kernel state。排程接縫不作對外擴充框架。`PathDiagnostics` 按 family 計數，builder 與 frame variant 留在內部。不為隔離分類需要的資料而新增 Protocol、registry 或通用 declaration；改一個算法時可能也要改 dispatch。

**依據。** 2026-10-18，使用者：「Q2只要是有助於分類的資訊，planner都可以知道。」現行 dispatch 沿用這條裁決。原裁決未附入庫討論，見 Git `ee1eeaa8d`。[Planner evidence 與固定成本討論](../discussions/2026-10-18-planner-evidence-and-overhead.md)是研究紀錄，不是現行 API 契約。

### 算法內聚，共用數值有單一 owner

**規則。** 算法的數值實作留在自己的模組與專用 helpers。多個算法共用的數值由單一模組擁有，kernel 不借用另一個 kernel class 的方法。Rates→Q／decay、period playback 與 fractional tail 的 owner 見 [Algorithm ownership and reuse](../../lib/zcu_tools/simulate/evolution/README.md#algorithm-ownership-and-reuse)。

**例子。** ADC 擁有 flat 腔場二階矩的精確結果 memo，負責 lossless key、有界容量與回傳陣列 ownership。FTF 在一次 acquire 的 span composition 建立最多 64 MiB 的 owner，供 g/e 各點串行共用。正常返回或拋出 exception 時都清理。`CavityDrive` 只轉交 owner，不組 ADC key。
現行的 flat span memo 也收各 state 的端點場；這項擴充是 designer 代裁，使用者未覆核。ADC 同次求解產出兩種產品，共用原本的容量與壽命。無法證明等價時分開求解、不收錄。詳細 key、容量、清理見 [ADC README](../../lib/zcu_tools/simulate/adc/README.md) 與 [FTF cavity drives](../../lib/zcu_tools/simulators/ftf/environment/README.md#cavity-drives)。

**邊界。** Memo 不留在 Engine、global 或長駐 source。FTF 仍選態、建 Stark／dephasing shift 並傳場給下一段。Shaped/callable 與主分支選擇不進 memo；projective 只重用既有的 flat 二階矩，不收各 state 端點場。跨 package 的依賴維持 [[0072]] 與 C8、C9、C12，不新增依賴方向或放寬 `ignore_imports`。這些 contract 不檢查內部子模組的匯入路徑；package root 的公開入口由介面與 caller review 維持。

**依據。**
- 2026-10-18，使用者：「之後我的規劃是讓你先整理目前模擬引擎，使其對外是黑盒子，對內是實質偶合，決策分類有全部算法的知識，算法部分內聚，以不影響優化與可讀性為主要指標，移除已經沒用的邏輯，將研究成果以及嘗試紀錄到repo文檔中供以後參考。然後就可以開始planner的重構，使各優化算法的實作張力能以自然的表達」。原裁決未附入庫討論，見 Git `ee1eeaa8d`。
- 2026-10-21，使用者對二階矩 memo 的方案答「前兩個採用建議」。原裁決未附入庫討論，見 Git `ffe98f8f7`；[MIST 腔場重用的先例整理](../discussions/2026-11-21-mist-cavity-reuse.md#先例與立場)列出這項裁決的適用範圍。
- 2026-11-21，designer 依使用者授權「我授权你的范围除了 push 还有真实硬体之外都可以做」代裁端點場擴充，使用者未覆核。正常參數重測為 281.5 s 與 163.3 s；兩次 commit 不同，不能把差值全歸因於快取。二階矩單獨重用的上限及比較見 [MIST 腔場重用](../discussions/2026-11-21-mist-cavity-reuse.md)。

### t6 只保留 direct

**規則。** `t6` 在 constant 單 tone 域只走 direct，保留原 guard、prepare-key 身份和失敗就 raise 的語意。不建 reusable channel，不做 Fourier playback，也不為 channel 收完整 acquire key inventory 或比價。

**例子。** 每列 density 經共用的 endpoint solver 在 H0 旋轉座標積分，用完整 secular bath 與原 budget-scaled 局部容差。實作入口見 [Kernel summary](../../lib/zcu_tools/simulate/evolution/README.md#kernel-summary)與[共用數值 owner](../../lib/zcu_tools/simulate/evolution/README.md#algorithm-ownership-and-reuse)。

**邊界。** 其他 family 的 period channel 不變。未來若出現高重用工作負載，再依新證據評估，不保留相容性分支。

**依據。** 2026-11-15，使用者裁定 `t6` 只保留 direct，答「2B」。量測與方案比較見[移除 t6 channel 的討論](../discussions/2026-11-15-t6-channel-removal.md)。

### 損耗漂移下的結構與 preparation lifetime

**規則。** Single、six 與 FTF 都支援損耗漂移，預設關閉。每次 acquisition 開始取一次有效損耗，整次固定。只在 acquisition 之間命中，不足以保留沒有同次命中的快取層。Engine 不保留跨 evolve 的數值 preparation 或 exact result，source 不保留跨 preparation 的 ADC solver。Engine 的 routing-facts cache 只以 inventory 與 pricing facts 驗證 caller-owned preparation 的當前路由，不存傳播結果。

**例子。** Single 的物理模型擁有損耗無關的結構，experiment 只組合結果。損耗改變時，source 保留該結構，關閉舊 ModelSource，丟棄完整 models 與 preparation，按本次固定的 losses 建 rates。溫度、DC、truncation 等結構依賴改變時才丟棄結構。FTF 漂移開啟時不拆結構，按每次 acquisition 的 effective settings 重建完整 facts，包含 eigensystem，並清掉舊的完整 preparation；points 與 rounds 共用這次的 losses。漂移關閉時不採這套重建策略。見 [Single reuse and lifetime](../../lib/zcu_tools/simulators/single_fluxonium/environment/README.md#reuse-and-lifetime)、[FTF acquisition clock and loss drift](../../lib/zcu_tools/simulators/ftf/environment/README.md#acquisition-clock-and-loss-drift)與 [FTF preparation reuse](../../lib/zcu_tools/simulators/ftf/environment/README.md#preparation-reuse)。

**邊界。** Rates、dissipative propagator、density 與完整 preparation 不跨不同 losses 重用。同輸入的不可變完整 preparation 可按設定、losses 與路由條件重用。FTF 的 complete preparation LRU、envelope fingerprints 與 acquire-local `CavityAverageCache` 仍可重用。即使不開漂移，FTF 重複 acquisition 也可能要重算。中途 projective readout 由環境在 read 起點投影並續演化；分支 density 只活在一次 span preparation，不作下一 shot、round 或 acquisition 的初態。能重用的是不可變的 FTF outcome weights 與 event populations、single／six 的初始分佈與轉移表，以及兩者的 conditional sums，不是抽樣後的 herald 結果。已交付 raw source 仍持有自己的不可變抽樣輸入。抽樣 RNG 與 raw 契約不因這項 lifetime 決定而變。數值、設定與 truth 的 owner 見 [[0072]]；C8、C9、C12、C19 不添依賴或例外。

**依據。**
- 2026-10-27（UTC），advisor 對快取分工的原話（不作現行契約）：「保留 single 的 _models／_prepared 與 FTF acquire-local 的 CavityAverageCache；拆 _facts 為結構與 rates。」同日使用者裁決：「我預期都會加入漂移機制」、「沒有不漂移的時候，我考慮C，應該是說，用不到的就可以移除」。原裁決未附入庫討論，見 Git `c6ff3f173` 的裁決與量測紀錄。
- 2026-10-27，advisor 依使用者授權「B你可以自己開task，不用我看過」訂定 FTF 重建策略。原裁決未附入庫討論，見 Git `257e8cc3a`；未見使用者覆核這項策略的紀錄。
- 2026-10-28，使用者以「A我認為可以完整版。」授權 single、six、FTF 的中途 projective readout。這句決定支援範圍，lifetime 沿用本節規則。原裁決未附入庫討論；[模擬環境草稿](draft/simulation-environments.md#acquire-接線與-projective-readout)也記有這項裁決。

### Projective read 的 Markov chain 表示

**規則。** Single／six 的 DC rank-1 投影後，下一段演化只取決於投影狀態。Single runner 輸出 Markov chain，不列舉 outcome history，不設 projection 記憶體預算。Six 對每個 cavity 用同一 runner。ADC 擁有 chain 的 raw／trace 取樣及 integrated 的逐態 flow counts 與累積邊界 bins；six 合併 integrated 時用同一編碼器。

**例子。** 每點先存第一次 read 的 populations，再存後續各次隨前一次投影狀態而變的轉移表。每點每次轉移各自套 floor。近似機率低於 floor 的 source state 用 identity，其餘採 Engine 算出的轉移列。某點的近似不看同次 acquire 的其他點；floor 為零就是精確 chain。資料形狀和抽樣見 [ADC README](../../lib/zcu_tools/simulate/adc/README.md) 與 [single environment README](../../lib/zcu_tools/simulators/single_fluxonium/environment/README.md)。

**邊界。** FTF 的 history 含腔記憶，單憑 DC 末態不能決定後續，仍用 `ProjectionReadout`。Single／six 的 chain 記憶體隨點數線性成長，raw 另隨 reps 成長。有 projective read 時，single／six 的同 seed 抽樣會因表示改變；無 projective read 的 single／six，以及 FTF，不因這項變更改輸出，結果逐位不變。

**依據。** 2026-11-16，使用者要求「模擬器不該有 OOM」，並授權「OOM你就改designer決定」。2026-11-17，designer 代裁 single／six 的表示；同日使用者覆核答「兩筆追認」。覆核涵蓋 Markov chain、逐點 floor、同 seed 的抽樣差異、人工 category bins，以及不設 projection 記憶體預算。見 [Projective read 的輸出改成 Markov chain](../discussions/2026-11-17-projection-markov-chain-readout.md)。

### Integrated raw 由環境生成，介面由 program 擁有

**規則。** `program.v2.simulation.RawShots` 定義延遲產生完整單發 ADC sums 的介面。環境透過 `IntegratedRound` 一起交付統計資料與 raw source。Program 只在單發 processing 或 getter 需要時呼叫 source，不重跑 backend。`simulate.boundary.RoundStatistics` 只有資料，不定義 materializer、Protocol 或 callback。

**例子。** FTF 的環境交付由 ADC 產生的完整 raw，包含物理訊號、receiver noise 和取整排列。單 carrier raw 抽 CTMC 的等待時間與轉移方向，保留每次跳躍，不重設腔場。介面見 [Compact timelines and statistics](../../lib/zcu_tools/program/v2/simulation/README.md#compact-timelines-and-statistics)，取樣見 [ADC README](../../lib/zcu_tools/simulate/adc/README.md)。

**邊界。** ADC 與 `simulate.white_noise` 各自產生完整 raw，以結構型別符合 `RawShots`，不 import program。White noise 由 [experiment 的環境組合根](../../lib/zcu_tools/experiment/README.md#explicit-simulation-environments)接線；FTF 由 [`simulators.ftf.environment`](../../lib/zcu_tools/simulators/README.md)組合框架機制。`simulate.rounding` 擁有兩種環境與 decimated transport 共用的取整和 shot-axis 排列，不負責 QICK averaging 或硬體操作。Raw 與直接平均須維持相同期望，不平移 raw 去配有限 shots 的平均。直接平均保留 conditional moments 與既有 RNG stream，不做逐 shot 工作。Caller 的完整 preparation 只存 noiseless inputs，不存 noisy shots 或 round identity。依賴方向維持 C8、C9、C12、C19，不加 `ignore_imports`。

**依據。** 2026-10-20，使用者答「1-1」，並問：「基本認同丙，但我質疑有需要進core嗎？」介面放到 program 後答「同意」。原裁決未附討論紀錄。

## 取捨

內部接縫隔離排程與數值表示，不保證換 kernel 時 dispatch 完全不用改。Guard 拒絕可以換候選，prepare 或 apply 的數值失敗直接傳給呼叫端。事後 accuracy exceedance 只記錄並警告，不因它修正 density、重算或重新分類；回傳 density 的合法性是[另一項檢查](#evolution-輸出合法-density)。

引擎不提供實驗 planner 的相容層，[[0077]] 不定義現行執行契約。本決策不保證消除所有中間結果的重算。若以後再設計 planner，資料壽命與重用要另行設計。

## 現行契約與研究依據

- [Evolution README](../../lib/zcu_tools/simulate/evolution/README.md) 擁有現行 API、決策圖、數值 owner 與失敗語意。
- [模擬環境草稿](draft/simulation-environments.md) 記錄 environment、model、evolution 與 ADC 的組合責任。
- [Planner evidence 與固定成本討論](../discussions/2026-10-18-planner-evidence-and-overhead.md) 保存已移除 planner 的實驗結果。
