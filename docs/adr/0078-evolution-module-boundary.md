---
status: accepted
---

# 0078 — Evolution 黑盒邊界與算法內聚

## 背景

演化引擎要比較各算法的適用條件、準備成本與資料重用。數值 certificate 依賴各算法的 eigensystem、週期表等中間結果。若把這些結果藏在算法後面，分類只能重算，或靠不完整的宣告選路。

## 決策

### 黑盒邊界是 simulate.evolution

**規則。** 外部只透過 `simulate.evolution` 的 package root 呼叫引擎及使用資料型別。呼叫端提供 numeric model source、timeline、初態與誤差預算。引擎回傳 readout 起點的量子態與 family 診斷。Environment 明示組合 population 轉換與 ADC，讀出樣本的加總歸 `simulate.adc`。`ModelSource` 可提供完整的 collapse operators。`Engine` 在求解座標以 `D[L]ρ = LρL† − {L†L,ρ}/2` 計算每個 jump，不把同一個矩陣拆成獨立方向率。

**例子。** 要查一列選到哪個 family，用 `Engine.inspect_dispatch`，不直接呼叫 dispatcher。查詢與 evolve 共用分類、guard 和完整 inventory 的計價，不演化呼叫端的 density。已決定的路由與相同初始 history 下的 evolve 一致。`scheduling_cost` 是交錯排程用的相對工作量，不是秒數或候選排序用的冷攤提估價。逐列回報、未決原因及查詢的快取行為見 [Dispatch inspection](../../lib/zcu_tools/simulate/evolution/README.md#dispatch-inspection)。例如呼叫端可把固定 Hamiltonian 下的腔衰減組成一個矩陣交給 `ModelSource`，由引擎保留該矩陣各分量間的相干轉移。

**邊界。** Kernel、dispatch、scheduler 與 cache 都是內部實作。查詢不寫 routing history 或登記重用事實，也不改變下一次 evolve 的路由或 density。Guard preparation 在查詢返回時釋放，source 仍可填入模型快取與 shift registration。量測層可借出 `RoutingHistory`，也可指定整次 acquire 的候選白名單；逐段仍由 dispatch 選路及驗證。這兩個控制不進物理 YAML、`AcquireRequest`、實驗 cfg 或 program binding。沒有逐段 family 對照、callback、偏好順序或公開 kernel registry。固定單一候選可以略過成本捷徑。白名單或 pin 都不能略過前置條件或 guard；`multitone_gate` 的建構後例外見[效率和驗證規則](#先求效率正確性事後補)。借用、清除及無可用候選時的錯誤契約見 [Accuracy, failures and lifecycle](../../lib/zcu_tools/simulate/evolution/README.md#accuracy-failures-and-lifecycle)。完整 collapse operators 只由固定 Hamiltonian 的 CPU `dop853` 求解。其他 family 和 backend 不接受帶完整 jump 的列，現有 guards 不放寬；不帶 jump 的路徑行為不變。FTF、Fock 排列及模型組成留在呼叫端，不為完整 jump 增設第二套 model-source API。

**依據。**
- 2026-10-18，使用者定案：「Q1A，是的，演化引擎黑盒子就是指simulate.evolution。」原裁決未附入庫討論，見 Git `ee1eeaa8d`。
- 2026-11-12，使用者對具名歷史與整次 acquire 的 family 白名單答「可以」。原裁決未附入庫討論，見 Git `a95315c32`。
- 2026-11-16，使用者核可公開 dispatch 查詢的實作計畫：「1核可」。Floquet guard 保留一週期 propagator 計算，使用者答「B」。對「查詢不改 routing history」與「欄位只回報既有排程工作量」答「追認」。原裁決未附入庫討論，核可與追認分別見 Git `2c59fcdec`、`0a81af3c7`。測試與量測的背景見[模擬器測試與量測的分層](../discussions/2026-11-15-simulator-test-measurement-layering.md)。
- 2026-11-22，designer 在使用者「我要离开一段时间，你继续代采」授權下代裁完整 jump 的公開數值契約。使用者覆核答「其他两个是接受。」見[完整 collapse operators 的取捨](../discussions/2026-11-23-evolution-full-collapse-operators.md)。

### `Engine` 的 GPU backend

**規則。** `Engine` 選的是 `dop853` 的執行裝置，不新增 family，也不讓呼叫端逐段選算法。預設等到第一列符合條件才偵測裝置，按成本選 CPU 或 OpenCL。

**例子。** 測試、配對量測或逐位比對要固定 CPU 時，明示 CPU backend。參數、安裝與啟用條件見 [GPU backend](../../lib/zcu_tools/simulate/evolution/README.md#gpu-backend)。

**邊界。** GPU 執行中出錯不靜默退回 CPU。平台與無可用裝置時的行為分別見[跨平台規則](#跨平台不綁作業系統)與[可選 GPU 規則](#gpu-是可選功能)。

**依據。** 2026-11-18，使用者對 `dop853` 提供 OpenCL 後端答「A」。見 [`dop853` 的 OpenCL GPU backend](../discussions/2026-11-18-ftf-gpu-dop853-backend.md)。

### 凍結數值輸入與參考解交接

**規則。** `simulate.evolution` 公開凍結數值的 codec、容器、identity 與 solver-free reference I/O。Tools 擁有場景與 metadata 的解讀，tests 擁有獨立 solver 與參考解 producer。凍結數值不入庫，codec 只接顯式路徑，不解析 Git 位置。存放、場景 producer 與跨 process 計時的分工見 [[0072]]。`FROZEN_PHYSICAL_VERSION=2`。Model row 缺 `collapse_operators` 表示沒有 jump；有 jump 時，依原順序保存完整複數矩陣，單位為 `ns⁻½`。顯式空集合不是缺欄位。沒有 jump 的模型保留原 identity、pin 和 reference key。

**例子。** 成對計時只讀已備妥的參考解。即使 cache miss 也不就地求解，由 tests 側 producer 按顯式的 manifest-to-solver 對應補出。只補一個場景時，producer 須先找到唯一同名列才讀 payload 或求解。沒有同名列就列出合法名稱；多筆同名則列出來源並報錯。容器欄位、hash 與 key、cache I/O 和單列選取的錯誤契約見 [Frozen numeric inputs and references](../../lib/zcu_tools/simulate/evolution/README.md#frozen-numeric-inputs-and-references)。例如重建原有無 jump 場景時，producer 在預設根 `scenes/v1-physical-v2` 寫入 v2 payload，沿用原 pin，對應的既有參考解仍可讀取。新 capture 寫到 `simulator-debug/v2`。

**邊界。** 讀回的是有限的數值模型，不是 Python builder、裝置佈局、routing history 或私有 preparation。容器不解讀 caller metadata，也沒有 solver 欄位。Producer 不 import tools，不從 metadata 或場景名稱猜 solver，失敗不換 solver。Physical 解碼契約變更由 producer 遷移；成對工具核對 physical、scene cache 兩個格式版本與完整 record 身份，不把算法數值版次當作相容性門檻。搬移 owner 不改既有 reference key。v2 readers 拒絕 v1 容器；新預設根不覆寫舊場景或 capture。遇到帶完整 jump 的 cache miss，不支援完整 L 的 solver 必須拒絕，producer 不得發布結果。正式 full-L 參考只由獨立、已驗證相同物理與完整 density 收斂的 solver 產生。

**依據。**
- 2026-11-16，使用者以「1核可」核可分層實作計畫。格式與缺資料時的驗證方式見[凍結輸入討論](../discussions/2026-11-16-frozen-simulator-inputs.md)，單列準備見[共用清單與 Pareto 選集討論](../discussions/2026-11-17-pareto-selection-and-scene-append.md)。
- 2026-11-22，designer 在代裁授權下選擇隔離 v2 容器並保留無 jump 的身份。使用者覆核答「其他两个是接受。」見 [physical v2 的取捨](../discussions/2026-11-23-frozen-physical-v2.md)。

### 模擬器的測試與量測分層，不做 e2e

**規則。** 模擬器測試與 kernel／family 比較只依賴模擬器，不把 program、編譯、演化和 I/Q 串成一項測試。Kernel／family 比較只計 `Engine.evolve` 的時間。各層各驗一件事：

- Kernel：用 `Engine.evolve` 固定候選，以一段輸入對獨立參考解量誤差與耗時。
- Dispatch：用 `Engine.inspect_dispatch` 查逐列 family、guard 與排程工作量，不演化呼叫端的 density。Guard 仍走 evolve 的路徑，包含自身需要的 ODE，故用小模型。
- 序列：用手寫多段 `TimelineBatch` 在 `Engine.evolve` 量最終 density 誤差及總耗時。整體有沒有變慢、family 是否保留都在這層判斷。場景按 gate 種類與個數宣告，不逐實驗增加。
- 前端：驗 program 編譯出的訊號，不演化。
- Environment：給定 density，驗讀出。FTF 的測試用不演化的 Engine 替身驗交接。

**例子。** 比較兩個 checkout 的序列總耗時時，成對計時直接呼叫 `Engine.evolve`，不經 acquire。精度案例與指標見[測試文件](../../tests/README.md#evolution-精度矩陣)。

**邊界。** 各層都重現不了使用者回報的錯時，可以跑一次 e2e 定位。腳本不入庫，不當驗收門檻。整實驗只允許 [`tools.experiment_wall_time`](../../tools/README.md#實驗-run-牆鐘) 入庫：它只量 [manifest](../../tools/data/experiment_wall_time/case-inputs.json) 中由維護者選入的場景，不以註冊清單決定列數。每個場景使用工具自持的 FTF 設定和乾淨 point，開現行 Form，套用場景覆寫，再跑完整 Run。它記錄 Run 牆鐘與分項，不當測試、gate 或精度證據。優化指標是模擬器核心耗時，即該 Run 各次 acquire 的編譯、演化、讀出與 raw materialize 四項之和。Controller、表單、Qt、繪圖、Analyze、Save 與 `ShotReducer` 等 host 處理不計入核心。Raw materialize 由呼叫它的模擬 binding 計時，寫進 [acquire telemetry](../../lib/zcu_tools/simulators/README.md#acquire-telemetry)，工具只彙總。不另建凍結的核心場景：改了實驗就重跑那一列；比較模擬器改動時，前後兩次在同一版實驗上跑。完整核心、下限與報表格式見[工具說明](../../tools/README.md#實驗-run-牆鐘)。速度判定依[模擬器速度目標](#模擬器速度目標)。工具在 `time_scale=0` 下只計送出 Run 到終態，不呼叫 Analyze 或 Save，也不做整庫 guide 盤點。速度判定只涵蓋已選場景；如果引用的 adapter 已不存在，該場景保留未量列。

**依據。**
- 2026-11-15，使用者：「我會預期模擬器測試腳本只會依賴模擬器模塊，後端與前端分開測，kernel與dispatch也分開測，不該有e2e的測試需求」。對「整條序列的最終誤差放哪一層」答「A」，即加序列層；同意只為定位而跑一次 e2e，答「我選B」。見[模擬器測試與量測的分層](../discussions/2026-11-15-simulator-test-measurement-layering.md)。
- 2026-11-20，使用者問：「為什麼要用不入庫的 scratch 腳本量?而不是入庫的？」2026-11-21，對工具入庫而不當 gate 答「都採用建議」。見[實驗 Run 牆鐘工具](../discussions/2026-11-21-experiment-run-wall-time.md)。
- 2026-11-21，使用者更正指標：「大头是模拟器本身，所以量模拟器速度就好了」。比較凍結核心場景與跑完整 Run 時，使用者問：「你觉得哪一种比较简单？是直接跑完整的1:1比较简单，还是复制场景比较简单？请你从维护性跟那个扩充性来考虑。」designer 建議照舊跑完整 Run、從 telemetry 讀核心並補 raw materialize、不建凍結場景，核心不計 host 處理；使用者答「那就采用建议吧」。`time_scale=0`、不呼叫 Analyze 或 Save，以及延後 materialize 的補充紀錄、核心下限與 180 秒判定的規則，是 designer 代裁，使用者未覆核。見[模擬器核心耗時](../discussions/2026-11-21-simulator-core-time.md)。
- 2026-11-23，使用者：「它其实不是量现在实验室有每个实验的时间，它是量特挑的场景模拟的时间。」對 guide 盤點答「A，不要做多餘的事，想檢查格式是否合理應該是其他工具負責，不是計時工具，類似的工具都該遵守KISS原則」。見[牆鐘工具量手選場景](../discussions/2026-11-23-curated-wall-time-scenes.md)。

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

### 模擬器速度目標

**規則。** 正常參數下，有掃描軸的 Run 以模擬核心耗時除以 Result 的掃描格點數，目標是平均不超過 0.1 秒。格點數是各掃描軸長度的乘積，不乘 software rounds 或每點 reps。無掃描軸、只取 shots 的實驗如 singleshot GE，仍以整個 Run 的核心耗時不超過 180 秒為目標。逐次評估型實驗也用 0.1 秒，但只計有 acquire 的評估。像 CZ optimize 一次候選可量多個 seed，仍只算一點。拒幅候選和還沒走到的候選各報數量，不進分母。

**例子。** 101 點的 onetone 目標為 10.1 秒，31×31 點的二維掃描為 96.1 秒。若將 31×31 的外層軸縮到 11 點，外推核心耗時是 11×31 實測值乘以 31/11。

**邊界。** 正常場景用現行表單預設的 reps 和 rounds。等距掃描用每軸至少 31 點攤平固定成本，離散掃描仍按其 Result 實際軸長計數。為減少量測成本，場景可明示縮減外層軸或 rounds。維護者在 manifest 記下縮減的 Form path 與完整點數或 rounds，工具不推導完整 Form 預設。只有完整核心耗時可按完整對實測的外層軸點數與 rounds 比率線性外推。表分欄報實測與外推的核心耗時、每點平均及判定。不將逾時的前綴外推，也不把估值當成完整 Run 的實測。下限只能在分母已知且已超過目標時證明超標。核心計時分項與 paired kernel 比較不變。

**依據。** 2026-11-23，使用者：「我意識到優化器評測時間依賴於點數，那應該改成平均每點0.1s，每軸至少會有31點來攤平固定成本」。對格點定義及無掃描軸的例外，使用者答：「都對，但我接受場景使用外插來減少測量時間成本」。這推翻 2026-11-21 的有掃描軸實驗整次 180 秒目標，保留只取 shots 的例外。比較過的方案見[模擬器速度改按每格點](../discussions/2026-11-23-simulator-speed-per-point.md)。

### 歷史控制與精度驗收

**規則。** 優先守[模擬器速度目標](#模擬器速度目標)。完整序列的最終 density 與逐讀出 population 對獨立參考解的最大絕對差均在 `1e-2` 以內時，視為一樣準，選耗時較短的做法。只有為了守速度目標而無法避免時，才接受超過 `1e-2`，並記錄誤差。任一誤差達 `1e-1` 時，寧可超時也不採用。

**例子。** 31 點場景的一個做法要 5 秒、誤差 `2e-3`，另一個要 2.5 秒、誤差 `3e-2`。選後者並記錄 `3e-2`。若後者誤差達 `1e-1`，改選前者。

**邊界。** Dispatch 仍可依量測歷史選不同 kernel。固定歷史或 family 是比較手段，不讓舊路徑成為真值。獨立精度以完整 Hamiltonian 參考解或對同一模型有誤差界的替代解判定；遷移比較只量前後差異。補驗時點見[效率和驗證規則](#先求效率正確性事後補)。既有 `2e-3` guard、驗收與 family 比較仍有效；原有 strict xfail 不放寬。新增或放寬容差時按本節驗；既有逐段 budget 不因此改值。見[精度矩陣](../../tests/README.md#evolution-精度矩陣)。

**依據。**
- 2026-11-12，使用者：「B，只要為了優化速度可以損失一點一致性，只要誤差在容許範圍」，並要求「應該已正確的物理演化為誤差參考，而不是當前實作，否則誤差會累計」。保留獨立參考的要求。
- 2026-11-22，使用者：「我会以模拟效率为优先考量。应该说，时间门槛控制在三分钟内为优先考量，再来才是精度。精度我希望能。控制在十的负二次方，但是有时候无可避免的会超过，那就算了吧。」「我认为，嗯，我设一个精度底线好了，就是大概是十的负一次方。」見[模擬器的時間門檻與驗證成本](../discussions/2026-11-22-simulator-speed-first-verification.md)。

### Family 去留：合格只比耗時

**規則。** 每點跑完整條 pulse sequence 後，最終 density 對獨立參考解的最大絕對差在該比較的預算內即合格。新取捨依[實驗精度規則](#歷史控制與精度驗收)，已採用 `2e-3` 的比較照原預算判讀。合格的 family 視為一樣準，只比耗時。另一個合格 family 快超過 10% 才算支配。

**例子。** 2026-11-13 討論的 Gaussian gate 序列 S 表中，`shaped_sambe` 的誤差是 `1.45e-4`、耗時 0.27 秒；`period_table` 是 `2.96e-5`、9.0 秒；`gate_cache` 是 `1.10e-5`、54 秒。三者都合格，`shaped_sambe` 支配準一個數量級但更慢的兩個。這組範例不替代後續量表的實測。

**邊界。** 獨占一個適用域的 family，或該域最後一道候選，不參加比較。單 kernel 的誤差與逐段 `budget` 都不是合格證據。要刪掉被支配的 family，先量到支配者在同型態長序列上仍在該比較的預算內，gate 數須遠多於 24，再請使用者看數據決定。量表機制見 [Engine family Pareto table](../../tools/README.md#engine-family-pareto-table)。

**依據。** 2026-11-13，使用者：「我通常是要求在给定最终精度容许范围内尽可能加速，所以通常这种精度差距极小的，我会优先选速度。」並澄清：「你要注意哦，这个精度是最终精度，也就是整个 post sequence 模拟最后的精度，而不是单个 kernel 的精度哦。」對整理後的規則答「確認」。見 [family 支配定義的討論](../discussions/2026-11-13-family-dominance-within-tolerance.md)。

<a id="先保證正確再優化瓶頸"></a>

### 先求效率，正確性事後補

**規則。** 優化先量目標實驗的前後完整 Run，各一次，非排他。不用全表當每項優化的關卡。只改 host 的路徑用同 seed 比逐位結果；數值路徑改動用同 seed 的縮樣與舊路徑比，最大差在 `1e-2` 內可交付。獨立參考能低成本取得就做，不足時標「未獨立驗證」，落地後補。完整量表在整批優化收尾時跑；遇到問題，先重驗受影響的場景。

**例子。** FTF projective 的 Markov 表示在縮樣遷移對照中，逐事件 population 最大差為 `3.272e-3`。使用者同意先交付，獨立精度取證另報。遷移差不能當獨立誤差，也不能證明 raw IQ 分類一致。見 [FTF projective 的 Markov 表示](../discussions/2026-11-22-ftf-projective-markov.md)。

**邊界。** 影響實機路徑、存檔格式或使用者看到的結果時，先驗受影響的契約再落地。FTF projective 的交付例外由使用者裁定，須保留其統計缺口。獨立參考太貴時，可用收斂外推或較小的同型問題估計，記錄假設、覆蓋範圍與不確定度。驗證成本以不超過優化省下的時間為目標。事後若證實誤差超過 `1e-2`，回頭修正或依[精度規則](#歷史控制與精度驗收)裁定時間例外。

既有 guard 與數值合法性照舊。已知不準且修法時程未定的路徑仍先收窄 admission，改走可用候選，修好再放回。Sampled multitone 的[過渡期拒收](../discussions/2026-11-14-sampled-multitone-interim.md)是這類處置。`multitone_gate` 的例外不變：比價選中就建構並使用，不在建構後另驗精度或因精度退回 `dop853`。數值失敗直接回報。其他 family guard、sampled `multitone_rwa` 抽查與動態維度 pilot 不適用這項例外。

**依據。**
- 2026-11-22，使用者：「我想优先把效率提高，再检查正确性」，對五項提議答「A」。同日說：「你要注意参考解的计算时间成本，不要花太多时间在这东西上面。你可以用比如说外推的方法」。見[模擬器的時間門檻與驗證成本](../discussions/2026-11-22-simulator-speed-first-verification.md)。
- 2026-11-14，使用者：「我预期就是首先先确保能全覆盖，然后再来是对于瓶颈场景做优化。」對時程未定先收窄 admission 答「A」。保留已知失準的 guard 處置，不再把全域精度證明當每次優化的前置關卡。原始討論未入庫。
- 2026-11-19，使用者：「我应该说过，不要因为听度而回退。比价的时候决定好就不要再改了，就算结果超标也是能接受。我不想白付时间去尝试错误的路径」。其中「听度」即精度。使用者把範圍限定在 `multitone_gate`。原裁決未附入庫討論，見 Git `7043877d5`。現行路徑見 [Kernel summary](../../lib/zcu_tools/simulate/evolution/README.md#kernel-summary)。

### Evolution 輸出合法 density

**規則。** `Engine.evolve` 在回傳點保證 readout density 符合 continuation 的 Hermitian、trace 與最小特徵值條件。不合法的輸出投影回合法 density，但最大元素絕對改動不得超過本次呼叫的 budget。

**例子。** FTF heralded 的純態經 50 ns 高 gain 雙 tone 段走 CPU DP5(4)，最小特徵值為 −2.4e−6。這次輸出的誤差符合 `2e-3` 的參考解精度目標，卻過不了 continuation 的合法性檢查。投影後兩個重現案例都通過；修正約占 evolve 總耗時的 0.05%–0.09%。

**邊界。** 修正超出 budget 就回報數值失敗，不放行、不改 kernel 選路，也不放寬 continuation 容差。投影步驟、容差與指出 point 的錯誤見 [Accuracy, failures and lifecycle](../../lib/zcu_tools/simulate/evolution/README.md#accuracy-failures-and-lifecycle)。

**依據。** 2026-11-19，使用者選定回傳前投影，修正量超過 budget 就回報失敗，並回覆「其余采用建议」。見 [Evolution 輸出合法 density](../discussions/2026-11-19-evolution-valid-density.md)。

### 固定預算，找適合的場景

**規則。** 按使用者宣告的預算評估算法。既有 `2e-3` 預算保留；新的實驗時間與精度取捨見[精度規則](#歷史控制與精度驗收)。在預算不變的前提下，找出划算的控制形狀、掃描點數、時長和初態。不適用的場景由 guard 或選路退回。

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

**邊界。** 查過與試修後仍找不到能算對的範圍，才收窄或刪除。過渡期另依[效率和驗證規則](#先求效率正確性事後補)。已算得準、卻不服務現有瓶頸的路徑可以移除，例如 [`t6` 的 channel](#t6-只保留-direct)。刪除 family 仍按[去留規則](#family-去留合格只比耗時)讓使用者看數據決定。

**依據。** 2026-11-14，使用者：「我不太想放弃快速的算法，目前的优化需求还蛮大的」。`single_tone_rwa` 去留與 `multitone_rwa` 超標的原始討論未入庫。

### 支援範圍不看使用頻率

**規則。** 物理與介面決定哪些輸入要支援、算對與合理計價；在[約化參數空間](#場景覆蓋靠約化的參數空間)上驗覆蓋，不看使用者常不常用。

**例子。** 量 prepare 價格如何隨重用攤提時，把同一段分別接 1、20、100 次。高重用的輸入即使用得少，也在量測範圍內。

**邊界。** 只有在已有耗時紀錄時，使用頻率才用來排優化先後。過渡期的速度代價照樣量，不以「這類序列較少使用」替代 benchmark。不為省時間而放棄算對高重用的輸入。

**依據。** 2026-11-16，使用者：「你為什麼老是問我常不常用？我常用與你模擬器要不要支援是兩回事」，對使用頻率只用來排優化先後答「A」。`multitone_rwa` 校價需求的原始討論未入庫。過渡期量測及高重用案例見 [Sampled multitone 的過渡期拒收](../discussions/2026-11-14-sampled-multitone-interim.md)及 [t6 只保留 direct](../discussions/2026-11-15-t6-channel-removal.md)。

### 模擬器不替使用者擋實驗參數

**規則。** 就實驗參數而言，模擬器只拒絕實機也會拒絕的程式。程式合法但參數不合理時，照算物理上預期的結果。模型算不準的部分用警告說明，不因參數不當報錯。

**例子。** FTF 規劃 ADC window 時，若整個 template 沒有 readout pulse，event 只有 receiver offset 與雜訊，不報「沒有 generator 候選」。ADC window 與 block 的配對在裝置時間進行；同一 ADC、同一 owner 的 window 可在同一事件中觀測聯集 pulse 留下的場。細節見[FTF 讀出配對](#ftf-讀出窗依裝置時間配對)及 [FTF readout and sampling](../../lib/zcu_tools/simulators/ftf/environment/README.md#readout-and-sampling)。

**邊界。** 型別、範圍、維度等契約外輸入仍在進入點 fast fail。模擬器缺少決定結果所需資訊的合法程式，在編譯時精確拒絕。例如 window 起點生效的 ADC 頻率字沒有具名來源時，模擬器不知道它代表哪個 DDS image，訊息請作者改用具名 ADC config。FTF 的讀出成員與 owner 依[裝置時間配對規則](#ftf-讀出窗依裝置時間配對)。無法表示的事件配置在規劃時拒絕，不回傳缺訊號的 I/Q。算法數值近似仍受[精度 guard](#先求效率正確性事後補)約束。

FTF cavity drive 使相鄰態能量反轉時，原色散模型仍能算，不重排狀態。警告指出受影響範圍與 gain 共同縮放上限；超限結果不在色散模型有效範圍內。計數方式與上限假設見 [FTF cavity drives](../../lib/zcu_tools/simulators/ftf/environment/README.md#cavity-drives)。

**依據。**
- 2026-11-15，使用者覆核 sideband reset 的殘餘光子檢查：「4我覺得有需要嗎？實驗參數設置不當就會發生的問題為什麼模擬器要擋？」對照算並警告答「A」。原始討論未入庫。
- 2026-11-19，使用者同意 cavity Stark 反轉照算並警告，答「1cA」。2026-11-20，designer 依授權代裁共同縮放上限的表示，使用者未覆核。見 [FTF cavity Stark 反轉：從報錯改成警告](../discussions/2026-11-20-ftf-stark-inversion-warning.md)。
- 2026-11-21，使用者定下「只拒絕實機也會拒絕的程式」這條規則，原話是「模擬器要給出合法的預期結果，只要合法，不因為不合理就抱錯」。FTF 的 receiver-only event、相干相加與在途漏配拒絕是 designer 代裁，使用者未覆核。見 [FTF 合法讀出時序](../discussions/2026-11-21-ftf-legal-readout-timing.md)。同日稍後，使用者在「保留 raw 字的 DDS 同餘機制」與「raw 字編譯時拒絕」之間改選後者，答「S」；先前選保留的理由（「那我选，K 吧，……解读弱智的问题。」）經 onetone homophasal 回歸測試證實不成立，table sweep 保留具名來源。見 [raw ADC 頻率字改成編譯時拒絕](../discussions/2026-11-21-raw-adc-word-reject.md)。
- 51 點、21 能階的合法輸入不受 projection 記憶體預算阻擋。案例與驗收見 [Projective read 的輸出改成 Markov chain](../discussions/2026-11-17-projection-markov-chain-readout.md)。

### FTF 讀出窗依裝置時間配對

**規則。** ADC 負責 ADC 時間轉換至裝置時間的單一換算。`simulate.adc` 匯出 `adc_to_device_time_ns(adc_time_ns, *, time_of_flight_ns)`，計算 `t−τ`；ADC solver 與共用 planner 呼叫同一函式。ADC 保留原始 sample grid、demodulation phase 與 trigger 順序。Planner 對每個 receiver 換算 window 端點，再以裝置時間和 block interval `[start,next_start)` 判交集，並包含 ringdown。最後相交的 block 擁有該 window；最後一個 block 延伸至 point 結束。每個 point、ADC、owner 的有場 event，使用其任一 window 相交 blocks 的聯集；每個 block 的 pulses 只收一次。每個 window 保留自己的交集集合。其他 ADC 或 owner 的 pulse 不會進入此 event 的聯集。

**例子。** 若同一 owner 的前窗與 blocks 62、63 相交，後窗只與 block 63 相交，兩窗仍各自保留交集與輸出 column；同一 ADC／owner event 以 blocks 62、63 的 pulses 建立一段場歷史，因此 block 62 的場可以延續到後窗。這不是跨 ADC 共用 pulse list。各 ADC 仍按自己的 receiver delay、grids 與 pulse union 讀出。

**邊界。** Projective collapse 仍在 readout pulse 的裝置時間起點，不隨 ADC delay 移動。跨投影 window 整窗使用最後相交 block 的 owner state，這是事件層近似；FTF 每個 ADC event 仍從 vacuum 起始，不帶入前一事件的殘場。兩項近似及量得的 I/Q 比較見 [FTF 已知近似](../../lib/zcu_tools/simulators/ftf/environment/README.md#known-approximations)。整窗早於第一個 block 的 ADC window 保留 receiver-only grid，只輸出 receiver offset 與 noise，不新增 projection；planner 把 grid 放到該 ADC 沒有 field grids 的既有 event，以維持跨點布局。若現有 event topology 無法表示該 grid，planner 精確指出 ADC、trigger 與原因後 fast fail。FTF 目前沒有 active reset；若新增依賴讀出結果的即時回饋，必須重新評估去程與回程延遲對 qubit 的影響。

**依據。**
- 2026-11-21，使用者授權 designer 處理細節，條件是計算正確、機制統一、好維護、知道近似且不造成 bug。原話：「读书时间，这种东西，你只要能确保计算是正确的，然后机制是统一的，好维护，你知道存在这个东西，而且不会造成什么 bug 之类的就好了。那具体上做细节，我没什么意见。」
- 2026-11-21，使用者選擇 window 跨投影時沿用單一 owner state，並確認投影時刻不移動：「R5 的话，这个算是模拟的一种误差。老实说，我不确定这是不是可以低成本解决的问题。比如说，可能只记录剩余光指数之类的，我不是很确定有没有好解法。 然后跨2个投影点的话，我觉得 A 就好了。没错，投影时刻只影响 ADC。 Qubit 不影响，这个是合理的，因为就算去程到 Qubit 有 delay， 那对于非 active reset 的操作来说是没有差的。我目前也没有设计 active reset。」
- 同 ADC／owner window 的 event pulse union 是 designer 依上述授權補定，**代裁，使用者未覆核（Review: pending）**。依據與選項比較見[同 owner 的讀出窗共用脈衝歷史](../discussions/2026-11-21-shared-owner-readout-membership.md)。
- 整體裝置時間模型、單一 owner 近似與 R5 延後見 [FTF 讀出窗依裝置時間配對](../discussions/2026-11-21-ftf-readout-device-time-pairing.md)。ADC 獨立 timeline 的既有邊界見 [ADR-0083](0083-program-macros-and-timelines.md)。

### 模擬器不從程式內容推斷實驗類型或硬體架構

**規則。** 選路、精度與維度不依「程式內容被辨認成哪種實驗或架構」切換。使用者宣告精度，沒宣告就用預設 budget。

**例子。** 有 flux 脈衝的程式照用預設 budget；CZ 也可能從 charge line 打。

**邊界。** 使用者宣告的精度和直接量到的數值可以改變選路。動態維度的 pilot 與 `drop(k)`、sampled `multitone_rwa` 抽查及各 family guard 都可以靠直接比對計算結果或讀數值指標來決定選路或維度，不看實驗名稱、脈衝種類或架構。

**依據。** 2026-11-18，使用者：「这不合理，CZ gate 可能也是从 charge line 打，而且模拟器理应不知道现在的架构是 FTF 还是只有单个 F 还是什么的。」見 [FTF 動態維度](../discussions/2026-11-18-ftf-dynamic-dimension.md)。

### FTF Stop 用合作式檢查點，設計上 5 秒內

**規則。** FTF Stop 不設強制上限，也不設延遲 gate。放點依據是正常參數下，相鄰兩次取消檢查之間的區段在 5 秒內結束。FTF 在每層展開前及每個 projective node 推進前檢查取消。Engine 在每批 geometry 完成後、geometry 合併後、chunk 邊界，以及 WorkPlan population 的 gate inventory 完成後、single-gate pricing 或 usage recording 前檢查。Binding 在 pacing 等待中檢查。當輪需要 raw 時，ADC 在 raw 入口、每個 logical event、每個 point group 與交付前檢查；Binding 在 required-raw 工作完成後、clock credit 前再檢查一次，SoC 只在成功後發佈並前移 index。

**例子。** 改為 chain 之前，`chain/amp_rabi` 的單點 caller 在第 8 次 `Engine.evolve` 的 6,614 個 geometry batches 中途送 Stop；第 3,307 批結束的檢查讀到旗標，沒有開始下一批。該次前綴的最長查詢間隔是 0.578580 秒。WorkPlan 的真實 caller 在 gate inventory 後讀到 Stop，最長前綴查詢間隔是 1.057840 秒；WorkPlan 關閉，沒有記錄該次的新 usage。必要 raw 的 Stop 測試在第一個 physical point group 結束時讀到取消，零輪沒有發布。這些都是局部診斷值，不是端到端延遲保證。實作接縫見 [Evolution README](../../lib/zcu_tools/simulate/evolution/README.md#workplan-and-cache-ownership)、[ADC README](../../lib/zcu_tools/simulate/adc/README.md) 與 [simulation README](../../lib/zcu_tools/program/v2/simulation/README.md)。

**邊界。** 5 秒不是驗收門檻。改為 chain 之前的離線、非排他診斷未達整體設計目標；改用 chain 後未重測端到端延遲。量測收據未入庫，不把單點或前綴外推到完整工作量。

- **已補檢查（附診斷值）。** 單點 #8 `plan` 原量 5.405 秒。重放外層 5.159 秒，其中 geometry batches 4.344 秒、merge/finish 0.815 秒；每批結束查詢，merge 後仍查詢。真實取消前綴最長間隔 0.578580 秒。該次只驗到第 3,307／6,614 批，未跑剩餘批次、merge、WorkPlan 或後段。
- **已補檢查（附診斷值）。** 單點 #9 `plan` 原量 7.494 秒。重放外層 6.813 秒，其中 geometry batches 5.967 秒、merge/finish 0.847 秒；沿用逐批及 merge 後查詢。#9 走完 geometry 進入 WorkPlan，真實取消由後一段驗證；不能把該前綴當成 #9 geometry 的最大間隔。
- **已補檢查（附診斷值）。** 單點 #9 `create_work_plan` 原量 7.160 秒，重放 5.649 秒。新檢查位於 gate inventory 後、single-gate pricing 與第一筆 usage recording 前；真實取消前綴最長間隔 1.057840 秒。該次沒有執行後續 single-gate pricing、planner 或本輪後段。
- **GUI 條件下不出現。** 原清單的空 Numba／ZCU cache、warmup 關閉首個 cold chunk 為 5.412／8.460 秒。保留磁碟 cache、`warmup=True` 的離線 GUI 設定樣本中，首個 chunk 為 0.007628 秒。這只代表該 retained-cache 首段，不代表其他 cache 狀態、後續 chunks 或 GUI 終態。
- **GUI 條件下不出現。** 同一個 retained-cache 樣本的首個 production WorkPlan 為 0.002580 秒；原空-cache、warmup 關閉的 cold WorkPlan 為 6.055 秒。這只覆蓋一個 segment row，不代表 #9 的大批次 WorkPlan。
- **GUI 條件下不出現 >5 秒，但空 cache 仍未達成。** Retained-cache 樣本的 `wait_ready` 為 1.962915 秒。另一個 `warmup=True` 的空-cache 樣本等待 115.738108 秒，期間沒有查 Stop；它不是 retained-cache GUI 工作路徑的重測。若要改啟動等待或取消行為，須另行取捨。
- **仍未達成（原因：尺寸未覆蓋）。** 在診斷 commit `0aeff3a658ddc1a02f656943f54d1e0c77cbfa98`，以 `ChainAmpRabiExp` 跑舊低 SNR preset 的工作副本，覆寫 projection floor 為 `1e-5`。這次非排他 CPU 診斷使用空快取、關閉 warmup 和 pacing，只跑原 Form 第一個 gain 點（−0.3）的首輪，沒有送 Stop。該輪有 554 outcomes、24 states、16 reads、ADC 0、每窗口 304 samples、100 reps。Engine 後的 compact populations、讀出分布、ADC 準備、statistics 準備、raw 準備、compact draw 與成功收尾分別量到 0.045507、0.001328、0.015142、0.006093、0.000057、0.001970、0.006212 秒；涵蓋前六段的相鄰查詢間隔為 0.205343 秒。51 點首輪與 floor `1e-9` 尺寸未覆蓋，不用這些數字宣稱達成或外推。
- **已補檢查（附診斷值）；仍未達成（原因：冷段與大尺寸未覆蓋）。** 必需 raw 現在在 ADC 的入口、logical event、point group、交付前及 SoC 發佈前查詢。上述單點診斷中，raw 的原相鄰查詢間隔為 5.035913 秒；送 Stop 的樣本中，event 內第一次查詢到第一群結束的查詢為 3.216835 秒，該輪沒有交付、完成輪數為零。另一個未送 Stop 的冷樣本，同一段查詢間隔為 5.296475 秒，raw 全段為 5.392256 秒；收據未分解 event preparation、群內執行與 JIT。不能據此宣稱冷 raw 已符合目標。51 點首輪、floor `1e-9` 與 GUI retained-cache raw 都未覆蓋。
- **仍未達成（原因：只量到外層）。** 上述單點診斷的首個 evolve 有兩個未分解區段，分別為 5.483342 秒與 8.105711 秒。收據沒有定位子段，不能據此另加檢查點或改寫 retained-cache 結論。

通用 Cancel 與 terminal 所有權遵循 [ADR-0066](0066-operation-lifecycle.md#取消與互動)。當輪 raw 的資料 owner 與 query 邊界見[Integrated raw 規則](#integrated-raw-由環境生成介面由-program-擁有)及 [required raw 討論](../discussions/2026-11-21-required-raw-stop-publication.md)。

**依據。** 2026-11-21，使用者對 Stop 上限答「3C，但設計上5秒內」，並說「我认为能及时中断也是一个挺重要的功能。」具體放點、覆蓋清單與落地檢查由 designer 依授權代裁，使用者未覆核。原始檢查點討論見 [FTF Stop 討論](../discussions/2026-11-21-ftf-stop-checkpoints.md)；必要診斷數字與尺寸限制已逐項寫在本節，未入庫的收據不作永久文件依賴。required raw 的 optional protocol 與發佈語意是 designer 代裁，使用者未覆核（Review: pending），選項與理由見 [required raw 討論](../discussions/2026-11-21-required-raw-stop-publication.md)。

### 模擬器帶通用調試介面

**規則。** 每個模擬器提供同一組調試介面，定位與驗證工具只依賴這組介面。共同欄位有型別；模擬器自行決定哪些額外值放進 dict。

**例子。** FTF、single 與 six 每次 acquire 以共用 writer 寫一行總 wall、編譯、演化、讀出與各 family 的分項，模擬器特有的欄位放在 `extras`。見[共用 telemetry](../../lib/zcu_tools/simulators/README.md#acquire-telemetry)。

**邊界。** Acquire telemetry 的量測 commit 與 dirty 標記由啟動擷取的呼叫端提供；lib 不查 Git。未提供的值寫 `null`。共用介面可以帶中性的模擬知識，不把 FTF 的腔記憶或 six 的 scatter 放成共用 hook。調試者就是模擬器維護者，介面可暴露內部資訊，不另為外部使用者承諾穩定 API。[模擬器的 debug capture](../../lib/zcu_tools/simulators/README.md#debug-capture)共用錄製與體檢介面，保存凍結輸入與實際選路。直接用 `ModelSource` 呼叫引擎仍能錄製與體檢，但沒有前端和讀出的耗時。

**依據。** 2026-11-16，使用者：「優化手段要看模擬器是否實作功能，應該是要規範模擬器應該帶有用於調試的通用接口，類似基類或protocol的概念」。對資料形狀答：「可以，但我認為這個資料不一定要完全強型別，可以用多餘資訊的接口，比如dict屬性，放值得記錄的資訊」。對使用者答：「可以，調試者應該就是模擬器維護者，我沒有意向去解耦合兩者責任」。對共通部分答：「中性的模擬知識是沒有問題的。場景特化的才不建議」。原始討論未入庫；實例見[共用 acquire telemetry writer](../discussions/2026-11-16-shared-acquire-telemetry.md)與 [Engine capture 的逐輪選路事件](../discussions/2026-11-17-engine-capture-events.md)。2026-11-22，使用者對由呼叫端提供量測 commit／dirty 及收窄未完成列的規則答「可以。」見[舊實測數字當現況](../discussions/2026-11-22-stale-measured-numbers.md)。

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

**例子。** ADC 擁有 flat 腔場 span 的精確結果 memo，負責 lossless key、有界容量與回傳陣列 ownership。同一次求解產出二階矩與各 state 的端點場，兩種產品共用容量與壽命；無法證明等價時分開求解、不收錄。FTF 的 projective 與非 projective 兩種 span composition 都經 `CavityAverageCache.span` 取得兩種產品。每次 composition 建立一個 owner，兩種 composition 用同一個 512 MiB 容量，供該次 composition 的各點與續演化節點串行共用；ADC 全庫預設仍是 64 MiB。正常返回或拋出 exception 時都清理。`CavityDrive` 只轉交 owner，不組 ADC key。FTF 為每個節點選態、建 shift。改用 chain 之前，`chain_length` 2.2 µs 的 projective 前綴有 29,921 次 span，其中 19,346 次命中別的 history 已解過的 key。這是舊路診斷，不是現行命中數。詳細 key、容量依據與清理見 [ADC README](../../lib/zcu_tools/simulate/adc/README.md) 與 [FTF cavity drives](../../lib/zcu_tools/simulators/ftf/environment/README.md#cavity-drives)。

**邊界。** Memo 不留在 Engine、global 或長駐 source。FTF 逐節點建 Stark／dephasing shift。同一節點的演化可把場傳給下一個 span；遇到 projective read 時，下一個節點從真空開始。非 projective composition 不受這項重設影響。Shaped/callable 輸入與主分支選擇不進 memo。跨 package 的依賴維持 [[0072]] 與 C8、C9、C12，不新增依賴方向或放寬 `ignore_imports`。這些 contract 不檢查內部子模組的匯入路徑；package root 的公開入口由介面與 caller review 維持。

**依據。**
- 2026-10-18，使用者：「之後我的規劃是讓你先整理目前模擬引擎，使其對外是黑盒子，對內是實質偶合，決策分類有全部算法的知識，算法部分內聚，以不影響優化與可讀性為主要指標，移除已經沒用的邏輯，將研究成果以及嘗試紀錄到repo文檔中供以後參考。然後就可以開始planner的重構，使各優化算法的實作張力能以自然的表達」。原裁決未附入庫討論，見 Git `ee1eeaa8d`。
- 2026-10-21，使用者對二階矩 memo 的方案答「前兩個採用建議」。原裁決未附入庫討論，見 Git `ffe98f8f7`；[MIST 腔場重用的先例整理](../discussions/2026-11-21-mist-cavity-reuse.md#先例與立場)列出這項裁決的適用範圍。
- 2026-11-21，designer 依使用者授權「我授权你的范围除了 push 还有真实硬体之外都可以做」代裁端點場擴充。使用者 2026-11-21 追認：「其余的第二批追认。」正常參數重測為 281.5 s 與 163.3 s；兩次 commit 不同，不能把差值全歸因於快取。二階矩單獨重用的上限及比較見 [MIST 腔場重用](../discussions/2026-11-21-mist-cavity-reuse.md)。
- 2026-11-21，使用者對模擬器下一批加速的選項答「6A」，選 projective 腔場的 span 重用。對本節原邊界句「projective 只重用既有的 flat 二階矩，不收各 state 端點場」答「2S，我對於模擬器沒那麼多限制，好維護與模擬效率是唯二考量，其他都只是建議」，改為 projective 與非 projective 走同一條 ADC span 合併解。512 MiB 的選法（再加倍多出的命中少於跨 history 重複機會的 10%，且 process peak RSS 增量不超過 512 MiB）與兩種 composition 共用一個容量，是 designer 代裁，使用者未覆核。見 [Projective 腔場改走 ADC flat span 合併解](../discussions/2026-11-21-projective-span-reuse.md)。

### t6 只保留 direct

**規則。** `t6` 在 constant 單 tone 域只走 direct，保留原 guard、prepare-key 身份和失敗就 raise 的語意。不建 reusable channel，不做 Fourier playback，也不為 channel 收完整 acquire key inventory 或比價。

**例子。** 每列 density 經共用的 endpoint solver 在 H0 旋轉座標積分，用完整 secular bath 與原 budget-scaled 局部容差。實作入口見 [Kernel summary](../../lib/zcu_tools/simulate/evolution/README.md#kernel-summary)與[共用數值 owner](../../lib/zcu_tools/simulate/evolution/README.md#algorithm-ownership-and-reuse)。

**邊界。** 其他 family 的 period channel 不變。未來若出現高重用工作負載，再依新證據評估，不保留相容性分支。

**依據。** 2026-11-15，使用者裁定 `t6` 只保留 direct，答「2B」。量測與方案比較見[移除 t6 channel 的討論](../discussions/2026-11-15-t6-channel-removal.md)。

### 損耗漂移下的結構與 preparation lifetime

**規則。** Single、six 與 FTF 都支援損耗漂移，預設關閉。每次 acquisition 開始取一次有效損耗，整次固定。只在 acquisition 之間命中，不足以保留沒有同次命中的快取層。Engine 不保留跨 evolve 的數值 preparation 或 exact result，source 不保留跨 preparation 的 ADC solver。Engine 的 routing-facts cache 只以 inventory 與 pricing facts 驗證 caller-owned preparation 的當前路由，不存傳播結果。

**例子。** Single 的物理模型擁有損耗無關的結構，experiment 只組合結果。損耗改變時，source 保留該結構，關閉舊 ModelSource，丟棄完整 models 與 preparation，按本次固定的 losses 建 rates。溫度、DC、truncation 等結構依賴改變時才丟棄結構。FTF 漂移開啟時不拆結構，按每次 acquisition 的 effective settings 重建完整 facts，包含 eigensystem，並清掉舊的完整 preparation；points 與 rounds 共用這次的 losses。漂移關閉時不採這套重建策略。見 [Single reuse and lifetime](../../lib/zcu_tools/simulators/single_fluxonium/environment/README.md#reuse-and-lifetime)、[FTF acquisition clock and loss drift](../../lib/zcu_tools/simulators/ftf/environment/README.md#acquisition-clock-and-loss-drift)與 [FTF preparation reuse](../../lib/zcu_tools/simulators/ftf/environment/README.md#preparation-reuse)。

**邊界。** Rates、dissipative propagator、density 與完整 preparation 不跨不同 losses 重用。同輸入的不可變完整 preparation 可按設定、losses 與路由條件重用。FTF 的 complete preparation LRU、envelope fingerprints 與 acquire-local `CavityAverageCache` 仍可重用。即使不開漂移，FTF 重複 acquisition 也可能要重算。中途 projective readout 由環境在 read 起點投影並續演化；分支 density 只活在一次 span preparation，不作下一 shot、round 或 acquisition 的初態。能重用的是 FTF、single 與 six 的不可變初始分佈、轉移表和 conditional sums，不是抽樣後的 herald 結果。已交付 raw source 仍持有自己的不可變抽樣輸入。抽樣 RNG 與 raw 契約不因這項 lifetime 決定而變。數值、設定與 truth 的 owner 見 [[0072]]；C8、C9、C12、C19 不添依賴或例外。

**依據。**
- 2026-10-27（UTC），advisor 對快取分工的原話（不作現行契約）：「保留 single 的 _models／_prepared 與 FTF acquire-local 的 CavityAverageCache；拆 _facts 為結構與 rates。」同日使用者裁決：「我預期都會加入漂移機制」、「沒有不漂移的時候，我考慮C，應該是說，用不到的就可以移除」。原裁決未附入庫討論，見 Git `c6ff3f173` 的裁決與量測紀錄。
- 2026-10-27，advisor 依使用者授權「B你可以自己開task，不用我看過」訂定 FTF 重建策略。原裁決未附入庫討論，見 Git `257e8cc3a`；未見使用者覆核這項策略的紀錄。
- 2026-10-28，使用者以「A我認為可以完整版。」授權 single、six、FTF 的中途 projective readout。這句決定支援範圍，lifetime 沿用本節規則。原裁決未附入庫討論；[模擬環境草稿](draft/simulation-environments.md#acquire-接線與-projective-readout)也記有這項裁決。

### Projective read 的 Markov chain 表示

**規則。** Single、six 和 FTF 在 triggered nonterminal read 的起點作 DC 能量基底的 rank-1 投影。FTF 把投影後續演化的入射腔場設為真空，同點同層只為來源機率達 floor 的 DC 能階各求一次轉移，不列舉 outcome history。這三種環境把第一次讀出的 populations 與後續各事件的轉移表交給 ADC 的 `ProjectionChain`。ADC 的 raw／trace 沿同一發的 chain 抽樣，保留 herald 與 terminal 的配對。Compact integrated 用逐態 flow counts 算出各事件的 occupancy，再以累積邊界編成共用 category 軸。Bins 給各事件的邊際計數，不是物理狀態的跨事件 joint。各讀出的 noiseless total 由該事件的 counts 乘條件均值後加總。

**例子。** 某點某層的來源態機率低於 floor，就在該點使用 identity 轉移列，即留在原態。不丟質量，不重新正規化，也不看其他點的機率。Floor 為零只移除這項近似。若兩次讀出間態必翻轉，compact 仍只給兩次各自的 g/e 計數；raw／trace 則保留每發的 g→e 或 e→g。見 [ADC README](../../lib/zcu_tools/simulate/adc/README.md) 與 [FTF environment README](../../lib/zcu_tools/simulators/ftf/environment/README.md#projective-migration-evidence)。

**邊界。** 在同一模型且不套 floor 近似時，沿轉移表和枚舉完整路徑再抽樣給出同一物理路徑分佈。這不承諾同 seed 逐發相同，也不把改了腔場、floor 或數值選路的跨版本結果視為已驗相等。Compact bins 不與 raw shots 配對，分類仍讀逐發 IQ。FTF 的跨版本分類與 Program 歸約前 compact 統計尚未驗證。Chain 用量隨點數線性成長，raw 另隨 reps 成長。無 projective read 的路徑及既有 outcome-mixture 入口不變。

**依據。**
- 2026-11-16，使用者要求「模擬器不該有 OOM」，並授權「OOM你就改designer決定」。2026-11-17，使用者以「兩筆追認」覆核 single／six 的 Markov chain、逐點 floor、同 seed 的抽樣差異、人工 bins 與不設 projection 記憶體預算。見 [Projective read 的輸出改成 Markov chain](../discussions/2026-11-17-projection-markov-chain-readout.md)。
- 2026-11-21，使用者：「我自己估计的事情是，由于每一次中间的演化，也就是两次 real 间的演化都长得一样，那其实只要算一次演化的转移矩阵，就可以完成模拟了。」「既然腔場礙事，那就每次開始都丟棄腔場如何？」對丟場並改用 chain 答「採用建議」，取代 FTF 須保留 history 的舊邊界。FTF 的 compact marginal 契約是 designer 代裁，使用者未覆核。見 [FTF projective 的 Markov 表示](../discussions/2026-11-22-ftf-projective-markov.md)。

### Integrated raw 由環境生成，介面由 program 擁有

**規則。** `program.v2.simulation.RawShots` 定義延遲產生完整單發 ADC sums 的介面。環境透過 `IntegratedRound` 一起交付統計資料與 raw source。Program 只在單發 processing 或 getter 需要時呼叫 source，不重跑 backend。`simulate.boundary.RoundStatistics` 只有資料，不定義 materializer、Protocol 或 callback。Program 另定義可選的 `CancellableRawShots` 能力，使用 `materialize_cancellable(reps, avg_level, *, cancel=...)` 對當輪必需 raw 借用唯讀取消查詢；source 不保存 query。唯一的 `Cancelled` 由 `simulate.boundary` 定義，`simulate.evolution.Cancelled` 保留相同類別身分的匯入入口。

**例子。** FTF 的環境交付由 ADC 產生的完整 raw，包含物理訊號、receiver noise 和取整排列。當 Program 的 threshold、mask 或 reducer 需要本輪逐 shot 資料時，SoC 將既有 acquire query 傳給支援 cancellable 的 ADC source；抽樣在入口、logical event、point group 與交付前檢查。成功的 shots 只抽一次，Program 與 SoC 各持有獨立可寫陣列。介面見 [Compact timelines and statistics](../../lib/zcu_tools/program/v2/simulation/README.md#compact-timelines-and-statistics)，取樣見 [ADC README](../../lib/zcu_tools/simulate/adc/README.md)。

**邊界。** ADC 與 `simulate.white_noise` 各自產生完整 raw，以結構型別符合 `RawShots`，不 import program。只有明確需要當輪 raw 時才傳 query；一般 compact statistics 路徑不抽 raw，已完成輪的 getter 也不帶當輪 query。沒有 `CancellableRawShots` 能力的來源維持兩參數 `materialize(reps, avg_level)`；SoC 只在同步呼叫前與交付前查詢，不宣稱能中斷來源內部工作。停止或 raw 驗證／抽樣失敗時，SoC 不替換上一個已完成輪的 source 或 buffers，也不前移 index；真實 raw 錯誤照常傳出。成功物化不表示 Program 的 processor 或 hook 也成功；發佈後的 Stop 或 processor 錯誤不回滾已發布 transport。Deferred getter 仍可從 retained source 重新 materialize，raw 與直接平均維持相同期望，直接平均保留 conditional moments 與既有 RNG stream，不做逐 shot 工作。依賴方向維持 C8、C9、C12、C19，不加 `ignore_imports`。

**依據。** 2026-10-20，使用者答「1-1」，並問：「基本認同丙，但我質疑有需要進core嗎？」介面放到 program 後答「同意」。2026-11-21，designer 依既有授權代裁當輪可選取消能力、required-raw 發佈邊界與 QICK／clock 接線，**使用者未覆核**；使用者的既有方向是「3C，但設計上5秒內」及「我认为能及时中断也是一个挺重要的功能。」比較與未覆蓋條件見 [required raw 討論](../discussions/2026-11-21-required-raw-stop-publication.md)。原裁決未附討論紀錄。

### ADC raw plan 的 owner 與壽命

**規則。** ADC 擁有 `IntegratedPaths` 等確定性 raw plan，也判斷 prepared batch 的身分。FTF 與 standalone single-fluxonium 的 acquire session 各持有一個 owner。每輪 integrated bind 傳入當前的 `PreparedRawShots`。換物件就退役舊 plan，A→B→A 不復活第一個 A。只有 materialize 到達某個 logical event 時才建該事件的 plan。不快取 RNG、counts、抽樣路徑、noise、結果或 Stop query。ADC 同步查找、發佈與退役，建構和求值不持 memo lock。兩個環境用既有的 `SimulationBinding.on_end` 在正常結束、Stop 或拋錯時退役 plan，即使診斷關閉也執行。Direct caller 沿用 matching finish、換 request 或 close。舊 raw source 用原輸入與 identity 冷重算，不讀其他代或回填退役的 memo。Plan 不進 source preparation、LRU 或 process 全域。

**例子。** `chain_power` 的固定 fixture 在一次 acquire 取 10 個 gain 點，共 320 輪。同一 prepared batch 的各輪重用已建立的事件 plan，換 prepared 物件才重建，不以 gain 點當失效邊界。用同一 fixture、非排他 1 worker 比較 `e602f0dc6` 與 `a4cd9f1bd`，`chain_power` core 為 350.160 → 165.755 s，materialize 為 343.737 → 159.082 s；Auto core 為 257.464 → 106.475 s。

**邊界。** 只在同一 active batch 的輪間重用。不採用跨內容的數值鍵或其他快取的 byte cap。若要擴大重用範圍，先驗證等價鍵及保留量。Six-fluxonium、decimated 與無 owner 的 `bind(identity)` 照舊。同 seed 輸出逐位元不變，不改既有 Stop 檢查點與 raw 發佈。

**依據。** 2026-11-22，designer 代裁，使用者未覆核。使用者當天說「好維護與模擬效率是唯二考量，其他都只是建議」與「b，但我重點想放acquire內」。取捨與實測條件見 [ADC raw plan 由誰持有](../discussions/2026-11-22-adc-raw-plan-acquire-owner.md)；原始工作檔未入庫。

## 取捨

內部接縫隔離排程與數值表示，不保證換 kernel 時 dispatch 完全不用改。Guard 拒絕可以換候選，prepare 或 apply 的數值失敗直接傳給呼叫端。事後 accuracy exceedance 只記錄並警告，不因它修正 density、重算或重新分類；回傳 density 的合法性是[另一項檢查](#evolution-輸出合法-density)。

引擎不提供實驗 planner 的相容層，[[0077]] 不定義現行執行契約。本決策不保證消除所有中間結果的重算。若以後再設計 planner，資料壽命與重用要另行設計。

## 現行契約與研究依據

- [Evolution README](../../lib/zcu_tools/simulate/evolution/README.md) 擁有現行 API、決策圖、數值 owner 與失敗語意。
- [模擬環境草稿](draft/simulation-environments.md) 記錄 environment、model、evolution 與 ADC 的組合責任。
- [Planner evidence 與固定成本討論](../discussions/2026-10-18-planner-evidence-and-overhead.md) 保存已移除 planner 的實驗結果。
