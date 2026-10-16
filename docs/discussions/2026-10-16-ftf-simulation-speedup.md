---
date: 2026-10-16
status: active
superseded_by: null
related:
  - docs/adr/draft/simulation-environments.md
  - docs/discussions/2026-10-15-offline-propagator-library.md
basis:
  - "repo@e7f32506c：prototype 量測用的 measure-simulation-environment 06 lane commit"
  - "repo@0130b613c：撰寫時的 main HEAD"
  - "本機：8 核 CPU，與其他 agent 共用；Intel Arc OpenCL（FP64）；Python 3.13.11、NumPy 2.4.6、SciPy 1.17.1、Numba 0.65.1、OpenBLAS 0.3.31、pyopencl 2026.1.4"
---

# FTF 模擬加速：快速路徑、shot 統計、分段快取與分派決策圖

## 結論

twotone flux-dependent spectrum 慢，主因不是單點物理慢。acquire 框架把同一份計算重複了 reps × rounds 次，這才是主要成本。

使用者的實際規模是 301 flux × 2001 freq，reps × rounds 最多 1000 × 100。只做物理加速時，這個規模推估約 8.1 天：

| 成本來源 | 推估時間 |
| --- | ---: |
| 逐 shot 抽樣 | 146 h |
| grouping | 23 h |
| 每個 round 重算物理 | 20 h |

改成每個 sweep 點只算一次、再用統計方式產生雜訊後，推估降到 13 分鐘到 2 小時。區間的寬度取決於逐 execution 的 bookkeeping 能不能一併移除。

採用的整體方向：

- **快速算法成為預設數值方法。** 使用者：「同意，這個task就是來優化算法的」。
- **精度判準。** 最終 readout basis 的 population 誤差 ≤ 1e-2。phase-sensitive 誤差（最低四態之間的 |Δρij|）也適用同一門檻。實驗最長約 10 µs，驗收要涵蓋誤差累積。
- **compact acquire。**
  - 每個 sweep 點只解譯一次 timeline，初態只演化一次。
  - reps 與 rounds 都是重複計算。每個 (round, 點) 只抽一次統計：shot 數用 multinomial 抽樣，表示投影雜訊；總和再加上 Gaussian(0, reps·Σ)。
  - raw 只在需要時，從同一個 realization 延後生成，型別維持 int64。
  - timeline 若隨 rep 改變，直接報錯。
- **分段分派。**
  - 每段依內容選路徑，詳見[決策圖](#分段分派決策圖)。
  - reps 內所有 sweep 點都預設批次計算。批次以 kernel 為單位，與段的位置和順序無關。
  - 使用者要求之後修改算法時，都以這張決策圖為基礎。
- **分段傳播子快取。** 多 gate 序列的每種 gate 只建一次與輸入態無關的傳播子，序列結果只剩矩陣乘法。RB 規模從約 20–40 小時降到約半分鐘。
- **兩 tone。** 採用 Floquet 等效：stark tone 先換成等效靜態生成元，probe 再當成弱驅動計算。
- **平行化。** 只用 numba，不做 GPU 與 Rust。只開一層平行，第一批不做 process pool。
- **刪除既有機制。** FTF 一定有耗散，所以離線 U 表永遠不會被用到，一併刪除。exact density record 也刪除。

正式實作還沒開工。使用者的指示是：「採用建議，先別開工」。

## 問題

使用者回報：「Twotone flux dep spectrum太慢」。

現況量測，base 為 06 lane commit：

- 一個 128 ns 的 twotone 點要 2.9 s。evolve 約 18 s/µs，readout 約 3 s/µs。
- 拿掉 `max_step` 上限後，步數幾乎不變。DOP853 的 tolerance 把步長壓在約 0.019 ns，原因是 lab frame 下 H0 約 7.7 GHz 的 coherence 振盪。所以放寬步長沒有用。
- frontend 對每個 shot 都解譯一次 timeline，每個 execution 約 73 µs。在使用者的規模下，只建 timeline 就要約 12.5 小時。

使用者的工作方式：先做 prototype 確認可行，再決定做哪些。原話是：「建一個新任務，先做調查與prototype，建議讓一些pi subagent實作來確認」。

使用者也兩度修正 prototype 的做法：

- 規模：「我只是要看可不可行而已，你規模跑那麼大，每次都跑舊路徑根本浪費時間，我要盡快確認優化方案」。之後的 prototype 改成小 workload、每項只跑一次、重用 baseline。
- 平行度：「你為什麼不多派幾個subagent?」。之後彼此獨立的評估項目，各派一個 subagent 並行。

## 考慮過的選項與比較

### 精確等價的優化

| 選項 | 結果 | 處置 |
| --- | --- | --- |
| readout 響應快取，加 constant readout 的 expm 精確解 | response 快 115–541 倍 | 採用 |
| 跨 round 重用演化 | 預設環境已注入 library，收益低 | 由 compact acquire 取代 |
| dense superoperator RHS | 只有舊路徑 0.44–0.52 倍的速度 | 否決 |
| 同 flux 多 freq 批次 DOP853 | 1.03–1.05 倍 | 否決 |

### 演化算法

| 選項 | 結果 | 處置 |
| --- | --- | --- |
| interaction picture | 1.08–1.40 倍。drive 經 U0 共軛後仍帶有 Bohr 相位 | 否決 |
| flux：H 仿射分解，bath 每個 interval 只設一次 | 4.5–4.8 倍，與舊路徑的 density 差 ≤ 5e-14 | 採用，第二批 |
| Liouville space 4 階 Magnus | 步長要縮到 0.03 ns 才通過，比舊路徑慢 | 否決 |
| 週期傳播子 | 13–1172 倍；10 µs 仍通過；強 drive 每週期需 8 步 | 採用，第一批 |
| 靜態 idle 用解析傳播子 | 97 倍 | 採用 |
| flux 值量化 | 30–79 倍，跨點 cache 命中率 87–91% | 採用，第二批 |
| 弱驅動二階 Dyson 加 reduced RWA | 每個 flux 2001 freq 約 1.9 s，95 個驗證點全部通過 | 採用，twotone 的主路徑 |

**flux tail。** 一階低通的 tail 理論上永遠不會歸零，flux 就一直在變，有 flux pulse 時分段演化無法使用固定生成元。使用者的裁決：「要做，不然分段演化在有flux pulse下就無用了，我會選flux絕對值，但還是以好實作與算的快為衡量辦法」。

**量化一律開啟。** 使用者：「同意，我認為甚至是總是量化，不給設定」。

**格點寬度 δ。** 比較了兩個值：

| δ | 最大 phase 誤差 | 對 1e-2 門檻的餘裕 |
| --- | ---: | ---: |
| 3e-3 Φ₀ | 3.9e-3 | 約 2.5 倍 |
| 1e-3 Φ₀ | 7.7e-4 | 約 13 倍 |

採用 1e-3 Φ₀，因為較大的餘裕可以容納多個 pulse 疊加後的誤差累積。

**週期傳播子提前到第一批。** 使用者要做 amplitude 與 length Rabi，強驅動是常態。原話是：「B，我需要做amplitude/length rabi」。兩條路徑的成本差異：

- 週期傳播子：每個像素約 0.2–0.3 s，和 pulse 長度無關。
- numba DOP853：成本和長度成正比。

### 精度門檻的演變

- 起初的門檻是 population ≤ 1e-3，理由是它低於 1e4 shots 的 shot noise。
- 使用者補充了兩次：「但要注意誤差累積，實驗長度大概10us內」、「我可能太嚴格，最後population到1e-2的誤差即可」。
- phase-sensitive 誤差也一起判定，因為 Ramsey 這類序列會把 coherence 誤差轉成 population 誤差。

### acquire 的 shot 統計

使用者提出的模型：「每次acquire只算一次timeline，初態批次演化一次，然後用機率線性組合結果，每個rep與round演化都一樣，再給每個點加上noise就好」。agent 補了兩點，使用者確認：

- 加入 multinomial 投影雜訊；
- raw 延後生成，與 averaged 結果共用同一個 realization。

raw 的型別比較了兩種做法：

- (a) 模擬時改用 float64；
- (b) 保留 int64，averaged 的雜訊變異另外加上取整造成的 reps/12。

使用者選 (b)，並訂下約定：「reps以內是獨立timeline，以外是重複計算」。

整合 prototype 的結果：

- 同一輸入下，1 flux × 6 freq 從 129.2 s 降到 0.25 s。
- 3 flux × 201 freq 全部優化開啟，共 3.2 s。
- 誤差都 ≤ 1e-3。
- 向量化統計抽樣後，每個 flux 的 201 freq × 2 rounds 只要 0.37 ms。

### 平行化與編譯

使用者提出：「我認為可以討論平行化，這些機制都沒碰到平行化」。可用的工具有 numba、BLAS、GPU 和 multi process。需要的話也可以用 Rust 寫庫，並安裝所需依賴。

量測方式：每項只跑一次，機器負載約 3–5。加速比只取同一 process 內的前後對照。

| Kernel | 方法 | 時間 | 加速 |
| --- | --- | ---: | ---: |
| Dyson（2001 freq） | numpy 單 thread | 1.37 s | 1.00× |
| | numpy thread pool 4 | 0.94 s | 1.47× |
| | numba prange 6 | 0.40 s | 3.47× |
| | OpenCL | 0.26 s | 5.40× |
| | Rust rayon 4 | 0.52 s | 約 2.6× |
| 退回路徑（128 ns 點） | 原 Evolution | 5.58 s | 1.00× |
| | secular dissipator 加 numpy RHS | 2.91 s | 1.92× |
| | 整個 DOP853 用 numba | 1.01 s | 5.50× |

**誤差。** 各方法和基準的差都 ≤ 1e-13。編譯版的累加順序不同，所以和 numpy 不逐位元相同；同一後端在不同 thread 數下逐位元相同。使用者接受這一點，不另外保留 numpy 後端。

**secular dissipator。** FTF 的 jump 都是 |i⟩⟨j|，dephasing 是對角項。所以 dissipator 可以拆成兩部分：population 經 rate matrix 轉移，coherence 逐元素衰減，不需要做 256×256 的 dense matvec。用隨機 ρ 驗證，殘差為 3e-18。

**process pool。** 用 8 個像素測，4 個 worker 快 3.05 倍，6 個 worker 快 2.96 倍。

**256×256 小矩陣。** BLAS 多 thread 和 GPU 批次 matmul 都沒有勝出。

**Rust。** 在兩個 kernel 上都沒有贏過 numba，卻要多維護 native 建置鏈和兩份數值 kernel。

**使用者的裁決：**「那先不用GPU與rust，用numba與compact acquire」。

**process pool 先不做。** 只開 numba prange 一層平行，等量到實際有多少退回像素，再決定要不要加。

**非 constant 包絡的強驅動。** gaussian、DRAG 這類包絡沒有週期傳播子可用。使用者選擇用 numba prange 沿退回點批次計算 DOP853，並說明「amplitude rabi都會用」。prototype 結果：

- 12 點 gaussian amplitude sweep，6 thread 快 3.5 倍，換算 101 點約 44 s。
- gaussian 與 DRAG 的誤差都 ≤ 1e-13。
- 各 thread 數的結果逐位元相同。
- 首次 JIT 約 12 s，所以 GUI 不能同步等待 JIT。

### 多 gate 序列與分段傳播子快取

使用者要求：「多gate優化必須做，請考慮快取，然後我以為分段演化自然支援快取演化，看起來你沒設計？」

現有 `EvolutionLibrary` 有兩種快取，都無法在序列中重用 gate：

- **離線 U 表**：只存 closed、static plateau，runtime 不會套用到含耗散的請求。
- **exact density record**：key 包含初始 density，而序列中間每個 gate 的輸入 density 都不同。

新方向是 runtime 的分段傳播子快取。每個段先算出一個與輸入態無關的傳播子，key 由模型與段內容組成，序列結果就是這些傳播子的乘積。

難點在 lab frame 的 carrier phase：同一個 gate 在不同時刻播放，有效相位 θ 是連續值。

prototype 使用 gaussian 名義 gate：X90 為 64 ns、X180 為 128 ns，carrier 在主躍遷 361.87 MHz。gain 用數值粗調，所以 Ω/ω 偏大，這組設定比實機保守。比較結果：

| 選項 | 結果 |
| --- | --- |
| θ 最近 bin 分箱 | K=256 時仍超過 1e-2 |
| θ 的 Fourier 三角插值 | K=8 的插值誤差 6.6e-7，K=16 為 1.6e-11 |
| 完整 superoperator | 一個 64 ns gate 的一個 phase 樣本就要 69–257 s |
| 閉系統 U 加 secular 耗散，整個 gate 做一次 Strang 分裂 | 兩種 gate、K=16，建表共 16.7 s，快取約 140 KiB |

採用 Strang 分裂加上 Fourier K=16。

- **速度。** 快取命中的 gate 比 numba DOP853 快 2500–8000 倍。50 個 gate 的序列從 58.9 s 降到 1.4 ms。
- **精度。** 50 gate、4 µs 的序列，population 誤差 4.1e-4，low-four 誤差 2.9e-3。誤差主要來自 Strang 分裂，增加 K 無法降低。
- **RB 規模換算。** 400 條序列、19–38 萬個 gate，包含建表共約 22–27 s。逐點 batch DOP853 則要 20–40 小時。
- **長序列對策。** 為了壓低長序列的累積誤差，每個 gate 再切成 m 個子段，各自做 Strang。

### 刪除離線 U 表與 exact density record

使用者指出：「因為一定有耗散，離線U表機制可以刪除了，除非你找到可以支援耗散的算法」。

- **離線 U 表：刪除。** 能支援耗散的替代方案是 superoperator 表加 amplitude 插值。但它要服務的固定 DC plateau，已經由 flux 量化加 secular 傳播子涵蓋，這條路徑含耗散，在 runtime lazy 建立。替代方案每個節點還會大 256 倍。
- **exact density record：一併刪除。** 有了 compact acquire 和傳播子快取之後，record 只剩「跨 process 重跑同一點」這個用途。它每次 direct solve 都要 fsync 寫檔，退回點一多就成為 I/O 瓶頸。private replay 改成在相同的數值與 RNG 版本下直接重算。

### 兩 tone 同時驅動

AC stark 與 dual-tone reset 是兩個 tone 同時開。兩個頻率沒有共同週期，所以週期傳播子和 Dyson 都不能直接用。使用者問：「我不知道interleave算法有沒有辦法近似」。

| 近似 | 結果 |
| --- | --- |
| 單純 Trotter 交錯 | 每一小段都取決於兩個 tone 當下的 phase，無法快取；dt 要遠小於 1/carrier。只做評估，預期不會比直接積分快 |
| 可公度頻率 | 使用者指出頻率約要量化到 0.1 MHz。這樣共同週期是 10 µs，比 pulse 還長，不可行 |
| stroboscopic interleave 加 probe phase 插值 | 精度足夠，但含單週期 superoperator 預算後，只有嚴格解 0.1–0.43 倍的速度。K=3 時出現約 1e-6 的負 Choi eigenvalue |
| Floquet k=0 | stark tone 先換成等效 H_F 與 micromotion，probe 在 H_F 上演化。12 個 reference 點（含 gain 1.2、失諧 8 MHz 的 stress 點）最大誤差 6.4e-5，快 7–15 倍 |

使用者：「1納入，可以補測」。

- ac_stark 的 21 gain × 201 freq，單核約 44–64 分鐘；嚴格解約 5–11 小時。
- 下一步要補測：probe 改成在 H_F 上做 Dyson，並在批次中向量化。
- dual-tone reset 用的是強 probe 與激發態初態，必須另外驗收；通過前一律退回兩 tone 的 DOP853。

### bath 與 secular 形式

FTF 的耗散速率由 YAML 參數算出：

- 電容與電感損耗的 Q；
- quasiparticle 比例；
- flux noise 振幅；
- coupler 的線路阻抗；
- 讀取腔的 κ 與熱占據數；
- 模型溫度。

每個躍遷的速率由三者決定：躍遷頻率、矩陣元、噪聲譜密度。所以速率會隨 flux 改變。

bath 在建構時就是 secular 的：jump 是 H0 eigenbasis 上的 |i⟩⟨j|，dephasing 是對角項，active 的簡併躍遷會直接報錯。

裁決：secular 形式（rate matrix、coherence 衰減、eigenbasis）改由模型直接提供。模型若沒有 secular 形式，就在進入點報錯。不保留執行期驗證與退回原 Evolution 的路徑。

### 分段分派決策圖

使用者要求：「還有請記錄這個決策圖，之後修改算法都要基於這個圖」。之後又修正了兩次批次原則：

1. 「不是只有freq要向量化，而是預設reps內都可以向量化」。
2. 「位置路徑重要嗎？不一定要照順序或同步計算吧？」

最後定下的批次原則：

- 批次以 kernel 與模型相容性為單位，與段的位置和順序無關。
- 傳播子類（靜態、週期、gate 快取、Floquet prepare）與輸入態無關。先對整個 acquire 的唯一段內容去重，再以任意順序平行計算。
- 狀態類（Dyson、RWA、DOP853）用 ready queue 動態組成 batch。唯一的限制是：同一條 timeline 內，輸入態必須先算好。

切段時機：任一 pulse 起訖、flat-top 的 ramp 與平台交界、flux 改變、readout 開始。

每段的描述包括：

- 模型 identity 與 secular 形式；
- 起點與長度；
- 段內 flux 是否為常數；
- 每個 tone 的 port、carrier、gain、包絡種類與參數、有效起始相位 θ；
- 推導出的驅動強度 r、二光子 r₂ 與各 guard；
- 這段內容在本次 acquire 中出現幾次。

以下是撰寫當時的版本。權威版本會隨 spec 移入模擬環境 ADR 或 `simulate/evolution` 的 README。

```
每個分段
├─ readout 段 ─▶ readout 快取 + expm 精確解
├─ 段內 flux 在變 ─▶ 量化成常數 flux 子段後重新分派（第二批之前退回原 Evolution）
├─ 模型沒有 secular 形式 ─▶ 報錯
└─ 同時播放的 tone 數
   ├─ 0 ─▶ secular 解析靜態傳播子
   ├─ 1
   │  ├─ constant（含 flat-top 平台）
   │  │  ├─ 弱驅動且輸入態已知：r ≤ 0.1 ─▶ 二階 Dyson；0.1 < r ≤ 0.2 或 guard 觸發 ─▶ reduced RWA
   │  │  └─ 其他 ─▶ 週期傳播子（每週期 1／4／8 步，θ 用 Fourier 插值）
   │  └─ 有形狀（gaussian、drag、cosine、arb、ramp）
   │     ├─ 相同內容重複出現 ─▶ gate 快取（閉系統 U + m 子段 Strang，θ 用 Fourier K=16）
   │     └─ 各點不同 ─▶ numba DOP853，沿工作項目 prange 批次
   ├─ 2
   │  ├─ 兩者皆 constant、一個是弱 probe、Floquet guard 通過 ─▶ Floquet k=0
   │  └─ 其他 ─▶ 兩 drive 相加的 numba DOP853
   └─ ≥ 3 ─▶ 多 drive 相加的 numba DOP853
```

段與段的組合方式：

- 傳播子類用矩陣乘法作用在目前的 ρ 上。
- 狀態類直接以目前的 ρ 為輸入。
- 共同的前綴只算一次。
- 每段實際走的路徑記錄在 author diagnostics 中。

## 被否決的選項與理由

- **dense superoperator RHS 與批次 DOP853**：沒有加速效果。
- **interaction picture**：drive 期間的步數只少三到四成，而且 population 誤差比舊路徑大。
- **Magnus 展開**：H0 頻寬 7.7 GHz 與時變 ΔH 形成巢狀 commutator，步長小到比舊路徑還慢。
- **時間上的 flux 量化（sample-and-hold）**：可能引入 GHz 等級的非絕熱假象。
- **flux 量化做成可選設定**：使用者要求一律開啟。
- **以 k·τ 時間倍數為 tail 門檻**：使用者選了 flux 絕對值。
- **保留現行精確路徑為預設**：與這個 task 的目的不符。
- **只做「平均值加 receiver noise」**：少了投影雜訊，SNR 會比真機好。
- **維持逐 shot 抽樣**：在使用者的規模下以小時計，而且最後只用到平均值。
- **raw 改 float64**：`get_raw` 的型別會和真機不一致。
- **timeline 隨 rep 變時退回逐 shot**：會留下一條很少用、卻要一直維護的慢速路徑。
- **flux 軸插值**：每個 flux 的預算只要幾十 ms；avoided crossing 附近還有 eigenvector 的 gauge 與排序風險。
- **3D flux 全表**：約 750 PB，改用 lazy 快取。
- **GPU（OpenCL）與 Rust**：使用者決定先不做。Rust 沒有勝過 numba。
- **第一批就做 process pool**：退回比例未知，收益不確定。
- **保留 numpy 後端供逐位元比對**：要維護兩份 kernel，而 1e-15 的差異對實驗沒有影響。
- **gate 快取用完整 superoperator 或 θ 分箱**：前者建構太貴，後者精度不足。
- **離線 superoperator 表**：用途已被 runtime lazy 路徑涵蓋，每個節點大 256 倍。
- **保留 exact density record**：剩下的用途很小，fsync 寫檔又會成為 I/O 瓶頸。
- **兩 tone 用 Trotter 交錯、可公度頻率、stroboscopic interleave**：理由見上文的比較表。
- **執行期驗證 secular 結構並退回原 Evolution**：對 FTF 永遠不會觸發，卻要一直維護一條退回路徑。
- **依段的位置與路徑同步分組**：傳播子類不需要對齊位置，狀態類只受同一條 timeline 內的先後限制。

## 留下的問題

- **待補的 prototype：**
  - Floquet 加 Dyson 的批次速度與精度。
  - Strang 子段數 m，以及 950 gate 長序列的誤差累積。
- **門檻：**
  - 相同內容重複幾次才建 gate 快取。粗估建表每種 gate 約 6–11 s，DOP853 每次約 0.5–1 s。
  - flat-top 平台多長才值得走週期傳播子。
- **未驗證的情境：**
  - avoided crossing 附近的分類與量化。範例 YAML 在 gain 0–1.0 內沒有 crossing，最小 gap 136 MHz。
  - DOP853 rejection 分支、10 µs 的強驅動與長序列。
  - flat-top 的接縫：carrier phase 要沿全域時間延續。
  - dual-tone reset。
- **原 Evolution：** 第二批的 flux 量化完成後，原 Evolution 就沒有呼叫端了，是否保留待決。
- **範圍與排程：**
  - 第一批包括：compact request、readout、靜態傳播子、週期傳播子、spectroscopy 快速路徑、統計抽樣、依 flux 批次分派與跨 round 重用，以及 numba 退回路徑。
  - gate 快取與 Floquet 已確定要做，但歸入哪一批還沒定。
  - 離線 U 表與 density record 要等 measure-simulation-environment 收票後才刪除。
  - orchestrator 由誰擔任、何時開工，由使用者決定。
- **ADR：** 快速算法取代預設數值方法、刪除離線表、replay 契約改為重算，這些都要寫進[模擬環境 ADR](../adr/draft/simulation-environments.md)。

## 依據

- 本文的數字都來自一次性 prototype（P1–P13 系列）的量測報告。報告是本機工作檔，沒有入庫；必要的結論與數字已寫在正文。
- 程式位置以 basis 中的 06 lane commit 為準：
  - `lib/zcu_tools/experiment/ftf/runtime.py`：`FTFSource.execute`
  - `lib/zcu_tools/simulate/evolution/solver.py` 與 `README.md`：快取種類
  - `lib/zcu_tools/simulate/baths.py` 與 `simulate/ftf/parameters.py`：bath 與耗散參數
  - `lib/zcu_tools/simulate/adc/response.py`
  - `lib/zcu_tools/program/v2/simulation/frontend.py`
  - `zcu_lab/v2/twotone/rb/program.py`：gate 的 phase 組成
  - `zcu_lab/v2/twotone/ac_stark/core.py`：兩 tone 同時播放

## 後記（2026-10-16）

快照 commit 之後，又定了三項。

### 排程：A 階段加波次

使用者指出：「既然計算不需要序列，除了需要初態的演化都可以考慮批次化+平行化，需要初態的演化可以等最後再一次批次計算，或是條件齊全在個別計算，但效率」。據此定下的排程，取代前文的 ready queue：

1. **規劃。** 切段，並對段內容去重、計數。波次數等於各 timeline 中狀態類段數的最大值。
2. **A 階段。** 使用兩次以上的傳播子類內容，依 kernel 分批平行計算，順序不限。
3. **波次。** 每條 timeline 先用傳播子乘積，把狀態推進到下一個狀態類段的入口。接著把這一波所有 timeline 的狀態類段，依 kernel 合成一個 batch 一次算完。
4. **收尾。** 推進到終點，再接 readout 與統計。

多數實驗只要 0–1 個波次：twotone、gaussian amplitude Rabi、ac_stark 是 1 波；constant Rabi、T1、Ramsey、echo、RB 是 0 波。

只用一次的傳播子不存，在波次中建好就作用到狀態上，用完即丟。2001 個週期傳播子各 1 MiB，存起來會吃掉 2 GiB。是否快取由使用次數決定，不由路徑種類決定。

不採用「條件齊全就個別計算」，原因有三：batch 太小時，prange 吃不滿核心；Dyson 的向量化效益下降；每次呼叫都有開銷。acquire 的結果本來就整批交出，低延遲用不上。

### GPU 與 joblib

使用者選 A：「A，好，但請你注意python迴圈低效的問題」。

- **只用 numba。**
- **GPU。** Dyson batch 在 OpenCL 上約快 1.5 倍，是唯一值得上 GPU 的 kernel。批次 expm、gate 建表、DOP853 都不划算。第一批不做 GPU，但 Dyson kernel 的介面寫成「數值陣列進、結果陣列出」，以後可以換成 OpenCL 後端。
- **joblib。** 已在 `client` extra 中。它只適合 numba 化不了的 Python 與 scipy 工作，例如 A 階段的 scipy expm、模型與 bath 建立、Floquet prepare。對 numba kernel 來說，process pool 和 prange 速度相當，還要多付 worker 啟動與 pickle 的成本。第一批完成後量各階段耗時，若上述工作成為最大項，再加 joblib。

### Python 迴圈的界線

這次找到的主要瓶頸，幾乎都是 Python 迴圈的次數隨 shot 或點數放大：逐 shot 解譯、分組、抽樣，以及逐 RHS 組 H(t)。

使用者確認的規則：Python 層迴圈的次數，只能隨模板數、唯一內容數、kernel 種類或波次數增長；不能隨 sweep 點數、段數 × 點數、reps 或 rounds 增長。

具體做法：

- 規劃以模板為單位，參數用 numpy 陣列沿 sweep layout 廣播。
- 去重用 `np.unique`。
- 傳播子的作用與連乘，由單一 numba kernel 依段索引表執行。
- batch 的組成與還原只用索引陣列。

驗收時，在使用者規模下單獨 profile Python 層。點數加倍時，這部分的耗時不能跟著加倍。
