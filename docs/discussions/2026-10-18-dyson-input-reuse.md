---
date: 2026-10-18
status: active
superseded_by: null
related:
  - docs/discussions/2026-10-17-ftf-dyson-guard-and-classification.md
  - docs/discussions/2026-10-18-constant-period-builders.md
  - lib/zcu_tools/simulate/evolution/README.md
basis:
  - "repo main@f152eff67：研究整理的截止版本"
  - "repo@30971bec7、c0a3077c0：早期診斷與正式驗收"
  - "repo@a5ffa6135、16b1dde43：共用表與 distinct-input 研究基線"
---

# Dyson 的解析分類、輸入共用與失敗的 sparse scan

## 結論

這輪研究把三種成本分開處理。解析 guard 避開不適合二階 Dyson 的輸入，共用表把相同輸入的路徑收縮一次，不共用輸入的短 plateau 則優先比較 Hilbert period。跳過零 coupling 的 sparse scan 沒有採用，因為完整 acquire 只省 12.6%，未達研究的 20% 目標。

本文保存截至 main `f152eff67` 的研究結果，不是後續版本的 dispatch 規格。數值契約與入口見 [evolution README](../../lib/zcu_tools/simulate/evolution/README.md)。

## 問題

早期單 row 的二階候選有 14,991,360 個，真正需要計算的 divided differences 只有 20,800 個。快取 scalar 函式不能消除枚舉 paths 的成本，也省不掉組係數與收縮 density。

另一個問題是輸入的身分。source 初態、相同前綴的輸出，以及每個頻率各自經過 ramp 後的輸出，不能都當成同一份 density。共用錯誤的輸入會同時改變 guard 與演化結果。

使用者在 2026-10-17 要求「分類要保持輕量，因為分類要做完才能開始批次計算」，也指出「把最慢的步驟算兩次真的很蠢」。因此 apply 後的檢查不能偷偷變成重新演化。

## 考慮過的選項與比較

### 解析 guard 與事後檢查各自回答不同問題

研究保留了三個分類量。r 與 r₂ 限制一階與二階的 drive ratios。B2 用 scalar transition bounds 估未剪枝二階 Dyson 的 driven-minus-free population change。B2 不枚舉 modal paths，也不先演化 density。

當時採用的門檻為：

- r ≤ min(0.1, 0.25 × budget^(1/3))。
- r₂ ≤ 0.05。
- B2 + budget/8 ≤ min(0.02, (1.5 × budget)^(2/3))。

在 budget=1e-3 時，r 上限為 0.025，B2 加上 0.000125 的保留額後約須低於 0.0131。這些係數經合成模型校準，不是任意多能階的嚴格誤差定理。

已知輸入使用 basis conversion 後的完整 density。實際 population differences、pair coherence magnitudes 與 off-diagonal trace norm 進入 B2，不能先把小 coherence 清零。未知前綴則使用 arbitrary-input 上界，不能讀取 worker 演化中的 density 來選路。已認證的 no-drive prefix 保留 B2 豁免，driven prefix 仍須通過 actual-input B2。

apply 後另算輸出 populations 相對同一輸入 free evolution 的 |Δp|。超標只記錄與 warning，不重算，也不修補結果。使用者原話是「就不重試了，誤差大就大吧，重試很浪費時間」。這項診斷不能證明對 exact dynamics 的誤差。

### 相同輸入先收縮，之後重用 paths

早期 `30971bec7` 的獨立 prototype，在單 flux × 2001 frequencies、單 worker 的 Engine 計時中，把 404.036 s 降至 7.933 s，density 最大差 2.19e-19。這不是完整 acquire，也不是後來的正式驗收。原摘要未列 interleaved 或 foreign fraction，不把這個探索值當 clean pair。

共用表之後仍有 Python 建表成本。以 `a5ffa6135` 的舊表為對照，研究改成按非零 coupling/input 拓樸預配 arrays，以 integer keys 配對共軛 paths，再由單執行緒 Numba 做 greedy discard。穩定排序、乘法與加總順序沒有改。

同 process 暖機後交錯三次，兩個 threading workers、12 chunks、native threads=1，S1 三個 flux 各 2001 點的完整 acquire 中位數如下：

| 路線 | Wall |
| --- | ---: |
| 舊共用表 | 5.617 s |
| 不建共用表 | 4.312 s |
| 新共用表 | 1.503 s |

S3 的 201 點 acquire 是 4.088／2.742／2.588 s，收益小得多。五組實際 model/input 的完整 candidate arrays、retained arrays 與 discarded bound 都 bitwise 相等。這次比較支持加速建表，不支持加 row-count 硬門檻。它不是不同 process 的 clean throughput gate，報告未列逐 run foreign fraction。

### 共用 driven prefix 與 coherent mode pruning

引擎先辨認完全相同的非空前綴，再演化一個代表。相同 final level 或相同 final state 不足以認證。前綴身分還要包含時間、port、carrier、gain、phase、envelope 內容及連續 flux 的插值邊界。

B2、input contraction、free baseline 與後續演化都採用同一份實算輸出。這份 density 只活在當次 evolve。對 coherent inputs，先成對剪除 driven-path 上界很小的共軛 modes，最多用 budget/16，剩餘 path-pruning 額度是 budget/8 減去已花額。free evolution 仍使用完整 modes，對角輸入不做這一層剪枝。

driven-prefix 實作 `979541911` 的單核、同 process 交錯、空 prepare cache 比較中，201 個 probes 前有共同 pre-pulse 的案例從 8.257 降至 0.384 s。低頻 π pulse 後掃 probe 從 3.219 降至 0.522 s。這些不是含 program planning、ADC 的完整 acquire。S7 echo 約為 0.432 → 0.433 s，沒有可辨識的收益，不能只報最快的兩案。

### 不同輸入不要硬套 shared contraction

在 `16b1dde43` 的研究中，R37 是兩個 1 µs Gaussian ramps 夾 50 ns plateau，201 個 carriers 掃共振附近。各 ramp 留下不同 density，不能共用 plateau 的 input table。F15wide 則用 10 ns ramps、80 ns plateau 掃 50–7000 MHz。

改成 period-first 後，兩 worker 的完整 acquire 分別從 37.525 → 4.252 s、29.013 → 3.204 s。每 cell 一次，獨立 process 交錯場景與版本，完整 target warmup 後清 prepare cache，不是重複中位數。21 點共振附近 strict 比較的 population 誤差低於 3.3e-11，peak 不移格。

period apply 仍隨 duration 增長。約 15 µs 的成本交叉是外推，不是實測門檻。後來採用的 ≤10 µs 偏好不能被解讀成量測已找到精確 crossover。

## 被否決的選項與理由

- sparse input modes 不適用 R37。其 201 個實際 plateau inputs，每 row 的 256 modes 都非零。
- 跳過零 coupling 把二階候選降至 5,630,400 個，但兩 worker acquire 只從 32.728 降至 28.606 s，省 12.6%。這是 `16b1dde43` 上每版一次、完整暖機後清 cache 的比較，R37 兩邊都含 input capture 成本。density 與 retained paths 逐 byte 相同，仍未解決每 row 重建大量 paths 的成本，故 prototype 未入庫。
- 以 row-count 門檻避開 shared build 只掩蓋舊建表太貴的問題。新建表在小群組也比不建表快，沒有採用這項門檻。
- 以 post-check 失敗重跑其他算法違反 no-retry 要求。guard rejection 與已開始數值執行後的失敗不能混為一談。

## 留下的問題

早期 `c0a3077c0` 正式驗收的九組 accuracy 都通過 1e-2。301 × 2001 與 301 × 4002 的 acquire wall 為 1023.359／1598.832 s，Python self CPU 比為 1.2581，當時 workers=8。這輪驗收已完成，但不能拿它與後來 workers=2 的研究相除，也不能用它代替新模型的 feature 驗收。

未知輸入的 ordinary Dyson 仍可能有合法用途。以上研究只證明所測短 plateau 的成本交叉，沒有證明可以刪除任意 duration 或模型上的 Dyson。

## 依據

本文轉錄早期效能診斷、正式驗收、共用表成本、driven-prefix、plateau routing 與 sparse-scan 報告。原始工作檔未入庫，本文保留必要條件與負面結果，沒有重跑計時。

[2026-10-17 的 guard 討論](2026-10-17-ftf-dyson-guard-and-classification.md)保留當時使用者裁決；[constant period 研究](2026-10-18-constant-period-builders.md)補充替代 builder 的演進。
