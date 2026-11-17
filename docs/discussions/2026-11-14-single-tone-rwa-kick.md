---
date: 2026-11-14
status: active
superseded_by: null
related:
  - docs/adr/0078-evolution-module-boundary.md
  - lib/zcu_tools/simulate/evolution/README.md
  - docs/discussions/2026-11-16-floquet-kept-price.md
basis:
  repo_commit: 91e3c54a4
  external_sources: none
---

# single_tone_rwa 以 endpoint kick 保留，guard 改用 κ2

本紀錄整理 2026-11-14 的裁決，並記下截至 2026-11-17 的落地驗收與延期範圍。

## 結論

當時選擇保留 `single_tone_rwa`，在每個 RWA 列演化前後各做一次一階 endpoint kick。
落地版本丟掉初始 kick 後小於 `1e-5` 的跨 component density 項，避免逐對演化的成本。
最後 admission 以被省略邊係數的譜範數指標 κ2 取代剩餘 population transfer 比較，門檻校準為 `0.08`。
Population-driven retention restoration 保留，沒有一起換掉。
唯讀診斷公開回報新指標，guard 沒算的數值以 `RWAUncomputed` 表示。

價格計入 kick 成本的工作延到下一個 effort。
因此這次交付的 plateau 預設選路仍先試 `single_tone_rwa`，再試 `floquet_period`，不比較兩者成本。
L、I 工作負載上的 kick 版 RWA 比 `floquet_period` 慢超過 10%，當時 warm 對照為 16–48%。
這個速度缺口經審閱後保留，沒有把 10% 判準改寬，也沒有宣稱價格已包含 kick。

實作契約與目前選路由 [evolution README](../../lib/zcu_tools/simulate/evolution/README.md) 擁有。
完整序列精度與 family 去留的規則見 [ADR-0078](../adr/0078-evolution-module-boundary.md#family-去留合格只比耗時)。

## 問題

舊 guard 在 221 個真實模型點中放行了 20 個超標點，完整 density 最大絕對誤差為 `1.08e-2`，超過 `2e-3`。
N 型序列換成隨機相位後，21 個點中有 15 個超標，最大 `1.07e-2`。
原 N 的 X、Y、−X、−Y 循環剛好抵消部分一階誤差，沒有揭露這個缺口。

RWA 丟掉的快速振盪項，在相干 density 元素留下約為耦合除以失諧的一階誤差。
Population transfer proxy 卻隨這個比值的平方變化，不能直接代表完整 density 誤差。
脈衝終點的相位也會改變誤差，整數個失諧週期的結果不能代表附近時長。

使用者在 2026-11-13 問：「我想知道的是，他的優勢來源是不用重建頻率，那誤差來源呢？能縮小嗎？」
這個問題把研究從刪除或收窄 guard，改成先查誤差來源並試著修正。

## 考慮過的選項與比較

當時的四組對照工作負載各有 24 點。
F 掃 carrier，L 掃脈衝長度，I 掃兩個 gate 中間的 idle，N 掃循環相位 gate 的個數。
場景定義由 [tools README](../../tools/README.md#freeze-numeric-scenes) 與對應 producer 擁有。

| 選項 | 當時比較的結果 |
| --- | --- |
| 刪除 RWA，交給其他 family | 四組改走 `floquet_period` 的非獨占對照合計約 340 ms |
| 一階 kick，加跨 component 小項丟棄 | 若按各場景較快的合格 family 分流，估計合計約 144 ms |
| 只收窄 guard | 試過多指標邊界，以及只用 transfer `T ≤ 1e-4`；單 gate 指標無法辨識跨 gate 累積 |
| 只在需要的列 kick | 以低能階單列指標決定跳過 kick，未覆蓋高能階與序列相位風險 |
| 全 kick，不丟小項 | 精度改善，但 L、I 相對 `floquet_period` 的耗時比約為 1.9、3.6 |
| 第一層 retention | 只補回靜態邊，仍受到 phase、pruning 與維度限制，沒有救回 F/L/I/N |

340 ms 與 144 ms 是當時非獨占研究對照的量級與分流估計，不是本次預設選路的正式加速比。
144 ms 的估計假設 L、I 會改選 `floquet_period`，這次交付尚未實作該價格選擇。

使用者在 2026-11-14 對「簡單和速度衝突時，你選哪個？」選「B」，即保留並用 kick 修好。
對「kick 落地前，要不要先讓 `single_tone_rwa` 退出預設選路？」選「2」，即維持現狀。

這次遵循「完整序列最終誤差合格後，只比耗時」的先例，沒有為了省實作而刪除仍有速度優勢的 family。
對簡單優先的取捨，使用者在這題明確選了保留與修正。
後來的 [Floquet 保留與價格重校](2026-11-16-floquet-kept-price.md) 同樣先研究快速路徑能否保留。
兩者的交付範圍不同，Floquet 的兩 family 價格已重校，這次的 RWA／FP 價格仍延期。

## 被否決的選項與理由

只收窄門檻的研究方案包括多指標 A/B、貼近樣本精度邊界的 C，以及單一 transfer 門檻 `T ≤ 1e-4`。
它們沒有提供跨 gate 誤差累積的界。
其中 `T ≤ 1e-4` 在隨機相位序列的 21 個點全數放行，卻有 15 個超標。
只在需要的列 kick 也漏掉同一批序列的累積誤差，結果等同 baseline。
因此沒有採用這兩條路線，沒有把單 gate 的樣本內合格當成序列保證。

全 kick 不丟項的方案，會讓原本可跳過的跨 component 對都進入演化。
L、I 的 1.9、3.6 倍耗時使它不能直接作為終點。
丟項容差 `1e-4` 雖改善速度，但在 200 gates 隨機相位序列累積到 `7.3e-4`。
當時決定取更嚴的 `1e-5`，再以長序列確認餘裕。
`7.3e-4` 是已量到的值，不是任意序列的誤差上界。

第一層 retention 只把邊補回靜態保留圖，撞到既有 phase、pruning 與維度限制。
它的失敗沒有證明頭尾解析修正也會失敗，因此後續仍試了一階 kick。
刪除則因使用者選擇保留並修正而未採用。

## 診斷介面

這次交付了公開的 `single_tone_rwa_diagnostic`、`SingleToneRWADiagnostic`、`RWAGuardMetric` 與 `RWAUncomputed`。
查詢低 carrier 或零 carrier 時，family 拒絕與輸入不合法是不同情況。
前者回報拒絕，guard 沒算的後續指標標成未計算，不為了診斷多算，也不填 0 或直接因低 carrier 而 raise。
契約外輸入仍由入口 raise。
使用者在 2026-11-14 以「一起追认」確認了這項低 carrier 語意。

新 `kappa2` 欄位取代 `omission_transfer`，沒有相容別名。
欄位的單位、門檻、可計算條件與失敗語意，由 [SingleToneRWADiagnostic 與查詢入口的 docstring](../../lib/zcu_tools/simulate/evolution/kernels/single_tone_rwa.py) 擁有。
這個唯讀查詢沒有改變演化、選路或 history。

## 留下的問題

價格計入 kick、FP 週期在一次 evolve 內重用，以及量表 benchmark 都延到下一個 effort，原判準保留。
FP 重用候選在 N 的 `warm_clear_history` 成對計時中，wall 比值中位數為 `1.079873`，超過不得變慢的 1。
當時仍未定出成因，沒有收下這個候選，也沒有將失敗解釋成價格已修好。

κ2 的 `0.08` 門檻在固定驗收集上有完整 density 餘裕。
這不證明它是最大可用門檻，也不是任意輸入或任意序列的精度定理。

## 依據

本 repo 的依據為 `91e3c54a4`，包含 endpoint kick、κ2 admission、公開診斷、測試修正與當時的文件。
研究數字與一次性驗收的原始工作檔未入庫；本紀錄保留比較所需的輸入範圍、結果與限制，沒有要求讀者取得那些檔案。

落地前的新 guard 驗收中，放行點的完整 density 最大絕對差全部不超過 `2e-3`。
最後的測試修正恢復了 12 個舊 gain-sweep 物理輸入，對獨立參考解的最大完整 density 誤差為 `6.480447e-4`。
κ2 邊緣內側案例的誤差為 `3.061429e-13`。
這些是不同驗收集合的結果，不是只用單一邊緣案例證明全部輸入。

對應的可重查入口是 [gain-sweep 契約測試](../../tests/simulate/evolution/contract/test_guard_sweeps.py)、[精度矩陣](../../tests/simulate/evolution/contract/test_accuracy_matrix.py) 與 [evolution README](../../lib/zcu_tools/simulate/evolution/README.md#kernel-summary)。
