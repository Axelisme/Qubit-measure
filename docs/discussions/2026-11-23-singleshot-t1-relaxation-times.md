---
date: 2026-11-23
status: superseded
superseded_by: 2026-11-23-singleshot-t1-amplitude-naming.md
related:
  - ../adr/0069-experiment-core-and-frontends.md#呈現規範
  - ../adr/0071-coherence-units.md#pair-lifetime-與-manifold-model-t1
  - ../adr/0073-measure-form-derivation.md#form-的形狀與初值
  - 2026-11-23-inverted-error-bars-and-figure-facts.md
  - 2026-11-23-singular-readout-analysis.md
basis:
  repo_commit: a242eeaeff3e57fce0674d1844776a4ec487f7b1
  external_sources: none
---

# Single-shot T1 的兩個衰減時間與 tone Stop 初值

## 結論

2026-11-23，designer 依夜間授權代裁，待使用者覆核。普通 single-shot T1 與固定 probe tone 版在 Analyze 圖由短到長列出兩個模型衰減時間，保留實驗名與 reduced χ²。不指定其中一個為 qubit T₁。固定 tone 版仍提議把較短時間存入 `$qubit.ext.t1_with_tone`。固定 tone、tone gain 和 tone frequency 表單以其五倍作為 exposure Stop 初值。普通版不寫回，`$qubit.t1` 不變。

## 問題

兩個實驗用同一組轉移率擬合 g、e 兩種準備的 G/E/Other 分類比例。Other 是分類器的結果，不是已識別的能階。`calc_lambdas` 將模型特徵值的負實部由小到大排列。兩個非定常衰減率取倒數後，`t1 = 1/lambdas[2]` 較短，`t1_b = 1/lambdas[1]` 較長。這個欄位順序在 `d0c2d75e7` 已定；`d90838e2f` 沿用數值，卻把 `t1` 標成 `Slow T₁`，把 `t1_b` 標成 `Fast T₁`。`12702c57c` 又把同樣的快慢寫進兩份 guide，並說 tone 版寫回慢模，但 adapter 一直提議 `t1`。

## 選項與取捨

維持原標籤會讓人讀反。只對調快慢則仍容易把兩個數當成獨立的 g、e 準備 lifetime，或誤認其中之一就是 qubit T₁。以模型時間升序呈現，也能容納同值。複共軛根有相同的實部，兩個數是同一振盪包絡的 e-folding 時間。

改存較長的時間可延長掃描，但不一定有可見的慢尾。調查時的單輪離線 Analyze 用 21 個 0–100 µs 的 delay、每個 delay 和準備 1,000 個合成 counts，得到 11.16261 和 23.16890 µs。當時的 HEAD 是 `c15a15706`，執行 tree 是 `b15f6735`。用該例推算，五倍初值會從 55.81 µs 變成 115.84 µs；不是 Run 結果。在較短的 Stop，慢模包絡還剩起始振幅約 9%。這不表示總布居還差 9%，因為該次 probe 沒有量慢模的絕對振幅。若慢模在所看的曲線上很弱，直接改用較長時間會多掃一段，也會讓同一 key 的新舊值出自不同規則。

若要用振幅選主模，先指定準備態和可觀察量。例如可比較有無 π pulse 的 `P(e)` 差值，但在這個差值上振幅較大的模不一定決定曲線尾端。[ADR-0071](../adr/0071-coherence-units.md#pair-lifetime-與-manifold-model-t1) 的 notebook manifold 使用 Gibbs 布居交換 g/e 的準備和 `1−p₀(t)`；[`chain/t1`](../../zcu_lab/v2/chain/t1/guide.md) 用 K-cycle 資料、detailed balance 模型和熱布居加權的 population-change amplitude。兩個先例都先定義要看的量，沒有提供本實驗可直接沿用的振幅判準。

另報 g/e pair equilibration time `1/(ge + eg)` 有明確定義，但排除了經 Other 的路徑。例如 g↔e 雙向率各 0.05 µs⁻¹、g/e→Other 各 0.01 µs⁻¹、Other→g/e 各 0.03 µs⁻¹，pair time 是 10 µs，三類模型的兩個衰減時間約為 9.09、14.29 µs。這是手算例。Tone 條件下的 pair 值也不能當成無 tone 時的 qubit T₁。

## 未採用

沒有增加 pair 或振幅選出的第三個數，兩者都不能回答 Stop 應覆蓋多長的尾端。沒有改存較長值，也沒有取消成功擬合的寫回。取消提案只會停止更新初值，已存值仍會優先。較短值保留為經驗初值，由操作員看兩種準備的尾端點並調整 Stop。既有 fallback 和校正無解時只顯示分類比例、不擬合也不提案的規則照舊。

## 留下的問題

若將來要讓 Stop 達到指定 plateau，再訂要看的準備態、可觀察量和容許偏差。也要定義擬合無法辨識尾端時怎麼處理，以及初值能否沿用到不同 tone 條件。這次沒有為初值加上 plateau 保證。

## 依據

- [ADR-0069〈呈現規範〉](../adr/0069-experiment-core-and-frontends.md#呈現規範) 與[一維反演量的誤差棒與圖上事實](2026-11-23-inverted-error-bars-and-figure-facts.md)將圖上的擬合事實與 Writeback 建議分開。這次沿用分工，由 core 報兩個時間與品質值，由 tone adapter 提議初值。
- [作者指南的 T1 初值契約](../../zcu_lab/v2/measure-authoring.md#t1-與等待)與 [ADR-0073](../adr/0073-measure-form-derivation.md#form-的形狀與初值)仍適用。這次未改表單的初值來源或觸發時機。
- 原始調查與歷史 probe 未入庫。落地以前後同一 post-R11 基底上的合成資料重跑，沒有以歷史數值作為驗收基準。現行讀法見[普通 T1 guide](../../zcu_lab/v2/singleshot/t1/guide.md)與[tone guide](../../zcu_lab/v2/singleshot/t1_tone/guide.md)。
