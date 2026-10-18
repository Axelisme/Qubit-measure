---
date: 2026-10-18
status: active
superseded_by: null
related:
  - docs/discussions/2026-10-18-direct-integration.md
  - docs/discussions/2026-10-18-planner-evidence-and-overhead.md
  - lib/zcu_tools/simulate/evolution/README.md
basis:
  - "repo main@f152eff67：研究整理的截止版本"
  - "repo@7fdd259ac、058963f1a：modal BLAS 與 frame 研究基線"
  - "repo@12f9e96a3：重掃與 direct-first screening 基線"
---

# Two-tone Floquet 的 frame、sidebands 與成本交叉

## 結論

Floquet 的三輪改進解決了不同問題。modal BLAS 去掉 apply 中間模式的逐項迴圈，frame 改寫降低平均化與 bath norm 成本，retained sidebands 則處理 qualitative workload 中不能直接平均掉的近共振分量。

後來的成本比較又讓短段 throughput workload 改走 DOP853。Floquet 變快不代表它應該永遠優先。

retained-sideband admission 最後以明示限制的 calibrated approximation 採用，沒有升格成對連續原方程的嚴格誤差保證。本文只記 main `f152eff67` 以前的證據，數值契約見 [evolution README](../../lib/zcu_tools/simulate/evolution/README.md)。

## 問題

S9 有兩個不能互換的場景。

| 場景 | Controls 與用途 |
| --- | --- |
| Throughput | 16 ns，Stark 500 MHz、nominal gain 0.02，probe 掃 400–450 MHz、gain 0.0001，201 點。用於成本與 throughput |
| Qualitative | 512 ns，Stark 約 7.289 GHz、gain 0.5，與 transition detuning 500 MHz，probe 掃共振附近 ±8 MHz，201 點。用於可見 Stark peak |

throughput 的 strict target contrast 只有 3.71e-10。raw argmax 一樣，仍不足以辨識 Stark shift。qualitative 的強長段則會被原本 k=0 averaging guard 拒絕。不能把短段速度套到長段，也不能把小 population error 當 feature 已驗收。

## 考慮過的選項與比較

### Modal apply 改用 BLAS

以 `7fdd259ac` 為基線，prototype 用 BLAS 收縮 modal Dyson 的中間模式。frame、k=0 模型、micromotion 與 periodic fallback 都沒有改。201 點完整 acquire 從 191.343 → 12.110 s，15.80 倍。

這組計時用兩 workers、native threads=1、CPU 0/1、獨立 process。原版 full-target warmup 會超過單 process 時限，故兩邊都只用同場景五點 warmup，再清 prepare cache。prototype 首次 foreign 15.58%，重跑為 4.79%，採用與 uncontended 原版的比較，不是三次中位數。

21 點 strict comparison 的 population 最大差 1.48e-12，原版與 prototype 完整 density 差約 4.21e-19。raw peak index 都是 15，但這沒有解除 throughput 場景的物理不可辨識性。

### Frame averages 與 bath norm

更早的五點 builder 驗收比較 `dcdc02353` 與 `58ca8b968`，使用兩 workers、同 process 完整暖機後交錯三次，acquire 中位數為 15.462 → 9.214 s，classification 為 10.393 → 4.153 s。該輪只驗五點，沒有補成 201 點驗收，也沒有解決 Stark shift。這組摘要未列逐 pair foreign fraction，不據此宣稱 isolated throughput。

`058963f1a` 上的下一個 frame prototype 用 Hilbert contractions 取代大型 Liouville einsum，bath norm 改用較便宜的保守上界，並共用 ±k 的 norm 工作。三組低負載 interleaved pairs 的 throughput acquire 中位數為 11.06 → 8.43 s，frame 約 4.1 → 1.46 s。host 同時有其他負載，不能當 isolated throughput。

qualitative 的 frame 只占 0.5%，所以這項改動不會解決它的主要瓶頸。當時繞過 guards 的 oracle 是 115.6 → 6.6 s，但沒有嚴格證明，未直接採用。

### Divisor 符號與 periodic roundoff floor

後續 admission study 發現先前 oracle 診斷把 divisor 寫成 ikΩ+λᵢ−λⱼ。依本研究的生成元與 FFT 慣例，應是 ikΩ−λᵢ+λⱼ。修正後，bath 的近共振項不再有 1e-6 級的 averaging 上界。找不到能便宜放行 qualitative 的 omission guard，不能把問題歸咎為單純 Dyson remainder 太保守。

另一個獨立缺陷是 periodic integrator 把 tolerance×step 要求縮到 roundoff 以下。兩能階解析 reproducer 在 time≈0.562 ns、step≈3.56e-17 ns 時卡住，估計 error 約 7.40e-18，要求卻約 6.95e-29。

修正將 acceptance 與下一步長控制共同使用的門檻設為 max(tolerance×step, 8×eps64×max(1, max|channel|))。只改 acceptance 不夠，controller 仍會繼續縮步。這項 roundoff 修正有獨立解析依據，不等於接受原本失敗的 omission prototype。

### Retain resonant sidebands，拒絕過大的 residual certificate

R 原型在 modal interaction picture 積分保留的近共振 bath/probe edges，只對其他 fast 分量求 omission bound。201 點 qualitative acquire 為 108.334 → 10.341 s，21 點 strict population 差 7.76e-7，peak index 同為 10。這是 study 的獨立比較，不是下述正式 pair 的基線。

P 原型改用 density-specific residual envelope，但上界為 0.109912，超過 1e-3 段落 budget，也超過 1e-2 population allowance。triangle inequality 丟失相位抵消，不能用 sampled residual 小來替代 certificate。

R 得到約 9.05e-4 至 9.32e-4 的 finite-Fourier bound，最後採用限制明確的 calibrated route。正式 qualitative 201 點 pair 為 95.10 → 10.04 s，foreign 1.1%／5.6%，21 點與 201 點都從 DOP853 改為 Floquet，其餘 34 案 routing 不變。

當時只在原 Duhamel guard 拒絕後試此 route。|k|≤4 中 |divisor|≤0.1/ns 的 edges 被保留，沒有第二個更大 window。證書按完整 2 MHz probe bins 共用，不是只認證 bin 中央點。

四項假設仍未認證：

- Fourier sample grid 以外的 aliasing tail。
- 完整 finite-Fourier generator 的 contractivity。
- ODE 的 global error。
- norm 與 eigensystem 計算的 directed rounding。

嚴格 local ODE tolerance 與 regression 只能校準結果，不能補齊這四項證明。

### 共用 certificate，再比較 direct 與 Floquet

bath 與 modal basis condition 不依 probe bin。正式 factoring 把它們每 certificate set 算一次，再合併各 bin 的 probe 部分。真實 acquires 的 q/r/e/tail 最大 delta=0，36 案 routing 不變。qualitative 201 點的一組 clean pair 為 10.503 → 8.055 s，foreign 5.70%／5.64%，後續 committed-tree pairs 仍 contended。不能再把這份 bath 共用列成尚未實作的候選。

下一輪成本排序同時估 duration、dimension、spectral width、signed carriers、gains、quadrature norms、budget、acquire-wide frame reuse 與 distinct probe bins。不讀 density、cache warmth、workers 或 chunks，順序在 guards 前固定。

throughput 201 點的兩組 clean pairs 為 10.043 → 1.760 s、8.639 → 1.509 s，比都約 0.175，routing 從 Floquet 改 DOP853。這不是前述 15.80 倍 modal BLAS 的重測。

qualitative 仍走 Floquet。它的 wall pairs 受干擾，但 work traces、九個 certificates、prepare keys、apply composition、chunks 與 density hash 都相同，只多 0.34 ms 的 cost-key 工作。因此沒有以 contended 的 7–11% wall 差宣稱實質退步。

## 被否決的選項與理由

- 直接繞過 guards 使用 17.6 倍 oracle。divisor 符號錯誤推翻了原本的證明希望。
- 以 P residual 放行 qualitative。0.109912 明確超標，不是接近門檻的浮點問題。
- 把 R 的 finite-Fourier bound 宣稱成原連續方程的 rigorous guard。aliasing、contractivity、global ODE 與 rounding 假設未完成。
- 固定 Floquet-first。短段 throughput 的 matched direct-first pairs 已提供反例。
- 用 raw peak index 相同判定 throughput 的 Stark shift 通過。strict contrast 太小，物理驗收仍不可辨識。

## 留下的問題

成本模型只在 16-level S9、約 0.5 與 7.29 GHz、16–512 ns、gain 0.02–0.5 校準。其他 spectra、ports 或 dimensions 的最佳順序仍是外推。

截止前最後一次重掃，qualitative guard 仍占約 74% diagnostic CPU，即使 bath factoring 已做完。這是新的瓶頸快照，不是新的 A/B 收益，也不授權放寬 guard。

## 依據

本文整合五點 frame 驗收、modal BLAS、frame、admission、retained-sideband/residual studies 與正式整合報告。原始工作檔未入庫，本文沒有新增數值驗證。

[Planner evidence](2026-10-18-planner-evidence-and-overhead.md)說明重掃與成本模型為何不能代替 whole-acquire admission；[direct integration](2026-10-18-direct-integration.md)保留 DOP853 的校準限制。
