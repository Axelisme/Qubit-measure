---
date: 2026-10-18
status: active
superseded_by: null
related:
  - lib/zcu_tools/simulate/adc/README.md
  - lib/zcu_tools/experiment/ftf/README.md
  - docs/adr/draft/simulation-environments.md
basis:
  - "repo main@f152eff67：研究整理的截止版本"
  - "repo@b0cb487f3：discrete-sum study 可取得的 baseline；prototype 未進入 repo 歷史"
---

# Readout 從 sampled fields 到 guarded discrete sums

## 結論

flat integrated readout 只需要 ADC samples 的離散和。研究因此改為對狀態條件場的 exponentials 求 geometric sums，省掉先建立 sampled fields 再加總的工作。

這不是連續時間積分，也不改 evolution populations。遇到 shaped pulses、nonuniform grids、trace requests 或不良 eigensystems，保留原 field route。正式採用時 readout backend 從 v5 升為 v6，不留舊 replay 相容層。

本文保存 main `f152eff67` 以前的推導與量測。[ADC README](../../lib/zcu_tools/simulate/adc/README.md)與 [FTF README](../../lib/zcu_tools/experiment/ftf/README.md)擁有契約。

## 問題

S2 是 2001 點 readout frequency sweep。早期重掃的 warm acquire 為 0.574 s，readout 約占 86% CPU。另一個 instrumented diagnostic 為 0.737 s，其中 responses 用 0.583 CPU-s。不能拿 instrumented wall 當 speedup 分母。

當時的 flat-field path 已共用 frequency-independent Schur bases 與 projected relative grids，並不是每個 carrier 都建立完整 dense moment tensor。但仍有 2001 次 Sylvester solves，以及 sampled-field、calibration、stack 與 sum 成本。只優化 evolution 不會解決 S2。

## 考慮過的選項與比較

### 在場模型內直接求樣本和

常數輸入的 moment solution 滿足 (M+iωI)X−XQ=B。Q 是 population generator，M 是 homogeneous moment generator。良態 eigenbases 下，係數分母是 μₐ+iω−λᵦ，輸出成為 exponentials 的線性組合。

對均勻 ADC grid tⱼ=a+jh，計算：

```text
sum_j exp(z*t_j) = exp(z*a) * expm1(N*z*h) / expm1(z*h)
```

sampled-alias 極限使用 N×exp(z×a)。這裡沒有乘 sample period，因為原 API 要離散 sample sum。pulse active incident、causal on-step 與 opposite off-step 分別加總，off-step 保留 exp(QT) 的 population transfer，不能省略 ringdown。

delay、mismatch、microwave conjugation、receiver gain/phase/offset 都保留原順序。noise 仍由原 sampling 加入。直接和不負責 normalization，也不畫 traces。

### 先做保守 guards，逐 content fallback

研究要求兩個 eigenbasis condition numbers<1e4、reconstruction residuals<1e-13、spectral separation≥1e-10，並保留原 Sylvester cancellation/residual guards。不符合時退原 field route，不假設所有 Q 都可 diagonalize。

正式版的 uniformity 依 endpoint-fitted lattice 與 coordinate ULPs 判定，避免大 absolute start 的浮點量化被固定 ns tolerance 誤判。single sample 也可處理。fallback 以 content 為單位，不讓一個 unsupported content 拖整批退回。

exact-content sums cache 與 trace field cache 分開。include_traces、decimated、shaped、nonuniform 或 defective-generator requests 都保留 field route 值，沒有為了加速而改 trace 契約。

### Study 與正式速度證據分開

study 比較 baseline 與 prototype。只有 baseline `b0cb487f3` 可取得，prototype 未進入 repo 歷史。正式機制見後來 land 的實作與 [ADC README](../../lib/zcu_tools/simulate/adc/README.md)。量測用兩 workers、native threads=1、CPU 0/1，獨立 process 依案交錯。每版先 full-target warmup，再清 Engine prepare caches，之後量一次完整 acquire，沒有 repeated medians。

| 場景 | Points | Acquire wall，舊 → 原型 | Foreign，舊／新 | 判讀 |
| --- | ---: | ---: | ---: | --- |
| L3 低頻 length | 201 | 0.2857 → 0.1966 s | 3.2%／3.7% | Clean，省 31.2% |
| L5 Ramsey | 201 | 0.2809 → 0.2070 s | 5.6%／4.5% | Clean，省 26.3% |
| F2 one-tone | 21 | 0.04648 → 0.03353 s | 4.7%／0.0% | Clean，省 27.9% |
| S2 readout | 2001 | 0.6707 → 0.4919 s | 24.3%／13.5% | Contended，只觀測省 26.7% |

S7 Ramsey 觀測省 42.3%，但 foreign 為 21.0%／4.1%。研究按當時 margin 規則接受其候選價值，沒有稱它 clean。S7 echo 與 F3 DC 仍 inconclusive，未做 timing 的其他 direct-route 場景也不由 CPU share 推定 ≥20% wall gain。

正式整合重新做 clean pairs，S2 為 0.5154 → 0.3823 s，foreign 5.5%／8.5%；L3 為 0.2876 → 0.1900 s，foreign 4.5%／0.0%；S6 為 0.5541 → 0.4813 s，foreign 5.4%／3.7%。這些才解除 study 的 S2 contamination 問題，不與前表跨版本相除。

34 個原尺寸場景的 acquired IQ 最大 relative difference 為 6.05e-14，evolution densities、populations 與 routing 全不變。direct sums 也覆蓋 echo、T1 與 zigzag 的完整 readout geometries，unsupported contents 保留 fallback。

## 被否決的選項與理由

- 以連續積分代替 sample sum。它改變 ADC 格點與 sampled alias，不符合原契約。
- 無條件使用 eigenbasis formula。defective 或近 defective generators 需要原 Schur／matrix-exponential field path。
- 整批 fallback。少數 shaped 或不良 contents 會丟掉其他可直接求和的收益。
- 將 study 的 S2 26.7% 稱為 clean PASS。兩次最終版本仍 contended，正式量測必須另做。
- 默默接受舊 readout replay。數值 backend 已改 v6，舊 replay 明確拒絕，沒有 v5 compatibility layer。

## 留下的問題

ADC 的 population tolerance=1e-6 與 segment budget=1e-3 服務不同邊界。environment 顯式把 raw density 轉為 populations 並 normalize，direct sum 不改這一步。sampled conditional moments 也不等於隨機 jump trajectories。

study 早期 timing 沒有逐 process 保存 MemAvailable，只手動觀測約 18 GiB，後四次記錄為 19.0–19.6 GiB。沒有看到低於 6 GiB，但早期證據仍不完整。研究也沒有驗證 301-flux sweep 或 production 六 workers 的重複 throughput。

## 依據

本文轉錄 constant-readout discrete-sum study 與正式整合報告。原始工作檔未入庫。delay、mismatch、ringdown、single sample、sampled alias、nonuniform、defective Q 與 decimated traces 的比較都在原研究完成，本文未重跑。

[模擬環境 ADR](../adr/draft/simulation-environments.md)提供 environment、evolution 與 ADC 邊界的背景，本文的 v6 描述是截止版本的歷史事實，不改寫 ADR 當時的版本文字。
