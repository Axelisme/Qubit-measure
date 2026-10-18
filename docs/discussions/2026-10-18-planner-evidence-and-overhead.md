---
date: 2026-10-18
status: active
superseded_by: null
related:
  - docs/adr/0077-numerical-plans-and-replay.md
  - docs/discussions/2026-10-18-dyson-input-reuse.md
  - docs/discussions/2026-10-18-two-tone-floquet.md
  - lib/zcu_tools/simulate/evolution/README.md
basis:
  - "repo main@f152eff67：研究整理的截止版本"
  - "repo@b5f8b8f23：span-aware planner 設計研究基線"
  - "repo@88215e1d6、16b1dde43：stage calibration 與 acquire validation 時的 main"
  - "repo@12f9e96a3、c9dd66071：最後兩輪重掃基線"
---

# Planner 的 evidence admission、固定成本與失敗記錄

## 結論

constant planner 的 accuracy 通過，但 whole-acquire performance admission 失敗，沒有任何 admitted production profile。144 組 matched 比較有 100 組 candidate 較慢。warm 中位數加總在 workers=1／2 分別退步 30.73%／4.37%。

stage model 能解釋算法成本，不能取代整個 acquire 的 wall gate。短 acquire 的 planning、cover、guards 與 admission 固定成本，足以吃掉數值部分的收益。

本文保存 main `f152eff67` 時已存在但未啟用的 planner 試驗。它不規定後續重構要保留哪些類別或 API。架構裁決由 [ADR-0077](../adr/0077-numerical-plans-and-replay.md)擁有。

## 問題

先按固定順序挑 family，再做 shared preparation，無法比較整個 acquire 的 builds、input reuse、apply 與 lifetime。相同輸入的 Dyson、不同輸入的 plateau、長 ramp 與 singleton T6 的成本結構不同，不能只給每 row 一個 family price。

另一個風險是把校準模型的估算當成 production 證據。candidate 即使更準，仍可能較慢；舊 kernel revision 的 coefficients 也不能直接沿用到新 numerical closure。

## 考慮過的選項與比較

### Span-aware whole-acquire 設計與後續修正

`b5f8b8f23` 的設計研究選 span-aware resource planner。它把 shared build 視為一個有 consumers 與 lifetime 的工作。candidates 可覆蓋完整 pulse 或 sequence，不先把每個 event interval 固定分派。prototype 只在保存的資料上做 scheduling arithmetic，沒有跑新 simulation，也沒有新 wall 收益。

原研究曾建議 kernels 用通用 declarations 描述能力。使用者 2026-10-18 明確修正：

「看起來planner與使用的算法是本質高偶合，那planner可以清楚的區分每個算法是合理的架構，不需要特別抽象化，然後我認為planner先彙整全部資訊再決定每段的演化算法是正確的思路」。

因此早期 declarations 提案不是後續架構的依據。使用者同日也說「Q2只要是有助於分類的資訊，planner都可以知道。」這裡保留原案與修正，不把討論當成另一份決策正本。

### 截止版本的未啟用機制

| 機制 | 試圖回答的問題 | 證據與限制 |
| --- | --- | --- |
| Declarations、family profiles | 記錄算法身分、stage work、resource estimates | 未讀資料不會自行改善 production routing |
| Shadow plan 與 cover graph | 全 acquire candidates 的 build/reuse/critical path 能否勝過固定順序 | 不執行演化，不能產生 wall gain 證據 |
| Constant planner | 直接辨認 T4、Dyson、Hilbert period、T6、DOP853，做 bounded deterministic cover search | 保留完整 incumbent；不能把較佳 estimated score 當作 admission |
| Calibration 與 evidence profile | 以 one-core coefficients、source closure、domain/resources certificates 固定校準適用範圍 | 實際沒有通過 whole-acquire 的 production dataset |
| Provider 與 legacy-plan adapter | 把候選／既有路由轉成有 ownership 的執行計畫 | 執行正確不代表規劃成本值得付 |
| Cache reservation、consumer references | 預先表達 shared build 的容量與生命週期 | 未建立一般 process RAM 上界，cache bytes 也不等於 RSS |

這些機制曾作為非預設基礎交付存在。constant-profile admission 在 source、kernel、dependency、revision 或實際 domain/resources 不符時，會在 assignment 前保留 legacy。developer evaluation 只容許缺少 acquire evidence，不會自動升級 profile。

pinned plan 的用意是重播既定 numerical assignment，不在 replay 時重新選算法。它也不能替候選省略原數值 guards 或吞掉 prepare/apply failure。

### 20 fit、16 held-out 與完整 acquire gate

stage calibration 先將 build、bind、apply 分開測量，再用 independent reference 驗 accuracy。20 fit 加 16 held-out 的固定 domain 是三能階 synthetic secular model，不是 16-level FTF。

關鍵輸入為 energies [0, 0.13, 0.26] GHz、固定 directional rates 與 dephasing、一個 Hermitian X/Y port，以及 mixed diagonal／coherent 兩個 exact inputs。每 acquire 四 rows 共一 key、duration=2 ns、budget=1e-3。carrier/gain controls 為 0.3/0.0001、0.3/0.04、−0.02/0.002、1.2/0.001 與 idle，另按 phase 切 fit／held-out。

實測只有 workers=1／2、cache limit=1 MiB。logical six-worker unit case 不是 actual six-core performance。沒有把 FTF、其他 dimensions、baths、inputs、multi-interval 或 cache capacities 加進 admitted domain。

whole-acquire 計時包住完整 Engine call，包括 admission/checks、planning、guards、prepare、bind、apply、pool 與 native-limit lifecycle。independent full Lindblad reference 在計時外，使用 SciPy DOP853，rtol=1e-10、atol=1e-12、max_step=0.05 ns，不共用受測 RHS。

acquire validation 時 main 為 `16b1dde43`，calibration lane 從 `7687b69ca` 開始，原 numeric source closure 經核對後沿用。第二個 bounded run 完成所有 36 cases，workers=1／2，cold 各一次、warm 各三次，legacy/candidate 交錯次序。cold 指 prepare cache，不是 JIT cold。每組 warm 取三次中位數。

| Workers | Prepare cache | Legacy matched median sums | Candidate sums | 判讀 |
| ---: | --- | ---: | ---: | --- |
| 1 | Cold | 0.317760 s | 0.172818 s | 加總較快，但 23/36 cases 較慢 |
| 1 | Warm | 0.112650 s | 0.147269 s | 36/36 較慢，退步 30.73% |
| 2 | Cold | 1.074498 s | 0.968736 s | 18/36 較慢 |
| 2 | Warm | 0.572267 s | 0.597278 s | 23/36 較慢，退步 4.37% |

各 cell 的加總不是一次大型 acquire wall。兩 workers 的短 calls 有明顯 spread，不能只取最好的一次。報告記錄 CPU affinity 0/1、native threads=1 與交錯方式，但沒有把本表標成逐 pair uncontended。

144 組有 100 組較慢，最大 ratio 約 2.185。candidate population maxabs 約 1.48e-6、完整三階 density maxabs 約 4.15e-6，均通過 1e-2。accuracy 不會抵銷 performance failure。

最後的 port binding 與 argument validation 修正後沒有重跑 heavy validation，因此舊 wall 不是 final tree 的新量測。既然前一狀態已未通過，也沒有因此發布 admitted profile。

## 被否決的選項與理由

- 以 stage predictions 取代 whole-acquire gate。固定成本恰恰在 stages 外，warm regression 已提供反例。
- 把 cold 加總較快當整體 PASS。它不能抵銷每 case 或 warm 回歸。
- 刪掉較慢 held-out、挑快 cases、擴大 domain 或拿舊 16-level wall 填補失敗。這些都會改變原先驗證問題。
- 建通用 memory accounting 或成本 plugin 作為 planner 前提。使用者要求直接認算法；大型 T6 allocations 可有限估算，其他未知配置仍是未知，不能填零冒充上界。
- 把 synthetic weak-coupling factors 的 kernel speedup 當 acquisition gains。尚未驗證的 guards 與 representation transitions，不能由 component timings 推成 production admission。

## 留下的問題

### 驗收與 profiling 走到哪裡

早期 `ee701ee7c`、workers=1、小型 warmup 的 profiling 含 target-specific JIT。S3／S5 singleton T6 prepare 是 36.197／20.294 s，S9 guard 與 prepare 重建相同 frame 約 10.3 s。S4／S6 被 readout 拒絕，RWA 沒有命中。這些證據不支持立刻另寫 coherent-input 算法，也不是穩態 throughput。

`dcdc02353` 對 `ee701ee7c` 的同輪驗收，把 S1 classification 3.813 → 0.095 s、acquire 16.762 → 6.889 s。S3 201 點 7.268 s、S9 五點 19.440 s 仍未達速度目標。S8 移出驗收，S9 shift 維持 unresolved。該輪報告、tables、conclusions 是同一次驗證，不算三份獨立證據。

`b5f8b8f23` 的後續全驗收仍有 F9／F10 warm 3.589／5.175 s，超過 3 s。F15 全 grid unresolved，F14 完整 strict 未完成，F1／F4／F8 的回退主要在 first process。主要驗收 workers=2、native threads=1；S 用 full-target warmup，L/F 沿用 tiny warmup，不能都稱 warm steady state。censored runs 保留，不算 PASS。

之後完整重掃分開 first-use、warm 與 instrumented diagnostics。r3 是 34 案，供 ordinary period 零 routing 的停止判斷。r5 在 `12f9e96a3` 擴為 36 案，24 個 numeric targets 過，L3／L4 仍只記 contended upper bounds。當時 direct-first 的 throughput screening 觀測約 82% 降幅，但尚無 accuracy，不能提前當 admission。含模板佔位的同輪草稿不是另一份實驗。

r6 在 `c9dd66071` 完成 36 案、workers=2、full-target warmup 後清 cache。L7strong／throughput／L8 wall 為 3.507／1.489／0.963 s。34 案至少一次 uncontended，兩個極短場景仍 contended。F15wide guard 占 31.7%、period prepare 占 28.5% diagnostic CPU，S9 qualitative guard 占 74%。

r5/r6 沒有 interleaved，instrumentation 與負載也不同。這些數字只定位歷史瓶頸，不能跨輪相除宣稱因果收益或 2% 不退步。worker elapsed 加總也不是 acquire wall。

### 後續需要的證據

新 planner 必須重新降低 planning 固定成本，再對當時最新 legacy 做完整 matched validation。以上資料未證明跨 evolve workspace、prefix lifetime 或重算消除的收益，也沒有替未驗證 representations 取得 admission。

## 依據

本文整合 span-aware 設計、stage observations、whole-acquire calibration、初始與完整驗收，以及多輪 reprofile 報告。原始工作檔未入庫。本文保留失敗、未量與 censored 邊界，不補成成功結果。

[Dyson 研究](2026-10-18-dyson-input-reuse.md)、[constant builders](2026-10-18-constant-period-builders.md)、[shaped preparation](2026-10-18-shaped-period-and-gates.md)、[two-tone Floquet](2026-10-18-two-tone-floquet.md)、[direct integration](2026-10-18-direct-integration.md)與 [readout sums](2026-10-18-readout-discrete-sums.md)各自保留算法證據，避免把已完成的數值優化再算成 planner 的預期收益。
