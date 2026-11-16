---
date: 2026-11-16
status: active
superseded_by: null
related:
  - docs/adr/0078-evolution-module-boundary.md
  - lib/zcu_tools/simulate/evolution/README.md
  - docs/discussions/2026-11-16-floquet-kept-price.md
basis:
  repo_commit: f33f7b51ff5d52326eb6db3c45f6cc4e44694f96
  external_sources: none
---

# Floquet 的拒絕成本與單次 evolve 內的 frame 重用

## 結論

當時採用兩項不改收拒的優化。
先把既有拒絕條件移到最早可判定的位置，再讓一次 evolve 內的 exact-key frame 只建一次。
沒有把拒絕機率算進價格，沒有跨 key 或跨 evolve 保留 frame，也沒有採用建 frame 前的 trace 下界。

Guard staging 的拒絕時間約下降 8% 至 19%。
Frame reuse 把指定 prefix／deferred workload 的每次 frame 建構由三次降為兩次，Floquet prepare CPU 從 1.99 s 降為 1.33 s。
這些分別來自下文的 staging 前後量測與 frame-reuse 配對量測，不能相加成整體加速。
兩次改動的完整 benchmark 都沒有超過 10% 的退步。

數值準備的壽命由 [ADR-0078](../adr/0078-evolution-module-boundary.md#損耗漂移下的結構與-preparation-lifetime) 擁有。
Frame fact 的契約與 `prepare_builds` 讀法見 [evolution README](../../lib/zcu_tools/simulate/evolution/README.md#facts-and-algorithm-groups) 及其 [telemetry 說明](../../lib/zcu_tools/simulate/evolution/README.md#routing-telemetry)。

## 問題

使用者於 2026-11-16 要求「找有沒有便宜的dispatch手段」。
Floquet guard 原本先解一週期 ODE，建立 frame 與 sideband 資料，之後才判定各列收拒。
放行時 apply 可沿用準備，被拒時準備就白費。
[價格重校](2026-11-16-floquet-kept-price.md) 能減少選錯先後的情況，不能消除仍被先試候選的拒絕成本。

調查一開始也誤讀了 admission。
正確區分是 legacy mask 通過就接受，modal 與 periodic 只是不同執行路徑。
只有 legacy 未通過的列，才看 sideband eligibility 與 bound。
所以證明 sideband 必拒，還不足以提前拒絕整個 Floquet。
這項修正決定了 staging 與 pre-frame predictor 的適用範圍。

## 考慮過的選項與比較

### 把已有拒絕條件提前，不新增 predictor

採用的 staging 只省掉已知失敗後不再需要的工作。
Frame residual／overlap 失敗就停，eigensystem 無效就不再算成功路徑的資料。
Legacy 未通過時，把既有 `spread*T > 14` 判定提前到 sideband candidates 與 probe-bin 證書之前。
證書累加非負項，部分 bound 已超過既有 `0.95 * budget`，且同 bin 所有待判列都必拒時才短路。
這兩個門檻沿用當時的 [Floquet 實作](../../lib/zcu_tools/simulate/evolution/kernels/floquet.py) 與 [sideband 實作](../../lib/zcu_tools/simulate/evolution/kernels/_floquet_sidebands.py)，沒有新設數值界線。

這個選項省不掉第一次一週期 ODE。
它與早先否決的 bath-only 方案不同，後者只查一個分量，收益也只是階段計時估算。
本次重排要求公開 Engine 收拒等價，並實測拒絕時間下降後才合併。

「Staging 前後量測」使用工具擁有的合成模型，沒有使用使用者 FTF 設定。
公開 Engine 的七種情境共十列，family 與收拒逐列相同，density 對 base 在 `2e-12` 內。
情境包括 legacy 接受、sideband 接受、全拒絕，以及同 frame／bin 內混合時長或 gain。
拒絕計時的結果如下，原始工作檔未入庫。

| 量測組 | 結果 | 限制 |
| --- | --- | --- |
| Single 12-level，首次／warm 分離補測 | warm median 0.639 s → 0.574 s，−10.26% | 兩輪 ABBA，fresh process 不等於空 disk cache |
| Single 16-level，首次／warm 分離補測 | warm median 2.675 s → 2.348 s，−12.22% | 與上一列相同協定 |
| 其餘原始有效拒絕對照 | 四例下降 8.41% 至 18.84% | 原先各側一次，不冒稱多次配對中位數 |
| 完整 simulator benchmark | 最大 wall 增幅 +4.87% | 所有 wall／CPU 比較均低於 +10% |

原始 single-12 對照曾出現 +160.07%，當時沒有直接歸咎編譯或忽略。
另做首次／warm 分離後，兩側都出現約 1.3 s 的首次額外成本，warm guard 沒有重現退步。
這些數字來自 staging 的補充量測，不是換輸入後挑最好樣本。
當時沒有 profiler，仍不能把首次額外成本細分成 import、JIT 或 machine-code 載入。

### 先數重算，再決定 memo 的單位

「拒絕重算 spike」用 committed `INTEGER_MODEL` 的 16 態來源，建立共用 prefix 的兩條 timeline，之後各自 deferred 解析。
每次 evolve 進入 guard 三次，但只有兩種 frame ODE input。
自然相位 control 的三個 row 都不同，仍有一個 frame input 重建一次。
刻意對齊相位後，才同時出現一個完全相同的負向 row。
這是合成 exact-repeat workload，不代表一般批次的重複率。

因此只記負向 row 會漏掉自然相位 control。
採用方案改成由 WorkPlan 持有 exact-key frame，prefix 與 deferred 的後續分類共用。
Key 包含 source 身份、budget 與完整 prepare key。
有效、無效，以及所有 row 都被拒的 frame 都適用。
Row admission 仍按各自參數重算，不把 frame 重用變成整個 key 的放行或拒絕。

「Frame-reuse 配對量測」沿用自然相位的 prefix／deferred workload，對兩側各用兩個 process。
十六次 evolve 的 family、拒絕列數與 density 完全相同，最大 density 差為零。
每次 frame 建構由三次降為兩次。
公開 telemetry 的 warm Floquet prepare CPU 從約 1.99 s 降到 1.33 s，約 −33%。
整條 evolve 的 warm wall 中位數只下降 0.61%，完整 benchmark 的最大 wall 增幅為 +3.90%。
這些數字來自 frame-reuse 的配對報告與驗收覆核，沒有將 prepare 收益冒稱為整條序列收益。

### 研究建 frame 前的 trace 下界

候選下界是 `spread(Re λ) ≥ −Re(trace(L0))/(d²−1)`。
在理想 unitary frame 與零模條件下，它可能由 rates 與 coherence decay 算出，避免先解 ODE。
但實作需要涵蓋 rotation 誤差與捨入方向的單側證書，還須獨立證明 legacy 必拒。

同一份「拒絕重算 spike」核對已審閱的拒絕例。
在 committed 單一 fluxonium 的八個歷史拒絕列中，trace predicate 命中 0/8，最大 lower-bound×duration 為 0.558，離既有門檻 14 很遠。
已審閱的拒絕例都沒有證明走到 `spread*T > 14` 分支。
因此該特定分支的命中率分母是零，不能把 0/8 當成分支 recall，也不能由零次預測聲稱 predictor 可靠。
研究沒有得到單側證明，當時關閉此方向，不落地新前置拒絕條件。

本次遵循 [2026-10-18 Floquet 討論](2026-10-18-two-tone-floquet.md) 的不放寬 guard 先例。
Frame 只在一次 evolve 內共用，遵循 ADR-0078 的 preparation lifetime。
原本一次 classify 內已有按 key 去重，本次區別的是 prefix 與 deferred 跨 classify 的重算，不把既有去重再算一次收益。

## 被否決的選項與理由

- 把拒絕機率算進價格，或固定改為 direct-first。這可能跳過會放行且更快的 Floquet。保留方案的 committed 11 點 detuning 配對已量到 Floquet 快 1.81 倍，來源見 [價格討論](2026-11-16-floquet-kept-price.md)。機率不是只擋必拒列的證明。
- 跨 key 重用 frame。不同 Stark 輸入沒有可用的誤差證書。
- 跨 evolve 保留 frame。這違反數值準備的既有生命週期。
- 任一 row 拒絕便記成整個 frame key 拒絕。Frame key 不含時長，保留研究的 committed 混合 duration 批次有七列接受、四列拒絕。若重播前已存整 key denial，會錯擋 7/11 列。這是由實測批次推導的反事實，不是實作 memo 後量到的錯擋率，也不否決已證明的 frame-wide invalidity。
- 用時長或 condition number 一刀切。審閱已找到相同時長有收有拒的輸入，condition number 也不能單獨預測 bound。本文不收錄使用者工作設定的衍生數字。
- 只存 invocation-local 負向 row。自然相位 workload 重複的是 frame input，不是 row，這種 memo 省不到該次重算。
- 直接採用 trace 下界。已審閱樣本沒有目標分支，也沒有浮點單側證書或 legacy 必拒證明。

## 留下的問題

跨 evolve 的拒絕記憶仍等實際跨呼叫重算證據。
單次 evolve 的 spike 沒有回答這個問題。
若之後做拒絕記憶，必須分清 exact row refusal 與證明過的 frame-wide invalidity。
查詢只能讀跨呼叫 history，不能因觀察而改變之後 evolve 的選路。

若出現實際 `spread*T > 14` 拒絕例，才重開 pre-frame trace 下界研究。
即使找到該例，仍須補 legacy 與數值單側裕度的證明。

這次也修正了效能驗收的量測單位。
多段 workload 的總 wall 只驗整體沒有變慢，改動段是否變快用同一 workload 的公開 telemetry 量。
這遵循 [planner 討論](2026-10-18-planner-evidence-and-overhead.md) 的整體退步檢查，沒有把階段收益拿來抵銷完整 benchmark 失敗。

首次呼叫還必須比較相同 Numba disk-cache 狀態。
Frame-reuse 原報告的 +16.55% 混合了 branch 的冷、熱 process，對上 base 的熱 process，不能當作改動造成的退步。
同為熱 cache 的首次中位數是 base 47.38 s、branch 47.04 s，約 −0.7%。
這組修正來自同輪原始 process 記錄的覆核，沒有刪除冷樣本或放寬 10% 門檻。
[Simulator cache 說明](../../lib/zcu_tools/simulate/README.md#numba-machine-code-cache) 記載 namespace 取決於 simulator 原始碼與依賴版本的 hash。
不同版本各自首次啟動所付的編譯成本，不能混進 warm 演化比較。

## 依據

本文整理便宜 dispatch 調查及其審閱修正、拒絕重算 spike、staging 前後報告與首次／warm 補測、frame-reuse 配對報告與驗收覆核。
原始工作檔未入庫，本文列出必要結果與限制。
本篇沒有新增模擬量測，也沒有收錄使用者 FTF 工作設定的衍生數字。

- 「Staging 前後量測」提供 Engine 等價結果、拒絕時間與 +4.87% benchmark 上限，首次／warm 補測修正了 single-12 異常的判讀。
- 「拒絕重算 spike」提供 row／frame 重複次數、0/8 trace 結果與證明缺口。
- 「Frame-reuse 配對量測」及驗收覆核提供 frame 次數、prepare CPU、完整 wall、benchmark 與 disk-cache 狀態的比較。
- [ADR-0078](../adr/0078-evolution-module-boundary.md) 與 [evolution README](../../lib/zcu_tools/simulate/evolution/README.md) 保存數值準備壽命、frame fact 與 telemetry 契約。
