---
date: 2026-11-14
status: superseded
superseded_by: 2026-11-15-t6-channel-removal.md
related:
  - docs/adr/0078-evolution-module-boundary.md
  - lib/zcu_tools/simulate/evolution/README.md
  - docs/discussions/2026-10-18-direct-integration.md
  - docs/discussions/2026-10-18-planner-evidence-and-overhead.md
basis:
  - "repo base@84bdeed3d92375c3a0f0856129dff8a22b51950d"
  - "repo integration@5214e2275e2bb7694a70cf77710b945554ea85f2：落地候選的實作，後續 promotion 只增加本文與索引"
---

# t6 的 direct 加速策略與職責歸屬

## 結論

這次保留了 `t6` 的名字、適用域、guard 與「失敗就 raise」。
它仍是 constant 單 tone 的最後一道精確解，不新增 family，也不做 RWA。
內部保留原有的 channel 做法，新增 direct 做法。
Direct 在 H0 旋轉座標直接積分每一列的 density，使用完整的 secular bath。

Direct 的局部容差跟每段 budget B 走，和 `dop853` 相同。
rtol = min(1e-4, 0.03B)，atol = min(1e-6, 0.0003B)。
兩者共用 `dop853.advance_endpoints`，不另留一套積分器。
這些局部控制不保證整條序列的最終誤差，仍須量測最終完整 density。

每組 prepare key 在 `N * C_direct < C_channel` 時選 direct，否則選 channel。
N 是完整 acquire 中該 key 的列數，不是 worker 子集或 routing history 的列數。
當時沿用 [evolution README](../../lib/zcu_tools/simulate/evolution/README.md#kernel-summary) 的價格式，單位是秒：

- `C_direct = 0.0012*d*max(1, f/0.582)*(M/21)^3`。
- `C_channel = 31*(M/21)^5`。

M 是能階數，d 是長度，單位 ns，f 是 carrier 絕對值，單位 GHz。
定價、完整 acquire 的 key inventory 與選擇由 dispatch helper `_t6_dispatch` 擁有。
第一次 `t6.prepare` 才觸發 inventory。
Kernel 消費選擇並建置數值產品，不在 worker 子集重新定價。
現行契約由 evolution README 與 [ADR-0078](../adr/0078-evolution-module-boundary.md) 擁有，本文只記錄當時的比較與裁決。

四個真實場景各有 21 能階、24 點。
誤差是整條序列最後完整 density 相對獨立 reference 的最大元素絕對差。
下表是 `t6` 的 first-evolve wall，單位為秒。

| 場景 | 誤差 | 新 wall | 舊 wall |
| --- | ---: | ---: | ---: |
| 掃長度 | 1.16e-4 | 6.3 | 736 |
| 固定 gate 掃 idle | 2.58e-5 | 1.15 | 31.4 |
| 掃 carrier | 5.95e-4 | 28.5 | 872 |
| 可變 gate 序列 | 6.08e-5 | 1.18 | 731 |

四個場景都通過最終 density 誤差 ≤ 2e-3。
Benchmark 的 30 格同 session 成對比值中位數介於 0.960 至 1.033，都通過不慢超過 10% 的門檻。
這些結果沿用先前完成的量測，不是上述最後實作 commit 的新耗時量測。
最後的 ownership 重構核對了四場景的輸入、reference 身分與全部 52 個 key，選擇都相同，才沿用結果。

方向與 direct 容差由使用者親裁。
兩種做法的成本選擇、dispatch helper owner 與 lazy 時點由 agent 依代裁授權決定，當時仍待使用者覆核。
若使用者之後推翻，本文將在文末加後記，不把當時的代裁改寫成親裁。

## 問題

`t6` 是第一批真實工作負載量表上最慢的路徑。
每組 flux、port、carrier、gain 與長度都先建一個 441×441 的 channel。
每組耗時 56 至 65 秒，幾乎不隨脈衝長度改變。
對掃長度這種每個 key 只用一次的工作負載，建置成本無法攤提。

Spike 改在旋轉座標直接積分一個 density。
四個單點都比原 `t6` 更準，耗時比從 20 ns 的 0.017 到 1000 ns 的 0.75。
但 direct 的成本隨長度近似線性增加，每個輸入狀態也都要各積一次。
單點較快不足以證明高重用輸入應放棄 channel。

## 考慮過的選項與比較

### 方向與做法

使用者在 2026-11-12 同意用旋轉座標加速精確解路徑，原話是「這可以，而且可以考慮擴展」與「就是用類似的思路來優化其他場景」。
對整理後的做法，使用者答「對」。
邊界是不做物理近似，不多加 family。

比較的做法有只用 direct、只用 channel、依成本兩者擇一、把 `dop853` 加進 plateau 候選，以及只把 channel 建置的積分改到旋轉座標。
Agent 採用依成本擇一，讓低重用輸入省掉 channel 建置，同時保留高重用輸入可攤提建置的路徑。

### Direct 的容差

選項是固定緊容差 rtol/atol = 2e-11/2e-13，或跟每段 budget 走。
使用者在 2026-11-13 選後者，答「A」。

提出選項時，比較的是原緊容差約 1e-11 級的誤差與 `dop853` 已量到的 1e-6 級誤差。
後者仍比 2e-3 門檻小約三個數量級，因此沒有理由為遠超需求的精度保留固定緊容差。
改完後，掃長度由約 48 秒降到約 6 秒。
四個完整場景的最終誤差則以上方表格為準，不能把單點的 1e-6 級當成整條序列的保證。
Channel 的數值控制沒有改。

### 定價與選擇的 owner

比較的 owner 有 `t6` kernel、kernel 首次 prepare 時呼叫的 dispatch helper、dispatch 預先發布而 kernel 只讀，以及在 family 層將 direct 交給 `dop853`。
另外也考慮移除 channel，或只比單列成本而不看完整 acquire 的列數。

採用 dispatch helper，遵循 ADR-0078 的決策知識分工。
使用者在 2026-10-18 的原話是「決策分類有全部算法的知識，算法部分內聚」。
ADR-0078 已把成本比較與 family 內 builder 選擇交給 dispatch。
Repo 內依整個 acquire inventory 做的選擇也在 dispatch，例如 multitone 的 20 次規則。

初版把選擇放在 kernel，理由是沿用既有 publisher 模式。
Review 指出既有 publisher 都在 dispatch，這個先例不能支持 kernel 擁有定價。
修正後由 `_t6_dispatch` 集中 inventory、價格與選擇，kernel 保留數值產品的建置。
這是遵循既有 ADR，不為 `t6` 加例外。

共用積分也遵循 ADR-0078 的單一 numerical owner。
[Direct integration 討論](2026-10-18-direct-integration.md)記錄了 H0 interaction picture 與 budget 容差的既有證據。
這次沿用其數值做法，但不把舊場景的收益當成新 `t6` 的驗收結果。

### Inventory 的建立時點

選項是在 `create_work_plan` 一律建立，或第一次 `t6.prepare` 才建立。
採用後者，因為不使用 `t6` 的 plan 不該掃描它的 candidate keys。
WorkPlan 保留完整 table，首次 prepare 由 helper 發布完整 inventory。
後續 prefixes 與 workers 共用發布結果，不在各自子集重算。
缺少已發布的 key 仍 fast fail。
Standalone 若沒有完整 geometry，也沒有已發布的 inventory，仍須提供完整 table 與 source。

最初曾把 1 至 2 ms 的變慢歸給一律建立 inventory。
同 session 成對量測後撤回這個歸因，跨時段漂移不能算成 inventory 的成本。
Lazy 時點仍有避免無用掃描的理由，但不以那個差額宣稱收益。

### 耗時驗收

選項是和存下的跨時段基準比，或同 session 和 base 做 ABBA 成對比較。
跨時段比較有 30 格超過 10%，這些失敗保留，不改寫成通過。
它們包含不同 baseline 的比較，不是 30 個獨立 workload。

改採同 session 成對比較後，30 格的比值中位數都在門檻內。
這支持當時的不退步驗收，不證明先前差額來自 lazy inventory。
[Planner 的 evidence 與固定成本討論](2026-10-18-planner-evidence-and-overhead.md)已顯示 stage 估算不能取代完整 acquire 的 wall gate。
這次遵循該先例，價格只用來選策略，耗時仍由實際執行驗收。

## 被否決的選項與理由

- 把 `dop853` 加進 plateau 候選。同一種輸入會多一條 family 路徑，而且當時還要求誤差不較大。
- 全面只用 direct。高重用輸入可能變慢，當時沒有量到足以移除 channel 的證據。
- 只用 channel。它保留了低重用 key 的建置瓶頸，無法處理掃長度的成本。
- 只把 channel 建置改到旋轉座標。沒有速度證據支持這條路。
- 固定緊容差。誤差比門檻小約八個數量級，速度卻不符合這次要解的問題。
- Owner 留在 kernel。需要替 ADR-0078 加例外，且找不到原則性的界線。
- Dispatch 預先發布、kernel 只讀。需要改 standalone 契約與大量測試，只換得誰觸發的差別。
- 在 family 層選擇。違反不多加 family 的邊界。
- 在 `create_work_plan` 一律建 inventory。沒有 `t6` 的 plan 也會付掃描成本。
- 只用跨時段基準作耗時驗收。漂移超過 10%，不能據此判定策略本身的成本。

移除 channel 與不看列數的選擇留待後續，當時沒有一併採用或宣告它們不可行。

## 留下的問題

- Channel 是否仍值得保留。現行價格讓已量的 52 個 key 全選 direct，舊價格則有 3 個 key 選 channel。
- 是否只比單列 direct 與一次 channel 建置。如果不看列數，inventory 機制可以消失，但高重用 key 不再自動選 channel。
- 價格式的能階數冪次與 carrier 因子沒有跨域校準。Direct 校準只有 21 能階、約 0.582 GHz 的資料。
- 價格沒有模型化 prefix 到期後的 channel 重建。
- 一個 shot 內 cavity drive、多次讀出、flux 脈衝，以及長序列，還沒有取得這次優化的量測證據。
- 這次沒有把 `t6` 的加速工作擴展到 `dop853`／`flux_dop853`。共用積分 owner 不代表兩者的擴展驗收已完成。

## 依據

本文以 base `84bdeed3d92375c3a0f0856129dff8a22b51950d` 與實作落地候選 `5214e2275e2bb7694a70cf77710b945554ea85f2` 為版本依據。
後續的本文與索引提交不改數值實作。

實作與現行契約位於以下 tracked 檔案：

- [t6 kernel](../../lib/zcu_tools/simulate/evolution/kernels/t6.py)。
- [t6 dispatch helper](../../lib/zcu_tools/simulate/evolution/_t6_dispatch.py)。

## 後記，2026-11-15

使用者後來選擇移除 channel 與定價分支，保留 direct。
本文的 direct 容差與共用 endpoint 決定未被推翻。
Channel 去留的新比較見 [t6 只保留 direct](2026-11-15-t6-channel-removal.md)，
現行決策見 [ADR-0078](../adr/0078-evolution-module-boundary.md#t6-只保留-direct)。
- [dop853 共用積分 owner](../../lib/zcu_tools/simulate/evolution/kernels/dop853.py)。
- [Evolution README](../../lib/zcu_tools/simulate/evolution/README.md)。
- [Family Pareto 工具入口](../../tools/family_pareto.py)與同組 `tools/family_pareto*.py` 工具。

Spike 報告、量測 receipt 與決策討論原始工作檔沒有入庫。
必要的機制、數字、比較理由與裁決狀態已寫在本文。
本文沒有把跨時段基準當成同 session 成對證據，也沒有把 ownership 重構前的耗時稱作最後 commit 的新量測。
