---
date: 2026-11-22
status: active
superseded_by: null
related:
  - ../adr/0078-evolution-module-boundary.md
  - ../../zcu_lab/simulation_envs/FTF/README.md#工作副本與調參
  - ../../lib/zcu_tools/simulators/ftf/environment/README.md
  - 2026-11-22-ftf-projective-markov.md
  - 2026-11-22-simulator-speed-first-verification.md
basis:
  repo_commit: 73ee9197d69bf1090cf2928e22a31ba463fc807b
  history_measurement_commit: 6f32b2f348715a2584699f55925be7461145d823
  markov_measurement_commit: cc6b252c02a0ee7f6e255bff6d57020dff988323
  external_sources: none
---

# FTF 樣板的 projective floor

## 結論

兩份 FTF 樣板各只把 `numerics.projection_weight_floor` 從 `1e-9` 改為 `1e-5`。欄位仍必填，schema 不提供預設；既有 `ftf.yaml` 不隨樣板更新。使用者 2026-11-21 對「只改樣板這一行」答「1A」。樣板與工作副本的入口見 [FTF 工作副本與調參](../../zcu_lab/simulation_envs/FTF/README.md#工作副本與調參)。

舊 history 模型的原表單首點沒有完整參考，當時接受單一交付例外。Markov 落地後，同模型驗收已涵蓋該點，例外不再適用。這份驗收只隔離 floor 的影響，不驗證模型對獨立參考的精度。

## 問題

舊 FTF 按投影結果保留 history。低機率 history 仍占後續演化列，串接多次讀出的實驗會累積這些列。提高 floor 可減少演化量，但也可能使 density 或逐讀出 population 改變。短 heralded 輸入與多次投影的 chain 都要看，不能把短例的結果外推到原表單。

原先的選項誤稱改 schema 預設就會套用到實驗。欄位其實必填，樣板與工作副本各自明填。使用者同意改兩份樣板，不改 schema 或工作副本。

## 考慮過的選項與比較

首選是 `1e-5`，讓新工作副本採用較高 floor。舊 history 模型的 chain amp_rabi 原表單首點，以同一輸入、seed 123 比較 `1e-5` 和 `1e-6`，最終完整 density 最大元素差為 `1.786e-3`。再細的 `1e-7` 與 `1e-9` 各在 1,800 秒內只完成 13／15 與 6／15 次非末端投影，無最終 density。這些數字來自 commit `6f32b2f34`，該輸入有 Stark 能階反轉警告。兩個候選接近，不等於任何一個對完整參考達標。

使用者 2026-11-21 答「floor選A」，維持 `1e-5`，不再為舊模型補更細參考。對明列原設定首點「精度未驗證」的單次例外，使用者答「那採用建議」。這不是放寬其他驗收，也沒有把逾時解作誤差超標。

[Markov 改動](2026-11-22-ftf-projective-markov.md)之後，floor 對低機率來源態用 identity 列，不丟機率。驗收改用同一模型、同 seed 比 `1e-5` 和不剪枝的 `0`，依 [ADR-0078 的精度門檻](../adr/0078-evolution-module-boundary.md)判讀。六列包括 chain amp_rabi 原表單首點、heralded CP、chain power、chain length、舊 phase 設定與一個 heralded RB seed。兩個 phase 輸入相同，共用一次配對。commit `cc6b252c0` 的全密度最大差為 `6.921e-6`，逐讀出 population 最大差為 `6.964e-6`，六列都低於 `1e-2`。原表單首點有了完整同模型對照；舊 history 的參考缺口仍只描述舊模型。

## 被否決的選項與理由

- `1e-6` 在舊 history 模型保留較多列。缺少完整參考，不能判定它比 `1e-5` 達到更合適的誤差與成本取捨。
- 維持 `1e-9` 或只改個人 `ftf.yaml`，新工作副本仍得原值。舊模型用 `1e-9` 也沒有算完原表單首點。
- `0` 不跳過任何來源列，包含零權重列。它適合當 Markov 同模型參考，不是省演化的樣板值。
- 用累積剪除權重 W 限制剪枝，會引入另一種預算與選列規則。它需要自己的數值與 caller 驗收，這次不採用。

## 留下的問題

[牆鐘工具的固定 YAML](../../tools/data/experiment_wall_time/ftf-settings.yaml)與 [family 比較的固定設定](../../tools/family_pareto_settings.py)不跟著樣板變更。它們留到工具是否改讀標準樣板的議題一併處理。中斷後從 checkpoint 恢復執行也另案，這次不加入。

## 依據

- [FTF environment 的 floor 契約與驗收範圍](../../lib/zcu_tools/simulators/ftf/environment/README.md#cavity-drives)。
- [Markov 模型的取捨](2026-11-22-ftf-projective-markov.md)與[時間門檻、驗證成本的裁決](2026-11-22-simulator-speed-first-verification.md)。樣板值沿用使用者的選擇；參考改成 Markov 同模型對照，因為舊 history 與現行模型不同。
- 原始工作檔未入庫。受測版本、輸入、未完成的範圍和必要數值已列在本文。
