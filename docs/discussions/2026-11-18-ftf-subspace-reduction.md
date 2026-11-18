---
date: 2026-11-18
status: active
superseded_by: null
related:
  - ../adr/0078-evolution-module-boundary.md
  - ../adr/0072-module-boundaries.md
  - ../../zcu_lab/simulation_envs/FTF/field-guide.md
  - 2026-11-17-dop853-speed-route-summary.md
basis:
  repo: c04fb38231de5aa65d01aba931a48db13d48d093
  external: none
---

# FTF 模擬的子空間約化：各做法的結果與結案

## 結論

模擬器不在 `Engine.evolve` 內部自動約化 Hilbert 空間。能量最低 N 態截斷在一般單 tone FTF 掃描上算得準，也比較快，但找不到便宜又嚴格的逐點誤差保證，不符合 [ADR-0078](../adr/0078-evolution-module-boundary.md#先保證正確再優化瓶頸)「guard 在整個參數空間不放行超標」。維度照舊由使用者在設定檔宣告 `retained_levels`，符合 [ADR-0072](../adr/0072-module-boundaries.md) 的「模型由使用方聲明」。FTF 樣板設定已是 local 4×4×4、`retained_levels: 24`，正好落在一般單 tone 掃描算得準的範圍；寬 carrier 掃描要提高到 28。

兩個 qubit 同時打成形 X90 的長序列（RB depth 500）上，選態、凍結小分量、分量分解、分塊演化都不划算。原因是熱平衡初態的主分量在 X90 pair 下自己就擴到全部 32 維，而 fluxonium 的 charge 驅動在 dressed 基底下連接幾乎所有能階對。

使用者 2026-11-18 對結案選項答「A」。

## 問題

使用者 2026-11-18：「關於子空間模擬優化與GPU優化，我比較看好，今晚你可以著重研究，建立一些prototype」，並要求做法由 agent 多輪嘗試、只看效果。要回答的是：在完整 density 對參考解 ≤2e-3 的固定預算下，FTF 模擬能不能只在一個較小的子空間演化，並且比現行預設選路快。

討論中使用者提出評估原則：「我認為不同場景的預算分配確實不同，但模擬算法的衡量不應該改變預算，而是固定預算並找適合的場景」。本研究照這個原則，預算固定，找每個做法適合的場景。

## 單 tone FTF 掃描

案例：tracked FTF 模型、固定 DC、parent 4×4×4=64 態；驅動 F1 charge 單一常數 tone（基準 carrier 5.4437 GHz、gain 0.1、40 ns）。一般掃描是 length 20–80 ns、gain 0.05–0.15、carrier ±50 MHz，各 5、21、101 點；壓力掃描是 carrier ±200 MHz 與 length 20–500 ns。基準是最低 32 態加 `Engine` 預設選路（全部走 `floquet_period`），每點約 19 ms。誤差是完整 density 的最大絕對元素差。

| 做法 | 結果 |
|---|---|
| 非連續選態（one-photon、two-photon、bath 閉包指標挑態） | 第一個案例挑 11 態，誤差 1.286e-3，演化 4.77 ms 對 32 態 20.62 ms，但選態本身 224 ms。選態成本優化到 31–53 ms，再用只算 seed 的 two-photon 近似降到 8.5 ms，選出的態不變。5 點 gain 掃描聯集 11 態、每點 12.68 ms 對 19.08 ms，誤差 1.873e-3，貼近預算；length 與 carrier 掃描誤差 3.11e-3、5.17e-3，不合格。每點壓到 1e-3 要 ≥36 態，比 32 態還慢。 |
| 二階 Floquet–Schrieffer–Wolff 消去虛態 | 虛態最小能量分母 5.70e-4 GHz，drive／分母 1.84–2.86，不在微擾區；誤差反而升到 4.93e-3。 |
| 能量最低 N 態截斷 | N=24 在三種 5 點掃描 15/15 點 ≤1e-3（最大 3.0e-4），每點成本約 32 態的一半。N=17、20 只有 6/15 點合格。用相鄰 N 互比估誤差不可靠：估計器在 N=17 停下，9/15 點實際超標，因為同時被刪的活躍態看不到。 |
| 3 個 pilot 點選 N（N=24 起，過則試 20、不過則試 28，對 32 態比，門檻 5e-4） | pilot 選擇與對 64 態真值的選擇 11/11 一致。一般掃描選 24，carrier ±200 MHz 選 28（N=24 在那裡誤差 2.21e-3）。含 pilot 成本對全用 32 態：5 點 1.17–1.27（變慢）、21 點 0.67–0.69、101 點 0.54–0.55。第二個 flux 點 21 點 0.68。pilot 只驗證那 3 點，內插點沒有逐點保證。 |
| 逐點 a-posteriori 截斷誤差界（defect／Duhamel：‖ρ32−ρ̃‖₁ ≤ ‖ρ32(0)−ρ̃(0)‖₁ + ∫‖L32ρ̃−ρ̃′‖₁ds） | 零階（ρ̃ 取截斷軌跡）的界 1.3e-2 到 2.3，太鬆。一階修正的界在 109 點中 81 點低於實際誤差（梯形取樣不是嚴格界），成本中位是截斷演化本身的 304–748 倍；「用界判定、超界退回」總成本是全用 32 態的 510 倍。 |
| 實際初態（0.05 K、32 levels 的 DC Gibbs density，最小 weight 7.64e-8） | 三種 21 點一般掃描都選 N=24，初態投影誤差 6.84e-6，全掃描 N=24 最大誤差 1.13e-4／7.65e-5／7.32e-5，成本 0.67–0.69；與純態結果相近。 |

## 兩個 qubit 同時打成形 X90 的長序列

案例：同一 FTF 模型、retained 32、Gibbs 初態、bath 保留；Gaussian X90 pair、單邊 tail、CZ 四類 gate；depth 500 inventory（max_pairs）2,581 個 gate。用含四類 gate 的 9-gate prefix 實測，再按 gate 數線性外推。32 態 `dop853` 9-gate prefix 20.3 s，外推 depth 500 約 5,830 s、誤差 8.5e-6。

| 做法 | 結果 |
|---|---|
| 舊選態器 | Gibbs 初態 32 態都有 population（前四個計算態合計 99.06%），選態保留全部 32 態。 |
| 截掉小 population 初態並凍結（ε=1e-4、1e-5、1e-6） | X90 pair 的閉包仍是 32 態；單 gate 最好 1.507 倍（tail_f1，26 態，凍結誤差 1.486e-6）；9-gate prefix 外推只快 1.012 倍。 |
| 分量分解（使用者提議：「就算初態有分佈，也不一定要納入演化維度來計算」） | ρ0 對角、演化線性，ρ(t)=Σp_iΦ(\|i⟩⟨i\|)。按 dominant coupler level 分三組，權重 0.99595／0.00403／2.0e-5；pair 閉包 32／30／20，9-gate prefix 32／32／26。三組各自演化比全空間慢 2.6 倍；只凍結最小組也慢 2.0 倍。主分量自己就要 32 維。 |
| 分塊演化（使用者提議：「也許可以拆分成多個解偶子空間來演化」） | dressed 32 態下 F1、F2 驅動的 496 個非對角元全部非零，超過最大值 1e-3 的有 371、357 個（最大 0.045、0.048 GHz）。按 coupler level 分 16／13／3 時，塊間驅動元最大 0.024、0.037 GHz，與塊內同級；bath 有 590 條跨塊通道。忽略塊間 coherence 快 1.09 倍，但 9-gate 誤差 2.38e-2；保留塊間 coherence 就回到完整 1,024 個 density 座標。 |

使用者 2026-11-18 說超長序列可以接受更多誤差，裁定長序列預算為 max(2e-3, 3.9e-6 × gate 數)，depth 500 約 1e-2。這個放寬不改變上表結論：X90 pair 需要的維度與預算無關，分塊的 2.38e-2 已超過 1e-2。

## 為什麼要那麼多能階

實驗常用 2–3 個能階的有效模型，參數由量測擬合，高能階的虛躍遷效應已折進擬合數字。這個模擬器從電路參數直接算、不做 RWA，那些效應只能靠高能階留在基底裡算出來。FTF 的能量排序讓需要的態排得很後面：兩個 fluxonium 的 0→1 很低（F1 約 0.60 GHz、F2 約 0.30 GHz），coupler 激發高出數 GHz；CZ 用到的 \|1,1,1⟩ 在整數磁通附近排第 23–24、中間磁通最高 28（[field-guide](../../zcu_lab/simulation_envs/FTF/field-guide.md)）。fluxonium 的 charge 算符連接不相鄰能階，加上近共振虛態，丟掉「沒被激發」的能階也會讓相位在百 ns 脈衝上偏離超過 2e-3。

## 被否決的選項與理由

- `Engine` 內自動能量截斷加 pilot 驗證：pilot 只驗證那 3 點，不是逐點 guard。
- 自動截斷加 a-posteriori 誤差界當 guard：界的成本是截斷演化的數百倍，一階版本又常低估。
- 選維度工具（3 個 pilot 比較 M=20／24／28／32，建議最小合格 M，由使用者改設定）：是精度輔助而非加速；field-guide 已有手動選 M 的指引。可另立題目。
- 非連續指標選態、Floquet–SW、分量分解、分塊演化：理由見上兩表。

## 留下的問題

- 寬 carrier（±200 MHz 以上）或中間磁通的掃描要把 `retained_levels` 提到 28；目前靠使用者依 field-guide 判斷，沒有自動提醒。
- 若實驗在序列前做主動 reset，初態接近 \|00⟩，雙 qubit 序列的分量結構會改變；本研究只量了熱平衡初態。

## 依據

- 本 repo commit `c04fb38231de5aa65d01aba931a48db13d48d093`；模型與設定取自 `tools/family_pareto_settings.py` 的 FTF 模型（0.05 K、retained 32）與 `zcu_lab/simulation_envs/FTF/default.yaml`（retained 24）。
- 參考解是 tests 端的完整 Lindblad、不做 RWA 的 interaction-frame 解（`tests/simulate/evolution/contract/interaction_reference.py`），兩級容差互比 ≤2e-6。
- 各輪 spike 的原型、逐點數據與報告是未入庫的工作檔；上文的數字取自那些報告，均為非 exclusive 單次計時。

後記（2026-11-18）：`retained_levels` 後來改為上限，FTF runtime 在符合條件的 acquire 以 pilot 選維度；`Engine` 仍不做約化。見 [FTF 動態維度](2026-11-18-ftf-dynamic-dimension.md)。
