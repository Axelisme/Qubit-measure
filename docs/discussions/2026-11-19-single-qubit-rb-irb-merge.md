---
date: 2026-11-19
status: active
superseded_by: null
related:
  - 2026-11-15-two-qubit-rb-simultaneous-gates.md
basis:
  repo:
    main: 8f754f26619957c2ecf8fdf5a49757d4c4d30caf
---

# 單 qubit RB 與 IRB 合併成 twotone/rb

## 結論

2026-11-19，單 qubit 的 RB 與 IRB 併成一個實驗 `twotone/rb`：

- `target_gate` 為 `None` 時只跑 reference，Result 的 arm 軸長度為 1。指定 gate 時跑 reference 與 interleaved 兩臂，兩臂共用同一組 seed。
- Result 是 `signals`，形狀 (round, seed, arm, depth)。rounds 是 host pass；每個 (seed, arm) 編一支程式並快取，depth 與 reps 在硬體上掃。兩臂時，每一輪輪流換先跑的臂，以平衡漂移。
- 只跑 reference 時，分析回傳 `RBAnalysis`：reference 衰減、每個 Clifford 的 error，以及 seed bootstrap CI。兩臂時回傳原本的 `IRBAnalysis`。
- `twotone/irb` 移除，不留 alias。
- 舊資料只保留原檔，不提供讀取相容或轉換：
  - tag `twotone/ge/rb` 的舊 run 存的是 `signals2D`，形狀 (seed, depth)；
  - tag `twotone/ge/irb` 的舊 run。
- 離線遷移的兩個舊宣告也一起移除。舊格式沒有 round 與 arm 軸，轉換屬於相容邏輯，這次不做。

## 問題

使用者：「我想把 IRB 跟 RB 实验合并成单个实验。」

合併前有兩個實驗，共用 `zcu_lab/v2/twotone/_rb_sequence.py` 產生序列：

- `twotone/rb`：每個 seed 一支程式，depth 與 reps 在硬體上掃，rounds 在程式內平均。Result 是 `signals2D`，形狀 (seed, depth)。
- `twotone/irb`：reference 與 interleaved 成對，rounds 是 host pass。Result 是 `signals`，形狀 (round, seed, arm, depth)，分析含 seed bootstrap CI。

`two_qubit/rb` 早已是 RB 與 IRB 合一的實驗，Result 帶 arm 軸，0 是 reference，1 是 interleaved。

## 考慮過的選項

| 選項 | 內容 | 只量 RB 時 |
| --- | --- | --- |
| A（採用） | 以 IRB 的結構為基礎，加一個可為 `None` 的 gate 欄位 | 只跑 reference 一臂 |
| B | 永遠跑兩臂，和 `two_qubit/rb` 一樣固定 arm = [0, 1] | 時間加倍 |
| C | 維持兩個實驗 | 兩份大致重複的實作 |

使用者回覆「單qubit的rb/irb合併」與「RB採用建議」，採 A。

欄位名在提案中寫作 `interleave`。實作沿用 IRB 原有的 `target_gate`，型別改為 `TargetGate | None`，意義與提案相同。

### 先例與立場

- **`two_qubit/rb` 的 arm 軸：** 遵循。單 qubit 用同一組 arm 座標（0=reference、1=interleaved）。區別在於：two-qubit 永遠跑兩臂；單 qubit 在 `target_gate=None` 時 arm 軸長度為 1。
- **移除舊 key：** 遵循同日命名盤點中使用者對改名的裁決：舊 key 不留 alias，舊 run 只保留原檔（使用者回覆「全采用建议」）。該裁決的工作檔未入庫。
- **只有 reference 時的分析回傳：** 依 repo 的強型別規則，回傳具名的 `RBAnalysis`，不以 NaN 或預設值填 interleaved 的欄位。

## 被否決的選項與理由

- **B：** 只想量 RB 時，時間加倍。
- **C：** 保留兩份大致重複的實作，和使用者要合併的目標相反。

## 留下的問題

- 使用者也要 two-qubit RB 改成在硬體上掃 depth（「我希望two_qubit版本也改成硬體掃描depth」）。調查發現：每個 CZ 的 local-Z 修正是任意角度，到第 k 個 CZ 時累積的相位每一步都不同，需要新增執行期讀取相位的機制。方案待裁決。
- 量測知識庫仍有經驗文件引用 `twotone/irb`。知識庫由使用者管理，是否更新待使用者決定。

## 依據

- 合併的實作：main 合併 commit `8f754f266`，含 `ac8aaddbf` 與 `bf05be8dc`。
- 合併前的程式：`f5e742a26` 的 `zcu_lab/v2/twotone/rb/core.py` 與 `zcu_lab/v2/twotone/irb/core.py`。
- 合併後的 Result 宣告：`zcu_lab/v2/twotone/rb/core.py` 的 `AXES_SPEC`。分析：`lib/zcu_tools/analysis/fitting/irb.py`。
- 測量策略：`zcu_lab/v2/experiment-authoring.md` 的〈Twotone RB 測量策略〉。
