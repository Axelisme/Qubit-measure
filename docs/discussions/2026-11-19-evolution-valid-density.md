---
date: 2026-11-19
status: active
superseded_by: null
related:
  - ../adr/0078-evolution-module-boundary.md
  - 2026-11-18-short-dop853-dp54.md
basis:
  repo:
    control: d45e87637975cbbfffa566a8888635dbebe28834
    first_bad: f7e610d7f6fe01931b40ab1616859a95958a613e
    fix_merge: 5f9b658b266204b22e6dce42053f00e0c14eb627
---

# Evolution 輸出合法 density

## 結論

2026-11-19，使用者選 A：`Engine.evolve` 在輸出端把不合法的 readout density 投影回合法 density，只在修正量不超過該次呼叫的 budget 時採用，超過就照數值失敗回報。規則寫在 [ADR-0078](../adr/0078-evolution-module-boundary.md#evolution-輸出合法-density)，現行契約見 [evolution README](../../lib/zcu_tools/simulate/evolution/README.md)。

## 問題

2026-11-19 的 FTF 離線模擬中，所有 two-qubit heralded 實驗都失敗，訊息是 `initial density must be positive and trace one`。

- 用 bisect 找到的第一個壞 commit 是 `f7e610d7f`〈Use DP5(4) for short CPU dop853 rows〉。它的前一個 commit `d45e87637` 能通過同一個判別案例。
- Heralded 路徑先把 density 投影成 rank-one 純態，接著演化 F1、F2 同時播放的 50 ns 高 gain ramp。CPU DP5(4) 的輸出 Hermitian、trace 為 1，最小特徵值是 −2.4e−6。
- 下一段的 continuation 檢查要求特徵值 ≥ −1e−8，因此拒收。這個檢查本身正確。
- 矛盾在兩個容差：ADR-0078 的精度目標是誤差 ≤ `2e-3`，DP5(4) 符合；continuation 檢查的容差是 1e−8，嚴格五個數量級。純態落在合法 density 的邊界上，任何近似積分器的微小誤差都可能把特徵值推成負值。
- 2026-11-18 採用 DP5(4) 時（[短 `dop853` row 改用 DP5(4)](2026-11-18-short-dop853-dp54.md)），驗收量的是 density 元素誤差，沒有檢查輸出是否仍是合法 density。
- 診斷時，OpenCL DP5(4) 的 warmup row 也輸出 −4.6e−5～−7.8e−5 的最小特徵值。

## 考慮過的選項與比較

- A：evolution 在輸出端投影回合法 density，只在修正量落在精度預算內時投影；超出預算照樣報錯。由數值 owner 自己保證輸出合法，CPU 與 OpenCL 一起修好。
- B：放寬 continuation 檢查的容差。
- C：純態起始的段改走 DOP853。
- D：撤回 DP5(4)。

Agent 建議 A，使用者回覆「其余采用建议」。

## 被否決的選項與理由

- B：負的 population 會流到讀出抽樣，下游各自處理。
- C：只是把問題移開，DOP853 的誤差也不保證小於 1e−8。
- D：失去 CPU 約 1.5 倍的加速。

## 留下的問題

- 修正耗時只在兩個 FTF 重現案例量過一次（約占 evolve 總時間的 0.05%–0.09%），沒有量更大的 density batch。
- 投影只作用在 `Engine.evolve` 回傳的 readout density。evolve 內部 prefix 的 density 不經過投影。
- OpenCL warmup 的負特徵值在 budget 內會被投影，實際取數的 OpenCL 路徑沒有另外驗證。

## 依據

- 本 repo：`lib/zcu_tools/simulate/evolution/_validation.py` 的 density 檢查與投影，`engine.py` 的 `Engine.evolve`；FTF heralded 投影在 `lib/zcu_tools/simulators/ftf/environment/cavity.py`。
- 診斷與重現的工作檔未入庫；本文已寫入結論所需的事實與數字。
