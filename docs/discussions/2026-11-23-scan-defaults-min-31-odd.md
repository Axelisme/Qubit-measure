---
date: 2026-11-23
status: active
superseded_by: null
related:
  - ../adr/0073-measure-form-derivation.md#掃描初值的點數
  - ../adr/0078-evolution-module-boundary.md#模擬器速度目標
  - 2026-11-23-simulator-speed-per-point.md
basis:
  repo_commit: 61137266ed954456448764f5d2da0ce23a997fd9
  external_sources: none
---

# 掃描初值至少 31 點、用奇數

## 結論

使用者把等距 Range 掃描的初值下限設為每軸 31 點。以參考值為中心的掃描用奇數，讓網格包含中心。變更只及於初值，Form 不阻擋操作者輸入偶數或更少的點，也不重寫已存 cfg。Protocol 決定的離散序列不套用這條規則。現行規則見 [ADR-0073〈掃描初值的點數〉](../adr/0073-measure-form-derivation.md#掃描初值的點數)。

## 問題

有些 Form 的初值只有 10、11、21 或 25 點。以中心為準的掃描若選 10 或 50 個等距點，兩個中間點會落在中心兩側。使用者看了 10 點的 ramp 後說「10點太少了，建議至少31點」。2026-11-23 的初步盤點在 `zcu_lab/v2/**/form.py` 的 121 個 `expts=` 初值中列出 16 個需調整的真初值，待填的一點佔位不算。

## 考慮過的選項與比較

單軸和多軸都可按每軸設下限。Raman g-f 從 11×10 改為 31×31，總點數由 110 增至 961。同樣的每點採集設定下，實機每輪須採更多點。使用者仍確認多軸也逐軸適用。與此並行的[模擬速度裁決](2026-11-23-simulator-speed-per-point.md)把耗時改按每格點比較，正常場景用每軸至少 31 點攤平固定成本。

## 被否決的選項與理由

- 只改單軸，放寬多軸。使用者確認每個軸都適用。
- 讓 Form 擋偶數或少於 31 點。這是新開表單的建議初值，不是限制操作者選掃描密度。
- 把 CPMG pulse count、RB 的 `LogPoints` depth、雙 qubit RB 的 depth／seed 清單一併加點。它們由 protocol 決定，不是等距 Range 初值；使用者答「A」，排除這些序列。

## 留下的問題

沒有替各實驗重量實機每輪時間。實際採集成本仍取決於各掃描的點數、reps 和 rounds。

## 依據

- 2026-11-23，使用者：「可以加一個小慣例，掃描點數建議是奇數，這樣可以包含中心點」。被問多軸是否逐軸套用時答「都對」。
- 遵循 [ADR-0073〈Form 的形狀與初值〉](../adr/0073-measure-form-derivation.md#form-的形狀與初值)：初值由 Form 作者宣告，不加驗證規則。
- 比較基線為審查時固定的 main `61137266e`，可查當時的 Form 宣告。盤點原始檔和對話未入庫，範圍及裁決記於上文。
