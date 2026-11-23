---
date: 2026-11-23
status: active
superseded_by: null
related:
  - ../adr/0074-flat-readout-module.md#shot-sums-與-heralding
  - ../adr/0069-experiment-core-and-frontends.md#能推導的誤差棒都畫
  - 2026-11-23-mist-trace-propagated-sem.md
  - 2026-11-23-errorbars-tier1-saved-stats.md
  - ../../lib/zcu_tools/analysis/README.md#逐點-shot-sem
basis:
  repo_commit: bcf894607f02a0a500b490a7740db1769fe13979
  external_sources: none
---

# IQ 六矩共用原始矩捨入界

## 結論

Designer 代裁，使用者 2026-11-23 追認。採 A1。矩陣與線性 SEM 從同一格式的六矩判斷中心量是否能以捨入解釋。兩入口共用相對界 `8·clip(Nu, 4u, √u)`，`N` 是實收 shot 數，`u=eps64/2`。界內的負中心量可歸零，但不把小的正估計清零。矩陣仍修復至正半定；線性入口保留直接投影、逐點權重及 overflow retry。

不可估時，MIST 和 Power 保留有限中心，記下點與數值原因，只略依賴失敗共變異的棒。MIST 不換參考。兩個實驗各自的零距離規則、取數與保存不變。

## 問題

原始平方和與均值平方項很接近時，相減會顯露先前累加的捨入誤差。固定容差會把合法常數序列的負殘差當作超界。MIST 的合成反例是 70 發 `I=.3, Q=0`；Power 當時用 12,200 發 `I=1/3, Q=0` 的合成資料驗略棒。兩例的真樣本散布都是零，不需要維護兩套原始矩判準。

## 選項與比較

- A1。共用按實收 N 的原始矩界，再各自做矩陣估計和投影。費用是接受一段更寬的捨入區間；不能從已捨入的六矩恢復丟失的散布。
- B。只放寬 MIST。可保留 Power 當時的略棒結果，但同一份六矩會依入口使用兩套原始矩允收界。
- A2。線性入口統一投影修復後的矩陣。兩入口的數值來源更接近，但直接乘公開的 float64 矩陣可能在開根前溢位，或先下溢丟掉仍可表示的 SEM。
- C。由取數端保存可合併的中心矩，避免最後相減的主要消去路徑。要改 producer、保存與 reader，也無法補回舊資料的低位。

## 未採用

不選 B 的雙門檻。A2 的尺度表示與極值驗證超出本次修正；C 要另立資料契約。A1 只共用原始矩門檻，不要求每個方向都可估。

## 留下的問題

這是有上限的串行累加工程界，不是任意 N 的誤差保證。投影與矩陣修復在正半定邊界可有不同結果，不把某個分歧鎖成介面契約。

## 依據

候選 `bcf894607` 的合成離線 Live／Analyze 驗證上述兩個常數序列回零。Power 三點棒為 `[1, 0, 3]`；換成超界中點後，中心和兩支鄰棒不變。MIST 分別驗壞非參考點、壞預設參考與固定參考，略棒範圍依實際相依關係。用相同候選的公開入口讀 `[70, 70, 70, 70, 70.0000000000065, 70]`，矩陣修復回零，直接沿 `(1, −1)` 的投影不可估。這一例說明差異，不是回歸斷言。

[Analysis README](../../lib/zcu_tools/analysis/README.md#逐點-shot-sem)和 [估計器](../../lib/zcu_tools/analysis/shot_uncertainty.py)記數值契約；原始調查及離線驗收收據未入庫。
