---
date: 2026-11-23
status: rejected
commit: 8cd18afcd
modules:
  - lib/zcu_tools/simulate/evolution
discussion: none
superseded_by: null
---

# Sampled multitone RWA 探針提早判死

**讀者：** framework-dev、agent。查為什麼不能靠探針中途越線判定終點不合格。

## 核心思路

現有 guard 以固定初態，分別用 RWA 與完整驅動參考算到列末。兩個列末密度矩陣的最大元素差不超過 `0.1 × B` 才放行。候選 E 想讓兩個積分器在相同 checkpoint 比較；中途一旦越線就拒收，省掉該探針剩餘的參考積分。

## 實測

在 `8cd18afcd` 的離線 FTF `two_qubit/chevron`，以 gg herald、2×2 格、24 retained levels、CPU 單 worker 非排他執行一次。48 個唯一探針全在普通端點比較中被拒收；它們的完整驅動參考積分合計 98.4 CPU s。這是要省的基準成本，不是 E 的實測節省。

同一 commit 上，以既有 `test_sampled_multitone_spot_edge_density` 的一列 d=4、1.5 ns 輸入做反例。門檻是 `1e-4`。普通端點差為 `9.869e-5`，但 10% span 的 checkpoint 差為 `2.117e-4`。既有規則放行，E 會拒收。

另一個檢查以現行 Gaussian 產生器重建 d=24 的 `simultaneous_drive_retained24` 正例，結果同樣是中途越線、終點通過。重建沒有記錄量測 commit，也不是[歷史放行輸入](../discussions/2026-11-18-sampled-multitone-rwa-admission.md#考慮過的選項與比較)的逐位重播，因此不列重建數字。原型與原始工作檔未入庫。

## 為何採用或否決

不採用 E。中間態的差可以回落，中途超標不代表端點超標。E 會把本來合格的列送回較慢的路徑。當時的實作先求完整 RWA 端點，再求參考端點；要測真正的早停成本，兩個積分器都得提供同一時刻的中間態。

## 留下的線索

[探針快取](./2026-11-23-ftf-rwa-probe-memo.md#核心思路)只消除重複的探針，唯一探針仍要付參考積分成本。若試用較便宜的 CPU DP5(4) 當參考，應對照通過、拒收和貼近門檻的探針，確認沒翻轉端點判決。不能把固定 probe 初態的端點直接當作 fallback 的輸出，因為實際演化可能有不同初態。
