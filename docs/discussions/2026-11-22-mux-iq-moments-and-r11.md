---
date: 2026-11-22
status: active
superseded_by: null
related:
  - ../../lib/zcu_tools/program/v2/README.md#mux-iq-sums
  - ../adr/0069-experiment-core-and-frontends.md#呈現規範
  - ../../lib/zcu_tools/program/README.md
basis:
  repo_commit: a90cb66bbaa7e5269572d375fef321936dbeb3cc
  external_sources: none
---

# Mux 逐 tone 校正 IQ 矩與 R11

## 結論

這次用了明示選用的 `mux_iq_sums`。Mux Amp Rabi、Len Rabi 和 T1 以同一批校正 shots 算平均和線性投影的 SEM。只合併完成的 rounds，結果依選取的 tone 順序排列。原生資料把矩存為可選欄位，舊檔仍可讀。寫成產品契約的範圍見 [Program v2 的 Mux IQ sums](../../lib/zcu_tools/program/v2/README.md#mux-iq-sums)。

## 問題

Mux 的原有平均不包含估計逐點誤差所需的矩。使用者在 2026-11-22 說：「像我其实更想要设计的，其实应该是平均标准差，也就是这个点的预期误差。」這句定的是點平均的不確定度，不是 reducer 或存檔格式。後兩項由 designer 代裁，當時未經使用者覆核。

## 選項與取捨

- 從同批 shots 歸約各 tone 的六矩，可重建平均與其線性投影 SEM，無需保存每發 IQ。例如先選 tone 1 再選 tone 0，第一列 signal 和棒都取自 tone 1，與 ADC 編號排序無關。
- 採用程式既有的單 ADC 路徑作先例，另開窄的多 tone opt-in，不改單 ADC 或一般 acquire。離線測試用不同的非零 ADC offset、當輪 PFB 校正和中斷輪。真 T1 的兩 tone 檔驗了 Native、mean-only 舊檔與 Labber。

## 未採用

不以第二批 raw 配 compact mean，也不要求 compact 與 raw 用同 seed 會抽出相同 IQ。不保存逐發 IQ、不算跨 tone 外積、不把 mask 或多次讀取納入此次 reducer；這些用途不在三個交付 caller 的需求內。

## 留下的問題

Mux Ramsey／Echo 的接合延期，沒有第四或第五個進版 caller。實機 offset、Stop 與呈現仍待驗。這次不把非線性衍生分數改用輸入均值的 SEM。

## 依據

原始討論與離線工件未入庫。比較單 ADC 的方案見 [Amp Rabi 的 shot moments](2026-11-22-amp-rabi-shot-moments.md)，完成輪的 owner 見 [ADR-0083](../adr/0083-program-macros-and-timelines.md)。
