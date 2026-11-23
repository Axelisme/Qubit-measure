---
date: 2026-11-22
status: active
superseded_by: null
related:
  - 2026-11-22-presentation-rules.md
  - 2026-11-22-data-figure-reading.md
  - 2026-11-22-mux-iq-moments-and-r11.md
  - 2026-11-22-fastflux-distortion-fit-failure.md
  - ../../zcu_lab/v2/figure-style.md
basis:
  repo_commit: a90cb66bbaa7e5269572d375fef321936dbeb3cc
  external_sources: none
---

# Families 部分交付與代表 R0 缺口

## 結論

交付已驗實驗的呈現、存檔接合與 mux reducer。Mux Amp Rabi、Len Rabi 和 T1 採同批校正矩；Mux Ramsey、Mux Echo、單 tone Echo 與 MIST 單功率軌跡維持進版基線。第 11 步是部分交付，代表與 R0 判準未變。未驗代表不補位、不寫通過。

## 問題

窄窗的 Singleshot GE Live／Primary 裁掉圖題或 counts。T1 tone sweep RF 在一般及窄 pane 都有圖面重疊，gain 與它共用 core。JPA Freq 缺有 pump-to-readout 耦合的同次來源；Power 的原圖例遮住 −10 dBm 峰，唯一 Analyze-only 補證在開始 Analyze 前被排序檢查攔下。Fast-flux Distortion Freq 舊檔可讀但 fit 失敗。Chain Amp Rabi 的完整 16-read 取證超出 600 秒窗口。

審查另發現 Coherence 的寫回理由進入 core 或由 adapter 疊到圖上。中途修正後，Mux Ramsey Q2 的 R² 為 0.8972，低於 0.9；圖不再說明門檻，只在 Writeback 顯示。MIST 位移量的近似棒不對應圖上的非線性幅值；去掉棒後只有普通 probe，缺最終版同次 FTF 的 Live 進度、末格與 Analyze。

## 選項與取捨

先補實測 GE 與 T1 圖面能回答的缺口。350×480 只作壓力圖，與實測正常和窄 pane 分列。仍不可讀時，原路徑不上新呈現。其餘已驗 leaf 照常交付，保留原代表身份。Coherence 和 MIST 的四個實驗整套保持進版契約，避免只移除某個畫法卻留下未驗取數、設定或保存接合。這保住三個已有證據的 mux caller。

## 未採用

不以普通 stage、舊圖或不同 Run 補代表證據，也不把 Writeback 當作 R0 C 所需的圖上說明。不在 scope 收尾臨時加 Coherence 評估／渲染介面、FTF pump 模型或 Fast-flux fitter fallback。只註缺口卻交付已知壞圖也不採。

## 留下的問題

GE、Mux Ramsey、MIST single trace、JPA Freq、Fast-flux Distortion Freq 與 Chain Amp Rabi 代表 R0 均未驗收。Coherence 需另驗選取 owner 把既有理由傳給繪圖 owner，MIST 需最終版一次同 Run 的讀者包。Fast-flux 先校準，Chain 需取得涵蓋原 16-read 工作量的上界。JPA、GE 與 T1 tone sweep 的遮擋／耦合另處理；離線證據不是實機驗收。

## 依據

遵循 [R0／R3b／R7](../../zcu_lab/v2/figure-style.md) 和 [ADR-0069](../adr/0069-experiment-core-and-frontends.md) 的不讀程式碼判讀、圖與 Writeback 分工。共用 [Data 閱讀](2026-11-22-data-figure-reading.md) 的驗收不能抵個別 pane 紅點，依 [Fast-flux 失敗語意](2026-11-22-fastflux-distortion-fit-failure.md) 保留原失敗。上述原始圖、probe 與答卷未入庫。
