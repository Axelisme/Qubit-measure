---
date: 2026-11-23
status: active
superseded_by: null
related:
  - ../adr/0069-experiment-core-and-frontends.md#呈現規範
  - 2026-11-22-mux-iq-moments-and-r11.md
  - 2026-11-22-amp-rabi-shot-moments.md
  - ../../zcu_lab/v2/mux/t2ramsey/guide.md
  - ../../zcu_lab/v2/mux/t2echo/guide.md
  - ../../zcu_lab/v2/twotone/time_domain/t2echo/guide.md
basis:
  repo_commit: 7f51de892318b9d75685acff8dee438c51a810e6
  external_sources: none
---

# Coherence 的 shot SEM 只隨當次 Run 交接

## 結論

2026-11-23，designer 夜間代裁（2026-11-23 授權），待使用者追認。Mux Ramsey、Mux Echo 與單 qubit Echo 從同一批完成輪的 shots 求平均 IQ 和逐點 shot SEM。累加矩隨當次 Run 的記憶體結果交給 Analyze，不寫入 Native 或 Labber。從新舊資料檔載入後重做 Analyze，圖仍有平均 IQ 和可得的擬合結果，但沒有棒。已匯出的圖維持匯出時畫出的棒。

這局部推翻了先前經使用者覆核、要求 **Mux Ramsey 與 Mux Echo 新檔加入可選 Native 矩**的裁決。原因是這次補回 coherence 呈現時明文凍結保存格式。例外只撤回這兩個 leaf 的新檔存矩要求，保留同批、完成輪、真實驗入口與 Echo Stop 驗收。不改 Mux Amp Rabi、Len Rabi 與 T1 已存的矩。單 qubit Echo 沿用暫態交接，但不在先前 mux 裁決的適用範圍內。

## 問題

既有 Run 留下平均 IQ，沒有可供同次 Analyze 算 shot SEM 的變異資訊。另取一批 shots 來配舊平均會破壞同源；只補 Live 又讓同次 Analyze 少一項可估的不確定度。任務同時要求 Native 與 Labber 格式、workflow 和既有 mean-only 呼叫端不變。

## 選項與取捨

Mux 使用現有多 tone reducer，按參與 tone 順序交接校正六矩。單 qubit Echo 使用現有單 ADC reducer，不借用 mux Result。兩邊都讓同一累加器的平均值供 Live、結果及既有 Echo Stop，讓 Analyze 按自己的選點及投影方向求棒。這多留六個 float64／tone／點，單 qubit Echo 沒有 tone 軸。可選欄位預設缺席，保留原建構方式，不另建 adapter cache。

這沿用[多 tone 校正 IQ 矩](2026-11-22-mux-iq-moments-and-r11.md)與[單 ADC Amp Rabi](2026-11-22-amp-rabi-shot-moments.md)的同批取數，區別它們的保存做法。[ADR-0069〈呈現規範〉](../adr/0069-experiment-core-and-frontends.md#呈現規範)仍定義 shot SEM、IQ 投影和缺少必要統計量時不畫棒。這次沒有把擬合標準誤或別的反演量誤差當逐點 shot SEM，也沒有改持久化的 owner。

## 未採用

- 維持舊圖或只補 Live，會讓同次 Analyze 仍缺可估的棒。
- 每個 mux leaf 各做一個私有 Result 子型別，會重複表達同一暫態欄位，還會使 Run 與原 Load builder 的欄位集不同。
- 在 Run 先算好 Analyze 的 SEM，可少留五個 float64／tone／點。但 Run 得同步 Analyze 的略點、有效值與投影 fallback，仍可能把棒配到不同方向或點。
- 加 Native 可選矩能讓新檔重載後重算棒，也能保留舊檔讀取。這會增加一個新檔變數，超出本次凍結的格式。Labber 即使仍匯出原選集，也不能消除 Native 的變更。

## 留下的問題

原檔沒有足以重建棒的統計量。若使用者不追認這項窄例外，另開任務為 Mux Ramsey 和 Mux Echo 恢復可選 Native 矩，檢查新舊檔讀取與原 Labber 選集。不從已保存的圖回填矩，也不變更其他已存矩的實驗。

## 依據

- 2026-11-22，使用者說：「像我其实更想要设计的，其实应该是平均标准差，也就是这个点的预期误差。」這句定的是統計量，不是這次保存例外的批准。
- 2026-11-23，使用者對平均 IQ 主軸投影說：「B，這種投影方式應該作為慣例存在，沒有獨特理由就保持一致」。見[平均 IQ 圖的投影慣例](2026-11-23-mean-iq-plot-projection.md)。
- 當時的原始裁決與選項調查未入庫。資料契約與重載的操作結果見頁首三份 guide。這篇記錄代裁的範圍，不新增 ADR 規則。
