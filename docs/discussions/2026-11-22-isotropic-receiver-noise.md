---
date: 2026-11-22
status: active
superseded_by: null
related:
  - ../adr/draft/simulation-environments.md
  - ../../zcu_lab/simulation_envs/FTF/field-guide.md
basis:
  main_base: 4b834a7c46f2050a57adeb02981c27cb4f1bbf83
  original_template: 4f812bfb57b54758959be5c0b5c1e4478783178f
  ge_measurement_commit: null
  external_sources: none
---

# 模擬器接收雜訊只保留等向設定

## 結論

使用者選 A。模型 YAML 每個 receiver 填一個白雜訊 PSD，I 與 Q 獨立且變異數相同。既有樣板的兩個對角值相同，所以只保留原對角值。舊矩陣會在載入時報出欄位路徑，不自動轉換。規則見[等向接收雜訊](../adr/draft/simulation-environments.md#等向接收雜訊)。

## 問題

使用者問為什麼 GE 讀出的垂直寬度比沿 g–e 連心線的寬度大：「这不应该是两个等向高斯的杂讯吗？」舊 FTF 低 SNR 設定的 I/Q PSD 對角值皆為 84000，非對角值為 16800。兩分量的相關係數是 0.2。這個值從 2026-10-17 的樣板沿用，沒有硬體校準依據。

| Qubit | GE 擬合實測 σ⊥/σ∥ | 舊 PSD 按 g–e 方向預測 σ⊥/σ∥ |
| --- | ---: | ---: |
| Q1 | 1.148 | 1.139 |
| Q2 | 1.214 | 1.225 |

原始 GE 量測資料與執行 commit 未入庫。右欄來自舊 PSD 在 g–e 連心線及其垂線的投影，不是另一次量測。

## 考慮過的選項與比較

- A：改用純量 PSD。I/Q 方差固定相同，沒有可重新開啟的相關係數。
- B：把非對角值改成 0，但保留 2×2 矩陣。樣板可得到相同的等向雜訊，schema 仍接受各向異性和 I/Q 相關。

GE 的單一 σ 及 [chain 的等向假設](../../zcu_lab/v2/chain/amp_rabi/guide.md)是既有分析前提。本次遵循這項前提，不改分析端。

## 被否決的選項與理由

否決 B，也不保留矩陣作為選項或讓分析去擬合 2×2 共變異矩陣。2026-11-22，使用者說：「我从来都没有要求杂讯要出现不同方向的异性我从来都是假设杂讯是各项同性所以这个机制完全没有必要留着，直接全部同向就好了。等向」。對實作答「我选 A」。

## 留下的問題

換成純量 PSD 後，舊樣板量出的 SNR 不是新樣板的校準結果。分析端讓所有態共用一個 σ 的裁決另行處理。

## 依據

以上比值與裁決來自未入庫的討論和分析紀錄。舊樣板的相關係數可從 Git 查到，現行欄位與取樣公式見 [FTF 欄位指南](../../zcu_lab/simulation_envs/FTF/field-guide.md#adc)。
