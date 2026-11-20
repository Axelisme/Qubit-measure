---
date: 2026-11-20
status: active
superseded_by: null
related:
  - docs/adr/0074-flat-readout-module.md
  - docs/discussions/2026-11-20-herald-reset-shot-masks.md
  - docs/discussions/2026-11-15-two-qubit-rb-simultaneous-gates.md
  - zcu_lab/v2/two_qubit/guide.md
basis:
  - "repo@c333da7d8：九個獨立 heralded 入口的分析與 mux herald reset"
  - "使用者 2026-11-20 裁決。數值核對腳本未入庫，計算條件與數字見正文。"
---

# 只為共用參數保留獨立 heralded 實驗

## 結論

使用者在 2026-11-20 決定只保留 `rb_heralded` 與 `conditional_phase_heralded`。其餘七個改在一般入口選 `reset/mux_herald`，只留 gg。被刪入口的舊存檔不再載入，現行規則見 [ADR-0074](../adr/0074-flat-readout-module.md#保留獨立-heralded-實驗的條件)。

## 問題

Reset 可以在序列前篩掉不在 gg 的 shots，但仍有九個實驗各自保存 81 類 herald 與終端聯合計數。保留這些計數要有分析上的理由。使用者說：「有一些实验呢，它天生不在意 GG 还是 EE，它在意的是 GE 的交换，或者是保留在 GGEE 的属性。这种情况下，利用 hurdling 去。诶，具体来说，用交叉矩以及 hurdling 这两种方案去分析，有助于提高拟合可信度。那其他实验就用 reset 模块就好了。」使用者接著問：「你确定对这个实验语义下 G G 跟 V 一的行为是等效的吗？」

「hurdling」是 heralding，「V 一」是 ee。「交叉矩」依上下文指 gg、ee 兩組聯合擬合。要問的是準備誤差會不會改變兩組共用的參數。若誤差只改變每組自己的振幅或偏移，才保留獨立入口。

## 考慮過的選項與比較

| 選項 | 保留的獨立入口 | 依據 |
| --- | --- | --- |
| A | RB、conditional phase | RB 的衰減參數不受準備誤差影響；conditional phase 的寫回只用 π 交點。 |
| B | A 再加 idle ZZ | 兩組都能擬合 ζ，但準備污染會移動 ζ。 |
| C | B 再加 joint readout | 兩組都有混淆矩陣，但矩陣本身包含準備誤差。 |

### 逐實驗盤點

| 實驗 | gg、ee 的分析用途 |
| --- | --- |
| `rb` | 兩組共用衰減 `p`。依 RB 模型，準備誤差只改各組的 `A`、`B`。這項沒有跑數值。 |
| `conditional_phase` | 兩組用相同的 π 交點。離開 π 時，準備污染可以改變擬合的 phase。 |
| `idle_zz` | 兩組擬合的 ζ 都會隨準備污染改變，低 ζ 的偏差尤其大。 |
| `joint_readout` | 混淆矩陣量的是準備加讀取的誤差。ee 多出的準備誤差直接進矩陣。 |
| `chevron` | 只用 gg 擬合；ee 不進 fit。 |
| `constructive_phase` | 只用 gg 擬合；ee 不進 fit。 |
| `cz_amplitude` | 只用 gg 擬合；ee 不進 fit。 |
| `conditional_spectroscopy` | 只用 gg 判讀校準；ee 不影響結果。 |
| `local_z` | 兩組各自擬合，只用 gg 寫回。ee 斜率包含 local Z 加上每個 gate 的 conditional phase，不是 gg 量的斜率。 |

### 數值核對

ε 是每組 herald 到序列開始時已不在所選 herald 態的比例。來源是 herald 誤判與等待期間的 T1 衰減。ee 的 ε 比 gg 大。

理想 conditional phase gate 下，將 ee 組的 σ 反號並對調名目 spectator 分支後，與 gg 組的 σ 最大差為 4×10⁻¹⁶。加入準備污染後，π 交點仍不變。163° 時已有偏移，所以不能用 π 交點的結果推論整條曲線都等效。

| 真實 conditional phase | ε = 2% 的相位誤差 | ε = 5% | ε = 10% |
| --- | ---: | ---: | ---: |
| 180° | 0.00° | 0.00° | 0.00° |
| 163° | −0.60° | −1.71° | −4.41° |

Idle ZZ 的核對對每個實際 spectator 態用單一阻尼 cosine 擬合。延遲窗為 0 到 10 µs，T2 為 20 µs。表中是擬合 ζ 除以真實 ζ。

| 真實 ζ | ε = 2% | ε = 5% | ε = 10% |
| --- | ---: | ---: | ---: |
| 20 kHz | 0.968 | 0.918 | 0.831 |
| 50 kHz | 0.996 | 0.987 | 0.964 |
| 200 kHz | 1.003 | 1.008 | 1.017 |

## 被否決的選項與理由

不採 B，因為 ee 的準備污染會拉動 idle ZZ 的 ζ。不採 C，因為混淆矩陣直接量到準備污染。使用者答「1A 2可以」，同意選 A，並將保留條件寫入 ADR-0074。

本次推翻[原 herald reset 裁決](2026-11-20-herald-reset-shot-masks.md)中保留全部九個獨立入口的部分。該裁決保留了全部條件資訊，卻沒有區分分析是否使用這些資訊、兩組是否在估同一個量。Reset 的放置和取數遮罩不變。[RB 並行排程紀錄](2026-11-15-two-qubit-rb-simultaneous-gates.md)也曾確認保留九個入口；RB 入口仍保留，本次不改當時的排程裁決。

## 留下的問題

這些數字只核對準備污染對模型參數的影響，沒有量到實機的 ε、分類誤判率或 T1。若 conditional phase 日後用離開 π 的點寫回，須重新核對兩組共用參數的條件。

## 依據

九個實驗的分析盤點基於 `c333da7d8`。數值核對腳本未入庫，計算條件與結果寫在上文。實驗入口和操作見 [two-qubit guide](../../zcu_lab/v2/two_qubit/guide.md)。
