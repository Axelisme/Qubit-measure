---
date: 2026-11-23
status: active
superseded_by: null
related:
  - zcu_lab/v2/README.md
  - docs/discussions/2026-11-20-heralded-experiment-pruning.md
basis: 8cd18afcda65bba24a81f0b7e6c03b8e7f56e613
---

# 以準備後的 herald 分支量條件頻譜

## 結論

新增獨立的 [heralded conditional spectroscopy](../../zcu_lab/v2/two_qubit/conditional_spectroscopy/guide_heralded.md)。先做四種名目 I／Xπ 準備，再用 mux herald 分類。按實測 gg、ge、eg、ee 畫終端 P(e)，不聯合擬合。一般入口的行為與存檔格式不變。

## 問題

一般條件頻譜選 gg herald 時，先篩選，再以 Xπ 準備其他分支。半磁通點的熱布居降低保留率。若 Xπ 準備失敗，該 shot 仍算在名目分支。使用者說：「我覺得你應該安排的不是準備在gg/ge/eg/ee，而是post-selection分類gg/ge/eg/ee」。

## 考慮過的選項與比較

獨立入口保留 herald 與終端的聯合計數。四種名目準備各取同樣多的 shots，讓熱布居分配到不同 herald 類別。Xπ 誤差會改變各類的保留數，分組仍依讀出分類。使用者說：「你還是可以用pi來製備來是分類比例與熱態無關」，接著澄清：「你搞錯了…當然是先打Xpi然後在herald讀出」。對更正後的獨立入口方案答「可以」。

[2026-11-20 的先例](2026-11-20-heralded-experiment-pruning.md) 只為 gg／ee 兩組共用參數而保留 RB 與 conditional phase 的獨立入口。本案要分別讀出四個初態條件，不估一個共用參數。這是原則的例外，兩者都記在 [Two-qubit 實驗清單〈何時保留獨立 heralded 入口〉](../../zcu_lab/v2/README.md#何時保留獨立-heralded-入口)。只需篩選初態的實驗仍用 herald reset。

## 被否決的選項與理由

在原入口加模式會讓同一 tag 兼用 9 類與 81 類資料，Analyze 也得分成兩條路。保持原狀則要增加嘗試數來補回 gg herald 排掉的 shots。只做 herald 分類、不先用 Xπ 平衡，保留數仍隨熱布居改變。把 herald 放在 Xπ 之前，無法依 Xπ 後的狀態分類。

## 留下的問題

用頻譜的峰谷找候選 RF。圖不辨識單一高能階，也不擬合峰位或寫回。準備 × herald 診斷提供各類的 shot 比例，不是態純度或 gate fidelity。實測分類的比例不保證等於四分之一。

## 依據

2026-11-23 使用者裁決。原始決策前工作檔未入庫。[一般條件頻譜 guide](../../zcu_lab/v2/two_qubit/conditional_spectroscopy/guide.md) 記錄名目準備入口的行為。
