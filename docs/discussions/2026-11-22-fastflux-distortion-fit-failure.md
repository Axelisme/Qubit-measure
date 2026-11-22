---
date: 2026-11-22
status: active
superseded_by: null
related:
  - ../../zcu_lab/v2/fastflux/distortion/freq/guide.md
  - ../adr/0085-numerical-fitting-owners.md
  - 2026-11-22-presentation-rules.md
basis:
  repo_commit: a90cb66bbaa7e5269572d375fef321936dbeb3cc
  clean_baseline: d7efc0e694c63f6f1bc3639dd65338aa33d061bd
  external_sources: none
---

# Fast-flux 頻譜擬合失敗

## 結論

保留 `fastflux/distortion/freq` 的整次 Analyze 失敗語意。略過含 NaN 的列不變；嘗試擬合有限資料的一列而失敗時，不發布部分成功圖。保留掃描資料與 Live 圖。這次只在 [leaf guide](../../zcu_lab/v2/fastflux/distortion/freq/guide.md) 說明，不改數值或取數。

## 問題

一組 11×15 的 FTF 掃描完成，Native 往返等值。Lorentzian 擬合超過評估次數，公開 Analyze 結束時沒有圖。另一組較早的輸入在 Run 就報 `NoValidKernel`；它與 Analyze 失敗不是同一個問題。

## 選項與取捨

先從舊 H5 在乾淨基準重讀。這能分辨資料相容與擬合結果，不必重採，也不會把早先不同狀態下的 Live 圖當乾淨前圖。如需新的同源前後比較，先找能覆蓋預期共振並足以判讀 flux 暫態的固定輸入。本案遵循 ADR-0085，該 leaf 擁有這組等待列與參考估計，不套其他實驗的分 tone 失敗隔離。

## 未採用

部分成功要另定失敗列怎麼呈現，以及第一個參考點或全部列都失敗時回傳什麼。只提高評估次數、改初猜或刪列亦未證實是數值修正。換相位實驗當代表會失去頻率峰位的覆蓋。

## 留下的問題

本批未做校準或新前後採集，代表 R0 未驗。另見 [Families 部分交付](2026-11-22-families-partial-delivery.md)。

## 依據

原始資料與診斷未入庫。上述結果分別來自不同 FTF 輸入；599 MHz 是其中一次的模型先驗，不是該次量到的共振。未用診斷結果改動 [呈現規範](../../zcu_lab/v2/figure-style.md)。
