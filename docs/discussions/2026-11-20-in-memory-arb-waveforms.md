---
date: 2026-11-20
status: active
superseded_by: null
related:
  - ../adr/0072-module-boundaries.md
  - ../adr/0063-persistence-ownership.md
  - 2026-11-20-arb-asset-writeback.md
  - ../../lib/zcu_tools/program/v2/modules/README.md
basis:
  repo: a987118b6
  external: none
---

# CZ 優化候選用記憶體 arb 播放

## 結論

CZ 優化每次試播直接把 time、I、Q 樣本交給 program 的 `InMemoryArbWaveformCfg`。它不讀檔，也不能序列化或寫進 library。候選與檔案 arb 使用同一套取樣規則。在同一個 generator 上，兩種 arb 用相同樣本會宣告出逐位元相同的 envelope。最佳樣本等到使用者接受寫回才成為 NPZ 資產。規則見 [ADR-0072](../adr/0072-module-boundaries.md#共享核心-core-與-cfg_model) 與 [arb 資產寫回討論](2026-11-20-arb-asset-writeback.md)。

## 問題

CZ 優化會反覆試播新的樣本，當時的 native `arb` 每次建 envelope 都從路徑讀 NPZ。若每輪都用這條路，就要先替候選寫檔。使用者在 2026-11-19 指出：「P 六，我不希望的一个情况是，随机演算法中途的波形被存到永久的资产库里。呃，我不知道目前有没有保存在记忆体中的宣告 arbitrary pose 的方法。如果没有，可能需要考虑设计一个。」語音轉寫的「pose」指 pulse。

## 考慮過的選項與比較

- 每輪把樣本寫進 run 專用的暫存目錄，結束時刪除。這可沿用既有 `arb`，也不會把候選寫進永久庫，但每輪仍需寫、讀 NPZ，run 還要管理暫存檔。
- Program 直接收 `WaveformSamples`。候選沒有路徑，registry 改用 generator channel 與樣本雜湊共用 envelope。Gain、phase 或 frequency 不同，不需要再宣告一次相同的 envelope。閉環優化也可重用這個入口。使用者在 2026-11-19 比較兩案後答：「可以」。

這次遵循 [ADR-0063](../adr/0063-persistence-ownership.md)，永久樣本仍用既有資產，library 仍保存 key。也遵循 [ADR-0072](../adr/0072-module-boundaries.md) 的分工。Core 負責共用的樣本格式與雜湊，program 負責執行時播放，持久 Spec 不新增 style。

2026-11-20 agent 代裁，待使用者覆核：雜湊把全零 Q 與未提供 Q 視為相同。兩者都不傳 Q envelope，不應在同一個 channel 各占一份，也不應讓相同波形得到兩個發布 key。

## 被否決的選項與理由

- 暫存 NPZ 能滿足不進永久庫的限制，但不能提供使用者想要的記憶體宣告方式。加入 native cfg 後，run 不必替每個候選準備路徑。
- 永久保存新增的 `spline` style 也曾是選項。使用者在 2026-11-19 裁定：「波形我觉得存 arbitrary 就好了，arbitrary 就是拿来存这种特别的东西的。其他采用建议。」`arb_memory` 只補試播入口，沒有改變這個存法。

## 留下的問題

存檔不能保存候選的 native cfg 供日後直接重播。CZ 優化的設計是保存每輪的 spline 參數，而非每輪的 I/Q 陣列。要讓其他實驗直接引用最佳波形，仍需發布資產。

## 依據

當時的設計比較與原始工作檔未入庫。實作可核對 [program 的波形設定](../../lib/zcu_tools/program/v2/modules/waveform.py)、[envelope registry](../../lib/zcu_tools/program/v2/modules/registry.py) 與 [core 的樣本雜湊](../../lib/zcu_tools/core/waveform_format/__init__.py)。

代裁依使用者 2026-11-20 的授權：「STDO:我去睡了，我看過目前規劃方向沒有問題，請你之後幫我代裁設計與推進任務，都不需要問我，你自己決定」。
