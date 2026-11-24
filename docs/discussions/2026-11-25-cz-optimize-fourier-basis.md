---
date: 2026-11-25
status: active
superseded_by: null
related:
  - 2026-11-24-cz-optimize-search-space.md
  - 2026-11-24-cz-optimize-native-labber-format.md
basis:
  repo: 13aa91a3c25169d7ed003e0b187e71c1cad5cf85
  external: null
---

# CZ 最佳化搜尋 FAST-DRAG 權重

## 結論

普通與 heralded CZ optimize 都改用前三個傅立葉諧波構成 I，Q 用 I 的解析導數乘 β / (2π)。CMA 每次搜尋兩個 I 權重比例、β、共同場倍率和兩個 local Z。評估 0 照原樣播放校準 pulse。FAST-DRAG 起點從設計權重與 β 建立，不擬合樣本。

## 問題

自然樣條讓 I 在兩端的值為零，但斜率不一定為零。樣條也不能用設計時的 FAST-DRAG 權重直接起步。2026-11-25，使用者指出：「我發現cz_optimize的波形在邊界的一階倒數不是0，而drag刻意設定成0，我認為有差別」，並問：「我開始懷疑，cz_optimzie是不是參數應該設定成傅立葉級數的權重，而不是時間上的節點」。

## 比較與裁決

- 把樣條改成 clamped 可以固定端點斜率，但起點仍要擬合，節點也不對應諧波。保留原樣條則兩個問題都留著。使用者在 2026-11-25 選傅立葉權重方案，回答「A」。
- 對 Q 考慮過獨立 sine 權重，或讓 β 控制同一組 I 諧波的導數。使用者在 2026-11-25 回答：「當然是fast-drag的形式但權重被優化啊，每個參數都要很重要，不然浪費優化迭代次數」。因此 Q 沒有獨立權重。I 在兩端值與斜率都為零，Q 在兩端只保證值為零。
- 同比放大所有 I 權重會同比放大 Q，複數峰歸一會消掉這個變化。因此固定起點絕對值最大的有號權重，只搜尋另外兩項的比例。用三個 I 諧波保留原 FAST-DRAG 的頻譜設計，不增加更高階權重。這個座標在一次 Run 內不跨過固定權重為零的形狀。
- 延續[等長 arb](2026-11-24-cz-optimize-search-space.md)及[具名欄位](2026-11-24-cz-optimize-native-labber-format.md)兩項做法。結果保存重建後的三個 I 權重與 β，重畫不需依賴起點的固定項。舊 `i_knot_*`、`q_knot_*` 欄位不能用新定義解讀。Native tag 加上 `fourier-v2`；依[舊結果格式裁決](../adr/0069-experiment-core-and-frontends.md#只有模擬測試資料時的舊結果格式)拒讀，不遷移舊檔。

## 留下的問題

原離線 FAST-DRAG 用有限差分算 Q，端點單邊差分會留下小幅非零 Q。新 CMA 起點用解析式；評估 0 仍保留原資產的實際樣本。起點和候選的頻譜可比較，但離線 probe 不驗閘保真度。

## 依據

[操作指南](../../zcu_lab/v2/two_qubit/cz_optimize/guide.md)列出欄位、播放與載入方式。[共用波形模組](../../zcu_lab/v2/support/cz_optimize_waveform.py)定義包絡和初始權重。原始裁決與離線樣本未入庫；本篇保留選擇所依據的數學關係和使用者原話。
