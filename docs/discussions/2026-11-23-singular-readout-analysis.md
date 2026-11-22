---
date: 2026-11-23
status: active
superseded_by: null
related:
  - ../adr/0069-experiment-core-and-frontends.md
  - ../adr/0074-flat-readout-module.md
  - 2026-11-22-fastflux-distortion-fit-failure.md
  - 2026-11-22-figures-only-analyze.md
basis:
  repo_commit: 563c5abffd3580b18aa3ff2866e6a8d64b365926
  external_sources: none
---

# 一維讀出校正無解時保留分類比例

## 結論

Designer 在 2026-11-23 代裁：MIST 的 frequency、gain、preparation frequency，另加 reset check、single-shot T1 與 T1 with tone，共六個 circle 一維 Analyze，若保存的讀出矩陣奇異，或把有限分類比例解成非有限值，只畫同次 Run 的 G/E/Other 分類比例。圖標為 `classified fractions`，附 `Readout correction unavailable`，log 留數值原因。這些是已分類的 shot 比例，不是原始 IQ、純態人口或新的校準樣本。

沿用既有 figures-only 發布。直接呼叫回 `None`，完成只代表發布了圖。GUI 與 MCP 保留本次 Run 的採集設定行。按操作查詢的數值摘要是 `{}`，沒有 fit 或 reset 統計，也不留下前一次的寫回提議。這項代裁待使用者追認，不是使用者親自採用的 fallback 原則。

## 問題

例如重開一個 T1 with tone Run，保存的矩陣可能連兩種準備狀態都分不開。先前 Analyze 會在畫圖前因求解失敗而中止。Load 不重播 Live，操作員因此連曝光期間兩種準備的分類趨勢也看不到。保存格式允許有限但不滿秩的矩陣；另有能求解、卻將有限輸入解成非有限值的矩陣。兩者都沒有可用的校正系列。

缺 error bar 是不同的問題。只要校正點有限，就保留不截斷的點；沒有有限 shot SEM 才省掉該棒，不按矩陣條件數另設門檻。原本缺測的分類點不算求解溢位，沒有保存矩陣也仍走原本路徑。

## 選項與取捨

- 照原樣失敗最省改動，但只留下 NumPy 例外。改成帶原因的 Analyze 失敗會讓訊息可讀，仍不會從這個 Run 發布新圖，也不會替換同源先前的分析。
- 六個 circle 入口發布分類圖需要各自保留座標與分支。好處是操作員仍能看已發生的分類計數，前端不必增加新的成功狀態。T1 with tone 可看到兩種準備的曝光時間曲線，但沒有衰減擬合或常數。
- 若也讓一維 Raman／STIRAP delay 只發布分類圖，兩個具名 recipe 就不能照現在的型別讀取結果。只替兩者換錯誤文字也不會產生可看的圖，還要把 Raman 的一、二維錯誤分支拆開。

## 未採用

沒有拿分類比例代校正數值擬合，也沒有用偽逆或 live library 補解。這些做法會換掉估計量或校準來源。也沒有新增能在 RPC 上解碼的校正結果型別。當時沒有 caller 需要機器判讀這個結果，已有具名圖加空摘要的通道。

Raman frequency／gain、STIRAP delay 與二維分析留在既有契約。兩個 recipe 在 Analyze 失敗後仍按原路徑嘗試保存 counts；不把失敗改成可供 recipe 消費的空摘要。

## 留下的問題

當時未找到需要變更 centroid3 recipe 契約的 caller。若日後需要一維 centroid3 的同類診斷，須先決定具名 recipe 要如何區分無校正結果與成功的數值摘要。

## 依據

- 2026-11-23 使用者對誤差棒說「反演出來的量如果可以從變異矩陣推導那也畫。」並答「1B 23可以」。討論中也說「我認同判斷標準是『事實』還是『判斷』」。這些話支撐固定矩陣、保留可算的點與棒，以及圖如實呈現觀察；不等於使用者選了這項失敗分支。
- 使用者曾說「分析是客觀的，可以有擬合品質之類的，完全不知道寫回的概念，這是有意的認知隔離。」這次沿用 [ADR-0069](../adr/0069-experiment-core-and-frontends.md#三種分析生命週期) 的 core 與 adapter 分工，也沿用 [ADR-0074](../adr/0074-flat-readout-module.md) 的 saved cfg 校準來源與 centroid3 preparation-label 量義。
- [Live-only figures-only Analyze](2026-11-22-figures-only-analyze.md) 已用同次 Run 的具名圖和採集行，不造空的分析型別。[Fast-flux 頻譜擬合失敗](2026-11-22-fastflux-distortion-fit-failure.md) 保留整次失敗，因一列失敗會影響主結果的參考；本題的分類比例可直接從各點 counts 計算，不依賴那個失敗的解。
- 2026-11-23 使用者授權「我要去睡了，之後由你代裁並推進，包含方向裁決」，並說「不應該有需要等我判斷的事項，一律都由你代裁後我回來追認」。Designer 據此代裁。當時的調查與 NumPy 驗算未入庫；驗算分開核對真奇異、近奇異但有有限解，以及有限分類比例的反解溢位。
