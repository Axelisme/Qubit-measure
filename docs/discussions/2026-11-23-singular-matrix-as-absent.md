---
date: 2026-11-23
status: active
superseded_by: null
related:
  - ../adr/0069-experiment-core-and-frontends.md#報告只寫本次做了什麼
  - ../adr/0074-flat-readout-module.md#classifier-隨-readout-保存
  - 2026-11-23-singular-readout-analysis.md
  - 2026-11-23-classified-raman-stirap-analysis.md
basis:
  repo_commit: d7911319985a4bfe17aec50b7ecb60953caa88b0
  external_sources: none
---

# 讀出矩陣奇異就當作沒有矩陣

## 結論

使用者在 2026-11-23 裁定：Analyze 從保存的 confusion matrix 得不到可用校正時，走與沒有矩陣相同的路徑。依各實驗的無矩陣契約，用未校正的分類比例出圖、擬合、回摘要，以及提出原本會有的 Writeback。沒有 Writeback 的實驗不新增提案。

Saved reader 遇到奇異矩陣回 `None`。數值求解端若用有限矩陣把原有效點解成非有限值，整次分析不用校正。兩種原因各記一筆 warning。原本缺測或僅缺 SEM 不觸發回退；正常有限校正不變。

## 問題

夜間代裁讓六個一維 circle 入口只發布分類圖，拿不到 fit、正常數值摘要或 Writeback。Raman／STIRAP 一維仍可能在出圖前失敗。同一組計數若從未保存矩陣，部分實驗原本卻可以擬合和提出寫回。兩種結果不一致。

## 選項與取捨

- 保留只出圖的代裁，能讓操作員看到分類比例，卻拿掉無矩陣路徑原有的 fit 和 Writeback。使用者否決。
- 只改六個 circle 入口，能縮小改動，但同樣的保存矩陣問題在其他 Analyze 仍有不同結果。使用者要求二維也照同一規則。
- 採無矩陣等價。Reader 處理矩陣奇異，數值 owner 判斷實際輸出是否非有限。各實驗沿用原未校正路徑，不再保留一套特殊摘要或圖。

## 留下的問題

若某入口原本沒有可用的無矩陣路徑，必須先定義其圖和摘要，不用空成功結果掩蓋失敗。Raman／STIRAP 在[後續裁決](2026-11-23-classified-raman-stirap-analysis.md)補這個契約。

## 依據

- 2026-11-23，使用者：「0423矩陣奇異就不校正我接受，但還是可以提writeback，還有recipe接口本來就可以改，他是連動修改的」；「2矩陣奇異就跟沒矩陣一樣處理」。原始問答未入庫。
- 推翻[只出分類圖的夜間代裁](2026-11-23-singular-readout-analysis.md)。沿用 [ADR-0074](../adr/0074-flat-readout-module.md#classifier-隨-readout-保存) 的 saved cfg 校準來源。取法與 [ADR-0078](../adr/0078-evolution-module-boundary.md) 對合法可算輸入照算並警告的先例一致；這裡仍拒絕無效 cfg，不改有限校正的估計量。
