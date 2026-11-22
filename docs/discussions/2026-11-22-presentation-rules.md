---
date: 2026-11-22
status: active
superseded_by: null
related:
  - ../adr/0069-experiment-core-and-frontends.md#呈現規範
  - ../adr/0075-agent-interface-layers.md#判讀依據放知識庫
  - 2026-11-20-experiment-naming-and-display.md
  - 2026-11-20-recipe-report-polish.md
  - 2026-11-20-rb-round-reduction-and-fit-quality.md
basis:
  repo:
    wave/presentation-rules-20261122/integration: 5832fe32321e68e7cde5b90a66528d32b8052a09
  external:
    - "Leon Ding, Novel Gates with Superconducting Fluxonium Qubits, MIT Physics PhD thesis, 2023"
---

# 實驗呈現規範：圖、摘要與讀者檢查

## 結論

使用者把圖留給物理量、模型與判讀。Run 身分不印在圖上，實際採集設定放在 GUI 和 MCP 的可見 Analyze 摘要。對數值大小敏感的一維資料畫 error bar；看峰位的 spectrum 和二維圖不用。這兩項決定與 2026-11-20 已採用的 R0–R9、統一顯示單位一起寫進 [ADR-0069〈呈現規範〉](../adr/0069-experiment-core-and-frontends.md#呈現規範)。[實驗呈現指南](../../zcu_lab/v2/figure-style.md)收錄可操作的條文、例子與讀者檢查。

Designer 補上 fit 品質、population 軸、分支標籤，以及表單、guide 和 recipe 說明。這部分為 designer 代裁，使用者未覆核。使用者指定 Ding 博論為版式參考，並未要求本地實驗復刻論文中的量測。

## 問題

讀者檢查讓懂物理、沒看過實作的人只讀表單、guide、liveplot、Analyze 圖與摘要。一次 joint readout 讀者包的表單寫 `1000×10`，圖只標 `acquired_shots=8192`。讀者不知道兩者是否來自同次取數。這不表示其中一個數字有錯，卻暴露出缺少來源與實際採集資訊。讀者也無法從只有 fit 誤差的圖判斷單點不確定度。其他答卷指出品質值沒有解釋、欄位語意與單位不清，以及沒有足夠資訊算出下一步該改多少。

2026-11-22，使用者給了圖的方向：「關於實驗liveplot與分析圖，我認為可以參考ding博倫的格式，這才是人看的圖。」問題是哪些內容值得借用，以及要怎麼避免把不同的 observable、掃描量或準備態當成同一個實驗。

## 考慮過的選項與比較

**採集資訊放哪裡。** 第一輪的 A-a 只在圖頁尾放，A-b 只放 GUI／MCP 摘要，A-c 在摘要和圖的短頁尾都放。Designer 原推 A-c，希望單看圖也能辨認來源。使用者答「1我以為圖的名字已經足夠定位」。Designer 因此重整 A-b：存檔路徑定位，圖不加 run 身分，實際採集設定留在摘要。使用者同意，理由是「圖應該放純理論實作者關注的重點，而不是實驗作者關注的資訊。」

**哪些資料要畫 error bar。** B-a 是能算就一律畫。B-b 按圖型，機率與計數畫二項 error bar、標 shot 數，平均訊號不強制。B-c 不訂規則。Designer 原推 B-b，使用者改按讀者要做的判斷來分：「我有考慮給部分實驗的點加上errorbar，通常是比如rabi/t1之類的一維數據，對數據值的大小比較敏感的。spectrum之類的或是二維圖就不用」。

**補足讀者答不出的問題。** 初稿把 fit 品質、表單語意和量化的下一步寫進規範，並用 Ding 的布居圖、分支標示與序列示意當參考。Challenge 收窄了八處容易被當成新功能的寫法：

1. R12 的「能換算成 population」限於現有 Result 已支持的物理量。把平均 IQ 縮到 0–1 不會得到分類機率，不能為了改圖先加跑校準。
2. R3b 只呈現分析 owner 已給的品質值和既有選取判定。繪圖端不另算、不訂門檻。這遵循[既有 fit quality 先例](2026-11-20-rb-round-reduction-and-fit-quality.md)。
3. F1 只改看得到的 label，保留 cfg key 和位置名。F2 沿用使用者已選的顯示單位，不另設一套詞彙。
4. F3 保留一句填寫說明，label 已表明效果時不再重述。表單的 module 或 slot 可以說明脈衝角色，不能因此承諾顯示解析後的參數。
5. Gd1 是內容檢查表，不固定章序。適用範圍談物理假設、observable 和可辨識性。依 [ADR-0075](../adr/0075-agent-interface-layers.md#判讀依據放知識庫)，板子限制與底層錯誤不放 guide。
6. R13 只用現有 Result 或 Analyze options 有的資訊補分支名，不取代位置名或元件顯示名。不為了排圖多量一個分支。
7. Gd2 用圖號和頁碼指向論文版式，再說明和本地實驗的差異。相同排版不等於測到相同的量。
8. S1 只把對應表單的量測參數用相同 label 和單位說清楚，控制欄位說明自己的作用。S2 對齊同一個量的可見名稱，不複製分析 owner 的摘要型別，遵循[既有 recipe 報告的做法](2026-11-20-recipe-report-polish.md)。

## 被否決的選項與理由

- 圖頁尾只放或重複放採集資訊（A-a、A-c）：實驗名和 run id 已可由存檔路徑定位，頁尾會把取數設定帶進以物理判讀為主的圖。
- 能算就一律畫 error bar（B-a）：峰位和二維圖不靠逐點大小判讀。若缺逐輪資料或二階矩，為了畫圖補取數也有成本。按機率／計數分類（B-b）則漏掉對數值大小敏感的其他一維資料。不訂規則（B-c）解決不了讀者無法評估單點不確定度的問題。
- 保留兩種時間單位寫法：`us` 與 `µs` 並存。使用者選統一顯示詞彙，不改保存單位。
- 每張圖都放序列示意，或一律改畫 population：單一序列圖不一定因此多提供判斷資訊；只有平均訊號時，population 可能需要額外分類校準。
- 要求表單顯示解析後的 module 參數：先查框架能否提供，再決定要不要做。

## 留下的問題

框架尚未統一供應可見摘要的完成輪數、提前停與 shot 數。標題、圖例和 error bar 所需資料的共用支援也另題研究。各家族改圖時先用現有資料，再以讀者檢查驗收，不把待做的框架能力寫成現況。

R12、R13、F1、F3／F4、Gd2 沒有使用者直接定下的原話。這些條文取自使用者的版式指示及讀者答卷，留待使用者覆核。

## 依據

裁決日期與使用者原話見 [ADR-0069〈呈現規範〉](../adr/0069-experiment-core-and-frontends.md#呈現規範)。使用者的原始問答、讀者包與逐圖對照紀錄未入庫；本篇保留用於比較的事實。Leon Ding 的 MIT 物理博士論文 *Novel Gates with Superconducting Fluxonium Qubits*（2023）的對應圖號與印刷頁碼列在[實驗呈現指南](../../zcu_lab/v2/figure-style.md)。
