---
date: 2026-11-21
status: active
superseded_by: null
related:
  - zcu_lab/v2/two_qubit/guide.md
  - zcu_lab/v2/two_qubit/rb/guide_heralded.md
  - zcu_lab/v2/two_qubit/conditional_phase/guide_heralded.md
  - zcu_lab/v2/mux/onetone/guide.md
  - lib/zcu_tools/program/v2/modules/herald.py
  - docs/adr/0074-flat-readout-module.md
  - docs/adr/0083-program-macros-and-timelines.md
  - docs/discussions/2026-11-20-herald-reset-shot-masks.md
  - docs/discussions/2026-10-30-readout-optimization-metrics.md
  - docs/discussions/2026-11-21-readout-timing-defaults.md
basis:
  - "repo@666eb48ac：調查時的 main"
  - "FTF conditional phase 與 joint readout 的模擬紀錄、離線編譯與調查報告未入庫，必要數字與判讀寫在正文"
---

# Herald 後的等待由 readout 的 `post_delay` 擁有

## 結論

2026-11-21，designer 依使用者授權代裁：herald 後沿用 readout 的 `post_delay`，讓腔在第一個 drive 前放空。Herald 與 terminal 讀出共用這個值。不增欄位、不改 cfg 或 schema，也不放寬兩次讀出設定必須相同的限制。此代裁尚待使用者覆核。

[Two-qubit 共用指南](../../zcu_lab/v2/two_qubit/guide.md#herald-reset-與獨立-heralded-入口)、[heralded RB](../../zcu_lab/v2/two_qubit/rb/guide_heralded.md)、[heralded conditional phase](../../zcu_lab/v2/two_qubit/conditional_phase/guide_heralded.md)和 [mux onetone](../../zcu_lab/v2/mux/onetone/guide.md)各自說明何時要設等待、太短時會看到什麼，以及怎麼驗證。操作者從線寬估場衰減時間 `1 / (π · FWHM)`，先取數倍，再用 `two_qubit/joint_readout` 比較 herald 後名目 Xπ 在所選等待與更長等待下的 terminal e 比例。

## 問題

新 mux readout module 的 `post_delay` 預設為 0 µs，沒有實驗校準它。FTF 的低 SNR 個案中，Round 13 的 conditional phase 掃描花了 1142 s，卻沒找到 π 交點。當時的 guide 沒提讀出後要等腔放空，操作者無從判斷該先查等待還是重掃 CZ。

Round 14 不打 CZ，先用 gg herald，再比較 F2 的 I 與名目 Xπ。等待 0 µs 時，F2 的 terminal e 比例從 I 的 6.01% 變成 Xπ 的 11.56%；等待 2 µs 時，從 4.71% 變成 92.90%。這是終端分類比例，沒有扣除 I 基線。Round 16 用 2 µs 在 Round 13 的同一掃描窗找到唯一 π 交點。沒有掃過兩值之間的等待，不能據此定出最短或最佳值。

## 成因

離線編譯顯示，等待 0 µs 時，herald 的 ADC 視窗從 1.16955 到 4.16434 µs；generator 波形在 4.16667 µs 結束，F2 Xπ 在同一時刻開始。ADC 視窗與 Xπ 沒有重疊。排程符合 [ADR-0083](../adr/0083-program-macros-and-timelines.md#timeline-只存結尾與-reference-分開)：等待從 generator 結束起算，ADC 視窗不延長作者的時間線。

工作假說是第一個 drive 落在殘留腔場裡。FTF 在同一 shot 內把腔場帶到下一段。殘留光子可透過 Stark shift 改變 qubit 轉移頻率並造成退相干，使固定頻率的 Xπ 轉不到位。R2 的線寬約 1.1 至 1.21 MHz，對應場衰減時間約 0.26 至 0.29 µs；2 µs 約為七倍。調查沒有直接量到失諧。

兩次讀出都加了等待。FTF 的各 shot 獨立初始化，腔從真空開始；terminal 之後多等 2 µs 不會改變已取得的 terminal IQ，也不會把狀態帶到下一個 shot。這個判斷限於 FTF。

## 考慮過的選項與比較

| 選項 | 做法 | 取捨 |
| --- | --- | --- |
| K：維持現狀 | 不改 guide 或設定。 | 不增加維護成本，操作者仍得從弱翻轉或缺 π 交點自行反查。 |
| N：沿用 `post_delay`（採用） | Guide 說明共用欄位，操作者驗證後設定自己的 readout module。 | 不改程式與 schema。新 module 仍從 0 µs 起步；加長 herald 後的等待，也加長 terminal 後的等待。 |
| S：加 `post_herald_wait` | 在 herald 後插入獨立等待，保留 readout 的 `post_delay`。 | 能獨立設定，但要解釋兩段等待如何疊加，還要改 11 個實驗 schema。 |
| E：讓 herald 用另一份 readout | 兩次讀出各自設定 `post_delay`。 | 能分開設定，但要改共用契約，另行維護兩次讀出的分類器與 ADC 一致性。 |
| D：表單依線寬給初值 | 建立新 module 時，由 resonator 的 κ 推導 `post_delay` 種子。 | 操作者不用從 0 起調；κ 缺值或不準時，種子也可能不適用。 |
| G：在零等待時警示 | 表單選 herald 且 `post_delay == 0` 時提醒，不擋 Run。 | 提醒操作者確認等待；若腔衰減夠快，0 µs 也可能適用。 |

## 被否決的選項與理由

- K 留下同一個操作盲點。操作者仍可能先跑完整掃描才發現 Xπ 幾乎沒翻轉。
- S 沒有證據顯示兩次讀出需要不同等待。它新增同義欄位，還得配合一次 schema major 升級；把新增欄位預設為 0 也不會解決問題。
- E 要推翻 [ADR-0074 的共用 readout 契約](../adr/0074-flat-readout-module.md#herald-reset-與逐-shot-遮罩)。如果兩套 ADC 或分類器設定日後分歧，herald 篩選的條件也會改變。

## 另立的表單問題

D 與 G 都能和 N 並用，沒有納入本次。D 可參考 onetone 從線寬估 ringup、替 `trig_offset` 給種子的做法。G 只提出非阻擋警示，不把 0 µs 判成非法。

## 先例與立場

- **遵循 ADR-0074。** 2026-11-19，使用者在討論 [herald reset](2026-11-20-herald-reset-shot-masks.md) 時問：「關於reset/herald，post_delay不是已經在readout裡就有了嗎？」Herald 沒有自己的 `post_delay`，兩次讀出使用相同設定。腔放空時間取決於讀出條件，例如線寬與功率，不是 herald 專屬的設定。
- **遵循讀出鏈。** [Chain power guide](../../zcu_lab/v2/twotone/ro_optimize/chain_power/guide.md) 已用 readout 的 `post_delay` 當相鄰讀出之間的光子衰減間隔。
- **區別實驗用的等待。** [Readout power guide](../../zcu_lab/v2/twotone/ro_optimize/power/guide.md) 的 `wait` 是第一個 readout 到中間 π pulse 的實驗變數。操作者可比較不同 `wait` 下的結果；這不需要另給 herald 一個欄位。
- **沿用驗證思路。** [2026-10-30 的討論](2026-10-30-readout-optimization-metrics.md)提出延長等待，直到 π pulse 的失準比例 f(τ) 不再下降。本次 guide 採用比較長短等待的做法，沒有新增 mux 版的 f(τ) 校準實驗。
- **預設值只是起點。** 2026-11-20，使用者說：「2基本上是A，但預設值可能沒設計好，所以實際上應該是看測量者的需求。」不符需求的初值另當表單問題處理。同日的 T2 guide 裁定也要求說明掃描需要多少點和範圍、為何需要，以及不足時圖上怎麼看。這次也把需求、原因與症狀寫給操作者，不把初值當校準結果。

## 留下的問題

- 硬體的 terminal 後等待會不會影響下一個 shot，沒有量測證據。不能從 FTF 的獨立初始化推論。
- D 的種子倍數未定。FTF 的 0 µs 與 2 µs 只證明這兩個設定的差異，不能定通用倍數。
- G 要先確認表單框架能否在不擋 Run 的情況下顯示警示。

## 依據

調查以 `666eb48ac` 的程式與文件為準。共用設定的契約見 [ADR-0074](../adr/0074-flat-readout-module.md#herald-reset-與逐-shot-遮罩)，起算點見 [ADR-0083](../adr/0083-program-macros-and-timelines.md#timeline-只存結尾與-reference-分開)。FTF 模擬 run、離線編譯記錄與調查報告未入庫；上文已記下用來比較選項的數字與限制。

## 後記（2026-11-21）

Herald 與 terminal 仍共用 `post_delay`。Mux Open／Reset 在可用線寬時用 `5/(2π·κ)` 給種子，不再一律從 0 µs 起步。保留原值與缺線寬的規則見[讀出時序預設](2026-11-21-readout-timing-defaults.md)。
