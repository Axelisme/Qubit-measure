**讀者：** operator、agent。從已校準的雙線 CZ 出發優化脈衝，再用獨立 seeds 驗收。

# CZ 優化

每發先執行可選的 Reset，每個隨機 Clifford 後播放一個 CZ，最後執行 recovery 和終端 mux 讀出。所有評估使用同一組 seeds。分數是各 seed 的終端 P(g,g) 等權平均，不是閘保真度。

## Run 前

1. 核對[位置與完整 mux](../guide.md#綁定位置與讀出-tones)，為兩個位置選已校準的 Xπ/2、Xπ 和 circle classifier。各 X pulse 與同位置 CZ pulse 要用同一 generator。若用 mux herald，核對[讀出與保留條件](../guide.md#herald-reset-與獨立-heralded-入口)。
2. 在 **Two-line CZ drive** 為兩線選相同形狀、時長的 `cosine`、`gauss`、`drag`、`const`、`arb` 或 `flat_top`。兩線可有不同的帶號 gain 和 phase。核對 **Local Z correction pair** 有序對應目前的 Qubit1、Qubit2，並使用 `$coupler.ext.cz_phase` 校準的角度。
3. **Interleaved Clifford depth** 選在已量的 IRB 衰減中段。**Clifford sequence seeds** 填 3–5 個不同整數。**Optimizer random seed** 控制候選，不改 Clifford 序列。
4. **Avoid frequencies (MHz)** 可填絕對實驗室頻率，例如量到但不想驅動的躍遷。留空時頻譜不畫頻帶。這個欄位不加入最優化目標。

評估 0 直接播放原 pulse。即使原 `arb` 不對稱，評估 0 也不改形狀。CMA 的初始均值另外取上升 ramp 做擬合，之後的候選都使用 I 鏡像、Q 反號鏡像和實數平台。Ramp 中點必須是正 I 且 Q 為零。`flat_top` 的 raise waveform 提供 ramp，剩餘長度是平台；其他包絡以整段作 ramp。`const` 的候選從短 cosine ramp 及等面積平台出發，評估 0 仍是原方波。

| 欄位 | 控制的物理量 |
| --- | --- |
| Initial I/Q control-point spread | 上升 ramp 的 I、Q 各三個控制點，歸一振幅範圍 −1 到 1。 |
| Minimum / Maximum ramp scale s；Initial ramp scale spread | `s` 乘上起點的參考 ramp 長。範圍須包含 1；spread 用自然對數尺度。 |
| Maximum plateau duration；Initial plateau duration spread | 平台長從 0 到所填上限，單位 ns。上限須涵蓋起點平台。 |
| Minimum / Maximum field multiplier g；Initial gain spread | `g` 同時乘上兩線物理場，不改兩線相對倍率。範圍須包含 1。切換表示時 raw gain 隨輸出尺度換算；若任一線的最大可達 raw gain 超出 ±1，Run 在取數前報錯。 |
| Initial local Z spread | 每個 CZ 後套用的兩個 virtual Z 角，單位 deg。 |

**Fix gate duration** 預設關閉。此欄控制 CMA 是否調整時長。

總評估數為 `1 + Optimizer generations × Candidates per generation`。每個實測候選以相同的 seed 數與 Reps 取數。包絡 `|I+iQ|` 超過 1 的 CMA 候選記零，不取數也不列入最佳。每個 seed 以保留 shots 為分母，包含 other；無保留 shots 時 Run 報該 seed。候選使用單一 arb 包絡，所需樣本隨 ramp 與平台總長增加。

## 比較圖與重載

- **Fit** 標出最高的完整實測評估。誤差棒是 retained-shot 的 P(g,g) SEM，不是 seeds 間散布。右欄顯示具名物理參數；評估 0 勝出時，右欄列的是 CMA 初始均值，不是原 pulse 的設定。
- **Envelopes** 疊畫評估 0 的顯示包絡、CMA 初始均值和最佳 CMA 候選。圖用 Qubit1 的 DDS gain 等效尺度，包含原 `flat_top` ramp 與平台的相對幅度。
- **Spectrum** 用複數 I/Q 比較評估 0 和最佳評估。0 MHz 是 CZ 載波，各曲線以自己的載波分量為 0 dB。避免頻率換算成相對載波偏移後以 ±0.5 MHz 帶標出。

**Save All** 在 Native `data.h5` 儲存具名參數、峰值、分類計數和起點顯示包絡。時間用秒、角度用 rad。Labber 匯出同一組具名參數和峰值。從 Native 檔重開完整結果。舊格式需用當初寫檔的版本開啟。若 Load 報 cfg 無法還原，仍可看圖，但沒有避免頻率帶，也不能寫回。起點引用 `arb` 資產時，Native 檔也保存資產路徑和 SHA-256，包括 `flat_top` 的資產 ramp。若資產缺檔或內容已變更，Analyze 摘要會指出路徑和原因，圖仍從已存樣本繪製。

## 審閱寫回並驗收

CMA 候選勝出時，提案包含一筆共用 arb 資產、兩筆 CZ pulse、兩筆 slot 和一筆 `$coupler.ext.cz_phase`。Pulse 的 gain 已按最佳 `g` 和新舊表示的輸出換算；載波、phase 和延遲不變。評估 0 勝出時不新增資產，pulse 與 local Z 維持原設定。如果原資產缺檔或內容已變更，拒絕寫回並指出路徑和原因。CMA 勝出的提案使用新資產，有效 cfg 下不依賴原資產。只有最佳分數高於評估 0 才預選提案。審閱後一起套用相依項目，再核對收據。

用[普通 paired RB／IRB](../rb/guide.md)驗收。核對 pulse、slot 和 phase pair，換一組未參與搜尋的 seeds，再比較 gate error 與 CI。不要用搜尋期間最高的 P(g,g) 當作改善證據。

共用操作見[量測流程](../../operator-workflow.md)。
