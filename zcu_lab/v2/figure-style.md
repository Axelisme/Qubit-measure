**讀者：** lab-author、agent。用這份指南撰寫表單、guide 和 recipe 說明、設計實驗圖，並檢查操作員能否只靠這些內容判讀結果。

# 實驗呈現指南

呈現規範適用於 GUI 與 recipe。操作員看到的是 cfg 表單文字、operator guide、recipe schema、liveplot 與 Analyze 圖和摘要。建圖、顯示與更新的 API 見[實驗作者契約](experiment-authoring.md#圖的建立顯示與更新)。

## R0：不讀程式碼也能判讀

懂物理但不知道實作的操作員，只看 cfg 表單（含 label、單位與說明）、guide、liveplot 與 Analyze 圖，就能說出實驗量什麼、每個參數做什麼、圖上每個元素代表什麼、結果怎麼判讀。recipe 入口也用同一套物理用語。

壞例：Ramsey 圖上只寫 `detune = 0.402 MHz`，讀者分不出那是施加的失諧還是條紋擬合值，也不知道頻率該往哪裡改。

好例（本地改寫示例）：圖內標明擬合的條紋頻率，Analyze 摘要列出同一次 run 的實際 detune，guide 說明如何找出當次 π/2 脈衝來源頻率，並寫出符號慣例和修正式。有有效的條紋 fit 才用修正式。例如來源為 5800.000 MHz、施加 0.400 MHz、擬合 0.402 MHz，則建議值為 5799.998 MHz，比來源低 0.002 MHz。Ding 圖 5-9(b,e)（頁 152）示範如何把 Ramsey 類序列與對應的相位判讀放在一起。

### 組讀者包

選一個要驗收的家族代表實驗。把操作員可見的內容放進一份讀者包：

| 檔案 | 內容 |
| --- | --- |
| `form.md` | 從選定的 point 與綁定推導出的表單。照畫面記錄 label、單位、description、初值或選項、唯讀與必填狀態；留空的欄位保持空白，不補猜測值。不把 Hidden 欄位當成操作員可見的欄位。 |
| `guide.md` | 實驗的 operator guide 原文。 |
| `recipe.md` | 已發布的 recipe 說明與完整 input schema；沒有 recipe 就註明「沒有 recipe」。 |
| `figures/` | 操作員實際看得到的 liveplot 與 Analyze PNG。保留 liveplot 進度畫面與最後一格供比對，別用重新排版或重畫的圖替代。 |
| `analysis-summary.md` | Analyze 圖同一 run 的實際摘要，包含操作員當時可見的值。 |

優先取同一次 run 的 liveplot、Analyze 圖與摘要。記錄來源、run、擷取階段與條件供歸類時核對，別用這份擷取紀錄替產品補答案。若表單是 point 初值、liveplot 是低 reps probe、Analyze 圖又來自另一個保存 run，點數與訊號不同可能只是包組錯了。先核對來源；必要時重做同源讀者包再判 A 或 F。若產品沒有足夠的身分或設定讓操作員區分來源，仍記為 A。

若家族驗收指定同一次 FTF Run，liveplot 的進度與末格、Analyze 圖與摘要都取自該次 Run。
[離線 stage probe](../../tests/README.md#實驗定義) 可以獨立驗各階段，通過不等於完成 R0 讀者包。
要確認 GUI 如何建立 Analyze options 和發布摘要，見 [共用 lifecycle hooks](gui-contracts.md#共用-lifecycle-hooks)。

### 派讀者並收答卷

請一位懂實驗物理、沒看過該實作的 agent 只看讀者包。它不能讀程式碼（包含分析源碼）、包外文件或 raw data，也不能向作者確認答案。請它用自己的話交下列五節，無法確定時寫「不知道」並指出缺少什麼；不要把猜測寫成答案。

1. 量什麼：說出實驗目的、脈衝序列與主要量測量。
2. 每個參數：用表格逐欄列「畫面 label｜填什麼與效果｜把握度（高／中／低）｜欠缺的資訊」。有 recipe 時也說明 input 對應的表單欄位。
3. 每張圖：寫檔名、用途、每軸的量與單位、各資料系列或色階、標題與註記的意思，以及如何判斷好壞。逐張解讀 liveplot 和 Analyze 圖，再比較兩者。
4. 結果判讀：根據圖和摘要說明這次結果能支持什麼，接下來做什麼。若要調參，寫出方向、數值與依據；無法算就列出缺的值。
5. 答不出或可能答錯：逐項列檔名與欄位或圖元素、目前解讀、哪件事無法確定、補哪項可見資訊才能回答。

### 歸類與驗收

作者用已知實驗行為核對答卷。答錯或答不出的呈現問題依下表記錄；每項保留對應的答卷項目與圖或欄位。H 不算缺陷，但 guide 必須說明限制，讀者也要能說出這個限制。

| 類別 | 缺陷或限制 | 例子 |
| --- | --- | --- |
| A | 圖與摘要無法辨認 run 身分或判讀所需的採集設定 | 只見表單初值，不知該次 cfg 的 Reps×Rounds 或實際施加的 detune。見 R10。 |
| B | 缺逐點不確定度 | 機率點沒有用 shot SEM 畫 error bar，只報了 fit 參數的誤差。見 R11。 |
| C | 缺已回傳的 fit 品質值、指標解釋，或既有選取判準的說明 | 圖列了殘差 RMS，guide 沒說這個數值代表什麼。若候選未通過既有選取判準，圖沒指出未通過的指標與門檻也算缺陷。沒有既有判準不算缺陷。見 R3b、Gd3。 |
| D | 表單欄位的語意或單位不清 | `Frequency` 沒說是 RF 還是 IF；`Reset` 欄位與 Reset 按鈕混淆。 |
| E | 圖上的標題、圖例或軸量不清 | y 軸只寫 `Real Signal`，沒說投影與單位。見 R6、R7。 |
| F | liveplot 與 Analyze 圖不一致 | 同一訊號在兩處用了不同的軸名，讀者無法對照。見 R9。 |
| G | 判讀與下一步缺量化依據 | 只給 Ramsey 擬合頻率，讀者無法算出 `qubit.freq` 要往哪改、改多少。見 R1、R3、R10。 |
| H | 實驗的物理範圍限制 | `other` 不等於已辨識的 leakage 能階，joint return 也不等於 gate fidelity。 |

另外記錄覆蓋範圍和實驗設計的問題。例如還沒有 recipe，或掃描窗口本身太短。若 guide 沒說明短窗口如何限制判讀，仍有呈現缺陷。修完 A–G 的缺陷後，用更新的包重做檢查。當讀者能答對量測目的、參數、圖與判讀，且沒有未處置的 A–G 缺陷，該家族的代表實驗通過 R0。

## 圖規則

以下好例引用 Leon Ding 的 *Novel Gates with Superconducting Fluxonium Qubits*，MIT Physics PhD thesis，2023。頁碼用論文印的頁碼；「本地示例」是寫法示範，不代表目前每個實驗已做到。目前每個 axes 最多 4 條序列、最小字級 10 pt、每張圖最多 6 個 panel、fit 網格至少 200 點。這四項門檻之後可調。

### R1：一張圖支援一個判斷

在 guide 寫出每張圖支援的判斷。圖只傳達一個數字時直接報數值。兩張圖畫同一份資料時，刪一張或讓第二張多出可用的比較。

- 壞例：把同一條 CZ amplitude 掃描以不同標題畫兩次，guide 沒說第二張要看什麼。
- 好例：Ding 圖 5-9(d)（頁 152）用掃描與擬合的極小值選 CZ drive amplitude。guide 可寫「以標出的布居最低點選振幅」。

### R2：同時畫量測點與密網格的 fit

畫 fit 時一定畫量測點。模型在至少 200 點的密網格計算。量測點與模型各有 legend 項目。擬合曲線用實線，不加 marker。

- 壞例：只畫 fit，或用同一組 marker 畫模型與資料。
- 好例：Ding 圖 5-13(b)（頁 157）可直接對照 RB 的量測點與模型線。本地作圖時，分列各組的量測與模型 legend，並用至少 200 點算模型線，讓模型網格不受量測點疏密限制。
- 頻譜要先選峰再畫 fit 時，用 [fitting 的 Peak candidates](../../lib/zcu_tools/analysis/fitting/README.md#peak-candidates) 對照 `twotone/freq` 的局部策略與 `twotone/chain` 的預設策略。

### R3：結果寫在圖上

把圖要回答的數值以「名稱 = 值 ± 誤差 單位」寫在圖內一角。用線或 marker 標出選中的座標。

- 壞例：只畫 amplitude 掃描，不標出要採用的 gain，也不報估值誤差。
- 好例：Ding 圖 5-9(d)（頁 152）用垂直線標出所選振幅；圖 5-13(b)（頁 157）直接標估值與不確定度。本地示例：標出所選 gain 的垂直線，圖角寫 `π pulse gain = 0.230 ± 0.006 a.u.`。

### R3b：結果旁寫 fit 品質

在結果旁列出 core 或分析已回傳的品質值，例如 `FitQuality` 的 R²、正規化殘差 RMS 或參數相對誤差。繪圖端不重算品質值。若已有選取判準，沿用負責選取的程式給出的結論與未通過理由。只有既有判定未通過時，才標「未通過選取判準」並列出未通過的指標、數值與門檻。繪圖端不重做選取，也不自訂門檻。沒有既有判準時只列數值。guide 依 [Gd3](#gd3解釋誤差棒與-fit-品質) 解釋各指標與既有判準。

- 壞例：實驗沒有選取判準，繪圖端自行以 R² < 0.9 標成未通過。
- 好例（repo 現例）：[Reset Rabi check](twotone/reset/rabi_check/core.py) 的殘差子圖在 legend 使用[各分支擬合回傳的 RMS](twotone/reset/rabi_check/fit.py)，不重算 RMS，也不標通過與否。[Guide](twotone/reset/rabi_check/guide.md) 說明這些診斷共用同一個 IQ 投影軸，並請操作員比較擬合殘差。
- 好例（本地改寫示例）：[T2 Echo 的 GUI adapter](twotone/time_domain/t2echo/gui.py) 已決定寫回候選是否預先勾選，並提供未勾選的理由。若把這份既有判定呈現到圖上，可寫「未通過選取判準：R² = 0.774，須 ≥ 0.9；正規化殘差 RMS = 0.113，須 ≤ 0.1；T2 相對誤差 = 0.83，須 ≤ 0.5」。這是呈現寫法，並非現行圖上的標註。Core 和繪圖端不另套這些條件。

### R4：尺度跟著資料走

y 軸與色階預設依資料範圍自動縮放。跨圖比較才用固定尺度，例如用 0–1 比機率；此時在 colorbar 標出實際範圍。寬掃描上的局部 fit 另加 zoom panel。IQ 散點軸範圍取自資料，用透明度或分 panel 讓 g、e 兩雲都看得見。

- 壞例：四張要互相比較的 population 熱圖各自縮放，同一顏色分別代表 0.2 與 0.8。
- 好例：Ding 圖 5-7（頁 150）用 0–1 色階比較四個初態。單張不做這種比較時，預設讓色階跟著資料走。

### R5：一個 axes 最多 4 條要辨認的序列

超過就拆成小 panel。顏色在全 repo 保持固定意義，例如 gg、ge、eg、ee 各用固定的顏色。

- 壞例：在同一 axes 疊八條分支，再用同一個藍色代表不同的態。
- 好例：Ding 圖 5-9(e–f)（頁 152）把 CZ 頻率與單 qubit 相位分開，各自用顏色辨認少數序列。在本 repo，實驗間共用態別時也沿用各態的顏色。

### R6：用操作員的物理用語

標題寫「實驗 · 目標 · 量」。免責聲明放 guide，圖上不印 `True`、`None`、enum 名或內部座標名。每個軸寫量與單位；多 tone 的每個 panel 都寫 qubit 或 resonator 名。

- 壞例：標題 `terminal_gg · herald=True`，軸名 `df`。
- 好例：Ding 圖 5-8(b)（頁 151）寫出驅動頻率、脈衝平台（plateau）長度與色階所指的態。[本地 CZ Chevron](two_qubit/chevron/guide.md) 掃的是兩路共用的驅動頻率與波形包絡長度，不是論文的 plateau。改寫時，標題可用「CZ Chevron · Q1/Q2 · P(g,g)」，軸標「共用驅動頻率 (MHz)」和「共用波形包絡長度 (µs)」，色棒標「雙 Xπ 後的 P(g,g)」。

### R7：在固定存檔幾何檢查版面

逐張在 12×9 in、150 dpi 的 PNG（1800×1350 px）檢查版面。字級至少 10 pt，tick 標籤不能重疊，legend 不能蓋資料。一張圖最多 6 個 panel。Analysis 與 Post-analysis 開啟匯出的 PNG 檢查。Run Save 只保存資料；驗收 Run 圖時，以相同幾何輸出供檢查的 PNG。

Run、Analysis、Post-analysis、Workflow 和 Data 都以 12×9 in 排版 Figure，再把整張圖等比放入圖磚。不依圖磚形狀重排。圖與字一起縮放，多餘空間留白。GUI 只驗整張圖等比嵌入、留白及無裁切，不在各磚形重驗字級或版面。

Data 縮圖用來選圖。`View figure` 在 Data 頁把同一張圖等比放入較大的閱讀面，不寫檔。見 [Measure〈Figure presentation〉](../../lib/zcu_tools/gui/app/measure/README.md#figure-presentation)。

- 壞例：存檔 PNG 的 colorbar 擠到標題，或 legend 遮住量測資料。
- 好例：Ding 圖 5-7（頁 150）的四分支分成 2×2 panel。採用這種分組後，在固定幾何的 PNG 核對字級、tick、legend 與資料位置。

### R8：圖型對應資料，liveplot 標明進度

離散結果分布用長條圖，二維掃描用熱圖。liveplot 讓讀者看出哪些部分已取完、累積多少 shots、目前估計值。

Run 和 Analyze 的一維掃描用相同規則：沒有擬合曲線，就用折線連接量測點；有擬合曲線，量測資料只畫點和可用的誤差棒，只有擬合曲線用實線。類別軸（例如 gg、ge、eg、ee 或 AllXY gate 名稱）的量測點不連線。這條規則不適用於 IQ 散點和二維圖。

Wrapped phase 的折線只連掃描順序中相鄰的有效點。圖上角度相差大於 180° 時，線會跨過分支切口；恰為 180° 時，無法判定方向。兩種情況都斷開該線段，保留兩個點和既有誤差棒。缺值也要斷線，不補插值點。

- Rabi 的 live 圖用點、折線和 shot SEM。Analyze 加上擬合曲線後，量測資料只留點和 shot SEM。
- Local Z 的 live 圖用點和折線顯示 X/Y 投影隨 CZ gate 數的變化。Analyze 的相位圖已有線性擬合，因此量測相位只畫點。

曲線圖沿用[共用樣式](../../lib/zcu_tools/plotting/experiment.mplstyle)的 `axes.grid: True`，不另外呼叫 `grid()`。影像與熱圖在 liveplot 和 Analyze 都明確呼叫 `grid(False)`。

- 壞例：用折線連接 gg、ge、eg、ee，讓離散分類看起來像連續掃描，也沒顯示取數是否完成。
- 好例（本地示例）：joint readout 用四種準備態 × 四種結果的分組長條圖，顯示已取資料的分布，liveplot 標題寫累積 shots。二維掃描則可參考 Ding 圖 5-8(b)（頁 151）的頻率 × 長度熱圖。

### R9：liveplot 與 Analyze 盡量一致

能共用就沿用相同的量、座標軸、軸標、單位、圖型與顏色。相同的量測點用同樣的 marker，同一張熱圖用同樣的 colormap。Analyze 第一張圖以 liveplot 最後一格為底，疊上 fit 與結果標註。其他 Analyze 圖依 R1 說明各自支援的判斷。
GUI 依 adapter 的 [capability](gui-contracts.md#capability-宣告與實作) 決定是否建立 Analyze 控制項。

- 壞例：同次 Ramsey 的 liveplot y 軸寫 `Real Signal`，Analyze 先開殘差圖，再另以 `Signal (a.u.)` 畫原訊號。
- 好例（本地示例）：liveplot 與第一張 Analyze 圖都以自由演化時間 (µs) 為 x 軸、IQ 實部 (a.u.) 為 y 軸，量測點樣式相同。Analyze 疊上擬合與 T2*，另用診斷圖判斷殘差是否有系統偏差。

### R10：圖放理論重點，採集設定放摘要

圖上放物理量、模型、擬合結果與判讀所需的標註。不要把 run 身分或採集設定印在圖上。存檔路徑已有實驗名與 run id。除了定位 run 的實驗名與 run id，GUI 與 MCP 的可見 Analyze 摘要只由 adapter 補上該次 cfg 的 Reps×Rounds，以及實驗已回傳的實際施加值，例如 detune。不列完成輪數、是否提前停或 shot 數。liveplot 的進度仍按 R8 顯示。

Run 身分與採集摘要是呈現資料，不寫進原生結果的 JSON `summary`，也不擴充 `RunMetadata`。各實驗依需要決定是否在 Result 保存實際施加值。MCP 的 Analyze 回覆沿用 [RPC handler 提供的 `presentation`](../../lib/zcu_tools/gui/app/measure/remote/handlers/analysis.py)，獨立傳遞來源 run id。GUI 與 MCP 共用的摘要讀取契約見 [Analysis presentation](../../lib/zcu_tools/gui/app/measure/remote/README.md#analysis-presentation)。原生結果欄位的命名見 [S2](#s2report-的數值沿用-analyze-摘要名稱)。

- 壞例：Ramsey 圖標題塞了 run id 和初始設定，摘要卻只重複表單初值，沒寫實驗回傳的實際 detune。
- 好例（本地改寫示例）：圖上只保留判讀條紋和 T2* 需要的量與標註。同次 Analyze 摘要在實驗名與 run id 旁列「Reps×Rounds 1000×10；實際 detune 0.400 MHz」。Ding 圖 5-9(e)（頁 152）的相位圖集中在判讀所需的 180° 參考線與候選頻率。

### R11：有 shot SEM 的一維資料畫 error bar

已有 shot SEM 的一維系列畫逐點 error bar，包括用來判讀峰位的 spectrum。二維圖不畫逐點 error bar。Guide 寫明 error bar 的來源。

逐點 shot SEM 估計該點平均值的標準差，代表這個平均值的預期誤差。shot SEM = 每發樣本標準差 ÷ √(保留 shot 數)。被 mask 的 shot 不算。它不是單發散布 σ、輪間散布或 fit 參數誤差。RB 的跨 seed 標準差也不是 shot SEM。估計器在 [`zcu_tools.analysis`](../../lib/zcu_tools/analysis/README.md)。只存平均或舊檔缺少估計所需資料時不畫 shot SEM。不為了套用連線規則改存檔格式；需要二階 moment 時留到家族取數修正。

- 壞例：T1 只報 `T1 = 80 ± 5 µs`，不顯示每個量測點的不確定度。
- 好例：Rabi 的逐點布居用 shot SEM 畫 error bar。Ding 圖 5-13(a)（頁 157）的 CZ fidelity 對脈衝寬度圖可作逐點誤差棒的版式參考；圖 5-8(b)（頁 151）的二維掃描則用熱圖。
- 好例：多維取數的一維 fit panel 可以用外層掃描座標當橫軸，而不是擬合用的時間軸。
  [CPMG](twotone/time_domain/cpmg/core.py) 對每個 π 脈衝數沿演化時間擬合衰減，再以脈衝數為橫軸畫 T₂。誤差棒是各次衰減擬合的 T₂ 標準誤（`decay_time_err`）。
  [`t1_with_tone_sweep`](singleshot/t1/t1_with_tone_sweep/core.py) 對每個外層掃描點沿探測脈衝長度擬合轉移速率。速率圖的橫軸是外層掃描座標（增益或頻率，也可換算成光子數）。速率誤差棒是擬合共變異數矩陣中相應對角元素的平方根。
  guide 要寫明各 panel 的座標與誤差棒的來源，區分擬合參數的不確定度與逐點 shot SEM。

### R12：y 軸畫物理 observable

現有 Result 已有物理上有效的 population 時才畫它，例如既有分類結果的態機率。值域是 0–1，軸名或色階標籤要寫明是哪個態。顯示範圍照 R4 選，不因值域是 0–1 就固定尺度。只有平均訊號時，寫明 IQ 投影方式與單位 `a.u.`，並在 guide 說明為何它不是 population。博論只作版式與標註參考。不為了改圖加跑校準或改變取數。

- 壞例：用最小值與最大值把平均 IQ 縮到 0–1，照抄博論的態名並標為該態的 population。
- 好例：Ding 圖 5-8(b)（頁 151）示範怎麼在色階標出態名與量，圖中量的是 `|101⟩` 的 population。本地 CZ chevron 只有雙 Xπ 後的終端分類結果時，色階標「雙 Xπ 後的 P(g,g)」，不沿用論文的態名。只有 IQ 實部的平均值時，軸標「平均 IQ 實部 (a.u.)」，guide 說明平均投影不是各態的分類機率。

### R13：分支與變體寫在圖上

多分支、多初態或多序列變體的圖，在每個 panel 或 legend 寫出對應的初態、分支、躍遷或變體。物理分支名取自現有 Result 或 analyze `options`。在原有標籤上加分支名，不取代位置名或 adapter 提供的元件顯示名。缺的分支不為了排圖而增測。若脈衝序列示意有助於 R1 的判斷，可放在資料 panel 上方。

- 壞例：legend 只寫 `case 0`、`case 1`、`case 2`，或把沒有指派的分支套上博論的躍遷名，蓋掉元件顯示名。
- 好例：參考 Ding 圖 5-7（頁 150）在各 panel 標初態與躍遷的排法，或參考圖 5-9(b,e)（頁 152）把序列畫在資料上方、用 legend 區分脈衝數。本地 conditional phase 若只量了 `n = 1／2／3`，就沿用元件顯示名，另標實際的 `n = 1／2／3`，不照抄論文的 `n = 1／3／5`。

## 單位

下表是圖與表單等操作員可見文字的寫法，不是 cfg 欄位名、儲存單位或程式值的變更規則。

| 量 | 壞例 | 好例 |
| --- | --- | --- |
| 時間統一用 `µs` | `Delay (us)` | `自由演化時間 (µs)`。Ding 圖 4-10(d)（頁 125）以 µs 標自由演化時間。 |
| gain、amplitude 與倍率用 `a.u.` | `Gain (normalized)`、`Scale (dimensionless)` | `驅動增益 (a.u.)`、`CZ 增益倍率 (a.u.)`。 |
| 機率寫出機率名稱，不加 `a.u.` | `Probability`、`P (a.u.)` | `P(g,g)` 或「基態存活率」。Ding 圖 5-8(b)（頁 151）在 colorbar 寫出 101 態的布居。 |
| 計數用 `count`，寫出在數什麼 | `Shots (None)` 或只給數字 | `累積 shots = 8192 count`、`掃描點數 (count)`。 |

可見單位不顯示 `dimensionless`、`None` 或空字串。沒有單位的機率直接用具名機率，如 `P(g,g)`。

## 表單文字

### F1：label 用操作員的物理用語

用量或脈衝的名稱，不用 cfg key 或內部名稱。只改顯示 label，保留 cfg key 及 `qubit1`、`qubit2` 等位置名。用不同的文字區分物理欄位和 GUI 操作。

- 壞例：把 Ramsey 的欄位標成 `detune_ratio`，或讓重置模組欄位與表單重設按鈕都叫 `Reset`。
- 好例（repo 現例）：[Ramsey 表單](twotone/time_domain/t2ramsey/form.py) 把 `detune_ratio` 顯示為 `Detune ratio (fringes/step)`，寫出每掃描一步的相位變化量。
- 好例：重置模組欄位維持 `Reset`，cfg key 仍是 `reset`。Run 頁的 `Defaults` 按鈕重新讀取來源並重建表單初值。

### F2：數值欄位標單位

每個數值欄位都標單位，統一的顯示詞彙以[單位表](#單位)為準。計數與比例要說清楚數的是什麼、比的是什麼。

- 壞例：CZ amplitude 表單的 `Pre-delay` 沒標時間單位，`Scale` 沒說倍率乘在哪裡。
- 好例：寫「CZ 脈衝前等待 (µs)」及「兩路 CZ 驅動增益倍率 (a.u.)」，並說明倍率 1 代表保留兩路原本的增益。

### F3：description 用一句填寫說明

用一句短句寫填什麼、單位（如適用）與來源（如前置實驗），必填欄位用「填入 …」開頭。只在 label 看不出來時補寫效果，例如頻率是 RF、IF 還是相對量，選項開關做什麼，或留空代表什麼。實驗專屬的理由與步驟是否寫入 guide，依 [Gd1–Gd3](#operator-guide) 判斷。

- 壞例：`Frequency` 的說明只寫「頻率」；`Mixer freq` 的說明沒有交代留空的意思。
- 好例：Ramsey 的必填 `Pi/2 Pulse` 寫「填入經 Rabi 校準的 π/2 脈衝。」two-tone 的 RF 掃描起點寫「填入驅動 RF 頻率 (MHz)。」若 generator 不要求填寫數位 mixer 頻率，`Mixer freq` 可寫「指定數位 mixer 頻率 (MHz)，留空表示不設定數位 mixer。」

留空不會關閉外部 LO；需要數位 mixer 的 generator 必須填值。見 [Program 的 Readout configuration](../../lib/zcu_tools/program/v2/modules/README.md#readout-configuration)。

### F4：module 與 slot 寫出序列中的角色

在 description 說明 module 或 slot 代表哪個脈衝或讀出。表單能否顯示由 slot 解析出的參數，由框架另行評估。

- 壞例：joint readout 的 `Mux readout` 欄位只列 library 路徑，沒說何時讀出。
- 好例：「填入已校準的 mux 讀出模組，用於每個 shot 完成準備後的聯合讀出。」

## Operator guide

### Gd1：guide 內容檢查表

寫給懂物理但沒看過實作的操作員。Guide 只寫下列六項，章節可合併，不必為不適用的項目留空段：

1. **量什麼。** 一至兩句說明量測目的與主要 observable。
2. **序列。** 用文字示意一個週期的脈衝序列與讀出。
3. **物理意義。** 說明擬合參數代表什麼，以及結果能辨識什麼。
4. **關鍵參數。** 只說操作員必須自行決定的欄位怎麼設、為什麼。實驗專屬的必要校準、前置條件和 Run 前檢查留在 guide。
5. **怎麼讀圖。** 說明每張圖支援的判斷與誤差棒的意義，見 [R1](#r1一張圖支援一個判斷) 與 [Gd3](#gd3解釋誤差棒與-fit-品質)。
6. **下一步。** 說明結果怎麼用。有 writeback 時用一句說明它改哪個值；要依結果修正設定時寫出修正式與適用條件。

約 40 行是篇幅目標，不設硬性檢查。不要為了縮短而刪去必留項或讓 [R0 讀者包](#組讀者包) 缺少必要資訊。特例只有在操作員需要它來設定關鍵參數或判讀結果時才留，放入對應的項目。

### Gd2：先核對操作員能在哪裡看到同一資訊

刪除下列內容前，逐項確認負責的介面或文件已向操作員呈現等價資訊。程式碼、文件標題或連結存在都不算。沒有可見替代時，在 guide 的對應項目留最短的必要說明。

| 內容 | 到哪裡核對 |
| --- | --- |
| 存檔格式、舊檔行為 | 存檔或載入介面，或負責該格式與相容性的文件。只留操作員需要的保存或載入限制，不抄內部資料結構。 |
| Recipe 留空行為 | 逐輸入核對已發布的 `Field.description` 或 schema 是否說清留空（`null`）的效果，以及確切的失敗階段。若缺 recipe 專屬的失敗時點，留在 guide。 |
| 通用 GUI 操作 | 連到 [`zcu_lab/v2/operator-workflow.md`](operator-workflow.md)，不重抄點擊流程。Guide 的連結文字寫出這個 repo 路徑，讓 GUI 操作員能從 checkout 開啟；Guide pane 不解析相對 Markdown 連結。 |
| 一般 writeback 門檻 | 在結果或 writeback preview 核對是否看得到同一規則。不能用只在失敗時出現的 note 取代完整規則；GE 的例外見 Gd3。 |

表單顯示初值不等於說明初值來源。只有操作員要用此來源設定關鍵參數，且 GUI 沒有等價的來源說明時，才在「關鍵參數」保留最短公式，例如 T2 Echo 視窗初值 `0…1.5·t2e`。

表單或結果直接可見、且會影響操作員的固定行為仍寫在 guide。例如 reset check 的 Gain Stop 初值取 4 × x180 gain 與 1.0 的較小值。不把它擴寫成板子限制。硬體限制、QICK 報錯、排錯與可重用的判讀經驗放知識庫，不從 guide 連到未追蹤的知識檔。見 [ADR-0075〈判讀依據放知識庫〉](../../docs/adr/0075-agent-interface-layers.md#判讀依據放知識庫)。模擬器限制放 [FTF README](../simulation_envs/FTF/README.md)，驗收缺口放相應任務紀錄，純進度文字不放 guide。維護者需要的實作理由寫在負責該行為的程式註解。

### Gd3：解釋誤差棒與 fit 品質

依 [R11](#r11有-shot-sem-的一維資料畫-error-bar) 寫明誤差棒的來源。逐點 shot SEM 是單發樣本標準差除以保留 shot 數的平方根，代表該點平均值的抽樣誤差，不是 fit 誤差、單發散布或輪間散布。若圖用的是其他統計量，例如 RB 的跨 seed 標準差，按實際量說明。

依 [R3b](#r3b結果旁寫-fit-品質) 解釋已回傳的品質指標及既有判準如何影響下一步。門檻依 Gd2 分工，guide 不另訂門檻。[Mux GE](mux/singleshot_ge/guide.md) 與 [single-shot GE](singleshot/ge/guide.md) 各留一行完整的 Primary 預選規則：聯合擬合的 R² ≥ 0.9、正規化殘差 RMS ≤ 0.1、相對寬度誤差 ≤ 0.5、g/e 相對中心誤差各 ≤ 0.5；四項全過才預選，任一不過或必要指標缺失、非有限或標為無效就不預選。只有 GUI 與 MCP writeback preview 都對成功和失敗顯示完整規則，才移除這一行。不在 guide 展開 note 格式或零方差等排錯細節。

逐一核對每個 family 適用的必留項，不以篇幅為由省略：

- T2 Ramsey／Echo 每個 fringe 至少取 10 點，掃描視窗約為預估 T2 的 3 倍，以看清條紋並量到大部分衰減。視窗太短只看到部分衰減，擬合得靠外推；取樣太疏會辨不清條紋。
- 逐點誤差棒若是 shot SEM，按上文寫明其意義，不寫成 fit 誤差或輪間散布。
- 校準 readout `post_delay` 從 `5/(2π·κ)` 開始；κ 為以 MHz 表示的線寬時，等待以 µs 計。Herald 後打名目 Xπ，和更長的等待比較。
- 保留表單或結果直接可見且影響操作員的固定行為，依 Gd2 核對。

## Recipe 說明

### S1：input 指明對應的表單欄位

對應表單量測參數的 input，description 用與表單相同的 label 與單位，並寫出它設定哪個欄位。若同名欄位出現在多處，連同上層 label 一起寫。控制欄位如 `target`、`reuse_tab_id` 與 `overrides` 說明自己的作用，見[Recipe 撰寫](measure-authoring.md#recipe-撰寫)。

- 壞例：Ramsey 的 `max_delay_us` 只寫「掃描上限」，沒有指出是時間掃描或對應哪一欄。
- 好例：若表單的 `Delay › Stop` 標 µs，description 就寫「設定 Delay › Stop (µs)，即自由演化時間掃描的終點。」若 recipe 的 `points` 設的是 `Delay › Expts`，就寫「設定 Delay › Expts (count)，即延遲掃描的點數。」

### S2：report 的數值沿用 Analyze 摘要名稱

Report 的數值帶名稱與單位。只有語意相同的量才和[可見的 Analyze 摘要](#r10圖放理論重點採集設定放摘要)共用顯示名稱與單位。原生 JSON `summary` 沿用 core 結果的 key。不要為顯示名稱改 schema、加重複欄位或另造 summary model。

- 壞例：Ramsey report 把擬合條紋頻率寫成 `detune = 0.402`，讀者可能誤認成 Analyze 摘要裡實際施加的 detune。
- 好例：report 與 Analyze 圖都寫「擬合條紋頻率 = 0.402 MHz」。原生 JSON `summary` 的 key 沿用 core 的 `detune`，不另造 `fringe_freq`。report 若也列施加值，沿用 Analyze 摘要的「實際施加的 detune」與單位 MHz。
