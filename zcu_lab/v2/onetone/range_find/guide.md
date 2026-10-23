# One-tone range find

**讀者：** operator、agent。從 GUI 或 MCP 大範圍掃 resonator 頻率，審閱凹陷與頻率寫回候選。

`onetone/range_find` 以普通 pulse readout 的 generator 與 ADC 硬體掃頻，一次取得整條絕對 MHz 曲線。它找出多個 resonator 的粗略頻率，只提議各 resonator 的 `freq`。它不建立 readout module，也不估計 linewidth 或 electrical delay。

## 準備與輸入

1. 先完成 lookback，確認 ADC window 與 trigger 能接到 readout。確認 generator、mixer、ADC 通道與功率已獲本次量測授權。能開表單不表示硬體設定安全。
2. 設定 alias `resonator`。Open／Reset 與 `onetone/freq` linear 使用相同的 readout Template、gain 與 timing 初值。`time_of_fly` 必須可讀。未設 kappa 時只用 time-of-flight；已設 kappa 時再加五個 ringup time。Readout 與 reset 的欄位規則沿用普通 one-tone，不使用 mux generator 或 tone 序號。
3. 從 New Tab 選 `onetone/range_find`。檢查 generator `gen_ch`、ADC `ro_ch`、mixer、gain、pulse 長度、ADC window 與 trigger。Open／Reset 產生可編輯的 readout 副本，不引用 `$mux` 的 readout 接線。
4. 展開 Sweep → Freq。Open／Reset 後 Start／Stop 空白且必填，Expts 初值為 301。填入絕對 MHz 範圍涵蓋所有預期凹陷，不是 mixer offset。用足夠點數讓每個凹陷有數個取樣點。這裡只有 linear sampling，不使用 homophasal 校準參數。
5. 填寫由低頻到高頻對應的 resonator 名稱。名稱與選中的 dip 依頻率升冪配對。Open／Reset 有 `$mux` 時讀 group resonators。全部成員都有目前 freq 才依 freq 升冪排序，否則保留 group 順序。沒有 `$mux` 則從空列表開始。這個可編輯的列表隨本次 cfg 保存，不影響 acquisition。Run 可留空，Analyze 前須核對名稱與頻率順序。圖形本身不能證明 component 身分。
6. 設定 Reps、Rounds 與 Relax delay，必要欄位有效後才按 Run。來源修改後須按 Reset 才重新建立初值。Reset 會丟棄目前 draft。

## FTF 成本例

離線 FTF 驗收以 7050 至 7220 MHz、501 點、Reps 2、Rounds 1、Relax delay 1 us 跑一次 Run。Pulse readout 使用 generator 0、ADC 0、gain 0.05、RO window 0.99 us、trigger 0.798 us。Run wall time 為 0.856 s，CPU slot 為 4–5，BLAS 單執行緒。此例使用 repo FTF preset 與模擬資料，不代表硬體成本或物理校準已通過。

一個點是一個絕對頻率，不是 resonator 數量乘以點數。計時從點擊 Run 到 Run 按鈕恢復可用，包含設定、編譯、acquisition 與宿主提交，不含 Analyze 或保存。FTF 的編譯與快取會影響時間，成本不保證與 reps 成正比。先用目標設定做短 Run 估成本，再增加點數。不同 CPU slot、設定或快取狀態的時間不能直接比較。

## Run 與停止

Live 圖只有一條 magnitude 曲線，橫軸是硬體量化後的絕對 MHz。先確認範圍涵蓋預期數量的凹陷。

Stop 沿用 `onetone/freq` linear 的硬體掃頻語意，保留已完成 acquisition 的資料，不逐點重新編譯。Analyze 拒絕部分或非有限資料。若未取得完整曲線，重新 Run 後再 Analyze。部分資料不會把 acquisition failure 改成成功。

## Analyze 與判讀

在 Analysis 檢查 Resonators (JSON) 與 Min separation。Analyze 參數從該次 Run 的 `cfg.resonators` 取初值。Load 從檔案 cfg 取，不從目前 Form 或 alias 重建。沒有可還原的 cfg 時，分析名單留空供手填。可修改 JSON 名單，例如 `["R2", "R1"]`。名稱必須是非空、不重複的字串列表。無效輸入會拒絕 Analyze，不會自動換成空列表。

Analyze 以這次 JSON 名單逐項讀取固定 Run／Load capture 的 `<name>.freq`。全部值都是有限數值且按名單嚴格升冪時，才有頻率排序依據。缺 capture、缺值、null、頻率相同或非升冪時，summary 的 `identity_diagnostic` 顯示「dip 與元件身分配對待確認」。此標示不改名單順序，也不阻擋分析。非 null 的 freq 若不是有限數值，Analyze 會報錯，bool 也不接受。不會以目前 alias 或新校準補歷史依據。即使有排序依據，也須人工核對身分，排序不等於配對已確認。

`min_separation` 是兩個選中凹陷間的最小 MHz 距離，必須為有限正數。預設 1 MHz。寬凹陷附近有多個細小結構時可提高；兩個 resonator 靠近時需減小並增加取樣點數。

Analyze 用兩端各 10% 點的 magnitude 中位數連成直線背景，再以一個取樣點的 Gaussian sigma 平滑去背景曲線。候選 prominence 必須超過五倍相鄰差值的穩健雜訊估計，也須超過原始 magnitude 動態範圍的百萬分之一。分析依 prominence 由大到小選取，排除距離太近的候選，直到取得名稱數量。掃描邊緣的最低點不算凹陷。曲線至少需要 11 個不同且有限的頻率點。

分析只顯示一張量測 magnitude 對絕對 MHz 的曲線圖。每個找到的凹陷以紅色星號標在量測曲線上，旁邊標名稱。Legend 顯示名稱、MHz 頻率與 depth，summary 也逐列列出這三項。Depth 是去背景平滑曲線的 prominence，單位 a.u.，不是 linewidth 或擬合誤差。非線性背景、邊緣凹陷或低訊噪比可能影響選點，須看圖判讀。

找到的凹陷少於名稱數量時，整次 Analyze 失敗，訊息列出找到與需要的數量，不產生部分 proposal。先檢查掃描範圍、點數、功率與名稱數量，再重跑或調整分析參數。不要替看不見的凹陷配名。

## 審閱與 Apply

成功後，每個 capture 中存在的 resonator 各有一列 `<name>.freq` MHz proposal。沒有 Run／Load capture 時全部省略，名稱不在 capture 時省略該名稱的列，summary 說明原因。不讀目前 alias 猜目的地。

沒有頻率排序依據時，每列 writeback 候選的 description 也顯示「dip 與元件身分配對待確認」。這個提示不封鎖 Apply。先核對 dip 與元件身分，再決定接受哪些列。需要改名單順序時，在 Analysis 修改 JSON 並重新 Analyze，候選才會重新配對。

對照圖上的名稱與位置，逐列檢查目的地和頻率。取消不接受的列，再按 Apply Selected。這裡不提出 module、kappa 或 edelay。確認粗頻率後，另開或 Reset `onetone/freq` 做普通 channel 窄掃。需要 mux readout 校準時，另開或 Reset `mux/onetone` 並核對 mux 接線，再執行窄掃。

## 保存與讀回

Save All 的 native 資料保存本次 cfg、量化頻率與 complex IQ。頻率在記憶體是 MHz，在檔案是 Hz。Load Data 還原該次 cfg 與原始資料，不從目前 Form 或 library 補歷史設定，也不覆寫目前 draft。

Labber 匯出包含絕對頻率與 complex IQ，不是 native Load 的替代來源。同 entry 且具有歷史 point 的 native Load 使用歷史 capture；沒有可用 capture 時仍可純分析，但不提議寫回。

## MCP recipe

呼叫 `onetone_range_find`，必填 `freq:{start,stop}` 與 `n`，端點使用絕對 MHz。
Recipe 不猜頻率初值。`target` 指定這次 Reset 的 resonator alias，`reuse_tab_id` 可重用 idle matching tab。
`resonators` 可提供由低頻到高頻的名稱，省略或 null 保留表單初值。
`reps`、`rounds` 省略或 null 保留表單值。Run 前驗證完整欄位。
`min_separation` 指定選中凹陷之間的最小 MHz 距離，預設 1 MHz，必須是有限正數。
例如 `min_separation: 5.0` 排除間距不足 5 MHz 的候選，不改 acquisition 的頻率點。
Recipe 在同一個 Analyze operation 傳入此值，重用 tab 時也不沿用先前的 GUI 值。
非正數或非有限值在開 tab 前拒絕。間距過大而找不到指定數量的凹陷時，分析失敗且不詢問寫回。
Run、Analyze 後回覆每個 dip 的 name、freq、depth，以及 `identity_diagnostic`、`writeback_diagnostic`。
Recipe 詢問 writeback，只接受選定候選，略過不寫入，最後保存 raw data 與分析。
