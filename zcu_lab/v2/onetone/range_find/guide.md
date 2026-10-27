# One-tone range find

**讀者：** operator、agent。從 GUI 或 MCP 大範圍掃 resonator 頻率，審閱凹陷與頻率寫回候選。

`onetone/range_find` 以普通 pulse readout 的 generator 與 ADC 硬體掃頻，一次取得整條絕對 MHz 曲線。單一 Analyze 入口開啟互動 picker，靜態找谷結果只作初始猜測。確認選點與元件對應後按 Done，才產生分析結果與各 resonator 的 `freq` 候選。它不建立 readout module，也不擬合 resonator linewidth 或 electrical delay。找谷的 FWHM 只作候選篩選。

修改 parameter source 後，按 Reset 才會重新取得初值。Refresh 只更新 source／choice，並保留已填輸入；Reset 可能覆蓋尚未儲存的輸入。

## 準備與輸入

1. 先完成 lookback，確認 ADC window 與 trigger 能接到 readout。確認 generator、mixer、ADC 通道與功率已獲本次量測授權。能開表單不表示硬體設定安全。
2. 設定 alias `resonator`。Open／Reset 與 `onetone/freq` linear 使用相同的 readout Template、gain 與 timing 初值。`time_of_fly` 必須可讀。未設 kappa 時只用 time-of-flight；已設 kappa 時再加五個 ringup time。Readout 與 reset 的欄位規則沿用普通 one-tone，不使用 mux generator 或 tone 序號。
3. 從 New Tab 選 `onetone/range_find`。檢查 generator `gen_ch`、ADC `ro_ch`、mixer、gain、pulse 長度、ADC window 與 trigger。Open／Reset 產生可編輯的 readout 副本，不引用 `$mux` 的 readout 接線。
4. 展開 Sweep → Frequency。Open／Reset 後 Start／Stop 空白且必填，Expts 初值為 301。填入絕對 MHz 範圍涵蓋所有預期凹陷，不是 mixer offset。用足夠點數讓每個凹陷有數個取樣點。這裡只有 linear sampling，不使用 homophasal 校準參數。
5. Acquisition 表單沒有 resonator 名單，Run 不需要預先命名。設定 Reps、Rounds 與 Relax delay，必要欄位有效後才按 Run。來源修改後須按 Reset 才重新建立初值。Reset 會丟棄目前 draft。

## FTF 成本例

離線 FTF 驗收以 7050 至 7220 MHz、501 點、Reps 2、Rounds 1、Relax delay 1 us 跑一次 Run。Pulse readout 使用 generator 0、ADC 0、gain 0.05、RO window 0.99 us、trigger 0.798 us。Run wall time 為 0.856 s，CPU slot 為 4–5，BLAS 單執行緒。此例使用 repo FTF preset 與模擬資料，不代表硬體成本或物理校準已通過。

一個點是一個絕對頻率，不是 resonator 數量乘以點數。計時從點擊 Run 到 Run 按鈕恢復可用，包含設定、編譯、acquisition 與宿主提交，不含 Analyze 或保存。FTF 的編譯與快取會影響時間，成本不保證與 reps 成正比。先用目標設定做短 Run 估成本，再增加點數。不同 CPU slot、設定或快取狀態的時間不能直接比較。

## Run 與停止

Live 圖只有一條 magnitude 曲線，橫軸是硬體量化後的絕對 MHz。先確認範圍涵蓋預期數量的凹陷。

Stop 沿用 `onetone/freq` linear 的硬體掃頻語意，保留已完成 acquisition 的資料，不逐點重新編譯。Analyze 拒絕部分或非有限資料。若未取得完整曲線，重新 Run 後再 Analyze。部分資料不會把 acquisition failure 改成成功。

## Analyze 與判讀

在 Analysis 檢查 Resonators (JSON)、Min separation、Smoothing sigma、Prominence noise factor 與 Max FWHM。這些值只設定 picker 的初始狀態。Run／Load 的名單初值都是 `[]`，不從 cfg、Form 或 alias 取名稱。保留 `[]` 按 Analyze 可做無名探索。若要對應元件，先填入可選名稱，例如 `["R2", "R1"]`。名稱必須是非空白且不重複的字串，無效 JSON 或名稱會拒絕 Analyze。

Picker 先把最顯著的 N 個候選依頻率升冪與名單配對，N 是名單長度。空名單則選取全部符合篩選的候選，沒有元件標籤可選。初始猜測不代表元件身分已確認。

開啟 picker 後可直接操作：

- 修改 Smoothing sigma、Prominence noise factor、Max FWHM、Min separation 或 Resonator count，按 Enter 或離開欄位即重算。Resonator count 為 0 時選全部候選。重算會取代手動選點，只保留仍在相同頻率的元件對應。數量只限制自動選點，手動可增選。
- 點選星號可刪除該谷，點圖中其他位置可新增最接近的量測頻率點。也可用 Remove selected dip 刪除目前選點。
- 在 Selected dip 選一個頻率，再用 Component 指定名單中的元件或 Unassigned。對應不受頻率順序限制。把同一元件移到另一點時，舊點會變成 Unassigned。
- 按 Done 接受目前 controls、選點與對應。至少須有一個選點，無效欄位或空選點會保留 picker 並顯示錯誤。Cancel 不取代前次成功結果。

若需增加可選元件名稱，先 Cancel，再改 Analysis 的 JSON 名單重新 Analyze。Done 後 summary 按頻率升冪列出最終 name／freq／depth／fwhm／score。手動點若不是目前的自動候選，depth／fwhm／score 為 null，不捏造谷深或寬度。

Done 以最終對應的名稱逐項讀取固定 Run／Load capture 的 `<name>.freq`。全部值都是有限數值且按最終選點順序嚴格升冪時，才有頻率排序依據。缺 capture、缺值、null、頻率相同或非升冪時，summary 的 `identity_diagnostic` 顯示「dip 與元件身分配對待確認」。此標示不重排最終對應，也不封鎖寫回。非 null 的 freq 若不是有限數值，Done 會報錯，bool 也不接受。不會以目前 alias 或新校準補歷史依據。即使有排序依據，也須核對身分，排序不等於配對已確認。

`min_separation` 是兩個選中凹陷間的最小 MHz 距離，必須為有限正數。預設 1 MHz。寬凹陷附近有多個細小結構時可提高；兩個 resonator 靠近時需減小並增加取樣點數。

Analyze 用兩端各 10% 點的 magnitude 中位數連成直線背景。`smoothing_sigma` 預設 0，不平滑，保留一至兩點寬的窄谷深度。大於 0 時使用 Gaussian sigma，單位是取樣點。`prominence_factor` 預設 5，候選 prominence 須超過此倍率乘以相鄰差值的穩健雜訊估計，也須超過原始 magnitude 動態範圍的百萬分之一。Sigma 與倍率必須是有限非負數。

每個候選的 FWHM 是半 prominence 位置的插值寬度，至少為谷兩側頻率步進的平均。`max_fwhm` 預設 25 MHz，必須是有限正數。寬度超過上限的候選先剔除；過耦合共振較寬時可調大。其餘候選依 `score = depth / fwhm` 由大到小選取，再排除距離太近的候選。有名稱時選到名稱數量為止。`[]` 則保留所有符合寬度與間距的候選。掃描邊緣的最低點不算凹陷。曲線至少需要 11 個不同且有限的頻率點。

Picker 顯示量測 magnitude 對絕對 MHz 的曲線，灰色圓點是自動候選，紅色星號是目前選點。Legend 顯示頻率順序編號、名稱或 unassigned 與 MHz 頻率。Done 後的圖保留同一組最終選點與對應。編號只用於看圖，不是元件名稱。Summary 的 depth 是去背景曲線經選定平滑後的 prominence，單位 a.u.。FWHM 使用 MHz，score 使用 a.u./MHz。這些值不是擬合 linewidth 或擬合誤差。

自動找到的谷少於名稱數量時，picker 仍可調參、增選或取消，不會自動完成。確認所有預期元件都已對應。非線性背景、邊緣凹陷或低訊噪比可能影響選點，不要替看不見的凹陷配名。

## 審閱與 Apply

只有 Done 後已指派元件的選點才提案寫回。每個 capture 中存在的已指派 resonator 各有一列 `<name>.freq` MHz proposal。Unassigned 不提案。沒有 Run／Load capture 時全部省略，名稱不在 capture 時省略該名稱的列，summary 說明原因。不讀目前 alias 猜目的地。

沒有頻率排序依據時，每列 writeback 候選的 description 也顯示「dip 與元件身分配對待確認」。這個提示不封鎖 Apply。先核對 dip 與元件身分，再決定接受哪些列。需要改對應時，重新 Analyze，在 picker 中調整 Component 並按 Done，候選才會更新。

對照圖上的名稱與位置，逐列檢查目的地和頻率。取消不接受的列，再按 Apply Selected。這裡不提出 module、kappa 或 edelay。確認粗頻率後，另開或 Reset `onetone/freq` 做普通 channel 窄掃。需要 mux readout 校準時，另開或 Reset `mux/onetone` 並核對 mux 接線，再執行窄掃。

## 保存與讀回

Save All 的 native 資料保存本次 acquisition cfg、量化頻率與 complex IQ。Cfg 不含 resonator 名稱，命名只屬於分析。頻率在記憶體是 MHz，在檔案是 Hz。Load Data 還原該次 cfg 與原始資料，不從目前 Form 或 library 補歷史設定，也不覆寫目前 draft。

分析 JSON 的 `summary.selection.options` 保存最終 controls，`summary.selection.selected` 保存最終頻率、component 對應與可用的 estimate。`summary.selection.candidates` 保留同次候選 metrics。文件的 `parameters` 仍是初始 Analyze 表單，不代表最終 controls。只有 Done 後的已提交結果會保存，未完成的 picker 草稿不算分析成果。

Labber 匯出包含絕對頻率與 complex IQ，不是 native Load 的替代來源。同 entry 且具有歷史 point 的 native Load 使用歷史 capture；沒有可用 capture 時仍可純分析，但不提議寫回。

## MCP recipe

呼叫 `onetone_range_find`，必填 `freq:{start,stop}` 與 `n`，端點使用絕對 MHz。
Recipe 不猜頻率初值。`target` 指定這次 Reset 的 resonator alias，`reuse_tab_id` 可重用 idle matching tab。
`resonators` 經 Analyze updates 傳入 picker，不寫入 acquisition cfg 或量測表單。省略保留分析名單，`[]` 明確做無名探索，null 原樣交分析 owner 驗證。非空名單提供可選元件標籤與初始配對，最終順序可在 picker 中調整。
`reps`、`rounds` 省略保留表單值，null 清空欄位。Run 前以 check_ready 驗證完整表單。
`min_separation` 指定選中凹陷之間的最小 MHz 距離。省略保留分析值，初值為 1 MHz。明填數字必須是有限正數，null 原樣交分析 owner。
例如 `min_separation: 5.0` 排除間距不足 5 MHz 的候選，不改 acquisition 的頻率點。
Recipe 在同一個 Analyze operation 傳入有提供的名單與間距，省略不覆寫分析值。非正數或非有限間距在開 tab 前拒絕。初始使用同一個找谷函式，不平滑、prominence 倍率 5、FWHM 上限 25 MHz，以 depth / FWHM 排序。初始猜測不足或為空時，仍等待互動修正，不自動送出 done。

操作順序如下：

1. Recipe 完成 Run 後呼叫 `tab.analyze("primary", interactive=True, updates=...)`，execution 回報 pending analysis。
2. Agent 以 `tab_interact` 查詢目前 state 與 command schema。核對初始猜測，必要時送 `set_parameters`、`add_dip`、`remove_dip` 或 `assign_dip`。`set_parameters` 須帶五個 controls，`max_dips: null` 表示全部。選點 index 為目前頻率升序的零起始位置，刪除或新增後須重新查詢。`assign_dip` 的 component 為提供的名稱或 null。
3. Agent 確認後送 `tab_interact` 的 `done`。直接接受初始猜測也必須明送 done。取消使用既有 cancel 操作，不能以 writeback 回答代替 done。
4. Done 成功後 recipe 才呼叫 `ask_writeback`。檢查最終 proposal 並回答選定項目或略過。空候選也不會寫入，不能靠初始 `resonators` 判斷最終寫回內容。
5. Recipe 只接受回答選定的候選，最後保存 raw data 與分析。Report 回覆最終各 dip 的可選 name、freq、depth、fwhm、score，以及 `identity_diagnostic`、`writeback_diagnostic`。分析失敗不詢問寫回，仍保存已取得的 raw data。

## Four starting situations

1. With an empty kickstart project, Open the Form and inspect its required blanks.
   Enter absolute frequency bounds and check readout timing before the coarse survey.
   Analyze with `[]` for discovery, or supply component labels before Analyze.
   Inspect and adjust the picker, then press Done. Cancel first if the label list must change.
   Nominal wiring and fallback values are candidates, not measured calibration.
2. With frequencies but no module slots, Reset to recapture those values.
   Fill the remaining pulse, timing and calibration errors before Run. Refresh keeps existing inputs.
3. With a readout slot, Reset and review the captured module and remaining errors separately.
   A slot does not prove that drive pulses, sweep bounds or GE calibration are valid.
4. Through the agent, call `onetone_range_find` after inspecting the same Form errors.
   freq with absolute MHz start/stop and n are required. n maps to frequency Expts.
   resonators supplies picker labels; omission retains names, [] discovers unnamed dips, and null reaches the owner.
   min_separation sets initial minimum MHz spacing. Inspect through tab_interact and send done before answering writeback.
   The Form chooses module sources. Omission retains values; null is passed to the Form or analysis owner. Overrides apply last, then check_ready validates before Run.
   Missing inputs return needs_parameters before acquisition. Reset may replace unsaved inputs.
