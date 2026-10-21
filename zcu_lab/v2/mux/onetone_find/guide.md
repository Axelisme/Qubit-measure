# Mux onetone find

用一個 mux tone 掃過指定的絕對頻率範圍，找出多個 resonator 的粗略頻率。每點只播放選定 tone，並使用該 tone 的 ADC。其他 tones 不播放。這一步只提議各 resonator 的 `freq`，不建立 readout module，也不估計 linewidth 或 electrical delay。

## 準備與輸入

1. 先完成 lookback，確認 ADC window 與 trigger 能接到 readout。確認 generator、mixer、ADC 通道及功率已獲本次量測授權。能開表單不代表硬體設定安全。
2. 在 Inspect 的 Parameters 建立 `group/mux`，填入 resonator 名稱列表，並設定 alias `mux` 指向這個 group。有 readout library 時，把 `$mux.module.readout` 指到該 `readout/mux` key。Resonator 的目前 `freq` 用來產生 tone 初值及分析名稱的排序，不是本次要找的答案。
3. 從 New Tab 選 `mux/onetone_find`。Open／Reset 複製 group readout 的 scalar 設定，再由 group resonators 產生完整 tones。沒有 readout slot 時使用可編輯的 blank 範本。初值沿用 fine scan 的 ringup 等待。檢查 readout pulse 長度、ADC window、trigger、gain、generator 與各 ADC channel。
4. Tone 是完整 tones 列表的原始序號，從 0 起算。選定後，該 tone 提供 gain、phase 與 ADC；掃描只取代它的 frequency。不接受越界或非整數序號。
5. 在 Frequency 填 Start、Stop 與 Expts。Start／Stop 是 MHz 絕對頻率，不是 mixer offset。先用範圍涵蓋預期的所有凹陷，再以點數讓每個凹陷有數個取樣點。頻率不可超出所選 ADC 量化 mixer 周圍的 generator mux 頻帶，Run 前檢查會指出允許的 MHz 範圍。
6. 設定 Reps、Rounds 與 Relax delay。確認所有必要欄位有效後才按 Run。來源修改後要按 Reset 才重新捕捉初值，Reset 會重建目前 draft。

沒有 mux 綁定時，可手填完整 readout 後量測。分析時必須手填 resonator 名稱。沒有 Run／Load capture 的結果仍可找凹陷，但不能提議寫回。

## Run 與停止

Live 圖只有一條 magnitude 曲線，橫軸是每點量化後的絕對 MHz。先檢查是否有預期數量的凹陷，以及頻率範圍是否涵蓋它們。

Stop 保留已取得的點，尚未取得的 IQ 為 NaN。Analyze 拒絕部分或非有限資料，避免把尚未掃到的 resonator 當成不存在。重新取得完整掃描後再 Analyze。取得部分資料不會把 acquisition failure 改成成功。

## Analyze 與判讀

在 Analysis 展開 Analysis parameters。Resonators (JSON) 欄位使用 JSON list 文字，例如 `["R2", "R1"]` 代表低頻凹陷叫 R2，高頻凹陷叫 R1。預設名稱來自該次 Run／Load capture 的 `$mux.resonators`，依其當時的 `freq` 排序。檢查這個順序是否符合接線；圖形本身不能證明 component 身分。名稱必須加雙引號，不能只填 `R2, R1`。無效 JSON、非 list、空值、重複或非字串名稱會拒絕 Analyze 並顯示原因，不會自動改成空 list。

`min_separation` 是兩個選中凹陷間的最小 MHz 距離，必須大於 0。預設 1 MHz，對應 MHz 級 readout linewidth。寬凹陷附近有多個細小結構時可提高；兩個 resonator 靠近時需減小並增加取樣點數。

按 Analyze 後，分析用兩端各 10% 點的 magnitude 中位數連成直線背景，再以一個取樣點的 Gaussian sigma 平滑背景移除後的曲線。候選凹陷的 prominence 必須超過五倍相鄰差值的穩健雜訊估計，並超過原始 magnitude 動態範圍的百萬分之一。分析依 prominence 由大到小選取，排除距離太近的候選，直到取得名稱數量。掃描邊緣的最低點不算凹陷。每條曲線至少需要 11 個不同且有限的頻率點。

上圖顯示原始 magnitude 與直線背景，下圖顯示去背景的曲線。每個位置的 legend 顯示名稱、MHz 頻率與 depth，summary 也逐列列出這三項。Depth 是去背景平滑曲線的 prominence，單位 a.u.，不是 linewidth 或擬合誤差。非線性背景、邊緣凹陷或低訊噪比可能影響選點，須看圖判讀。

找到的凹陷少於名稱數量時，整次 Analyze 失敗，訊息列出找到與需要的數量，不產生部分 proposal。先檢查掃描範圍、點數、功率與名稱數量，再重跑或調整 Options。不要為了取得 proposal 而替看不見的凹陷配名。

## 審閱與 Apply

成功後，每個 capture 中存在的 resonator 各有一列 `<name>.freq` MHz proposal。沒有 capture 時全部省略，名稱不在 capture 時省略該名稱的列，summary 說明原因。不讀目前 alias 來猜目的地。

對照圖上的名稱與位置，逐列檢查目的地和頻率。取消不接受的列，再按 Apply Selected，只套用勾選的列。這裡不提出 module、kappa 或 edelay 更新。需要精細頻率與 readout 校準時，Apply 已確認的粗頻率後，另開或 Reset `mux/onetone_fine`，讓它讀取更新後的來源，再執行窄掃。後續 `mux/twotone_spectrum` 仍需已確認的 qubit drive 與 readout 設定。

## 保存與讀回

Save All 的 native 資料保存完整 cfg、原始 tone 序號、全部 readout tones、量化頻率與 complex IQ。頻率在記憶體是 MHz，在檔案是 Hz。Load Data 還原該次 cfg 與原始資料，不從目前 Form 或 library 補歷史設定，也不覆寫目前 draft。

Labber 匯出沿既有單曲線政策，包含絕對頻率與 complex IQ。Labber 不是 native Load 的替代來源。同 entry 且具有歷史 point 的 native Load 使用歷史 capture；沒有可用 capture 時仍可純分析，但不提議寫回。
