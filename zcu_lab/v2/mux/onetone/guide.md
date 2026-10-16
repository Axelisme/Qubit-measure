# Mux onetone

掃共同的名目頻率偏移，同時取得選定 tones 的複數訊號。每個 sweep 點的頻率是原始 tone 的 base frequency 加上 offset，不累加。圖上不是 DDS 實際量化頻率。

## 準備輸入

1. 在 Inspect 的 Parameters 建立 `kind: group/mux` 元件，例如 `MG12`。設定有序 `members: [R1, R2, R3]`，並把 `module.readout` 指到已有的 `readout/mux` library key。完整 tones[i] 對應 members[i]，label 不決定綁定。
2. 在 Aliases 設定 scalar alias `mux = MG12`。開啟 `mux/onetone` 時，Readout 引用 `$mux.module.readout` 的條目。
3. 核對 generator、mixer、每個 tone 的 ADC channel、frequency、gain，以及共用播放長度、ADC window 和 trigger timing。範本的 scalar 初值只供編輯，不代表硬體安全設定。
4. 可選已有的 readout/mux key，或用 Convert to inline 複製內容再修改。Tones 是單行 JSON，例如 `[{"freq":5980,"gain":0.1,"ro_ch":4,"label":"R1"},{"freq":6020,"gain":0.1,"ro_ch":5,"label":"R2"},{"freq":6040,"gain":0.1,"ro_ch":6}]`。這是格式例子，不是建議的硬體輸入。
5. Active tones 選 None 量全部，或輸入原始序號 JSON list，例如 `[0,2]` 只量第一及第三個 tone。`[2,0]` 也有效，program、曲線與資料 columns 都保持此順序。不接受空 list、重複、越界、bool 或非整數 index。宿主的 result／Run 診斷會指出 `active_tones` 的錯誤並拒絕 Run，不保證欄位本身變紅。
6. 設定 Frequency offset 的 Start、Stop、Expts，以及 Reps、Rounds、Relax delay。執行前確認所有欄位有效。

沒有 group 或 alias 時，必要 readout 欄位保持未填或 Invalid。仍可手選或填寫有效 Readout 後 Run，不要求綁定 group。

## Run 與圖形

Run 使用這個 tab 在 Open 或 Reset 時捕捉的來源。其後修改 aliases、group members 或 library 不會改動目前 draft。需要新來源時明確 Reset，這會重建初值。Run 不把 subset 覆蓋到完整 readout。

Live 圖以共同 offset 為橫軸，顯示 K 條 active magnitude 曲線。Legend 使用對應原始 tone label，空 label 顯示原始序號，例如 `tone 2`。Label 可以重複。資料軸仍是原始 tone 序號，與 component、label 或 ADC channel 無關。停止後保留已取得的資料；未取得的點保持 NaN。失敗狀態不因有部分資料而變成成功。

## 保存與讀回

保存使用當次 RunRecord 的完整 cfg 與原始資料，包含全部 base tones、labels、active_tones 與 sweep。不保存 members 或 group snapshot。Native 讀回沿既有 cfg restore，不從目前表單補來源。Load 不覆寫目前 draft，也不改其固定來源。

這一版只有 Run、live 圖與資料存取，沒有 FIT 或 writeback。既有 Labber 匯出保留共同 offset、原始 tone 序號及複數訊號，columns 與 tone_indices 同序。Run／Load 的來源由宿主捕捉。跨 entry 或沒有 point 的 Load 不會建立分析參數來源。檔案 cfg 缺失或無法恢復時，原始資料仍可讀取，但不能以目前表單代替歷史 cfg。
