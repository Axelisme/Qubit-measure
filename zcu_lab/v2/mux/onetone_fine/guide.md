# Mux onetone fine

掃共同的名目頻率偏移，同時取得選定 tones 的複數訊號。每個 sweep 點的頻率是原始 tone 的 base frequency 加上 offset，不累加。圖上不是 DDS 實際量化頻率。

## 準備輸入

1. 在 Inspect 的 Parameters 建立 `kind: group/mux` 元件，例如 `MG12`。設定有序 `resonators: [R1, R2, R3]`，並把 `module.readout` 指到已有的 `readout/mux` library key。完整 tones[i] 對應 resonators[i]，label 不決定綁定。
2. 在 Aliases 設定 scalar alias `mux = MG12`。Open／Reset 時，Readout 複製 `$mux.module.readout` 的 scalar 設定，tones 由 group resonators 校準值產生。缺 slot 時沿 blank mux 範本加 group tones。共同 trigger 增加最小 kappa 對應的五倍 ringup 等待，pulse 至少涵蓋 ADC window。
3. 核對 generator、mixer、每個 tone 的 ADC channel、frequency、gain，以及共用播放長度、ADC window 和 trigger timing。範本的 scalar 初值只供編輯，不代表硬體安全設定。
4. 可選已有的 readout/mux key，或用 Convert to inline 複製內容再修改。Tones 是單行 JSON，例如 `[{"freq":5980,"gain":0.1,"ro_ch":4,"label":"R1"},{"freq":6020,"gain":0.1,"ro_ch":5,"label":"R2"},{"freq":6040,"gain":0.1,"ro_ch":6}]`。這是格式例子，不是建議的硬體輸入。
5. Active tones 選 None 量全部，或輸入原始序號 JSON list，例如 `[0,2]` 只量第一及第三個 tone。`[2,0]` 也有效，program、曲線與資料 columns 都保持此順序。不接受空 list、重複、越界、bool 或非整數 index。宿主的 result／Run 診斷會指出 `active_tones` 的錯誤並拒絕 Run，不保證欄位本身變紅。
6. 設定 Frequency offset 的 Start、Stop、Expts，以及 Reps、Rounds、Relax delay。執行前確認所有欄位有效。

Group 可選填 `qubits: [Q1, Q2, Q3]`，不填或 null 適用於 resonator bring-up。提供時必須與完整 resonators 等長且沒有重複 qubit。Open／Reset 檢查每個 qubit 和 resonator 是否存在、kind 是否正確，以及 `Q1.resonator == R1` 等逐項對應。缺失或 null 也會拒絕。錯誤顯示欄位、index 與實際值，修正來源後再開啟或 Reset。不排序或按頻率猜配對，也不額外拒絕重複 resonator。

沒有 group 或 alias 時，必要 readout 欄位保持未填或 Invalid。仍可手選或填寫有效 Readout 後 Run，不要求綁定 group。

## Run 與圖形

Run 使用這個 tab 在 Open 或 Reset 時捕捉的來源。其後修改 aliases、group resonators 或 library 不會改動目前 draft。需要新來源時明確 Reset，這會重建初值。Run 不把 subset 覆蓋到完整 readout。

Live 圖以共同 offset 為橫軸，顯示 K 條 active magnitude 曲線。Legend 使用對應原始 tone label，空 label 顯示原始序號，例如 `tone 2`。Label 可以重複。資料軸仍是原始 tone 序號，與 component、label 或 ADC channel 無關。停止後保留已取得的資料；未取得的點保持 NaN。失敗狀態不因有部分資料而變成成功。

## 保存與讀回

保存使用當次 RunRecord 的完整 cfg 與原始資料，包含全部 base tones、labels、active_tones 與 sweep。不保存 resonators 或 group snapshot。Native 讀回沿既有 cfg restore，不從目前表單補來源。Load 不覆寫目前 draft，也不改其固定來源。

既有 Labber 匯出保留共同 offset、原始 tone 序號及複數訊號，columns 與 tone_indices 同序。Run／Load 的來源由宿主捕捉。跨 entry 或沒有 point 的 Load 不會建立分析參數來源。檔案 cfg 缺失或無法恢復時，原始資料仍可讀取，但 FIT 會拒絕，不以目前表單代替歷史 cfg。

## FIT

所有 active tones 共用 onetone/freq 的四欄設定。Model type 可選 hanger `hm`、transmission `t` 或 `auto`，預設 `hm`。Fit amplitude background 預設開啟，Fit phase curvature 預設關閉，兩者可獨立切換。

Electrical delay seed 單位為 us，預設留空。留空時，每個 tone 各自使用對應 resonator.ext.edelay 的有限 generator／ADC route 校準作 branch seed。沒有匹配時，該 tone 使用內部固定 100 us 上限作 global 搜尋。明填有限數字時，所有 active tones 共用此 branch seed，0 與負值也有效。Seed 只選擇擬合起點，不固定模型 delay。非有限或非數字 delay 輸入會拒絕整次 FIT。

FIT 只處理 active columns。每條曲線以保存的原始 base frequency 加共同 nominal dfs 作 MHz 座標，去掉首尾點。這不是 DDS 實際頻率。Cfg、selection、tone_indices 或 shape 不符時整次拒絕。單一 tone 的數值擬合失敗會留下原因與診斷圖，其餘 tone 繼續。

K 張圖使用 `fit_i` 命名，i 是原始 tone 序號。例如 `[2,0]` 產生 `fit_2`、`fit_0`。Title 顯示 label，label 重複也不會覆蓋圖。Summary 列出每個 active tone 的成功 scalars 或失敗原因。成功列的 `fit_quality` 比較去端點後的原始 IQ 與模型，以串接 I、Q 的實數向量計算殘差指標；公開 fit 不提供 covariance，各具名模型參數的相對誤差為 null 並附 `covariance_unavailable`，品質不決定 proposal 是否提出。

## Writeback

成功 FIT 提出一個完整 readout/mux module，只換成功 active tone 的 freq。Inactive 或失敗 tone 保留原值，全部失敗時不提 module。固定 target 是 `$mux.readout.readout_rf`。描述列出 active tone 的頻率變化或失敗。

有 Run／Load capture 時，下一列 ParameterWriteback 提案寫到 `$mux.module.readout`，值是同份 capture 解析的 module root。Apply 模組列與 slot 列後，library 保存擬合後的 readout，slot 指向該 root。下次開表單沿 slot 讀取該 library 模組。

Adapter 只從 Run／Load capture 讀取 `$mux.resonators`。Resonators 長度必須等於完整 readout 的 N，不是 active 數 K。成功 tone i 透過 resonators[i] 各提一列 freq、kappa、ext.theta0，可以逐列取消。Delay 沿單 tone 政策，使用 seed 或非等距 grid 時才提 route-qualified ext.edelay。Inactive 和失敗 tone 不提單一參數。

沒有 qubits 的 bring-up 不以 resonators 長度限制 Run。Resonators 長度不符時，summary 顯示實際數量和完整 N，保留有效 fit 的 readout module 與 capture 可解析的 readout slot 提案，省略逐 resonator scalar。Alias 未綁或來源故障仍會報錯，不改成長度不符，也不讀目前表單補綁定。Resonators 可以重複；若所選 proposals 有重複 destination，Apply 會在寫入前拒絕，可取消其中一列再套用。

Run 的 capture 固定解析 module alias 與 resonator 資料，其後修改全域 aliases 不影響這次提案。同 entry 的 Load 使用檔案 history 的 point／roles 讀取目前持久值，不承諾未保存的歷史 revision。無分析 capture 的 Load 可純計算並列 module proposal，但不能非空 Apply，也不另造 group snapshot。
