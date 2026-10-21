# Mux onetone fine

**讀者：** operator、agent。從 GUI 或 MCP 掃描 mux resonator 的共同頻率偏移，分析並套用校準結果。

**Last updated:** 2026-10-22

每個 sweep 點的名目頻率是原始 tone 的 base frequency 加上共同 offset。
圖與擬合都使用這個名目座標，DDS 實際量化頻率可能有差異。

## 準備輸入

1. 在 Inspect 的 Parameters 建立 `kind: group/mux` 元件，例如 `MG12`。設定有序 `resonators: [R1, R2, R3]`，把 module 槽 `module.readout` 指到已有的 `readout/mux` library key。完整 tones[i] 對應 resonators[i]，綁定依據是位置。
2. 在 Aliases 設定 scalar alias `mux = MG12`。Open／Reset 時，Readout 複製 `$mux.module.readout` 的 scalar 設定，tones 由 group resonators 校準值產生。缺 slot 時使用 blank mux 範本加 group tones。共同 trigger 增加最小 kappa 對應的五倍 ringup 等待，pulse 至少涵蓋 ADC window。
3. 核對 generator、mixer、每個 tone 的 ADC channel、frequency、gain，以及共用播放長度、ADC window 和 trigger timing。範本的 scalar 初值只供編輯。Run 前由操作員確認硬體安全值。
4. 可選已有的 readout/mux key，或用 Convert to inline 複製內容再修改。Tones 是單行 JSON，例如 `[{"freq":5980,"gain":0.1,"ro_ch":4,"label":"R1"},{"freq":6020,"gain":0.1,"ro_ch":5,"label":"R2"},{"freq":6040,"gain":0.1,"ro_ch":6}]`。此例只示範格式，硬體值須另行確認。
5. Active tones 選 None 量全部，或輸入原始序號 JSON list。例如 `[0,2]` 量第一及第三個 tone，`[2,0]` 則依第三、第一的順序執行、畫圖與保存 columns。序號須為唯一、有效範圍內的整數。空 list、重複、越界、bool 或非整數會在 Run 診斷中報錯並阻擋執行。請查看診斷內容判斷問題欄位。
6. 設定 Frequency offset 的 Start、Stop、Expts，以及 Reps、Rounds、Relax delay。執行前確認所有欄位有效。

Group 可選填 `qubits: [Q1, Q2, Q3]`。省略或 null 適用於 resonator bring-up。
提供時，qubits 必須與完整 resonators 等長，且每個 qubit 唯一。
Open／Reset 檢查每個元件是否存在、kind 是否正確，以及 `Q1.resonator == R1` 等逐項對應。
缺失或 null 元件會報錯。錯誤列出欄位、index 與實際值，修正來源後重新開啟或 Reset。
配對保留原始順序。Resonators 可以重複。

未綁 group 或 alias 時，必要 Readout 欄位保持未填或 Invalid。
手選或填寫有效 Readout 後仍可 Run。

## FTF 成本例

一次雙 tone 離線 Run 掃共同 offset −2 至 2 MHz，共 12 點，100 reps、1 round。
Readout 長 1 us，ADC 積分窗 0.5 us，recovery 為 0.1 us。
Driver 的 Run action 實測 2.807 s，除以共同掃描點數約為每點 0.23 s。
這是短掃描的成本量級，不代表 readout 校準已通過。

計時包含按 Run、等待終局與截圖，不是純 acquisition，也不含 Analyze 或 Save。
不可拿不同 slot 或不同實驗的秒數作速度比較。
共同 offset 每點同時量 active tones，不是各 tone 點數的乘積。
點數、active tone 數、readout 長度、積分窗與 recovery 都會改變成本。
硬體會隨 reps、rounds 增加重複 acquisition。
FTF 的編譯與快取可能主導時間，不能假設成本與 reps 成正比，也不能視增加 reps 為免費。
先用目標 readout 設定做短 Run，再估完整掃描的預算。

## Run 與圖形

Run 使用 tab 在 Open 或 Reset 時捕捉的參數來源。
其後修改 aliases、group resonators 或 library，現有 draft 仍保留原來源。
需要新來源時執行 Reset，這會重建初值。Active tones 只決定本次量測範圍，完整 readout 保留全部 tones。

Live 圖以共同 offset 為橫軸，每個 active tone 各有一張 `measurement_i` magnitude 圖。
標題使用原始 tone 序號與 label，空 label 顯示例如 `tone 2`。Label 可以重複。
資料軸使用原始 tone 序號。停止後保留已取得資料，未取得點保持 NaN。
失敗操作即使取得部分資料，仍保留失敗狀態。

## 保存與讀回

保存當次量測的完整設定與原始資料，包含全部 base tones、labels、active_tones 與 sweep。
檔案保存量測設定；resonators 與 group 綁定由 Run／Load 的參數來源提供。
Native 讀回使用檔案設定，現有 draft 及其固定來源保持原樣。
Labber 匯出保留共同 offset、原始 tone 序號與複數訊號，columns 與 tone_indices 同序。

同 entry 且有 point 的 Load 可建立分析參數來源。
跨 entry 或缺少 point 的 Load 只提供資料。
檔案設定缺失或無法恢復時，仍可讀取原始資料，FIT 會報錯並要求有效歷史設定。

## FIT

所有 active tones 共用 onetone/freq 的四欄設定。
Model type 可選 hanger `hm`、transmission `t` 或 `auto`，預設 `hm`。
Fit amplitude background 預設開啟，Fit phase curvature 預設關閉，兩者可獨立切換。

Electrical delay seed 單位為 us，預設留空。
留空時，每個 tone 使用對應 resonator.ext.edelay 中符合 generator／ADC route 的有限校準值作 branch seed。
缺少匹配時，該 tone 使用固定 100 us 上限作 global 搜尋。
明填有限數字時，所有 active tones 共用此 branch seed，0 與負值也有效。
Seed 選擇擬合起點，模型仍會擬合 delay。非有限或非數字輸入會拒絕整次 FIT。

FIT 處理 active columns，以檔案原始 base frequency 加共同名目 offset 作 MHz 座標，並去掉首尾點。
設定、selection、tone_indices 或 shape 不符時整次報錯。
單一 tone 的數值擬合失敗會留下原因與診斷圖，其餘 tone 繼續。

圖名使用原始 tone 序號。例如 `[2,0]` 產生 `fit_2`、`fit_0`，重複 label 仍保留各自圖形。
Summary 列出每個 active tone 的成功 scalars 或失敗原因。
`fit_quality` 比較去端點後的原始 IQ 與模型，以串接 I、Q 的實數向量計算殘差。
模型參數的相對誤差為 null 並附 `covariance_unavailable`，表示此擬合未提供 covariance。
Quality 供操作員審閱；是否提出候選值依成功擬合結果判定。

## Writeback

成功 FIT 提出完整 readout/mux module，只更新成功 active tone 的 freq。
Inactive 或失敗 tone 保留原值，全部失敗時省略 module 候選。
固定 target 是 `$mux.readout.readout_rf`，描述列出 active tone 的頻率變化或失敗。

有 Run／Load 參數來源時，另一列候選將 `$mux.module.readout` 指向同份來源解析的 module root。
Apply 模組列與 slot 列後，library 保存擬合後的 readout，slot 指向該 root。
下次開表單沿 slot 讀取該 library 模組。請分別選取要套用的列。

逐 resonator 候選使用當次 Run／Load 來源中的 `$mux.resonators`。
Resonators 長度須等於完整 readout 的 N。
成功 tone i 透過 resonators[i] 各提一列 freq、kappa、ext.theta0，可以逐列取消。
使用 seed 或非等距 grid 時，delay 才提出帶 generator／ADC route 的 ext.edelay 候選。
Inactive 或失敗 tone 的單一參數保持原值。

省略 qubits 的 bring-up 可在 resonators 長度不符時 Run。
長度不符時，summary 列出實際數量與完整 N。
有效 fit 的 readout module 及可解析的 readout slot 候選仍保留，逐 resonator scalar 候選省略。
Alias 未綁或來源故障會報告各自錯誤。
若所選候選有重複 destination，Apply 會在寫入前拒絕。取消重複列後再套用。

Run 固定捕捉 module alias 與 resonator 資料，其後修改全域 aliases，提案仍使用當次來源。
同 entry 的 Load 依檔案 history 的 point／roles 讀取目前持久值。
可用的歷史值以已保存內容為準。
缺少分析參數來源的 Load 可以純計算並列 module 候選；選取非空候選後 Apply 會報錯。
套用前檢查目的地、頻率變化與 fit quality。
