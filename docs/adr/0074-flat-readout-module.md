---
status: accepted
---

# ADR-0074 — 獨立的平鋪 Readout module

**讀者：** lab-author、framework-dev、agent。查閱 readout 配置、classifier 所有權與 module 引用的跨模組契約。

## 背景

內嵌 Pulse 與 DirectReadout 讓作者重複填寫 generator 與 ADC 的頻率和通道。
Readout 的引用、表單繼承與校準也各自維護同步規則。

使用者於 2026-10-15 提出：「我在考慮將Readout Module改成獨立一個Module，而不是內嵌Pulse/DirectReadout，同時移除後者。這樣之後MuxPulse/MuxPulseReadout出來後應該比較一致」。

## 決策

### Readout 擁有 generator 與 ADC 配置

持久 `PulseReadoutSpec` 與 materialized `PulseReadoutCfg` 使用 `readout/pulse`。
Generator 通道叫 `gen_ch`，ADC 通道叫 `ro_ch`。
使用者於 2026-10-15 選定：「1保留readout/pulse吧，與readout/mux一致。2叫gen_ch好了」。

Pulse readout 平鋪保存 `gen_ch`、`nqz`、`freq`、`gain`、`phase`、`waveform`、
`mixer_freq`、`pre_delay`、`post_delay`、`ro_ch`、`ro_length`、`trig_offset` 與 `desc`。
它不保存 `pulse_cfg`、`ro_cfg` 或獨立的 `ro_freq`。
MHz 頻率 `freq` 同時決定 generator 與 ADC。時間欄位使用 us，phase 使用 deg，gain 使用 normalized gain。

Pulse readout 的 `set_param` 接受 `freq`、`gain`、`length` 與 `ro_length`。
`length` 委派 waveform，其餘三項更新同名欄位。其他名稱回報 ValueError。

整個 Readout 可以引用 library key。Pulse readout 的 waveform 仍可內嵌或引用 waveform key，
包含 flat-top ramp 的遞迴引用。Readout 不引用 Pulse 條目，Reset 的 Pulse 引用保持不變。

使用者於 2026-10-15 對純 ADC Direct readout 回覆「不保留」，對獨立 ADC 頻率回覆「onetone/sa可以刪除。C」。
因此不提供 `readout/direct`、Direct 範本或 onetone/sa。
Onetone 的 linear 與 homophasal 取樣繼續使用同一個 Readout。

### Mux readout 保存有序 tones

持久 `MuxReadoutSpec` 與 materialized `MuxReadoutCfg` 使用 `readout/mux`。
Module 明填 `gen_ch`、`nqz` 與 `mixer_freq`，共用 const 播放 `length` 與 ADC window。
有序的 1 至 8 筆 tones 各自保存 `freq`、`gain`、`phase` 與 logical ADC channel `ro_ch`。
每筆 tone 的 generator 與 ADC 使用同一個 MHz 頻率，不引用 Pulse 或 waveform。
Mux readout 不接受 sweep 或逐 tone library key，`set_param` 一律拒絕更新，caller 重建 cfg 與 program。

Tone 的選填 `label` 預設空字串，隨 Spec、cfg 與 library 保存。
Label 只作顯示，可以重複，不代表 component；program 不解讀它。
Tone 序號定義資料軸順序，label 與 ADC channel 都不改變這個順序。
Form 沿用整個 module 的 ref／inline 與 Convert to inline，tones 使用 model-list picker 逐項編輯。
每筆 tone 的 freq、gain、phase、ro_ch 與 label 都可編輯。Label 不當固定 identity。

Library 新建入口識別 MuxReadoutSpec，`readout/mux:blank` 只提供 scalar 編輯初值。
Mux 實驗的 lab Form 從固定 capture 的 mux group 產生清單，每個 resonator 一個槽位。
Freq 取該 resonator 校準值，gain 使用 lab 初值，ro_ch 依 group 順序。生成器不猜硬體或校準值。
Lab 以 Template.set／Slot.set 提供依原 Spec schema 驗證的 typed model-list 初值。
Form adapter 集中生成與 nested ref／inline 路徑處理，Core 只處理普通 list input。
FixedElements 的無 participants 模式只固定目前 inline 清單的槽位數，見 [[0079]]。

使用者於 2026-10-16 曾同意 scalar 初值加必填 tones JSON 的方案。
使用者於 2026-10-20 對從 group 產生可逐欄編輯的固定清單回覆：「可以」。
本次取代 JSON leaf 候選，不擴充 InputPolicy，不新增持久 identity，也不提供硬體安全保證。

2026-10-21 註：目前的表單設計採用固定 model-list 與逐欄編輯，JSON leaf 是已被取代的候選。

### Classifier 隨 readout 保存

Pulse readout 與 mux 的每筆 tone 各有可選 `classifier`，預設 None。
數值由 library readout module 擁有，module 本身只保存而不使用分類參數。
使用者於 2026-10-29 提出：「將center/radius/confusion matrix等資訊直接放到readout cfg裡的可選欄位中」，並說明「module本身不利用，但可以自然綁定到對應的readout module上」。

Circle 保存 g/e 複數中心、可缺的 radius 與 3×3 confusion matrix。
Linear 保存 radians 的 angle、threshold 與可缺的 2×2 confusion matrix。
`Centroid3Classifier` 使用 `mode="centroid3"`，保存必填的 g_center、e_center、f_center 與 3×3 confusion_matrix。
三中心須有限且彼此不同。Matrix rows 為準備態 g/e/f，columns 為量到的 g/e/f。
分類座標為原始積分 ADC 值除以 rounded integration sample count，不扣 IQ offset；linear 以 `exp(1j*angle)` 旋轉。
矩陣 rows 為真實／準備態，columns 為量到的態，circle 順序 g/e/other，linear 順序 g/e。
矩陣只驗有限值與形狀，不轉置、不限制值域或 row sum。
Radius 提供時須有限且 >0；None 表示尚未提供 Post 半徑校準。
複數中心沿用 point 參數的 `{__complex__: [re, im]}` tag，由 `core.complex_codec` 編解碼。
Core/native conversion、library 與 data.h5 cfg 都保留 classifier，不升容器版本。
舊條目沒有 classifier 時讀為 None，不從其他參數補值。
Point 的 `readout_cal` 停用，不再讀寫，也不把舊數值搬入 classifier。
改版後重跑 singleshot GE 並 Apply 到 readout library module。
使用者於 2026-10-29 對 library 所有權與不搬舊值的選項同意：「2A 3A 4同意」。

Classifier 不保存校準時的 readout 設定，也沒有自動失效機制。
修改 readout 設定不會清除既有 classifier，取用時也不比對校準設定是否仍適用。
操作員須在修改 readout 後重跑對應校準，circle 使用 singleshot GE 與 Post，centroid3 使用三態 reference。Apply 新校準後再使用分類。
使用者於 2026-10-29 裁決：「QC3我認為分類參數比對需求還不確定，先A」。
Writeback proposal 的 materialized 設定比對只防止把新校準寫給不同設定，不檢查既有 classifier 的有效性。

Readout 工具從 cfg 產生以 ADC channel 為 key 的 detached processors，由作者明確交給 `acquire_processed`。
Mapping 必須完整覆蓋已宣告 ADC，acquisition 不掃 program registry。
每個 processor 接收 normalized float64 `(shots, reads, 2)` IQ，回傳 `(shots, out_reads, N)` float64 payload。
框架平均 shots 與 rounds，回傳 `(out_reads, *point_dims, N)`，output axes 在同次 acquire 每個 ADC 固定。
Circle 輸出 [g,e,other]，半徑邊界與等距屬 other。
Linear 輸出 [g,e]，旋轉後實部大於 threshold 判 excited，等於時判 ground。
Centroid3 輸出 [g,e,f]，每 shot 選 Euclidean 距離最近的中心，等距時依 g/e/f 順序選第一個，沒有拒判區。
F 不代表 circle 的 other，`require_other` 與 heralding 明確拒絕 centroid3。
普通 `acquire` 保留原生 IQ 與 QICK threshold／angle，不提供 host classify 或 herald option。
Builder 的 processed methods 要求 explicit converter，不把 population 當作 IQ。

三態 reference 只用 acquired IQ 與 preparation labels 建立中心及 training confusion。
它經既有完整 readout module proposal／Apply 流程，只更新 Run 捕捉的 library 目標的 classifier，保留其他欄位與 child keys。
沒有捕捉的 library 目標或非更新欄位與實測 readout 不符時明確失敗，不自動建立或猜測目標。
Training confusion 包含準備污染，correction 回到 preparation-label 座標，保留負值與大於 1 的結果。
這不是 pure energy-state population 或 readout fidelity。
STIRAP 與單頻 g↔f Raman 從該次 run 保存的 readout cfg 要求 centroid3。
缺少 classifier、型別不符或未知 ADC 明確失敗，不退回 circle、不從 live library 補值、不讀 simulator truth。
校準沿 Spec、native cfg、library 與 data.h5 的 run cfg 保存，不另建 experiment calibration storage。
Reference 分析與 centroid3 processor 共用 `utils.shot_classification` 的最近中心規則。

2026-10-29，使用者對三態校準交接選 A，答「3A」：新增 readout classifier 型別，由三態 reference Apply 到 library，後續實驗讀 saved run cfg。
對 acquire 輸出答「5A，但這不是長久之計，必須設計一個方案來回傳任意維度的數據。而不是2維」。
使用者於 2026-10-29 後續選擇逐 shot 後處理方法，保留每 ADC 一個 ndarray 與單一可變長 payload 軸。
分類 processor 明列每個狀態，不從 g/e 重建第三類；`acquire_decimated` 不變。

單 qubit GE Primary 提出只有中心的 circle，Post 提出含中心、radius 與 matrix 的完整 circle。
兩者都是完整 module replacement，同次批次依 Primary→Post 接受。
分兩次 Apply 時最後一次生效，Post→Primary 會回到只有中心。
需要 radius 的 consumer 因此提示做 Post。T1 需要 other，不接受 linear 或 centroid3。
舊 cfg 沒有 classifier 時，T1 分析略過 confusion 修正，不讀 point 校準。
GE 以 run 的 readout slot 解析目標，proposal builder 與 materialized 比對契約見 [[0008]]。
Mux GE 每個 phase 合成一個完整 proposal，只更新成功 tone 的精確 classifier 路徑。
雙 qubit 的 nearest-GE 分析只從 saved readout cfg 的角色 tone 派生 circle centers，不要求 radius。
Joint readout、chevron、constructive phase、CZ amplitude 在 cfg 驗證時拒絕 missing／非 circle。
Conditional phase、local Z、RB 保留 raw acquisition，分析將 missing／非 circle 轉為帶原因的結果，不產生 population 或 fit。
Readout owner 的單 ADC circle lookup 不理解 lab 角色，也不讀 live library。
其餘 singleshot population consumers 同樣明傳完整 ADC mapping，confusion correction 只讀 saved cfg。
Amplitude／Length Rabi 保留單 ADC raw IQ 與聯合 fit，未校準時只略過 live population preview。
有效 Rabi fit 經同一中立 builder 更新 captured resonator readout 的完整 classifier，
Pulse 更新 classifier，單 tone Mux 更新 tones.0.classifier，不支援多 tone 分析。
Check 的 raw Run 不要求 classifier，到分析才從 saved cfg 要求 circle 與 radius。

### Herald 規格與 host post-selection

Library readout cfg 擁有 herald 使用的 classifier。
單 qubit heralded T1、Amp Rabi、Len Rabi、Ramsey 與 Echo 各用兩個 `PulseReadout` instance，分別發射 herald 與 signal。
兩個 instance 共用 `cfg.readout` 與 classifier，不另存一組 herald 校準。
作者從實際發射 herald 的 instance 以 `occurrence=0` 取得規格。
同一個 instance 也可以初始化一次後多次發射；這是公開能力，不是實驗必須採用的配置。
2026-10-29，advisor 更正舊 spec 與本 ADR 將「同一 readout 設定」誤述為「同一 instance」的文字，沒有改變使用者要求的設定共用。

Readout 公開工具依自身發射序產生具名 `HeraldSpec`。
作者用 `heralded(spec, inner)` 包裝該 ADC 的 signal processor，明傳給 `acquire_processed`。
`acquire` 與 `acquire_processed` 都不接受 herald option，也不掃 registry 或查 live library。
Caller 的 processor 執行 ground classification，移除 herald read，輸出 signal numerator 與 acceptance 分量。
框架平均 numerator 與 acceptance，builder converter 呼叫 `herald_ratio` 還原 conditional signal，將 fraction 寫入實驗 Result。
Cumulative round hooks 使用相同公式，等同依 accepted shots 數加權。
零通過時 signal 為 NaN，不發出除零 warning。
Program 不保存 integrated mask/count state 或 pass-fraction accessor。
模擬把每輪 retained raw shots 交給同一 processor path，不設獨立 herald mode，也不重新呼叫 backend。
沒有新增逐發持久資料。
使用者於 2026-10-29 對以 processor helper 取代 integrated herald 與 accessor 的方案選 A，答「6A」。
實際 API、資料形狀與失敗條件見 [program 契約](../../lib/zcu_tools/program/README.md#herald-declarations)。

Lab 擁有 host scan 跨 acquire 的結果合併，使用 `herald_ratio` 回傳的通過比例，不重做分類或遮罩。
Shaped Len Rabi 保留 round-major，掃完一條曲線才判斷 early-stop SNR。
Lab 只合併實際交付的取數結果，依通過數加權；停止而略過 acquire 時，不把 cached program 的上一輪統計算成新資料。
合併公式、零通過與未量測點的行為由 [Len Rabi guide](../../zcu_lab/v2/heralding/len_rabi/guide.md#run-and-inspect) 與 core 說明。

實驗層以獨立的 `heralding/*` 組合序列，不在既有實驗加入 Reset 模組。
目前提供單 qubit pulse-readout T1、Amp Rabi、Len Rabi、Ramsey 與 Echo。
實驗的 Result 保存同網格 `pass_fraction`，與 `signals` 的名稱、shape 及 MCP recipe 呼叫保持不變。
四個第二階段成員沿用一般版本的 writeback 目標與規則，描述標明 Heralded，不另設 heralded 校準欄位。
這是 host post-selection，不是 active reset；不包含 mux herald 或雙 qubit 四態聯合分類。
操作、保存與 writeback 目標見各 guide：[T1](../../zcu_lab/v2/heralding/t1/guide.md)、
[Amp Rabi](../../zcu_lab/v2/heralding/amp_rabi/guide.md)、[Len Rabi](../../zcu_lab/v2/heralding/len_rabi/guide.md)、
[Ramsey](../../zcu_lab/v2/heralding/t2ramsey/guide.md)、[Echo](../../zcu_lab/v2/heralding/t2echo/guide.md)。

使用者於 2026-10-29 回答先做取數層 post-selection、直接做 FPGA active reset 或兩者並行的選項時裁決：
「那不做active reset，讓底層支援post-selection，上層使用新實驗組，比如heralding/*」。
同日，對分類參數由作者自取、acquire 自動掃描，或 readout 工具產生具名規格再由作者明傳三個選項，使用者選 C：「我也傾向C」。
對只做 mux 或也做單 qubit pulse readout 的範圍問題，使用者選後者並要求先做單 qubit：「3我反而想先測試在單qubit，因為比較簡單，B」。
對實驗層不存通過比例、Result 存同網格 `pass_fraction`，或只記整體平均三個選項，使用者選 B：「5B」。

使用者於 2026-10-29 對「第二階段先做單 qubit Rabi／Ramsey 成員，mux 與雙 qubit 四態聯合分類留待後續討論」的建議回答：「herald採用建議」。
對 Amp Rabi、Len Rabi、Ramsey、Echo 四個成員的清單回答：「herald第二階段清單採用建議」。
對 writeback 的三個選項，A 與一般版相同並標明 Heralded、B 只顯示不提 writeback、C 寫入 heralded 專用欄位，使用者選 A：「Herald選A」。

依賴沿用 [[0072]] 的分層與 `.importlinter` C8、C15、C16。
Experiment 使用 program 的後選擇能力，lab core 不依賴 GUI，framework 不反向載入 lab。
這些規則不改變真實 acquire 與模擬的執行邊界，也不新增依賴方向。

### Owner 不變，caller 不重建舊配置

`cfg_model` 擁有 Spec、cfg 驗證、引用展開與參數設定。
`program` 負責 native conversion、QICK 宣告與排程。
`resources` 擁有 library 文件與 I/O。
`gui.form_adapter` 從 core cfg 與持久 Spec 推導 Form，session 擁有 Library entry 表單。
這些分工沿用 [[0072]]、[[0065]] 與 [[0073]]，不保留 editable shape catalog。

依賴沿用 `.importlinter` 的 C11、C16、C19 與 `form-adapter-no-host`。
Readout 不新增 registry、factory 或跨 owner resolver。
Core/native factory 使用 caller 捕捉的 resolver 與 asset capability，不查 process-global library。

Pulse readout runtime 可將平鋪 generator 欄位投影成內部 PulseCfg，以重用 waveform 與播放邏輯。
Caller 不維護另一份 ADC cfg。Pulse cfg 不含 `ro_ch`，runtime Pulse 另接收 ADC matching 通道。
PulseRegistry 的硬體定義身分包含該通道，避免相同 pulse cfg 在 ADC 0 與 1 間錯誤共用。

Pulse readout 在 module 起點送出 readout configuration，在 `trig_offset` 觸發 ADC，
在 `pre_delay` 播放 generator。Blocking duration 覆蓋 ADC window 和含 post-delay 的 generator 播放，
各自沿既有硬體 clock rounding。

### 高階頻率與低階 words 分開

`TablePulseReadout` 保留成對的 generator／ADC frequency words。
`freq_val`、`ro_freq_val`、`freq_words` 與 `ro_freq_words` 是 runtime register／table 契約，
不是持久 cfg 的兩個頻率。兩種 words 表示同一語意頻率，由各硬體的 mixer／downconversion 規則編碼。
模擬的 accumulated、raw 與 decimated 路徑使用同一份平鋪 readout plan，數值單位與物理模型不變。

### 表單與持久資料

Pulse readout Form 直接編輯 `gen_ch`、`ro_ch`、`freq`、`gain`、`waveform` 與 ADC window。
`gen_ch`、`ro_ch` 必填且可編輯，不從其他欄位推導或補值。
Form 只在整個 module 和 waveform 的 inline input 加入 wrapper。
例如 recipe 頻率路徑為 `modules.readout.module.freq`，持久 Spec 路徑為 `freq`。
Readout 不使用舊 shape inheritance 或自動補值，表單由共用 Form 生成器建立。

舊 nested Readout 與 Direct 條目不遷移，也不加相容讀取。
正常驗證入口回報具體 library key，作者重建條目。
Artifact restore 的 typed cfg 失效與原始 data 可讀性由 [[0063]] 的 owner 負責。

Module library 版本只跟容器結構改變，不因 module／waveform 欄位變動升版。
使用者於 2026-10-15 定案：「只是改Module而已真的需要升Library版本嗎？就像加一個Component原件定義，不應該讓舊版本不能讀對吧」「1可以，2A，3保留2.0」。
Library 維持既有 2.0，不加 readout 條目版本。
實驗 cfg schema major 與宿主 wire／GUI revision 依各 owner 的不相容契約更新。
Parameter container、native artifact 與 ledger 的外層格式不因 Readout 欄位形狀改變。

## 取捨

共用某個 Pulse 條目的 Readout 改為自行保存 generator 設定，兩者需要分別維護。
換來的是一個頻率來源和一套表單路徑，runtime 仍可重用 Pulse 的播放實作。
純 ADC 與獨立 ADC 頻率不在 Readout 契約內。
Mux readout 共用 generator 與時間欄位，不能逐 tone 設定 waveform 或時間窗。
