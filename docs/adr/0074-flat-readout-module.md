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

Classify 軸以 one-hot 表示分類結果。
Circle 輸出 [g,e,other]，半徑邊界與等距屬 other。
Linear 輸出 [g,e]，旋轉後實部大於 threshold 判 excited，等於時判 ground。
Centroid3 輸出 [g,e,f]，每 shot 選 Euclidean 距離最近的中心，等距時依 g/e/f 順序選第一個，沒有拒判區。
F 不代表 circle 的 other。
普通 `acquire` 保留原生 IQ 與 QICK threshold／angle，不提供 host classify 或 herald option。
分類取數的 processor、累加與保存分工見下節 [Shot sums 與 heralding](#shot-sums-與-heralding)。

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
同日，對沿用 QICK 的 float64 I/Q 分量或先轉成 complex128 的選項，使用者回答：「我想推進Program設計一致性，選A」。
Processor 接收 I/Q 分量，需要複數運算時自行轉換。`acquire_decimated` 不變。
原先每 ADC 一個 processor、單一 payload 軸與框架取平均的方案，由 2026-10-30 的 shot-sums 契約取代。

單 qubit GE Primary 提出只有中心的 circle，Post 提出含中心、radius 與 matrix 的完整 circle。
兩者都是完整 module replacement，同次批次依 Primary→Post 接受。
分兩次 Apply 時最後一次生效，Post→Primary 會回到只有中心。
需要 radius 的 consumer 因此提示做 Post。T1 需要 other，不接受 linear 或 centroid3。
舊 cfg 沒有 classifier 時，T1 分析略過 confusion 修正，不讀 point 校準。
GE 以 run 的 readout slot 解析目標，proposal builder 與 materialized 比對契約見 [[0008]]。
Mux GE 每個 phase 合成一個完整 proposal，只更新成功 tone 的精確 classifier 路徑。
雙 qubit 的 9 個實驗各有 ordinary 與 heralded 兩版，兩版都在 Run 前要求兩個角色 tone 的 circle 與 GE Post 的 radius，缺少或非 circle 時 cfg 驗證報錯。
取數端以 shot sums 分類：ordinary 存終端讀出的 9 類計數，heralded 存 herald 加終端的 81 類計數；Result 保存原 float64 總和，不再保存逐 shot IQ，也不在 analyze 分類。
2026-10-30 使用者裁定：「不要再有逐shot記錄的實驗了，除了singleshot/ge這種本來就是用於生成分類參數的實驗，其他都請在平均時分類」。gg／ee 分析與 writeback 細節由 `zcu_lab/v2/two_qubit/guide.md` 擁有。
Readout owner 的單 ADC circle lookup 不理解 lab 角色，也不讀 live library。
其餘 singleshot population consumers 使用 shot-sums 分類取數，confusion correction 只讀 saved cfg。
Amplitude／Length Rabi 保留單 ADC raw IQ 與聯合 fit，未校準時只略過 live population preview。
有效 Rabi fit 經同一中立 builder 更新 captured resonator readout 的完整 classifier，
Pulse 更新 classifier，單 tone Mux 更新 tones.0.classifier，不支援多 tone 分析。
Check 的 raw Run 不要求 classifier，到分析才從 saved cfg 要求 circle 與 radius。

### 原生 QICK projection 的 readout 與 caller 分工

普通 `acquire` 保留原生 QICK threshold／angle projection。
Caller 依實驗語意決定是否傳入 threshold／angle，並負責 ADC 宣告順序、完整覆蓋與校準是否仍適用。
原生 projection 將 excited fraction 放在 I，Q 為零，讓既有 IQ analysis 可使用這個等效 IQ。
使用者於 2026-10-29 說：「1混入IQ是故意的，就是要變成等效IQ，好讓分析程序可以不用動。2所以是由實驗方手動傳遞3這是用戶責任。4目的是自帶0~1的合理範圍，這樣數據的值本身就有意義。」

使用者於 2026-10-31 裁定：「7,8,9,13照建議」。
沒有 caller 的 classifier-to-QICK options helper 已移除。
Caller 自行準備原生 projection 參數，readout 不提供缺少校準時整體退回普通 IQ 的 helper。
此分工沿用 [[0072]] 與 `.importlinter` C8、C15、C16 的依賴方向，不新增跨層入口。

### Shot sums 與 heralding

`acquire_summed` 接受單一 processor，同時交付所有已宣告 ADC 的正規化 IQ。
輸入以 ADC channel 為 key，每個陣列是 float64 `(points, reps, reads, 2)`。
Processor 每點回傳一個總和陣列，框架只累加已完成 round 的回傳值，不除以 reps 或 rounds。
硬體與模擬使用同一條 processor 與累加路徑。

讀出軸只有 `Classify` 與 `IQ`。
Classify 把一次讀出的分類結果轉成 k 維 one-hot；IQ 把一次未分類讀出轉成 `(1, I, Q)`。
Processor 對每個掃描點計算 `Σ_shots v₁ ⊗ v₂ ⊗ …`，輸出形狀是 `(points, *axes)`。
軸順序由作者列出的讀出軸決定。同一次讀出可列多次，以保存二次矩或交叉項。
所有 IQ 軸都取第 0 格時得到計數；只有 Classify 軸時，整個輸出就是各聯合類別的計數。

讀出以字串 label 指定，不要求作者計算 ADC 的 read index。
`shot_sums(prog, axes)` 在 compile 後、取數前解析 label。
Timeline 發射用 emit tag，未標記的發射以 module 名稱識別。
Label 必須唯一指向一次讀出發射，否則在取數前報錯。
Classifier 取自該 readout 的 cfg 或作者明確提供的 override，不查 live library。

Result 原樣保存 processor 的 float64 總和，不在 run 中換成平均或比例。
分析與繪圖從計數和總和計算比例、平均 IQ 或條件平均。
Schedule 的 `run_acquire_summed` 讓 caller 以 `RunOptions(converter=...)` 選擇 live signal；未提供 converter 時原樣寫入總和，不把類別計數當作 IQ。
搬遷後的實驗拒絕缺少 counts／sums 的舊格式，不從舊比例推算計數，也不加相容路徑。

單 qubit heralding 使用 `[Classify("herald"), IQ("readout")]` 保存各 herald 類別下的終端 IQ 總和。
分析選取 herald 判為 g 的類別，再由其計數算出條件平均；其他 herald 類別的資料仍保留。
Library readout cfg 擁有 classifier。現有實驗用兩個 PulseReadout instance 發射 herald 與 signal，共用同一份 readout cfg，不另存 herald 校準。
Host scan 跨 acquire 時由 lab 加總實際取得的 sums，不把上一輪 cached 結果算成新資料。
這是 host post-selection，不是 active reset。實際介面與資料形狀見 [program 契約](../../lib/zcu_tools/program/README.md#summed-acquisition)。

使用者於 2026-10-30 要求單一 processor：「H1 , 我認為processor就不要每個adc分開給，統一只給一個就行」。
同日明定分類時機：「分類請不要在analyze裡做，浪費記憶體，分類請在processor裡做，將nread, 2的資料轉成M維資料」。
對讀出軸，使用者提出：「不用堆到一個M維度裡，而是一個嵌套維度，一般iq訊號被轉成3維(1,i, q)，而k分類器轉成k維，幾次readout就幾層」。
整理成逐 shot 外積總和後，使用者確認：「1是 2你說得Result是什麼？ 3可以 我認為當前設計夠簡單但靈活」。
同日的實驗保存約束是：「不要再有逐shot記錄的實驗了，除了singleshot/ge這種本來就是用於生成分類參數的實驗，其他都請在平均時分類」。
本次總和保存裁定取代 2026-10-29「5B」的同網格 pass_fraction 保存方案。
Result 改存所有 herald 類別的總和，通過比例改由計數衍生，不再只留下通過類別的資料。

實驗層保留獨立的 `heralding/*` 組合序列，不在既有實驗加入 Reset 模組。
T1、Amp Rabi、Len Rabi、Ramsey 與 Echo 沿用普通版 writeback 目標與規則，描述標明 Heralded，不另設校準欄位。
使用者於 2026-10-29 裁決：「那不做active reset，讓底層支援post-selection，上層使用新實驗組，比如heralding/*」。
對 Amp Rabi、Len Rabi、Ramsey、Echo 的成員清單，使用者回答：「herald第二階段清單採用建議」。
對沿用普通版 writeback 並標明 Heralded 的選項，使用者回答：「Herald選A」。
操作、保存與 writeback 目標見各 guide：[T1](../../zcu_lab/v2/heralding/t1/guide.md)、
[Amp Rabi](../../zcu_lab/v2/heralding/amp_rabi/guide.md)、[Len Rabi](../../zcu_lab/v2/heralding/len_rabi/guide.md)、
[Ramsey](../../zcu_lab/v2/heralding/t2ramsey/guide.md)、[Echo](../../zcu_lab/v2/heralding/t2echo/guide.md)。

依賴沿用 [[0072]] 的分層與 `.importlinter` C8、C15、C16。
Experiment 使用 program 的取數能力，lab core 不依賴 GUI，framework 不反向載入 lab。
本次修改不改變真實 acquire 與模擬的執行邊界，也不新增依賴方向。

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
