---
status: accepted
---

# 0074 — 獨立的平鋪 Readout module

## 背景

內嵌 Pulse 與 DirectReadout 讓作者重複填寫 generator 與 ADC 的頻率和通道。Readout 的引用、表單繼承與校準也各自維護同步規則。

## 決策

### Readout 擁有 generator 與 ADC 配置

**規則。** 持久 `PulseReadoutSpec` 與展開後的 `PulseReadoutCfg` 都使用 `readout/pulse`。Readout 平鋪保存 generator 通道 `gen_ch`、ADC 通道 `ro_ch`、waveform 和讀出時序，不內嵌 `pulse_cfg` 或 `ro_cfg`。單一 `freq` 以 MHz 同時指定兩端的頻率。

**例子。** Onetone 的 linear 與 homophasal 取樣共用同一個 Readout。`set_param` 接受 `freq`、`gain`、`length` 與 `ro_length`。只有 `length` 委派給 waveform，其餘更新同名欄位。平鋪欄位另有 `nqz`、`phase`、`mixer_freq`、`pre_delay`、`post_delay`、`trig_offset` 與 `desc`。時間用 us，phase 用 deg，gain 用 normalized gain。

**邊界。** 不提供 `readout/direct`、Direct 範本、獨立的 `ro_freq` 或 `onetone/sa`。不支援的 `set_param` 名稱回報 `ValueError`。整個 Readout 可以引用 library key；waveform 可 inline 或引用 waveform key，flat-top ramp 也可遞迴引用。Readout 不引用 Pulse 條目，Reset 的 Pulse 引用不變。共享配置由 [cfg_model](../../lib/zcu_tools/cfg_model/README.md) 擁有。

**依據。** 2026-10-15，使用者提議：「我在考慮將Readout Module改成獨立一個Module，而不是內嵌Pulse/DirectReadout，同時移除後者。這樣之後MuxPulse/MuxPulseReadout出來後應該比較一致」。同日選定：「1保留readout/pulse吧，與readout/mux一致。2叫gen_ch好了」。對純 ADC readout 答「不保留」，對獨立 ADC 頻率答「onetone/sa可以刪除。C」。原始討論未入庫，見 Git `f647c9a9c`。

### Mux readout 保存有序 tones

**規則。** `MuxReadoutSpec` 與 `MuxReadoutCfg` 使用 `readout/mux`。Module 明填 generator、Nyquist zone 與 mixer，共用 const 播放長度和 ADC window。每筆 tone 的 generator 與 ADC 共用一個 MHz 頻率。Tone 在清單中的位置決定資料軸，顯示用的 `label` 與 ADC 編號都不改變軸順序。配置、tone 數量及 label 的保存契約見 [cfg_model](../../lib/zcu_tools/cfg_model/README.md)。

**例子。** Mux 實驗的 lab Form 用固定 capture 中的 mux group 建初始清單，每個 resonator 一個槽位。`freq` 取校準值，`gain` 取 lab 初值，`ro_ch` 依 group 順序。Form 沿用整個 module 的 ref／inline 與 Convert to inline，再用 model-list picker 逐欄編輯 tone。Library 新建入口識別 `MuxReadoutSpec`；`readout/mux:blank` 只提供 scalar 編輯初值，見 [lab 的初值說明](../../zcu_lab/v2/README.md)。Typed model-list 初值與 nested ref／inline 由 [Form adapter](../../lib/zcu_tools/gui/form_adapter/README.md#native-model-lists) 處理，Core 仍收普通 list input。

**邊界。** Mux 不引用 Pulse、waveform 或逐 tone library key，不接受 sweep，`set_param` 拒絕更新。要改設定，由 caller 重建 cfg 與 program。Label 可以重複，不代表 component 或持久 identity。Lab 提供初值，生成器不猜硬體和校準值，也不保證硬體安全。不擴充 `InputPolicy`；`FixedElements` 不帶 participants 時只固定目前 inline 清單的槽位數，見 [[0079]]。

**依據。** 2026-10-20，使用者對從 group 建立可逐欄編輯的固定清單答「可以」，2026-10-21 的註記確認採用固定 model-list。2026-10-16 的舊 Form 方案由此取代，歷程見 Git `2277294b9` 及本 ADR 的 Git 歷史；原始討論未入庫。Mux 不接受 sweep 等配置規則是 agent 的選擇。使用者 2026-11-18 說「我大概看了一遍，基本上是没问题」，也說這類規則「并不算是裁决，而算是建议」。這次過目不構成裁決，引用時只作參考。原始討論未入庫，見 Git `1f4b16aa0`。

### Classifier 隨 readout 保存

**規則。** Library readout module 擁有分類校準。Pulse readout 與 mux 的每筆 tone 都有可選 `classifier`，預設 `None`。Readout module 只保存，不在宣告或播放時分類。Spec、native cfg、library 與 `data.h5` 的 run cfg 都保留欄位，不升容器版本；校準不另建 experiment storage。分類取自 run 使用的 cfg，不查 live library。取數與結果保存見 [Shot sums 與 heralding](#shot-sums-與-heralding)。

**例子。** 三態 reference 只用 acquired IQ 和 preparation labels 建立中心及 training confusion，並用既有的完整 module proposal／Apply 流程更新 Run 捕捉的 library 目標。只改 classifier，保留其他欄位與 child keys。Reference 與 centroid3 分類共用 `utils.shot_classification` 的最近中心規則。STIRAP 的 [delay](../../zcu_lab/v2/twotone/stirap/delay/guide.md)、[amplitude](../../zcu_lab/v2/twotone/stirap/amplitude/guide.md)、[detuning](../../zcu_lab/v2/twotone/stirap/detuning/guide.md) 與單頻 [g↔f Raman](../../zcu_lab/v2/twotone/raman_gf/guide.md) 在 Run 前都要求指定 ADC 有 centroid3 classifier。

- 型別、矩陣方向與驗證、正規化 ADC 座標及複數編碼見 [cfg_model](../../lib/zcu_tools/cfg_model/README.md)。[Program 的 readout 契約](../../lib/zcu_tools/program/v2/modules/README.md#readout-configuration) 說明按 ADC 取校準與各模式的分類軸。Circle 的半徑邊界和等距算 other；linear 旋轉後實部等於 threshold 算 ground，大於才算 excited；centroid3 選 Euclidean 距離最近的中心，平手依 g/e/f 順序取第一個，沒有拒判區。f 不是 circle 的 other。
- 單 qubit GE Primary 提案只有 circle 中心，Post 提案有中心、radius 與 matrix。兩者都是完整 module replacement，批次依 Primary→Post 接受，分開 Apply 則最後一次生效。Post→Primary 會回到只有中心。Mux GE 每個 phase 合成一個完整 proposal，只更新成功 tone 的 classifier 路徑。GE 的 Run slot、proposal builder 與 materialized 比對見 [[0008]]。

**邊界。** 沒有捕捉的 library 目標，或非更新欄位與實測 readout 不符時，writeback 失敗，不建也不猜目標。這項比對只防止新校準寫到不同設定；已存 classifier 不附校準時的 readout 設定，改設定不清除它，取用時也不比對。改 readout 後，操作員重新做相應校準再 Apply：circle 做 singleshot GE 與 Post，centroid3 做三態 reference。舊條目沒有 classifier 時讀作 `None`，不讀寫 point 的 `readout_cal` 或搬用舊值。

- Raman 與上述三個 STIRAP 的 Analyze 讀保存的 g/e/f counts，不再分類。它們要求合法完整的 saved cfg，錯 classifier 型別或未知 ADC 仍拒絕。合法 cfg 缺 classifier 時只畫分類比例；矩陣奇異或把有效點解成非有限值時，記一筆 warning 後走相同路徑。兩階段都不退回 circle、不查 live library，也不讀 simulator truth。Training confusion 包含準備污染。成功校正回到 preparation-label 座標，保留負值與大於 1 的結果，不代表 pure energy-state population 或 readout fidelity。
  Run 的嚴格 ADC lookup 見 [`centroid3_classifier`](../../lib/zcu_tools/program/v2/modules/readout.py)。[Analysis 子模組](../../lib/zcu_tools/analysis/README.md#子模組)說明 `analyze_centroid3_populations` 與 `Centroid3Summary.from_applied_matrix`；實際欄位見 [recipe 回報](../../zcu_lab/recipes/README.md#操作文件與驗收)。
- 雙 qubit 分類實驗在 Run 前驗證兩個角色 tone 都有 circle 與 GE Post 的 radius，否則 cfg 驗證報錯。一般入口保存終端讀出的 9 格計數，[獨立 heralded 家族](../../zcu_lab/v2/heralded/README.md) 保存 herald × 終端的 81 格計數。Result 存原 float64 總和，不存逐 shot IQ，也不在 Analyze 分類。Readout 的單 ADC circle lookup 不解讀 lab 角色。
- T1 需要 other，不收 linear 或 centroid3。舊 cfg 沒有 classifier 時略過 confusion 修正。需要 radius 的 consumer 提示做 Post。其餘 singleshot population consumers 在取數時分類，confusion correction 只用 saved cfg。Amplitude／Length Rabi 保留單 ADC raw IQ 與聯合 fit；未校準只略過 live population preview。有效 fit 透過同一中立 builder 更新捕捉的 resonator readout，Pulse 改 `classifier`，單 tone Mux 改 `tones.0.classifier`，不支援多 tone 分析。Check 的 raw Run 不要求 classifier，Analyze 才從 saved cfg 要求 circle 與 radius，見 [Check guide](../../zcu_lab/v2/singleshot/check/guide.md)。

**依據。**
- 2026-10-29，使用者提議「將center/radius/confusion matrix等資訊直接放到readout cfg裡的可選欄位中」，並說「module本身不利用，但可以自然綁定到對應的readout module上」。對 library 所有權與不搬舊值答「2A 3A 4同意」，對比對校準設定答「QC3我認為分類參數比對需求還不確定，先A」。見 [Singleshot 分類參數放進 readout cfg](../discussions/2026-10-29-readout-classifier.md)；其中早期取數與兩次 Apply 的說法不代表現行契約。
- 2026-10-29，使用者對三態校準交接答「3A」；對沿用 QICK 的 float64 I/Q 分量答「我想推進Program設計一致性，選A」。Reducer 收 I/Q 分量，需要複數時自己轉換，`acquire_decimated` 不因這項選擇變更。原始討論未入庫，見 Git `9005d4ede`。同日提出的每 ADC processor、單一 payload 軸及框架取平均，在 2026-10-30 由 [shot-sums 裁決](#shot-sums-與-heralding) 取代。
- 使用者 2026-11-18 過目分類軸等 agent 選擇時說「我大概看了一遍，基本上是没问题」，並說「并不算是裁决，而算是建议」。引用這些選擇時只作參考。原始討論未入庫，見 Git `1f4b16aa0`。
- 2026-11-23，使用者對保存矩陣奇異答「0423矩陣奇異就不校正我接受，但還是可以提writeback，還有recipe接口本來就可以改，他是連動修改的」、「2矩陣奇異就跟沒矩陣一樣處理」；對 Raman／STIRAP 無校正的 Analyze 答「M1、邊界對，D」。Run 仍要求 centroid3；Analyze 只讀已分類 counts。見[讀出矩陣奇異就當作沒有矩陣](../discussions/2026-11-23-singular-matrix-as-absent.md)與[無校正的 Raman／STIRAP 分類圖與摘要](../discussions/2026-11-23-classified-raman-stirap-analysis.md)。

### 原生 QICK projection 的 readout 與 caller 分工

**規則。** 普通 `acquire` 保留原生 IQ 與 QICK `threshold`／`angle` projection，不提供 host classify 或 herald option。Caller 依實驗語意準備原生參數，負責 ADC 宣告順序、完整覆蓋及校準是否仍適用。

**例子。** 要讓既有 IQ analysis 讀取 projection，caller 明示 `threshold`／`angle`。QICK 將 excited fraction 放在 I，Q 為零，見 [Native acquisition](../../lib/zcu_tools/program/README.md#native-acquisition)。

**邊界。** Readout 不提供 classifier-to-QICK options helper，也不在校準不足時替 caller 整體退回普通 IQ。Host 分類走 [shot sums](#shot-sums-與-heralding)；帶遮罩的原生取數限制見 [Herald reset 與逐 shot 遮罩](#herald-reset-與逐-shot-遮罩)。依賴沿用 [[0072]] 與 `.importlinter` C8、C15、C16，不新增跨層入口。

**依據。** 2026-10-29，使用者說：「1混入IQ是故意的，就是要變成等效IQ，好讓分析程序可以不用動。2所以是由實驗方手動傳遞3這是用戶責任。4目的是自帶0~1的合理範圍，這樣數據的值本身就有意義。」2026-10-31，對 caller 分工答「7,8,9,13照建議」。原生取數的初期討論見 [Singleshot 分類參數放進 readout cfg](../discussions/2026-10-29-readout-classifier.md)；移除 helper 的原始討論未入庫，見 Git `dda262afd`。

### Shot sums 與 heralding

**規則。** `acquire_summed` 與 Schedule 的 `run_acquire_summed` 接收單一 `reducer=`。框架持有 state，reducer 定義如何累加。硬體與模擬共用處理路徑；取消時只保留已完成 rounds 的 state。Round hook 與最終結果都取 reducer 輸出，框架只還原點的維度，不另加總或平均。Result 原樣保存 float64 總和，分析與繪圖才從總和與計數算比例、平均 IQ 或條件平均。需要分類時在取數時完成，不在 Analyze 分類。

**例子。** `twotone/t1_cov` 不需要 classifier。它在同一 shot 用相同的 Pulse readout cfg 發射 `pre_readout` 和 `post_readout`，`shot_sums` 保存兩個二階 `IQ` 軸的 float64 `(N, 6, 6)` 外積總和。兩側共用的中心與 Gaussian width 由二階矩自校準。J 是兩次讀出的聯合機率，K 的每列是條件轉移機率；用 `C = 1 − tr J` 擬合與更新即時圖，K 只供診斷。播放、Analyze、Summary、live 與存檔契約由 [T1 cov guide](../../zcu_lab/v2/twotone/time_domain/t1_cov/guide.md) 擁有，自校準由 [analysis](../../lib/zcu_tools/analysis/README.md#雙讀出矩的自校準) 擁有。Cov 比較掃描點的變化，接受讀出期間衰減與拖尾帶來的偏差，不宣稱無偏的 gate fidelity。

**邊界。** Reducer 可用自己的 state 做不可相加的統計，例如 KMeans；取數層不另留 tracker。Reducer 的 `start`／`add`／`output`、輸入形狀、遮罩及加總型 reducer 見 [Summed acquisition](../../lib/zcu_tools/program/README.md#summed-acquisition)。來源消耗的讀取不交給 reducer。Schedule 沒有 `RunOptions(converter=...)` 時原樣交付總和，不把類別計數當作 IQ。

- 只有 `Classify` 與 `IQ` 兩種軸。前者把一次讀出的類別轉成 one-hot，後者預設二階六分量。軸階數不是態數。作者顯式列軸及順序，可重列一次讀出做外積。`shot_sums` 只加總遮罩留下的 shots；全取 IQ 軸第 0 格是計數，只有 `Classify` 時則是聯合類別的計數。Classifier 取自 readout cfg 或顯式 override，不查 live library。Label 在 compile 後、取數前解析，使用 emit tag 或未標記發射的 module 名稱；缺失或有歧義即失敗，不猜 ADC read index。
- Readout 與 JPA optimization 顯式選 `IQ(read, order=3)`，live 與最終分數用同一份已完成 rounds 的總和，不另存統計狀態。十個分量及解出的計數與原始矩見 [Program](../../lib/zcu_tools/program/README.md#summed-acquisition)，sample covariance、三階中心矩用於 [SNR 與 skew penalty](../../lib/zcu_tools/experiment/v2/README.md#v2-工具與實驗輔助)。
- 搬遷後的實驗拒絕沒有 counts／sums 的舊格式，不從比例逆推計數。T1 cov 只收單 tone `readout/pulse` 和可選脈衝類 reset，不用 herald 遮罩。不載入舊 `heralding/t1`，也不提供 alias。這裡 cov 指二階矩，heralding 指分類後的 post-selection，並非 active reset；放置規則見[下節](#herald-reset-與逐-shot-遮罩)。雙 qubit heralded 保留分類。
- Experiment 使用 program 的取數能力；lab core 不依賴 GUI，framework 不反向載入 lab。真實 acquire 和模擬的執行邊界不改，依賴沿用 [[0072]] 與 `.importlinter` C8、C15、C16。

**依據。**
- 2026-10-30，使用者要求單一處理器：「H1 , 我認為processor就不要每個adc分開給，統一只給一個就行」。分類時機：「分類請不要在analyze裡做，浪費記憶體，分類請在processor裡做，將nread, 2的資料轉成M維資料」。軸設計：「不用堆到一個M維度裡，而是一個嵌套維度，一般iq訊號被轉成3維(1,i, q)，而k分類器轉成k維，幾次readout就幾層」。整理為逐 shot 外積總和後答「1是 2你說得Result是什麼？ 3可以 我認為當前設計夠簡單但靈活」。保存原則：「不要再有逐shot記錄的實驗了，除了singleshot/ge這種本來就是用於生成分類參數的實驗，其他都請在平均時分類」。分類軸的嵌套保留，IQ 分量以後續裁決為準。原始裁決討論未入庫，見 Git `05df4f6bb`。2026-10-29 的舊 pass_fraction 方案見 [Heralding post-selection 討論](../discussions/2026-10-29-heralding-post-selection.md) 與 ADR 的 Git 歷史。
- 2026-11-02，使用者提出「讓iq模式的維度變成[1,i1,q1,1i,1q,ii,iq,qi,qq]就好了」，對去重成六個分量答「a」，並說「先固定2階，3階有需要再考慮支援」。對保留兩種軸說：「實作上，我考慮你的簡單方案，因為使用IQ與Classify的實驗似乎都分開了？至少在使用時就知道。」對分類使用範圍說：「a，但有些實驗，可能特別要求取得population，還是要支援分類，但一般只是為了訊號比較好，取出initial population影響的實驗，可以不給classifier」。對 cov 偏差與擬合說：「老實說有偏差沒關係，重點是我能不能很好的看到不同掃描點的K的變化，訊號是相對的」「那就都用C，先不要直接擬合K，除非有必要我再討論」。見[二階矩自校準的討論](../discussions/2026-11-02-herald-transfer-matrix.md)。
- 2026-11-19，使用者說「可以加回來，應該說，我刪掉的不是cov版，而是heralding版」，並要求「把前一种基于共变因素的实验，把它纠正其义，不要再叫其 hurdling，呃，改成加上什么 COV 的 postfit 之类的。」2026-11-20，對先恢復 T1 答「舊題都採用建議」；對名稱、readout 與 reset 答「關於t1_cov，1A，名字叫pre_readout/post_readout比較明確。2採用建議。3我認為可以加可選reset，作為穩定ge分佈使用。」見 [T1 二階矩實驗以 t1_cov 恢復](../discussions/2026-11-20-t1-cov-restore.md)。該討論也記錄 2026-11-09 使用者說「舊 heralding 我完全沒用過也沒讀過，可以直接刪掉。」2026-10-29 舊實驗組的成員及 writeback 不再適用，見 [早期討論](../discussions/2026-10-29-heralding-post-selection.md) 與 Git `188409d1d`。
- 2026-11-20，使用者將 tracker 與 processor 的重疊分工改成顯式 reducer 與 state。語音原話：「Jiki 基本没问题，但我有发现一个设计上的取舍。我发现有 processor 机制之后，我好像不需要 tracker 机制，后者是不是可以删掉了？」「但这两个东西的定位重复了，我不是很想要重复定位的东西出现在城市里，这对于使用者跟维护者来说都是负担。无单调缺可，就必须放弃。一些不可加的统计能力，比如说 K-means，K-means，所以这是我烦恼的。」「那我觉得还是用比较显示的方法比较好，也就是一个 reducer，然后再一个 stage，然后者是状态，前者是累加器，这个抽象会比较合理一点，比较自然一点。」其中「Jiki」讀作「幾題」，「城市」是「程式」，「stage」是 state，「显示」是「顯式」。對一併實作與採用關鍵字 `reducer=` 答「采用建议」。見 [tracker 與 processor 的討論](../discussions/2026-11-20-herald-reset-shot-masks.md#tracker-與-processor-的定位重複)。

#### 選用校正後的 IQ 矩

**規則。** 作者用 `IQ("readout", offset_corrected=True)` 明示需要原生 QICK 平均的零點。Program 解析編譯後的 logical read，以讀出積分長度正規化每發 I、Q，再依 QICK 的 offset policy 各扣同一 scalar，最後加總 `[N, ΣI, ΣQ, ΣI², ΣIQ, ΣQ²]`。靜態 PFB 由該 ADC 的取數設定決定倍率；處理當輪時缺設定就報錯，不猜 offset。

**例子。** 傳統 Amp Rabi 用一個 summed slot。若原平均為 `5+7j`，QICK 要求各扣 `2`，就用 `3+5j` 作為 live、SNR 早停及最終 `signals` 的平均。`iq_moments` 留在同一零點，SEM 使用完成 rounds 中保留的 N，不用 cfg 補算。

**邊界。** 此 opt-in 只接受單一可見 ADC、每發一次積分讀出、單一二階 IQ 軸，涵蓋 Pulse 與單 tone Mux。不支援 edge counting、多 tone、多 read、遮罩或額外矩軸。未 opt-in 的 IQ、分類、遮罩與原生取數預設不變。模擬在同一輪使用原有 raw source，只物化一次，不重跑 backend，不改普通 compact 路徑。只存平均而無矩的舊資料仍可讀，保存規則由 [[0069]] 擁有。

**依據。** 2026-11-22，designer 代裁，使用者 2026-11-23 追認（「都採用建議」）。比較、非零 offset 驗證及前例見 [Amp Rabi 的 shot moments](../discussions/2026-11-22-amp-rabi-shot-moments.md)。

#### Herald reset 與逐 shot 遮罩

**規則。** 只需篩選初態時，在既有 `reset` 欄位選 `reset/herald` 或 `reset/mux_herald`。兩者只保留 `keep` 指定的一類，不另設欄位或實驗入口，也不做 FPGA active reset。Herald 與終端發射共用 readout 設定。Module 或作者只註冊遮罩來源，`program` 的取數層在每個完成 round 算遮罩，與 round hook 同層。所有 ADC 共用同一張 bool `(points, reps)` 遮罩，沒有讀取軸。多個來源依註冊順序執行，以 AND 合併；沒有來源時全為 True，取數結果逐位不變。註冊位置與 owner 見 [[0083]]〈Builder 與 Schedule 的職責〉。取數時檢查索引未超出編譯後的讀取數，宣告與形狀見 [Registered shot masks](../../lib/zcu_tools/program/README.md#registered-shot-masks)。

**例子。** Mux 實驗以 `reset/mux_herald` 選 `keep: {'0': g, '2': g}`。鍵是完整 tones 清單的位置，不是 ADC 編號。只在兩個 tone 都是 g 時保留該 shot，所有 ADC 一起保留或排除。`reset/herald` 搭配 `readout/pulse`，`keep` 是單一態。可選 `prepare` 是一個 `pulse`、`reset/pulse`、`reset/two_pulse` 或 `reset/bath`，可 inline 或引用 library key。先 prepare，再做 herald 讀取、實驗序列和終端讀取。Herald 的 `readout` 可 inline 或引用 library key。Herald 沿用 readout 的 `pre_delay`／`post_delay`。共享 readout 物件的兩種建法見 [Herald resets](../../lib/zcu_tools/program/v2/modules/README.md#herald-resets)。

**邊界。** `ResetWithHeraldCfg` 才包含兩種 herald，普通 `Reset.Cfg` 不含。支援 herald reset 的雙 qubit 入口用 `reset/mux_herald`，不收只搭配 `readout/pulse` 的 `reset/herald`。T1 cov 不收 herald，遮罩會改變它用來自校準的聯合二階矩。需要保留多類時，見[保留獨立 heralded 實驗的條件](#保留獨立-heralded-實驗的條件)。

- Circle 可選 g、e、other，需 radius；linear 可選 g、e；不接受 centroid3。Mux `keep` 鍵須為非負十進位整數的字串，否則 cfg 驗證報錯。Build 拒絕與終端不同或有掃描的 herald readout、未知 tone、缺少 classifier 或 circle radius、非法 `keep` 態，以及各可行路徑消耗的實體讀取不一致。掃描會改變已校準 classifier 所對應的設定，所以不收 swept herald readout。
- 硬體觸發照實計數。Mux 的 herald 讀取包括未列入 `keep` 的 tone；每個觸發的 ADC 都移除這一次讀取。實驗用可見的 label 和索引，`shot_sums` 按邏輯讀取數檢查，不必改實驗的 converter。平均、reducer、`ShotResult` 不含來源消耗的讀取，沒有剩餘讀取的 ADC 也省略。`get_raw()` 仍含全部 trigger。`run_acquire_shots` 給可見讀取的原始 int64 積分 IQ 與 bool mask，不由框架套遮罩；實驗決定如何使用。
- 有遮罩來源時，普通平均在 host 算 `Σ m·x / N`。`x` 是逐 shot 長度正規化並扣 IQ offset 後的值，`m` 為 0 或 1，`N` 是 reps 乘已完成 rounds。接受率留在幅度，不另存。Reducer 收遮罩，`shot_sums` 只加保留的 shots。Herald reset 不存被排除的類別，也不改 Result 形狀與存檔欄位。
- 分類計數的分母是每格總計數，機率以通過 herald 的 shot 為條件。例：RB 的 joint00 是兩個 tone 都以 g 通過時的存活率。接受率可由計數總和除以 reps×rounds 得到，不另存。有限計數代表已量，少跑幾輪也可用，統計誤差按實際計數。未量格維持 NaN，已量格的計數總和為零時機率是 NaN，分析附原因。
- 帶遮罩來源時，decimated 取數會報錯，因為時間軌跡沒有可分類的逐 shot 積分值。`acquire`／`run_acquire` 有來源且收到原生 `threshold` 時也在進入點報錯。FTF 每次投影後從真空續演化。Herald 與終端讀取沿同一發的投影狀態 chain 取樣。有來源或 reducer 時逐 shot 抽 IQ，和硬體走相同的遮罩、平均與 reducer 路徑。

**依據。** 選項、實測與契約核對見 [Herald 改為 reset 模塊與逐 shot 遮罩](../discussions/2026-11-20-herald-reset-shot-masks.md)。

- 放置：2026-10-29，使用者說「那不做active reset，讓底層支援post-selection，上層使用新實驗組，比如heralding/*」。底層 post-selection 與不做 active reset 仍有效；上層放置已被取代，見[早期討論](../discussions/2026-10-29-heralding-post-selection.md#後記)。2026-11-19，使用者說：「H我自己的想法是，如果是reset模塊，那就放棄可以取得多餘的資訊，比如說把分類非g的shot值都設定成0來使其對平均沒有貢獻。對於想要利用更多資訊的實驗，保持分化是合理的。」2026-11-20，對整批建議答「好，我沒問題了，全部採用建議」。
- 取數層與多來源：2026-11-19，使用者說「我傾向統一的處理方式，統一知道權重的存在，而也代表權重的Mixin要在比較底層，作為通用底層功能存在，而不屬於Module層級。」2026-11-20 說「2我認為可以支援多次register_shot_weights，照順序執行，並合併mask，考慮這這個機制，我傾向mask先只有0/1就好，用int?目前沒想出用float的需求，看你的想法。」Bool 是 agent 的具體型別選擇，可直接用 `&` 合併，乘 IQ 時自動轉型。同日使用者問「關於mask，我擔心的是mux與多個ro ch的多次readout，據我所知，不同ro ch儲存欄位是分開的，那mask的形狀就需要討論一下。」對全 ADC 共用的逐 shot 遮罩答「M1」。
- 平均、計數與 shots：2026-11-19，使用者說「比如說把分類非g的shot值都設定成0來使其對平均沒有貢獻。」「由於掃flux本來就會正規化，所以都沒差……2我認為先不用存，但平均就能分析出來。」「平均公式可以調查一下設計張力，沒問題就採用。」扣 offset 後才置零，否則被排除的 shot 會留下負 offset。對條件計數的分母答「5a」，並說「run_acquire_shots我傾向遮罩，應該說，怎麼使用遮罩應該是各實驗的自由。」
- 隱藏讀取：2026-11-19，使用者說「3希望能設計的乾淨些，實作前請與我討論具體細節。」「6為什麼不支援run_acquire_shots？為什麼不支援tracker？作為一個獨立於processor的機制，我希望能保持彼此透明，如果不能，我希望瞭解原因。」2026-11-20 答「herald spec 的測試接縫和 contracts 我確認」。同日回答「B, 我傾向一次就設計完整。23可以」，其中「2」同意省略全被消耗的 ADC，以及有來源時拒收 `threshold`。
- Herald cfg：2026-11-19，使用者說「也許herald cfg可以設定要留哪種。」「關於reset/herald，post_delay不是已經在readout裡就有了嗎？pre_reset是為什麼需要呢？……mux與一般的readout要用同一個module type還是不同的？我以為通常都是不同的。mux的module通常都是特化。」2026-11-20 說「pre_reset應該可以不只是reset type，普通的qubit pulse或onetone pulse應該都可以達到穩定ge分佈的效果。」對只放一個 `prepare` 答「C，目前沒想到需要多個prepare的場景，那就先簡單做法。」同日對 JSON 相容的帶引號 `keep` 鍵答「2-1」。
- 模擬：2026-11-19 提出的統計生成方案，見[模擬抽樣比較](../discussions/2026-11-20-herald-reset-shot-masks.md#模擬器的統計抽樣提出後撤回)。使用者看過躍遷機率、偏差和逐 shot 抽樣耗時後，於 2026-11-20 答「shot抽樣就e吧」。

#### 保留獨立 heralded 實驗的條件

**規則。** 分析需要保存一類以上的 herald 結果時，使用獨立的 `heralded/` 實驗。它取得可見的 herald 與終端讀出，在取數時分類並保存聯合計數，不把 herald 讀出交給 reset 遮罩。GUI New Tab 把 `heralded` 與 `two_qubit` 並列。Registry key 跟 `zcu_lab/v2/` 下的模組目錄一致；同一模組有多個入口時，在目錄後加變體名。分組目錄也出現在選單，不另設例外。

**例子。** `heralded/rb` 放在 `zcu_lab/v2/heralded/rb/`，用四個 herald 類別估計每臂的共同衰減。`heralded/conditional_spectroscopy` 則在名目 I／Xπ 準備之後量 herald，按實測類別畫四支條件頻譜，不共用擬合參數。若只要選出 gg 再分析終端讀出，用支援 `reset/mux_herald` 的普通入口即可。

**邊界。** 家族規則不指定一種共用擬合，也不要求每類都進主估計。各入口決定用途與資料不足時的發布條件，見 [家族 README](../../zcu_lab/v2/heralded/README.md)。Registry key 不是 Native tag。搬移既有入口不改 Native tag、cfg schema 或 HDF5 映射；用新入口載入同 schema 的舊結果，不加舊 key、import 轉接或舊 schema reader。Lab 與各家族的依賴沿用 [[0072]]，本節不設 import 例外。

**依據。** 2026-11-24，使用者說：「如果是 reset herdering 的话，那就是真的就是只留一种。如果你想要全部保留，或是保留一种以上的，请独立的分化实验。」引文的「herdering」「分化」依語境指 heralding、heralded。對新增五個入口答「那就采用建议吧。」對選單與目錄一併改答「我会选 C two，我认为表单的结构应该要跟实际结构对应，或者说一致，这样才好维护啊。」對以目錄定 key、不為分組層開例外的提案答「採用建議吧，對於3，新名字你要叫什麼？」本次只處理 heralded 的 key。比較與先例見 [Heralded 獨立家族](../discussions/2026-11-24-heralded-family.md)。

### Owner 不變，caller 不重建舊配置

**規則。** `cfg_model` 擁有 Spec、cfg 驗證、引用展開與參數設定。`program` 擁有 native conversion、QICK 宣告與排程。`resources` 擁有 library 文件與 I/O。`gui.form_adapter` 從 core cfg 與持久 Spec 推導 Form；session 擁有 Library entry 表單。

**例子。** Pulse readout runtime 可把平鋪欄位投影成內部 `PulseCfg`，重用 waveform 與播放，caller 不重建 ADC cfg。Pulse cfg 沒有 `ro_ch`；runtime 接收 ADC matching 通道，`PulseRegistry` 用它區分硬體定義，避免 ADC 0 與 1 誤共用。Readout 在 module 起點送設定，在 `trig_offset` 觸發 ADC、在 `pre_delay` 播放 generator。結尾及 clock rounding 由 [program 的 readout 契約](../../lib/zcu_tools/program/v2/modules/README.md#readout-configuration) 與 [[0083]]〈Timeline 只存結尾，與 reference 分開〉擁有。

**邊界。** 不保留 editable shape catalog，不新增 registry、factory 或跨 owner resolver。Core/native factory 用 caller 捕捉的 resolver 與 asset capability，不查 process-global library。

**依據。** Readout 沿用 [[0072]]、[[0065]]、[[0073]] 的既有責任和 `.importlinter` C11、C16、C19 及 `form-adapter-no-host`，不另立 owner。分工的原始討論未入庫，見 Git `f647c9a9c`。

### 高階頻率與低階 words 分開

**規則。** `PulseReadout` 只收一個高階 `freq`。底層成對的 generator／ADC words 表示同一語意頻率，各按硬體 mixer／downconversion 規則編碼。

**例子。** 查表掃描將 `TableParam` 傳給 `PulseReadout`，QICK 逐點產生兩個 words，原生 loop 在收尾時更新，見 [Readout table sweeps](../../lib/zcu_tools/program/v2/modules/README.md#readout-table-sweeps)。

**邊界。** `freq_val` 與 `ro_freq_val` 是 runtime register 契約，不是持久 cfg 的兩個頻率。模擬的 accumulated、raw 和 decimated 共用平鋪 readout plan，不改數值單位或物理模型。

**依據。** 單一高階頻率沿用使用者 2026-10-15 的 [Readout 配置決定](#readout-擁有-generator-與-adc-配置)。查表掃描與底層的分工見 [[0083]]；原始討論未入庫。

### 表單與持久資料

**規則。** 共用 Form 生成器直接編輯平鋪 Readout；`gen_ch`、`ro_ch` 必填且可編輯，不推導、不補值。Module library 只在容器結構改變時升版本，不因 module 或 waveform 欄位改變升版。

**例子。** Inline readout 的頻率在 canonical Form input 用 `modules.readout.module.freq`，在持久 Readout Spec 用 `freq`。只有整個 module 和 waveform 的 inline input 加 wrapper，不使用舊 shape inheritance。Recipe 路徑省略生成的 wrapper，見 [Measure MCP 的 tab cfg 協定](../../lib/zcu_tools/mcp/measure/README.md#tab-cfg-協定)。編輯欄位及 waveform 引用見 [Form adapter](../../lib/zcu_tools/gui/form_adapter/README.md)。

**邊界。** 舊 nested Readout 與 Direct 條目不遷移、不加相容讀取。驗證入口指出 library key，作者重建條目。Artifact restore 的 typed cfg 失效及原始 data 可讀性由 [[0063]] 的 owner 負責。Library 保持 2.0，不加 readout 版本；實驗 cfg schema major 與宿主 wire／GUI revision 依各 owner 的不相容契約更新。Parameter container、native artifact 與 ledger 不因 Readout 欄位形狀改變外層格式。

**依據。** 2026-10-15，使用者問「只是改Module而已真的需要升Library版本嗎？就像加一個Component原件定義，不應該讓舊版本不能讀對吧」，並定案「1可以，2A，3保留2.0」。原始討論未入庫，見 Git `f647c9a9c`。

## 取捨

Readout 與獨立 Pulse 分別維護 generator 設定。Readout 因此只有一個頻率來源和一套表單路徑，runtime 仍可重用 Pulse 的播放。純 ADC 和獨立 ADC 頻率不在契約內。Mux 共用 generator 與時間欄位，不逐 tone 設 waveform 或時間窗。
