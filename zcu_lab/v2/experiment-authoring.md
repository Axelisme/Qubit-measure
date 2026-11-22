**讀者：** lab-author、framework-dev、agent。查閱 v2 實驗的 cfg、資料形狀、來源選擇、執行與保存契約。

寫或改實驗時，先從 [experiment-style.md](experiment-style.md) 選使用者範本。

# v2 experiment authoring

本頁是契約參考。操作教學見 [tutorial](../tutorial/README.md)，GUI adapter 見 [measure authoring](measure-authoring.md)。
`zcu_lab/v2/` 擁有使用 program/v2 的具體實驗，各 leaf 的 `core.py` 擁有 cfg、Result、run 與分析政策。
共同介面、RunRecord、保存映射與 cfg 組裝由 [experiment README](../../lib/zcu_tools/experiment/README.md) 定義。
GUI adapter 提出持久 ModuleSpec，由共用 Form 編輯，再交 Library owner 保存。
Core 接收執行用 typed cfg，與 GUI 表單、draft 分離。跨宿主契約見 ADR-0065、ADR-0008。

## 讓使用者只看一個檔案就能理解實驗

### 1. 在 core 直接寫出 pulse 序列與分析政策

序列與局部寫法依 [風格指南](experiment-style.md#2-照範本的形狀寫-corepy)。
Experiment core 擁有 run、analyze、逐 tone 計算、失敗隔離與分析結果型別。
GUI 擁有來源、route prior、writeback 與 operator guide，為 writeback 做的有限值檢查與摘要整理也留在 GUI。
共用 helper 限於驗證、整形、平均或明確參數化的數值工具。各實驗在本地決定序列與分析政策。
擬合 helper 接明確的 skip、數值 phase 或代表自由擬合的 None，政策選擇留在 caller。
各實驗的 analyze 以自己的 Result 呼叫共用數值工具，分析圖也在本地完成。

Mux 平行 drives 同時起跑，readout 等最長 branch 結束，短 pulse 保留自己的結束時間。
Echo core 明示跳過第 1 點，Ramsey core 依資料選固定 0°／180° phase。
壽命模型的跨次比較屬於 workflow，mux T1／T2echo core 只計算本次選定的模型。
Mux onetone 以 `Options.edelay_seeds` 接每 tone 種子。
Adapter 決定手動值、capture prior 與全域搜尋的優先序，記錄來源並決定 writeback。

### 2. 在 recipe 函式本體寫出操作流程

Recipe 主體依序開 tab、選 module、逐欄 set、套 overrides、check_ready、Run、Analyze，再詢問 writeback。
來源選擇與掃描端點留在主體，helper 只共用數字呈現、逐 Q 分析 updates 等機械對應。
接受答案後才呼叫 `tab.accept_writeback(decision.items)`。
Analyze 的 fitting 等例外交作者決定是否捕捉，未捕捉就使 recipe failed。
每次 Analyze 後先檢查 `analysis.result`，子層取消可能回 `result=None, status="failed"`。
此時回報 `analysis.reason` 並結束相依流程，跳過 Ask、接受 writeback 與相依 Post。此分支只處理取消。

Run 放在 try 外，成功返回後以既有 finally 同步 `tab.save()`。
根取消使下一個 await raise 普通 ScriptCancelled，進行中的 Run 先 drain 並保留 facts。
該 Run await 被根取消時跳過後續 Save finalizer，partial 留在 tab 交 agent 決定。
沿用此取消與保存流程，不新增取消分支或 Save finalizer。
Inputs 繼承 RecipeInputs，框架只作 pydantic／MCP JSON schema 驗證，recipe inputs 沒有 Form。
可省略欄位用 KEEP，null 原樣交 owner。讀寫都用 dotted string。
明填 scalar 前直接 command `to_inline`，保留 waveform family；已 inline 時保留原值。
宣告、Analyze 例外與取消契約見 [recipe 教學](../tutorial/04-recipe.md)。

### 3. Docstring 說明量測、序列與結果

Docstring 的篇幅與範本見 [風格指南](experiment-style.md#3-寫法判斷)。
Module 與公開函式交代量測、序列、輸入、結果的單位與 shape，以及必要的失敗條件。
Run 回傳量測資料，擬合值歸 analyze。操作教學、使用範例與環境指引放 leaf 的 `guide.md`。
必要的 lifecycle、rollback 與 caller 責任用一句說明並連到框架契約。

### 4. 用註解與空行分開階段

準備、序列、取數與平均的分段依 [風格指南](experiment-style.md#3-寫法判斷)。
Recipe 同樣按設定、量測、分析與 writeback 問答分段。
註解說明動作與原因，語言依 [CLAUDE.md](../../CLAUDE.md#先判定角色)。

## 寫法慣例

本節定義 cfg、存檔軸、分析摘要與來源契約。選範本與 core 結構見 [風格指南](experiment-style.md)。

### Cfg 名稱、順序與取數欄位

連續物理量掃描用 `SweepCfg`，離散整數序列用 `list[int]`。
整數是否可為零、是否須唯一，由各實驗的需求決定。
Cfg 掃描欄位用被掃量的單數名，如 `delay`、`gain`、`freq`、`depth`，硬體與 host 掃描都適用。
Result 用複數名，如 `delays`、`gains`、`freqs`、`depths`；沒有自然複數時用 `_values`。
`declare_sweep`／`add_sweep` 名稱與 cfg 掃描欄位相同。
只有單一計數 N、由它推導 0 到 N 的軸時，cfg 用 `n_<軸名>`，軸與 loop 用該量的名字。
此軸包含 0 與 N，共 N+1 點，`add_loop` 與 Result 使用相同軸名。

下表的五個一般入口，以及 conditional phase 和 RB 保留的獨立 heralded 入口，已將 cfg 掃描欄位改為單數並各自提升 schema major。
Result 欄位、儲存軸名與 label／unit 維持原樣。
這些入口載入舊 major 的 run 時保留 Result 與 metadata，cfg 為 None 並發出 `CfgUnavailableWarning`。
舊 cfg 不遷移、不提供 alias，也不支援重新分析，避免忽略舊 key 後靜默套用新預設值。
已移除的獨立 heralded 入口不載入舊 run。現有入口見 [two-qubit 實驗清單](README.md#two-qubit-實驗清單)。

| 實驗 | 舊 cfg 欄位 | 目前欄位 |
| --- | --- | --- |
| `two_qubit/conditional_spectroscopy`、`two_qubit/idle_zz` | `dc_values` | `dc` |
| `two_qubit/conditional_phase`、`two_qubit/local_z` | `gate_counts` | `gate_count` |
| `two_qubit/rb` | `depths`、`seeds` | `depth`、`seed` |

新實驗的軸 label 沿用同一個字，如 `Delay`。既有存檔 label／unit 保持不變以便載入舊檔。
Module 名稱等於提供設定的 cfg 欄位名，例如 `Pulse("qub_pulse", cfg.qub_pulse)`。
欄位名寫角色，如 `stark_tone`、`probe`，角色對稱時才用 `gain1`／`gain2`。
持久 cfg 改名時，明定歷史 cfg 的處置並更新 `cfg_schema_version`。

Cfg 依序放播放順序的 modules、外到內的掃描量、取樣方式、取數欄位。
Form 預設按 cfg 宣告順序顯示，adapter 有獨立輸入型別時按該型別排序。
只有需調整頂層欄位時才設 `field_order`，並在原地註明原因。
指定欄位按 tuple 順序前移，其餘保留相對順序，nested model 維持原樣。
`tree` 的寫入順序與畫面排序無關。未知或重複欄位在 Form 建構時 raise ValueError。
持久 Spec 的 Form 只接受空 `field_order`。
GUI 作者依 [表單設計與預演](measure-authoring.md#表單設計與預演) 核對實際順序。

時間掃描需選取樣方式時用 `sampling: Literal["linear", "decay"]`。
Linear 是線性格點，decay 將同一時間窗的點集中在衰減段，這類時間掃描只接受這兩種模式。
各實驗保留原取樣預設，既有 `uniform` 欄位遷移時改名。
取數欄位在每個 cfg 最後完整宣告，保持欄位可見。
平均類建議提供 `reps`、`rounds`、`relax_delay`，有 SNR 定義才加 `earlystop_snr`，需要才加 `initial_delay`。
這是建議，缺少其中一項本身不阻擋 review。預設值與欄位 description 依 [風格指南](experiment-style.md#3-寫法判斷)。
`acquire_retry` 只開放超長掃描 `twotone/fluxdep` 與 `singleshot/t1/t1_with_tone_sweep`。
Workflow 從自己的 settings 傳 retry 給取數 helper，core cfg 不承接 workflow retry。

Readout 用實驗接受的最窄型別：pulse 用 `PulseReadout.Cfg`，mux 用 `MuxReadout.Cfg`，兩者都支援才用 `Readout.Cfg`。
其他 module 同樣用自己的 `.Cfg`。宣告順序參考 [`AmpRabiCfg`](twotone/rabi/amp_rabi/core.py)。
Module 建構子的 keyword 覆寫產生有效 cfg，caller cfg 保持不變，None 表示沿用原值。
Pulse 支援 freq、gain、phase、length；PulseReadout 支援 freq、gain、length、ro_length。
Reset 參數見 [module 建構參數表](../../lib/zcu_tools/program/v2/modules/README.md#constructor-overrides)。
Waveform 的 length sweep 限制仍適用，MuxReadout 不接受覆寫。
`ro_length=TableParam(...)` 在 PulseReadout 建構時 raise NotImplementedError。
線性 ADC window sweep 保留 QICK 編譯限制。`set_param` 會修改接收呼叫的 cfg，只對自己持有的副本使用。

### v2 cfg 組合慣例

Cfg 從 `zcu_tools.cfg_model` import `ConfigBase`，modules 與掃描欄位平鋪在頂層。
Module 保留 selector 與子表單，掃描保留 SweepCfg 子表單，FieldMeta 提供 label、unit 等提示。
Pulse Advanced group 保留 NQZ、Phase、Pre-delay、Post-delay、Mixer freq 的既有摺疊。
Tone lists 保留逐元素結構。只宣告實驗會使用的取數欄位。
Initial delay 需編輯或驗證時才宣告，否則沿用 program 的 1.0 µs。
平鋪 cfg 使用 `cfg_schema_version="3.0"`，後續契約變更依各實驗提升版本。
Load 保留舊 major 的資料與 metadata，回 `record.cfg=None` 與 `cfg_issue`，不轉換舊巢狀 cfg。
Core 按需 import 實際使用的 module、SweepCfg 與硬體指令。
父層型別與執行前組裝見 [父層 README](../../lib/zcu_tools/experiment/README.md#cfg-與父層支援)。

### 存檔軸用關鍵字，固定座標不塞進 Result

`Axis`／`Z` 在 field_name 後用 keyword 寫 label、unit、scale、dtype，axes 是 inner-first。
Scale 定義為 `disk = memory * scale`，MHz→Hz 用 1e6，us→s 用 1e-6，原值保存省略 scale，預設 1.0。
記憶體保留習慣單位，盤上頻率與時間用 SI 單位。
固定類別軸用 `values`。field_name 與 values 必須恰給一個，否則建構時 raise ValueError。
Z 各維度須匹配軸長，固定軸也計入；load 不將固定值寫進 Result。
`GroupedAxesSpec` 的 `VariableAxisSpec` 全用 keyword，`VariableZSpec` 只有 field_name 可用位置參數。
`VariableAxisSpec.generated_arange(label=..., unit=...)` 也用 keyword。
可選 native 變數與 Labber 選集各自宣告，見 [ADR-0069](../../docs/adr/0069-experiment-core-and-frontends.md#native-可選變數與-labber-投影)。
單變數映射見 [PowerExp.AXES_SPEC](twotone/power_dep/core.py)，二維硬體掃描見 [FreqGainExp.AXES_SPEC](twotone/ro_optimize/freq_gain/core.py)。

Native HDF5 的變數名（`VariableSpec.variable` 或 `AxesSpec.data_variable`）與存檔軸的 `label` 都是結果格式的一部分，不是圖上的標籤。Cfg 改名與舊檔相容性見 [ADR-0069〈Cfg 欄位改名與 schema 版本〉](../../docs/adr/0069-experiment-core-and-frontends.md#cfg-欄位改名與-schema-版本)。現行例子見 [Two-qubit 持久名稱](README.md#two-qubit-持久名稱)。

### 分析選項自行驗證，fit 結果用屬性讀取

分析選項用 pydantic dataclass 的欄位限制驗證，GUI 沿用此驗證。
負 skip 在建構時拒絕，GUI 顯示表單錯誤，remote 回 `INVALID_PARAMS`。
實例見 [`AmpRabiAnalyzeOptions`](twotone/rabi/amp_rabi/core.py)；新增檢查的授權依 [風格指南](experiment-style.md#3-寫法判斷)。
`fit_decay`、`fit_dual_decay`、`fit_rabi`、`fit_decay_fringe` 回傳具名結果。
新程式碼讀屬性，tuple 解構僅保留既有 caller。`fit_decay` 的時間與誤差沿用輸入時間單位。
輸入 delays 與 real_signals 是同長一維 float64 陣列，delays 非負。
讀屬性的範本見 [`AmpRabiExp.analyze`](twotone/rabi/amp_rabi/core.py)。

### Analysis 回傳值只放摘要

Analysis 大小最多 O(log N)，N 是掃描點數或 shot 數。
回傳擬合結果、純量、固定小值，或每 tone／qubit 一筆的結果。
隨 N 成長的陣列留在 analyze 畫圖，只發布圖時回 None。
GUI 在 primary、interactive、post-analysis 發布前呼叫 `summarize_analysis`，存檔與 RPC 也沿用此轉換。
任一 ndarray、list、tuple 攤平後超過 64 元素會 raise ValueError，GUI 顯示該次分析錯誤。
訊息包含欄位路徑、元素數、上限與不能隨掃描大小成長的規則。
64 是檢查上限，較短的 O(N) 陣列仍違反摘要契約。

### 來源選擇與分類 read axis

在 `make_form(ctx)` 用 `source(ctx, path, template)` 選來源，path 可為 module slot 或 library root，支援 `$alias`。
Helper 查固定 capture 的 `ctx.has(path)`，存在時回保持 reference 的 Slot，既不 copy 也不附加 set。
缺 capture、key、未設定或懸空 slot、缺或未綁 alias 都回 False，source 原樣回 Template，保留 set 與 seed_metadata。
路徑語法、其他 alias 解析錯誤與來源故障直接傳出。
來源可解析但 family 不相容時，由 Form 報錯，不能退回 Template 掩蓋錯誤。
Metadata 寫在 Template，只有採用它時 Form 才合併 metadata。
Key 用 Form root-relative canonical path，同路徑以 `Form.with_seed_metadata` 顯式值優先。
只有 Slot、沒有 Template 時，缺來源的 required 欄位保持 Invalid；nullable 且預設 None 的欄位保持停用且有效。

兩分支共用 set 時，對選出的 node 做一次 `dataclasses.replace`，合併原 set。
Run 會覆寫 waveform.length 時，可在合併的 set 放 `"waveform.length": Hidden(0.0)`。
Slot 與 Template 都有 set，沿用 source 的三參數介面即可。
Slot.set 初值只套用到 copy，這項設定本身不把 reference 改成 copy。
來源 helper 契約見 [ctx_helpers.py](support/measure/ctx_helpers.py)。
單 tone 分類用 `Classify("readout")`，編譯後由 `shot_sums(prog, [axis])` 解析 label，自動選 pulse 或單 tone mux。
多 tone mux 明寫原始 tone 序號，例如 `Classify("readout", tone=2)`，省略時錯誤列出可選 tones。

### 圖的建立、顯示與更新

建圖、預設樣式、show、owner-thread 更新與 finish／release 依 [plotting README](../../lib/zcu_tools/plotting/README.md)。
Live artists 與多面板格式依 [v2 README](README.md#即時圖工具)。
建立非均勻熱圖時，在 `live_heatmap(..., uniform=False, cmap="viridis")` 傳入 Matplotlib 已註冊的色盤名稱。省略時用 `RdBu_r`。在 placeholder 載入前選定色盤；第一次 `set` 後不能更換。
實驗自行決定座標、labels 與圖名。show 前可填資料，show 後的修改也經 plot-thread callback。
選 y 軸要畫的物理量時看 [R12：y 軸畫物理 observable](figure-style.md#r12y-軸畫物理-observable)，標示方式見[〈單位〉](figure-style.md#單位)。
一維建圖與 update 見 [`AmpRabiExp.run`](twotone/rabi/amp_rabi/core.py)，熱圖與最近兩條切線見 [PowerExp 的 live plot](twotone/power_dep/core.py)。

## 目錄佈局

Leaf 位置與附件見 [套件 README](README.md)。量測核心與 GUI 分離，import 邊界依 [風格指南](experiment-style.md#4-自查)。
跨實驗 singleshot 數值 helpers 放 `support/singleshot/`，reset fit 與 JPA optimizer 留在各自 leaf。
通用 runtime 與量化工具放框架。儀器角色 helper 位於 `support/device.py`，見 [device 操作](README.md#實驗角色與-device-操作)。

## 實驗特有的分析與保存

共同 RunRecord、explicit source 與具名圖契約見 [父層介面](../../lib/zcu_tools/experiment/README.md#實驗介面與資料)。
Run 以 explicit RunContext 取得硬體與 devices，回純 Result，由 adapter 配 cfg／Result 成 RunRecord。
同步 analyze 接 source、typed options 與本次 Plots，數值直接回傳，圖發布到本次 Plots，跨次成功狀態由 record 表達。
有分析選項時以 `Experiment.Options` 引用既有 options 類別，GE 的 post-analysis 另有 `GE_Exp.PostOptions`。
類屬性保存型別而非共用 instance，欄位與預設由原 options 擁有。無選項時省略 Options，基底不要求它。
各 leaf core 定義輸入與結果型別，guide 定義操作與 writeback。
Mux zip acquisition、participants、座標與 partial data 見 [v2 README](README.md#mux-qubit-numeric-cores)。
Readout power 的八分支 QND、native tag、指標與 gain 選擇見 [Power guide](twotone/ro_optimize/power/guide.md)。

### CKP 與 reset 診斷

CKP 從 ground／excited maps 抽 resonance traces，用 `analysis.fitting.shared` 共擬合 Lorentzian baseline、scale、width。
兩個 resonance frequency 各自 local，兩個 slope 固定零。
Chi／kappa uncertainty 從具名 global covariance 投影，包含 local frequency cross-covariance，保留完整 covariance。
Backend minimum 或 covariance 無效時 fast-fail。CKPAnalysis 只含 chi、kappa、res_freq，圖交本次 Plots。

`twotone/reset/RabiCheckExp` 保存三分支 gain sweep，run 回純 Result。
Analyze 接 RunRecord 與 None options，接受 cfg=None，回 RabiCheckFit 並發布 fit。
Reset 前分支決定共同 IQ 投影與基頻，reset-only 用同基頻，reset 後追加 Rabi 分支包含基頻與二次諧波。
半峰對峰振幅、phase、相對 contrast 與殘差只描述平均 IQ，不代表 reset fidelity。
數值由同目錄 `fit.py` 擁有，GUI 只投影純量。

`singleshot/ResetCheckExp` 的 `ResetCheckResult.counts` shape 是 `(Ngain, 3, 3)`，依序為 gain、reset branch、G/E/Other。
三 branch 是 reset 前、reset-only、reset 後追加 Rabi，屬獨立 preparations，不是配對 shot 軌跡。
Counts 累加已取得 rounds，analyze 先正規化，再選用 `ResetCheckAnalyzeOptions.confusion_matrix` 修正。
回傳 reset-only 的平均／最差 excited population 與最大 Other population，發布 populations 圖，不提供 IQ 校準 writeback。
Other 不等於校準後 leakage，population 不等於 reset-channel fidelity。
Run 回純 Result，analyze 接 RunRecord 與 ResetCheckAnalyzeOptions，回 ResetCheckAnalysis。

### Singleshot raw IQ 與分類

Amp Rabi 保留硬體 gain sweep，每個 host round 取所有 gain 的 Reps 筆 raw IQ，跨 round 串接 shots。
Canonical complex128 shape 為 `(Ngain, Reps * Rounds)`，shot_indices 是 inner axis，gains 是 outer axis。
完整 round 立即更新 live populations，取消後未完成 round 保持 NaN。
Analysis 只移除完全缺失的 shot columns，部分缺失的 sweep 明確拒絕。
Live preview 用 saved readout circle／radius，缺 classifier 時略過 preview；raw acquisition 與 joint fit 可在未校準時執行。

Len Rabi 用 host `Schedule.scan` 逐 length 取 shots，再寫入 `(Nlength, Nshot)` complex128 row。
此安排限制 FPGA raw buffer 為單一 length。Stop 保留完整 rows，其餘維持 NaN。
Amp／Len 共用 `analysis.fitting.singleshot_rabi` raw-IQ joint likelihood，各自在 core 畫 population、histogram、confusion matrix。
Amp 固定無衰減／零 phase，Initial State 描述 zero-gain 前主要狀態，支援 signed gain。
Amp 從 joint frequency 與 covariance 推導 pi／pi2 gain 與誤差，回 AmpRabiFit；Len 回 RabiJointFitSummary。
兩者 readout 診斷只讀 source.cfg，GUI 校準 writeback 共用 backend validity 與有限校準值檢查。
Analysis 不用外部 confusion matrix。舊 population-only 檔缺 raw IQ，canonical loader 明確拒絕。

Check 在 Analyze 才要求 saved cfg 的 circle／radius，只發布分類 scatter，回 None。
AC Stark 回 AcStarkAnalysis，另有 explicit source／AcStarkPlotOptions 的 population 圖入口。
Ground／Excited／Other 即時熱圖與 current trace 放本次 measurement 圖的面板。
其餘分析沿用各自 nullable cfg 契約。
分類 consumers 從 RunRecord.cfg.readout.classifier 取得參數，build 後解析 Classify 再交 `run_acquire_summed`。
Counts 保存各 round 完整計數總和，live 與 analyze 才除以每點總計數，analysis 保留取得的 Other。
需要 Other 時要求 circle 與 radius，linear／centroid3 無法表達 Other。
舊比例資料缺 counts 時拒絕，不能由比例推算計數。
GUI confusion correction 只讀 saved readout matrix，缺 cfg／classifier／matrix 時略過。
Rabi 與 Check 保留單 ADC raw IQ，Rabi 無 classifier 仍可 Run／fit。
Classifier 複數持久格式由 cfg_model 與 core complex codec 擁有。

### Singleshot initial state

GE、Len Rabi、Amp Rabi 的 Initial State 只影響分析。
Radius 自動搜尋上限為 g/e 中心距離，手動分類可用更大 radius。
Raw-IQ projection likelihood 與 radius 無關，derived confusion／population 用圓內且最近中心的互斥區域。
GE `consider_other` 只控制 radius 選擇的 midpoint-other 懲罰，校正矩陣保留 isolated-other row。
數學、零座標外推與 joint likelihood 見 [analysis owner](../../lib/zcu_tools/analysis/README.md#singleshot-初態與-joint-fit)。
操作、phase offset 與 writeback 見 [GE](singleshot/ge/guide.md)、[Len Rabi](singleshot/len_rabi/guide.md)、[Amp Rabi](singleshot/amp_rabi/guide.md)。
一般 Len Rabi phase 見 [leaf guide](twotone/rabi/len_rabi/guide.md)，fringe phase 見 [Ramsey](twotone/time_domain/t2ramsey/guide.md) 與 [Echo](twotone/time_domain/t2echo/guide.md)。

### 版本實驗的資料形狀

以下軸序指 Result-native shape，spec axes 仍是 inner-first。
Grouped CPMG variables 是 lengths／signals，axes 是 Time Index、Number of Pi，lengths 在盤上為 s、記憶體為 us。
RO auto-optimize variables 是 readout_freq／readout_gain／readout_length／snr。
JPA auto-optimize variables 是 jpa_flux／jpa_freq／jpa_power／jpa_phase／snr，phase 為 integer index。
Jpa_flux 是 device-native value，unit a.u.、identity scale；strict loader 拒絕舊 grouped variable 的 unit A。
Grouped 頻率與時間在盤上為 Hz／s，typed loader 回復 MHz／us。
`onetone/flux_dep` canonical axes 是 `(freqs, values)`，signals shape 是 `(Nflux, Nfreq)`。
舊 Labber Frequency MHz/Hz、Yoko flux 軸、ADC unit channel 由 [離線 migrations](../../migrations/README.md#legacy-resultdatabase--native-storage) 轉換。
Runtime／GUI 僅載入符合 spec 的 native 資料，legacy npz、sidecar 與 Labber 匯出不屬 native 格式。

Bath reset freq-gain 的四點 pi/2 tomography phase 是第三 sweep axis，shape 為 `(4, Ngain, Nfreq)`。
Length 的 phase 是第二 axis，shape 為 `(Nlength, 4)`。
CKP 的 prepared state 是 initial_states，GE 是 prepared_states，GE raw IQ shape 為 `(2, Nshot)`。
分類 counts 的 population_states=[0, 1, 2] 是最內層軸，保存完整 g/e/other：

| 實驗 | counts shape | 其餘軸，外到內 |
| --- | --- | --- |
| MIST power／freq／pre_freq | `(Nsweep, 3)` | gain／frequency |
| Singleshot ac_stark、MIST power_freq | `(Ngain, Nfreq, 3)` | gains、freqs |
| Singleshot t1／t1_with_tone | `(Nt, 2, 3)` | lengths、initial_states |
| Singleshot t1_with_tone_sweep | `(Nx, 2, Nt, 3)` | xs、initial_states、lengths |

舊兩欄 population 與新 analyze 不相容，不提供轉換 adapter。
`cfg_schema_version` 只標記 cfg 契約，不當作 Result 資料版本。
現行 AC Stark／MIST 為 4.0，T1／T1WithTone／T1WithToneSweep 為 5.0。
Homogeneous Result 保存單一 native data.h5，Labber 另匯出單一 .hdf5；離散狀態是 axis，不拆 sidecar variables。
Runtime 不載入 legacy artifact，也不重排舊 `(2, Nsweep)` population HDF5 或 multi-sidecar z 方向。
MIST Freq／Power／PreFreq／FreqPower 回純 Result，分類校準屬 typed cfg。
其 analyze 接對應 options，允許 cfg=None，只發布 fit。Power 支援 photon 軸與 log scale，FreqPower 只提供已實作的軸選項。
單 sweep 與二維 Ground／Excited／Other 面板發布為 measurement，Population 表示 counts 正規化的三態比例。
Singleshot T1／T1WithTone／T1WithToneSweep 用 typed sampling，analyze 允許 cfg=None。
Tone 回 t1／t1_b，其餘只發布 fit，保留 transition-rate fitting、host rounds 與完整 outer-row 篩選。
T1／Tone measurement 含 ground／excited traces，Tone sweep measurement 含六張 population 熱圖與兩張 current traces。
格式細節見 [datafile README](../../lib/zcu_tools/datafile/README.md)。

## Timeline program 與 Schedule 取數

TimelineBuilder 記錄動作，build 產生並編譯 TimelineProgramV2；它不接 soc、不取數，也不是 context manager。
自訂 register 或 dmem 操作時寫 TimelineProgramV2 子類別，兩種 program 都可交 Schedule／step 的 `run_acquire*`。
SimulationSoc 從 compiled stream 建 acquisition request，編譯與 IR timing analysis 可獨立執行。
離線取數需對應的 [simulation environment](../../lib/zcu_tools/program/v2/simulation/README.md)。
`delay_by_timeline` 按逐點 runtime 結尾推進 reference，shell 收尾 WAIT 用編譯期常數。
合法點的已知剩餘時間差大於零時 warning，超過 relax_delay 時 raise ValueError；未知差距只警告上界。
兩種 program 共用 [ADR-0083 收尾 WAIT 契約](../../docs/adr/0083-program-macros-and-timelines.md#收尾-wait-與-slack-護欄)。
Scheduled timing analysis 不驗證 CPU 等待或跨 shot 實機排程。

### Builder 與 module 的生命週期

一維序列與取數見 [`AmpRabiExp.run`](twotone/rabi/amp_rabi/core.py)，以 branch 比較 g/e 的二維掃描見 `twotone/ro_optimize/freq_gain/core.py` 的 `FreqGainExp.run`。
Builder 明傳 reps、relax_delay 與需要的 initial_delay，取數時另傳 rounds。
`add_sweep` 同時宣告 loop 並回 QickParam，獨立成行，loops 按宣告順序外到內。
`add_loop` 搭 LoadValue 用於整數 selector 或候選 pulse 選擇。
Emit、delay、timeline、repeat、branch 等方法沿用 [program API](../../lib/zcu_tools/program/v2/README.md#原生寫法與具名-timeline)，布林控制選項用 keyword。
Build 成功後 builder 凍結，再錄製或 build 都 raise ValueError。
平行序列先 fork、依作者順序 emit、再 join；join 只合併結尾，reference 保持原位。
TwoPulseReset／BathReset 的 child 與 branch case 不能推進 reference。
Timeline 範圍只切換預設名字，reference 與控制流契約見 [ADR-0083](../../docs/adr/0083-program-macros-and-timelines.md#timeline-只存結尾與-reference-分開)。

手寫 program 在 `_initialize` 建 module 並 declare，在 `_body` emit，需持有 cfg 時自行 deepcopy。
同一 module 物件可跨 builder／program 重用，各 program 依物件身分保存宣告並在 compile 配置自己的資源。
錄製或編譯後保持 module 與 cfg 不變，不同掃描設定建不同物件。
同 program 多次 emit 同物件只宣告一次，每次時間與 tag 獨立，重複 readout 用不同 tag，如 herald／terminal。
不同物件重複占用 ADC／register 時，編譯指出兩個物件；Pulse 家族既有 generator／template 共用規則仍適用。

### Sweep 參數與 cfg 的歸屬

掃描值透過 module constructor keyword 傳入。Schedule／step 不持有 cfg，需修改時在 run 自行 deepcopy。
`sweep2array` 只讀 sweep 與 channel，建立座標與 pulse mutation 無關。
均勻掃描用 add_sweep，非均勻用 add_table_sweep，兩者建立頂層 loop，重名 raise ValueError。
查表回 TableParam，單位由 consumer 決定，如 MHz、us；空表或非有限值 raise ValueError。
同 loop 的另一張表用 `TableParam(name, values)`，表長與 loop 不符在 build raise ValueError。
TableParam 可與數字或未掃描 QickParam 作正反向 `+ - * /`，產生同 loop 新表。
與任何另一掃描參數運算都 raise TypeError，包含同 loop 的另一張表。
每圈結尾載入下一點，即使該發沒播放 pulse，最後回第一點供外層 repetition 使用。
Const／flat-top length 可查表，flat-top 只改 flat 段；其他 waveform 限制仍適用，MuxReadout 不支援查表。
時間 getter 在量化後逐點值為常數或仿射時回等值 QickParam，runtime-only／非仿射時 raise RuntimeError。
具名 timeline duration 逐點 ceil，原生 QICK 轉換保留 nearest。
查表每點硬體字等於該值的純量 program，線性浮點值回編 gain 等截斷欄位可能差 1 最小單位。
比較 linear／table parity 前，以同 profile 確認各點純量字與線性字相同。
轉換入口與值型別限制見 [program 查表掃描](../../lib/zcu_tools/program/v2/README.md#查表掃描)。

### Host loop、device setup 與時間取樣

Host loop 用 sched.scan／repeat，program loop 用 add_sweep／add_loop。
Host 掃描與逐列取數參考 [PowerExp.run](twotone/power_dep/core.py)。
需重用各點 program 時，run 維護 dict cache，miss 才建新 builder 並 build，每次仍由 step 取數。
Device setup 留在 run，先 set_*_in_dev_cfg，再用 setup_devices 初始化掃描起點。
傳 context.devices、progress=True、cancel_signal=context.scope.stop，helper 到 driver 邊界才轉 Event。
Progress=True 已提供通知，直接 acquire 路徑也須傳 cancel flag。
Raw-shot GE／Check／Len Rabi／Mux GE cfg 以 shots 宣告取數量，builder 用 reps=cfg.shots，取數固定 rounds=1。
有 initial_delay 時明傳，這些 runtime 固定值留在 run。Lookback 固定 reps=1，rounds 仍由 cfg 編輯。

Onetone freq 的 linear 用 SweepCfg，homophasal 從 fitted resonator circle 產生非均勻頻率表。
兩者保留同一 freq UI、存檔 freqs 軸與 cfg，由 PulseReadout 覆寫 freq。
QICK 逐點轉 generator／readout words，loop 每發後載下一點，末點回第一點，DDS phase accumulator 保持連續。
非均勻點先 round 到硬體格點，相鄰 collapse 時 fast-fail。
T1 family 用 `experiment.v2.utils.t1_sampling` 沿 normalized decay curve 等弧長配置 delay，保留時間窗與點數。
各路徑依 delay 或 probe 的 grid 量化，collision／非嚴格遞增直接失敗，保留原點數與順序。
Direct delay list 沿用 conversion contract，不重採樣。
Singleshot Tone 的 decay 路徑在 host 逐點執行，zero-length 略過 probe；linear 路徑在板上執行。
Tone sweep 在 device setup 前要求所有時間嚴格大於零，linear／decay 都以 TableParam length 在板上掃描。
外層 gain／frequency 留在 host，readout 對齊各點實際 pulse 結尾，短點保留其時長。
Mux Len Rabi 同 point loop 中每個參與者有自己的 length 表。

### 用 run_acquire* 寫入與停止

SignalBuffer 接完整 array 的 on_update，Schedule 接 scope=context.scope 並把轉換資料寫入 slot。
一維接線見 [`AmpRabiExp.run`](twotone/rabi/amp_rabi/core.py)。它把同一份校正 IQ 矩用於 live update、SNR 停止條件和最終結果。
Buffer、retry、partial 與 lifecycle 見 [runtime README](../../lib/zcu_tools/experiment/v2/runtime/README.md)。
Integrated、summed、decimated 三個 slot 入口用 RunOptions(converter=...) 轉形狀或算 conditional mean。
Summed 先 build，再 `shot_sums(prog, [IQ("terminal")])`，將 `[N, ΣI, ΣQ, ΣI², ΣIQ, ΣQ²]` 寫入 float64 slot。
若與 native 平均 IQ 共用零點，且符合單 ADC、單積分讀出與無 mask 的前提，傳 `IQ("terminal", offset_corrected=True)`。
分類軸、校正的前置條件與 split_sums 見 [summed acquisition](../../lib/zcu_tools/program/README.md#summed-acquisition)。
多 ADC mux 須提供符合 slot shape 的 converter；`MuxReadout.collect_raw_iq(prog)` 在 completed-round hook 讀 raw buffers。
逐 shot IQ 必須來自 raw buffer，不能以累積 averaged IQ 代替。
Decimated 不用 sweep，caller 按 rounded readout window 配 trace buffer，shape 匹配 backend repetitions／samples，可附 on_update。
Readout 設定不變可重用 module，各 program 仍各自持有宣告。

`run_acquire_shots` 要已進 context 的 Schedule，不需預建 buffer，只取一輪，不轉換或更新 live。
它回傳 `ShotResult`。從 `.shots[adc_channel]` 讀取 `(points, reps, reads, 2)` 的 int64 IQ；讀取軸不含遮罩來源消耗的讀取。
`.mask` 是 `(points, reps)` 的 bool 陣列。取數不刪除或歸零被拒的 shot，由實驗決定如何用遮罩。
Rounds 省略或為整數 1，其他值 raise ValueError；多輪由 host 重複呼叫。
接受 RunOptions 的 retry、progress、on_stop，converter／stop_condition 非 None 時 raise ValueError。
成功取數後沒有 raw buffer 時 raise RuntimeError；外部取消回 None，且不讀 raw buffer。
Acquire-only cancel 不改父 outcome，下次 acquire 可繼續；只有父 stop token 已設才記 stopped。
Producer 失敗用同 retry 預算，raw-shot 無 slot 還原；raw-buffer 讀取失敗直接傳出。
SNR early stop 用三個 slot 入口的 `RunOptions(stop_condition=snr_checker(...))`。

Program.acquire* 只取數回傳。直接呼叫時作者承接以下責任：
- 傳 `cancel_flag=sched.stop` 或 step.stop。
- 接 StoppedPartialAcquireError，決定 set_stop 與未完成資料的處理。
- 自行管理取消 scope，直接呼叫沒有 acquire 子 scope，scope.cancel("acquire") 停不到它。
- 自行寫 retry；Schedule runner 只在 run_acquire* 代管取消轉譯、slot、live update 與 retry。
直接取數仍是作者可選路徑。外部取消、partial、retry 與 on_stop 見 [Stop 與 Retry](../../lib/zcu_tools/experiment/v2/runtime/README.md#stop-與-retry)。

## Signal/Raw 處理慣例

逐 shot 保存限於校準／分布分析：singleshot GE、Mux GE、qutrit_population、Amp／Len Rabi、Check。
其他實驗在取數時分類，Result 存計數或總和；校準實驗保留 raw 分布供擬合與驗證。
QICK raw 通常為 list[ndarray]，首元素 shape 為 `(nro, ..., 2)`，尾軸是實數 I/Q。
`default_raw2signal_fn` 位於 runtime.acquisition，預設 ADC／read 限制依 [runtime owner](../../lib/zcu_tools/experiment/v2/runtime/README.md#schedule-run-acquisition)。
Integrated／decimated converter 收每輪累積與最終 raw，回符合 slot shape 的 ndarray。
Live signal2real 可在本地定義或取 lab support，magnitude／phase／real 由實驗選，analyze 可選不同轉換。
座標工具可回 scalar 或 ndarray，後續需 indexing、min/max、軸對齊時，在邊界用 np.asarray(..., dtype=...) 正規化。
SNR／map argmax 的 ro-optimize／reset 預設 smooth_method="wavelet"，gaussian 留作舊結果對照。
Smooth 對 Gaussian 是 sigma，對 wavelet 是 threshold scale。
RO length 的 `SNR/sqrt(length + t0)` 用 t0 作 duration normalization，正 t0 啟用短 readout bias，越小的正值 bias 越強。
None／0.0 都是純平滑 SNR argmax。

## v2 工具與實驗輔助

Sweep 座標參考 [`AmpRabiExp.run`](twotone/rabi/amp_rabi/core.py) 的 `sweep2array` 呼叫，量化契約由 [round_zcu.py](../../lib/zcu_tools/experiment/v2/utils/round_zcu.py) 擁有。
`round_zcu_*` 處理單點，`round_sweep_dict` 處理規則 sweep，`sweep2array` 展開座標。
目前 time／frequency 依 QICK start/span 與 signed-step 量化，gain 將 start／derived step 各自截斷成 words。
量化後多點 step 為零或相鄰碰撞時 fast-fail，要求加大 span 或減少 expts。
`merge_result_list` 將 list[dict[name, ndarray]] 遞迴堆成 dict，原外層 list 成新最外維。
ResultTree.measurement_result(name) 只 merge 並 cache 該 measurement，其他 measurement 的更新互不影響。
`snr_checker` 用 `estimate_snr` 判斷 sweep 曲線是否達到 early stop 門檻。
RO 與 JPA optimization 用 `run_acquire_summed` 與 `shot_sums(prog, [IQ("readout", order=3)])`。
把 `snr_as_signal` 接到 `RunOptions(converter=...)`，即時圖與最終結果都使用已完成 rounds 的總和。
`snr_as_signal` 預設 g/e 是第一軸，只回傳剩餘的掃描軸。`skew_penalty=0` 回傳 pooled-sigma SNR，正值降低投影寬度不等或合計偏態較大的候選分數。
Comment 與 sweep helpers 見 [父層 utils](../../lib/zcu_tools/experiment/README.md#cfg-與父層支援)。

## Lookback

Lookback Result 是 us time axis 與 complex signals，cfg 由 RunRecord 保存，run 用 decimated Schedule。
LookbackAnalyzeOptions 預設 ratio=0.3、smooth=None，先對 complex signals 作 Gaussian smoothing，再從 magnitude 峰向前找最後低於門檻的 sample。
Analyze 可接受 cfg=None，回只有 predict_offset 的 LookbackAnalysis，發布 fit；預設 saver 拒絕缺 cfg。
Time 盤上仍為 seconds，Notebook 使用共用 NotebookAdapter，以 RunRecord 保存結果。

## Twotone RB 測量策略

`twotone/rb` 同時涵蓋 reference RB 與 IRB。`target_gate=None` 只跑 reference，arm 軸長度為 1；選 X90／X180／Y90／Y180 時另跑 interleaved arm，arm 0=reference、1=interleaved，使用校準 X90／X180。
兩種模式都保存 `(seed, arm, depth)` IQ。

每個 seed／arm 一個 program，板上以 sweep index 遍歷 depth 與 reps。Host 依 seed 跑 reference，再跑可選的 interleaved，分別呼叫 `run_acquire(..., rounds=cfg.rounds)`。取數回傳已完成 rounds 的平均 IQ；沒有已完成 round 的格子維持 NaN。
Recovery 是 24-element Clifford group 的完整 inverse，以 6-state permutation 程式化生成 CAYLEY／INVERSE_INDEX。
`CAYLEY[i][j]` 是先 C_j 再 C_i，累積為 `acc = CAYLEY[next][acc]`，兩處註解同步維護。
Random prefix 長度與 recovery id 用 LoadValue 從 dmem 查，random 段用 register-driven repeat，固定 n="rand_len"。
空 random 序列交 LoadValue(values=[]) no-op。
Recovery 最多兩個 physical pulses，用 recovery_gate_0／1 兩個 ComputedPulse slot，共用完整 gate_pulses candidates。
不足處用 BasicGate.Id、gain=0 補位，保持 recovery 時長與 depth 無關。
Interleaved arm 與 reference 共用 random seed，每個 random Clifford 後插 target，相鄰 gates 保持分開，各算完整 inverse 並延續 virtual-Z frame。
序列表 owner 是 twotone/_rb_sequence.py。

分析只選所有 arm、depth 均為有限值的 seeds，等權平均。所有 arm 共用一條 IQ 投影軸，各擬合 A*p**depth+B。舊檔帶 round 軸，Load 會因軸數不合拒絕。
Reference 回報 error per Clifford=(1-p_reference)/2；有 interleaved arm 時另報 target error=(1-p_interleaved/p_reference)/2，負值保持原值。
95% interval 對整條 seed curves bootstrap，不包含 gate-dependent／coherent noise 系統誤差。
中止保留 raw 與未完成 slot 的 NaN；少於兩個 seed 或四個 distinct depths 時分析失敗。

## 寫新 Experiment 時的檢查清單

1. 已依 [風格指南自查](experiment-style.md#4-自查) 核對 core 的局部性、import、description、掃描與檢查授權。
2. Cfg 名稱、順序與 module 型別符合 [cfg 契約](#cfg-名稱順序與取數欄位)，版本與表單符合 [cfg 組合](#v2-cfg-組合慣例)。
3. Experiment[Result, Cfg] 的 Result dataclass 只含資料，RunRecord 持有 cfg 與歷史 metadata。
4. 需要 typed 持久化時宣告 class-level AXES_SPEC，single／grouped 共用入口，tag 使用有層次的 on-disk 名字。
5. AxesSpec.Axis／Z 的軸序、單位與 scale 已依 [存檔軸](#存檔軸用關鍵字固定座標不塞進-result) 核對。
6. `save_run(record, path, spec=..., replace=False)` 保存 native，`save_labber(record, path, spec=...)` 另匯出 Labber，兩者明選格式。
7. `restore_run(stored, spec=...)` 還原 native Container，`load_run_with_metadata(path, spec=...)` 讀 native data.h5 再還原 typed record。
8. Typed cfg 不相容時保留有效 data，回 cfg=None 與 cfg_issue；缺 cfg envelope 或壞資料仍拒絕。
9. 基底只宣告 run／analyze 簽名，保存與讀取用上述函式。GUI tag、restore、保存 hooks 預設沿用同 spec，見 [GUI contracts](gui-contracts.md#共用-lifecycle-hooks)。
10. Run／analyze 接本次 context／Plots，host／program loops 與取數取消責任符合 [Timeline 與 Schedule](#timeline-program-與-schedule-取數)。
11. 實驗定義改動依 [離線驗收](../README.md#實驗定義的離線驗收) 做 probe，測試邊界依 [程式碼品質](../../docs/code-quality.md#測試驗證可觀察契約) 與 [tests 驗收](../../tests/README.md#實驗定義)。
