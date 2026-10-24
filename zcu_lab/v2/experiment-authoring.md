# v2 experiment authoring

**Last updated:** 2026-10-22

**讀者：** lab-author、framework-dev、agent。撰寫實驗 core，查閱 records、資料形狀、執行與保存慣例。

使用者入口見 [tutorial](../tutorial/README.md)。

## 目錄

- [實驗序列與 recipe 撰寫規則](#讓使用者只看一個檔案就能理解實驗)
- [使用者實驗的測試](#使用者實驗的測試)
- [實驗家族 records](#實驗家族-records)
- [目錄佈局](#目錄佈局)
- [實驗特有的分析與保存](#實驗特有的分析與保存)
- [典型 Exp.run 範本](#典型-exprun-範本以-onetonefreqcorepy-為例)
- [v2 cfg 組合慣例](#v2-cfg-組合慣例)
- [Signal/Raw 處理慣例](#signalraw-處理慣例)
- [v2 工具與實驗輔助](#v2-工具與實驗輔助)
- [Lookback 與 Tracker](#lookback--tracker)
- [Twotone RB 測量策略](#twotone-rb-測量策略)
- [外部中斷支援](#外部中斷支援cancel_flag)
- [新 experiment 檢查清單](#寫新-experiment-時的檢查清單)

`zcu_lab/v2/` 擁有使用 program/v2 的具體實驗。每個實驗的 `core.py` 保存 cfg、Result、run 與分析政策。共同實驗介面、Result 保存映射與 cfg 組裝見[框架 README](../../lib/zcu_tools/experiment/README.md)。本頁保留實驗家族、具體 workflow 與撰寫慣例。

GUI adapter 的 module writeback 直接提出持久 ModuleSpec，由共用 Form 編輯，再交 Library owner 保存。Core 仍接收執行用 typed cfg，不依賴 GUI 表單或 draft。作者入口見 [measure authoring](measure-authoring.md)，跨宿主契約見 ADR-0065、ADR-0008。

一般實驗以 [runtime](../../lib/zcu_tools/experiment/v2/runtime/README.md) 的 `SignalBuffer`、`Schedule` 與 `ProgramBuilder` 編排 host loop 與 program acquire。Runtime 的 buffer、stop、retry 和 lifecycle 機制見其文件。

## 讓使用者只看一個檔案就能理解實驗

以下四條規則適用於 `zcu_lab` 的實驗與 recipes。序列與流程由各實驗自己寫出，
不要為了減少相似程式碼，把使用者要讀的步驟藏到共用 helper。

### 1. 在 core 直接寫出 pulse 序列與分析政策

`core.py` 的 `run()` 直接列出 `Pulse`、`Delay`、`Join` 與 readout 的先後關係。
共用 helper 只做驗證、資料整形、平均等機械工作，不用 mode 或旗標切換不同實驗。
兩個實驗可以共用平均工具，但各自在 `run()` 寫出自己的序列。

例如 [mux Amp Rabi](mux/amp_rabi/core.py) 每個掃描點先 reset，
所有參與 drive 同時起跑，最後共同 readout。以下是序列片段，省略準備與取資料：

```python
builder = (
    step.prog_builder(context.soc, context.soccfg)
    .add_reset("reset", modules.reset)
    .add(
        Join(
            *[
                Pulse(f"drive_{index}", pulse)
                for index, pulse in zip(scan.indices, pulses, strict=True)
            ]
        ),
        MuxReadout("readout", readout),
    )
)
```

短 branch 等最長 branch 結束才 readout，不把短 pulse 的末端對齊到最長 pulse。
Ramsey 與 Echo 也各自列出 pulse 與 delay，不能交給 `acquire(mode="ramsey")` 選序列。

同一條規則也適用 `analyze()`。Echo 在自己的 core 明示跳過第 1 點，Ramsey 在自己的
core 依資料選固定 0°／180° 相位。共用 fitting helper 只接收明確的 skip、phase 等
擬合參數，不用實驗名稱或 mode 選擇不同實驗的政策。phase 只傳數值或代表自由擬合的
`None`，不傳選政策的 callback。搬移政策時，保持數值結果與失敗語義不變。

### 2. 在 recipe 函式本體寫出操作流程

Recipe 本體依序開 tab、選 module、逐欄 set、依序套 overrides、check_ready、Run、Analyze，再詢問 writeback。
來源選擇與掃描端點直接留在主體，helper 只共用數字呈現或逐 Q 分析 updates 等機械對應。

例如 [mux Amp Rabi recipe](../recipes/mux_amp_rabi.py) 應直接讀得到
`session.open_tab(...)`、`tab.select(...)`、`tab.set(...)`、
`await tab.run()`、`await tab.analyze("primary", result=T)` 與
`await tab.ask_writeback()`。接受答案後才 `tab.accept_writeback(decision.items)`。
每次 Analyze 後先檢查 `analysis.result`。預期擬合失敗時 result=None、status="failed"，立即回含 `analysis_failure=analysis.reason` 的 report，不 Ask、不接受 writeback、不跑相依 Post。Post 失敗保留 Primary 估計；成功時 analysis_failure=None。任意程式錯誤仍 raise。

Run 留在 try 外，成功返回後的流程以既有 finally 同步 `tab.save()`。根取消使下一個 await raise 普通 ScriptCancelled，進行中的 Run 先 drain 並保留 facts。該 Run await 被根取消時不進後續 Save finalizer，partial 留在 tab 交 agent 決定。作者不新增取消分支或 Save finalizer。
Inputs 繼承 RecipeInputs，框架只作 pydantic／MCP JSON schema 驗證，不渲染 recipe inputs Form。
可省略欄位用 KEEP，null 原樣交 owner。寫入用 dotted string，read 保留 typed tuple。
明填 scalar 前只在 selector.mode 為 ref 時 command to_inline，保留 waveform family。
完整宣告、Analyze 失敗值與根取消契約見 [recipe 教學](../tutorial/04-recipe.md)。

### 3. Docstring 說明量測、序列與結果

公開 docstring 先說量什麼、序列怎麼做、回傳結果代表什麼。
單位、資料形狀及必要的失敗條件仍要交代。Lifecycle、rollback 與 caller 責任
只在必要時用一句帶過，詳細契約連到 framework 文件，不在每個實驗重貼。

例如 [mux Ramsey](mux/t2ramsey/core.py) 的 `run()` 可用這樣的量測說明開頭：

```python
"""Measure Ramsey fringes for the selected qubits with shared mux readout.

Apply simultaneous x90 pulses, wait the requested delay, then apply the
final x90 pulses and read out together. Return requested delays in us
and mean complex IQ per participating tone; the fringe decay measures T2.
"""
```

這裡的 IQ 是量測結果，T2 由後續分析取得，不把擬合值說成 `run()` 的回傳值。

### 4. 用註解與空行分開階段

在準備、序列、取資料與平均等階段之間留空行。註解說明這段做什麼、為什麼，
不逐行重述賦值或函式名稱。程式註解用英文。

例如 [mux Amp Rabi](mux/amp_rabi/core.py) 的階段可以標成：

```python
# 1. Prepare selected tones without changing their original readout indices.
...

# 2. Start drives together; read out after the longest branch finishes.
...

# 3. Collect mean IQ and restore tone-major result order.
...
```

這是分段示意，不是可執行的範本。Recipe 同樣按設定、量測、分析與 writeback 問答分段。

## 使用者實驗的測試

本套件的 adapter、workflow 與 recipes 是實驗定義，不寫個別實驗測試。實驗錯誤由 GUI 啟動或載入直接回報，不逐一走過 lab registry 來選取測試材料。

框架 owner 用測試自己定義的最小合成 workflow／adapter 驗證公開 seam 與對外契約，不 import `zcu_lab`。刪掉 `zcu_lab/` 後整套 pytest 仍須通過，包含 slow。實驗共用 helper 留在本套件，不因可抽成 helper 就新增 pytest；框架共用機制依責任放在 lib，由該 owner 的合成材料測試驗證。

案例歸屬、fixture 與搬遷規則見 [tests README](../../tests/README.md)。

---

## 實驗家族 records

### T1、singleshot GE 與 OneTone 的核心入口

T1、T2Echo、T2Ramsey、AmpRabi、LenRabi 的數值 Analysis 保存本次 fit_quality，普通擬合名為 fit。品質只使用 skip／mask 後的資料、已提交模型與 optimizer covariance，不重新擬合。GE 保存 joint、ground、excited 三階段，各自使用自己的資料、模型與 covariance；post 沿用 primary calibration。品質不改曲線、校正有效性或 writeback policy。

`twotone.time_domain.T1Exp` 使用無跨次狀態的 `run(config, *, context)` 和 `analyze(source, options, *, plots)`。T1Result 只含 times／signals；RunRecord 將 typed cfg 與資料配成來源，`T1Cfg.uniform` 隨 record cfg 保存。T1AnalyzeOptions 提供 skip 與 dual_exp 的 defaults，T1Analysis 只含數值。T1 分析接受 cfg=None 的來源，預設 canonical saver 則拒絕缺 cfg。Run 建立具名 measurement liveplot，分析建立 fit 圖；caller 負責操作結束後 finish 及釋放呈現，不以新操作關閉舊圖。

`singleshot.ge.GE_Exp` 的 run 回傳純 GE_Result，只有 signals／shot_indices／prepared_states。Analyze 接收 RunRecord；`post_analyze(source, primary, options, *, plots)` 使用同筆來源與已採用的 FIT 校準，不重新擬合。FIT 與 post 分別向本次 Plots 發布 `fit`／`post` 圖，analysis 只含數值。兩種分析都接受 cfg=None；預設 canonical saver 拒絕缺 cfg。Notebook 使用共用 NotebookAdapter 與獨立 GEPostAnalyzer，不由核心持有上次操作狀態。

`onetone.flux_dep.FluxDepExp` 用 explicit context 執行同步 run，回傳只有 values／freqs／signals 的純 Result。`measurement` 使用共用 2D 熱圖與掃描線 handle，也能繪製反向通量掃描。核心只負責 acquisition 與 RunRecord 的 native restore／持久化宣告，不提供互動 analyze。Notebook 與 GUI 各自捕捉來源並完成選線，重用 Qt-free kernel 與原生圖 builder。GUI preview 不作結果圖。

`onetone.freq.FreqExp` 的 run 回傳純 freqs／signals，支援 linear 與 homophasal 取樣。同步 analyze 接 explicit source 與 FreqAnalyzeOptions，回傳純 FreqAnalysis，向 Plots 發布 `fit` 圖。`onetone.power_dep.PowerDepExp` 只提供量測及保存／載入，earlystop_snr 歸 typed cfg。兩者都用 explicit context 執行 run，保留 canonical Hz 與 complex data。

一般 T1、GE 與 OneTone 的 Notebook run／同步 FIT 入口是 先建立 `nb_adapter = zcu_tools.notebook.NotebookAdapter(...)`，再呼叫 `nb_adapter(core)`，不由核心 namespace 轉接。GE 的專用 post 工具由 [`singleshot/ge/notebook.py`](singleshot/ge/notebook.py) 擁有。FluxDep 的獨立選線工具由 [`zcu_tools.notebook.flux_dep`](../../lib/zcu_tools/notebook/flux_dep.py) 擁有。FluxDepAnalyzer 明確接收 RunRecord 與 caller 填入的 FluxPickInputs，不讀 NotebookAdapter 的目前來源。相關 Notebook 與 GUI callers 已採用 records 與 explicit context，下方 run 範本使用同一契約。T1WithTone／ScanT1WithTone 也使用 explicit context／source／options／Plots。T1WithTone 共用純 T1Result 與 T1Analysis，但保留原有 dual-decay 分量選擇；ScanT1WithTone 的分析輸出只有 gains／t1s／t1errs。

T2Echo／T2Ramsey 的 detune 放入 typed cfg，run 回傳純 Result，analyze 使用 explicit source 與 typed options，向 Plots 發布 fit 圖。硬體 rounding 後的 true_activate_detune 是 run-only metadata；canonical axes／complex data 不包含它，load 後值為 None，不推測實際 detune。兩者分析均允許 cfg=None。

### FastFlux records

FastFlux 六個核心使用 `run(config, *, context) -> Result`，不保存跨次 cfg／result。
同步分析接收 explicit `RunRecord`、options 與 `Plots`。T1 與三個 distortion 回傳
具名數值欄位，MIST 的 `MistAnalyzeOptions` 保留 `ac_coeff`，其餘無可調選項時傳入
`None`。三個 distortion 需要來源 cfg 來取得 pulse 時間，缺少時明確拒絕。共用
NotebookAdapter 建立 records；核心保留既有 canonical axes、單位與分析公式。

### TwoTone spectroscopy records

`twotone` 的 Freq、FreqFlux、Power 與 Dispersive 使用 explicit RunContext，run 回傳純 Result。
Freq 與 Dispersive 同步 analyze 接 RunRecord 與各自 typed options，回傳數值 Analysis，
向本次 Plots 發布 `fit` 圖。FreqFlux 與 Power 只提供量測及保存／載入，不提供 core analyze。
FreqFlux 的 `fail_retry` 是 typed cfg 欄位；GUI 選線由 plugin 擁有。Notebook 使用
NotebookAdapter 執行量測，互動選線由獨立 FluxDepAnalyzer 處理，不由核心轉接。

### TwoTone Rabi records

`twotone.rabi` 的 AmpRabi／LenRabi 核心以 RunContext 執行量測，回傳純 Result。同步 analyze 接 explicit RunRecord 與各自的 typed AnalyzeOptions，回傳數值 Analysis，接受 cfg=None。量測與分析分別發布 `measurement`／`fit` 具名圖；caller 負責 finish／release。LenRabi 保留 const／flat_top 的板端 sweep 與 arb waveform 的 host scan、跨 rounds 平均及部分成果。Notebook callers 透過 NotebookAdapter 保存 run 與 analysis records。

### Mux qubit zip acquisition

`mux.twotone_spectrum.MuxFreqExp` 與 `mux.amp_rabi.MuxAmpRabiExp` 提供無 GUI 的 `run(config, *, context)`。
完整 `tones: list[MuxToneCfg]` 保留 readout 原始順序。Prepare 在 numeric 邊界一次選出 active indices，
產生各 drive 自己的量化座標 `(K,P)`。只有 Amp Rabi 另建 nominal `point_pulses (P,K)`。
Inactive tone 不限制掃描點數。
Freq 的座標是 MHz，Amp Rabi 是 normalized gain。不同範圍按 index zip，不展開 Cartesian product。

兩個 run 各自在函式本體直接組成 reset、`Join(Pulse(...), ...)` 與 active
`MuxReadout`，不經共用 acquisition helper 選擇或隱藏序列。
Freq preparation 回傳 `MuxFreqScan`，只包含原始 indices 與量化座標。
Amp Rabi preparation 仍回傳含逐點 pulses 的 `MuxPulseScan`。兩者都不 acquire。
Freq 的各 drive 用 `sweep2param("point", tone.sweep)` 共用同一硬體 loop，只 compile 一次。
Readout DAC tones 與 ADC 頻率保持固定。Freq 座標按硬體 start/span/step 量化，
不逐點獨立 round。Amp Rabi 仍每點 compile 一次。
所有 branch 同時開始，短 branch 完成後等待；readout 接在最長 branch 之後。
沒有末端 padding，也沒有第二次 compile 或自訂 Module。
既有 Pulse 與 Join 擁有 native timing，不改 pulse 的 pre/post delay。

`SignalBuffer` 以 `(P,K)` 儲存，讓 Schedule 每步寫入整個 tone vector，回傳時轉為 `(K,P)`。
Program 擁有 reps/rounds 平均；取資料時只經初始化後的 `MuxReadout.collect_iq` 取得平均 IQ。
Freq 的 round hook 保留全 sweep 已完成 rounds 的平均；第一個 round 未完成時保留 NaN。
Amp Rabi 的 round hook 保留目前點已完成 rounds 的平均，取消後未寫入點仍是 NaN。RunContext 的
`context.scope.stop.raise_if_error()` 區分 acquisition failure 與正常取消；有 partial data 不代表成功。
Plots 以原始 index 命名 `measurement_i`，每條曲線用自己的座標，不靠 label 去重。
Caller 負責 finish/release、組成含完整 cfg 的 RunRecord，以及保存 completion 狀態。

兩個 leaf 的分析與 native save/load 都讀 explicit source，不從目前 cfg 補歷史輸入。
公開入口與單位見 [Mux two-tone spectrum](mux/twotone_spectrum/guide.md) 及 [Mux Amp Rabi](mux/amp_rabi/guide.md)。

### TwoTone pulse calibration records

`AcStarkExp`、`AcStarkRamseyExp`、`CKP_Exp` 的 run 使用 RunContext，回傳純 Result。AcStark 的 earlystop_snr 與 Ramsey 的 acquisition detune 歸 typed cfg。Analyze 接 explicit source 和 typed options，接受 cfg=None。AcStark 回傳 ac_coeff；Ramsey 只發布 fit 圖，分析 detune 仍可獨立指定；CKP 回傳 chi／kappa／res_freq。兩種 AcStark 使用 measurement 2D with line，CKP 在同一張 measurement 圖並排 ground／excited 熱圖；三者的分析圖都具名 fit。Caller 擁有 finish／release，NotebookAdapter 管理 Notebook 的 records 與呈現。

### Twotone sequence records

AllXY／RB／ZigZag／ZigZagScan 使用 explicit RunContext、純 Result 與具名 measurement。AllXY 的 gate ticks／style 由 host owner 在呈現前初始化，analyze 接 AllXYAnalyzeOptions，回傳 AllXYAnalysis（相對振幅誤差、detune 參數、Bloch-z 平均偏差與 fit residual），並發布 fit 圖。RB analyze 接 None options，回傳 EPC／fidelity；ZigZagScan 接 find_range options，回傳 min_value。兩者 fit 圖另交 Plots。ZigZag 的 repeat_on 歸 typed cfg，沒有 analyze。Canonical 軸與 RB seed／recovery 演算法不變。

### MIST records

`mist` 的 FluxDep／DriveFreq／PowerDep 核心使用 explicit RunContext，回傳純 Result。Analyze 接 explicit RunRecord，接受 cfg=None，只發布具名 fit 圖並回傳 None。FluxDepAnalyzeOptions 保留通量換算、photon 軸及第二座標刻度；PowerDepAnalyzeOptions 保留 g0／e0／ac_coeff，DriveFreq 使用 None options。FluxDep 的熱圖為原生 Matplotlib 圖，不再接收 Plotly fig／fig_kwargs；caller 從具名圖集合取得 Figure 進行原生操作。Notebook callers 使用 NotebookAdapter 與核心 typed cfg／options。

### Bath reset records

Bath reset 的 FreqGain／Length／Phase 核心以 RunContext 執行，回傳純 Result。FreqGain analyze 接 smoothing options 並回傳 gain／freq；Phase 使用 None options、回傳 max_phase／min_phase；Length 使用 None options、只發布 fit。分析接受 cfg=None。四點 tomography、Length 的 host rounds 平均與 program cache、Phase 的 cosine fitting 不變。GUI 使用明確 RunRecord，數值結果與具名 fit 圖分開。

### Reset tone records

Single-tone 的 Freq／Length 與 dual-tone 的 Freq／Length／Power 使用 explicit RunContext、純 Result 和 measurement／fit 具名圖。Single Freq 分析回傳 freq／fwhm，兩 Length 只發布曲線。Dual Freq／Power 接 typed smoothing options，回傳兩軸最佳頻率／gain。分析皆允許 cfg=None。Dual Freq 的 method 歸 typed cfg，預設 soft；GUI 固定 hard 並保存於來源 cfg。Soft host scan 與 hard 雙軸 sweep 保留相同 canonical 軸序，雙 pulse 長度差與 gain 掃描的背景選擇不變。

### Readout optimization records

`twotone/ro_optimize` 的 Freq／FreqGain／Length／Power 核心使用 explicit RunContext，回傳純 SNR Result。Run 保留既有 acquire kwargs forwarding 與 MomentTracker／g-e branch；Length 用 host scan，其餘用硬體 sweep。同步分析接 RunRecord、typed AnalyzeOptions 與 Plots，允許 cfg=None，回傳純最佳 frequency／gain／length，另發布 fit 圖。Smoothing、length duration normalization、power penalty 與 canonical axes 不變。Auto optimizer 使用下述 grouped record 契約。

### JPA records

JPA Freq／Flux／Power／Check／OneToneFlux 以 explicit RunContext 執行，回傳純 Result；cfg 與資料由前端配成 RunRecord。前三者的分析接 explicit source、None options 與 Plots，回傳最佳 scalar；Check 只發布 fit 圖，OneToneFlux 不提供 analysis。Run 的 measurement 圖歸本次 Plots。Freq／Power 保留隨機內部掃描點與無連線點圖，Flux 保留 Gaussian smoothing，pump off/on check 結束時仍保持 ON。AutoOptimize 同樣接 explicit records 與 grouped persistence，num_points 的整數及最低預算驗證由 typed cfg 擁有。最佳三參數分析發布原生四 axes fit 圖；sample_params 診斷接 explicit source，發布3D圖。即時四投影使用具名 typed scatter，保留 optimizer phase 色彩。

Readout AutoOpt 同樣使用 records 與 grouped persistence，num_points 屬於 typed cfg。分析回傳最佳頻率／增益／長度，四 axes 的 fit 圖獨立發布；即時 iteration／freq／gain／length 投影各自具名並由 Plots 管理呈現。

### CPMG grouped records

CPMG 使用 explicit RunContext／RunRecord／Plots，不保留跨次結果。detune_ratio 與 earlystop_snr 屬於 run cfg，分析選項屬於 CPMGAnalyzeOptions，回傳 CPMGAnalysis 的 ns／t2s／t2errs；T2 與誤差使用 us，fit 圖另交 Plots。共用 GroupedAxesSpec 保存與載入 record；lengths role 保留每列時間座標，signals role 與既有 grouped v2 schema 不變。

## 目錄佈局

每個 leaf 的位置與附件見[套件 README](README.md)。量測核心不 import GUI。跨實驗的 singleshot numerical helpers 留在 `_support/singleshot/`，reset fit 與 JPA optimizer 留在各自的 leaf，通用 runtime 與量化工具留在框架。


---

## 實驗特有的分析與保存

實驗基底、RunRecord 與 Result 契約見[父層實驗介面](../../lib/zcu_tools/experiment/README.md#實驗介面與資料)。以下記錄 v2 實驗特有的分析與資料形狀。

有分析選項的實驗以 `Experiment.Options` 公開其既有 options 類別，例如 `T1Exp.Options(skip=1)`。GE 另以 `GE_Exp.PostOptions` 提供 post-analysis 選項。類屬性引用型別，不保存共用 instance；欄位與預設仍由原 options 定義擁有。沒有選項的實驗不補空 Options。這些屬性不是共用 Protocol 的要求。

CKP numeric analysis先從ground/excited maps抽取resonance trace，再透過
`analysis.fitting.shared`共同擬合Lorentzian baseline、scale與width；兩個resonance
frequency維持local，兩個slope各自固定為零。Chi/kappa uncertainty直接由named global
covariance投影，包含local frequency cross-covariance；analysis不重建per-trace covariance
blocks。Backend minimum或covariance無效時fast-fail；`CKPAnalysis`只含chi、kappa與res_freq，圖交本次Plots。

`twotone/reset/RabiCheckExp` 以 explicit RunContext 執行，回傳純 Result，發布三分支 measurement。Analyze 接 RunRecord 與 None options，回傳純 `RabiCheckFit`，另向本次 Plots 發布 fit；接受 cfg=None。reset前分支決定
共同IQ投影與基頻，reset-only以同基頻擬合，reset後追加Rabi的分支包含基頻與二次諧波。
半峰對峰振幅、相位、相對contrast與殘差只描述平均IQ資料，不推導reset fidelity。
數值分析由同目錄`fit.py`擁有，GUI adapter只投影其純量；acquisition與持久化
資料格式維持三分支gain sweep。

`singleshot/ResetCheckExp`沿用MIST power的population acquisition，以硬體gain／branch
sweep保存`(Ngain, 3, 2)` G/E populations。analysis可用外部confusion matrix修正，
回傳reset-only平均／最差excited population，不提供IQ校準writeback。
Other不是校準後leakage，這些population不是reset-channel fidelity。
Run 使用 explicit RunContext，回傳純 Result。Analyze 接 RunRecord 與 ResetCheckAnalyzeOptions，回傳純 ResetCheckAnalysis，另向本次 Plots 發布 `populations` 圖。

`singleshot/AmpRabiExp`保留硬體gain sweep，每個host round擷取所有gain的Reps筆raw IQ；
不同round的shots串接而不平均，canonical complex128 shape為`(Ngain, Reps * Rounds)`。
`shot_indices`是inner axis，`gains`是outer axis。完成的round立即更新live分類population；
取消時未完成round保持NaN，analysis僅移除完全缺失的shot columns，部分缺失的sweep明確拒絕。
run使用resolved cfg snapshot中的GE centers/radius作live分類，analysis不使用外部confusion matrix。

Amp／Len Rabi、Check、ResetCheck 與 AC Stark 使用 explicit RunContext 執行，GUI 將同次 cfg／Result 配成 RunRecord。同步 analyze 接 source／typed options／Plots，允許 cfg=None，不保存跨次成功狀態。Amp 回傳 AmpRabiFit，Len 回傳 RabiJointFitResult，兩者的 readout 診斷只讀 source.cfg。Check 只發布分類 scatter，無數值結果。AC Stark 回傳 AcStarkAnalysis，另提供 explicit source／AcStarkPlotOptions 的 population 圖入口；即時 Ground／Excited／Other 熱圖及 current trace 各自具名。

Amp與Len Rabi共同使用`zcu_tools.analysis.fitting.singleshot_rabi`的raw-IQ joint likelihood與
`zcu_tools.plotting.singleshot_rabi`的population、histogram及confusion matrix診斷圖。
Amp固定無衰減／零相位，Initial State描述zero-gain前的主要狀態，支援signed gain。
Amp額外由joint frequency及其covariance推導pi/pi2 gain與誤差，保留原有scalar summary。
兩者的GUI校準writeback共用backend validity及finite校準值檢查。
舊Amp population-only檔案缺少IQ shots，無法轉換成新資料，也不在load時虛構raw IQ。

---

Singleshot T1／T1WithTone／T1WithToneSweep 使用 explicit RunContext，uniform 歸 typed cfg。同步分析接 RunRecord、typed options 與本次 Plots，允許 cfg=None。Tone 回傳純 t1／t1_b，其餘兩者只發布 fit 圖。即時 ground／excited traces 各自具名；Tone sweep 的六張 population 熱圖與兩張 current trace 各自具名。Transition-rate fitting、host rounds 與完整 outer-row 篩選保持不變。

### 版本實驗的資料形狀

`PersistableExperiment`、`AxesSpec`、inner-first 軸序及 grouped variables 的共同映射見[父層 README](../../lib/zcu_tools/experiment/README.md#實驗介面與資料)。以下保留 v2 實驗的具體資料形狀。

- **grouped experiment variables**：`CPMG_Exp` 使用 variables `lengths` / `signals`，axes 為 inner-first 的 `Time Index`、`Number of Pi`，盤上 `lengths` 單位為 seconds，記憶體內仍回復為 us。RO auto-optimize 使用 variables `readout_freq` / `readout_gain` / `readout_length` / `snr`；JPA auto-optimize 使用 variables `jpa_flux` / `jpa_freq` / `jpa_power` / `jpa_phase` / `snr`，其中 `jpa_flux` 以中性 device-native value 寫盤（unit `a.u.`、identity scale、數值不縮放），舊 auto grouped file 若 `jpa_flux` variable unit 為 `A` 不是 canonical，strict loader 不做 `A` fallback。頻率與時間在 disk 上使用 SI units（Hz、s），typed loader 重建回 Result 記憶體單位（MHz、us）；JPA phase 是 integer index。這些 runtime `load()` 讀取與 grouped spec 相符的 native `data.h5`，並透過 `from_stored` 還原 typed record。Labber 匯出由獨立的 `save(source, destination)` 提供，不能直接交給 native loader。既有 Labber 舊檔依 [migrations](../../migrations/README.md#legacy-resultdatabase--native-storage) 提供歷史證據並離線轉換；legacy `.npz` 或 sidecar 不是 runtime 可載入格式。
- **legacy single-file**：舊 Labber HDF5 的 `Frequency` `MHz/Hz`、`Yoko` flux 軸或 `ADC unit` signal channel 不符合當前 `AXES_SPEC`，不由 runtime/GUI 隱式轉換。`onetone/flux_dep` 的 canonical axes 是 `(freqs, values)`，對應 Result-native `signals.shape == (Nflux, Nfreq)`。
- **single-variable 離散狀態軸**：bath reset freq-gain 把四點 pi/2 tomography phase 視為同一個 Result 的第三個 sweep axis；bath reset length 把 phase 視為第二個 axis，Result-native shape 為 `(Nlength, 4)`；`CKP_Exp` 把 ground/excited prepared state 視為 `initial_states` axis；`GE_Exp` 把 ground/excited prepared state 視為 `prepared_states` axis，Result-native shape 為 `(2, Nshot)`；singleshot `len_rabi`以`shot_indices`作inner axis，canonical `complex128` raw-IQ shape為`(Nlength, Nshot)`；analysis將pooled IQ投影至共同PCA axis，以固定共同bins的integrated readout-transition multinomial likelihood joint-fit 可選衰減包絡及phase offset的Rabi dynamics，重建g/e centers並推導nearest-center-region radius與other row為identity的confusion matrix；population points與fit curves皆從raw result衍生；舊population-only檔案缺少IQ shots，canonical loader明確拒絕而不虛構資料；MIST `power` / `freq` / `pre_freq` 把 `g/e` population components 視為 `population_states=[0, 1]` axis，canonical shape 為 `(Nsweep, 2)`；singleshot `ac_stark` 與 MIST `power_freq` 使用 `population_states` 加兩個 sweep axes，canonical shape 為 `(Ngain, Nfreq, 2)`；singleshot `t1` / `t1_with_tone` 使用 `population_states`、`initial_states` 與 `lengths`，canonical shape 為 `(Nt, 2, 2)`；`t1_with_tone_sweep` 使用 `population_states`、`lengths`、`initial_states` 與 generic `xs`/`Sweep Value` axis，canonical shape 為 `(Nx, 2, Nt, 2)`，只存 Result 的 g/e components，`other` 由 analysis 推導。這類 homogeneous Result 的 native 資料存成單一 `data.h5`，Labber 另匯出單一 `.hdf5`。離散狀態不是 Data Variable，不拆成多個 sidecar artifact；legacy artifact 不由 runtime 載入；舊 singleshot population HDF5 的 `(2, Nsweep)` 或 multi-sidecar z 方向也不在 runtime 重排。

Singleshot MIST 的 Freq／Power／PreFreq／FreqPower 使用 explicit RunContext，回傳純 Result。所有 classification 校準值屬於 typed cfg。同步 analyze 接 RunRecord、對應 AnalyzeOptions 與 Plots，允許 cfg=None，只發布 fit 圖。Power 支援 photon 軸與 log scale；FreqPower 不宣告未實作的軸選項。單 sweep 即時圖使用 measurement，二維掃描的 Ground／Excited／Other 熱圖各自具名；canonical population shapes 不變。

Singleshot Len Rabi的length軸由host-side `Schedule.scan`逐點執行；每個program只擷取
單一length的`shots`筆raw IQ，再寫入sweep-first Result row，避免FPGA同時配置完整
`Nlength × Nshot` raw buffer。stop保留已完成rows，其餘維持partial-result NaN。

Len Rabi numeric analysis以backend minimum validity作為finite calibration與writeback的前置條件。raw-IQ initializer以pooled PCA two-cluster assignment取得各群median center、群內pooled MAD noise scale與per-length粗略population；high-shot histogram超過coarse resolution時，同時保留quantile initializer作為另一個deterministic basin，先比較較粗的integrated-bin likelihood，再以較佳candidate回到原始共同bins求最終minimum，必要時才嘗試另一個candidate。coarse stage不取代或放寬final validity。各預設Migrad在invalid時最多續跑一次；caller提供explicit `max_calls`時略過coarse stage且只執行一次Migrad，不把該budget延伸成restart。Analysis Figure由experiment Module負責，以上方population estimates/global fit、左下第一個acquired point的integrated-bin histogram decomposition及右下derived confusion matrix呈現同一份joint-fit證據；valid histogram依cfg acquisition length與fitted length ratio顯示effective T1，並列出該point落入G/E classification circles與circle外L區域的observed fractions，不顯示Rabi pulse length。layout在GUI preview與fixed-size save geometry都維持panel、labels與annotations分離；invalid結果保留observed histogram，但不顯示fitted decomposition、effective T1或calibration matrix。

Singleshot Amp／Len Rabi、ResetCheck、T1 family、AC Stark 及 MIST freq/power/power_freq/pre_freq 的 classification
校正值由各自 cfg 的 `g_center/e_center/radius` 提供，run 不接受獨立校正參數。RunRecord.cfg 保存這次校正值。Complex centers 在 experiment comment 中以可逆 complex literal 序列化，還原由 Pydantic complex 欄位處理。

Singleshot Rabi joint fit依analysis選擇有衰減或純cosine population dynamics：
`len_rabi`預設擬合衰減包絡，也可選擇純cosine。純cosine模型不擬合
`t_r`，結果以`None`表示該參數不適用；兩種模式共用raw-IQ calibration與confusion
matrix的估計流程。

檔案格式細節見 [datafile README](../../lib/zcu_tools/datafile/README.md)。

---

## 典型 `Exp.run()` 範本（以 `onetone/freq/core.py` 為例）

一般核心實驗的入口是 `run(cfg, *, context: RunContext) -> Result`。硬體與具名 devices 由 context 傳入，cfg 只含這次操作的設定。Result 只含資料。NotebookAdapter 或 GUI adapter 將原始 cfg 與 Result 配成 RunRecord，不在核心快取上一輪結果。

1. 以 `sweep2array` 將 sweep 展成硬體格點。Device setup 使用 `context.devices`。
2. 從 `context.plots.liveplot_1d("measurement", ...)` 建立具名 viewer，其圖由本次 Plots 持有。需要自訂 layout 時，明確建立 Figure／Axes，不使用 pyplot current state。
3. `SignalBuffer` 的 on_update 將完整 buffer 交給 viewer。Schedule 使用 runner-owned cfg 副本，透過 `ProgramBuilder` acquire 並寫回 buffer。
4. Schedule 接 `scope=context.scope`，將外部取消傳給 acquire。Experiment setup helper 接 `cancel_signal=context.scope.stop`，只在 helper 到 driver 的邊界轉成 Event；直接 acquire 的路徑也必須傳入 cancel flag。
5. 回傳純 Result。同步分析接 `analyze(source, options, *, plots)`，數值直接回傳，圖向本次 Plots 發布。

Schedule／ProgramBuilder 的 buffer、program cfg、retry 與 partial-result 規則見 [runtime README](../../lib/zcu_tools/experiment/v2/runtime/README.md)。
目前所有 Experiment 均走新 runtime 或直接 `SignalBuffer` path，包含 `lookback/core.py`、`onetone/*`、`twotone/*`、`singleshot/*`、`jpa/*`、`fastflux/*` 與 `mist/*`。

`onetone/freq` 的 frequency sampling 有兩種 mode：`linear` 沿用 program-side
`SweepCfg` sweep；`homophasal` 保留同一個 `sweep.freq` 使用者介面，但由已擬合的 resonator
circle 參數產生非等距 frequency array，再把 gen/readout raw frequency words 放進
`TablePulseReadout` 的 dmem tables。每點沿用一般 `PulseReadout` 的 wmem-backed WPORT，
並在 sweep loop 的 exec-after hook 更新下一點 generator/readout frequency words；最後一點
回繞第一點供下一個 outer repetition 使用。這條路徑不重置 DDS phase accumulator，更新
位置與 QICK native `QickParam` sweep 一致。非等距模式仍先把點 round 到硬體格點，並在
相鄰點 collapse 時 fast-fail。

`zcu_tools.experiment.v2.utils.t1_sampling` 擁有 T1 family 的 non-uniform sampling contract，供
`twotone/time_domain/t1` 與 singleshot `t1`、`t1_with_tone`、`t1_with_tone_sweep` 共用：
在硬體量化前沿 normalized T1 decay curve 等弧長配置 delay，保留 configured window 與 point
count。各路徑依 delay 或 probe-pulse 所屬的硬體 grid 量化；任何 collision 或非嚴格遞增都
fast-fail，不以 sorting、`unique` 或自動減點掩蓋。Direct delay list 走相同 conversion
contract，但不重新採樣。Singleshot `t1_with_tone` 的 zero-length 點略過 probe pulse，避免
建立硬體不接受的 zero-cycle pulse；`t1_with_tone_sweep` 則要求完整 time axis 嚴格大於
零，並在 device setup 前 fast-fail。其 uniform 與 non-uniform time axis 都由
`TableLengthPulse` 在板上執行：外層 gain/frequency 維持 host scan，內層 length loop 從
dmem 載入 pulse length 與實際 duration；const/flat-top 共用單一 wmem template，readout
依每點真實 pulse 結束時間對齊，不把短 pulse 補到最長點。

### sweep 參數 mutation 的歸屬：搬進 runner-owned cfg

把 sweep 參數綁到 pulse（`modules.xxx.set_param("freq"/"gain"/"length", ...)`）一律在 runner-owned cfg 上做：一般 Schedule 寫法改 `sched.cfg` 或 `step.cfg`。不要在 `run()` 頂層直接改 caller 傳入的 `cfg`。`sweep2array` 只讀 `cfg.sweep.*`（sweep 定義）與通道，不依賴 pulse param，所以 `set_param` 不需要提前到外層。

只有兩類「副本外操作」是有意保留的：

- **device setup**：`set_*_in_dev_cfg(cfg.dev, ...)` + `setup_devices(cfg, context.devices, progress=True, cancel_signal=context.scope.stop)` 需要在掃描前先把硬體帶進度地初始化到起點，留在 `run()` body；`progress=True` 本身即通知，不另加 warn。
- **raw-shot program 設定**：`singleshot/ge`、`singleshot/check`、`singleshot/len_rabi` 與 `mux/singleshot_ge` 的實驗 cfg 不含 reps／rounds。`run()` 另建 `ProgramV2Cfg(reps=cfg.shots, rounds=1, ...)`，保留原始 initial／relax delay，再交給 builder。Lookback 同樣另建 program cfg，固定 reps=1，rounds 沿用可編輯的實驗欄位。這些 runtime 固定值不出現在 Form。

---

## v2 cfg 組合慣例

具體實驗使用自己的 module／sweep cfg。需要 reps、rounds、initial_delay、relax_delay 的 cfg 從 `zcu_tools.experiment` import `ProgramExpCfg`，再以 `class XxxCfg(ProgramExpCfg)` 加上實驗欄位。不含 program 欄位的 cfg 仍繼承 `ExpCfgModel`；在 run 內另建設定時保留 `ProgramV2Cfg`。

Core import `Pulse`、`Readout`、`Reset` 等實際使用的模組，欄位型別寫 `Pulse.Cfg`、`Readout.Cfg`、`Reset.Cfg`。具體 readout／reset／waveform 也使用自己的 `.Cfg`，例如 `MuxReadout.Cfg` 與 `BathReset.Cfg`。`SweepCfg` 與硬體指令仍按需 import。`SweepCfg` 定義在 `program/v2/sweep.py`。父層 cfg 型別與執行前組裝方式見[父層 README](../../lib/zcu_tools/experiment/README.md#cfg-與父層支援)。

---

## Signal/Raw 處理慣例

- **`raw` 格式**：QICK 回傳的 raw 一般是 `list[ndarray]`，第一個元素 shape 為 `(nro, ..., 2)`（IQ 兩個實數）。
- **`default_raw2signal_fn`**：Schedule 路徑使用 `zcu_tools.experiment.v2.runtime.schedule` 的預設轉換：`raw[0][0].dot([1, 1j])`，取第 0 個 RO channel、該 channel 的第 0 次 readout，再把 IQ 轉成 complex。
- **客製化 `raw2signal_fn`**：integrated acquire 用 `build_and_acquire(raw2signal_fn=...)` / `run_program(raw2signal_fn=...)`；decimated trace 用 `build_and_acquire_decimated(raw2signal_fn=...)` / `run_program_decimated(raw2signal_fn=...)`。`ProgramBuilder.set_raw2signal_fn(...)` 可設定同一個 builder 的預設轉換。
- **`signal2real` 函式**：每個 Exp 檔案會定義 local 的 `xxx_signal2real`（通常 `np.abs`），給 liveplot 用；analyze 階段可能換成 phase / real。
- **scalar/array 邊界**：座標轉換工具（例如 value↔flux）可接受 scalar 或 ndarray；若後續 plotting/analysis 需要 indexing、min/max 或與另一個 sweep array 對齊，呼叫端在邊界用 `np.asarray(..., dtype=...)` 正規化成 ndarray，而不是用型別宣告假設回傳一定是 array。
- **peak-picking smoothing**：ro-optimize 與 reset 這類以 SNR/map argmax 找最佳點的分析預設使用 `smooth_method="wavelet"`；`smooth_method="gaussian"` 保留為舊 Gaussian 對照。`smooth` 是通用強度：Gaussian 時是 sigma，wavelet 時是 threshold scale。ro-optimize length 的 `t0` 是 `SNR/sqrt(length + t0)` 的 duration-normalization term；`t0 > 0` 啟用短 readout bias，且較小的正值 bias 較強。`None` 與 `0.0` 都是純平滑 SNR argmax。

---

父層 `experiment/utils/` 的 comment 與 sweep helper 見[父層 README](../../lib/zcu_tools/experiment/README.md#cfg-與父層支援)。儀器角色 helper 位於 `_support/device.py`，見 [lab v2 說明](README.md#實驗角色與-device-操作)。

## v2 工具與實驗輔助

- **`sweep2array(sweep_cfg, name, {"soccfg", "gen_ch", "ro_ch"})`** — 展開 `SweepCfg` 為 numpy array，已套 ZCU 量化（`round_zcu_freq/time/gain/phase`）。Exp 幾乎都靠這個產生 x 軸。
- **`round_zcu_*`** — 單點版本的量化函式；`round_sweep_dict` 同時處理整個 sweep dict。時間的量化有特殊處理：預先減去 `0.5 * one_cycle` 以匹配 QICK sweep 用 `np.trunc` 的行為。多點 sweep 的 step 若在量化後變成 0，會 fast-fail 並要求放大 span 或減少 expts，避免 GUI/agent 看到低階 `SweepCfg` 一致性錯誤。
- **`merge_result_list(list_of_results)`** — 把 `list[dict[name, ndarray]]` 遞迴轉成 `dict[name, ndarray]`（外層 list 變成最外層維度）。`ResultTree.measurement_result(name)` 只對單一 measurement 呼叫它，並快取 stacked view；更新某個 measurement 時不重算 unrelated measurement。
- **`estimate_snr` / `snr_as_signal` / `snr_checker`** — SNR 估計與 early-stop：`estimate_snr` 搭配 `snr_checker` 用於曲線 early-stop；`snr_as_signal(raw, ge_axis, skew_penalty=0.0)` 從 `MomentTracker` 的 g/e IQ moments 計算 pooled-sigma separation SNR。`skew_penalty=0.0` 是純 SNR；提高 `skew_penalty` 會以連續 rational penalty 降低 projected skew 與 g/e shape mismatch 較大的候選點，供 readout optimization 與 JPA optimization 共用。

---

## Lookback & Tracker

- `LookbackExp`（`lookback/core.py`）使用 `run(config, *, context)` 與 `analyze(source, options, *, plots)`。Result 只含 us time axis 與 complex signals，cfg 隨 RunRecord 保存。Run 沿 Schedule 的 decimated acquire，runner reps 強制為1，caller cfg 不改。LookbackAnalyzeOptions 保留 core ratio0.3／smoothNone，分析回傳只有 predict_offset 的 LookbackAnalysis，fit 圖寫入本次 Plots。分析從 magnitude 峰值向前找最後一個低於門檻的 sample，Gaussian smoothing 先作用於 complex signals。Cfg None 仍可分析，預設 saver 則拒絕；canonical Time 軸在磁碟上仍為 seconds。Notebook 使用共用 NotebookAdapter，沒有核心 last_result。
- **`KMeansTracker`**（`zcu_tools.experiment.v2.utils.tracker.kmeans`）：線上維護 `(..., 2)` IQ 樣本的動態多群統計（每群 `cluster_mean` / `cluster_covariance` / `cluster_center` / `cluster_weight`），支援 leading dims 與 `share_axis` 共用群組。內部以增量統計維護每群矩，無需保留原始樣本；singleshot 家族可直接用其 cluster 統計估計 SNR。

---

## Twotone RB 測量策略

`twotone/irb` 是獨立的 paired reference/interleaved 實驗，固定存
`(round, seed, arm, depth)` IQ；arm 0 是 reference，arm 1 是 interleaved。
GUI target 預設 X90，支援 X90、X180、Y90、Y180，沿用校準的 X90/X180 pulse。
兩組使用相同 random Clifford seed，target 在每個 random Clifford 後實際插入，
不合併相鄰 gate；各自計算完整 inverse，沿用連續 virtual-Z frame。
共享序列與 program 建構分別在 `twotone/rb/sequence.py` 與 `program.py`。
IRB 外層由軟體遍歷 round、seed、arm，每個 seed/arm 的 program 快取重用，
program 固定 rounds=1，depth/reps 在硬體完成。Arm 先後順序隨 round/seed 交替。
原 `twotone/rb` 的二維存檔與每個 program 內完成 rounds 的行為維持不變。

IRB analysis 只選完整 round/seed 配對，先平均同 seed 的 rounds，再等權平均 seeds。
兩組共用 IQ 投影軸，各自 fit `A*p**depth+B`；target error 為
`(1-p_interleaved/p_reference)/2`。95% interval 以整條配對 seed 曲線 bootstrap，
不包含 gate-dependent/coherent noise 的系統性誤差。負 error 不 clamp。
中止時未完成的 slot 維持 NaN，原始資料保存；不足兩個 paired seeds 或四個
distinct depths 時分析明確失敗。

- RB 量測採「每個 seed 一個 program」的結構：在 program 內用 sweep index 遍歷 depth，而不是在 host 端對 `(seed, depth)` 重複建立程式。
- recovery gate 是累積 Clifford 的**完整 group inverse**（24 種），不是只把態送回 +Z 的 state-restoring gate：module load 時從 `CLIFFORD_GROUP` 的 6-state permutation（對 24 元 quotient group faithful）程式化生成 `CAYLEY` / `INVERSE_INDEX` 查表，順序約定 `CAYLEY[i][j]` = 先作用 C_j 再 C_i、累積寫 `acc = CAYLEY[next][acc]`，兩處註解互相錨定，改其中一邊必須同步。
- seed 對應的 random gate prefix 長度與 recovery gate id 以 `LoadValue` 從 dmem 查表；random 段用 register-driven `Repeat`；recovery 因 inverse decomposition 最多含 2 個 physical pulse，使用兩個獨立 `ComputedPulse` slot（`recovery_gate_0/1`），不足處以 `BasicGate.Id`（gain=0）補位。兩個 slot 必須共用完整 `gate_pulses` candidate list，使 recovery 段時長與 depth 無關。
- random 段 `Repeat` 固定使用 `n="rand_len"`；當 random 序列可為空時，交由 `LoadValue(values=[])` 的 no-op 行為吸收，不在 RB 層做 sentinel value 邏輯。
- rounds 交給 `acquire()` 原生流程處理，host 層只掃 seed。

---

## 外部中斷支援（cancel_flag）

新 Schedule 寫法由 `ProgramBuilder.build_and_acquire()` / `run_program(...)` 自動把 acquire-local composite `cancel_flag` 傳給 program acquire；它會觀察 `Schedule.stop` 的 external stop，但 data-driven early stop 只停止目前 program acquire，不會把 `Schedule.outcome` 改成 `stopped`。direct ProgramBuilder path 在 external stop / `KeyboardInterrupt` / acquire error 時會保留目前 buffer partial result，並把狀態寫入 `Schedule.outcome`（`completed` / `stopped` / `interrupted` / `failed`）。外部 stop 若在 current round 未完成時被 acquire loop 觀察到，會丟棄該 partial round、保留先前 completed rounds；first round 尚未完成就 stop 時，runner 保留 NaN partial 並標記 `stopped`，不對空 rounds 平均。`failed` / `interrupted` 仍保留 partial result，但本次 `RunContext.scope.stop` 會攜帶第一個非取消錯誤 cause，GUI 與 Notebook 在提交成果前呼叫 `raise_if_error()`，不將失敗的 partial data 當成成功；retry 成功會清除 transient failure cause。

- `AcquireOptions` 從 `zcu_tools.experiment.v2.runtime` 匯入。若使用 SNR early stop，將 `snr_checker(signals_buffer[step], threshold, signal2real_fn)` 傳給 `ProgramBuilder(...).build_and_acquire(options=AcquireOptions(stop_condition=...))`；runner 只在 completed round 寫入 buffer 後檢查，命中時呼叫 acquire-local `cancel_flag.set()`，保留目前 round 並讓 `Schedule.outcome` 維持 `completed`。
- `singleshot/ge/core.py`、`singleshot/check/core.py` 的 raw-shot acquire path 不經 `ProgramBuilder.run_program(...)`，因此在實驗邊界直接傳入 `cancel_flag=sched.stop` 或 `cancel_flag=step.stop`，並把 first-round no-data stop 視為 stopped partial。

---

## 寫新 Experiment 時的檢查清單

1. 定義 `XxxModuleCfg` / `XxxSweepCfg`，通常繼承 `ConfigBase`。Module 欄位用 `Pulse.Cfg`、`Readout.Cfg` 等型別。需要 program 欄位時，`XxxCfg` 繼承 `ProgramExpCfg`，否則繼承 `ExpCfgModel`。
2. 需要 typed record 持久化時，繼承 `PersistableExperiment[T_Result, XxxCfg]` 並宣告 class-level `AXES_SPEC`，single 與 grouped mapping 共用此入口。Result dataclass 只放資料，cfg 由 RunRecord 持有。`from_stored(stored)` 還原已讀取的 native data，`load(source)` 是讀檔包裝；`save_run(source, destination, *, metadata, replace=False)` 寫 native，`save(source, destination)` 另作 Labber 匯出，兩者不猜格式。GUI 透過 adapter 的 restore 與保存內容 hooks 使用同一宣告，見 [GUI contracts](gui-contracts.md#共用-lifecycle-hooks)。
3. 實作 `run(cfg, *, context)`，沿用上方的 RunContext／Plots／Schedule 流程。同步分析另實作 `analyze(source, options, *, plots)`，互動分析由前端 helper 擁有。
4. `ProgramBuilder.build_and_acquire()` / `run_program(...)` 自動注入 Schedule `cancel_flag`；若直接呼叫 `program.acquire(...)`，必須明確傳入 `cancel_flag=sched.stop` 或 `cancel_flag=step.stop`。SNR early stop 走 builder 的 `options=AcquireOptions(stop_condition=snr_checker(...))`。
5. 持久化由 `AXES_SPEC` 宣告。軸用 `AxesSpec.Axis`，signal 用 `AxesSpec.Z`。MHz 轉 Hz 寫 `scale=1e6`，us 轉 s 寫 `scale=1e-6`；原值保存時省略 scale，預設為 1.0。盤上是 SI 單位，記憶體內維持習慣單位。`AXES_SPEC.tag` 取有層次的 on-disk 名字，例如 `"twotone/rabi/len"`，axes 以 inner-first 排列。Native 保存與還原依同一 spec 換算；`save` 的 Labber 匯出不是 `load` 的逆操作。Typed cfg 不相容時 native restore 保留有效 data、回 cfg=None 與 cfg_issue，缺 cfg envelope 或壞資料仍拒絕。來源與失敗契約見 [experiment owner](../../lib/zcu_tools/experiment/README.md#實驗介面與資料)。
6. 如果有多個 sweep 軸，先區分 host loop 與 program loop：host loop 用 `sched.scan(...)` / `sched.repeat(...)` / `sched.batch(...)`，program loop 用 `ProgramBuilder.declare_sweep(...)`；batch child 必須是 replayable callable，buffer 寫入由 child 明確指定，batch 本身不做 per-child retry。host soft sweep 若需要重用每個點的 program，在 `run()` 裡維護 dict：cache miss 時 `builder.build()`，每次量測時 `builder.run_program(program)`。

Singleshot GE/Rabi 的 radius 自動搜尋上限為 g/e 中心距離，手動分類可使用更大 radius。Raw IQ 投影 likelihood 不依賴 radius；derived confusion 與 population 使用圓內且最近中心的互斥區域。GE `consider_other` 僅控制 radius 選擇的 midpoint-other 懲罰模型，回傳校正矩陣維持 isolated-other row。

### Singleshot initial state

`GE`、`len_rabi` 與 `amp_rabi` 的 `initial_state` 是 analysis-only 的 `ground` / `excited` 選項，預設 `ground`，描述 probe / swept drive pulse 之前的主要狀態，不代表純態。GE raw rows 固定是 probe off/on；分析與 confusion diagnostic 共同映射到主要 g/e 順序，持久資料不重排。Rabi joint fit 以 pulse 前 excited population 所在半區間限制初態，對同一 pooled PCA histogram 的兩個方向進行候選擬合，以有效性及 likelihood 選擇，不將第一個非零掃描點當成初態。Population、centers 與 confusion matrix 一律保留物理 g/e 語意；len_rabi 保留可選衰減包絡。`max_calls` 限制每個方向候選的 Migrad call budget；兩個方向都會評估。

GE confusion diagnostic 使用兩組各自 refined 的 preparation populations 建立 initial matrix；不以第一列推定第二列。兩組 refinement 均從 joint fit 的同一組 seeds 出發，避免分析順序影響結果。


兩條 Len Rabi analysis 都提供 `fit_phase: bool = False`，與 `decay` 獨立。
一般 Len Rabi 預設固定 0°/180°（由 signed amplitude 決定方向）；啟用後沿用自由
phase 擬合及包含 phase-frequency covariance 的 π/π/2 length 誤差傳播。
這將一般 Len Rabi 先前的預設自由 phase 改為 opt-in；amp Rabi 不受影響。

Singleshot 模型是 `p_inf + (p_e0 - p_inf) exp(-t/t_r) cos(omega*t + phase)`；
無衰減時包絡為 1。`initial_populations` / `p_e0` 表示 pulse 前狀態；
`zero_length_populations` 是模型外推到 length=0 的值，phase 非零時兩者不同。
`phase` 以 radians 保存、圖上以 degrees 顯示。自由 phase 限制在初態方向的
[-π/2, π/2] branch，以避免 180° offset 與 g/e 交換混淆；此模式另限制 baseline
使振幅方向符合初態，且非負 length 的整條 population 曲線維持 [0, 1]。
因此這個選項用於初態附近的 shape offset，不辨識跨半週期的未知 rotation。
固定模式維持原有 parameterization；free-phase 模式的 backend 額外包含 phase
參數及其 covariance，失敗結果同樣保留所選模型的參數形狀。

`T2RamseyExp.analyze()`與`T2EchoExp.analyze()`提供`fit_phase: bool = False`。
啟用時由既有`fit_decay_fringe`自由擬合phase；停用保留各自的固定相位策略。
此選項僅作用於fringe模式，純decay分析忽略它；T2、detune與誤差仍由所選模型推導。
