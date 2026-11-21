# program/v2

**Last updated:** 2026-11-22，校正 IQ 軸的取數前置條件、`shot_sums` 的分析用途

**讀者：** lab-author、framework-dev、agent。查找 V2 program 公開 API、型別契約與編譯規則的 owner。

## 原生寫法與具名 timeline

`TimelineProgramV2` 繼承 `MyProgramV2` 與 `ProgramResourcesMixin`，共用既有 IR compiler 與 acquisition 擴充。
`TimelineBuilder` 從 package root 匯出，只記錄動作，`build()` 產生並編譯 `TimelineProgramV2`。
它不接收 SoC、不取數，也不依賴 experiment runtime。

| 入口 | 用途與限制 |
| --- | --- |
| `TimelineProgramV2(soccfg, reps=1, initial_delay=1.0, relax_delay=1.0)` | Program 不持有 experiment cfg。Delay 單位為 us，`initial_delay` 為 float，不接受 `None`。Shot 外殼為 `final_wait=0`、`final_delay=relax_delay`。 |
| `_initialize(cfg)`／`_body(cfg)` | 前者宣告通道、pulse、readout、module 與 loops，後者發射原生 QICK 操作或 `emit(module)`。建構子不承接 `AveragerProgramV2` 的全部選項。 |
| `TimelineBuilder(soccfg, ..., **program_kwargs)` | `build()` 原樣轉交 QICK compilation options，例如 `reps_innermost=True`。Compiler 在 build 時回報不支援或衝突的選項。取數選項不放這裡。 |
| `declare(module)`／`emit(module, timeline=..., tag=...)` | 手寫 program 顯式宣告。Builder 第一次 emit 時記錄宣告。同一物件可跨 program 重用，錄製或編譯後不要修改它或其 cfg。每個 program 依物件身分保存自己的宣告結果。 |
| `build()` | 所有錄製 context 關閉後才能呼叫，否則 raise ValueError 且仍可繼續錄製。成功後 builder 凍結，再錄製或 build 都 raise ValueError。 |
| `add_sweep(name, spec)`／`add_table_sweep(name, values)`／`add_loop(name, count)` | 前兩者宣告均勻或非均勻的頂層掃描，回傳參數可傳給 module constructor。`add_loop` 配合 `LoadValue` 保留逐點選擇用途。 |
| `fork(source, into=...)`／`join(*sources, into=target)` | Fork 宣告新 timeline，join 將既有 target 與 sources 的結尾取逐點 max。兩者只記帳，不發射動作或推進 reference。 |
| `with timeline(name)` | 只切換範圍內省略或傳入 `None` 的 timeline 預設名字，顯式名字優先，離開後還原。 |
| `timeline_inc(t, *, timeline=..., range_hint=...)` | 只增加結尾。Scalar／QickParam 單位為 us，register 名稱代表非負 tProc cycles。Hint 是作者保證的整數 cycle 閉區間，不增加硬體檢查。 |
| `delay_by_timeline(timeline=...)` | 將 reference 推進到所選結尾，全部 timeline 扣除實際推進量並 floor 0。原生 delay／delay_auto 仍推進 reference，不受 timeline 範圍影響。 |
| `with repeat(name, n, *, timeline=None, range_hint=None)` | 逐圈累積結尾，入口與回邊不自動同步。`n` 為非負整數或既有 count register 名稱。固定次數不接受 hint。 |
| `with branch(by=..., align_timeline=..., timeline=None) as cases` | 依 sweep 的 zero-based index 選 `with cases.case(index)`。未宣告點執行空 case，重複、非整數或超界 index 報錯。必填 bool `align_timeline` 選全部 case 的 max 或實際 case 結尾，兩者都不推進 reference。 |
| `label(label)`／`jump(label)`／`cond_jump(label, arg1, test, op=None, arg2=None)` | Program 與 builder 支援 `_body` 最外層往後跳。Label 名為非空字串，在 body 內唯一。條件跳轉對 register 做 `Z`／`NZ`／`S`／`NS` 測試；`arg2` 可為 register 或整數。Builder 在 build 時重播並報錯。 |
| `pulse.pulse_name(prog)` | 取得該 program 的 pulse 宣告名稱。Pulse template 按硬體定義去重。相同 generator channel 與取樣定義共用 envelope，不影響 pulse template 的身分。 |
| `timing_layout`／`ir.analyze_timing` | 從 linked boundary metadata 觀察逐點事件、body_end 與含外殼的 shot_end。DMEM lookup 傳公開 `compile_datamem()` 給 `CompiledTimingInput.dmem`。另傳 `ir.compiled_wmem(prog)` 給 `wmem` 可觀察每次 wave 輸出的六個硬體字。未傳時 `wave_words` 為 None。Resync 讀硬體 s11，analysis 明確拒絕。 |

每條 timeline 保存相對 reference 的結尾，`default` 不是 reference。
帶 timeline 的 pulse／trigger 在結尾加 t 發射。Readout 的作者結尾只含 generator waveform 與 pre/post delay，ADC 視窗留在 ADC channel，shot 收尾仍等待全部 ADC。
未帶 timeline 且未進範圍的原生輸出保留 QICK 行為。
`pulse_by_reg` 選到 timeline 時，須提供配對的 `selection` metadata。
同 generator 的 pulse 重疊不能靠另一條 timeline 豁免。帶 timeline 的後一個 trigger 必須證明不與同 ADC 前一視窗重疊，否則在 build 時報錯。
原生 `t=None` 無法定位 ADC 起點，會使後一個同 ADC timeline trigger 無法證明不重疊。此未知標記不捏造 shot 收尾的 runtime 結尾。
Branch 的每個 case 從入口的 ADC 狀態開始編譯。任一 case 在出口的 ADC 結尾未知，Branch 後的同 ADC timeline trigger 就會在編譯時報錯。
Builder 原生布林控制選項只接受 keyword，其餘參數沿用同名 program 方法。

編譯器在 `PathState` 記錄逐路徑的 ADC 讀取數。Branch 的全部 case（含省略的空 case）必須各 ADC 一樣多，否則在 Branch 出口報錯。
固定 Repeat 按次數累加 body 讀取。Register 次數只有在 `range_hint` 上下限相同時才算固定，否則 body 不得讀取 ADC。
Build 後 `ro_chs[ch]["trigs"]` 已是每發讀取數，不用再宣告。`set_reads_per_shot` 只核對傳入值，值不符時報錯。

Case 與 Repeat 只可使用母結構允許的 timeline 及局部 fork，局部名字離開後失效。
Case 內不得經 owner 推進 reference，包含巢狀 Repeat。TwoPulseReset／BathReset 的 children 也有此限制。
`jump` 不保留 fall-through；只允許放 label，直到有可到達入邊的 label 為止。`cond_jump` 的跳走邊與 fall-through 都參與編譯期合併。
目標必須在跳轉之後，不得放在 Repeat body 或 Branch case 內。`with timeline(name)` 內可以跳轉，它只切換預設 timeline。
Label 只保留各入邊都有的 timeline 名稱。已知結尾取逐點最大值，執行期結尾必須相同；每個 ADC 的讀取數也必須相同。
Label 不推進 reference。若各路徑須保留自己的時序，在跳轉或 fall-through 抵達 label 前呼叫 `delay_by_timeline`。
`shot_sums` 不解析跳轉到目標 label 之間的 trigger，請將共用 readout 放在合併 label 後。
`_body` 之外的 label 與跳轉沿用 QICK 行為。Raw `asm_inst` 或自訂 `Macro.translate` 不經 timeline 記帳與 reference 守衛，結果由作者負責。
`DelayAuto` module 只在 timeline 範圍外且未指定 timeline 時包裝原生 delay_auto／delay_reg_auto；任何顯式或範圍選出的 timeline 都報錯。

編譯產物超過 tProc 記憶體時，program 拋出 `TProcMemoryError`，它是 `RuntimeError` 的子類。
從 `zcu_tools.program.v2` 匯入。`memory` 是 `pmem`、`wmem` 或 `dmem`，`used` 與 `capacity` 分別是使用量與容量，單位都是 word。

Generator／ADC 宣告沿用 QICK 的後宣告覆蓋規則，作者負責一致性，需要時讀 `gen_chs`／`ro_chs`。
Timeline module 不自動保護 route，明確呼叫 `protect_readout_routes` 才啟用保護。
不同 owner 重複保護報錯，失敗不保留部分 protection。Register 名稱與地址衝突會列出兩個 module。

完整操作範例見 [experiment authoring](../../../../zcu_lab/v2/experiment-authoring.md#timeline-program-與-schedule-取數)。
跨 owner 與控制流契約見 [ADR-0083](../../../../docs/adr/0083-program-macros-and-timelines.md)，其中包含
[timeline 與 reference](../../../../docs/adr/0083-program-macros-and-timelines.md#timeline-只存結尾與-reference-分開)、
[runtime 範圍證明](../../../../docs/adr/0083-program-macros-and-timelines.md#runtime-end-與範圍證明)、
[branch／repeat 邊界](../../../../docs/adr/0083-program-macros-and-timelines.md#forkjoin-與-repeat-邊界) 與
[收尾 WAIT／slack](../../../../docs/adr/0083-program-macros-and-timelines.md#收尾-wait-與-slack-護欄)。
公開方法的參數與錯誤契約另見 [builder.py](builder.py) 與 [timeline.py](timeline.py)。

Register 次數不能與 Repeat counter 的 `name` 相同。省略 count hint 只略過 generator 回邊的同 channel overlap 證明。
ADC 回邊仍須證明下一圈不早於前一視窗結尾，重疊或無法證明時 raise ValueError。
固定次數或 hint maximum 小於 2 時，兩項回邊檢查都跳過。
`fixed_loop_indices` 提供固定 counter 名稱、點數與 writer metadata，counter 不加入 QICK top-level sweep metadata。

## 查表掃描

`TimelineBuilder.add_table_sweep(name, values)` 與 `TimelineProgramV2.add_table_sweep(name, values)`
建立頂層 loop，回傳 `TableParam`。Program 子類別在 `_initialize` 呼叫。
它等同 `add_loop(name, len(values))` 加 `TableParam(name, values)`，對應線性掃描的
`add_sweep(name, spec)`。兩種掃描依宣告順序由外到內排列，重名會 raise ValueError。
`values` 必須是非空、有限的內建 `int`／`float` 物理值序列，單位由消費參數決定。
其他元素型別（包含 bool 與 NumPy scalar）在建構 `TableParam` 時 raise TypeError，訊息指出值與 loop。
NumPy array 的呼叫端先用 `.tolist()` 轉成內建數字。空表或非有限值 raise ValueError。

```python
from zcu_tools.program.v2 import PulseReadout, TableParam

freq = builder.add_table_sweep("freq", freqs_mhz)
gain = TableParam("freq", gains)  # One gain per frequency point.
builder.emit(PulseReadout("readout", readout_cfg, freq=freq, gain=gain))
```

範例的 `builder` 是尚未 build 的 `TimelineBuilder`，`readout_cfg` 是 `PulseReadout.Cfg`。
`freqs_mhz` 與 `gains` 是相同長度的序列。同一 loop 可有多張獨立的表，不需再宣告 loop。
表長與 loop 次數不符會在 build 時 raise ValueError。
`TableParam` 可與 `int`、`float` 或未掃描的 `QickParam` 做 `+ - * /`，包含反向運算。
結果仍是同一 loop 的新表。與另一個掃描參數運算會 raise TypeError，即使兩表屬於同一 loop。
`TableParam` 繼承 `QickParam`，但沒有仿射的 `start`、`spans`、`raw_param` 或 `derived_param`，讀取這些屬性會 raise TypeError。

QICK 將每個物理值轉成 `TableRawParam` 的硬體字，每點嚴格等於以該值編出的純量 program。
所屬 loop 每圈結尾從 rotated DMEM 表更新 wave 欄位或時間 register。
每發都前進，與該發是否播放該 pulse 無關。最後一點後回到第一點，供下一次外層 loop 或 reps 使用。
同一 waveform 的不同欄位可分別使用線性掃描與查表掃描。

將線性掃描的實際浮點值交給 `add_table_sweep`，不保證重現線性掃描的每個硬體字。
Gain 等截斷取整欄位可能差 1 個最小單位，因為浮點回編不保證冪等。
查表每點仍嚴格等於該浮點值的純量 program。
比較 linear/table parity 前，先以同一 profile 確認每個實際點的純量字等於線性掃描字。
實際點可從 `get_pulse_param(..., as_array=True)` 或參數的 `get_rounded` 取得。

Pulse、PulseReadout 與 Reset 的參數沿用各自的單位與 waveform 限制。
Const 與 flat-top 的 `length` 可查表，flat-top 只更新 flat 段長度。
Arb、gauss、drag 與 cosine 的 length sweep 仍會 raise。
`ro_length` 不支援 `TableParam`，建構 `PulseReadout` 時 raise NotImplementedError，不會進入 build。
`MuxReadout` 不支援查表掃描。
時間參數與 getter 的取整及失敗條件見 [Timeline 取整](#timeline-取整)。

## 型別與 cfg 契約

線性的 `QickParam` runtime 物件提供 `is_sweep`、`minval`、`maxval`、`start`、`spans`、`to_array`，但第三方 stubs 可能未宣告。
Program code 透過 `utils.is_qick_param()`／`QickParamLike` 做 runtime guard 與型別收窄，不各自使用 `cast()`。

`AbsModuleCfg` 與 `AbsWaveformCfg` 是抽象契約。具體 cfg 實作 `build()` 與 `set_param()`，未知參數名稱 raise ValueError。
Pulse、PulseReadout 與 reset 的 keyword-only scalar／sweep 覆寫使用 detached effective cfg，`None` 沿用原值。
Registry identity 與 lowering 使用有效 cfg，`set_param()` 則修改接收呼叫的 cfg。
線性的 `ro_length` sweep 保留 QICK ADC-window 的既有編譯限制。`TableParam` 的拒絕時機見[查表掃描](#查表掃描)。
參數表見 [modules](modules/README.md#constructor-overrides)，跨家族 audit 見 [set_param contract](../../../../tests/cfg_model/test_set_param_contract.py)。

`SweepCfg` 的 start、stop、step 必須是有限值，並與 expts 一致。它不限制單位或 gain 範圍。
`PulseCfg.gain` 與 `PulseReadoutCfg.gain` 共用 `Gain`。建構或更新 cfg 時，純量 gain 必須在 [-1, 1]，單位為 DAC full-scale fraction，a.u.。
Host 迴圈逐點傳入純量 gain 覆寫。超界點在 module 建構時寫入 runtime cfg 就會拋出 `ValidationError`，該點不進 acquisition。
FPGA sweep 的 `QickParam` 仍在編譯時由 `MyProgramV2.add_pulse` 檢查。它在 QICK 轉換前檢查原生呼叫傳入的純量 gain，
以及 `QickParam` 與 `TableParam` 運算後的 `minval()`／`maxval()`。
超出 [-1, 1] 時拋出 `ValueError`，列出 pulse、generator、實際範圍與允許範圍。兩條路徑的診斷不保證相同。
Mux 的 `declare_gen(mux_gains=...)` 不經此入口。Mux tone 的 gain 由 `MuxReadoutTone.gain` 的 `Gain` 欄位限制在 [-1, 1]，建構與賦值時都會驗證。

## 呼叫端 YAML 編譯設定

[`program.build_soccfg(profile)`](../README.md#yaml-compile-profiles) 從 caller-owned YAML mapping 推導時脈。
Channel 數、型別、時脈、記憶體限制與 QICK metadata 都必填，builder 不查已安裝 QICK 版本、不建立硬體或 acquisition source。
Tests 自備 [`tests/program/profiles/`](../../../../tests/program/profiles/)。Lab 板子 YAML 位於 `zcu_lab/simulation_envs/`，由 startup 或範例呼叫端選定路徑。
板子設定與物理模型 YAML 分開。
欄位與數值由各 YAML 擁有，測試的 fixture 與替身慣例見 [tests README](../../../../tests/README.md#qick-編譯-fixture)。

`pfb_channel(soccfg, ro_ch, frequency_mhz)` 透過 QICK 公開 readout 計算解析絕對物理頻率，使用該 ADC 的 PFB 時脈與 channelization。
Caller 不另做 folding 或 rounding。`SimulationSoc` 的 `routes` 宣告 `(gen_ch, ro_ch)` 接線，`drive_generators` 宣告不接 ADC 也能發射 pulse 的 DAC；使用未宣告的 DAC 或未接線的 ADC 路由會失敗。
白噪音環境另接受 `routes="all"`，將每個 generator 接到每個 ADC。
Request、數值契約與環境組裝見 [simulation](simulation/README.md)，startup 與 YAML 入口見 [simulation environments](../../../../zcu_lab/simulation_envs/README.md)。

## 取數與 readout observation

`acquire(..., rounds=1)` 與 `acquire_decimated(..., rounds=1)` 接收軟體平均次數。
`reps` 與 `rounds` 必須為正整數，bool 不算整數，無效值在編譯或 acquire admission 前 raise ValueError。
`TimelineBuilder.register_shot_mask_source(consumed_reads, source)` 記錄來源與它消耗的實體 ADC 讀取，並在 build 時轉交 program。
手寫與 builder program 都在組裝 body 後呼叫已宣告 module 的 `register_shot_mask_sources(prog)`。Module 可解析自己的發射並註冊來源，預設不註冊。Program 在取數時呼叫這些來源。
來源、布林遮罩與取數的契約見 [registered shot masks](../README.md#registered-shot-masks)。
`acquire_summed` 每輪把可見 ADC 的正規化 IQ 與遮罩交給單一 reducer。
框架用 `start(points)` 建立 state 並保存它；round hook 與最終結果都從 reducer 的 `output(state)` 取得。
`shot_sums(prog, axes)` 在 compile 後以 emit tag 或未標記 module 名稱解析 `Classify` 與 `IQ` 軸。
Label、讀取索引與讀取數以移除遮罩來源消耗的讀取後的資料為準。
`IQ(read, offset_corrected=True)` 在加總前對單 ADC、單次積分讀取扣除 QICK IQ offset；其餘 IQ 軸仍使用未扣 offset 的正規化 IQ。
Static PFB 在 reducer 處理該輪時讀當次 acquisition 的 ADC 設定；缺設定拋 `RuntimeError`。
校正零點與 opt-in 邊界見 [ADR-0074](../../../../docs/adr/0074-flat-readout-module.md#選用校正後的-iq-矩)，使用限制見 [summed acquisition](../README.md#summed-acquisition)。
`chain_sweep_sums(prog, read)` 處理單一 fixed Repeat 內的 PulseReadout，回傳每點 12 × 12 augmented sums。
`thermal_chain_sums(prog, read)` 處理同一 PulseReadout 的 16 次連讀，可展開或放在單一 fixed Repeat。
每點回傳 108 欄：shot count 與 107 個 sums，跨保留的 reps 與已完成的 rounds 只加總。
Sums 保留 R1..R16 的 I/Q means、R1 self moments、全部 15 個相鄰對與 (R1,R3)、(R1,R4)、(R1,R5)。
每對保留 II′、IQ′、QI′、QQ′ 四個 components，reducer 不套分析權重。
Caller 安排熱平衡等待與沒有 drive 的連讀；reducer 檢查 owner、read 數與同 ADC 沒有其他 triggers，不檢查 drive 或 timing 內容。
`chain_cycle_sums(prog, read)` 處理每 rep 16 次 PulseReadout 的 cycle，回傳每點 66 × 66 augmented sums；D 排列由 caller 保證。
`rpi_sums(prog, read)` 選一個 fixed Repeat（至少 16 reads）內的 PulseReadout，
每個 rep 是獨立 cycle，lag 1..15 只在 cycle 內配對。
多 setting 時必須將 reps 放在最內層。統計量跨 reps 與 rounds 累加，
35 欄輸出保留完成的 cycle 數與 sums/counts，固定 H=1 的單段長度由總 reads／cycle 數恢復。
Caller 負責 Rπ 排程與無 relax 等待。Rep 之間的斷流不影響 lag，
reducer 不檢查 π 校準，也無法偵測 cycle 內未觀察到的物理斷流。

這些 reducer 在編譯後解析 label；pulse 與 mux readout owners 一起判定唯一性。
Read-count 限制、外積總和與取消契約見 [summed acquisition](../README.md#summed-acquisition)。
`shot_sums` 等 reducer 回傳的 sums 與 moments 是分析輸入，不是誤差估計。
框架不提供通用的 uncertainty estimator；各分析 consumer 依模型選擇估計方法，見[分析子模組](../../analysis/README.md#子模組)。

`readout_observation.observe_readouts(prog)` 由 program owner 解包 compiler wrappers，提供 readout provenance、Trigger 順序及控制 scope。
它靜態訪問 branch 各路與 repeat body 一次，不是 runtime trace 或 linked instructions。
`observe_readout_paths(prog)` 另回傳含 selector／case 與 observations 的 `ReadoutPath`，同 selector 的多次 Branch 共用 case index。
兩種 observation 皆以 `repeat_counts` 保留由外至內的 Repeat 次數，整數代表固定次數，字串代表 count register。
Path 保留 Repeat scope 與未宣告的空 case，不改 flatten traversal。Consumer 保留 module identity、label、read-count 與取數驗證，不自行解析 wrappers。

<a id="timeline-rounding"></a>

## Timeline 取整

具名 timeline 的操作將結果 ceil 到 tProc cycle 一次，事件起點也用 ceil。
原生 QICK 時間轉換保留 nearest，不改成 timeline 的 ceil。
`math.isclose(ticks, nearest, rel_tol=1e-12, abs_tol=1e-10)` 將接近整數的值視為該整數，吸收換算雜訊。
例如 119/430.08 us 保持 119 cycles。在 2³¹ cycles 附近，相對容差也可能吸收約 0.002 cycle 的真實尾段，超出兩項容差的正尾段才向上取整。
靜態 waveform 與 ADC 長度先作有理數時鐘換算，再將 post 加入 aggregate 後 ceil。
線性 sweep 保留 QICK 原生 start rounding 與 step 截斷，不配置逐點時間表。
`TableParam` 的 timeline duration 逐點 ceil，原生 delay 時間逐點沿用 QICK 轉換。
`delay`、`delay_auto`／`DelayAuto(t=...)`、`timeline_inc`、pulse `t`、`pre_delay`、`post_delay`
與 `trig_offset` 可用查表時間。編譯器以 `TableRawParam` 初始化時間 register，再於所屬 loop 結尾載入下一點。
查表 length 保留 generator cycle 格點。Native auto follower 使用 QICK nearest 轉出的 channel 結尾，timeline finish 仍用 ceil。
編譯器保留查表 timeline end 的原始 `TableParam`，供後續 ADC channel end 沿用逐點純量的 QICK nearest 轉換。
非查表 register origin 仍使用 runtime ceil。Branch、fork 與 module 還原的 end 保留原來源，新寫入的非查表 end 不沿用舊表。

Runtime 跨格點 generator 長度使用 k=8 上界定點換算。
Compiler 選 canonical signed-digit 項數最少的 W，平手取較小 W。
500 us 內的合法整數長度滿足 `ceil(n*W/256)` 不早於物理結束且最多晚 1 us，超出 500 us 不沿用此誤差上限。
每個操作加入自己的 post 後才做最後 ceil，靜態與整數仿射 tProc 長度保持精確。
換算只用至多 15-bit shifts 與加減，不用 ARITH、DIV 或時間表。時脈比為 1 不發出換算指令。
乘積或中間值 signed32 overflow 在 build 時拒絕。

現有 native length sweep 與 `Pulse(length=TableParam)` 提供 runtime 長度。
Waveform-address selection 透過 `pulse_by_reg` 播放，置於 timeline 時要求配對 address producer 與候選 waveform 的 selection metadata。
這不新增任意作者 length register 或 ADC length API。Ratio 7/5 的 W=359，ratio 56/5 的 W=2872，但目前 API 未開放 ADC／PFB runtime length consumer。
編譯期已知 bias 使用字面值，零 span 不配置 sweep register。Runtime 結尾保留既有 register 加字面值偏移，只有需要獨立 register operand 的 consumer 才物化。
同步可拆成既有非負 register 與非負字面值的 reference advance，不改事件時間或使用 s14。
時間 getter 依量化後逐點值的可表示性回傳結果，不依來源型別。
常數表與仿射表回傳等值 `QickParam`，不原樣回傳 `TableParam`。
Runtime-only 或非仿射時間 raise RuntimeError，不能表示成線性值。
作者資料表與 IR dispatch jump-address table 仍受支援。

## IR 與硬體對齊

[tProc v2 ASM reference](../../../../docs/reference/tprocv2-asm-notes.md) 擁有指令、register、時序與 waveform memory 限制。
IR 分層、pipeline、pass owner 與 linking 契約見 [IR README](ir/README.md)。
`IRCompileMixin.optimize_asm()` 使用 `make_default_pipeline`，IR 最佳化可透過 `disable_all_opt` 完全繞過，module／macro 已發出可執行 ASM。
`Nop` 保留為顯式指令，不當作 dead-code noise。

`IRNode` 表達 loop、branch、dispatch 與 block 結構；`BasicBlockNode` 保存 labels、線性 insts 與可選末端 branch。
原子 `Instruction` 不繼承 `IRNode`。`IRLinker.link()` 依 `addr_inc` 配置 `P_ADDR` 並解析 labels，最終 QICK dictionary 不輸出 `ADDR_INC`。
`IRLinker.unlink()` 從 QICK `prog_list` 還原含 label 與結構 meta 的邏輯指令。
一般指令佔 1 word；WAIT 展開為比較與條件跳躍，佔 2 words，後續地址為 N+2。

### 打包的 data-memory lookup

tProc v2 ALU 只使用 shift count 的低四位，包含 register operand。單次 shift 無法取出 bit 16 以上的 slot。
`LoadValue` 保持完整 32-bit packing，將擷取拆成至多 15-bit shifts。
Lookup 驗收見 [tests](../../../../tests/README.md#loadvalue-壓縮閾值)，pass 驗收見 [IR pass 測試](../../../../tests/README.md#ir-pass-測試)。
Pass 行為速查見 [simulation](simulation/README.md#ir-pass-behavior)。

## 頻率 word 解碼

`utils` 的 scalar、paired 與 array decoder 接受 `cfg=PulseReadoutCfg`，只借用 generator／ADC channels、mixer frequency 與 Nyquist zone。
它們不修改 cfg，也不使用 waveform、gain 或 readout window。`readout_freq_words` 編碼仍明確接收上述四項設定。
`generator_frequencies_from_words` 接收已取整的物理 mixer 頻率與 generator channel，不建立 readout cfg。
`readout_frequencies_from_words` 接收已宣告 ADC 與同 shape 的物理 reference array，選最近 ADC-DDS image。
Compiled request builder 用 trigger 起點生效的具名 ADC config 供應頻率 reference，不用 carrier。
沒有具名來源的 raw 頻率字會在模擬 request 編譯時報錯，訊息列出 ADC 和 trigger 時間。請改用具名 ADC config。
Cfg-based API 仍委派給相同 word decoder。

## Macro 層

`add_table_sweep("freq", freqs)` 回傳的參數可傳給 `PulseReadout(..., freq=freq)`，由 QICK 逐點轉成 generator／ADC 字。
原生 loop 收尾從 rotated dmem table 更新下一點的 wmem 欄位，最後一點後還原第一點。
更新在 shot timing 後、loop test／jump 前執行，即使該發未播放也會前進，不啟用 DDS phase reset。

`MetaMacro.regs: dict[str, str]` 將 info key 對應到 logical register 名稱、ASM address 或 QICK alias。
`translate()` 透過 `IRCompileMixin.record_meta()` 複製 info 並解析 register，在目前 instruction address 記錄 metadata，不發出 ASM、不修改輸入。
解析失敗不記錄 marker。此路徑不經 QICK `expand()`，info 中的 loop 名稱不會自動解析，必須放在 regs，否則 assembler 會報 `Source Data not Recognized`。

## ComputedPulse 與 DMEM lookup

`ComputedPulse.declare` 宣告候選 waveform 與 wmem-index table。每次 `emit` 捕捉 selector，由同次選擇決定 playback 與 post-delay。
Program owner 保留 waveform 分數尾段，加入所選 post 後取整一次，不使用最大候選 duration。
Address producer、捕捉 selector 與依賴接受全流寫入稽核。

`ComputedPulse.lower` 透過 `LoadValue.lower_selected` 查出候選的起始 wmem index，再交給 `pulse_by_reg`。
候選使用各自的 pulse 名稱，lookup table 保存實際起始 index，不假設候選連續。
Flat-top table 存第一個 ramp_up entry，`flat_top_pulse=True` 播放 ramp_up／flat／ramp_down 三個 entry。
ComputedPulse 拒絕 int4 flat-top 的額外 dummy wave，避免候選 stride 與實際 layout 不符。
`ProgramResourcesMixin` 擁有全程式共享的順序 dmem buffer，`add_dmem()` 回傳 slice 起點。
`compile_datamem()` 維持 QICK 的 `np.int32` array contract。`MyProgramV2` 建立完整 binary program 後，將 remote SoC 的 dmem 轉成一維 built-in `list[int]`，保留 signed int32 與 two's-complement bit pattern。

## Big-jump 閾值

`ir/hw_semantics.py::needs_big_jump(pmem_size)` 集中判斷 `pmem_size > 2048`。
小 PMEM 用 label-mode JUMP，大 PMEM 用 `WriteLabel(s15) + JUMP ADDR=s15`。
`macro/loop.py` 與 `ir/dispatch.py` 匯入同一函式。`UnpackIRBranchPass` 依 `PipeLineConfig.pmem_capacity` 使用相同閾值。

## LoadValue 值表

`LoadValue` 在 dmem 保存整數 lookup table，`auto_compress=True` 且值個數至少 30 時才考慮 bit packing。
`values=[]` 只供上層確定不執行 lookup 的路徑，例如零次 loop。`init()` 依 `use_existed` 宣告 destination register，不配置 dmem 或補值；`run()` 不發指令並回傳傳入時間。這不支援 runtime 讀空表。

未壓縮值域為 `0..2**31-1`，避免 bit 31 使讀回的 register 值變負。
Packed word 可設定 bit 31，超過 INT32_MAX 時以 `word -= 2**32` 存為負 int32，硬體 bit pattern 不變。
Lowering 的 `SR` 加 `AND value_mask` 清除符號擴展，算術或邏輯 shift 都可還原正值。
`bits_per_value` 圓整到 2 的冪次，`values_per_word = 32 // bits` 小於 2 時不壓縮。因此 max value > 65535 時不壓縮。

## ArbWaveform 參考時間軸

`ArbWaveformCfg` 保存邏輯 data key 與必填本機絕對 path，length 是該 NPZ 的 `time[-1]`，不另存播放長度。
`ModuleCfgFactory.from_core` 透過明示 `context.assets.path_for` 轉換 nested arb，成品不持有 provider。
Program 與 sim 從 path 讀 bytes，交 `core.waveform_format` 解碼 samples，不查 global DB 或解讀 formula。
`ArbWaveform.make_iqdata` 在完整 reference time axis 插值到硬體 timing，不伸縮或裁切。覆寫同 path 會改變之後播放內容。
