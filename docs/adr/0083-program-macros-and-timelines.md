---
status: accepted
date: 2026-10-29
---

# 0083 — Program macro 與具名 timeline

Builder 契約更新：2026-10-30。Timeline、Repeat、收尾證明與全面遷移裁定更新：2026-11-01。
查表掃描與 native 時間來源裁定更新：2026-11-02。
相關：[[0072]]、[[0074]]、[[0078]]

## 脈絡

Program 作者要在同一個 body 混用原生 QICK 操作與可重用的 Pulse、Readout、Reset，也要排定跨 channel 的事件。Module 若繼承 Macro，每次發射的狀態就會混入資源宣告。具名 timeline 若把 default 綁到 reference，省略名字和明示名字又會遵循不同的推進規則。

## 決策

Program 擁有資源與發射時序。Builder 只組裝 program，Schedule 擁有實驗的取數政策。全部實驗使用 `TimelineBuilder`／`TimelineProgramV2`，不保留舊 program 或 control module。

### Program 基底與 module owner

**規則。** `TimelineProgramV2` 繼承 `MyProgramV2` 與 `ProgramResourcesMixin`，沿用 `SimulationAcquireMixin`、`ImproveAsmV2`、`IRCompileMixin`、classify／herald acquisition 及唯一的 [compiled simulation frontend](#compiled-snapshot-與-simulation-入口)。原生操作與 module lowering 共用 IR 管線與 `TimedMergePass` 的安全條件，不按來源分流或另做 canonicalization。Module 描述資源需求，不是 QICK Macro。Program 以 module 的物件身分保存宣告結果，每次 emit 產生新的 `ModuleEmission`，時間與 tag 屬於該次發射。Module owner 提供 declare 與 lowering，不在物件上保存跨 program 宣告狀態。

**例子。** 手寫 program 在 `_initialize` 建立並 `declare(module)`，在 `_body` 用 `emit(module, timeline=..., tag=...)`。同一 module 可宣告到兩個 program，重新 compile 只重建當前 program 的宣告。用 `Pulse("probe", cfg.probe, freq=freq)` 套用數值或 sweep，不必先複製 caller cfg 或處理 waveform 路由。[Module 原生寫法](../../lib/zcu_tools/program/v2/modules/README.md#原生寫法)及[建構子覆寫](../../lib/zcu_tools/program/v2/modules/README.md#constructor-overrides)說明有效 cfg、`None` 與各參數的限制。

**邊界。** Module 不能傳給 `append_macro`，也不由 cfg 自動生成或透過 handle 使用。只有 preprocess 或 expand 確實不同時，才另建 macro 類別。`ProgramResourcesMixin` 只集中 DMEM 配置與 readout route 保護。Generator／ADC 重複宣告沿用 QICK 的後者覆蓋；program 不檢查兩次設定是否相等，也不另設查詢介面。作者讀 `gen_chs`／`ro_chs` 查設定，明確呼叫 `protect_readout_routes` 才啟用保護。Register 名稱或地址衝突由 program 檢查，錯誤指出兩個 module 的名稱與物件。明示借用與 Pulse template 共用不變。tProc readout 用各自具名的 config 在 runtime 切換 freq／phase／outsel，沿用 QICK `add_readoutconfig` 的 template 契約。每 shot 依[編譯出的 trigger 數](#每發-adc-讀取數)取數，不按宣告或單一 module 的發射次數限制共用。
Program／emission owner 配置每次 lower 的內部名稱。Scope 每次 compile 重建，巢狀發射或 lower 失敗不污染呼叫端；builder 不組這些名稱。LoadValue owner 重用 table 與 destination，提供當次 index 的 lookup。作者用 Repeat 或頂層 sweep index 存取，不複製或改寫 loader 宣告。固定 Repeat 的 counter metadata 由 program 提供，不混入 QICK 頂層 sweep。入口與資源衝突見 [ProgramV2 README](../../lib/zcu_tools/program/v2/README.md#原生寫法與具名-timeline)，各 owner 分工見 [modules README](../../lib/zcu_tools/program/v2/modules/README.md#design-boundaries)。

**依據。**
- 2026-10-29，使用者說：「我覺得不太好，用program寫法就該接近原生的寫法，而builder才該設計成一個with scope，內部是像body的寫法」。手寫入口保留原生寫法，builder 是否用 `with` 依[下一節](#builder-與-schedule-的職責)的後續裁決。對分開宣告和發射答「那用A好了，但代價應該就是會出現很多新macro與module」；對 cfg 直接生成 macro 答「A'的機制可能比較複雜，收益相比A比較小，除非有很一致且簡單的declare機制，否則還是顯式module好了」。見 [module 身分與作者入口](../discussions/2026-10-29-module-to-macro.md#module-身分與作者入口)。
- 2026-10-30，對 program 持有宣告結果答「我也傾向B1，原因是對作者而言最小驚訝」。使用者也說：「11其實我覺得set_param有點複雜，不夠python原生，作者不容易直接看出來要怎麼寫新實驗。」並選「11B」，用建構子 keyword 覆寫。原始討論未入庫，宣告 owner 見 Git `e57265396`，builder 更新見 `e999cab42`。
- 2026-10-31，使用者對宣告檢查說：「既然做这些检查这么的麻烦，那就干脆不检查了，就跟原生机制一样就好。」取消自動 route 保護、保留 register 檢查與原生 template 契約是 designer 代裁，使用者未覆核。Program 與 module owner 的其他代理細節依 2026-10-30 授權，同為 designer 代裁，使用者未覆核。見 [宣告與時間模型的討論](../discussions/2026-10-31-timeline-variable-model.md)及 Git `3a8adfa93`。

### Builder 與 Schedule 的職責

**規則。** `TimelineBuilder` 只記錄序列和第一次 emit 的 module 宣告，`build()` 產生 `TimelineProgramV2`。Schedule 與 scan step 用 `run_acquire`、`run_acquire_summed`、`run_acquire_decimated` 接收已建好的 program，program 在 SoC 之前。`RunOptions` 管理轉換後寫入 slot、即時顯示、retry 與 `stop_condition`。Program 自身的 `acquire*` 只取數並回傳。逐 shot 遮罩屬於 program 的取數層，`ImproveAcquireMixin` 持有來源，`acquire`／`acquire_summed` 在每輪完成後呼叫來源。

**例子。** Builder 第一次 emit 記錄宣告，對同一物件的第二次 emit 沿用宣告。手寫 program 與 builder 產物交給同一個 Schedule 入口，不各寫一套 retry。Builder 收到 `register_shot_mask_source` 只錄下，在 program 的 `_initialize` 依錄製順序轉交；手寫 program 直接註冊。組完 body，program 依宣告順序呼叫各 module 的 `register_shot_mask_sources(prog)`，預設 hook 不註冊來源，program 不檢查具體型別。Herald reset 在 lowering 標記讀取，到 hook 才解析各路徑的實體索引，每次發射註冊一個來源。[Program README](../../lib/zcu_tools/program/README.md#registered-shot-masks)記錄遮罩的取數契約。

**邊界。** Builder 不是 context manager，不接收 SoC 或依賴 Schedule，不按內容合併不同 module 物件。資源共用或衝突由各家族或 program 處理。Schedule 與 scan step 不保存 cfg，run 讀實驗自己的 cfg；步驟差異以區域變數或 `with_updates` 表達。Schedule 不呼叫來源或套用遮罩。`run_acquire_decimated` 與 `run_acquire` 在 retry 前透過 `check_shot_mask_options` 讓 program 拒絕含來源的 decimated 或原生 `threshold`。沒有 precheck 的 producer 視為沒有來源。

**依據。**
- 2026-10-30，使用者：「我認為builder也許該專注於build program，能與sched解偶就解偶，讓acquire由sched提供方法而不是builder」。確認時說：「採用，除此之外，builder應該不用with寫法，他沒有要管理生命周期的需求吧？program可以顯式呼叫構建，」。對物件身分與資源共用答「採用建議」。入口依「然後，我突然想到run_acquire與RunOptions比managed這名字好懂」及「用run option就好」命名。原始討論未入庫，見 Git `e999cab42`。
- 2026-10-30，使用者確定參數順序：「沒問題，採用建議，我思考soc/prog的順序，感覺prog應該要是第一個才對」。2026-11-01，使用者答「B, 我要睡了，預期起來後我規劃都落地」，將 Schedule 移除 cfg 與實驗 cfg 寫法納入遷移。原始討論未入庫，見 Git `7cab54392`。
- 遮罩的使用者裁決見 [[0074]]〈Herald reset 與逐 shot 遮罩〉。2026-11-20 review 後使用通用 hook 的理由見 [herald reset 討論後記](../discussions/2026-11-20-herald-reset-shot-masks.md#後記)。

#### Round 在取數端歸約

**規則。** 一般實驗把 round 數交給 acquire。Reducer 加總或平均已完成的 rounds，Result 不留 round 軸。這是建議，不是強制。

**例子。** `two_qubit/rb` 與 `two_qubit/rb_heralded` 加總分類計數，`twotone/rb` 平均 IQ。每支 seed／arm program 只取數一次。

**邊界。** 需要逐輪資料，例如 trace by trace 平均時，作者可以把 round 迴圈放在最外層。

**依據。** 2026-11-20，使用者：「其实结果的 round 的轴是多余的，我想应该是之前实做的人的瑕疵。round 的轴应该在 round 函数内就已经被平均掉了，而且呢，这个平均应该是 reducer。负责的吧？」同日補充：「這是建議而不強制，如果有強烈的trace by trace平均需求，實驗作者當然可以將round做在最外面，但一般實驗使用acquire內建的rounds就好了」。比較見 [RB 取數歸約](../discussions/2026-11-20-rb-round-reduction-and-fit-quality.md)。

### 查表掃描與 QICK 協定

**規則。** `add_table_sweep` 與 `add_sweep` 都建立頂層 loop，回傳的參數交給既有 consumer。`TableParam` 沿 QICK `QickParam.to_int` 協定轉成 `TableRawParam`，各點硬體字等於該點純量編譯的結果。Generator 與 readout manager 不用另建編碼路徑。`MarkedCloseLoop` 在同一個原生 loop 收尾處處理查表 waveform 及 register，再交回 QICK 的線性 sweep 收尾。

**例子。** 從 rotated DMEM 表載入下一點。即使分支沒播放該 waveform，loop 也更新其查表值。最後一點後載回第一點，下一個外層 loop 或 rep 可以照常開始。[ProgramV2 README](../../lib/zcu_tools/program/v2/README.md#查表掃描)示範同一 loop 用頻率與 gain 兩張表。

**邊界。** `TableParam` 可與純量或未掃描的 `QickParam` 算術，不可與另一個 sweep 參數運算，也不能當成零 span 的純量。`ro_length` 不支援查表，`PulseReadout` 在建構時就拒絕；`MuxReadout` 不支援查表掃描。單位、waveform 限制與錯誤型別見[查表掃描](../../lib/zcu_tools/program/v2/README.md#查表掃描)與 [module 覆寫](../../lib/zcu_tools/program/v2/modules/README.md#constructor-overrides)。

**依據。** 2026-11-02，使用者：「我的想法是：add_table_sweep能支援的與add_sweep一致，實驗方式也接近，都在loop尾端更新對應register」。對沿 QICK 四層接線答「採用提案」。型別、算術、拒絕時機與 native 時間來源是同日 designer 代裁，使用者未覆核。原始討論未入庫，見 Git `f633572b0`。

### Timeline 只存結尾，與 reference 分開

**規則。** 每條 timeline 只存相對 `ref_time` 的結尾 E，下限為 0。`default` 是普通 timeline，不是 reference，實體 channel timestamp 也不是具名 timeline。範圍外的 emit 省略名字時選 default。一般 emit 從所選結尾放置事件，按規則 b 用實際 lowering 結果更新結尾，不讀 `total_length` 或串接原先重疊的事件。`timeline_inc` 只增加所選結尾。`delay_by_timeline` 按各點實際結尾推 reference，不以 symbolic max 或統一長度代替。跨零或 max 交叉時也按各點真值算。Reference 推進 d 後，含 default 在內的每條結尾變成 `max(0, E-d)`。只有作者明示操作及每 shot 收尾會推進 reference。

**例子。** Readout 作者 timeline 只含 pre-delay、generator waveform 與一次 post-delay。最後取整，不先各自取整尾段。ADC 視窗只記到該 ADC 的物理結尾，不 join 回作者 timeline。九個 chain 實驗因而不用 `delay_auto(ros=True)` 等 ADC，只等 generator 結束加 `post_delay`。上一視窗可與下一個 pulse 的發射重疊，因為下一個 pulse 尚在飛行；同一 ADC 的 trigger 仍不得重疊。[讀出鏈間隔討論](../discussions/2026-11-21-chain-readout-spacing.md)另記 pulse 長度與緩衝。

**邊界。** `with timeline(name)` 只替換省略或為 `None` 的預設名。Emit、pulse／trigger、兩種 readout config、register playback、timeline 增量與同步、Repeat／Branch 的母 timeline 都遵守這個預設。顯式名字優先，巢狀範圍退出時還原，未知名字報錯。Scope 不 fork、join 或推 reference，case 另用自己的 timeline 屏蔽外層。範圍外未指定 timeline 的原生 pulse／trigger 照 QICK；原生 `delay`、`delay_auto`、`wait_auto`、`resync` 不加 timeline keyword。
帶 timeline 的 pulse／trigger 在 E+t 發射，pulse 結尾到 generator 結束，trigger 結尾只計 trigger 寬度。兩種 readout config 在 E+t 瞬時生效，不加 config 長度或 ADC 視窗。Delay 與 SoftDelay 只加結尾。Reset 留各 child 的 post-delay，沒有 composite post；BathReset 的 pi/2 在 cavity end 加 post-delay 並向上取整後開始，取 qubit 與 pi/2 的實際結尾合流。實際發射的長度與 [nominal duration](../../lib/zcu_tools/program/v2/modules/README.md#readout-configuration) 分開看。
另開 timeline 不豁免同 generator 的可證明重疊。帶 timeline 的後續 trigger 必須能證明不早於同 ADC 前一視窗結尾；原生 `t=None` 留下不可定位的 ADC 結尾，任一 Branch case 帶出此結尾都會攔下後續同 ADC 的 timeline trigger。純原生的後續 trigger 仍走 QICK。Shot 收尾仍等待所有 ADC。
DelayAuto module 是唯一的 emission 例外，只在範圍外且未指定 timeline 時包裝原生 delay_auto／delay_reg_auto，保留 gens／ros 並推 reference。其他情況在 emit 報錯。Repeat 內可明用原生操作，case 與 Reset 的守衛照常。原生 delay_reg_auto 先消耗選定實體 channel，再推進非負 register cycles；尚未消耗的 timeline 依 producer bounds 扣實際推進量，全消耗後則沿用原生非負時間契約。PulseReadout 的 runtime gain／frequency words 不是時間 consumer，不受這項時間限制。不提供 `delay_auto_timeline`，也不為 `delay_timeline`／`sync` 留別名。

**依據。**
- 2026-10-29，使用者：「原生qick其實有timeline機制，只是其timeline固定綁定到channel, 且與ref_time是分開的…我們新設計的timeline，其實就是不綁定channel的原生timeline」。並選「我倒是認為只要存當前結尾就好，不用存開始，我傾向B1」。見 [B1 時間模型的討論](../discussions/2026-10-29-module-to-macro.md#timeline-模型不綁定參考時間只存結尾)。
- 2026-10-30，使用者對 scope 答「1a 採用建議」。2026-10-31，使用者以「7,8,9,13照建議」確認 timed API 的一致性及刪除 `qick_threshold_options`。命名選「名字我考慮：timeline_inc, delay_by_timeline，這樣有delay的方法都是動ref的比較一致」，並說「ModularProgram那個是沒有rintime timeline功能時的無可奈何辦法，新program有就不要模仿了」。原始問答未入庫，命名見[時間模型討論](../discussions/2026-10-31-timeline-variable-model.md)；交付見 Git `3a8adfa93`、`870d10f17`。
- 2026-11-01，使用者：「18b請用gen的就好，adc讓其在獨立timeline不join，因為adc目前不影響後續pulse」。對編譯期報錯答「B」。原始討論未入庫，見 Git `57f83c88e`、`2e85c8e3d`。Timeline 的 owner 邊界與 lowering 細節是同日 designer 代裁，使用者未覆核，授權原話見[範圍證明的依據](#runtime-end-與範圍證明)。
- 2026-11-21，使用者：「B''，ii，應該改的是readout module計算timeline的方式，不join ro window，然後實驗cfg要多預留先緩衝pulse長度，不要對齊到極限。」實作沿用 module 既有的不 join 規則。見 [讀出鏈的 pulse 長度與間隔](../discussions/2026-11-21-chain-readout-spacing.md)。

### Runtime end 與範圍證明

**規則。** 時間與結尾在 `[0, 2**31)`，中間算術須符合 signed32。Program／timeline owner 保存 producer bounds、來源及實體 channel 的編譯期結尾式，並稽核翻譯後的所有寫入及初始化。只在實際發射時計入 channel，reference 推進時扣減並 floor 0；Repeat／Branch 保留迭代與路徑關係。這些是 WAIT／overlap 的證明資料，不是第二個 scheduler，也不用 channel-end register。

**例子。** Register 次數的 Repeat 中，先執行的 LoadValue 提供該圈 table 值域，後續 ComputedPulse 或時間 consumer 可用；沒執行的圈或別的路徑不建立定義。Runtime 整數起點 R 加小數 f 後，再放 t 的事件時用 `R+ceil(f+t)`。單次操作內合計實際起點、長度與 post，再取整，不把不同操作合成一次 ceil。原生 QICK 的 scalar channel timestamp 仍用 nearest。[Timeline 取整](../../lib/zcu_tools/program/v2/README.md#timeline-取整)記錄容差、2³¹ 附近約 0.002 cycle 的效果、查表與線性 sweep、getter 及跨格點換算。

**邊界。**
- `timeline_inc` 的 scalar／QickParam 是 µs，register 名稱是非負 tProc cycles。作者可省略 `range_hint=(lo, hi)`。只有既無 bounds 又無 hint 的作者時間段才記 lo=0、hi 未知。該段 max 用 runtime 真值，overlap 與 overflow 由作者負責，不加有限上界或硬體檢查，也不只因無法證明就拒絕。既知的常數、sweep、table 或 hint 仍供檢查與省指令使用。Producer 本身仍須已知、已初始化且未越界；錯誤指出 register 與指令。這個時間視圖不改已捕獲來源的身分，也不縮窄非時間的 signed 值或 uint32 words。Getter 只回實際可量化表示的 scalar／仿射值，不以 ndarray 或 bounds 上界代替，runtime-only 或非仿射值報 `RuntimeError`。
- Native pulse 起點若有原 `TableParam`，先加實長再按 QICK nearest 換算，不逐段 ceil。Trigger 查完整 scheduled end 的查表來源才可用同一物理時間做 ADC tail 的 nearest；混入其他 sweep／register 或沒有表來源時仍採 runtime ceil。`TimelineBook` 用 `RuntimeEnd` 物件 identity 與強引用記來源，compile reset 清掉記錄。恢復舊 end 可查回來源，新寫入非查表 end 不借用舊表。只保存原表或允許的查表加靜態值結果，不從 register 名、點值或 proof recipe 猜來源，也不改 `RuntimeEnd`、`CycleProducer` 或 Branch 合流。
- 靜態 generator fabric／ADC 長度用有理數時鐘換算。既有 runtime length sweep 與 selected waveform／length table 用 k=8 上界定點換算，範圍為 2²³ tProc cycles，超限與 signed32 overflow 在 build 報錯。500 µs 內不得早於物理結尾，最多晚 1 µs，超過不承諾此誤差上限。不開一般 length register 或 ADC length API，換算不用 ARITH 或除法器。
- 證明按軸 ID 對齊時間分量。已知軸存一維值，未知軸存範圍與係數，常數是大小 1 的固定軸。加減合併同軸、消去零項；逐點全等歸到常數軸，常數結尾不配 end register。跨格點常數 bias 用字面值，零 span 不配 sweep register；runtime 結尾只差一個常數時沿用原 register，consumer 要獨立 operand 才物化。同一已知軸的非線性逐點算，跨軸 ceil／定點換算用有界誤差軸，不建網格。跨軸 max／min 以線性部分及有理係數的 Abs 項表示，容許負係數及巢狀 Abs。取界按共用軸分連通塊，塊內分支定界；每塊超過 20 個未定號 Abs 項才退回區間界，不改真值。不可變節點快取範圍，支配判斷快取正規化差式，`RuntimeEnd` 的界由式子推出。
- 合流共用 N 路 max／min。展平同向候選，提出共項，合併相同餘項，按差值界刪掉被支配者，只在剩下的 k 項做 k−1 次選值。物化入口讀已捕獲的 register，不重算證明式或重讀被覆寫來源。差值能證明在 signed32 內時，max 用 TEST、複製 x、`if(S)` 寫 y，min 用 `if(NS)`；可覆寫 x 時省複製。k 路選值是 `1+2(k−1)` 條，不含候選求值與共項。不能證明差值安全時用 sign-corrected mask，max 11／min 12 條、3 個 scratch，不依賴溢位後的 S。UF 後由硬體 hazard stall，不加 NOP 或用外部 F／NF。合法非負時間的差不會溢位；無上界仍可能走 mask，但溢位修正沒有合法公開 consumer，min 也沒有公開 codegen consumer。這兩條內部路徑與成本用直接審閱，不宣稱已有端到端驗證。
- 常數事件用 `@t`；runtime 事件寫 s14，經 preprocess 設 `t_regs["t"]="s_out_time"`，讓 QICK 展開 Pulse、ConfigReadout、Trigger，不另配事件時間 register。同基準後續事件只加差值。Native macro／raw asm 寫 s14 或 Branch 合流時，下一次須載入基準；timeline 自我複製 s14 不使記錄失效。原生掃描仍用 QICK sweep register。這個接縫不複製 expand、不加 QICK 私有呼叫、不 monkeypatch 或更換 QICK。
- 不自動配置時間或 selected-end DMEM。作者明示的查表掃描、資料／waveform table 與 IR jump-address table 照常使用。Branch 已是 tProc cycles 的結尾以整數乘加與 max／選值合流，不建逐點網格或分段插值。

**依據。**
- 2026-10-31，使用者：「F3同意，range_hint是可選的」。2026-11-01，未知上界與保留已知來源由 designer 代裁。2026-11-18，使用者過目此條說「我大概看了一遍，基本上是没问题」，並說這類規則「并不算是裁决，而算是建议」。這些細節仍是僅供參考的先例，不當成使用者裁決。LoadValue 的同圈定義是 2026-11-01 designer 代裁，使用者未覆核，不是一般未知 register 的 fallback。原始問答未入庫，見 Git `3a8adfa93`、`1f4b16aa0`。
- 2026-10-31，使用者說「你永遠都不能先加總再ceil，誤差會很大」，補充「純delay的應該也是量化」。對定點換算答「那k=8,我知道為什麼優先用dmem了」，要求「500us任意長誤差晚1us內，不可以早」。2026-11-01，以「B，而新設計沒有查表fallback吧？」確認相對容差方案。精確有理數時間模型另行評估。原始問答未入庫，見 Git `3a8adfa93`。2026-11-02 的查表 native timestamp 與 ADC tail 來源細節是 designer 代裁，使用者未覆核，見 Git `f633572b0`。
- 2026-10-31，使用者說「計算的原理是軸比對，源變數只是軸數為1的變數而已」，並答「我也傾向Abs寫法，而且k可以開足夠大，因為一次只求一個節點的範圍，還可以考慮快取」。要求「所以应该是要设计一个多个线性式都可以适用的 join 的算法」，提醒「要確定一下這個flag真的會影響結果，有可能是假flag。」對記憶體說「Dmem資源很珍貴，除了作者明確知道有用的位置，其他地方少用」，並答「Ir跳躍那個可以，因為屬於log級別的使用量，我接受」。對事件時間要求「我想進一步討論timeline利用s14實現的通用方案，而不只是repeat」。見 [Timeline 變數、join 與 s14 的討論](../discussions/2026-10-31-timeline-variable-model.md)，其中實作前的 mask 成本不是現行規則。
- 2026-10-31 的實體 channel 記帳，以及 2026-11-01 的 signed32 guard、mask 成本、preprocess 與失效細節是 designer 代裁，使用者未覆核。2026-11-01 的代理授權原話是「1代理模式就都給你決定，我事後看就好」。字面 bias 與 register 位移也由 designer 依此代裁，使用者未覆核；目的在無同步 chevron 及遷移後 RB／IRB 不超過 16 個 data register。原始工作檔未入庫，見 Git `3a8adfa93`、`05aadfb44`。
- 2026-10-29，使用者：「ABs/asr是安全可用的，可以不用測」。這不是已完成實機驗證的聲明，原始問答未入庫。

### 收尾 WAIT 與 slack 護欄

**規則。** Body 先在 runtime 以真實 maximum 同步所有 timeline 與 ADC 實際結尾，再記 `body_end`，最後進 shot 外殼。Shell WAIT 是編譯期常數，不用 variable WAIT，也不以上界替代 runtime 同步。先配對 raw 與已消耗的結尾再相減取界，保留共同來源與逐點相關性，不枚舉多維笛卡兒積。

**例子。** ADC end 是 O+190、body end 是 O+200，消耗後剩餘 0。O 的範圍不增加 WAIT。剩餘量為常數時，WAIT 用精確值，不警告；即使該常數超過 `relax_delay` 也不拒收。

**邊界。** 有精確 min／max 時 WAIT 用 max，`slack=max-min` 以 tProc cycles 表示合法點的實際差距。Slack 為 0 不警告；大於 0 時指出變數與差距；等於 `relax_delay` 仍只警告，超過才 raise `ValueError`。只有安全界時，警告說明 slack 是 upper bound，不據此拒絕。可消去的未知來源先消去，仍保留其他已知 ADC 來源與逐點資料；沒有上界的實際 ADC 結尾照樣參與同步，不拒絕或假造上界，已消耗的 tail 為 0。Abs 預算用盡只會放寬證明界，不能把下界當真實 min 或改 runtime 同步。

**依據。** 2026-10-30，對常數 WAIT 答「A」，補充「WAIT可以用常數，但sync要考慮runtime」；對未知 slack 不按上界拒絕另答「A」。此處 sync 的現名是 `delay_by_timeline`。2026-11-01，對未知 ADC 結尾沿作者時間規則處理答「3A」。來源配對與相減取界是同日 designer 代裁，使用者未覆核，不加入一般 CFG analysis。原始討論未入庫，見 Git `ae2632f13`、`3a8adfa93`。

### Fork、join 與 Repeat 邊界

**規則。** `fork` 從已存在的 timeline 複製結尾到一個或多個新名字，`join` 只把 target 與 sources 的 max 寫回 target，不發射或消耗 sources。Branch 按 sweep 的 zero-based index 選 case，不支援依量測結果選擇。`align_timeline=True` 以所有 case 的結尾 max 合回母 timeline，`False` 用實際 case 的結尾。Repeat 逐圈累積母 timeline，入口與回邊不同步，零圈不執行或宣告 body。兩種結構都把省略時的 scope 預設或 default 作母 timeline，body 用自己的 timeline 屏蔽外部預設。

**例子。** `with cases.case(index) as name:` 從母 timeline 隱式 fork `"<by>[<index>]"`，沒宣告的點是空 case。True 模式的未選 case 不執行，結尾必須能從入口重建；全靜態用靜態 max，否則按點算 runtime max。兩種模式都不推 reference，不改其他外部 timeline。Branch／Repeat 後可直接在外部 timeline 發射，從實際剩餘結尾用 s14 放事件，不以範圍上界放置，也不必先 join、fork 或同步。

**邊界。** 只有 default 預先存在。Fork 重名或使用未宣告名字報錯，不要求每條 fork 都 join 回來。`align_timeline` 是必填 keyword-only bool，重複、非整數或超界 case index 報錯。Case 只能用自己的 timeline 和局部 fork，不能用母 timeline 或外部名字。Repeat 只能用自己的母 timeline 與局部 fork，其他外部名字也不能作 fork source。局部名字離開即失效，未 join 的局部 end 丟棄，外部可重用名字。Branch 區塊內、case 外不能放操作；case 不能經 owner 推進 reference，連巢狀 Repeat 也一樣，等待改用 `timeline_inc`。不在 case 內時，Repeat body 可以明寫 `delay_by_timeline`、原生 delay／delay_auto 或巢狀 Repeat。跨 Branch 的 timeline 先在入口前宣告。
只有 Repeat body 頂層最後一個操作是針對循環 timeline 的 `delay_by_timeline`，且入口結尾為常數 0，才省略尾端 register。固定圈長用常數 `inc_ref`，其餘用變數尾端／s14。若以該操作收尾，但入口結尾不是常數 0，編譯器發出 `UserWarning`，提示作者在 Repeat 前同步。編譯器不自行插入同步。原生同步不算作 `delay_by_timeline`。不做定點分析、重播 preprocess 或按後續事件數自動推 reference。固定 n 為非負整數，不收 `range_hint`。Register n 不能和 counter 同名，非負閉區間 hint 可省略，不從初值猜上界或新增硬體檢查。回邊的 generator 與 ADC 檢查見 [ProgramV2 README](../../lib/zcu_tools/program/v2/README.md#原生寫法與具名-timeline)。
每 shot／sweep 點重建 loop 狀態，counter 不進頂層 sweep metadata。作者明寫推進時外部結尾扣每圈實際推進量並 floor 0，否則不變。固定且可靜態表示的總量保持靜態；runtime 次數或路徑用逐圈 debit storage。後續 pulse、trigger、config、emit 或 `timeline_inc` 可沿用原本為 shot 收尾、join、fork 或同步所需的 selected-end／debit storage，不預配事件 register。仍受 16 個 data register 上限及不得自動配置時間 DMEM 的限制，超限保留 QICK 原錯誤。

**依據。**
- 2026-10-29，使用者：「我覺得join的工作好像可以簡化，不是平行好幾個branch並合併，而是單純的將給定的幾個timetime結尾合併到指定目標timeline」。並要求「那會需要一個宣告新timeline的機制，從當前指定timeline尾端分岔」。原始問答未入庫。
- 2026-10-31，使用者：「case()輸入應該要有對應的值，值沒有宣告對應case應該視為空branch」；「我會預期branch實際長度可以選擇所以case的最大值，或是runtime時走的那個branch」；「我不想自動推進ref time」；「局部fork自動並回是超出預期的行為」。對命名提出「branch初始化帶timeline參數應該就不用再一層with timeline的寫法，你覺得case隱式建立的timeline名叫什麼，感覺with case as timeline_name是好設計，且一樣屏蔽預設timeline」，並答「45確認」。原始問答未入庫，見 Git `dfc7bfa87`、`3a8adfa93`。
- 2026-10-31，使用者：「repeat屏蔽預設timeline，尾端就是一個變數，這樣內部操作自然變成基於這個變數」。並問「讓作者顯式在body想要的位置插入delay或sync效果是否一樣？」對依結構選 lowering 答「確認」。具體 lowering 是 designer 代裁，使用者未覆核，見 [Repeat 與 IR 的討論](../discussions/2026-10-31-timeline-variable-model.md#repeat-與-ir-展開)。
- 2026-11-01，對外部 timeline 不再只許等待、可從實際結尾繼續發射，使用者答「1B」。Storage 重用是同日 designer 代裁，使用者未覆核。原始討論未入庫，見 Git `3a8adfa93`。

#### 往後跳轉與 label 合併

**規則。** `TimelineProgramV2` 在 `_body` 最外層提供 `cond_jump`、`jump` 與 `label`，builder 只記錄並在 build 轉交同名動作。目標必須在後方，迭代用 Repeat 或 sweep。Branch、Repeat 與作者 label 共用不可變 `PathState` 保存結尾、ADC 讀取數與證明，`TimelineBook` 擷取與還原，各控制結構定義自己的合流。Label 合併所有可到達的入邊與 fall-through，不替作者同步。

**例子。** `cond_jump` 記目標入邊並繼續編譯原路，不求值條件。`jump` 後不能放動作，直到有可到達入邊的 label。可跳到 `_body` 最後的作者 label 提早結束本發，shot 外殼仍收尾。若要保留各路徑自己的時間，先在入邊用 `delay_by_timeline` 同步，再進 label；否則 max 會影響下一個動作與 shot 收尾的放置。

**邊界。** 各邊編譯期的 timeline／generator／ADC 結尾取 max，sweep 交叉時以 sweep register 重建逐點最大值。Runtime 結尾在各邊不同就報錯，列出 label、timeline 或 channel 與入邊，不在邊上補 register 寫入。只留各邊共有的 timeline 名稱。未知 ADC 結尾取聯集，producer 定義取交集，selection 證明依 Branch 規則。Source 表達式各邊均有且一致才保留；未另定合併的欄位必須相同。每個 ADC 讀取數須[對齊](#每發-adc-讀取數)。
Label 名非空且唯一，編譯器避免撞到機器 label。位置不合法、重複定義、缺目標或往回跳在 build 報錯並指出 label。Repeat／case 內不能跳，`with timeline(name)` 只改預設名，不限制跳轉。條件只用 register 的 `Z`、`NZ`、`S`、`NS`，不用 peripheral flag。`_body` 外仍照 QICK，raw `asm_inst` 不經 timeline 記帳。`shot_sums` 等 label consumer 不沿跳轉邊列舉讀取；在跳轉與目標之間的 trigger 會被拒絕並指出目標，區間外共用讀取可解析。這不改 `trigs` 或 raw IQ。尚未實作的 continue／break 見[草稿](draft/control-flow-completeness.md)。

**依據。** 2026-11-20，使用者：「真正通用的机制应该是 condition jump 跟 label。」同日答「可以接受统一只能往后跳」，並以「都采用建议」接受其餘第一版規則。最外層限制、補定的合併、拒收與模擬細節是 designer 代裁，使用者 2026-11-20 追認：「確認」。見[往後跳轉的討論與覆核](../discussions/2026-11-20-timeline-forward-jumps.md#後記)。

#### 量測能力不抵可讀性成本時不加分支

**規則。** 新增的量測範圍不足以抵銷第二條編譯或取數路徑的閱讀成本時，保留一條路徑，超出容量就報錯。

**例子。** Two-qubit RB 在板上掃 depth，每個 seed、arm 編一支 program，取數前全部編完。較淺 depth 在自己的前綴後播放 recovery，同步到該路徑的結尾，再跳到共用讀取。最深路徑直接接讀取。PMEM 對半分塊的深度收益與額外編譯、主機迴圈和 slice 寫入成本見[量測比較](../discussions/2026-11-20-rb-depth-program.md#被否決的選項與理由)。

**邊界。** 實際要量的深度以較少點仍放不下，而且增加的可量範圍有明顯收益時，才重評額外路徑。

**依據。** 2026-11-20，使用者裁定：「拿掉粉块，它带来的效益没有比可读性的破坏来得多」。見 [板上 RB 程式改寫](../discussions/2026-11-20-rb-depth-program.md)。

### 每發 ADC 讀取數

**規則。** Timeline compiler 在 macro preprocess 取得 QICK 的逐 ADC trigger 增量，原生 trigger、module readout 與原生 macro 的內部 trigger 走同一路徑。`PathState` 累計各路徑的讀取數。每個 Branch 出口與作者 label 的可到達入邊（含 fall-through）須逐 ADC 相同。編譯器寫結果到 `ro_chs[ch]["trigs"]`，不重寫 Branch 或 Repeat 指令。

**例子。** 兩個 case 各讀一次，本發只算一次。若其中一個是空 case，就在本次 Branch 出口報錯，列出各 case 的計數；不能用同 selector 的第二個 Branch 補讀。Label 不符時列出 label、ADC 與各邊計數。

**邊界。** 固定 Repeat 的出口是入口加上圈數乘單輪增量，`range_hint` 的上下限相等也算固定；其他 runtime 次數的 body 不得讀 ADC。`set_reads_per_shot` 只核對，不改配置。Raw ASM 的 trigger 不經編譯器記帳，CPU 模擬器仍逐發比較實際 trigger 數與 `trigs`。

**依據。** 2026-11-20，使用者選擇在每個 Branch 出口對齊，而不允許後續 Branch 互補，答「3-1」。見 [Timeline 編譯器推算每發 ADC 讀取數](../discussions/2026-11-20-timeline-read-counts.md)。Label 的合併依[往後跳轉的覆核](#往後跳轉與-label-合併)。

### 並行序列與 reference 守衛

**規則。** 作者用 builder／program 從共同 origin 為各條 sequence fork timeline，依作者順序 emit，再 join 回母 timeline。Join 只併結尾，不發射 children 或重排序列。分支與迴圈用 program 的 `branch`／`case` 與 `repeat()`，不用 control module。

**例子。** TwoPulseReset 或 BathReset 的 child，以及 Branch case，若呼叫會推進 reference 的 owner 接縫，build 報錯。守衛看實際推進，不按 macro 名或效果猜，也不把會推進的 sequence 移到最後。

**邊界。** Producer provenance、全流寫入稽核、實體結尾與 case／Repeat 回邊照常檢查，不加 padding 或 rewind。Raw `asm_inst` 或自訂 `Macro.translate` 若直接推 reference，不經 timeline 記帳，守衛與 Repeat 契約都不保護，作者負責。

**依據。** 2026-10-31，使用者對統一並行寫法答「f12同意，刪掉，不要有兩種寫法」。同日的 fork／join lowering 與 reference owner 守衛是 designer 代裁，使用者未覆核。原始討論未入庫，見 Git `3a8adfa93`。

### Compiled snapshot 與 simulation 入口

**規則。** IR compiler 在成功 compile 後發布不可變、具型別的 `compiled_stream`，保存優化前的 logical instructions、labels 與結構 metadata。手寫與 builder program 共用此契約。Compiler 擁有內容，不借用可變的 module／macro graph，不依賴 simulation 或 observer callback；simulation 不管理 compile lifecycle。`SimulationAcquireMixin` 對 `SimulationSoc` 使用唯一 compiled request builder，明傳 SoC routes。Builder 交付 detached numeric requests，物理 backend 仍由 numeric owner 負責；其他 SoC 照 QICK 取數。

**例子。** 發布 body 邊界、LOOP／BRANCH／SOURCE、已解析的起點／步長與有限 table definitions，由 frontend 對照指令。編譯器為作者跳轉附 `CONTROL_LABEL` 與 `CONTROL_JUMP`，不寫入 binary。Label 標記貼著機器 label，jump 標記在指令前，只帶 `target`、`conditional` 與 `test`，無條件時 `test=None`。Frontend 核對最外層、唯一的前方目標、TEST 與跳轉指令，包括 direct JUMP 和 `WriteLabel(s15)` 加間接 JUMP。門檻見 [QICK big-jump](../../lib/zcu_tools/program/v2/README.md#big-jump-閾值)。同一個向量步驟中，跳走的 lanes 等到目標才恢復，其他 lanes 照常執行；等待中的 lane 不更新時間、register、memory、FIFO 或輸出事件。

**邊界。** 支援與否看 instruction effect，不按 macro 類別或來源白名單。Frontend 不從 raw JUMP、仿射 shell 或 module tree 猜結構，metadata 與指令不符、無標記的 body JUMP、依硬體 flag 或量測回填 register 的跳轉，都在送 request／呼叫 backend 前拒絕。結構標記沒有通用 payload、`kind` 或 `region`。各 sweep point 獨立，reps 只在 metadata，不執行；非 sweep loop 依標記與 active mask 迭代，不展開。同值或輸出不變也不能讓 register、memory、readout config FIFO 或 pending update 跨 point／rep；prologue 後未再寫的常數或 config 可用。它保留 shot-local 運算、記憶體與 FIFO 時序，逐 lane 核對實際讀取數，不做回饋或 tProc latency 模擬，不提供 ordered engine 或 legacy fallback。[IR snapshot](../../lib/zcu_tools/program/v2/ir/README.md#pre-optimization-compile-snapshot)記存取、複製、recompile 與位址；[Simulation README](../../lib/zcu_tools/program/v2/simulation/README.md#compiled-request-builder)記 admission、routes、FIFO 與指令效果；[module owner](../../lib/zcu_tools/program/v2/modules/README.md#design-boundaries)擁有來源名稱。

`SimulationSoc` 在 request 生命週期持有取消查詢。`install_request` 收下該次 acquire 的旗標，`release_request` 清除，`run_round()` 把同一查詢交給 backend 和 pacing。QICK 0.2.436 的 decimated round 入口是 `start_tproc()`，accumulated 是 `start_readout()`；兩個入口都經 `run_round()`。QICK 簽名不改，旗標不放進 `AcquireRequest`。Backend 只讀查詢，不寫 caller 的旗標。Backend 拋 `Cancelled` 時，`SimulationAcquireMixin` 在 round 邊界清理中斷輪，不留下該輪資料。零完成輪拋 `StoppedPartialAcquireError`；已有完成輪則照 QICK finalize 回傳其平均。實機路徑不變。

**依據。**
- 2026-10-29，使用者：「優化是一定要做的，你可以安排流程」。對 compiled frontend 答「A, 不是瓶頸就好」，對舊 frontend 答「舊前端不留，第二題你決定」，對運算支援答「運算的採用建議」。以指令效果判定，取代早期 runtime 運算禁令，早期範圍見[原型討論](../discussions/2026-10-29-module-to-macro.md#模擬器的範圍)。
- 2026-10-30，使用者：「每個內部迴圈都當作獨立點向量化模擬，但內部非sweep迴圈會視為一個shot內部的order，但不會單純的展開」。對獨立 shot 與拒絕跨 shot carry 答「採用建議」；通用標記暫緩後答「那採用建議做法」，用有限的 LOOP metadata，跳轉也用有限標記。Snapshot 存取、routes 與 FIFO 細節是 designer 依 compiled frontend 授權代裁，使用者未覆核。上述 compiled 選項的原始問答未入庫，交付見 Git `182b9bede`。
- 2026-11-20 的跳轉標記與 lanes 行為是 designer 代裁，使用者 2026-11-20 追認：「確認」。見[往後跳轉的討論後記](../discussions/2026-11-20-timeline-forward-jumps.md#後記)。
- 2026-11-21，使用者答「3C，但設計上5秒內」，選擇讓 decimated 的取消進入 FTF 求解。取消傳輸與零輪轉換由 designer 依授權代裁，使用者未覆核。見 [FTF Stop 討論](../discussions/2026-11-21-ftf-stop-checkpoints.md)與 [simulation acquisition 契約](../../lib/zcu_tools/program/v2/simulation/README.md)。

### IR analysis 與驗收責任

**規則。** `program/v2/ir` 擁有唯讀時序 analysis，不讀 frontend 私有 macro 或從 module `total_length` 猜行為。Program owner 透過既有 IR metadata 傳遞 body boundaries 及 loop 名稱／次數，依最終 linked addresses 公開為 `timing_layout`。Loop 資料只標掃描座標，不能代替指令中的初值、更新與跳轉。驗收以合法輸入和現行 B1／規則 b 獨立手算逐點事件以及 body／shot 結尾，不把新舊結果相等、指令數、register 排法或逐位相同當正確性。

**例子。** 由同次編譯的 `prog_list`、labels、layout 與 `compile_datamem()` 建 `CompiledTimingInput`。每次到 `body_start` 開始一發，下一次 `body_start` 前的 reference 是前一發 `shot_end`，最後一發用程式終點，含 `final_wait`／`final_delay`。另報 `body_end` 供觀察 body 本身推進的量。事件 cycle 從初始 reference 0 起算。驗證 WAIT 時，從公開事件與 `body_end` 加上獨立換算的 generator／ADC 時鐘和長度求剩餘量；隨機和固定案例驗每點都夠等，能證明精確時用精確 max，並核對 slack 的 warning 與 relax 拒絕邊界。例見 [Compiled timing 驗收](../../lib/zcu_tools/program/v2/modules/README.md#compiled-timing-驗收)。

**邊界。** Analysis 不模擬 CPU latency，不能單獨算 waveform 或 ADC end，也不是正式 WAIT guard 的 oracle。Dmem 用 signed int32 words，位址是從零起算的 word offset，None 或空資料不是零填。只讀已知且在範圍內的地址；缺資料、未知位址、越界、寫入或不可決定的控制流均報錯。Caller 不重建 dispatch table 或掃 opcode 猜邊界。Resync 讀硬體 s11，analysis 拒絕，不假設為 0。已知 register 時間及唯讀 DMEM 仍可分析。IR 窗口看字面值 s14 增量及 TIME inc_ref；register 增量使窗口未知，不展開。`TimeInst` 不讀寫 s14，自我複製交給通用 peephole，不在 IR 放 timeline 特例。一次性遷移報告不是現行 gate，不為還原舊指令排法新增 scalar＋sweep 合併。參見 [IR analysis 契約](../../lib/zcu_tools/program/v2/ir/README.md#compiled-timing-analysis)。

**依據。**
- 2026-10-29，使用者：「第一件選A，語意與行為一致就可以了，舊設計的怪癖不一定要對齊」，並強調「我沒有要求必須與舊的一致，我要求的是與預期行為一致」。見 [驗收改為預期行為的討論](../discussions/2026-10-29-module-to-macro.md#驗收基準改為預期行為)。
- 2026-10-30，一次性遷移報告由當時的 advisor 依使用者授權代裁，使用者未覆核。原始工作檔未入庫，報告規則與例子見 Git `3a8adfa93`。
- 2026-11-01，IR／program 的分工是 designer 代裁，使用者未覆核。見 [s14 與 IR 的討論](../discussions/2026-10-31-timeline-variable-model.md#s14-與-ref)及 Git `3a8adfa93`。

### 現行交付範圍與裁定沿革

**規則。** 全部實驗用 Timeline program；module lowering 採規則 b，runtime timeline、builder／Schedule 分工與 compiled simulation frontend 都是現行路徑，不是階段開關。

**例子。** `MuxReadout.collect_iq`／`collect_raw_iq` 從 Timeline program 已編譯的 readout emission 取數，保留平均、ADC 長度正規化、offset、tone／ADC 順序與輸出 shape。降低 register 用量後，四個 chevron／constructive_phase 實驗的硬體與 host 路徑共移除八處 drive join 後同步。舊 cfg 仍能編譯，公開編譯事件時間逐點不變。

**邊界。** 不保留 `ModularProgramV2`、`BaseCustomProgramV2`、`ProgramBuilder`、`Schedule.prog_builder`，也沒有 `Join`／`Branch`／`Repeat`／`ScanWith` control module。Module 作者不用 legacy `init`／`run`／`block_mode`／`tag` 或 `set_param`；每次 emit 指定 tag，數值覆寫走建構子。原生 program 仍有分支與迴圈能力。舊 Module-tree frontend 不留 selector、oracle 或 fallback。

**依據。**
- 2026-11-01，使用者裁定「不要留Join」，並要求「ModularProgram連同Join/Branch/Repeat/ScanWith都刪掉，全面用新的TimelineProgram，我應該早就這樣要求了」。同日對不影響時序的同步答「13沒影響就移除多餘的同步」。原始問答未入庫，見 Git `7cab54392`、`57f83c88e`。
- 2026-10-29 的階段安排見[原型與遷移討論](../discussions/2026-10-29-module-to-macro.md#驗收基準改為預期行為)。當天回答「1A  2B 3A」中，以 register 名稱傳入 timeline cycles 的設計仍適用；Module Join 與當時的 Mux 取數範圍由全面遷移取代，見 Git `bbcd98bd5`。2026-10-31 重新排定大量實驗遷移，原始問答未入庫，見 Git `7cab54392`。

## 依賴與後果

Program 組裝序列，experiment runtime 擁有取數政策。Experiment 可依賴 program，program 不反向依賴 experiment。沿用 [[0072]] 及 `.importlinter` 的 C8、C9、C12、C19 與 `program-simulation-entry`。IR analysis 只分析編譯結果，不產生模擬量測資料。模擬遷移見 [simulation-environments](draft/simulation-environments.md#compiled-frontend-與小介面依賴)。

作者須明示何時同步 timeline 與原生 reference，不能用未同步的原生 pulse 代替序列發射。無法靜態表示的時間由 runtime timeline 承接，未知的作者時間段由作者負責；仍有 16 個 data register 上限。[原生 program 寫法](../../lib/zcu_tools/program/v2/README.md#原生寫法與具名-timeline)與 [module 宣告](../../lib/zcu_tools/program/v2/modules/README.md#原生寫法)見各 owner README。
