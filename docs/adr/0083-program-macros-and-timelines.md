# 0083 — Program macro 與具名 timeline

日期：2026-10-29。Builder 契約更新：2026-10-30。Timeline、Repeat、收尾證明與全面遷移裁定更新：2026-11-01。
狀態：Accepted。全部實驗使用 TimelineBuilder／TimelineProgramV2，legacy program 與 control module 已移除。
相關：[[0072]]、[[0074]]、[[0078]]

## 脈絡

Program 作者需要在同一個 body 混用原生 QICK 操作與可重用的 Pulse、Readout、Reset。
跨 channel 的排程也需要獨立於 QICK channel timestamp 的具名時間線。
若以 Module 繼承 Macro 表達兩者，同一物件的資源宣告與每次發射狀態就會混在一起。
若把 default timeline 綁到 reference，省略 timeline 的 module 與具名 module 又會遵循不同的推進規則。

## 決策

Module 與 builder 的 owner 細節由 advisor 依使用者 2026-10-30 代理授權裁定。
Compiled snapshot 的存取形式、routes 與 FIFO 細則由 advisor 依下述 compiled frontend 授權裁定。
使用者的設計選擇另附日期與逐字原話，與 advisor 的細節裁定分開記錄。
2026-11-01 的 timeline 整理由 advisor 依使用者「1代理模式就都給你決定，我事後看就好」的代理授權確認，涵蓋下述 owner 邊界、證明模型與 lowering 細節。

2026-11-01，使用者裁定：「不要留Join」，並要求：「ModularProgram連同Join/Branch/Repeat/ScanWith都刪掉，全面用新的TimelineProgram，我應該早就這樣要求了」。
全部實驗使用 `TimelineBuilder`／`TimelineProgramV2`。`ModularProgramV2`、`BaseCustomProgramV2`、`ProgramBuilder`、`Schedule.prog_builder` 與 control module `Join`／`Branch`／`Repeat`／`ScanWith` 已移除。
Module 的 legacy `init`／`run`／`block_mode`／`tag` 與 `set_param` 不再是作者介面。Tag 由每次 emit 指定，數值覆寫由 constructor keyword 指定。
並行只用 builder／program 的 `fork`／`join`，分支用 `branch`／`case`，迴圈用 `repeat()`。這不移除原生 program 的分支與迴圈能力。
使用者同日選擇「B, 我要睡了，預期起來後我規劃都落地」，將 Schedule 移除 cfg 與實驗 cfg 寫法慣例納入遷移。

### Program 基底與 module owner

`TimelineProgramV2` 繼承 repo 的 `MyProgramV2` 加 `ProgramResourcesMixin`。
它承接既有 `SimulationAcquireMixin`、`ImproveAsmV2`、`IRCompileMixin` 與 classify／herald acquire 擴充。
`TimelineProgramV2` 使用唯一的 compiled simulation frontend。
支援範圍依指令效果判定，不代表所有 native 操作均可模擬；admission 契約見 [simulation README](../../lib/zcu_tools/program/v2/simulation/README.md#compiled-request-builder)。
原生操作與 module lowering 共用既有 IR 管線及 `TimedMergePass` 的安全條件，不按來源分流或另建 canonicalization。
共用 `ProgramResourcesMixin` 只集中 dmem 配置與 readout route 保護。
`TimelineProgramV2` 的 generator／ADC 宣告沿用 QICK，後宣告覆蓋先宣告，由作者負責設定是否一致。
它不做宣告等值檢查或提供查詢介面，作者需要資訊時讀 QICK 的 gen_chs／ro_chs。
Timeline module 不自動保護 readout route，只有作者明確呼叫繼承的 protect_readout_routes 才啟用保護。
Register 名稱／地址衝突仍由 program owner 檢查，錯誤指出兩個 module 的名稱與物件。
明示 register 借用與 Pulse registry 的 template 共用規則不變。
tProc readout 仍以各自具名 config 在 runtime 切換 freq／phase／outsel，add_readoutconfig 保留 QICK template 契約。
每 shot 取數以編譯出的 trigger 總數為準，不按宣告或單一 module 的 emit 次數拒絕共用。
2026-10-31，使用者裁定：「既然做这些检查这么的麻烦，那就干脆不检查了，就跟原生机制一样就好。」
取消自動 route 保護、保留 register 衝突與原生 template 契約的細節由 advisor 依代理授權整理，builder 不解析 module 內部資源。

作者在 `_initialize` 建立並 `declare(module)`，在 `_body` 呼叫 `emit(module, timeline=..., tag=...)`。
Module 描述資源需求，不繼承 QICK Macro，也不能直接傳給 `append_macro`。
原生路徑由 `TimelineProgramV2` 以 module 物件身分為 key 持有宣告結果，宣告時寫入，lowering 從該 program 讀取。
同一 module 物件可同時宣告進多個原生 program，手寫 program 與 `TimelineBuilder` 都適用。
重新 compile 只重建該 program 的宣告結果，不覆寫其他 program 的資源。

2026-10-30，對宣告結果改由 program 持有的 B1 方案，使用者回答：「我也傾向B1，原因是對作者而言最小驚訝」。
Builder 深複製只解決 builder 的重用，且會分離原物件與複本的身分，手寫 program 不受益。
切換 program 時重新綁定也無法保留多個 program 各自的宣告結果，因此兩者都不能取代 program ownership。

作者以 `Pulse("probe", cfg.probe, freq=freq)` 這類 constructor keyword 傳入數值或 sweep。
支援覆寫的 module 建構子先複製 cfg，再套用覆寫並保留有效 cfg，caller cfg 不變。
`None` 沿用原值。
Registry identity、read axes 與 lowering 都使用有效 cfg。
作者不必只為套入掃描值先 deepcopy cfg，也不必知道 waveform 的內部路由。
各 module 的 keyword、單位與限制見 [modules README](../../lib/zcu_tools/program/v2/modules/README.md#constructor-overrides)。
2026-10-30，對掃描值的作者寫法，使用者指出：「11其實我覺得set_param有點複雜，不夠python原生，作者不容易直接看出來要怎麼寫新實驗。」
使用者選擇「11B」，採用建構時 keyword 覆寫。

每次 emit 建立新的通用 `ModuleEmission`，時間與 tag 屬於那次發射。
各 module owner 提供宣告與 lowering，module 不保存跨 program 的宣告狀態。
只有確實需要不同 preprocess／expand 行為時，才另建 macro 類別。

使用者於 2026-10-29 對原生作者入口指出：「我覺得不太好，用program寫法就該接近原生的寫法，而builder才該設計成一個with scope，內部是像body的寫法」。
對宣告與發射分開的選項，使用者選擇：「那用A好了，但代價應該就是會出現很多新macro與module」。
這裡的 A 指顯式 module 宣告、每次 emit 產生 macro，不是交付階段名稱。
以通用 emission 加 owner lowering 避免每種 module 都需要一個對應 macro 類別。
對 cfg 直接產生 macro 的替代方案，使用者裁定：「A'的機制可能比較複雜，收益相比A比較小，除非有很一致且簡單的declare機制，否則還是顯式module好了」。
Module 家族沒有一致的 cfg-only 宣告入口，部分 owner 還需要 cfg 以外的參數，因此不從 cfg 自動建 module 或加入 handle 層。
原生 program 仍顯式宣告 module；下節的 builder 自動宣告作者交來的 module 物件。

Program／emission owner 配置每次 lower 的內部名稱，scope 隨 compile 重建，nested emission 與失敗返回不污染呼叫端。
平行排程由作者或 module owner 以 fork／emit／join 表達，保留作者次序。
Builder 只重播公開方法，不自行組 emission 的內部名稱。
LoadValue owner 提供本次 index 的 lookup，重用已宣告的 table 與 destination。
作者以原生 `repeat()` 或 top-level sweep index 配合該 lookup，不複製或改寫 loader 的宣告狀態。
Program 的 repeat owner 提供固定 counter 的名稱、次數與 writer metadata；nested loop 不加入 QICK 的 top-level sweep metadata。
這些分工讓 emission 狀態與宣告資源分開，也避免呼叫端鏡射 QICK 私有 discovery 流程。

### Builder 與 Schedule 的職責

`TimelineBuilder` 位於 program 層，只記錄序列並以顯式 `build()` 產生 `TimelineProgramV2`。
Builder 不管理取數生命週期，不是 context manager，也不接收 SoC 或依賴 Schedule。
它在 module 物件第一次 emit 時記錄宣告，同一物件再次 emit 沿用宣告。
不同物件不按內容合併；硬體資源的共享與衝突交給各 module 家族和 program owner。

Schedule 與 scan step 不保存 cfg。實驗 run 讀取自己的 cfg 參數，步驟差異由區域變數或 `with_updates` 表達。
Schedule 與 scan step 的 `run_acquire`、`run_acquire_summed`、`run_acquire_decimated` 接收已建立的 program。
它們以 `RunOptions` 管理轉換後寫入 slot、即時顯示、retry 與 stop_condition。
Builder 產生的 program 與手寫 program 使用相同入口；program 自身的 `acquire*` 只取數並回傳。
序列組裝不需知道這些執行政策，作者也不必為手寫 program 另做 retry 與停止邏輯。

使用者於 2026-10-30 提出：「我認為builder也許該專注於build program，能與sched解偶就解偶，讓acquire由sched提供方法而不是builder」。
確認職責切分時，使用者補充：「採用，除此之外，builder應該不用with寫法，他沒有要管理生命周期的需求吧？program可以顯式呼叫構建，」。
同日，對 module 以物件身分宣告、資源由各家族共用的方案，使用者回答：「採用建議」。
取數入口最後依使用者「然後，我突然想到run_acquire與RunOptions比managed這名字好懂」及「用run option就好」命名。

### Timeline 只存結尾，與 reference 分開

採 B1 模型。每條 timeline 只存相對 `ref_time` 的目前結尾 E，下限為 0，不另存開始或 channel end 集合。
`"default"` 是普通 timeline，不是 reference。一般 emit 省略 timeline 時選 default，不移動 reference。
範圍外未帶 timeline 的原生 pulse／trigger 保留 QICK reference timing。
不建立每 channel 一條預設 timeline，原生 channel timestamp 保持原用途。

使用者於 2026-10-29 說明定位：「原生qick其實有timeline機制，只是其timeline固定綁定到channel, 且與ref_time是分開的…我們新設計的timeline，其實就是不綁定channel的原生timeline」。
對儲存狀態裁定：「我倒是認為只要存當前結尾就好，不用存開始，我傾向B1」。

`with builder.timeline(name)` 與 `with program.timeline(name)` 只切換範圍內省略 `timeline=` 或傳入 `None` 時的預設。
`emit`、pulse／trigger、兩種 readout config、register playback、`timeline_inc`、`delay_by_timeline` 及 Repeat／Branch 的母 timeline 都遵守此規則。
顯式名字優先，包含 `"default"`；巢狀範圍覆蓋外層，離開時還原。
名字須是 default 或已 fork 的 timeline，未知名字報錯。範圍不 fork、不 join，也不推進 reference。
Branch 的 case 另以自己的 timeline 屏蔽預設，不能藉外層 scope 使用母 timeline。
原生 `delay`／`delay_auto` 推進 reference，不受這個預設影響。
2026-10-30，使用者對預設名字、顯式值優先、巢狀還原與 program 同名入口回答：「1a 採用建議」。
2026-10-31，使用者以「7,8,9,13照建議」確認同族 timed API 的 scope 一致性與刪除 `qick_threshold_options`。

一般 emission 先取所選 timeline 的結尾作為內部事件的共同起點。
完成後採規則 b，不從 `total_length` 重建 duration，也不把 composite 內原本重疊的事件串接。
單一 post-delay owner 取實際 waveform／ADC event end 的最大值，加一次 post-delay，再對完整結尾向上取整到 tProc cycle。
分數尾段保留到 aggregate 收尾，不先各自取整。
Reset 保留各 child 的 post-delay，不新增 composite post。
BathReset 的 pi/2 在 cavity end 加 post-delay 向上取整後開始，最後合併 qubit 與 pi/2 的實際結尾。
Delay 與 SoftDelay 都只加結尾，不推進 reference。
帶 timeline 的 pulse／trigger 在 E+t 發射，結尾更新到事件結束點。
兩種 readout config 是瞬時事件，結尾更新到 E+t，不加 config length 或其他 ADC window。
同 generator 可證明的 pulse 重疊在編譯期報錯，另一條 timeline 不能豁免此檢查。

`timeline_inc(t, timeline=X)` 只加 X 的結尾，`delay_by_timeline(timeline=X)` 才將 reference 推到 X 的結尾。
Reference 推進 d 後，全部 timeline 包含 default 一起變成 `max(0, E-d)`。
原生 delay／delay_auto／wait_auto／resync 沿用 QICK，不增加 timeline keyword。
不提供 `delay_auto_timeline`。跨零或 max 交叉時按各點真值計算，不補成統一長度。
2026-10-31，使用者選定命名：「名字我考慮：timeline_inc, delay_by_timeline，這樣有delay的方法都是動ref的比較一致」。
舊 `delay_timeline`／`sync` 入口不保留別名。Reference 只因作者明確要求的操作及每 shot 收尾推進。

DelayAuto module 是一般 emission 規則的唯一例外。
它只在 timeline 範圍外且未指定 timeline 時包裝原生 delay_auto／delay_reg_auto，保留 gens／ros，移 reference 並扣減全部結尾。
顯式指定任何 timeline，包含 default，或由範圍選出 timeline，均在 emit 入口報錯。
在 Repeat 等範圍內需要此效果時，作者直接呼叫原生操作，仍須遵守 case 與 Reset 的 reference 守衛。
原生 delay_reg_auto 先消耗選定 native channels，再推進非負 register cycles。
尚未完整消耗的 timeline 依 producer bounds 扣除實際推進量；全部已消耗時，register 部分沿用原生非負時間契約。
PulseReadout 的 gain／frequency runtime words 是非時間 consumer，不受此時間限制排除。

使用者於 2026-10-31 指出：「ModularProgram那個是沒有rintime timeline功能時的無可奈何辦法，新program有就不要模仿了」。
新 program 的 runtime delay_by_timeline 依逐點實際結尾對齊，不模仿 legacy DelayAuto symbolic max 權宜行為。
已移除的 legacy program 行為不構成 Timeline 的相容性要求。

### Runtime end 與範圍證明

`timeline_inc` 的 scalar／QickParam 單位為 us，register 名稱代表非負 tProc cycles。
Timeline 結尾與作者時間的合法範圍為 `[0, 2**31)`，中間算術須符合 signed32。
有已知 bounds 時，編譯器照常檢查 overflow 與可證明的同 channel 重疊。
作者自行準備的 register 時間可不提供 `range_hint=(lo, hi)`。
只有既有 bounds 與 hint 都不存在的時間段，才以作者的非負契約記 lo=0、hi 未知。
該段 max 在 runtime 算，overlap 與 overflow 由作者負責，不加有限上界或硬體檢查，也不因無法證明而拒絕。
已知的常數、sweep、framework table 與 hint 仍供證明及省指令使用；hint 不會關閉既有資訊。
這個時間視圖不改 signed 非時間資料，也不改已捕獲來源的身份。
2026-10-31，使用者裁定：「F3同意，range_hint是可選的」。lo=0／hi 未知與保留已知來源的細節由 advisor 於 2026-11-01 依代理授權裁定。

Program／timeline owner 擁有 framework producer bounds 與翻譯後的全流寫入稽核。
原生常數、sweep、loader table 與 playback owner 提供來源關係、合法 index 及依賴。
這些已登記來源仍須符合寫入與初始化契約，錯誤指出 register 與指令位置。
Register 次數 Repeat 內，先執行的 LoadValue 以作者 table 值域定義同圈來源，ComputedPulse／時間 consumer 可隨後使用。
未執行的圈或其他路徑不建立定義，未知、未初始化與越界來源仍拒絕。
這項同圈定義由 advisor 於 2026-11-01 依代理授權裁定，不是一般未知 register 的範圍 fallback。
非時間 raw-word consumer 保留原契約，不因時間稽核而縮窄 uint32 words。
Getter 對 scalar／仿射時間回實際量化值；runtime-only 或非仿射時間明確 raise RuntimeError，不回 ndarray 或 bounds 上界。

每個 timeline 操作把自己的結果 ceil 到整數 tProc cycle 一次，事件起點也用 ceil，不用 nearest。
小數只保留在單一操作內，例如實際起點、waveform／ADC 長度與 post 的合計。
Runtime 整數起點 R 加靜態小數 f，下一操作計算 R＋ceil(f＋t)。
Ceil 以 `math.isclose(ticks, nearest, rel_tol=1e-12, abs_tol=1e-10)` 吸收接近整數的換算雜訊，其餘小數照常向上取整。
在 2³¹ 附近，相對容差最多也會吸收約 0.002 cycle 的真實小數。
作者 sweep 沿 QICK 原生 start round 與每圈 step 往零截斷，實際值可能不同於預期值，不配置逐點時間 table。
2026-10-31，使用者要求：「你永遠都不能先加總再ceil，誤差會很大」，並補充「純delay的應該也是量化」。
2026-11-01，使用者以「B，而新設計沒有查表fallback吧？」確認相對容差方案。精確有理數時間模型另行評估。

靜態 generator fabric／ADC 長度先作有理數時鐘換算，再於該操作 aggregate 收尾 ceil。
現有 runtime 長度 sweep 與 selected waveform／length table 使用 k=8 上界定點換算，不新增一般作者 length register 或 ADC length API。
編譯器以移位加減計算乘積，選符合誤差預算且 CSD 項數最少的 W，平手取較小值，不用 ARITH／除法器。
換算範圍為 2²³ tProc cycles，超限或乘積／中間值 overflow 在 build 報錯。
500 µs 內的合法長度不早於物理結束，最多晚 1 µs；超出 500 µs 不沿用此誤差上限。
2026-10-31，使用者選定「那k=8,我知道為什麼優先用dmem了」，並要求「500us任意長誤差晚1us內，不可以早」。
換算比、W 與目前可達 consumer 見 [program README](../../lib/zcu_tools/program/v2/README.md#timeline-rounding)。

編譯期以軸 ID 對齊時間分量，已知軸保存一維值，未知軸保存範圍與係數，常數使用大小 1 的固定軸。
加減先合併同軸分量並消去零項。已知軸上逐點全等的分量移到常數軸，結尾因此為常數時不配置 end register。
編譯期已知的跨格點換算 bias 以字面值進入運算；span 為 0 的換算參數不配 sweep register。只差編譯期常數的 runtime 結尾沿用既有 register，以字面值位移表示，consumer 需要獨立 register operand 時才物化。2026-11-01 advisor 依使用者「1代理模式就都給你決定，我事後看就好」的代理授權，為無同步 chevron 與遷移後 RB／IRB 的 16 data register 容量定下這兩條。
同一已知軸的非線性運算逐點算；跨軸 ceil／定點換算保留原式，另加有界誤差軸，不建多維網格。
跨軸 max／min 以線性部分加有理係數的 Abs 項表示，允許負係數及巢狀 Abs。
取界依共用軸分連通塊，塊內分支定界。每塊超過 20 個未定號 Abs 項才退回區間界，退回不改 runtime 真值。
不可變節點快取範圍，支配判斷快取正規化相減式；RuntimeEnd 的界從式子推出，不另存 min／max。
2026-10-31，使用者說明「計算的原理是軸比對，源變數只是軸數為1的變數而已」，並確認「我也傾向Abs寫法，而且k可以開足夠大，因為一次只求一個節點的範圍，還可以考慮快取」。

所有 timeline 合流共用 N 路 max／min：展平同向候選，提出公共部分，合併相同變數餘項，再依差值界剔除被任一候選支配的項。
剩一項不作 max，剩 k 項作 k−1 次真實選值，再加回公共部分。
已物化的進入點是不透明綁定，後續讀捕獲 register，不從證明式重建 max 或重讀被覆寫來源。
差值可證明在 signed32 內時，兩路 max 使用 TEST、複製 x、`if(S)` 寫 y，共 3 條；可覆寫 x 時省去複製。
k 路選值成本為 `1+2(k−1)`，不含候選求值與公共部分。Min 用 `if(NS)`。
不能證明差值安全時，使用 sign-corrected mask，max 11／min 12 條、3 個 scratch，不依賴溢位後的 S。
UF 後由硬體 hazard stall，不加 NOP，不使用外部 F／NF。
2026-10-31，使用者要求「所以应该是要设计一个多个线性式都可以适用的 join 的算法」，並提醒「要確定一下這個flag真的會影響結果，有可能是假flag。」
Signed32 guard 與上述 mask 成本由 advisor 於 2026-11-01 依代理授權確認。
合法非負 timeline 時間的差不溢位；無上界仍可選到 mask，但溢位修正沒有合法公開 consumer。
該修正與目前無公開 codegen consumer 的 min 不宣稱已有端到端驗證；路徑與成本直接審閱，公開測試保留合法輸入的實際選值驗收。

框架不自動配置時間或 selected-end DMEM table。作者明示的資料／waveform table 與 IR dispatch 的 jump-address table 保留。
2026-10-31，使用者說明：「Dmem資源很珍貴，除了作者明確知道有用的位置，其他地方少用」。
對 IR jump-address table，使用者回答：「Ir跳躍那個可以，因為屬於log級別的使用量，我接受」。
已是 tProc cycles 的 Branch 合流以整數常數乘加及 max／選擇重建結尾，不用逐點網格或分段插值。

常數事件時間用字面值 `@t`；runtime 事件由 timeline 寫 s14，再沿 preprocess 接縫設 `t_regs["t"]="s_out_time"`。
Pulse、ConfigReadout、Trigger 的 expand 由 QICK 負責，不另配事件時間 data register。
同基準的下一事件使用 s14 差值增量。Native macro／raw asm 寫 s14 與 Branch 合流使記錄失效，下一事件重新載入基準。
Timeline 自己事件的 s14 自我複製不使記錄失效，原生 sweep 時間仍用 QICK sweep register。
不複製 QICK expand、不新增 QICK 底線私有呼叫、不 monkeypatch 或更換 QICK。
2026-10-31，使用者要求：「我想進一步討論timeline利用s14實現的通用方案，而不只是repeat」。
QICK preprocess 適配與記錄失效細節由 advisor 於 2026-11-01 依代理授權確認。

Program／timeline owner 為每個 native channel 保留編譯期結尾式子，涵蓋 scalar 與 runtime 起點。
只在真正發射時記錄，reference 推進時扣除並 floor 0，Repeat／Branch 保留路徑與迭代關係。
這是 WAIT／overlap 證明的 metadata，不是第二個 scheduler，不新增 channel-end register。
Native channel 記帳由 advisor 於 2026-10-31 依代理授權裁定。
使用者於 2026-10-29 裁定：「ABs/asr是安全可用的，可以不用測」。這不是已完成實機驗證的聲明。

### 收尾 WAIT 與 slack 護欄

Body 收尾同步全部 timeline 及 ADC 實際結尾，再記錄 body_end，然後進 shot 外殼。
`delay_by_timeline` 在 runtime 取真實 maximum；shell WAIT 仍是編譯期常數，不使用 variable WAIT，也不以上界代替 runtime 同步。
ADC 結尾沒有上界時仍參與實際同步，不因無 hint 而拒絕或假造上界。
已同步消耗的 tail 剩餘量為 0，其他已知 ADC 仍保留來源資訊與既有護欄。
2026-11-01，對未知 ADC 結尾沿作者時間規則處理的方案，使用者回答：「3A」。

收尾先配對 raw 與已消耗的結尾，再相減取界，以保留共同來源與逐點相關性，不枚舉多維笛卡兒積。
剩餘量為常數時，WAIT 使用精確值，不發 warning，即使常數大於 relax_delay 也不拒絕。
例如 ADC end＝O+190、body end＝O+200，剩餘量是 0，不因 O 的範圍增加等待。
能證明精確 min／max 時，WAIT 使用 max，`slack=max-min` 是合法點的實際差距，單位為 tProc cycles。
Slack 為 0 不警告，大於 0 時 warning 指出變數與差距。
已知 slack 超過 relax_delay 才 raise ValueError；等於時警告，不拒絕。

只有安全界而非精確差距時，warning 明說 slack 是 upper bound，不以該上界超過 relax_delay 為由拒絕。
可消去的未知來源先消去，可逐點求值的部分仍保留，不能因另一個未知值丟掉全部 provenance。
Abs 預算退回區間只使證明保守，不改同步真值，也不把保守下界冒充實際 min。
2026-10-30，使用者選擇常數 WAIT 方案「A」，並補充「WAIT可以用常數，但sync要考慮runtime」。
對未知 slack 只警告、不按上界拒絕的方案，使用者同日回答：「A」。引文 sync 的現名為 delay_by_timeline。
來源配對、相減後取界及證明模型由 advisor 於 2026-11-01 依代理授權同步，不加入一般 CFG analysis。

### Fork、join 與 Repeat 邊界

只有 default 預先存在。其他名字先在 body 用 `fork(source, into=...)` 宣告，複製 source 的結尾。
Into 接受新名字或名字 list，重名與未宣告使用報錯。
`join(*sources, into=target)` 將既有 target 與 sources 的結尾取 max 寫回 target，不消耗 sources，也不移 reference。
它不發射 child Module，不要求每條 fork 都 join 回來。
2026-10-29，使用者提議：「我覺得join的工作好像可以簡化，不是平行好幾個branch並合併，而是單純的將給定的幾個timetime結尾合併到指定目標timeline」。
使用者並要求：「那會需要一個宣告新timeline的機制，從當前指定timeline尾端分岔」。

`branch(by=..., align_timeline=..., timeline=None)` 依 sweep 點索引選 case，不支援量測結果分支。
母 timeline 省略時選目前範圍，範圍外選 default。
每個 `with cases.case(index) as name:` 從母 timeline 隱式 fork，名字為 `"<by>[<index>]"`，並屏蔽範圍預設。
Case 內只能使用 case timeline 與其局部 fork，不能使用母 timeline 或其他外部名字。
Index 是 by 迴圈的 zero-based 點索引。未宣告的點執行空 case，重複、非整數或超界 index 報錯。
`align_timeline` 是必填 keyword-only bool。True 把各 case 結尾 join 回母 timeline；False 使用實際執行 case 的結尾。
True 需要結尾能從入口重建，不執行未選 case 的動作。全靜態取靜態 max，否則取逐點 runtime max。
兩種模式都保留入口 reference，不自動 delay_by_timeline，其他外部 timeline 不變。
Case 內經 owner 推進 reference 的操作在 build 報錯，等待改用 timeline_inc。
Branch 區塊內、case 外的操作也報錯，無條件動作寫在區塊前或後。
Case 及局部 fork 的名字離開 case 即失效，未 join 的局部 end 直接丟棄，不合回祖先。
跨 Branch 使用的 timeline 先在入口前宣告；register 超限保留 QICK 原錯誤，由作者負責。

2026-10-31，使用者要求：「case()輸入應該要有對應的值，值沒有宣告對應case應該視為空branch」。
對結尾選擇說明：「我會預期branch實際長度可以選擇所以case的最大值，或是runtime時走的那個branch」；「我不想自動推進ref time」。
對局部 fork 說明：「局部fork自動並回是超出預期的行為」。
使用者提出「branch初始化帶timeline參數應該就不用再一層with timeline的寫法，你覺得case隱式建立的timeline名叫什麼，感覺with case as timeline_name是好設計，且一樣屏蔽預設timeline」，並以「45確認」確認母／case timeline 方案。

`repeat(name, n, *, timeline=None, range_hint=None)` 使用逐圈累積的 timeline。
母 timeline 省略時選目前範圍，範圍外選 default。Body 以循環 timeline 屏蔽預設，只能使用母 timeline 與其局部 fork。
入口與回邊不自動同步，n=0 保留母 timeline，不執行或宣告 body 內容。
Case 外的 body 可明寫 delay_by_timeline、原生 delay／delay_auto 或巢狀 Repeat。
Repeat 可以放進 Branch case，但 body 及巢狀結構仍不得經 owner 推進 reference。
其他外部 timeline 可橫跨 Repeat，不能在 body 內使用，連作為 fork source 也不行。
離開時局部名字失效，外部可重新 fork 同名。
2026-10-31，使用者說明：「repeat屏蔽預設timeline，尾端就是一個變數，這樣內部操作自然變成基於這個變數」。
對 reference 推進，使用者提出：「讓作者顯式在body想要的位置插入delay或sync效果是否一樣？」

只有 body 頂層最後操作為循環 timeline 的 delay_by_timeline，且入口結尾為常數 0，才採不攜帶尾端 register 的 lowering。
固定圈長編成每圈常數 inc_ref，其他情況使用變數尾端／s14 指針。
若 body 以該同步結尾但入口不為常數 0，發 UserWarning 提示循環前同步，編譯器不自行加入。
不分析定點、不重播 preprocess，不因後續事件多就自動推進 reference。
這項結構規則由使用者於 2026-10-31 以「確認」採用，具體 lowering 由 advisor 依代理授權整理。

固定 n 是非負整數，不接受 range_hint。Register n 是已準備的 register 名字，不能與 counter 的 name 相同。
Register 次數的非負閉區間 hint 可選，不從初值假造上界，也不增加硬體檢查；沒有 hint 時不證明回邊同 channel overlap。
每 shot／sweep 點重建循環狀態，counter 不加入 QICK top-level sweep metadata。
作者明寫 reference 操作時，外部 end 按每圈實際推進量 debit 並 floor 0；無 reference 操作則不變。
固定且可靜態表示的總量保持靜態，runtime 次數或路徑需要時重用逐圈 debit storage。
Branch／Repeat 之後可直接在外部 timeline 放 pulse／trigger／config／emit 或使用 timeline_inc，不必先 join、fork 或同步。
後續事件從實際 remaining end 經 s14 放置，不用範圍上界，也不為可能的事件預配 register。
Shot 收尾、join、fork、delay_by_timeline 本來需要的 selected-end／debit storage 可直接重用，不另配事件時間 register。
2026-11-01，對取消外部 timeline 的 wait-only 拒絕、從實際結尾繼續放事件的方案，使用者回答：「1B」。
Storage 重用範圍由 advisor 同日依代理授權裁定，仍保留 16 個 data register 上限與不用時間 DMEM 的限制。

### 並行序列與 reference 守衛

作者以 builder／program 為每個 sequence fork timeline，依作者順序 emit，最後 join 回母 timeline。
各 sequence 從共同 origin 開始，sequence 及內部 Module 都不重排。`join` 只合併結尾，不發射 children。
分支與迴圈直接使用上述 `branch`／`case` 與 `repeat()`，沒有 control module 包裝。
TwoPulseReset／BathReset 的 child 與 branch case 不得經 reference owner 推進 reference。
守衛設在實際推進 reference 的 owner 接縫，不按 macro 名稱或效果分類，不保留交換 admission。
Producer provenance、全流寫入稽核、physical-end 與 case／Repeat backedge guards 照常執行，不加 padding 或 reference rewind。

2026-10-31，使用者對統一並行寫法回答：「f12同意，刪掉，不要有兩種寫法」。
Fork／join lowering 與 reference owner 守衛由 advisor 同日依代理授權裁定，取代將推進型 sequence 排到最後的方案。
Raw asm_inst 或自訂 Macro.translate 推進 reference 不經 timeline 記帳，也不受這些守衛或 Repeat 契約保護，結果由作者負責。

### Compiled snapshot 與 simulation 入口

IR compiler 每次成功 compile 後在 program 發布具型別、不可變的 `compiled_stream`。
它保存優化前的 logical instructions、labels 與結構 metadata，是與 simulation 無關的 compiler artifact。
`TimelineProgramV2` 的手寫與 builder 路徑使用相同契約。
Snapshot 不借用可變的 Module／macro graph，與後續 program 變更隔離；consumer 取得 detached typed instructions。
Compiler 擁有結構與資料，不依賴 simulation，也不透過 observer callback 管理 consumer。
Simulation 不接管 compile lifecycle。
存取、copy、recompile 與位址語意見 [IR README](../../lib/zcu_tools/program/v2/ir/README.md#pre-optimization-compile-snapshot)。

Producer 以 LOOP／BRANCH／SOURCE 與 body 邊界描述結構、標記作者來源，並提供 resolved sweep 起點、步長與有限 table definitions。
Simulation frontend 核對 metadata 與實際指令，不從 jump 或仿射 shell 反推結構，也不重新解釋 Module tree。
不一致在建立 request 前拒絕。
來源名稱由 producer 擁有，分工見 [modules README](../../lib/zcu_tools/program/v2/modules/README.md#design-boundaries)。

Sweep points 是獨立向量 lanes，reps 只保留在 request metadata，不執行。
Shot 內的非 sweep loops 依標記順序迭代，以 active masks 處理零次、固定與 register 次數，不展開指令流。
跨 point／rep 的 register、memory、readout config FIFO 與 pending update 延續在建立 request 前拒絕，同值或輸出不變也不豁免。
Prologue 後不再寫入的常數與 config 可用，沒有 ordered engine 或 fallback。
支援範圍由 instruction effect 決定，不依 macro 類別或來源白名單；不支援的效果在呼叫 backend 前拒絕。
Frontend 保留 shot-local 運算、memory 與 FIFO 時序，不執行量測結果 feedback，也不模擬 tProc 執行 latency；它不是通用 tProc emulator。

`SimulationAcquireMixin` 對 `SimulationSoc` 呼叫唯一的 compiled request builder，傳入 SoC routes。
Builder 讀 snapshot，輸出 detached numeric requests；physical backend 仍由既有 numeric owner 負責。
其他 SoC 保留 QICK acquisition。
舊 Module-tree simulation frontend 已移除，不留 selector、legacy oracle 或 fallback。
呼叫與拒絕契約見 [simulation README](../../lib/zcu_tools/program/v2/simulation/README.md#compiled-request-builder)。

2026-10-29，使用者要求：「優化是一定要做的，你可以安排流程」，對 compiled frontend 選擇回答：「A, 不是瓶頸就好」。
同日，對移除舊 frontend，使用者裁定：「舊前端不留，第二題你決定」。
對運算支援，使用者回答：「運算的採用建議」，因此採 instruction effects 判定，取代早期 runtime 運算禁令。
2026-10-30，使用者描述執行模型：「每個內部迴圈都當作獨立點向量化模擬，但內部非sweep迴圈會視為一個shot內部的order，但不會單純的展開」。
對獨立 shot 與拒絕跨 shot carry 的提案，使用者回答：「採用建議」。
同日，通用標記暫緩後，使用者回答：「那採用建議做法」。
現行方案只擴充有限 LOOP metadata，不採 generic payload。

### IR analysis 與驗收責任

`program/v2/ir` 擁有唯讀時序 analysis，不 import frontend 私有 macro，也不從 module total_length 回推編譯行為。
Program owner 產生 boundary metadata，經既有 IR meta 通道傳到最終 linked addresses，公開為 `timing_layout`。
Layout 提供 body_start、body_end 與由外到內的 loop 名稱／次數。Loop 資訊只用於掃描座標，不取代編譯指令的初值、更新與跳轉。
`CompiledTimingInput` 接收公開 prog_list、labels、layout，以及 `compile_datamem()` 的 dmem 結果。
Dmem 是 signed int32 words，位址是從零起算的 word offset；None 或空資料不是零填記憶體。
Analysis 只讀已知位址及資料並處理可決定的控制流，缺資料、未知或越界位址、寫入等未支援效果均報錯。
Caller 不重建 dispatch table，不用固定指令數、手填位址或掃 opcode 猜 shot 外殼。

每次執行 body_start 開始一個 shot。
Shot_end 取下次 body_start 前的 reference，最後一個取程式結束時的 reference，包含 final_wait／final_delay 外殼。
結果另提供 body_end reference，供觀察 body 自身推進量。
事件 cycle 的原點是初始 reference 0，analysis 不模擬 CPU latency。
Resync 讀硬體 s_usr_time／s11，analysis 明確拒絕，不預設為 0，也不宣稱其後精確時序。
編譯期可知的 register 時間與唯讀 dmem 照常分析，不套用硬體時間例外。
IR 窗口同時計入字面值 s14 增量與 TIME inc_ref，register 增量視為未知，不展開該窗口。
TimeInst 不宣告讀寫 s14，TIME 不碰 s14；純 register 自我複製的刪除由通用 peephole 處理，不把 timeline 規則放進 IR。
這項 IR／program 分工由 advisor 於 2026-11-01 依代理授權確認。

Analysis 沒有 waveform／ADC 積分長度，不能單獨算出 event end，也不作 production WAIT guard 的 oracle。
WAIT 驗收從公開逐點事件與 body_end 出發，另以已知硬體時鐘和播放／積分長度獨立計算剩餘時間。
隨機 program 與固定回歸案例檢查 WAIT 不小於任何合法點的剩餘時間，已證明精確的來源使用精確 max。
能證明合法點剩餘相同時須精確且無 warning；變動時區分實際 slack 與保守上界，再檢查 warning 及 relax 的拒絕邊界。

使用者於 2026-10-29 裁定：「第一件選A，語意與行為一致就可以了，舊設計的怪癖不一定要對齊」，並強調：「我沒有要求必須與舊的一致，我要求的是與預期行為一致」。
因此驗收從具體輸入與現行 B1／規則 b 獨立手算逐掃描點的 pulse／trigger 絕對 cycle、body_end 與 shot_end，不由受測 total_length 或新舊輸出生成 expected。
指令數、register 排法與逐位相同都不是驗收條件，不為還原它們新增 scalar＋sweep 合併。
新舊六項輸出 prog_list／labels／pmem／wmem／dmem／asm 連同輸入、版本、重現命令只作一次性遷移差異報告。
規則 a→b 報告使用真實 A baseline 與 B 編譯結果。兩側支援且輸入合法時，逐點列 event、body／shot end 與 b−a delta。
A 不支援的 owner／runtime 組合列具體原因，A timing 與 delta 記 N/A；B 仍以真 producer 完成獨立手算驗收。
不為比較另寫 rule-a adapter。Legacy 原始輸出只能標示「沒有 boundary metadata」，不冒充 rule a、不猜時序，也不為它新增 producer。
報告區分預期改變、舊怪癖與新缺陷，說明正負 delta，不設只許提前或統一 N cycle 門檻。
這項報告政策由 advisor 依使用者 2026-10-30 代理授權裁定，沿用使用者要求的預期行為驗收。
例如修正漏算的較長 qubit tail 可以讓 B 的 body 結尾晚於 A，不能以 delta 的正負取代正確性判斷。
新缺陷須修正，未知差異須回報；報告不成為常駐的新舊 equality test。

### 現行交付範圍與裁定沿革

Timeline module lowering、規則 b、runtime timeline、builder／Schedule 分工與 compiled simulation frontend 均已交付。
全部實驗已改用 Timeline；舊 program、builder、module lifecycle 與 control module 不再保留為平行路徑。
MuxReadout 的 `collect_iq`／`collect_raw_iq` 依 Timeline program 已編譯的 readout emission 取數，保留平均、ADC 長度正規化、offset、tone／ADC 順序與輸出 shape。

早期 A／B／C／D 是交付順序，不是現行功能開關。A 的規則 a、靜態證明限制與 register Repeat 等待上界已由本篇的規則 b 與 runtime 契約取代。
2026-10-29，對規則 b 與差異報告提前到 B、D 只刪舊路徑的安排，使用者回答：「採用建議」。
同日，對 B 的 register delay、Module Join lowering 與 Mux 取數範圍，使用者回答：「1A  2B 3A」。
當時三項分別採用 register 名稱表示 timeline cycles、保留 Module Join 的原生 lowering，以及只轉 Mux 宣告／播放、不擴 collect_raw_iq。後兩項已由 2026-11-01 的全面遷移裁定取代。
2026-10-31，對 D 移回大量實驗遷移結尾的安排，使用者回答：「採用建議」。遷移與刪除現已完成，不再留待 D 交付。

2026-10-30，對先交付 builder／Schedule 取數、實驗遷移另開工作且等待模擬 frontend 的安排，使用者回答：「沒問題，採用建議，我思考soc/prog的順序，感覺prog應該要是第一個才對」。
因此現行 run 入口以 program 在前、soc 在後。Builder 不管理取數，Schedule 不組裝 program。
Compiled frontend 已取代舊 Module-tree simulation frontend，沒有 legacy selector 或 fallback。

## 依賴與後果

序列組裝留在 program，取數執行政策由 experiment runtime 擁有。
Experiment 可依賴 program；program 不反向依賴 experiment，此方向由 `.importlinter` C8 檢查。
沿用 [[0072]] 的分層與 `.importlinter` 的 C8、C9、C12、C19，以及 `program-simulation-entry` contract。
IR analysis 是編譯結果分析，不是物理模擬 frontend，不產生模擬量測資料。
模擬遷移的狀態另見 [simulation-environments](draft/simulation-environments.md#compiled-frontend-與小介面依賴)。

作者必須明確同步 timeline 與原生 reference，不能以未同步的原生 pulse 代替序列發射。
Runtime timeline 承接無法靜態精確表示的時間。已知來源保留 bounds、provenance 與可證明重疊的拒絕，無 hint 作者時間的未知段由作者負責合法性。
Data register 上限仍為 16，不保證任意序列都能編譯。實驗仍保留 drive Join 後的明確同步。
原生寫法與公開 API 範例由 [program README](../../lib/zcu_tools/program/v2/README.md#原生寫法與具名-timeline) 維護，module 宣告與 owner 邊界見 [modules README](../../lib/zcu_tools/program/v2/modules/README.md#原生寫法)。
