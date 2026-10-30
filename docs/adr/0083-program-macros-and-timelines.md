# 0083 — Program macro 與具名 timeline

日期：2026-10-29。B 與 builder 契約更新：2026-10-30。
狀態：Accepted。A、B 與 builder 已交付；C frontend 遷移、實驗遷移與 D 舊路徑移除另行交付。
相關：[[0072]]、[[0074]]、[[0078]]

## 脈絡

Program 作者需要在同一個 body 混用原生 QICK 操作與可重用的 Pulse、Readout、Reset。
跨 channel 的排程也需要獨立於 QICK channel timestamp 的具名時間線。
若以 Module 繼承 Macro 表達兩者，同一物件的資源宣告與每次發射狀態就會混在一起。
若把 default timeline 綁到 reference，省略 timeline 的 module 與具名 module 又會遵循不同的推進規則。

## 決策

B 與 builder 的 owner 細節由 advisor 依使用者 2026-10-30 代理授權裁定。
使用者的設計選擇另附日期與逐字原話，與 advisor 的細節裁定分開記錄。

### Program 基底與 module owner

`TimelineProgramV2` 繼承 repo 的 `MyProgramV2` 加 `ProgramResourcesMixin`，不經 `ModularProgramV2`。
它承接既有 `SimulationAcquireMixin`、`ImproveAsmV2`、`IRCompileMixin` 與 classify／herald acquire 擴充。
承接 acquire 介面不代表模擬 frontend 已支援新寫法。
原生操作與 module lowering 共用既有 IR 管線及 `TimedMergePass` 的安全條件，不按來源分流或另建 canonicalization。
共用 `ProgramResourcesMixin` 只集中 dmem 配置與 readout route 保護。
`TimelineProgramV2` 在 ADC／register 宣告邊界記錄 module resource owner。
不同物件要求不可共用的資源時，宣告失敗並指出兩個 module 的名稱與物件。
明示的 register 借用與 module 家族共享契約保留，例如 Pulse registry 的 generator／template 共用。
這些檢查由 program 擁有，builder 不解析 module 的內部資源；`ModularProgramV2` 的規則不變。

作者在 `_initialize` 建立並 `declare(module)`，在 `_body` 呼叫 `emit(module, timeline=..., tag=...)`。
Module 保存資源宣告結果，不繼承 QICK Macro，也不能直接傳給 `append_macro`。
每次 emit 建立新的通用 `ModuleEmission`，時間與 tag 屬於那次發射。
各 module owner 提供宣告與 lowering，不經舊 `run`。
只有確實需要不同 preprocess／expand 行為時，才另建 macro 類別。

使用者於 2026-10-29 對原生作者入口指出：「我覺得不太好，用program寫法就該接近原生的寫法，而builder才該設計成一個with scope，內部是像body的寫法」。
對宣告與發射分開的選項，使用者選擇：「那用A好了，但代價應該就是會出現很多新macro與module」。
這裡的 A 指顯式 module 宣告、每次 emit 產生 macro，不是交付階段名稱。
以通用 emission 加 owner lowering 避免每種 module 都需要一個對應 macro 類別。
對 cfg 直接產生 macro 的替代方案，使用者裁定：「A'的機制可能比較複雜，收益相比A比較小，除非有很一致且簡單的declare機制，否則還是顯式module好了」。
Module 家族沒有一致的 cfg-only 宣告入口，部分 owner 還需要 cfg 以外的參數，因此不從 cfg 自動建 module 或加入 handle 層。
原生 program 仍顯式宣告 module；下節的 builder 自動宣告作者交來的 module 物件。

Program／emission owner 配置每次 lower 的內部名稱，scope 隨 compile 重建，nested emission 與失敗返回不污染呼叫端。
Program body 可直接呼叫 `emit_parallel`，由 program 建立這次發射的命名 scope。
Module lowering 內呼叫 `emit_parallel` 則沿用當下 scope；builder 只重播公開方法，不自行組名稱。
LoadValue owner 提供本次 index 的 lookup，重用已宣告的 table 與 destination。
ScanWith 使用固定次數 Repeat 與該 lookup，不複製或改寫 loader 的宣告狀態。
Repeat owner 提供固定 counter 的名稱、次數與 writer metadata；nested loop 不加入 QICK 的 top-level sweep metadata。
這些分工讓 emission 狀態與宣告資源分開，也避免呼叫端鏡射 QICK 私有 discovery 流程。

### Builder 與 Schedule 的職責

`TimelineBuilder` 位於 program 層，只記錄序列並以顯式 `build()` 產生 `TimelineProgramV2`。
Builder 不管理取數生命週期，不是 context manager，也不接收 SoC 或依賴 Schedule。
它在 module 物件第一次 emit 時記錄宣告，同一物件再次 emit 沿用宣告。
不同物件不按內容合併；硬體資源的共享與衝突交給各 module 家族和 program owner。

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
`"default"` 是普通 timeline。未進入 timeline 範圍時，一般 emit 省略 timeline 便選 default，不會移動 reference。
未進入範圍且不帶 timeline 的原生 pulse／trigger 仍相對 QICK reference。
不建立每 channel 一條預設 timeline，原生 channel timestamp 保持原用途。

使用者於 2026-10-29 說明定位：「原生qick其實有timeline機制，只是其timeline固定綁定到channel, 且與ref_time是分開的…我們新設計的timeline，其實就是不綁定channel的原生timeline」。
對儲存狀態裁定：「我倒是認為只要存當前結尾就好，不用存開始，我傾向B1」。

`with builder.timeline(name)` 與 `with program.timeline(name)` 只切換範圍內省略 `timeline=` 或傳入 `None` 時的預設。
它涵蓋 `emit`、`emit_parallel`、`pulse`、`trigger`、`delay_timeline`、`sync` 與 `repeat` 的母 timeline。
顯式名字優先，包含 `"default"`；巢狀範圍覆蓋外層，離開時還原。
名字須是 default 或已 fork 的 timeline，未知名字報錯；範圍本身不 fork、不 join，也不推進 reference。
`branch` 仍只有 `by`，沒有母 timeline 參數；case 內的發射沿用當下範圍預設。
範圍不增加 branch 的同步或 join，case used/local 與不等長規則維持不變。
原生 `delay`／`delay_auto` 只推進 reference，不受此預設影響。
使用者於 2026-10-30 對只切換既有名字的預設、顯式值優先、巢狀還原與 program 同名入口，回答：「1a 採用建議」。

一般 emission 先取所選 timeline 的結尾作為內部事件的共同起點。
完成後採規則 b，不從 `total_length` 重建 duration，也不把 composite 內原本重疊的事件串接。
單一 post-delay owner 取實際 waveform／ADC event end 的最大值，加一次 post-delay，再對完整結尾向上取整到 tProc cycle。
實際 event end 使用編譯後的發射時刻與播放／積分長度；分數尾段保留到 aggregate 收尾，不先各自取整。
Reset 保留各 child 的 post-delay，不新增 composite post。
BathReset 的 pi/2 在 cavity end 加 post-delay 向上取整後開始，最後合併 qubit 與 pi/2 的實際結尾。
相依 placement 的 ceil 防止事件提早，不能用各 child duration 先取整再相加代替。
Delay 與 SoftDelay 都只加結尾，不產生 reference advance。
帶 timeline 的 pulse／trigger 在 E+t 發射，結尾更新到事件結束點。
`send_readoutconfig` 與 `send_readoutconfig_from_regs` 是瞬時事件，結尾更新到 E+t，不加 config length 或其他 ADC window。
經 timeline 或 emit 放置的 pulse 若早於同 channel 前一個 pulse 結束，編譯報錯；另一條 timeline 不能豁免此檢查。
未進入範圍且未帶 timeline 的原生 pulse 保留 QICK 行為。

`delay_timeline(t, timeline=X)` 只加 X 的結尾，`sync(timeline=X)` 才將 reference 推到 X 的結尾。
Reference 推進 d 後，全部 timeline 包含 default 一起變成 `max(0, E-d)`。
原生 delay／delay_auto／wait_auto／resync 沿用 QICK，不增加 timeline keyword。
不提供 `delay_auto_timeline`。
掃描量跨零或結尾 max 交叉時，使用逐點的 floor 0／maximum，不整條歸零、硬選一側或補成統一長度。

DelayAuto 是一般 emission 規則的唯一例外。
它只在 timeline 範圍外且未指定 timeline 時包裝原生 delay_auto／delay_reg_auto，保留 gens／ros，移 reference 並扣減全部結尾。
它不假造 total_length，也不額外推進 default。
顯式指定任何 timeline，包含 default，或由範圍選出 timeline，均在 emit 入口報錯。
delay_reg_auto 先消耗選定的 native channels，再推進非負 register cycles。
尚未被 native 部分完整消耗的 timeline，依已知 producer bounds 扣除實際推進量。
若 native 部分已完整消耗全部 timeline，register 部分沿用原生的非負時間契約。
PulseReadout 的 gain／frequency runtime words 是非時間 consumer，不受這項時間限制排除。

### Runtime end 與範圍證明

Scalar 與可精確表示的整數仿射 QickParam end 保持靜態。
跨零 floor、交叉 max、path-selected end 與 runtime length 需要時才物化為 cycle register。
`delay_timeline` 的 scalar／QickParam 單位為 us；register 名稱代表非負 tProc cycles，只更新目標結尾，不移 reference。
Runtime 算術只用不更新旗標的 ABS、ASR、加與減。
End 的非負 cycle 範圍為 `0..2**31-1`，中間運算也須符合 signed 32-bit；可能溢位就在編譯期報錯，不依賴硬體 wrap。

Program／timeline owner 擁有 typed producer bounds 與翻譯後的全流寫入稽核。
原生常數、sweep、loader table 與 runtime playback owner 提供各自的來源關係；Repeat count 使用作者保證的 range_hint。
時間來源或 bounds 依賴有未登記 producer 的寫入，編譯器拒絕並指出 register 與指令位置。
少數掃描點的 timing 結果不能代替範圍證明。
自訂 macro 的 register 初值不構成時間證明，也不新增作者替任意時間 producer 宣告 range 的介面。
非時間 raw-word consumer 保留原契約，不因這項時間稽核而縮窄 uint32 words。

編譯期已知的 sweep 若在 aggregate ceil 後不再是整數仿射序列，由 program／timeline owner 產生逐點精確 dmem table。
Table index、bounds 與 writes 沿用所屬 producer 的介面及稽核，不複製 loader packing 或反向依賴 semantic Module。
表放不下 dmem 時編譯報錯，指出 emission 與所需大小，不截斷或用 padding 代替。
因此靜態值的選擇取決於能否精確表示，不只是能否在編譯期知道各點時間。
不加入一般 interval／CFG analysis。

使用者於 2026-10-29 指出：「可以先簡單做法，但我確實想探索runtime timeline機制，使用abs實作max語法，用於無法編譯器決定的時間」。
對將 runtime timeline 納入 B 的範圍，使用者回答：「1可以」。
對硬體指令假設，使用者裁定：「ABs/asr是安全可用的，可以不用測」。
因此 ABS／ASR 視為可用，不要求額外硬體 smoke test；這不是已完成實機驗證的聲明。

### Fork、join 與 Repeat 邊界

只有 default 預先存在。其他名字先在 `_body` 用 `fork(source, into=...)` 宣告，複製既有 source 的結尾。
Into 接受一個新名字或名字 list；重名與未宣告使用報錯。
`join(*sources, into=target)` 將既有 target 與 sources 的結尾取 max 寫回 target，不消耗 sources，也不移 reference。
它只合併時間記帳，不發射 child Module；靜態 max 在編譯期計算，runtime max 由上述 cycle 運算產生。
不強制 fork 後 join 回來。Body 收尾仍同步全部 timeline，再記錄 body_end，然後進入 shot 外殼。

使用者於 2026-10-29 提議：「我覺得join的工作好像可以簡化，不是平行好幾個branch並合併，而是單純的將給定的幾個timetime結尾合併到指定目標timeline」。
並要求：「那會需要一個宣告新timeline的機制，從當前指定timeline尾端分岔」。
因此採記帳式 fork／join，不提供 `with join`／`j.branch`。

`with self.branch(by=...)` 依 sweep selector 選 case，不支援量測結果分支。
每個 case 只同步自己使用的 timeline 與局部 fork，可不等長，不補到共同最長 reference。
Case 內 fork 的名字在 case 結束後失效；跨 Branch 要使用的 timeline 必須在入口前宣告。
入口前既有 timeline 的 path ends 不同時，保留實際選中 end 並標為只能等待。

`with self.repeat(name, n, timeline=P, range_hint=...)` 的母 timeline P 預設為 default。
入口先 `sync(timeline=P)`，避免第一圈與迴圈前 P 的內容重疊。固定 n=0 仍同步入口，但不執行 body。
內部只能用 P，或從 P／局部 timeline fork 的局部名字。
其他外部 timeline 可以橫跨迴圈，但不能在內部使用，連作為 fork source 也不行。
每圈同步 P 與局部結尾。離開後 P 為 0，局部名字失效，可在外部重新 fork。
巢狀 Repeat 遞迴套用此規則。

使用者於 2026-10-29 要求考慮「一開始一個超長flux pulse在獨立timeline，中間有branch或repeat的情況」，並對允許外部 timeline 橫跨結構的修正規則回答：「我同意」。
這取代進入 Repeat 時拒絕尚未結束的外部 timeline 的做法。

固定 n 為非負整數，不接受 range_hint。外部結尾先扣入口同步量。
每圈推進相同且可靜態表示時，退出後精確扣 n 倍推進量；否則按實際執行路徑逐圈 debit 並做 floor 0。
Register n 是既有 register 名字，不得與 name 命名的 counter 同名。
它一律要求作者保證的非負整數閉區間 `range_hint=(minimum, maximum)`，minimum 不得大於 maximum。
編譯器不從 register 初值推定 range，也不新增硬體檢查。
Register 次數或不同 case 的推進需要 runtime storage 時，外部結尾按實際路徑扣除，不採 minimum count 的等待上界。
路徑不同或 register 次數後尚未消耗的原 timeline 保持只能等待。

只能等待的來源可供 join／fork、sync 或 body 對齊使用。
在它自己上面放置 pulse／trigger／emit／兩種 readout config 或使用 delay_timeline 都報錯，config 的 t=0 也不例外。
Join 目標與 fork 新 timeline 是普通狀態，從實際 remaining end 建立可繼續發射的排程點，來源本身仍只能等待。
Sync 或原生 reference 推進完整消耗該 end 後，來源恢復普通狀態；部分消耗仍只能等待。

Repeat body 支援具名 timeline 的 register 時間與各 case 不同的 Branch advance。
各 case 分別累積 pulse guard，回邊使用全路徑保守 envelope 證明不重疊。
Envelope 只作重疊與範圍證明，不改各 case 的 reference advance、實際 end、body_end 或 shot_end。
Repeat body 內的 DelayAuto(register) 與原生 delay_reg_auto 仍拒絕；它們是未知的 native reference advance，不是具名 timeline duration。
迴圈外 consumer 不受這項 Repeat 限制。

### Module Join 與 program join 的分工

Module Join 發射平行 branch sequences，保留每個 sequence 內的 Module 次序及共同 origin。
它與只合併既有結尾的 program join 是兩個契約；Module Branch 則 lower 成上述 branch/case 結構。
Program emission owner 依 lowered operations 彙總 reference effects，不從 Module 名稱猜排程。
零個推進型 sequence 時保留次序；唯一推進型 sequence 最後發射，其他 sequences 保持次序；兩個以上或未知 reference effect 都拒絕。

交換完整 sequences 前，owner 須以實際 register、wmem 與 alias identity 證明無 RW、WR、WW 衝突。
未知 resource effects 只在需要交換時拒絕，compiler 配置的 register 不自動視為可忽略的 scratch register。
這項 admission 不取代 producer provenance、翻譯後寫入稽核、每個 generator 的 physical-end 檢查或全部 case／Repeat backedge guards。
不重排 sequence 內部，不加入一般 scheduler，也不用負相對時間、reference rewind 或 padding 維持共同 origin。

這項 admission 由 advisor 依使用者 2026-10-30 代理授權裁定。
逐個 child 直接 lower 會讓推進 reference 的 child 改變後續 sibling 起點；上述限制保留可證明安全的共同 origin，而不引入多序列排程器。

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

### 已採用的階段邊界

| 階段 | 範圍與狀態 |
|---|---|
| A，已交付底層 | B1、顯式 module／emission、fork／join、Repeat 與 IR 時序 analysis。A 交付時採規則 a，無法靜態證明時拒絕，register Repeat 使用等待上界；這些限制由 B 的現行契約取代。Builder、實驗與模擬 frontend 不切換。 |
| B，已交付 | 其餘 Module lowering、規則 b、runtime timeline 與 a→b 時序差異報告。新路徑不讀 block_mode，legacy 欄位與 init／run 保留到 D。B 不遷移實驗、不改模擬 frontend，也不刪舊路徑。MuxReadout 只轉換 declaration／playback，collect_raw_iq 仍只接受 legacy program。 |
| Builder，已交付 | 在 B 的 TimelineProgramV2 上提供序列錄製與 build，Schedule 提供共用 run_acquire*。與舊 builder 並存，不遷移實驗，不等待模擬 frontend 切換。 |
| C，另行交付 | 模擬 frontend 遷移，現有每種 Module form 與可模擬實驗須維持支援。其編譯結果介面、優化與切換驗收由 C 維護。Builder 已交付不代表 frontend 已切換；大量實驗遷移仍待 C 完成後另行交付。 |
| D，另行交付 | 刪除舊 init／run 路徑與退役的 legacy 入口，不再於此切換規則 b。 |

2026-10-29，對規則 b 與差異報告提前到 B、D 只刪舊路徑的安排，使用者回答：「採用建議」。
同日，對 B 的 register delay、Module Join lowering 與 Mux 取數範圍，使用者回答：「1A  2B 3A」。
三項分別採用 register 名稱表示 timeline cycles、保留 Module Join 的原生 lowering，以及只轉 Mux 宣告／播放、不擴 collect_raw_iq。
新路徑不讀 block_mode、legacy 保留到 D 的細節由 advisor 依既有階段裁定具體化，不要求 B 改寫實驗。

2026-10-30，對先交付 builder／Schedule 取數、實驗遷移另開工作且等待模擬 frontend 的安排，使用者回答：「沒問題，採用建議，我思考soc/prog的順序，感覺prog應該要是第一個才對」。
因此 run 入口以 program 在前、soc 在後；這次交付不代表 lab 實驗已遷移或 emit 寫法已可模擬。

## 依賴與後果

序列組裝留在 program，取數執行政策由 experiment runtime 擁有。
Experiment 可依賴 program；program 不反向依賴 experiment，此方向由 `.importlinter` C8 檢查。
沿用 [[0072]] 的分層與 `.importlinter` 的 C8、C9、C12、C19，以及 `program-simulation-entry` contract。
IR analysis 是編譯結果分析，不是物理模擬 frontend，不產生模擬量測資料。
模擬遷移的狀態另見 [simulation-environments](draft/simulation-environments.md#module-frontend-與小介面依賴)。

作者必須明確同步 timeline 與原生 reference，不能以未同步的原生 pulse 代替序列發射。
Runtime timeline 承接無法靜態精確表示的時間；無法證明 producer bounds、資源交換或不重疊的排程仍在編譯期拒絕。
原生寫法與公開 API 範例由 [program README](../../lib/zcu_tools/program/v2/README.md#原生寫法與具名-timeline) 維護，module 宣告與 owner 邊界見 [modules README](../../lib/zcu_tools/program/v2/modules/README.md#原生寫法)。
