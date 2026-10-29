# 0083 — Program macro 與具名 timeline

日期：2026-10-29
狀態：Accepted。A 階段底層已交付；B、C、D 的已採用邊界見下文，不代表已實作。
相關：[[0072]]、[[0074]]、[[0078]]

## 脈絡

Program 作者需要在同一個 body 混用原生 QICK 操作與可重用的 Pulse、Readout、Reset。
跨 channel 的排程也需要獨立於 QICK channel timestamp 的具名時間線。
若以 Module 繼承 Macro 表達兩者，同一物件的資源宣告與每次發射狀態就會混在一起。
若把 default timeline 綁到 reference，省略 timeline 的 module 與具名 module 又會遵循不同的推進規則。

## 決策

### Program 基底與 module owner

`TimelineProgramV2` 繼承 repo 的 `MyProgramV2` 加 `ProgramResourcesMixin`，不經 `ModularProgramV2`。
它承接既有 `SimulationAcquireMixin`、`ImproveAsmV2`、`IRCompileMixin` 與 classify／herald acquire 擴充。
承接 acquire 介面不代表模擬 frontend 已支援新寫法。
原生操作與 module lowering 共用既有 IR 管線及 `TimedMergePass` 的安全條件，不按來源分流或另建 canonicalization。
共用 `ProgramResourcesMixin` 只集中 dmem 配置與 readout route 保護。

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
Module 家族沒有一致的 cfg-only 宣告入口，部分 owner 還需要 cfg 以外的參數，因此不加入自動宣告或 handle 層。

### Timeline 只存結尾，與 reference 分開

採 B1 模型。每條 timeline 只存相對 `ref_time` 的目前結尾 E，下限為 0，不另存開始或 channel end 集合。
`"default"` 是普通 timeline。一般 emit 省略 timeline 時選 default，不會移動 reference。
不帶 timeline 的原生 pulse／trigger 仍相對 QICK reference。
不建立每 channel 一條預設 timeline，原生 channel timestamp 保持原用途。

使用者於 2026-10-29 說明定位：「原生qick其實有timeline機制，只是其timeline固定綁定到channel, 且與ref_time是分開的…我們新設計的timeline，其實就是不綁定channel的原生timeline」。
對儲存狀態裁定：「我倒是認為只要存當前結尾就好，不用存開始，我傾向B1」。

一般 emission 先取所選 timeline 的結尾作為內部事件的共同起點。
完成後結尾為起點加 A 階段規則 a 的 `total_length`，不把 composite 內原本重疊的事件串接。
已對齊 tProc cycle 的 owner 長度不再向上取整第二次。
Delay 與 SoftDelay 都只加結尾，不產生 reference advance。
帶 timeline 的 pulse／trigger 在 E+t 發射，結尾更新到事件結束點。
`send_readoutconfig` 與 `send_readoutconfig_from_regs` 是瞬時事件，結尾更新到 E+t，不加 config length 或其他 ADC window。
經 timeline 或 emit 放置的 pulse 若早於同 channel 前一個 pulse 結束，編譯報錯；另一條 timeline 不能豁免此檢查。
未帶 timeline 的原生 pulse 保留 QICK 行為。

`delay_timeline(t, timeline=X)` 只加 X 的結尾，`sync(timeline=X)` 才將 reference 推到 X 的結尾。
Reference 推進 d 後，全部 timeline 包含 default 一起變成 `max(0, E-d)`。
原生 delay／delay_auto／wait_auto／resync 沿用 QICK，不增加 timeline keyword。
不提供 `delay_auto_timeline`。
掃描量的下限與結尾 max 必須能逐點證明符號或大小；跨零、交叉或無法證明時編譯報錯，不整條歸零或硬選一側。

DelayAuto 是一般 emission 規則的唯一例外。
它在未指定 timeline 時包裝原生 delay_auto／delay_reg_auto，保留 gens／ros，移 reference 並扣減全部結尾。
它不假造 total_length，也不額外推進 default。
顯式指定任何 timeline，包含 default，均在 emit 入口報錯。
Runtime register 時間只經 delay_reg_auto。所有 timeline 結尾都須證明不大於靜態推進部分，才能在加上未知非負 register cycles 後歸零。
PulseReadout 的 gain／frequency runtime words 是非時間 consumer，不受這項時間限制排除。

### Fork、join 與 Repeat 邊界

只有 default 預先存在。其他名字先在 `_body` 用 `fork(source, into=...)` 宣告，複製既有 source 的結尾。
Into 接受一個新名字或名字 list；重名與未宣告使用報錯。
`join(*sources, into=target)` 將既有 target 與 sources 的結尾取 max 寫回 target，不消耗 sources，也不移 reference 或產生指令。
不強制 fork 後 join 回來。Body 收尾仍同步全部 timeline，再記錄 body_end，然後進入 shot 外殼。

使用者於 2026-10-29 提議：「我覺得join的工作好像可以簡化，不是平行好幾個branch並合併，而是單純的將給定的幾個timetime結尾合併到指定目標timeline」。
並要求：「那會需要一個宣告新timeline的機制，從當前指定timeline尾端分岔」。
因此採記帳式 fork／join，不提供 `with join`／`j.branch`。

`with self.repeat(name, n, timeline=P, range_hint=...)` 的母 timeline P 預設為 default。
入口先 `sync(timeline=P)`，避免第一圈與迴圈前 P 的內容重疊。固定 n=0 仍同步入口，但不執行 body。
內部只能用 P，或從 P／局部 timeline fork 的局部名字。
其他外部 timeline 可以橫跨迴圈，但不能在內部使用，連作為 fork source 也不行。
每圈同步 P 與局部結尾。離開後 P 為 0，局部名字失效，可在外部重新 fork。
巢狀 Repeat 遞迴套用此規則。

使用者於 2026-10-29 要求考慮「一開始一個超長flux pulse在獨立timeline，中間有branch或repeat的情況」，並對允許外部 timeline 橫跨結構的修正規則回答：「我同意」。
這取代進入 Repeat 時拒絕尚未結束的外部 timeline 的做法。

固定 n 為非負整數，不接受 range_hint。外部結尾先扣入口同步量，再精確扣 n 倍每圈 reference 推進量。
Register n 是既有 register 名字，不得與 name 命名的 counter 同名。
它一律要求作者保證的非負整數閉區間 `range_hint=(minimum, maximum)`，minimum 不得大於 maximum。
編譯器不從 register 初值推定 range，也不新增硬體檢查。
外部非零結尾改為只能等待，以 minimum 計算最晚剩餘結尾作等待上界。

只能等待的來源可供 join／fork、sync 或 body 對齊使用。
在它自己上面放置 pulse／trigger／emit／兩種 readout config 或使用 delay_timeline 都報錯，config 的 t=0 也不例外。
Join 目標與 fork 新 timeline 是普通狀態，可在該上界定義的保守排程點繼續發射，來源本身仍只能等待。
Sync 或原生 reference 推進逐點完整消耗上界後，來源恢復普通狀態；部分消耗仍只能等待。
A 階段不在 Repeat body 接受未知 runtime reference advance，包含 DelayAuto(register) 與原生 delay_reg_auto，因每圈推進量無法靜態表達。
迴圈外的既有 consumer 與每圈長度可證明的 register count Repeat 保持可用。

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
因此驗收從具體輸入與 B1 規則 a 獨立手算逐掃描點的 pulse／trigger 絕對 cycle、body_end 與 shot_end，不由受測 total_length 或新舊輸出生成 expected。
指令數、register 排法與逐位相同都不是驗收條件，不為還原它們新增 scalar＋sweep 合併。
新舊六項輸出 prog_list／labels／pmem／wmem／dmem／asm 連同輸入、版本、重現命令只作一次性遷移差異報告。
新側用真 producer 做完整時序驗收。Legacy 標示「沒有 boundary metadata」，不猜時序、不宣稱時序等價，也不為它新增 producer。
報告區分 B1 預期改變、舊怪癖與新缺陷；新缺陷須修正，未知差異須回報。

### 已採用的階段邊界

| 階段 | 範圍與狀態 |
|---|---|
| A，已交付底層 | B1、規則 a、顯式 module／emission、fork／join、Repeat 與 IR 時序 analysis。原生 lowering 涵蓋 Pulse、Delay、SoftDelay、DelayAuto、PulseReadout 與 Reset 系列。無法靜態證明時報錯，register Repeat 只提供上述等待上界，不引入 runtime timeline。Builder、實驗與模擬 frontend 不切換。 |
| B，已採用待實作 | 其餘 module macro 化、規則 b、移除 block 並改寫使用它的三個實驗。規則 b 取事件最大結束點，再由 lowering 用 delay_timeline 加 post_delay；沿用 IR analysis 產出 a→b 時序差異報告。加入 runtime timeline，見下文。 |
| C，frontend 遷移待實作 | 模擬 frontend 遷移到新 program 寫法，現有每種 module 的 macro 形式須可模擬，現有可模擬實驗不能失去支援。改讀編譯結果與向量直譯方案仍在討論，不把原型列為已採用架構。新 builder 與實驗遷移排在 C 之後，不屬於 A。 |
| D，待實作 | 只刪舊 init／run 路徑與舊 Join 等退役入口，不再於此切換規則 b。 |

2026-10-29，對規則 b 提前到 B、同時移除 block、D 只刪舊路徑的建議，使用者回答：「採用建議」。
這取代先前將規則 b 與差異報告放到 D 的安排，A 維持規則 a。
同日，對 runtime timeline 寫進 B 的問題，使用者回答：「1可以」。
B 的範圍含掃描量碰下限 0、依掃描點選 case 的 Branch 與跨 Repeat 的 timeline。
能編譯決定的時間仍用靜態值，只有無法決定者才放進 register，依量測結果分支仍不支援。
Branch 各 case 自行推 reference，可不等長，不補成共同最長 reference；這是待實作的 B 規則，不是 A 的 API。
這項安排不放寬 A 的靜態證明與只能等待限制，也不代表 C 已改用編譯結果。

## 依賴與後果

本決策在既有 program package 內分配 owner，不新增跨 package 依賴。
沿用 [[0072]] 的分層與 `.importlinter` 的 C8、C9、C12、C19，以及 `program-simulation-entry` contract。
IR analysis 是編譯結果分析，不是物理模擬 frontend，不產生模擬量測資料。
模擬遷移的狀態另見 [simulation-environments](draft/simulation-environments.md#module-frontend-與小介面依賴)。

作者必須明確同步 timeline 與原生 reference，不能以未同步的原生 pulse 代替序列發射。
靜態證明會拒絕部分硬體能執行的排程；B 的 runtime timeline 處理此限制，不在 A 猜測結果。
原生寫法與公開 API 範例由 [program README](../../lib/zcu_tools/program/v2/README.md#原生寫法與具名-timeline) 維護，module 宣告與 owner 邊界見 [modules README](../../lib/zcu_tools/program/v2/modules/README.md#原生寫法)。
