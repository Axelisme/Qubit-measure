# FTF bring-up 與 CZ 校準操作指南

本指南先用既有單 qubit 實驗分別準備 F1、F2，再說明 microwave-activated CZ 的校準次序。F1、F2 是兩個 fluxonium，coupler 是這一對共用的 transmon。CZ 由兩條 fluxonium charge line 同頻、相鎖地驅動，coupler 只設 DC bias，沒有第三條 coupler AC drive。

## 目前可用與待交付能力

既有單 qubit 實驗的入口與操作見下方連結。Catalog 的 `ftf/joint_readout` 已提供四分支 raw 保存、離線分類與熱 population matrix，操作見 [joint readout 指南](joint_readout/guide.md)。`ftf/idle_zz` 也已提供 target F1/F2、spectator I/Xπ 的 conditional Ramsey、可選具名 DC 掃描與單頻模型頻差分析，操作見 [idle ZZ 指南](idle_zz/guide.md)。`ftf/conditional_spectroscopy` 已提供四名目分支的雙線 weak probe、可選雙 Xπ 對比與 frequency／具名 DC 掃描，操作見 [conditional spectroscopy 指南](conditional_spectroscopy/guide.md)。`ftf/cz_amplitude` 提供保持 signed gain 比例與 relative phase 的共同 scale 掃描、零相位完整週期擬合及兩線 pulse／slot proposals，見 [CZ amplitude 指南](cz_amplitude/guide.md)。`ftf/conditional_phase` 提供 target F1/F2 的 I/Xπ spectator、獨立 X/Y projection、共同 frequency 與 n 掃描及 phase 診斷，見 [conditional phase 指南](conditional_phase/guide.md)。Frequency 依 n=1 唯一 π 交點提出兩線 pulse 與獨立 slots，n=3/5 交點僅供人工確認。這些離線驗證不代表實際 FTF 控制鏈 smoke 已通過。

MuxGroup 綁定、tone-ordered raw、雙線 drive 配置與 FTF mux acquire 能力已落地。其餘校準實驗及首批 FTF 控制鏈驗收仍待交付。實際 mux 模擬須顯式設定 readout kind 與 receiver list，不能直接使用預設 pulse readout。

| 能力與交付狀態 | 操作前要確認的內容 |
| --- | --- |
| MuxGroup 綁定，已提供 | 有序 `resonators`／`qubits` 對應已可用。F1/F2 tone 取角色在 `qubits` 的位置。舊 `members` 不受支援。Form 以固定 capture 驗證 component 關係。 |
| Tone-ordered raw shots，已提供 | `MuxReadout.collect_raw_iq` 回最後一輪、原 loop 與 tone 順序的 normalized raw IQ。每 completed round 收集，host rounds 串接，不重複 normalization。平均 mux 頻譜不能代替同 shot IQ。 |
| 雙 charge line，已提供 | `f1_charge`／`f2_charge` 預設為 gen2／gen3，省略 operator 表示 charge。Form 仍依固定 capture 綁定，不硬編 channel。串擾係數不代表新增 coupler drive。 |
| FTF mux acquire，已提供 | Program 提供有序 tones。YAML 須顯式填 `readout.drive.kind: mux` 與 receiver list；預設仍是 pulse。所有 ADC 的同 shot 共享同一 joint 態，依 declaration／tone 順序輸出，不用兩路獨立抽樣冒充 joint shots。 |
| FTF 實驗與前端，部分提供 | Joint readout、idle ZZ、conditional spectroscopy、CZ amplitude 與 conditional phase 已提供 Form、Run、native save/load 及分析。CZ amplitude 與 conditional phase 分別提供 amplitude 與 frequency proposals。Constructive phase、Rabi chevron、local Z 及端到端 FTF smoke 仍待交付。本文的校準路線不是自動 workflow。 |

不要為了先跑通而另寫 raw parser、猜 ADC 排序，或從 simulator truth 補校準。本指南也不提供模擬器配置或啟動命令。

## Capture、Reset 與人工 Apply

Open 或表單的 Reset 會固定捕捉當時的 aliases、parameters 與 ModuleLibrary，並建立可覆寫的初值。Run 使用已解析的 cfg，不在執行時重新綁角色。表單 Reset 是重新取初值，與實驗中的 reset pulse 不同。

改 aliases、工作點或 library 後，先保留所需的手填值，再重新 Reset 或開新頁，檢查整份 cfg。只改全域 alias 不會更新已開表單。Apply 也不會讓其他已開表單自動取得新校準。

每次 Run 後都先看資料圖，再做 FIT 或互動分析。檢查 preview 的來源、目的 component、數值、單位與 pulse 內容。只勾選要接受的 rows，最後按 Apply。FIT 產生 proposal，不等於已套用校準；下一個實驗不能把未 Apply 的 proposal 當成有效 library。

保存本次資料及 cfg。Analyze、writeback 與重新載入後的分析都應依該次來源，不用目前表單補歷史缺值。Load 不會把歷史 cfg 回填成目前表單。缺少來源時，有些既有分析仍能顯示數值，但不能據此拼出未量測的 pulse proposal。

## 分別準備 F1 與 F2，實驗 0a

既有單 qubit 表單使用 `qubit`、`resonator` 與 DC 掃描所需的 `flux_dev`，不會自動跟隨 `f1`／`f2`。先把這些 scalar aliases 綁到 F1、它的 readout resonator 與對應 DC source，完成下列步驟。再改綁 F2，重新 Open／Reset，重做同一套步驟。每次 preview 都核對實際 component，避免把 F1 的結果套到 F2。

掃描前確認接線、device 名稱、native 單位與已授權範圍。缺必要校準時，從已量到的資料或人工確認的範圍填值。可編輯初值、template 與等待時間 fallback 都不是校準結果，也不是硬體安全預設。不要用 predictor、已知 Hamiltonian 或 simulator truth 代替找線與看圖。

| 次序與既有指南 | 要觀察的圖與判讀 | 可接受的校準與 preview |
| --- | --- | --- |
| 1. [Lookback](../lookback/guide.md) | 看 decimated ADC trace 的 baseline、訊號到達與完整 pulse。窗口要涵蓋訊號，不要把 baseline ripple 當成到達點。 | 檢查偵測的 arrival time，接受後寫到捕捉的 `resonator.ext.time_of_fly`，不是 pulse module。 |
| 2. [Resonator spectroscopy](../onetone/freq/guide.md) | 先用 linear sampling 看 resonance dip、complex response 與 fit，確認掃描跨過共振並可辨識 linewidth。 | Preview `resonator.freq`、`kappa`、`ext.theta0`。有 pulse-readout source 時可提 `resonator.readout.readout_rf`。Electrical delay 的寫回需可辨識的 branch 或可信 prior，不因有 fit 就全收。 |
| 3. [Resonator/DC map](../onetone/flux_dep/guide.md) | 看 DC source value × frequency 熱圖。從實測 map 拖線標 half-flux 與 integer-flux，按 Done。訊號太小時先縮頻窗、檢查 SNR，不自動找 sweet spot。 | Preview `qubit.flux_half`、`flux_int` 及 `flux_period = 2 × abs(flux_int − flux_half)`。Picker 的初始線位置不是實測校準。確認 `flux_dev` 與 `flux_unit`，表單不替你換單位。 |
| 4. 選定 DC 工作點後重跑 [resonator spectroscopy](../onetone/freq/guide.md) | 既有首次 bring-up 通常移到已辨識的 integer-flux 點，再看此處 resonance。若 FTF 選別的工作點，也須在該設定重校 readout。 | 人工設定並記錄 DC source values，再審閱新的 readout 頻率、linewidth 與 module proposal。套用 flux 位置校準不等於已把 source 移到該處。 |
| 5. [Qubit spectroscopy](../twotone/freq/guide.md) | 先在確認的 passband 寬掃找 qubit feature，再窄掃看 Lorentzian／sinc fit。雜訊或未辨識峰不算找到目標躍遷。 | Preview `qubit.freq`、`qubit.kappa`，確認是目標 qubit 線再 Apply。此步不寫 x180/x90 module。 |
| 6. [Amplitude Rabi](../twotone/rabi/amp_rabi/guide.md) 或 [length Rabi](../twotone/rabi/len_rabi/guide.md) | 在已量得的 qubit 頻率固定 length 掃 gain，或固定 gain 掃 length。圖中需看得到可辨識的振盪週期；只見一段斜率時先調整範圍。 | Preview 來源 pulse 的 π／π/2 結果。Pulse 寫到 `qubit.pulses.pi_amp`／`pi2_amp`，`qubit.module.x180`／`x90` 是另列的 slot proposals。核對 frequency、generator、waveform 與 gain／length，再選要套用的 module 和 slot rows。 |
| 7. [GE single-shot](../singleshot/ge/guide.md) | 用校準 π pulse 比較 probe off／on 的 IQ scatter、histogram 與 fit。確認兩個中心有限、可區分，且分支標記正確。Initial State 標記主要 population，不表示純態。 | Preview `qubit.ext.readout_cal` 下的 `g_center`、`e_center`、`sigma`、`fidelity`。Post-analysis 的 radius 與 3×3 diagnostic 不是後續 FTF 的 4×4 joint matrix。 |
| 8. [T1](../twotone/time_domain/t1/guide.md) | 看 π pulse 後的 delay decay、fit 與誤差。時間窗要涵蓋衰減到接近平坦的區域。 | Preview `qubit.t1` 與誤差。更新後重新 Reset 後續表單，才會重算依 T1 建立的等待初值。 |
| 9. [T2 Ramsey](../twotone/time_domain/t2ramsey/guide.md) 與 [T2 echo](../twotone/time_domain/t2echo/guide.md) | Ramsey 看 fringe 與衰減。Echo 看重聚焦後的總 delay decay。缺 T2 初值時人工填時間窗，不拿假設值當量測結果。 | Preview `qubit.t2r`／`t2e`。Ramsey 只有在來源 cfg、實際 detune 與 fringe fit 齊全時才另提 qubit frequency；echo 不寫頻率。兩者都不更新 pulse module。 |

Rabi 的 module 與 slot rows 可分開 Apply。只接受 module、不更新 slot 時，後續表單可能仍用舊 pulse。確認 x180/x90 指向這次接受的 keys；若 Ramsey 更新了 qubit frequency，也要核對後續實際使用的 pulse frequency，不能假設 scalar 更新會改寫 library pulse。

單 qubit GE 校準要記錄 readout route、gain、frequency、積分窗口與其他讀出條件。後續 joint 分類需要與實際 mux IQ 相容的兩組 GE 中心。改讀出條件後要重新確認中心；完成既有單 qubit GE 不代表其中心可直接用於不同 mux 讀出條件，也不代表實際 FTF joint 模擬已通過。

這裡重用既有序列，不另做 FTF 版單 qubit 實驗，也不接管 mux 單 qubit Rabi／T1／T2。FTF 後續分支保留熱 population，不藉 heralding、純態注入或 post-selection 改善圖形。

## FTF 的四個角色與 tone 對應

Joint readout 使用以下固定角色綁定，後續 FTF 表單沿用同一契約。MuxGroup `qubits` 支援已落地。

| Scalar alias | 指向 | 用途 |
| --- | --- | --- |
| `f1` | 第一個 fluxonium component | 取 `$f1.module.x180`／`x90`，供準備與 projection。 |
| `f2` | 另一個 fluxonium component | 取 `$f2.module.x180`／`x90`，與 F1 使用不同 charge generator。 |
| `coupler` | 本對的 transmon component | 集中保存這一對的兩線 CZ pulses 與 local-Z pair。 |
| `mux` | MuxGroup component | 從 `$mux.module.readout` 取完整 MuxReadoutSpec。缺來源時仍須填有效 readout，不以 template 代替校準。 |

F1 的 tone 是 `$f1` component 在固定 capture 的 `$mux.qubits` 中的位置，F2 同理。舉例，若 `f1=FA`、`f2=FB`，而 `qubits=[FB, FA]`，則 F1 對應 `tones[1]`，F2 對應 `tones[0]`。這是從 component 綁定推得的結果，不是讓操作員手填 tone index。

保留 group 與完整 tone 列表的原順序，不用 label、ADC number、generator 排序或 IQ cluster 排序猜 qubit 身分。MuxGroup 的 resonator／qubit 對應由 mux owner 驗證。缺 `qubits` 綁定或不包含 F1/F2 時，必須在 acquire 前明示錯誤，不以舊 `members` 或兩份獨立讀出替代。

改角色後重新 Reset，核對 F1/F2 pulses、兩個 generator、mux readout 與 DC source。Core 使用具名 device 及 native 單位，不把 alias 語法送給 driver。F1/F2 bias 由操作員提供；掃 coupler source 不等於只改 coupler loop flux，也不會自動補償 F1/F2 的偏移。

## 第一批 0b–6 的校準路線，0b／1／2／4／5 已提供離線驗證

以下次序以 0a 的單 qubit pulses、readout 與人工記錄的 bias 為起點。各實驗分別 Run、分析、preview 與 Apply，不會自動啟動下一步。

`B(ab)` 是在熱初態上對 F1/F2 分別施加 I 或 Xπ 的名目準備，a 對應 F1，b 對應 F2。`B(11)` 不宣稱得到純 `|11⟩`。全部 shots 保留。

| 次序 | 操作與圖 | 判讀與後續動作 |
| --- | --- | --- |
| 0b. [Joint readout](joint_readout/guide.md) | 四個 `B(00)`／`B(01)`／`B(10)`／`B(11)` 分支各接一次 mux readout，收同 shot 兩 tone IQ。保存 raw IQ，顯示 acquired shots、counts 與 4×4 confusion matrix。 | 行是名目準備，列是同 shot 的 00／01／10／11 分類，兩軸都依 F1、F2 順序。逐 tone 使用有效 GE 中心做最近中心 G/E 分類，再直接 joint 計數，不能乘兩張 marginal matrices。 |
| 1. [Idle ZZ](idle_zz/guide.md) | 選 F1 或 F2 為 target，spectator 做 I／Xπ 兩分支。人工填 detune，掃 Ramsey delay，看兩條曲線、fit 與 ζ；需要時外層掃具名 DC source。 | ζ 是兩名目分支在單頻模型假設下的擬合頻差。符號依人工 detune 的同側前提，近 0／2×detune 時拒絕 ζ。有限 fit 不排除熱混合；看 fit_quality 與曲線判斷模型，不稱 pure ZZ、不 writeback 或自動調偏。 |
| 2. [Conditional higher-level spectroscopy](conditional_spectroscopy/guide.md) | 四個名目分支做雙線 weak probe，掃共同 frequency，可加 coupler DC 軸，並比較可選雙 Xπ 對比 pulse 的訊號。看四分支頻譜／熱圖。 | 人工從圖選目標 transition 與後續 probe 窗口。不自動把最深峰當 CZ frequency，不替峰命名 bare coupler 能級，也不擬合 EC/EJ/J。 |
| 3a. Constructive phase | 在 `B(11)` 固定兩線 gain 比例，掃 relative phase × duration。看 Rabi 干涉圖與可辨識的 Rabi rate。 | 選目標 transition 的建設性 phase。有效結果可提出 F2 相對 F1 的 pulse phase 更新；不掃寄生線抵消，也不最佳化 relative gain ratio。 |
| 3b. Rabi chevron | 固定 relative phase 與兩線 gain，掃 common frequency × duration。看 chevron 與回返曲線。 | 人工選 CZ width 與 frequency 窗口。單一 population 谷值不是高 fidelity gate 的證據。兩線同頻、同 duration、同時間原點，waveforms 由 cfg 明示；cosine 包絡只是起點。 |
| 4. [CZ amplitude](cz_amplitude/guide.md) | `B(11)` 後播雙線 CZ，再做雙 Xπ 對比與 mux readout。掃共同 gain scale，保持原 signed gain 比例、relative phase、frequency 與 waveform。 | 用零相位 cosine 擬合 joint 00 的第一正回返週期。實測須從0覆蓋完整週期，否則不提案、不外插。Preview 後逐列接受兩線 pulse 與獨立 slots。回返只是 thermal leakage proxy；看 fit_quality 與大振幅 AC Stark 偏離，不當 gate fidelity。 |
| 5. [Conditional phase／frequency](conditional_phase/guide.md) | Spectator 做 I／Xπ，target 做 π/2，接 n 次 CZ 與 X/Y projection。掃共同 frequency 與 n，起點可用 n=1、3、5。 | 看 quadratures、contrast、conditional phase 與奇數 n 交點。X/Y 是獨立 acquisitions，不是同 shot 連續測量。核對 phase wrapping 與 branch 標記。只以 n=1 相鄰點 unwrap 後線性內插的唯一 π 交點提 frequency；缺失、無交點、多交點或退化時不提案，n=3/5 僅診斷。兩線 pulse 與 slots 各自 preview／selected Apply。 |
| 6. Local Z | 幅度與頻率選定後，從 `B(00)` 分別量 F1/F2 的 π/2、n 次 CZ、X/Y projection，另一 qubit 不做準備翻轉。掃 n，看兩個 phase slopes。 | 兩個分量都可辨識時才提 local-Z pair。熱態成分仍在數據內，不能把 slope 稱為已純化分支真值，也不能把缺失 correction 填 0。 |

Joint matrix 的圖意是「名目準備→觀測結果，含熱 population」。它包含準備及讀出效果，不是 pure-state assignment matrix。不要設 matrix 對角值門檻、用 inverse 修正後續 population，或由它推 gate fidelity。最近中心二分類不以 radius 丟掉 Other shots，因此也不聲稱能辨認 leakage。缺失、非有限或重合的 GE 中心須明示無效。

### Amplitude 與 frequency 人工交替三輪

完成 chevron 選窗後，建議以同一組雙線 gate cfg 人工做三輪 `4 → 5`。

1. 做 amplitude scan，檢查回返 fit，preview 後接受有效的兩線 pulse 更新。
2. 重新 Reset 下一個表單，確認它使用剛接受的 amplitude，再做 conditional phase／frequency scan。檢查交點與 contrast，preview 後接受有效 frequency 更新。
3. 將這一對步驟重做至共三輪。每輪保留 cfg、圖與選擇理由，縮窗以實測結果為準，不要求系統計數或自動串接。

沒有交點、contrast 不可辨識或 fit 失敗時，先保留圖與原因，重新檢查範圍、讀出和 pulse，不選最近預設值冒充成功。三輪是操作建議，不是驗收門檻，也不保證 conditional phase 達 π。

### Coupler 保存與 local-Z 重測

初版每個 coupler 保存一對 F1/F2 的 CZ 校準。Preview 的目的地如下，不另建 gate registry 或多版本 CZ module kind。

| 目的地 | 內容 |
| --- | --- |
| `$coupler.pulses.cz_f1`、`$coupler.pulses.cz_f2` | 兩個普通 Pulse ModuleSpec，各自包含本次實測 generator、frequency、gain、phase 與 waveform。 |
| `$coupler.module.cz_f1`、`$coupler.module.cz_f2` | 分開的 slot rows，指向同份 capture 解析出的上述 library keys。 |
| `$coupler.ext.cz_phase` | 一筆 compound parameter，包含 F1/F2 component 名稱與 `correction_f1_deg`、`correction_f2_deg`。兩個 correction 來自同次 local-Z 實驗。 |

Amplitude／frequency proposal 只換本次擬合的共同 scale 或 frequency，保留來源 cfg 的 gain 比例、relative phase 與 waveform。不要用目前 library 重拼一組未量測的 CZ pulse。

若測得每 CZ 的 logical Bloch phase accrual 是 θ1、θ2，local correction 就是 −θ1、−θ2，單位 degree。後續序列把 correction 用在 single-qubit logical frame，不額外播放 Z pulse，也不改兩線 constructive relative phase。Program pulse phase 的符號須沿既有 frame convention，不能直接假設與 logical Z 同號。

先完成 amplitude／frequency 選定，再量 local Z。之後改任一線的 amplitude、frequency、duration、waveform、relative phase 或 DC 工作點，都要重新檢查 CZ 並重測 local Z，不能沿用舊 pair。交換 F1/F2 也不能靜默沿用；先核對保存的 component 名稱與目前角色，再重做對應校準。本流程沒有自動 staleness 判定。

Module、slot 與 phase rows 都可獨立勾選 Apply，不是跨檔原子交易。只套用部分 rows 後，務必檢查下一次 Run 的完整 cfg，避免新 pulse 配到舊 phase correction。

## 第二批 IRB 與結果解讀

Two-qubit RB／IRB 是第一批控制鏈驗收後的獨立交付，不以既有單 qubit `twotone/irb` 代替。它需要配對的 two-qubit Clifford reference／interleaved sequences、各自完整 inverse、實際雙線 CZ 與 local frames。

後續 IRB 要保留相同 seed 的配對資料與 round 軸，從 same-shot joint 00 計算 survival。它使用 d=4 的 error estimate，不套 d=2 的單 qubit 係數。熱 population、準備／讀出與模型假設都要留在解讀中，不宣稱重現論文 fidelity。

第一批只驗收控制鏈 smoke，不設 fidelity、leakage、π phase 或 matrix 對角值門檻。但低 contrast、無法辨識的 fit 與執行錯誤是不同結果。低 contrast 可保留為資料；無法辨識就回報原因且不提無效校準；接線、時序、cfg 或 acquisition 錯誤不能當成成功 smoke。

後續 FTF 資料須保留 raw complex IQ、branch／sweep 座標、tone 順序、shot 對應與本次 cfg，不能先平均再稱 single-shot。取消時保留已取得資料與未完成位置。分析只用完整分支或完整配對，缺一個 quadrature 不推 phase。這條路線不做 heralding、readout 錯誤反演、寄生抵消或自動校準 workflow。
