# FTF bring-up 與 CZ 校準操作指南

**讀者：** operator、agent。從 GUI 或 MCP 準備 F1、F2 並校準 microwave-activated CZ。

**Last updated:** 2026-10-21

F1、F2 是兩個 fluxonium，coupler 是這一對共用的 transmon。CZ 由兩條 fluxonium charge line 同頻、相鎖地驅動，coupler 只設 DC bias。

## 校準前確認

先分別完成下方的單 qubit 校準，再依序量 joint readout、idle ZZ、conditional spectroscopy、constructive phase、chevron、CZ amplitude、conditional phase 與 local Z。最後用 paired RB／IRB 評估結果。每一步都需要獨立 Run、看圖、分析與人工 Apply。

確認接線、device 名稱、native 單位與已授權範圍。`f1_charge`／`f2_charge` 的預設接線為 gen2／gen3，省略 operator 表示 charge；表單實際使用 Open／Reset 時的角色接線。兩角色須使用不同 generator，串擾係數仍只描述這兩條 charge line。

實際 mux 模擬須明填 `readout.drive.kind: mux` 與 receiver list，預設 readout kind 為 pulse。每個同 shot 的完整 tones 必須對應同一個 joint 態。使用保存的同 shot raw IQ 與 tone 順序分析，平均頻譜或兩路獨立抽樣的 IQ 無法提供 joint shots。校準來源限實測資料或人工確認的範圍；predictor、Hamiltonian 假設、simulator truth、template 與等待時間 fallback 都只是假設或初值。每個初值都須逐項核對硬體安全範圍。

離線合成資料可檢查操作與分析流程。現場仍須驗證 FTF／GUI 控制鏈、接線、時序、cfg 與 acquisition。低 contrast 是可保留的資料結果；fit 無法辨識時保存原因並停止該校準提案；執行錯誤須先排除再繼續。現場驗證以流程與資料完整性為準，fidelity、leakage、π phase 與 matrix 對角值留作物理解讀。

## Open、Reset 與人工 Apply

Open 或表單的 Reset 會固定捕捉當時的 aliases、參數與 module library，建立可覆寫的初值。Run 全程使用這份已解析 cfg。表單 Reset 是重新取初值，實驗中的 reset pulse 是另一個操作。

改 aliases、工作點或 library 後，先保留所需的手填值，再重新 Reset 或開新頁，檢查整份 cfg。已開表單維持原來源，其他頁面 Apply 的校準也須經 Reset 才會取得。

每次 Run 後先看資料圖，再做 FIT 或互動分析。檢查 preview 的來源、目的 component、數值、單位與 pulse 內容。只勾選要接受的 rows，最後按 Apply。只有已 Apply 的 proposal 才能作為後續校準來源。

保存本次資料及 cfg。重新載入後的分析與提案只用該次來源；Load 保持目前表單原值。歷史來源缺失時，部分分析仍可顯示數值，pulse proposal 則需要完整的量測來源。

## Run 前的表單與環境檢查

MCP `status` 的 `ready.can_run` 只檢查 entry、active context、SoC 與至少一張 tab 是否存在。它不表示這張 FTF 表單有效。即使回覆 `can_run=true`、`missing=[]`，仍先用 `tab.get_cfg` 讀目標 tab 的 `input`、`values`、`errors`，補齊必填值並重新檢查。查單一欄位的 metadata 用 `tab.describe_cfg`。呼叫方式見 [MCP 連線與觀察](../../../lib/zcu_tools/mcp/measure/README.md#連線與觀察)。

表單有效只表示 cfg 通過驗證，不表示初值已校準、硬體操作已獲授權或 Run 一定能取得控制權。先核對實際接線、安全範圍與量測來源，再 Run。

## 採樣數與成本

Run 前先從目前 cfg 計算規模。`reps` 是每 round 的重複次數，`rounds` 是 software acquisition rounds。完整量測的每座標、每分支採樣數是 `reps × rounds`。有多個 sweep 軸時，座標數是各軸點數的乘積；分支數另外乘入。Mux 的兩個 tones 屬於同一 shot，不是兩份獨立 joint samples。

以下是單 qubit 準備流程的初開值，尚未計入對 F1、F2 各做一次的倍數。Open／Reset 的來源或人工編輯可能改變目前 cfg，Run 前以表單為準。

| 入口 | 初開採樣設定 | 一次完整 Run 的採樣數 |
| --- | --- | --- |
| Resonator spectroscopy，onetone freq | 301 個頻率，reps=1000、rounds=10 | 每頻率 10,000 次，共 3,010,000 次。 |
| Qubit spectroscopy，twotone freq | 301 個頻率，reps=1000、rounds=10 | 每頻率 10,000 次，共 3,010,000 次。 |
| GE single-shot | shots=100000，probe off／on 各一分支 | 每分支 100,000 shots，共 200,000 shots。此入口內部使用 reps=shots、rounds=1，不再乘一組平均次數。 |

兩 qubit FTF 表單 Open／Reset 的初值為 reps=1000、rounds=10，每座標、每分支有 10,000 個 raw joint shots。這個數量不代表分類、擬合或校準已有足夠統計。各入口的總量依下列軸計算，DC sweep 關閉時 DC 點數取 1。

| FTF 入口 | 總 joint shots |
| --- | --- |
| Joint readout | `4 × reps × rounds`，四個名目準備分支。 |
| Idle ZZ | `DC 點數 × 2 spectator 分支 × delay 點數 × reps × rounds`。 |
| Conditional spectroscopy | `DC 點數 × 4 準備分支 × frequency 點數 × reps × rounds`。 |
| Constructive phase／chevron | `phase 或 frequency 點數 × duration 點數 × reps × rounds`。Chevron 初開為 25×101 點，端點仍須人工提供。 |
| CZ amplitude | `scale 點數 × reps × rounds`。 |
| Conditional phase | `frequency 點數 × gate-count 個數 × 2 spectator 分支 × 2 projections × reps × rounds`。 |
| Local Z | `gate-count 個數 × 2 targets × 2 projections × reps × rounds`。 |

估算硬體採集時間時，先用 `總 shots × 每 shot 序列時間`。序列時間要包含 preparation、drive、readout 與 relax delay。假設每 shot 共 10 µs，上表 onetone、twotone 的純序列時間皆為 30.1 s。這只是換算例，不是執行時間保證。編譯、傳輸、round 交接、DC ramp／settle、繪圖與保存都另計；gate-count 改變序列長度時，要分座標加總。

FTF raw IQ 保存完整 tones 與 shots。Complex128 IQ 的陣列大小約為 `總 joint shots × tone 數 × 16 bytes`，尚未計入分析陣列與保存開銷。例如 chevron 初值 25×101 點、兩 tones、reps=1000、rounds=10，raw IQ 約 808 MB。模擬器可重用相同物理條件的計算，因此 wall time 不一定隨 reps 線性增加；抽樣與 raw 輸出仍有成本。先以同一 backend、相近序列與點數實測，再外推，不把硬體序列時間當成模擬秒數。

要縮短操作診斷時，明確減少點數或採樣數並保存該次 cfg，標記為「診斷、非校準」。Open／Reset 會恢復 reps=1000、rounds=10。低採樣圖只能用來排查接線、流程與掃描窗口，接受校準前仍須取得可判讀的資料與 uncertainty。

## 分別準備 F1 與 F2，實驗 0a

既有單 qubit 表單使用 `qubit`、`resonator` 與 DC 掃描所需的 `flux_dev`。先把這些 scalar aliases 綁到 F1、它的 readout resonator 與對應 DC source，完成下列步驟。再改綁 F2，重新 Open／Reset，重做同一套步驟。每次 preview 都核對實際 component，避免把 F1 的結果套到 F2。

| 次序與既有指南 | 要觀察的圖與判讀 | 可接受的校準與 preview |
| --- | --- | --- |
| 1. [Lookback](../lookback/guide.md) | 看 decimated ADC trace 的 baseline、訊號到達與完整 pulse。窗口要涵蓋訊號，並區分 baseline ripple 與到達點。 | 檢查 arrival time，接受後寫到捕捉的 `resonator.time_of_fly`。 |
| 2. [Resonator spectroscopy](../onetone/freq/guide.md) | 用 linear sampling 看 resonance dip、complex response 與 fit，確認掃描跨過共振且 linewidth 可辨識。 | Preview `resonator.freq`、`kappa`、`ext.theta0`。有 pulse-readout source 時可提 `resonator.readout.readout_rf`。Electrical delay 寫回需可辨識的 branch 或可信 prior。 |
| 3. [Resonator/DC map](../onetone/flux_dep/guide.md) | 看 DC source value × frequency 熱圖。從實測 map 拖線標 half-flux 與 integer-flux，按 Done。訊號太小時先縮頻窗、檢查 SNR。 | Preview `qubit.flux_half`、`flux_int` 及 `flux_period = 2 × abs(flux_int − flux_half)`。Picker 的初始線只是初值。確認 `flux_dev` 與 `flux_unit`，使用 source 的原單位。 |
| 4. 選定 DC 工作點後重跑 [resonator spectroscopy](../onetone/freq/guide.md) | 首次 bring-up 通常移到已辨識的 integer-flux 點，再看此處 resonance。選其他工作點時也須在該設定重校 readout。 | 人工設定並記錄 DC source values，再審閱新的 readout 頻率、linewidth 與 module proposal。Apply flux 位置校準與移動 source 是兩個獨立操作。 |
| 5. [Qubit spectroscopy](../twotone/freq/guide.md) | 在確認的 passband 寬掃找 qubit feature，再窄掃看 Lorentzian／sinc fit。確認峰可辨識且屬於目標躍遷。 | Preview `qubit.freq`、`qubit.kappa`。x180/x90 pulse 另由 Rabi 校準。 |
| 6. [Amplitude Rabi](../twotone/rabi/amp_rabi/guide.md) 或 [length Rabi](../twotone/rabi/len_rabi/guide.md) | 在已量得的 qubit 頻率固定 length 掃 gain，或固定 gain 掃 length。圖中需看得到振盪週期；只見斜率時先調範圍。 | Preview 來源 pulse 的 π／π/2 結果。Pulse 寫到 `qubit.pulses.pi_amp`／`pi2_amp`，`qubit.module.x180`／`x90` 是獨立 slot rows。核對 frequency、generator、waveform 與 gain／length，再勾選。 |
| 7. [GE single-shot](../singleshot/ge/guide.md) | 用校準 π pulse 比較 probe off／on 的 IQ scatter、histogram 與 fit。確認兩中心有限、可區分且分支標記正確。Initial State 標記主要 population。 | Preview `qubit.ext.readout_cal` 的 `g_center`、`e_center`、`sigma`、`fidelity`。Radius 與 3×3 diagnostic 供單 qubit 判讀，FTF joint 分類使用下方的 4×4 matrix。 |
| 8. [T1](../twotone/time_domain/t1/guide.md) | 看 π pulse 後的 delay decay、fit 與誤差，時間窗涵蓋衰減至接近平坦的區域。 | Preview `qubit.t1` 與誤差。更新後 Reset 後續表單，重算依 T1 建立的等待初值。 |
| 9. [T2 Ramsey](../twotone/time_domain/t2ramsey/guide.md) 與 [T2 echo](../twotone/time_domain/t2echo/guide.md) | Ramsey 看 fringe 與衰減，echo 看重聚焦後的總 delay decay。缺 T2 初值時人工填時間窗。 | Preview `qubit.t2r`／`t2e`。Ramsey 需來源 cfg、實際 detune 與 fringe fit 齊全才另提 frequency；echo 只提 T2。Pulse module 維持原值。 |

Rabi 的 module 與 slot rows 可分開 Apply。確認 x180/x90 指向這次接受的 keys；只更新 module 時，slot 可能仍指向舊 pulse。Ramsey 更新 qubit frequency 後，也須核對後續實際 pulse frequency，scalar 與 library pulse 是獨立設定。

記錄單 qubit GE 的 readout route、gain、frequency、積分窗口與其他讀出條件。Joint 分類需要與實際 mux IQ 相容的兩組 GE 中心，改讀出條件後要重新確認。後續 FTF 分支保留全部熱 population，從原始熱態開始，保留每個 shot 與其熱 population。單 qubit Rabi／T1／T2 使用上述既有實驗入口。

## FTF 的四個角色與 tone 對應

| Scalar alias | 指向（kickstart／simulation 新 entry 的綁定） | 用途 |
| --- | --- | --- |
| `f1` | 第一個 fluxonium component，`Q1` | 取 `$f1.module.x180`／`x90`，供準備與 projection。既有 entry 缺此角色時，在 Aliases 加入 `f1` 並綁到 `Q1`。 |
| `f2` | 另一個 fluxonium component，`Q2` | 取 `$f2.module.x180`／`x90`，使用另一個 charge generator。既有 entry 缺此角色時，在 Aliases 加入 `f2` 並綁到 `Q2`。 |
| `coupler` | 本對 transmon component，`coupler` | 保存這一對的兩線 CZ pulses 與 local-Z pair。 |
| `mux` | MuxGroup component，`MG12` | 從 `$mux.module.readout` 取完整 mux readout。缺來源時須手填有效設定；template 只是起點。 |

F1 的 tone 是 `$f1` component 在 Open／Reset 所固定的 `$mux.qubits` 中的位置，F2 同理。例如 `f1=FA`、`f2=FB`，`qubits=[FB, FA]`，則 F1 對應 `tones[1]`，F2 對應 `tones[0]`。Tone index 由綁定決定，畫面只供核對。

保留完整 tone 列表原順序。MuxGroup 的 `resonators`／`qubits` 須逐項對應且包含 F1/F2；缺少綁定會在 acquire 前報錯。使用 `qubits` 指定身分，label、ADC number、generator 或 IQ cluster 排序都無法替代此綁定。

改角色後 Reset，核對 F1/F2 pulses、兩個 generator、mux readout 與 DC source。DC 操作使用具名 device 與 native 單位。F1/F2 bias 由操作員提供並記錄；掃 coupler source 也可能影響其他 loop，F1/F2 偏移須另行確認與處理。

## Joint readout 到 local Z 的校準路線

以下以 0a 的 pulses、readout 與已記錄 bias 為起點。`B(ab)` 是熱初態上對 F1/F2 分別施加 I 或 Xπ，a 對應 F1，b 對應 F2。`B(11)` 仍是熱 ensemble，全部 shots 保留。

| 次序 | 操作與圖 | 判讀與後續動作 |
| --- | --- | --- |
| 0b. [Joint readout](joint_readout/guide.md) | 四個 B00／B01／B10／B11 分支各接一次 mux readout，保存同 shot 完整 tone IQ。看 acquired shots、counts 與 4×4 matrix。 | 行是名目準備，列是觀測 00／01／10／11，兩軸都依 F1、F2 排序。逐 tone 最近 GE 中心分類，再直接 joint 計數。 |
| 1. [Idle ZZ](idle_zz/guide.md) | Target 選 F1 或 F2，spectator 做 I／Xπ。人工填 detune，掃 delay，看曲線、fit 與 ζ；需要時外層掃具名 DC source。 | ζ 是單頻模型下的兩分支擬合頻差，符號依 detune 同側前提。近 0／2×abs(detune) 時回不可辨識。依曲線與 fit_quality 判讀熱混合，調偏由操作員決定。 |
| 2. [Conditional higher-level spectroscopy](conditional_spectroscopy/guide.md) | 四分支雙線 weak probe，掃共同 frequency，可加 coupler DC 軸與雙 Xπ 對比。 | 人工選 transition 與後續 probe 窗口。頻譜是診斷訊號，能級命名、EC/EJ/J 與 CZ frequency 需另行確認。 |
| 3a. [Constructive phase](constructive_phase/guide.md) | B11 固定 signed gain 比例，掃 relative phase × duration。 | 無衰減 cosine 求各 trace rate，rate < 1/span 為未解析。以 `rate² = A + B cosφ + C sinφ` 求最大 rate 的 `atan2(C,B)`，只提 F2 pulse 與獨立 slot。Nyquist 與 signed gain ratio 由操作員確認，此掃描只調 relative phase。 |
| 3b. [Rabi chevron](chevron/guide.md) | 固定 relative phase 與兩線 gain，掃 common frequency × duration。 | 人工選 CZ width 與 frequency 窗口。Population 谷值只供選窗。兩線同頻、同 duration、同時間原點，waveforms 由 cfg 明示，cosine 包絡只是起點。 |
| 4. [CZ amplitude](cz_amplitude/guide.md) | B11 後播雙線 CZ、雙 Xπ 對比與 mux readout。掃共同 gain scale，固定原 signed gain 比例、relative phase、frequency 與 waveform。 | 零相位 cosine 擬合 joint 00 的第一正回返週期。實測從 0 覆蓋完整週期才提案，範圍不足即停止提案。逐列 preview 兩線 pulse 與 slots。回返是 thermal leakage proxy，須看 fit_quality 與大振幅 AC Stark 偏離。 |
| 5. [Conditional phase／frequency](conditional_phase/guide.md) | Spectator I／Xπ，target π/2，n 次 CZ，獨立 X/Y projection。掃共同 frequency 與 n，起點可用 `[1,3,5]`。 | 以 n=1 相鄰點 unwrap 後線性內插的唯一 π 交點提 frequency。缺失、無交點、多交點或退化均保留原因並停止提案。n=3/5 僅診斷。核對 wrapping、contrast、branch 標記，逐列接受 pulse／slots。 |
| 6. [Local Z](local_z/guide.md) | 從 B00 分別量 F1/F2 的 π/2、n 次 CZ、獨立 X/Y projection，另一 qubit 維持原熱態。至少三個連續非負 n，整列 X/Y 完整才相鄰 unwrap 與含 intercept 的 OLS。 | Correction=-slope，保持原值。兩分量可辨識才提一筆具 component 名稱的 pair。採樣假設為相鄰真實增量小於 180°，每 gate 額外 360° 無法辨認。看 fit_quality 與 uncertainty，保留熱態模型限制；缺 correction 回報缺失。 |

Joint matrix 表示「名目準備→觀測結果，含熱 population」，包含準備與讀出效果。Matrix 的用途限於診斷原始熱 population 與讀出結果；後續分析保留原 population。Matrix inverse、對角值門檻與 gate fidelity 推論都超出此校準的適用範圍。最近中心二分類保留 Other shots，leakage 仍未解析。缺失、非有限或重合的 GE 中心均回報無效。

### Amplitude 與 frequency 人工交替三輪

完成 chevron 選窗後，建議以同一組雙線 cfg 人工做三輪 `4 → 5`。

1. 做 amplitude scan，檢查回返 fit，preview 後接受有效兩線 pulse 更新。
2. Reset 下一個表單，確認使用剛接受的 amplitude，再量 conditional phase／frequency。檢查交點與 contrast，preview 後接受有效 frequency 更新。
3. 重做至共三輪，每輪保留 cfg、圖與選擇理由，依實測結果縮窗。

無交點、contrast 未解析或 fit 失敗時，保存圖與原因，先檢查範圍、讀出與 pulse，再決定重測。三輪是人工操作建議，conditional phase 是否達 π 仍須由資料確認。

### Coupler 保存與 local-Z 重測

每個 coupler 保存一對 F1/F2 的 CZ 校準。

| 目的地 | 內容 |
| --- | --- |
| `$coupler.pulses.cz_f1`、`$coupler.pulses.cz_f2` | 兩個 pulse，含本次實測 generator、frequency、gain、phase 與 waveform。 |
| `$coupler.module.cz_f1`、`$coupler.module.cz_f2` | 獨立 slot rows，指向同次 Open／Reset 解析的上述 library keys。 |
| `$coupler.ext.cz_phase` | 一筆 parameter，含 F1/F2 component 名稱與 `correction_f1_deg`、`correction_f2_deg`，兩 correction 來自同次 local-Z 量測。 |

Amplitude／frequency proposal 只換擬合的共同 scale 或 frequency，保留來源 cfg 的 gain 比例、relative phase 與 waveform。所有提案都必須能追溯到本次量測的 pulse。

每 CZ 的 logical Bloch phase accrual 為 θ1、θ2 時，local correction 是 −θ1、−θ2，單位 degree。後續序列以 virtual Z 更新 single-qubit logical frame，兩線 constructive relative phase 維持原值。Pulse phase 的正負換算須依既有 frame convention 核對。

先選定 amplitude／frequency，再量 local Z。之後改任一線的 amplitude、frequency、duration、waveform、relative phase 或 DC 工作點，都要重新檢查 CZ 並重測 local Z。交換 F1/F2 後也須先核對保存的 component 名稱，再重做對應校準。校準時效由操作員確認。

Pulse、slot 與 phase rows 都可獨立 Apply。部分套用後，檢查下一次 Run 的完整 cfg，確認 pulse 與 phase correction 屬於同一組已接受校準。

## Paired RB／IRB 與資料解讀

[Two-qubit RB／IRB](rb/guide.md) 使用配對的 two-qubit Clifford reference／CZ-interleaved sequences、各自完整 inverse、實際雙線 CZ 與 local frames。它與單 qubit `twotone/irb` 分別量不同的控制序列。

保留相同 seed 的配對資料與 round 軸，從 same-shot joint 00 算 survival。使用 d=4 的 error estimate，保留負值及 paired-seed bootstrap interval。熱 population、準備／讀出效果與模型假設均須納入判讀，物理 gate 表現需由實際量測確認。

所有 FTF 資料保留 raw complex IQ、branch／sweep 座標、完整 tone 順序、shot 對應與本次 cfg。取消時保存已取得資料與未完成位置。Phase 只用完整 X/Y 分支；RB／IRB 只用完整配對。後續量測保留原始熱 ensemble 與原始讀出效果。寄生抵消需另行研究，各校準由操作員逐項執行。
