# Two-qubit bring-up 與 CZ 校準操作指南

**讀者：** operator、agent。從 GUI 或 MCP 準備 Qubit1、Qubit2 並校準 microwave-activated CZ。

**Last updated:** 2026-11-22，位置命名與舊 point 重綁

`qubit1`、`qubit2` 是量測位置，不代表元件種類。下文以 FTF 為例：兩條 qubit charge line 同頻、相鎖地驅動 CZ，coupler 只設 DC bias。

本指南的順序對應 Leon Ding, *Novel Gates with Superconducting Fluxonium Qubits*（MIT 物理博士論文，2023） 的校準流程：§4.1.3 以 heralding 初始化，§4.3.2 校準單 qubit gate，§5.3.2–5.3.5 校準 CZ。偏離該流程時，在量測紀錄寫明原因。

## 校準前確認

先分別完成下方的單 qubit 校準，再依序量 joint readout、idle ZZ、conditional spectroscopy、constructive phase、chevron、CZ amplitude、conditional phase 與 local Z。最後用 paired RB／IRB 評估結果。每一步都需要獨立 Run、看圖、分析與人工 Apply。

確認接線、device 名稱、native 單位與已授權範圍。FTF 模型的 `f1_charge`／`f2_charge` 預設接 gen2／gen3，省略 operator 表示 charge。表單使用 Open／Defaults 捕捉的接線。兩條 drive line 須用不同 generator，串擾係數仍只描述這兩條線。

FTF 的 mux 模擬在 `readout.mux` 宣告 generator 與 receiver 清單 `adc`；tones 仍填在 mux 讀出模組。[欄位指南](../../simulation_envs/FTF/field-guide.md#generatoradc-與-entry-接線對照)列出接線對照。`readout.drive` 屬於獨立的單 tone pulse 路徑。每個同 shot 的完整 tones 必須對應同一個 joint 態。取數時依固定 tone 順序聯合分類 Qubit1、Qubit2，保存類別計數。平均頻譜或兩路獨立抽樣的 IQ 無法提供 joint shots。校準來源限實測資料或人工確認的範圍；predictor、Hamiltonian 假設、simulator truth、template 與等待時間 fallback 都只是假設或初值。每個初值都須逐項核對硬體安全範圍。

離線合成資料可檢查操作與分析流程。現場仍須驗證 FTF／GUI 控制鏈、接線、時序、cfg 與 acquisition。低 contrast 是可保留的資料結果；fit 無法辨識時保存原因並停止該校準提案；執行錯誤須先排除再繼續。現場驗證以流程與資料完整性為準，fidelity、leakage、π phase 與 matrix 對角值留作物理解讀。

## Open、Defaults 與人工 Apply

開啟 tab 或按 Defaults 時，表單會固定捕捉當時的 aliases、參數與 module library，建立可覆寫的初值。Run 全程使用這份已解析 cfg。Defaults 只重填表單初值，不發射 reset pulse。

改 aliases、工作點或 library 後，先保留所需的手填值，再按 Defaults 或開新頁，檢查整份 cfg。已開表單維持原來源，其他頁面 Apply 的校準也須按 Defaults 才會取得。

每次 Run 後先看資料圖，再做 FIT 或互動分析。檢查 preview 的來源、目的 component、數值、單位與 pulse 內容。只勾選要接受的 rows，最後按 Apply。只有已 Apply 的 proposal 才能作為後續校準來源。

保存本次資料及 cfg。Load 保持目前表單原值。寫回使用該次 Run／Load 的 capture，不使用目前 aliases。沒有 capture 就不能寫回。

新版 HDF5 的位置軸使用新名，例如 `terminal_qubit1`、`terminal_qubit2`。若舊檔的軸名不符新版宣告，Load 會拒絕。保留原檔，用對應的舊版軟體讀取。若結果格式相容但 cfg major 不同，Load 保留 Result 與 metadata，不還原 cfg。需要 cfg 的分析會報錯。

## Run 前的表單與環境檢查

MCP `status` 的 `ready.can_run` 只檢查 entry、active context、SoC 與至少一張 tab 是否存在。它不表示這張 two-qubit 表單有效。即使回覆 `can_run=true`、`missing=[]`，仍先用 `tab.get_cfg` 讀目標 tab 的 `input`、`values`、`errors`，補齊必填值並重新檢查。查單一欄位的 metadata 用 `tab.read_recipe_field`，路徑是 logical path。呼叫方式見 [MCP 連線與觀察](../../../lib/zcu_tools/mcp/measure/README.md#連線與觀察)。

表單有效只表示 cfg 通過驗證，不表示初值已校準、硬體操作已獲授權或 Run 一定能取得控制權。先核對實際接線、安全範圍與量測來源，再 Run。

## 採樣數與成本

Run 前先從目前 cfg 計算規模。`reps` 是每 round 的重複次數，`rounds` 是 software acquisition rounds。完整量測的每座標、每分支採樣數是 `reps × rounds`。有多個 sweep 軸時，座標數是各軸點數的乘積；分支數另外乘入。Mux 的兩個 tones 屬於同一 shot，不是兩份獨立 joint samples。

以下是單 qubit 準備流程的初開值，尚未計入對 Qubit1、Qubit2 各做一次的倍數。Open／Defaults 的來源或人工編輯可能改變目前 cfg，Run 前以表單為準。

| 入口 | 初開採樣設定 | 一次完整 Run 的採樣數 |
| --- | --- | --- |
| Resonator spectroscopy，onetone freq | 301 個頻率，reps=1000、rounds=10 | 每頻率 10,000 次，共 3,010,000 次。 |
| Qubit spectroscopy，twotone freq | 301 個頻率，reps=1000、rounds=10 | 每頻率 10,000 次，共 3,010,000 次。 |
| GE single-shot | shots=100000，probe off／on 各一分支 | 每分支 100,000 shots，共 200,000 shots。此入口內部使用 reps=shots、rounds=1，不再乘一組平均次數。 |

two-qubit 表單 Open／Defaults 的初值為 reps=1000、rounds=10，每座標、每準備或 projection 分支取得 10,000 個 joint shots。這個數量不代表擬合或校準已有足夠統計。RB 的 `rounds` 交給取數端，reducer 加總已完成 rounds。

下表的 P 是掃描點數，已包含準備態、spectator 與 projection 軸。DC sweep 關閉時 DC 點數取 1。一次完整 Run 的總 shots 都是 `P × reps × rounds`。

| Two-qubit 入口 | P |
| --- | --- |
| Joint readout | `2 prepared_qubit1 × 2 prepared_qubit2 = 4`。 |
| Idle ZZ | `DC 點數 × 2 spectator states × delay 點數`。 |
| Conditional spectroscopy | `DC 點數 × 2 prepared_qubit1 × 2 prepared_qubit2 × frequency 點數`。 |
| Constructive phase／chevron | `phase 或 frequency 點數 × duration 點數`。Chevron 初開為 25×101 點，端點仍須人工提供。 |
| CZ amplitude | `scale 點數`。 |
| Conditional phase | `frequency 點數 × gate-count 個數 × 2 spectator states × 2 projections`。 |
| Local Z | `gate-count 個數 × 2 targets × 2 projections`。 |
| Paired RB／IRB | `depth 點數 × seed 個數 × 2 arms`。每點保存已完成 rounds 的計數總和。 |

取數使用含 radius 的 circle classifier，把 Qubit1、Qubit2 分成 g／e／other。缺 radius 或使用 linear classifier 時，Run 前會要求先做 GE Post。Result 的 `sums` 保存 float64 未正規化總和，未量點為 NaN。Other 留在機率與 `⟨σ⟩=(N_e−N_g)/N_total` 的分母。不保存逐 shot IQ，舊逐 shot 格式不轉換成新 Result。

| 版本 | 每 shot 的 mux 讀出 | 每掃描點的類別數 | `sums` 陣列大小，不含座標與 cfg |
| --- | --- | --- | --- |
| 一般入口 | 1 次 terminal。若選 `reset/mux_herald`，再加 1 次 herald。 | `3×3=9` | `P × 9 × 8 = 72P bytes`。|
| Heralded | 1 次 herald＋1 次 terminal | `3×3×3×3=81` | `P × 81 × 8 = 648P bytes`。|

Chevron 的 25×101 點保存 181,800 bytes 的 `sums`。選 mux herald 不改欄位大小。以 reps=1000、rounds=10 計，不選 herald 發射 25,250,000 次 mux 讀出。選 herald 發射 50,500,000 次。完整 mux 的其他 tones 照設定讀出；Result 只保存 Qubit1/Qubit2 的終端聯合計數。

估算硬體序列時間時，逐座標加總 `shots × 該點每 shot 序列時間`。序列時間包含可選 reset、herald、preparation、drive、terminal readout 與等待。使用 herald 多一次讀出及其 post_delay，不能直接當成牆鐘時間加倍。編譯、資料傳輸、round 交接、DC ramp／settle、繪圖與保存另計。這裡沒有實測牆鐘效能，不由類別數或保存量推論加速比。

要縮短操作診斷時，明確減少點數或採樣數並保存該次 cfg，標記為「診斷、非校準」。Open／Defaults 會恢復 reps=1000、rounds=10。低採樣圖只能用來排查接線、流程與掃描窗口，接受校準前仍須取得可判讀的資料與 uncertainty。

## Herald reset 與獨立 heralded 入口

只有 RB 與 conditional phase 另有 `_heralded` 入口。兩者保存 81 類，分析同時使用 herald g,g 與 e,e 的終端計數。本次移除的七個獨立入口改在對應的一般入口用 reset 選擇 herald。入口清單見 [v2 README](../README.md#two-qubit-實驗清單)。

在支援的入口選 `reset/mux_herald`，用與終端相同的完整 mux readout。Qubit1、Qubit2 在 tone 位置 0、1 時，用 `keep: {'0': g, '1': g}`。Herald 在序列前讀出一次。Result 只保留通過篩選的終端計數。

使用 herald 前，先設定 readout 的 `post_delay`，讓腔內光子在後續 qubit drive 前衰減。Herald 讀出和終端讀出共用這個值（[ADR-0074](../../../docs/adr/0074-flat-readout-module.md#herald-reset-與逐-shot-遮罩)）。等待從 readout generator 的波形結束算起，不等 ADC 積分窗結束（[ADR-0083](../../../docs/adr/0083-program-macros-and-timelines.md#timeline-只存結尾與-reference-分開)）。`mux/onetone` 依線寬給可編輯的初值，但 FIT 不校準這段等待。

用 `mux/onetone` 的 FIT 取得各 resonator 的 `kappa`，也就是以 MHz 記的半高全寬。從 `5 / (2π · κ)` µs 開始，這是五個光子衰減時間，也是表單估算 ring-up 等待的公式。共用一個 `post_delay` 時，用最小的 κ，讓線寬最窄的腔有時間放空。

等待太短時，殘留的腔光子會讓 qubit 轉移頻率偏移並造成退相干。Herald 後的名目 Xπ 可能幾乎不翻轉，conditional phase 掃描也可能找不到 π 交點。

在 FTF 的低 SNR 例子中，R2 的線寬約 1.2 MHz。五個光子衰減時間約 0.66 µs；2 µs 約是單個光子衰減時間的 15 倍。gg herald 後打名目 Qubit2 Xπ，等待 0 µs 時，Qubit2 的終端 e 比例為 11.6%；等待 2 µs 時為 92.9%。比例未扣除 I 基線。2 µs 只是當時的比較值，不是最短或最佳等待。

在 `two_qubit/joint_readout` 的 Reset 欄位選 `reset/mux_herald`。用 Qubit1、Qubit2 在完整 readout 中的 tone 位置作為 `keep` 的鍵，兩個值都填 g。Herald 和 terminal 使用同一份完整 mux readout。Run 後看 FIT 的 4×9 列正規化機率矩陣。取目標 qubit 打名目 Xπ、另一顆打 I 的那一列，把目標 qubit 的 terminal 分類為 e 的欄位加總。也可與兩顆都打 I 的 gg 列比較。這是通過 gg herald 的 shots 中，目標 qubit 的 terminal e 比例，不是 gate fidelity。只延長 `post_delay`，其餘設定不變，再 Run 和 FIT。若比例仍隨等待延長而上升，先調整等待再跑 heralded 掃描。

`reset` 預設 None，不發射 reset。脈衝 reset 在每發 shot 開始時發射。按 Defaults 只會重新擷取表單初值。

用 mux herald 時，分析將每格計數除以該格總和。接受率是該格總和除以 reps × rounds。這是分類後保留的比例，不是基態純度。

兩個 `_heralded` 入口的 **pass fraction** 是 herald g,g 計數除以全部已取得 shots。它不含 e,e 的接受數。先核對各組 shot 數與 uncertainty，再判讀選取後的差異。

## 分別準備 Qubit1 與 Qubit2，實驗 0a

既有單 qubit 表單使用 `qubit` 與 `resonator` aliases。先把 `qubit` 綁到 Qubit1，`resonator` 綁到它的 readout resonator，完成下列步驟。DC 掃描在 Open／Defaults 從 `$qubit.flux_dev` 取得可覆寫的裝置名稱。再改綁 Qubit2，重新 Open／Defaults，重做同一套步驟。每次 preview 都核對實際 component，避免把 Qubit1 的結果套到 Qubit2。

| 次序與既有指南 | 要觀察的圖與判讀 | 可接受的校準與 preview |
| --- | --- | --- |
| 1. [Lookback](../lookback/guide.md) | 看 decimated ADC trace 的 baseline、訊號到達與完整 pulse。窗口要涵蓋訊號，並區分 baseline ripple 與到達點。 | 檢查 arrival time，接受後寫到捕捉的 `resonator.time_of_fly`。 |
| 2. [Resonator spectroscopy](../onetone/freq/guide.md) | 用 linear sampling 看 resonance dip、complex response 與 fit，確認掃描跨過共振且 linewidth 可辨識。 | Preview `resonator.freq`、`kappa`、`ext.theta0`。有 pulse-readout source 時可提 `resonator.readout.readout_rf`。Electrical delay 寫回需可辨識的 branch 或可信 prior。 |
| 3. [Resonator/DC map](../onetone/flux_dep/guide.md) | 看 DC source value × frequency 熱圖。從實測 map 拖線標 half-flux 與 integer-flux，按 Done。訊號太小時先縮頻窗、檢查 SNR。 | 分項 preview `qubit.flux.period.<device>` 與預設勾選的 `qubit.flux.offset`。Period 大小為兩條擬合線距離的兩倍，保留原方向。Offset 扣除其他儀器的 Run snapshot 貢獻，note 列值與扣除項。Picker 的初始線只是初值。確認 `flux_dev` 與 `flux_unit`，使用 source 的原單位。 |
| 4. 選定 DC 工作點後重跑 [resonator spectroscopy](../onetone/freq/guide.md) | 首次 bring-up 通常移到已辨識的 integer-flux 點，再看此處 resonance。選其他工作點時也須在該設定重校 readout。 | 人工設定並記錄 DC source values，再審閱新的 readout 頻率、linewidth 與 module proposal。Apply flux 位置校準與移動 source 是兩個獨立操作。 |
| 5. [Qubit spectroscopy](../twotone/freq/guide.md) | 在確認的 passband 寬掃找 qubit feature，再窄掃看 Lorentzian／sinc fit。確認峰可辨識且屬於目標躍遷。 | Preview `qubit.freq`、`qubit.kappa`。x180/x90 pulse 另由 Rabi 校準。 |
| 6. [Amplitude Rabi](../twotone/rabi/amp_rabi/guide.md) 或 [length Rabi](../twotone/rabi/len_rabi/guide.md) | 在已量得的 qubit 頻率固定 length 掃 gain，或固定 gain 掃 length。圖中需看得到振盪週期；只見斜率時先調範圍。 | Preview 來源 pulse 的 π／π/2 結果。Pulse 寫到 `qubit.pulses.pi_amp`／`pi2_amp`，`qubit.module.x180`／`x90` 是獨立 slot rows。核對 frequency、generator、waveform 與 gain／length，再勾選。 |
| 7. [GE single-shot](../singleshot/ge/guide.md) | 用校準 π pulse 比較 probe off／on 的 IQ scatter、histogram 與 fit。確認兩中心有限、可區分且分支標記正確。Initial State 標記主要 population。 | Preview readout module 的 circle classifier。FTF 實際 mux 路徑另做 [mux GE](../mux/singleshot_ge/guide.md)，每個 tone 保存自己的 g/e centers 與 GE Post radius；不寫 point 校準。後續 FTF 使用 circle 的 g/e/other 三類，radius 決定 other 區域。 |
| 8. [T1](../twotone/time_domain/t1/guide.md) | 看 π pulse 後的 delay decay、fit 與誤差，時間窗涵蓋衰減至接近平坦的區域。 | Preview `qubit.t1` 與誤差。更新後，在後續表單按 Defaults，重算依 T1 建立的等待初值。 |
| 9. [T2 Ramsey](../twotone/time_domain/t2ramsey/guide.md) 與 [T2 echo](../twotone/time_domain/t2echo/guide.md) | Ramsey 看 fringe 與衰減，echo 看重聚焦後的總 delay decay。缺 T2 初值時人工填時間窗。 | Preview `qubit.t2r`／`t2e`。Ramsey 需來源 cfg、實際 detune 與 fringe fit 齊全才另提 frequency；echo 只提 T2。Pulse module 維持原值。 |

Qubit1/Qubit2 drive 分別填入各自的 charge-line pulse，包含 frequency、signed gain、NQZ 與 envelope duration。Frequency 單位為 MHz。Signed gain 單位為 a.u.，範圍為 [-1, 1]。Envelope duration 單位為 us。先做 Qubit1 與 Qubit2 的 frequency 與 amplitude／length Rabi，取得各自的 pulse 參考。高能階 drive 範圍及 CZ pulse slot 仍由操作者確認。

Rabi 的 module 與 slot rows 可分開 Apply。確認 x180/x90 指向這次接受的 keys；只更新 module 時，slot 可能仍指向舊 pulse。Ramsey 更新 qubit frequency 後，也須核對後續實際 pulse frequency，scalar 與 library pulse 是獨立設定。

記錄 GE 的 readout route、gain、frequency、積分窗口與其他讀出條件。Joint 分類只取 saved mux readout cfg 中角色 tone 的 circle centers 與 radius，不從單 qubit point 校準搬值。改讀出條件後要重跑 mux GE 及 GE Post。未啟用 reset 的一般入口從熱 ensemble 開始。使用 mux herald 時先篩選 shot，再累加終端計數。單 qubit Rabi／T1／T2 使用上述既有實驗入口。

進 CZ 前，依論文 §4.3.2 逐顆精校單 qubit gate：

1. 在候選 bias 附近取多個 DC 點量 [T2 Ramsey](../twotone/time_domain/t2ramsey/guide.md)，把 fringe 頻率對 bias 擬合拋物線，頂點就是 sweet spot。兩顆都在 sweet spot 後再往下做。
2. 在 sweet spot 重做 [mux GE](../mux/singleshot_ge/guide.md)，更新 centers 與 radius。
3. 用 cosine 包絡的 x180，以 [zig-zag](../twotone/zigzag/guide.md) 的奇數 π 脈衝串精定振幅。
4. 用 [single-qubit RB](../twotone/rb/guide.md) 逐顆驗收 gate。論文另做 DRAG 與兩顆同時 RB；本 repo 目前沒有這兩個實驗。

## 綁定位置與讀出 tones

| Point alias | FTF 範本綁定 | 用途 |
| --- | --- | --- |
| `qubit1` | `Q1` | 第一個位置。依實驗從 `$qubit1.module.x180`／`x90` 取準備或 projection 脈衝。 |
| `qubit2` | `Q2` | 第二個位置。依實驗從 `$qubit2.module.x180`／`x90` 取脈衝。 |
| `mux` | `MG12` | 從 `$mux.qubits` 定位兩顆 qubit，從 `$mux.module.readout` 取完整 mux readout。 |
| `coupler` | `coupler` | 存放這一對的 CZ 脈衝 slots 與 local-Z 校準。 |

`joint_readout`、`idle_zz`、`conditional_spectroscopy` 不要求綁 `$coupler`。其餘九個入口要求綁定：`chevron`、`constructive_phase`、`cz_amplitude`、`conditional_phase`、`conditional_phase_heralded`、`local_z`、`rb`、`rb_heralded`、`cz_optimize`。

Open／Defaults 檢查 `$mux` 是否綁到 `group/mux`，以及 group 所列的 qubit、resonator 是否存在且種類正確。每個 `qubits[i].resonator` 必須等於 `resonators[i]`。兩個位置須綁到不同元件，且都在 `$mux.qubits` 中。這些檢查不限 qubit 的物理子種類，也不檢查 coupler 的種類。

若已綁 `$mux.module.readout`，它必須是 mux readout，且 tone 數須等於完整 group 的大小。缺 slot 時，按實驗表單提示選擇或填寫 readout。無論引用 library 還是 inline 編輯，表單提交時都核對實際 tone 數；數量不符會在 `readout` 回報預期與實際數量並阻止 Run。等長不代表物理順序正確。請自行核對 readout 的順序，表單不以 label、頻率、ADC number 或 IQ cluster 推定對應。

Refresh 保留原稿，不重算初值。它重建表單，用新的 capture 判定 tone 數並更新隱藏的索引。若需要按新來源重填初值，使用 Defaults。

例如 `qubit1=QA`、`qubit2=QB`，而 `$mux.qubits=[QB, QA]`，則 Qubit1 用 `tones[1]`、Qubit2 用 `tones[0]`。表單隱藏 `qubit1_tone` 與 `qubit2_tone`，可在保存的 run cfg 查到。

Notebook 的內建角色也叫 `qubit1`、`qubit2`，但與 GUI 的 point aliases 分開，不會自動互綁。直接建立 Notebook cfg 時，按完整 readout 的順序填兩個從 0 起算的 tone index。

### 重綁舊 point

程式不會遷移舊 point 或翻譯舊名。先確認要更新的 point、兩個元件的順序與對應的 CZ 校準，再依序更新：

1. 在 Aliases 用 `qubit1`、`qubit2` 取代舊 `f1`、`f2` 綁定，保留各自原指向。保留適用的 `mux`、`coupler` 綁定。
2. 將 coupler 的 CZ pulse library keys `cz_f1`、`cz_f2` 改為 `cz_qubit1`、`cz_qubit2`，保留原脈衝內容。
3. 將 coupler 的 CZ module slots `cz_f1`、`cz_f2` 改為 `cz_qubit1`、`cz_qubit2`，讓兩個 slot 各指向相應的新 library key。
4. 若已有 `$coupler.ext.cz_phase`，把 `f1`、`f2` 改為 `qubit1`、`qubit2`，把 `correction_f1_deg`、`correction_f2_deg` 改為 `correction_qubit1_deg`、`correction_qubit2_deg`。保留元件名、兩角度與配對順序。

重新 Open／Defaults，核對兩路脈衝、generator、完整 mux readout 與 DC source。RB、heralded RB 與 CZ optimize 也會核對 phase pair 的元件名與順序。不符時不要交換 label 來使用舊校準，應核對綁定或重測對應的 local Z。

## Joint readout 到 local Z 的校準路線

以下以 0a 的 pulses、readout 與已記錄 bias 為起點，表格連到一般入口的 guide。`B(ab)` 表示對 Qubit1/Qubit2 分別施加 I 或 Xπ，a 對應 Qubit1，b 對應 Qubit2。未啟用 reset 時，`B(11)` 仍是熱 ensemble；選 mux herald 並保留 g,g 時，從通過該篩選的 shots 解讀準備。

| 次序 | 操作與圖 | 判讀與後續動作 |
| --- | --- | --- |
| 0b. [Joint readout](joint_readout/guide.md) | 一般入口 `two_qubit/joint_readout` 測四個 I／Xπ 準備；Run 看終端分類長條圖和保留 shots。Reset 可選 `reset/mux_herald`，此時只計通過篩選的 shots。 | Analyze 保留長條圖與 shot SEM，另列單一 4×9 計數／機率矩陣。每列是一種名目準備，每欄是 Qubit1／Qubit2 的 g/e/other 聯合結果。沒有獨立 heralded 入口或分開的 g,g、e,e herald 矩陣。 |
| 1. [Idle ZZ](idle_zz/guide.md) | 選 Ramsey target，spectator 做 I／Xπ。填 programmed Ramsey detuning，掃自由演化延遲；需要時再掃 DC bias。 | Analyze 疊上擬合並列兩分支的頻率與殘差診斷。ζ 是兩個名目分支的 signed 頻差；符號無法辨識時不報 ζ，熱混合可能使單頻模型偏離資料。 |
| 2. [Conditional higher-level spectroscopy](conditional_spectroscopy/guide.md) | 四個名目準備各掃雙線 probe 的共同 RF，可掃 DC bias 或加雙 Xπ 對比。 | 從同一 qubit 的四個 P(e) 頻譜比較峰谷位置，再選窄掃窗口。能級、EC/EJ/J 與可用的 CZ frequency 需另行確認。 |
| 3a. [Constructive phase](constructive_phase/guide.md) | B11 固定 signed gain 比例，掃 relative phase × duration。 | 無衰減 cosine 求各 trace rate，rate < 1/span 為未解析，圖會標在可辨識 rates 之外。以 `rate² = A + B cosφ + C sinφ` 求最大 rate 的 `atan2(C,B)`，只提 Qubit2 pulse 與獨立 slot。Nyquist 與 signed gain ratio 由操作員確認，此掃描只調 relative phase。 |
| 3b. [Rabi chevron](chevron/guide.md) | 固定 relative phase 與兩線 gain，掃共同 CZ 載波 × 共同波形 envelope 時長（µs）。 | 看 P(g,g) 谷值人工選時長與頻率窗口。用 duration slices 的量測點、shot SEM 和密網格 fit 核對。兩線同頻、同 duration、同時間原點，waveforms 由 cfg 明示，cosine 包絡只是起點。 |
| 4. [CZ amplitude](cz_amplitude/guide.md) | B11 後播雙線 CZ、雙 Xπ 對比與 mux readout。掃共同 gain scale，固定原 signed gain 比例、relative phase、frequency 與 waveform。 | 零相位 cosine 擬合終端 P(g,g) 的第一正回返。實測從 0 覆蓋完整一週期才提案；兩支來源 pulse 的 signed gain 各乘以該倍率。逐列 preview 兩線 pulse 與 slots。回返是 thermal leakage proxy，須看 fit_quality 與大振幅 AC Stark 偏離。 |
| 5. [Conditional phase／frequency](conditional_phase/guide.md) | Spectator I／Xπ，target π/2，n 次 CZ，獨立 X/Y projection。掃共同 frequency 與 n，預設 `[1,2,3]`。 | 以計數加權的 X/Y 模型求每 gate slope，再由兩 spectator 的 slope 差找唯一 π 交點，線性內插提 frequency。Gate-count 差值 gcd 必須為 1，`[1,3,5]` 有 slope alias，不能使用。缺失、無交點、多交點或退化均保留原因並停止提案。核對 wrapping、contrast 和 spectator 分支。只有一個 π 交點才將兩線頻率改成內插值，再逐列接受 pulse／slots。 |
| 6. [Local Z](local_z/guide.md) | 從 B00 分別量 Qubit1/Qubit2 的 π/2、n 次 CZ、獨立 X/Y projection，另一 qubit 維持原熱態。至少三個連續非負 n，整列 phase 完整才相鄰 unwrap 與含 intercept 的計數加權最小平方。 | Correction=-slope，保持原值。兩分量可辨識才提一筆具 component 名稱的 pair。採樣假設為相鄰真實增量小於 180°，每 gate 額外 360° 無法辨認。先看分類 X/Y 的 shot SEM，再看 slope 的 fit SE 與品質；後者不是逐點 shot SEM。缺 correction 回報缺失。 |

Joint matrix 表示「名目準備→觀測結果，含熱 population」，包含準備與讀出效果。Matrix 的用途限於診斷原始熱 population 與讀出結果；後續分析保留原 population。Matrix inverse、對角值門檻與 gate fidelity 推論都超出此校準的適用範圍。Circle 分類把半徑外資料列為 other，不能直接把 other 當成 leakage。缺失、非有限或重合的 GE 中心與缺 radius 均回報無效。

### Amplitude 與 frequency 人工交替三輪

完成 chevron 選窗後，建議以同一組雙線 cfg 人工做三輪 `4 → 5`。

1. 做 amplitude scan，檢查回返 fit，preview 後接受有效兩線 pulse 更新。
2. 在下一個表單按 Defaults，確認使用剛接受的 amplitude，再量 conditional phase／frequency。檢查交點與 contrast，preview 後接受有效 frequency 更新。
3. 重做至共三輪，每輪保留 cfg、圖與選擇理由，依實測結果縮窗。

無交點、contrast 未解析或 fit 失敗時，保存圖與原因，先檢查範圍、讀出與 pulse，再決定重測。三輪是人工操作建議，conditional phase 是否達 π 仍須由資料確認。

### Coupler 保存與 local-Z 重測

每個 coupler 保存一對 Qubit1/Qubit2 的 CZ 校準。

| 目的地 | 內容 |
| --- | --- |
| `$coupler.pulses.cz_qubit1`、`$coupler.pulses.cz_qubit2` | 兩個 pulse，含本次實測 generator、frequency、gain、phase 與 waveform。 |
| `$coupler.module.cz_qubit1`、`$coupler.module.cz_qubit2` | 獨立 slot rows，指向同次 Open／Defaults 解析的上述 library keys。 |
| `$coupler.ext.cz_phase` | 一筆 parameter。`qubit1`、`qubit2` 保存實際元件名；`correction_qubit1_deg`、`correction_qubit2_deg` 保存同次 local-Z 量測得到的角度。 |

Amplitude／frequency proposal 只換擬合的共同 scale 或 frequency，保留來源 cfg 的 gain 比例、relative phase 與 waveform。所有提案都必須能追溯到本次量測的 pulse。

每 CZ 的 logical Bloch phase accrual 為 θ1、θ2 時，local correction 是 −θ1、−θ2，單位 degree。後續序列以 virtual Z 更新 single-qubit logical frame，兩線 constructive relative phase 維持原值。Pulse phase 的正負換算須依既有 frame convention 核對。

先選定 amplitude／frequency，再量 local Z。之後改任一線的 amplitude、frequency、duration、waveform、relative phase 或 DC 工作點，都要重新檢查 CZ 並重測 local Z。交換 Qubit1/Qubit2 後也須先核對保存的 component 名稱，再重做對應校準。校準時效由操作員確認。

Pulse、slot 與 phase rows 都可獨立 Apply。部分套用後，檢查下一次 Run 的完整 cfg，確認 pulse 與 phase correction 屬於同一組已接受校準。

## Paired RB／IRB 與資料解讀

[Two-qubit RB／IRB](rb/guide.md) 使用配對的 two-qubit Clifford reference／CZ-interleaved sequences、各自完整 inverse、實際雙線 CZ 與 local frames。它與[單 qubit RB／IRB](../twotone/rb/guide.md) 分別量不同的控制序列。

Result 依 (seed, arm, depth) 排列，再接 Qubit1／Qubit2 分類軸。Heralded 在終端分類軸前多兩個 herald 分類軸。Stop 保留已完成 rounds 的計數，未量格子是 NaN。舊檔有 round 軸，Load 因軸數不合而拒絕。

Plain RB 從終端 joint g,g 計數算 survival。Heralded 另以終端回到各自 herald g,g／e,e 的比例作 survival，依各組完整 seeds 聯合擬合。使用 d=4 的 error estimate，保留負值及 paired-seed bootstrap interval。熱 population、準備／讀出效果與模型假設均須納入判讀，物理 gate 表現需由實際量測確認。

RB 的 liveplot 與第一張 Analyze 圖用保留 shots 的計數畫機率。帶端帽的棒是 shot SEM；淡色的棒是 seed 間 SD。擬合圖用完整 seeds 的等權平均，棒仍是 seed SD，1/e depth 標註的 ± 才是 fit 標準誤。若掃描窗無法解析衰減，先依 [RB 判讀](rb/guide.md#interpreting-the-result)檢查深度與不確定度，不用 ratio 建議 CZ 調參。

取得第一個 IRB 結果後，依論文 §5.3.3 掃 CZ 的 cosine 寬度。長閘的誤差由被驅動非計算態的 T1 主導，短閘由同調 leakage 主導，最佳寬度在兩者之間（論文裝置為 85 ns）。每個寬度重做 4 → 5 → 6 與 IRB。最短寬度受驅動強度限制，先由實測 Rabi rate 換算 gain 1 時的 2π 週期；非 const 波形的最大振幅只有 const 的一半。

要調整已校準 CZ 的 I/Q 形狀，依[每輪優化與驗收](cz_optimize/guide.md#每輪優化與驗收)操作。優化、寫回，再用 plain RB 的新 seeds 做 paired IRB。驗收的 gate error 仍有改善才按 Defaults，從寫回值開始下一輪。

Two-qubit 實驗保存未正規化 sums、準備態／spectator／sweep 座標與本次 cfg，cfg 保留完整 tone 順序。取消時保存已取得計數，未量點為 NaN。Phase 依各實驗的完整性規則分析 X/Y；RB／IRB 只用完整配對。診斷型 joint readout 與 conditional spectroscopy 可顯示已完成 rounds 的分布及每點 shot 數。寄生抵消需另行研究，各校準由操作員逐項執行。
