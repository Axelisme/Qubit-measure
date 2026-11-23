# Two-qubit bring-up 與 CZ 校準操作指南

**讀者：** operator、agent。準備兩個 qubit 位置並依序校準 microwave-activated CZ。

兩條 qubit charge line 同頻、相鎖地驅動 CZ；coupler 只設 DC bias。把第一回返、conditional π phase、local Z 和 paired IRB 分開驗收。任何一項都有自己的可觀察量；高回返不能代替相位或 gate error。

## 綁定位置與讀出 tones

在同一個工作點校準兩顆 qubit。切換單 qubit 表單時，將 `qubit`、`resonator` 綁到當次的元件：先做 [Lookback](../lookback/guide.md)、[resonator 頻譜](../onetone/freq/guide.md)與必要的 [DC map](../onetone/flux_dep/guide.md)，再校準 [qubit frequency](../twotone/freq/guide.md)、[Xπ 與 Xπ/2](../twotone/rabi/amp_rabi/guide.md)及弛豫等待。改用雙 qubit 同時工作點後，重新確認兩顆的頻率、脈衝與讀出。scalar 頻率、library pulse 和 module slot 是獨立設定，核對實際脈衝來源。

- `qubit1` 和 `qubit2` 是位置，須綁不同元件。兩者須出現在同一 `mux` 的有序 qubit 清單；相同位置的 resonator 和 readout tone 須屬於同一 qubit。若 qubit 列表是 Qubit2、Qubit1，兩位置的 tone index 就是 1、0。表單核對完整 tone 數，不會從 ADC 編號或頻率猜順序。Notebook 用同名位置，但不讀取 GUI aliases。
- **Mux readout** 引用 `$mux.module.readout`，缺 slot 時補全空白表單。保留完整 mux（包括其他 tones）。兩個位置都要有相同讀出設定下的 [mux GE 與 GE Post](../mux/singleshot_ge/guide.md) 圓形分類器及半徑；缺失或使用 linear classifier 會擋住 Run。更改讀出頻率、gain 或積分窗後重校。兩條 tone 必須來自同一 shot；不以各自的平均 IQ 代替聯合計數。
- 除 joint readout、idle ZZ 及兩個 conditional spectroscopy 入口外，其餘入口須綁 `coupler`，作為雙線 CZ 的 pulse、slot 或 local-Z 校準來源。核對兩線各自的 generator 與 bias 設定，而不只核對表單是否有效。

### 重綁舊 point

沒有自動遷移。把 aliases 的 `f1`／`f2` 改為 `qubit1`／`qubit2`，保留原元件與順序；把 coupler 的 library keys 和 slots `cz_f1`／`cz_f2` 改為 `cz_qubit1`／`cz_qubit2`，讓各 slot 指向相應新 key。若已有 `$coupler.ext.cz_phase`，同樣更新兩個元件鍵，將 `correction_f1_deg`／`correction_f2_deg` 改為 `correction_qubit1_deg`／`correction_qubit2_deg`。不交換兩支 pulse 或兩個角度來規避配對檢查；重讀來源後核對脈衝與元件順序。

## Herald reset 與獨立 heralded 入口

一般入口的 **Reset** 留空則無 reset。在支援 mux herald 的入口，選 `reset/mux_herald` 並用與終端相同的完整讀出；`keep` 的鍵是完整 tone 的零起算位置（不是 ADC），例如所選位置為 0、1 才填 `{'0': g, '1': g}`。一般入口只計通過篩選的終端分布。獨立 heralded 入口不再加一層 herald reset。RB 與 conditional phase 按 herald 分出 gg、ee 兩組做聯合分析。[Heralded conditional spectroscopy](conditional_spectroscopy/guide_heralded.md) 在名目準備後 herald，按實測類別分成 gg、ge、eg、ee 四支頻譜。**Pass fraction** 是 herald gg 數除以所有取得的 shots，不是純度。

從 `post_delay = 5/(2π·κ)` µs 開始。κ 是 [mux Onetone](../mux/onetone/guide.md) 量得的 linewidth（FWHM，MHz），共用等待取讀出中最小的 κ。等待從讀出波形結尾起算，不等 ADC 積分窗；herald 與終端共用此值。先用 [Joint readout](joint_readout/guide.md) 選 gg herald，對每顆取「該顆名目 Xπ、另一顆 I」的列，加總該顆終端為 e 的機率。只延長等待再比較。若較長等待仍提高 e 比例，先調等待再做 heralded 掃描；比例不是 gate fidelity。

## 採樣數與成本

每座標、每分支的嘗試 shots 為 **Reps × Rounds**；總數再乘掃描點、準備／投影分支及適用的 seed、arm 數。Mux tones 不是獨立 shots。可選 herald 多一次讀出與等待；不能據此推論牆鐘時間加倍。分析用保留 shots；確認完成輪數，不能把圖上的點數當成所有 rounds 均完成。

## Joint readout 到 local Z 的校準路線

| 順序 | 用結果做什麼 |
| --- | --- |
| [Joint readout](joint_readout/guide.md) → [Idle ZZ](idle_zz/guide.md) | 先檢查四個名目準備的聯合計數，再比較 spectator I／Xπ 的 Ramsey 頻差。 |
| [Conditional spectroscopy](conditional_spectroscopy/guide.md) 或 [heralded 入口](conditional_spectroscopy/guide_heralded.md) → [Constructive phase](constructive_phase/guide.md) → [Chevron](chevron/guide.md) | 先按名目準備或實測 herald 找候選 RF，再定兩條線的相對相位，最後用回返谷選頻率與時長窗口。 |
| [CZ amplitude](cz_amplitude/guide.md) ↔ [Conditional phase](conditional_phase/guide.md) | 固定波形、時長與帶號 gain 比例，先量完整第一回返，再找唯一 π 交點。建議人工交替三輪；輪數不代替結果判準。 |
| [Local Z](local_z/guide.md) → [Paired RB／IRB](rb/guide.md) | 分別測兩顆的每 gate phase，以其負值修正邏輯 frame，再用同 seed 成對序列比較衰減。 |

### Coupler 保存與 local Z 重測

CZ 結果可提案更新 `$coupler.pulses.cz_qubit1`、`cz_qubit2` 及對應的獨立 module slot；local Z 寫入一筆 `$coupler.ext.cz_phase`，含有序元件名和兩個角度。只採用部分 rows 時，要核對兩線 pulse、slot 與 phase 是否仍屬同一組。改 amplitude、frequency、duration、waveform、relative phase、bias 或交換位置後，重測 local Z；不沿用不合配對的舊值。

## Open、Defaults 與人工 Apply

共用操作見 [`zcu_lab/v2/operator-workflow.md`](../operator-workflow.md)。Analyze 用來源 Run／Load 的 cfg 與計數，不用目前表單替代；沒有來源 capture 不能寫回。Load 不改現有表單。

軸名、tag 或 layout 不合的舊檔（含舊獨立 heralded 入口、早期 IQ 格式、RB 的 round 軸）不自動轉換，保留原檔並用寫出版本開啟。結果相容但 cfg major 不合時保留結果和 metadata，不還原 cfg。需 cfg 的分析不能據此給提案。原生結果保存計數，不保存逐 shot IQ，見 [Two-qubit 實驗清單](../README.md#two-qubit-實驗清單)。
