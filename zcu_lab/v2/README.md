# v2 experiments

**讀者：** lab-author、agent。查找實驗的檔案分工、作者契約與各家族入口。

**Last updated:** 2026-11-23，獨立 heralded 入口的取捨

先從 [tutorial](../tutorial/README.md) 新增實驗，再按下列入口查契約與操作。
實驗定義改動的驗收方式與命令範例見 [離線驗收](../README.md#實驗定義的離線驗收)。

## Reading routes

- 寫或改實驗：讀 [實驗寫法指南](experiment-style.md)，從其中的範本開始寫。
- 設計 liveplot 與 Analyze 圖、檢查操作員能否判讀：讀 [實驗呈現指南](figure-style.md)。圖顯示後的 artist 更新與 refresh 契約見 [plotting〈原生 matplotlib 更新〉](../../lib/zcu_tools/plotting/README.md#原生-matplotlib-更新)。
- [Two-qubit bring-up 與 CZ 校準](two_qubit/guide.md) 說明 Qubit1、Qubit2 的單 qubit 校準、位置綁定、舊 point 重綁與資料保存。FTF 控制鏈可執行仍不代表物理校準或 fidelity 已驗收。
  - 一般入口用 reset 選擇 herald。獨立 heralded RB 與 conditional phase 聯合分析 g,g／e,e；heralded conditional spectroscopy 改在名目準備後分類，分出四個實測分支。見[完整清單](#two-qubit-實驗清單)。
- [K-cycle 讀出鏈 amp Rabi](chain/amp_rabi/guide.md)、[len Rabi](chain/len_rabi/guide.md)、[T1](chain/t1/guide.md)、[T2 Ramsey](chain/t2ramsey/guide.md) 與 [T2 echo](chain/t2echo/guide.md) 說明每個 rep 是一個 cycle（K 後 (D, 讀, 讀) × 8）、兩條 live 曲線、聯合擬合與 joint residual 的判讀方式。數值契約見 [analysis〈Readout chain K-cycle〉](../../lib/zcu_tools/analysis/README.md#readout-chain-k-cycle)。
- [`twotone/time_domain/t1_cov`](twotone/time_domain/t1_cov/guide.md) 用兩次單 tone 讀出的二階矩自校準，擬合狀態改變機率以估計 T1。可選脈衝 reset，不需 classifier。
- [Experiment authoring](experiment-authoring.md) 說明核心 records 與資料形狀，新實驗從[寫法慣例](experiment-authoring.md#寫法慣例)選 cfg、分析與圖 API 的寫法。
- [Measure authoring](measure-authoring.md) 說明 cfg defaults、operator guides、writeback 與 reload。
- [GUI contracts](gui-contracts.md) 說明 concrete adapter hooks 與 capability。
- [Measure support](support/measure/README.md) 提供共享 domain mechanics。
- [Program 的 Readout configuration](../../lib/zcu_tools/program/v2/modules/README.md#readout-configuration) 說明 Pulse／Mux Readout 的 generator、ADC、時序與 IQ 契約。
- [Framework runtime](../../lib/zcu_tools/experiment/v2/runtime/README.md) 擁有 Schedule、buffer、acquire 與 stop lifecycle。

## Two-qubit 實驗清單

Open／Defaults 使用 point aliases `qubit1`、`qubit2` 和 `mux`。表單不限制 qubit 的物理子種類或 coupler 的種類，仍要求 `group/mux` 並檢查有序的 qubit／resonator 綁定。兩個位置須綁不同元件，且都在 `$mux.qubits` 中。`joint_readout`、`idle_zz` 和兩個 conditional spectroscopy 入口不要求 `$coupler`。其餘入口須綁 `$coupler`，它是這對 qubit 的 CZ 校準來源或寫回目標。

下表列出 two-qubit 的 GUI／MCP registry keys。每個連結指向該入口的 guide。另有 [`two_qubit/cz_optimize`](two_qubit/cz_optimize/guide.md)，以固定 depth 的 interleaved RB 比較原 CZ 與等長 arb 候選，搜尋形狀、共同物理 gain 及 local Z。

| 實驗 | 一般 key／guide | Heralded key／guide |
| --- | --- | --- |
| Joint readout | [`two_qubit/joint_readout`](two_qubit/joint_readout/guide.md) | — |
| Conditional phase | [`two_qubit/conditional_phase`](two_qubit/conditional_phase/guide.md) | [`two_qubit/conditional_phase_heralded`](two_qubit/conditional_phase/guide_heralded.md) |
| Chevron | [`two_qubit/chevron`](two_qubit/chevron/guide.md) | — |
| Conditional spectroscopy | [`two_qubit/conditional_spectroscopy`](two_qubit/conditional_spectroscopy/guide.md) | [`two_qubit/conditional_spectroscopy_heralded`](two_qubit/conditional_spectroscopy/guide_heralded.md) |
| Constructive phase | [`two_qubit/constructive_phase`](two_qubit/constructive_phase/guide.md) | — |
| CZ amplitude | [`two_qubit/cz_amplitude`](two_qubit/cz_amplitude/guide.md) | — |
| Idle ZZ | [`two_qubit/idle_zz`](two_qubit/idle_zz/guide.md) | — |
| Local Z | [`two_qubit/local_z`](two_qubit/local_z/guide.md) | — |
| Paired RB／IRB | [`two_qubit/rb`](two_qubit/rb/guide.md) | [`two_qubit/rb_heralded`](two_qubit/rb/guide_heralded.md) |

一般入口保存兩個位置的 9 類 terminal 聯合計數。除了 `conditional_phase`，表內一般入口可用 `reset/mux_herald` 在序列前篩選 shot，仍只保存這 9 類終端計數。三個 `_heralded` 入口保存 81 類 herald × 終端計數。RB 與 conditional phase 聯合分析 herald g,g 與 e,e；conditional spectroscopy 合併名目準備後畫 herald gg、ge、eg、ee 四支頻譜。採樣與保存量見 [two-qubit guide](two_qubit/guide.md#採樣數與成本)。

### 何時保留獨立 heralded 入口

#### gg／ee 共用參數

**規則。** 只有分析在 gg 和 ee 兩組共用一個參數，且準備誤差只進各組自己的振幅或偏移、不移動共用參數，才保留獨立的 heralded 入口。

**例子。** RB 的兩組各自擬合振幅與偏移，共用衰減參數 `p`。Conditional phase 只用 π 交點寫回；兩組在 π 交點相符，不代表離開 π 時也等效。

**邊界。** 準備誤差若會拉動共用參數，或分析本就不共用參數，就用 `reset/mux_herald` 只留 gg。例如 idle ZZ 的 ζ 會受準備污染拉動，joint readout 矩陣也直接含準備誤差。

**依據。** 2026-11-20，使用者說：「有一些实验呢，它天生不在意 GG 还是 EE，它在意的是 GE 的交换，或者是保留在 GGEE 的属性。这种情况下，利用 hurdling 去。诶，具体来说，用交叉矩以及 hurdling 这两种方案去分析，有助于提高拟合可信度。那其他实验就用 reset 模块就好了。」使用者追問 gg／ee 是否等效，核對後同意保留獨立的 RB 與 conditional phase。語音用詞、數值及選項比較見[只為共用參數保留獨立 heralded 實驗](../../docs/discussions/2026-11-20-heralded-experiment-pruning.md)。

#### 依 probe 前實測類別分開分析

**規則。** 分析要依 probe 前的實測類別分開，且 gg、ge、eg、ee 各自都是分析對象時，保留獨立 heralded 入口，不要求各分支共用擬合參數。先做四種名目 I／Xπ 準備，使各類取得相近的 shot 數，再做 herald 分類。不要用 reset 先丟掉其他類別。

**例子。** [`two_qubit/conditional_spectroscopy_heralded`](two_qubit/conditional_spectroscopy/guide_heralded.md) 依準備後的 herald 類別畫四支頻譜，不做聯合擬合。

**邊界。** 只需要篩選初態的實驗使用 `reset/mux_herald`，不另開入口。

**依據。** 2026-11-23，使用者說：「我覺得你應該安排的不是準備在gg/ge/eg/ee，而是post-selection分類gg/ge/eg/ee」、「你還是可以用pi來製備來是分類比例與熱態無關」、「你搞錯了…當然是先打Xpi然後在herald讀出」，並對更正後的獨立入口方案答「可以」。見[以準備後的 herald 分支量條件頻譜](../../docs/discussions/2026-11-23-heralded-conditional-spectroscopy.md)。

### Two-qubit 持久名稱

`qubit1`、`qubit2` 是位置，不是元件名。下表的 `$coupler` 是來源路徑：保存時以擷取到的實際 coupler 元件名取代。

| 用途 | 名稱 | 產生方 | 讀取方 |
| --- | --- | --- | --- |
| Point alias | `qubit1`、`qubit2`、`mux`、`coupler` | 操作員在 Inspect 綁定並保存到 point 的 `aliases.yaml`。[FTF kickstart](../kickstart/FTF/startup.py) 和 [FTF simulation startup](../simulation_envs/FTF/startup.py) 首次建立 point 時，各從自己的範本填入。 | [`captured_qubits`](two_qubit/_measure.py) 在 Open／Defaults 讀固定 capture。只有需要 coupler 的表單才要求該綁定。 |
| CZ module slot | `$coupler.module.cz_qubit1`、`$coupler.module.cz_qubit2` | [CZ amplitude](two_qubit/cz_amplitude/gui.py)、[conditional phase](two_qubit/conditional_phase/gui.py)（含 [heralded](two_qubit/conditional_phase/gui_heralded.py)）與 [CZ optimize](two_qubit/cz_optimize/gui.py) 的 writeback 提案；[constructive phase](two_qubit/constructive_phase/gui.py) 只提案更新 `cz_qubit2`。Apply 後 slot 保存對應的 library key。 | Open／Defaults 的表單讀 slot，例如 [chevron](two_qubit/chevron/form.py) 與 [RB](two_qubit/rb/form.py)。 |
| CZ pulse library key | `$coupler.pulses.cz_qubit1`、`$coupler.pulses.cz_qubit2` | 同一批 adapter 另用 `ModuleWriteback` 提案保存 pulse。操作見 [Two-qubit guide 的 Coupler 保存段](two_qubit/guide.md#coupler-保存與-local-z-重測)。 | [表單](../../lib/zcu_tools/gui/form_adapter/_seeds.py) 從 slot 取得 library key，再從擷取的 library 讀 pulse。 |
| CZ phase 校準 | `$coupler.ext.cz_phase.qubit1`、`$coupler.ext.cz_phase.qubit2`；`$coupler.ext.cz_phase.correction_qubit1_deg`、`$coupler.ext.cz_phase.correction_qubit2_deg` | [Local Z](two_qubit/local_z/gui.py) 與 [CZ optimize](two_qubit/cz_optimize/gui.py) 的 writeback 提案；Apply 後同一筆參數記錄有序元件名與兩個角度。 | RB（含 heralded）與 CZ optimize 表單共用 [`captured_phase_pair`](two_qubit/_cz_phase_form.py)，核對來源元件名再取角度初值。 |
| Native HDF5 變數名 | `sums`；CZ optimize 另存具名參數（如 `i_knot_1`、`q_knot_1`、`gain_scale`）、`peak` 和起點顯示資料（如 `start_iq`、`start_style`） | 各 core 的 `AXES_SPEC` 宣告 `VariableSpec.variable`，例如 [joint readout](two_qubit/joint_readout/core.py) 與 [CZ optimize](two_qubit/cz_optimize/core.py)；[`save_run`](../../lib/zcu_tools/experiment/native_persistence.py) 寫入 `/vars`。 | Native Load 的 [`restore_run`](../../lib/zcu_tools/experiment/native_persistence.py) 按同一宣告讀取，再交給實驗的 `result_builder`。 |
| CZ 起點時長 | `/docs/run_values` 的 `duration`（秒） | [CZ 的 Result spec](two_qubit/cz_optimize/_result.py) 宣告必填 `RunValueSpec`。 | Native Load 還原為 `CZOptimizeResult.start_duration_s`；缺欄或缺文件時拒讀。 |
| Native HDF5 的分類軸名 | `terminal_qubit1`、`terminal_qubit2`；三個 `_heralded` 入口另有 `herald_qubit1`、`herald_qubit2`。CZ optimize 的分類軸是 `Qubit1 class`、`Qubit2 class`。 | 各 core 的 `AXES_SPEC` 宣告 `VariableAxisSpec.label`，例如 [heralded RB](two_qubit/rb/core_heralded.py)、[heralded conditional phase](two_qubit/conditional_phase/core_heralded.py) 與 [CZ optimize](two_qubit/cz_optimize/core.py)。 | Native Load 用 [`VariableSpec.loaded_from_variable`](../../lib/zcu_tools/experiment/axes_spec.py) 核對同一宣告的軸名。 |

## 套件邊界

每個實驗有自己的資料夾。`core.py` 擁有 cfg、Result、量測與分析政策。
`gui.py` 以具名參數宣告可選的 measure-gui Adapter 值，接上 Form、run、guide 與 Analysis／Interactive／PostAnalysis。
Core 擁有 run 與 analyze 的計算，GUI 只接線、解析來源與決定 writeback。
`twotone/freq` 的分析只提供 Model type、Expected center、Min SNR，並選用局部選峰策略。
`twotone/chain` 保留預設策略。兩者的選峰與擬合窗口見 [fitting〈Peak candidates〉](../../lib/zcu_tools/analysis/fitting/README.md#peak-candidates)，責任分工見 [ADR-0072](../../docs/adr/0072-module-boundaries.md#two-tone-廣掃的選峰與窗口)。
Mux/onetone core 也擁有逐 tone 數值 outcomes 與擬合失敗隔離。
GUI 只接線、解析固定 capture 的 route prior／resonator 來源與提出 writeback。
Leaf 的 `__init__.py` 只提供 core exports，不載入 GUI。

同目錄的獨立實驗各有量測入口。TwoTone T1 的 [core.py](twotone/time_domain/t1/core.py)
只放基本 T1，[t1_with_tone.py](twotone/time_domain/t1/t1_with_tone.py) 與
[scan_t1_with_tone.py](twotone/time_domain/t1/scan_t1_with_tone.py) 各放固定 tone 與 gain sweep。
兩者共用基本 T1 的 Result、Analysis 或訊號轉換，不增加 GUI adapter。
AC Stark spectroscopy 與 Ramsey 各有自己的實驗目錄。
`twotone/ac_stark` 對應 [spectroscopy core](twotone/ac_stark/core.py)，
`twotone/ac_stark_ramsey` 對應 [Ramsey core](twotone/ac_stark_ramsey/core.py)。
註冊位置與 Ramsey guide 見 [Lab 組裝與註冊](../README.md#lab-組裝與註冊)。
Singleshot GE 與 mux GE 的混淆矩陣求解由 `zcu_tools.analysis.singleshot` 擁有。
兩個 core 在 Analyze 都啟用 [共用 GE 診斷](../../lib/zcu_tools/experiment/README.md#ge-模型診斷)。擬合成功時，Single 發布 `diagnostic`，mux 以原始 tone 序號發布 `diagnostic_i`；各 core 仍另外產生 Fit 圖。
各實驗的 `core.py` 自行選擇分析資料並繪製分類圖，數值擬合工具由 `zcu_tools.analysis.fitting` 擁有。
已分類 counts 的 saved-matrix reader 與跨實驗 row mapping 在 [support/singleshot/util.py](support/singleshot/util.py)。
Raman／STIRAP 共用已分類 counts 的[可選校正](support/singleshot/centroid3_analysis.py)，各 core 自己繪圖並回傳摘要。
Mux IQ 平均轉換在 [support/mux.py](support/mux.py)，
AC Stark／CKP 的連續共振中心擷取在 [support/resonance.py](support/resonance.py)。
兩個 GE core 各自排 Post 版面，並用 [support/singleshot/classification_plot.py](support/singleshot/classification_plot.py) 畫分類色的密度、疊分類線與中心，再將兩格準備圖的 I/Q 範圍對齊。
Two-qubit chevron／constructive phase 各自在 `run()` 寫出序列，
共用型別在 [\_rabi_scan.py](two_qubit/_rabi_scan.py)，數值擬合在
[\_rabi_analysis.py](two_qubit/_rabi_analysis.py)。兩個實驗各自在 `analyze()` 畫自己的圖，
[\_sums.py](two_qubit/_sums.py) 檢查分類校準並換算計數，不讀 alias 或元件。
Two-qubit 的 conditional phase 與 heralded conditional phase 呼叫 `fit_phase_slopes`；斜率的單位與週期代表值見 [fitting〈Phase regressions〉](../../lib/zcu_tools/analysis/fitting/README.md#phase-regressions)。
Readout-chain 的 chain_power／chain_freq／chain_length 各自在 `run()` 寫出掃描與 16-read shot，
並在自己的 core 建立及更新 live／metric 三 panel，數值分數交給 `zcu_tools.analysis.readout_chain`。
[\_chain_common.py](twotone/ro_optimize/_chain_common.py) 只保留保存映射、完成點篩選與最佳 setting 摘要型別。
[chain_length](twotone/ro_optimize/chain_length/guide.md) 與 [rpi_length](twotone/ro_optimize/rpi_length/guide.md) 在每個 ADC window 加上同一個可編輯的 `length_margin` 設定 pulse 長度。初值取擷取模組的 pulse 長度減 ADC window，沒有模組才用該 Registry 的 readout 範本。
Rπ 的 rpi_power／rpi_freq／rpi_length 各自在 `run()` 寫出掃描與 R→π 鏈，並保留自己的 cfg、Result 與分析摘要型別。
三者各自在 core 畫 score 與診斷圖；[\_rpi_common.py](twotone/ro_optimize/_rpi_common.py) 只保留共用的 `rpi_sums` 解碼、選點與格式化。

K-cycle 讀出鏈實驗各自在 `run()` 寫出 (D(x), 讀, 讀) × 8 的 `builder.repeat("cycle", BLOCKS_PER_CYCLE)` 迴圈；
各 chain core 保留可用的 first-read 起點。第一讀無估計時改用保存軸的單一起點，first-read 圖保留實測點，既有 first-read 估計欄位回傳 null。兩種情況都用完整 sums 聯合擬合 65 個特徵一次。
`fit_cycle` 的 joint 估計、`J/df` 與 `cycle_residual_projection` 的畫圖契約見 [analysis〈Readout chain K-cycle〉](../../lib/zcu_tools/analysis/README.md#readout-chain-k-cycle)。
可沿 live 方向投影時，Analyze 畫出實測點、聯合模型曲線與有正負號的逐點殘差。圖與摘要都呈現 `J`、`df`、`J/df`、χ² 期望帶的數值端點與白化殘差的 `|r| > 3` 比例。逐點序列只用於畫圖，這些數值不作為接受或拒絕擬合的門檻。
五個 core 各自在 `run()` 建立及更新 all-number 與 first-read 的 live 曲線。
[chain/\_cycle.py](chain/_cycle.py) 保留共用的 CycleSums 存檔變數與第一讀 IQ 投影。[`chain/_joint.py`](chain/_joint.py) 提供 joint residual 的欄位與圖元，各實驗決定圖面配置。模型留在 `zcu_tools.analysis.readout_chain.curve_models`。

`v2` 延續 program/v2 的版本脈絡，不表示 GUI framework 的第二版。通用 runtime、量化與 SNR 工具留在 `zcu_tools.experiment.v2`。Core 不依賴 GUI，import-linter C15 檢查 core 到 GUI framework 的直接與間接依賴。

Save／MCP 透過單一 `summarize_analysis` 轉換 native 結果，不使用已退休的實驗 serializer 或結果 base。
結果根為 dataclass、pydantic model 或字串 key 的 Mapping，summary 是 strict JSON object，key 沿用原生欄位名。
非有限數值轉為 null，Live RPC 另以 `invalid` 回報路徑。保存文件只含 pane、parameters、summary，不保存 invalid。

## 取數路徑分流慣例

實驗有不同取數路徑時，例如 FPGA sweep 與 host 逐點取數，讓 `run` 只做共同前置與分派。
每條路徑寫成一個私有 `_run_*` 方法，各自建立 buffer、`Schedule` 與程式，並在方法內寫出取數流程。
各路徑回傳同一種 Result，座標與資料沿用相同的軸順序與 shape 契約。
呼叫端與 `analyze` 只讀這份 Result，不依取數路徑分支。
程式結構不同就拆成不同路徑。
即使共用同一個硬體迴圈，只要 delay 機制或 module 組合不同，也要拆分。
只差座標數值、程式結構相同時，可以保留單一路徑。
不強制合併作者已拆開的實驗，保留時在原地說明理由。

以 [twotone len_rabi/core.py](twotone/rabi/len_rabi/core.py) 為例，`run` 先拒絕原生 arb 波形、複製 cfg，再準備量化長度座標。
`const`／`flat_top` 交給 `_run_for_flat`，在 FPGA sweep 長度。
其他可調長度的波形，例如 `cosine`，交給 `_run_for_arb`，由 host 逐長度取數並重用同長度的程式。
`_run_for_arb` 沿用歷史名稱，不表示它接受原生 arb 資產。
兩條路徑都回傳 `LenRabiResult`。`lengths` 與 `signals` 都是一維 `(N,)`，每個長度對應一個平均 complex IQ。
新 run 另回傳 `iq_moments`；舊檔缺少這個欄位仍可載入。
Host 路徑先合併各長度已完成的 shot 矩，再除以保留 shot 數求平均。未取數的點保留 NaN，不把內部的 round 軸交給呼叫端。

## 即時圖工具

`zcu_lab.v2.support` 匯出 `live_line`、`live_errorbar`、`live_scatter` 與 `live_heatmap`。
實驗的呼叫順序見[圖的建立、顯示與更新](experiment-authoring.md#圖的建立顯示與更新)。
`@plots.on_plot_thread` 的 owner 執行、artist 歸屬驗證與 refresh 規則見
[plotting〈原生 matplotlib 更新〉](../../lib/zcu_tools/plotting/README.md#原生-matplotlib-更新)。

```python
fig, ax = context.plots.subplots(num="signal")
ax.set(xlabel="Time (µs)", ylabel="IQ real (a.u.)")
line = live_line(ax)


@context.plots.on_plot_thread
def update(data):
    return line.set(delays, data)


fig.show()
```

[`support/live.py`](support/live.py) 提供掛到 axes 的工具，`_live_attached.py` 擁有資料檢查與 artists，
`_live_format.py` 處理格式。`support.live.AttachedLine` 可用於作者的型別標註。
`live_errorbar(ax).set(xs, signals, yerr)` 更新點與對稱誤差棒；NaN 誤差只省略誤差棒，保留資料點。
作者直接在 axes 設 ticks、顏色與標題。掃描歷史線由作者在另一個 axes 建立並更新。
Reference marker 與動畫錄影不在這些工具中。

共用的 `live_heatmap` 讓所有欄共用一組 `ys` 座標。[CPMG core](twotone/time_domain/cpmg/core.py) 的 Live 與 Analyze 都用 `_count_map`，讓每個 pulse count 欄各用自己的 delay grid，不呼叫 `live_heatmap`。

Singleshot Check 在自己的 `analyze` 畫分類 IQ／區域圖；GE 的 Post 版面在 `singleshot/ge/core.py`。
Amp／Len Rabi 在各自的 core 畫 joint-fit population、histogram 與 confusion 圖，
分析計算仍由原分析 owner 負責。

## Mux readout 與表單初值

有 reps／rounds 欄位的表單在 Open／Defaults 時填入 1000／10，初值可編輯。
Lookback 只有 rounds，初值為 10，core 固定 reps=1。
以 shots 控制的 singleshot 表單不改 shots，也不增加 reps／rounds 欄位。
Mux twotone spectrum 的 relax_delay 初值固定為 1.0 µs，不取 captured T1。

`onetone/range_find` 使用普通 pulse readout 的 generator 與 ADC 硬體掃絕對 MHz。
它與 `onetone/freq` linear 共用純 core cfg 與頻率格點。
兩個 core 的 `run` 各自列出 reset → PulseReadout → declare_sweep → acquire 序列，並建立及更新自己的圖。
Raw 是一維頻率與 complex IQ。
單一 primary 入口為 Interactive，靜態找谷只提供初始猜測。預設不平滑，以 depth / FWHM 排序，
剔除超過 25 MHz 的寬谷。分析名單只在 Analyze 參數，不進 cfg 或量測表單。
名單預設 `["R1"]`，至少提供一個元件標籤，初始依頻率升序配對。
Picker 可調五個找谷 controls、增刪選點與任意調整元件對應。按 Done 才提交最終結果，
每個 capture 中存在的已指派名稱只提議一列 `freq`，不提出 module、kappa 或 edelay。
最終 controls／選點／對應保存在分析 summary，parameters 仍是初始表單。
Recipe 等待 agent 透過 `tab_interact` 確認並送出 done，才詢問 writeback。
操作與判讀見 [range find guide](onetone/range_find/guide.md)。

`mux/onetone` 使用 `readout/mux`。**Frequency offset** 是選定 tone 共用、可編輯的 MHz 偏移；每個 sweep 點把名目偏移加到各 tone 的基準 RF。
Open／Defaults 的 `df` 初值由 [`mux/onetone/form.py` 的 `make_form`](mux/onetone/form.py) 定義。
每條曲線透過原始 tone 序號對應完整 cfg，資料保存共同偏移軸、原始序號與二維 signals。
Tone label 是可重複的顯示文字，不是 component 名稱，也不改變資料軸。
`mux/onetone` adapter 從 group 的 resonators 產生 readout tones。
有 readout slot 時以副本保留 scalar 初值，再套用 ringup timing。
缺 slot 時使用 blank mux 範本，gen_ch 取 `$mux.wiring.gen_ch`，第 i 個 tone 的 ro_ch 取 `$mux.wiring.ro_ch[i]`，gain 填 0.1。
ADC window 固定填入 1.0 µs，mixer 取 group 的 `mixer_freq`。初值來源及缺值處理見 [mux onetone guide](mux/onetone/guide.md#無-readout-slot-的候選來源)。
`participants` 是非空、唯一的原始整數 index list，按輸入順序量測。Open／Defaults 將 alias `$participants` 的名稱依 `$mux.resonators` 位置轉成 indices；缺列、未綁或空清單預設全部。之後改 aliases 不更新現有輸入，需按 Defaults 或重新開啟 tab。
Lab 將 subset 交 program，不改完整 cfg。
RunRecord 與 native cfg restore 保留完整 readout、labels 及 selection，不以目前表單補歷史資料。
Adapter 提供 K 條 active live 曲線及 K 張 `fit_i` 圖，i 是原始 tone 序號。
Mux onetone 的 GUI 宣告自己的分析選項，包含模型與 delay 選擇。
Core 呼叫 `support/resonator.py` 的 `fit_resonance` 做數值擬合，逐 tone 隔離數值失敗。
Core 的 `analyze` 自己持有 Figure，選擇既有 lib model 的公開 renderer 畫擬合圖。
沒有歷史 cfg 時拒絕 FIT。
Module proposal 只更新成功 active tone 的頻率，保留其他完整設定，固定寫到 `$mux.readout.readout_rf`。
單一參數從 Run／Load capture 的 `$mux.resonators` 依完整原始序號對應。Resonators 數不等於完整 tone 數時，保留 readout module 與 capture 可解析的 readout slot 提案，省略逐 resonator scalar，summary 顯示原因。
MuxGroup 可不填 qubits 做 resonator bring-up，readout 各欄位仍可編輯，不增加完整 qubit 綁定門檻。
填了則由 model 檢查等長與重複 qubit。Form 在 Open／Defaults 呼叫
`support.measure.mux_bindings.validate_mux_bindings(ctx.parameters)`，以固定 capture
檢查 component 存在、kind 及逐項 qubit.resonator 對應，無效就拒絕來源。
其他 mux Forms 共用此入口，呼叫例見 [Measure support](support/measure/README.md#mux-綁定驗證)。
Two-qubit Forms 用 [`captured_qubits`](two_qubit/_measure.py) 要求完整的有序 mux 綁定，從 `$mux.qubits` 找出兩個位置並核對已綁 readout 的類別與 tone 數。這不限制 qubit 的物理子種類或 coupler 的種類。正常 lowering 後，`TwoQubitForm.to_cfg` 用同次 capture 核對實際 ref／inline readout 的完整 tone 數。Refresh 保留草稿，以新 capture 重建 Form 和 hidden indices。操作員仍須核對 tone 順序，見 [Two-qubit guide](two_qubit/guide.md#綁定位置與讀出-tones)。
Module 可整組引用或 inline 編輯，tones 使用 picker 逐項編輯。
七個 mux qubit Forms 有 readout slot 時引用 library，完整 tones 與 scalar 設定取自該模組。
缺 slot 時從 blank mux 範本加 group tones，填入 resonator 校準頻率與名稱 label。
Gen_ch 取 `$mux.wiring.gen_ch`，各 tone 的 ro_ch 取 `$mux.wiring.ro_ch[i]`，順序與 group resonators 相同，gain 填 0.1。
Pulse length、ro_length 與 trig_offset 留空必填。
即使 group 有 time_of_fly，也不替 qubit Forms 推導 timing。
頻率缺失或無效時也留空必填。
`mux/onetone` 有 readout slot 時使用副本，以 group tones 與 ringup timing 作初值。
Ringup-copy 的 tone gain 填 0.1，ADC 取同序的 `$mux.wiring.ro_ch[i]`。
Mux onetone 缺 slot 時仍使用 blank 範本，ADC window 固定填入 1.0 µs，不依賴 kappa 或 time_of_fly。
無 slot 的 mixer 從 `$mux.mixer_freq` 取初值，缺失或 null 時留空必填，不依賴 tone 頻率或 kappa。
Trigger 取 `$mux.time_of_fly` 加五倍 ringup 時間，pulse 長度取 trigger 加 window。
任一 kappa 或 group time_of_fly 缺值時，trigger 與 pulse 長度留空必填，不讀 resonator 的 pulse 路徑 TOF。
`post_delay` 用最小 κ 估五個光子衰減時間 `5 / (2π · κ)`。有 slot 時，用已設 κ 計算並保留較長的原值；全未設時保留原值。無 slot 時，只有 κ 齊全才推導這段等待，不要求 `time_of_fly`，否則保留範本的 0 µs。
Readout 欄位說明列來源與未經裝置確認的限制，Run 仍執行 capability 檢查。
Label 可編輯且可重複，不當 identity。固定槽位只限制增刪，不防止整列表交換值。
Library 的 `readout/mux:blank` 不預填必需的通道、gain、時長或 trig_offset，不以 0 代替未校準的 trigger timing。Mux Forms 的 fallback 透過 typed model-list seeds 加入 tones。
九個 FTF Forms 有 readout slot 時同樣引用 library。八張缺 slot 時提供 blank mux 範本，rb 缺 slot 時保持 Invalid。
操作員必須確認通道與功率；範本不是硬體安全預設。

## Mux flux maps

`mux/onetone_flux_global` 與 `mux/twotone_flux_global` 掃 group 的 coil source。
`mux/onetone_flux_local` 與 `mux/twotone_flux_local` 同步掃各 participant 的局部 DC 線。
各實驗在自己的 core 建立 program 與 flux 取數迴圈，不呼叫其他實驗的 run。
外層以軟體設值並等待 settle，內層由 onetone 在 host 掃頻，twotone 在硬體掃頻。
Local 每 tone 有自己的 Start／Stop，共用 `flux_expts`。Global 使用一個 `flux` 範圍。
四份 cfg 各自完整宣告，Form 依 cfg 宣告順序顯示，不使用 `field_order`。
順序為 `reset`、`readout`、`tones`、外層 DC 掃描、內層頻率點數、`participants`、DC 單位與 settle、取數欄位。
模組仍有 selector 與子表單，掃描仍有 `SweepCfg` 子表單。
Pulse 的 Advanced group 保留 NQZ、Phase、Pre-delay、Post-delay 與 Mixer freq 的既有摺疊。
四實驗的 cfg schema 為 5.0。Load 舊 major 時保留資料與 metadata，cfg 為 None，不轉換舊設定。
Onetone 的 `df` 是共用 MHz offset；twotone 的單一 `tones` 清單保存各 Q 的 drive 與 MHz 範圍，
`expts` 是共用硬體點數。
Guide 說明 kickstart、已有頻率、已有 readout slot 與 recipe 的起手步驟，以及 Open／Defaults／Refresh 的差異。
`participants` 保存原始 tone index 與執行順序，index 始終對應完整 group 的原始序號。
Stop 保留已完成的 flux 點，未完成點留 NaN。Analyze 每次用原始 `tone` index 選一個 participant，
沿既有 half／integer line picker 校準。Local 與 global 分開提出該 Q 的 `flux.period.<被掃 device>` 與預設勾選的 `flux.offset`，
不覆蓋其他 periods 或 Q。Global 使用 cfg.flux_dev 的實際名稱，不把 coil 當別名。
目的地只取 Run／Load capture。四個 guide 列出初值與操作路徑。
Native 保存 tone-major 三維 IQ 與各 tone 的 DC／MHz 座標。
Labber independent 匯出沿第一軸拆分三維 IQ，每個 tone 一個二維 log，同檔保存多個 logs。
Reader 回傳各片二維資料，不重組三維陣列。命名與維度限制見 [Labber owner](../../lib/zcu_tools/datafile/README.md#labber)。

## Mux qubit numeric cores

`mux/twotone_spectrum` 與 `mux/amp_rabi` 提供可 import 的 typed cfg、public `run`、分析與 native persistence。
兩個 `run()` 各自在 core 直接寫出掃描與 pulse 序列。
Twotone spectrum 用一個 program 的共用硬體 loop 掃 drive DAC 頻率，mux readout 保持固定。
Amp Rabi 也只編譯一個 program，在共用的 `point` 硬體 loop 同步掃各 drive 的 gain。
共同 reset 後，TimelineBuilder 以 fork／emit 讓 drives 同時起跑，再以 join 取得當點最長結尾，
接著做共同 mux readout。不補齊末端，不以全 sweep 最大長度 padding。
Amp Rabi 的 shot count 與 IQ moments 由 [Program 的 summed acquisition](../../lib/zcu_tools/program/README.md#summed-acquisition) 累加。
需要用 sums 估計逐點不確定度時，見 [analysis〈逐點 shot SEM〉](../../lib/zcu_tools/analysis/README.md#逐點-shot-sem)。
七個 mux qubit adapters 已註冊在 catalog，透過各 leaf 的 native `form.py` 生成完整
固定具名 tones 及獨立 Participants 輸入。元素 picker 只選編輯對象；名稱 readonly。
普通改名、增刪及重排由公開 FixedElements owner 原子拒絕。
未參與及未顯示元素錯誤照常顯示並擋 Run。七個同名 mux recipes 已明列於
`zcu_lab.recipes.RECIPES`，typed overrides、逐元素來源與操作見
[mux recipe guide](../recipes/mux-guide.md)。
各 leaf guide 說明 GUI、保存來源及 explicit Apply 操作，core 本身仍不依賴 GUI。

Mux 實驗每個參與 tone 各有一張 `measurement_i`，
不把多 tone 疊在同一 axis。每張圖保留自己的 actual x，delay 類使用共用
requested-delay 軸。

[Mux Ramsey](mux/t2ramsey/core.py) 與 [Mux Echo](mux/t2echo/core.py) 的 `analyze` 各自把作圖用的密網格 `curve_times` 傳給 [`fit_coherence_trace`](support/mux_coherence_analysis.py)。
Helper 用取得的資料擬合並評估品質，再在 `curve_times` 上計算同次模型的曲線。
密網格只用來畫模型，不是擬合的量測點，也不是 shot SEM 的來源。

Mux GE Live（`measurement_i`）以藍／紅區分 probe off／on 取數列，不讀 Analyze 的 **Initial State**。
Analyze Fit（`fit_i`）用 [`ge_signals_by_state`](support/singleshot/util.py) 依 **Initial State** 將同一批 IQ 排成 g／e 準備（藍／紅），圖例標出各準備對應的 probe row。
`diagnostic_i` 也用準備色。Post（`post_i`）沿用所選來源的 Primary 準備映射，分格畫出 g／e 準備的 IQ；各格的顏色表示 g／e／Other 分類結果。
讀圖見 [Mux GE guide](mux/singleshot_ge/guide.md#read-the-plots-and-use-the-result) 與
[單 qubit GE guide](singleshot/ge/guide.md#how-to-read-the-results)，IQ 密度的取色規則見 [figure-style R4](figure-style.md#r4尺度跟著資料走)。

單 qubit 與 mux GE 的 Live 更新先複製 IQ，
再經公開的 `@plots.on_plot_thread` 在 host owner 繪圖。Worker 不持有 active artists，
partial NaN 不補零。Mux GE 逐 tone 直接呼叫數值工具，不委派單 qubit 實驗的分析。

七個 mux qubit cfg 使用完整固定 `tones` 與獨立 `participants: list[int]`。
每個元素有固定 qubit `name` 與 nullable drive；pulse sweep 類另有只含 `start`／`stop` 的 `ToneRangeCfg`，
T1／Ramsey／Echo 使用共用 requested-delay 軸。Participants 選擇使用的 drive 與逐 tone sweep。
同 generator 的脈衝若重疊，program compiler 在 build 時拒絕。Cfg 仍拒絕 drive 與 readout 共用 generator。
未參與欄位保留草稿，結果為 None。全 readout 身分始終驗證。三個 pulse scan 的頂層 `expts` 是共用正整數，
表單在 Tones 後只顯示一個 Expts，預設為頻譜 301、Amp Rabi 51、Len Rabi 101。
每 tone 只有 Start／Stop，不接受舊的每 tone 點數。
Len Rabi 參與元素支援 const／flat_top 與取樣波形，例如 cosine、gauss、DRAG。
全體 participants 都用 const／flat_top 時，保留硬體長度表路徑。
任一 participant 使用取樣波形時，全體 participants 一起走 host 路徑。
Host 每點只編譯一次，跨 rounds 重用程式。
原生 arb asset 的長度由資產決定，掃描前即拒絕，不伸縮或裁切。
Participants 非空、不重複，索引必須在完整 tones 範圍內。Runtime 直接保留清單順序，
例如 `[2, 0]` 依序量 Q3、Q1。GE raw、analysis 與 native persistence 同樣保留該順序。
RunRecord 保存完整具名 cfg 與 participants，不另保存目的地或 group snapshot。

`prepare_frequency_scan(cfg, soccfg=...)` 直接使用 participants 原始 indices，
回傳 `mux.twotone_spectrum.MuxFreqScan`。
其 `indices` 保留 participants 順序的原始序號，`coordinates` 是各 drive 硬體 grid
上的 `(K,P)` float64 MHz 陣列。座標沿 `sweep2array(..., "freq", ...)` 的硬體
start/span/step 量化。Frequency preparation 不建立逐點 pulse；
Core 以各 tone 的 start／stop 與共用 expts 建立 `SweepCfg`，
`run()` 再用 `sweep2param("point", sweep)` 建立 symbolic drive。
`prepare_amplitude_scan(cfg, soccfg=...)` 仍回傳 `mux.amp_rabi.core.MuxPulseScan`，
包含原始 `indices`、以 `round_zcu_gain` 獨立量化的 gain `coordinates (K,P)` 及
固定 numeric `point_pulses (P,K)`。`run()` 不用這批 pulse 逐點編譯，並以編譯出的硬體 gain grid 取代準備階段的座標。
這些入口只準備資料，不 acquire；各 public `run()` 直接寫出 common-start zip 排程，
不展開 Cartesian product。

`MuxSweepResult` 保存原始 `tone_indices`、各自量化的 `coordinates` 及平均 complex IQ `signals`。
兩個矩陣都是 `(K,P)`。NaN IQ 表示未取得的 partial data，不表示成功。
各 core 的 `analyze(source, options, plots=...)` 使用 explicit RunRecord，允許 cfg=None。
每個原始序號各有 `fit_i` 圖、校準值／誤差／quality，或保留原因的 `MuxFitFailure`。
Freq 沿用單 qubit 的 magnitude/background transform 與 linewidth fitter；Amp 沿用零 gain phase、無衰減 fit。

Native mapping 使用 `coordinates`／`signals` 兩個 variables，共享 Point／Tone axes。
Frequency 座標在磁碟上是 Hz，在記憶體是 MHz。Gain 不縮放。
Analyze／Save 使用 explicit RunRecord；Load 還原同次完整 tones cfg、原始 indices、獨立數值座標與 IQ。
它們不讀目前 Form 或 library。Freq 與 Amp Rabi 取消時都保留全 sweep 已完成 rounds 的平均。
Amp Rabi 若沒有完成任何 round，IQ 維持 NaN。
Acquisition 例外原樣傳出；core 回傳的資料由作者決定是否可分析。
`MuxOnetoneExp.run` 與 `MuxFreqExp.run` 可接 `on_acquire_stop`，在每次外部取消的 acquire 清理後同步通知。
Mux flux scan 用這個作者 callback 丟棄被取消的當前 flux point，不從 parent stop 推斷子層資料完整度。
Numeric core 不保存 writeback destination，也不發 proposal。

## Qubit flux helpers

無 GUI 的 `support/flux.py` 接受 qubit 的具名 `flux` 校準區塊，不讀檔或操作儀器。
`qubit_flux_frame(flux)` 複製 offset 與帶號 period mapping，包含尚未校準的 null offset。
`device_flux_points(flux, device_name, values)` 委派共用投影，以其他儀器的值算實際 half／integer，回傳 `QubitFluxCalibration`。
未校準 offset 或被選儀器沒有 period 時不能投影。其他儀器缺值視為 0，明寫無效值仍報錯。
各 caller 明確選擇 point frame 與 device values 的來源及取值時點。
Onetone、twotone 與四個 mux fluxdep adapter 在 Analyze 時呼叫
`support/measure/writeback_helpers.fluxdep_writeback_items`。
它以 Run 的 ctx.parameters 複製原 frame，以 `RunMetadataContent(record.metadata).device_snapshot` 計算新的 offset。
被掃儀器使用擬合 half 與新 period 大小，方向沿用原設定，首次校準採正方向。
其他儀器保留 period，以 Run-start 值貢獻 flux；缺值視為 0。
Helper 分開提出帶號 period 與預設勾選的 offset，並在 offset note 列出其他儀器的值、單位與貢獻。
同 Run 同步掃描其他線不阻擋 proposal。
同次提出 ordinary record，保存原始 fitted half／integer／正 period、被掃 device、完整 connected snapshot 與原 frame。
接受時不重讀 point 或儀器，不把本次新校準放入 record frame。

## 實驗角色與 device 操作

`support/device.py` 擁有儀器角色 helpers，供實驗 core 與 workflow 使用。DeviceInfo 是資訊，不再提供 set_flux／set_freq／set_power／set_output 控制命令。Device 層只提供儀器本身的操作，不知道 flux 或 RF 角色；實驗層選 driver、明示單位與操作用途。

- `check_flux_unit(cfg: BaseDeviceInfo, unit: str) -> None` 是純檢查，只比 YOKO／Fake Info 儲存的 value_unit。YOKO Info 驗 current／A 與 voltage／V，Fake 為建立時固定的 native、V 或 A。Mismatch 或契約外 unit 報 ValueError，其他 Info 報 TypeError 並指名不支援 flux 設定的類別。不換算、不改 cfg，不讀寫硬體或 cache。
- `set_flux(device, value, *, unit, stop, progress=False)` 委派同一 checker，保既有 mode 查詢、stop、progress 與 driver safety，不新增 get_info 或其他硬體查詢。接受 A、V 或 native。YOKO 的 A 要求 current 模式並呼叫 set_current，V 要求 voltage 模式並呼叫 set_voltage。Fake 比對 driver 實際 get_info() 回報的固定單位，再呼叫 set_value，不新造 native Info。單位或模式不符在寫入前拒絕。Helper 不換算、不切模式、不改 rampstep、不自動開 output，YOKO ramp 使用同一 stop 並沿既有安全限制。
- `set_rf_frequency(device, freq_hz)` 呼叫 SGS100A 的 set_frequency 或 Fake 的 set_value。單位為 Hz，MHz 等換算由 caller 在呼叫處完成。
- `set_rf_power(device, power_dbm)` 呼叫 SGS100A 的 set_power 或 Fake 的 set_value，單位為 dBm。
- `set_output(device, "on" | "off")` 呼叫支援 driver 的 set_output。

Helpers 對其他 device 種類報 TypeError，driver 錯誤原樣傳播。FakeDevice 維持單一 value，不新增 RF 欄位、protocol 或 registry；模擬頻率與功率仍寫同一值。

## 序列與量化入口

- [ZigZag core](twotone/zigzag/core.py) 的 `run()` 明列 register loop 的 pulse 序列。
- [Actual detune](support/detune.py) 的公開 docstring 說明 Ramsey／Echo 的 time_scaler、MHz 計算與原生量化失敗語意。

## Layout

```text
zcu_lab/
├── definitions.py        # explicit measure catalog
├── templates.py          # startup-only template catalog
└── v2/
    ├── <family>/<experiment>/
    │   ├── core.py
    │   └── gui.py         # optional measure adapter
    └── support/          # shared domain helpers
```

`support/resonator.py` 集中單 tone 與 mux 的模型 fitting 及 route prior 解析，不 import GUI。各實驗自己持有 Figure 並選擇 renderer，各 caller 保留分析選擇、失敗隔離與 writeback 政策。Core 不經 `support/measure/` 取得這些 mechanics。

組合根以 `Registry(templates=TEMPLATES)` 注入預設持久範本。Library 新建與表單範本展開都讀 `registry.templates`。`definitions.register_all` 明列 measure adapters，並把收到的範本綁給 chain_length 與 rpi_length 的 Form factory。Import 不註冊或操作硬體。

Reload 重建 leaf、adapter catalog 與 core-typed Notebook helper，保留 measure／singleshot shared support、workflow support、templates 與框架 identity。兩個 Form factory 使用該次註冊收到的 Registry 範本。修改固定依賴需要重啟。

框架 owner 用測試自己定義的最小合成 workflow／adapter 驗證公開接縫，不 import `zcu_lab`。本套件的實驗定義不寫個別測試，也不以 lab registry 選取測試材料。刪掉 `zcu_lab/` 後整套 pytest 仍須通過，包含 slow。Core-only caller 明確 import `core`；GUI callers 明確選擇其附件。
