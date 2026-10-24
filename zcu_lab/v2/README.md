# v2 experiments

**Last updated:** 2026-10-24，實驗表單平鋪與 pulse Advanced

**讀者：** lab-author、agent。查找實驗的檔案分工、作者契約與各家族入口。

先從 [tutorial](../tutorial/README.md) 新增實驗，再按下列入口查契約與操作。

## Reading routes

- [FTF bring-up 與 CZ 校準](two_qubit/guide.md) 說明既有 F1/F2 單 qubit 校準、角色與保存方式。FTF 控制鏈可執行仍不代表物理校準或 fidelity 已驗收。
  - 四分支 [conditional spectroscopy](two_qubit/conditional_spectroscopy/guide.md)、[relative-phase × duration constructive phase](two_qubit/constructive_phase/guide.md) 與 [common-frequency × duration Rabi chevron](two_qubit/chevron/guide.md) 提供雙線校準圖與分析。
  - [CZ common-amplitude](two_qubit/cz_amplitude/guide.md) 保留 same-shot raw，擬合零相位完整回返週期。操作員逐列接受 coupler 的雙 pulse／雙 slot proposals。
  - [Conditional phase](two_qubit/conditional_phase/guide.md) 提供獨立 X/Y acquisition、phase 診斷與 n=1 frequency proposal。
  - [Local-Z](two_qubit/local_z/guide.md) 以雙 target 完整列估計 slope，依 correction=-slope 提供 correction pair 與手動 parameter Apply。
- [Experiment authoring](experiment-authoring.md) 說明核心 records、資料形狀與 run 慣例。
- [Measure authoring](measure-authoring.md) 說明 cfg defaults、operator guides、writeback 與 reload。
- [GUI contracts](gui-contracts.md) 說明 concrete adapter hooks 與 capability。
- [Measure support](support/measure/README.md) 提供共享 domain mechanics。
- [Framework runtime](../../lib/zcu_tools/experiment/v2/runtime/README.md) 擁有 Schedule、buffer、acquire 與 stop lifecycle。

## 套件邊界

每個實驗有自己的資料夾。`core.py` 擁有 cfg、Result、量測與分析政策。`gui.py` 是可選的 measure-gui adapter。Leaf 的 `__init__.py` 只提供 core exports，不載入 GUI。

`v2` 延續 program/v2 的版本脈絡，不表示 GUI framework 的第二版。通用 runtime、量化、SNR 與 tracker 工具留在 `zcu_tools.experiment.v2`。Core 不依賴 GUI，import-linter C15 檢查 core 到 GUI framework 的直接與間接依賴。

## 即時圖工具

從 `zcu_lab.v2.support` import `live_line`、`live_scatter`、`live_heatmap` 與
`live_heatmap_with_line`，以 `live_line(context.plots, name, xlabel, ylabel, ...)`
建立 typed handle。工具與格式直接放在公開支援套件，不另建 plotting 子套件。
`support/live.py` 選擇格式與 scan-line 呈現，私有格式及 handle 檔保存 artists 與輸入檢查。
作者要改 ticks、顏色、最近掃描線或增添工具時，從這裡修改。

四工具保留既有 defaults、數值、具名圖及既有 axes 接入。Line 的 configure_axes 在 owner
初始化後、呈現前執行。Handle 更新實數資料，保留 NaN 未測點與可選 title；
`refresh=False` 可批次更新。Lib 的通用 LiveUpdate 負責複製資料、owner 執行與 refresh，
finish／release 後拒絕更新。Reference marker 與動畫錄影不在這些工具中。

Singleshot 的分類 IQ／區域圖在 `singleshot/plot.py`，joint-fit population／histogram／
confusion 圖在 `singleshot/plot_rabi.py`。它們保留既有用途，分析計算仍由原分析 owner 負責。

## Mux readout 與表單初值

有 reps／rounds 欄位的表單在 Open／Reset 時填入 1000／10，初值可編輯。
Lookback 只有 rounds，初值為 10，core 固定 reps=1。
以 shots 控制的 singleshot 表單不改 shots，也不增加 reps／rounds 欄位。
Mux twotone spectrum 的 relax_delay 初值固定為 1.0 µs，不取 captured T1。

`onetone/range_find` 使用普通 pulse readout 的 generator 與 ADC 硬體掃絕對 MHz。
它與 `onetone/freq` linear 共用純 core cfg、頻率格點與繪圖 buffer。
兩個 core 的 `run` 各自列出 reset → PulseReadout → declare_sweep → acquire 序列。
Raw 是一維頻率與 complex IQ。
分析名單只在 Analyze 參數，不進 cfg 或量測表單。名單預設空，先探索符合最小 MHz 間距的無名凹陷，
依頻率升序回報 freq 與 depth，不提出寫回。先看圖，再於 Analyze 填入不重複的 resonator 名稱。
分析去除直線 magnitude 背景；非空名單選取最顯著的 N 個凹陷，再依頻率升序配名稱。
每個 capture 中存在的名稱只提議一列 `freq`，不提出 module、kappa 或 edelay。
不足 N 個凹陷時整次分析失敗。操作與判讀見 [range find guide](onetone/range_find/guide.md)。

`mux/onetone` 使用 `readout/mux`，在每個 sweep 點對選定 tone 加上共同的名目頻率偏移。
每條曲線透過原始 tone 序號對應完整 cfg，資料保存共同偏移軸、原始序號與二維 signals。
Tone label 是可重複的顯示文字，不是 component 名稱，也不改變資料軸。
`mux/onetone` adapter 從 group 的 resonators 產生 readout tones。
有 readout slot 時以副本保留 scalar 初值，再套用 ringup timing。
缺 slot 時使用 blank mux 範本，gen_ch 取 `$mux.wiring.gen_ch`，第 i 個 tone 的 ro_ch 取 `$mux.wiring.ro_ch[i]`，gain 填 0.1。
ADC window 固定填入 1.0 us，完整 group 的 kappa 都有效時再推導其餘 mixer／timing 候選，來源見 [mux onetone guide](mux/onetone/guide.md#無-readout-slot-的候選來源)。
`participants` 是非空、唯一的原始整數 index list，按輸入順序量測。Open／Reset 將 alias `$participants` 的名稱依 `$mux.resonators` 位置轉成 indices；缺列、未綁或空清單預設全部。之後改 aliases 不更新現有輸入，需 Reset 或重新 Open。
Lab 將 subset 交 program，不改完整 cfg。
RunRecord 與 native cfg restore 保留完整 readout、labels 及 selection，不以目前表單補歷史資料。
Adapter 提供 K 條 active live 曲線及 K 張 `fit_i` 圖，i 是原始 tone 序號。FIT 共用 onetone/freq 的模型與 delay 選項，逐 tone 隔離數值失敗。沒有歷史 cfg 時拒絕 FIT。
Module proposal 只更新成功 active tone 的頻率，保留其他完整設定，固定寫到 `$mux.readout.readout_rf`。
單一參數從 Run／Load capture 的 `$mux.resonators` 依完整原始序號對應。Resonators 數不等於完整 tone 數時，保留 readout module 與 capture 可解析的 readout slot 提案，省略逐 resonator scalar，summary 顯示原因。
MuxGroup 可不填 qubits 做 resonator bring-up，readout 各欄位仍可編輯，不增加完整 qubit 綁定門檻。
填了則由 model 檢查等長與重複 qubit。Form 在 Open／Reset 呼叫
`support.measure.mux_bindings.validate_mux_bindings(ctx.parameters)`，以固定 capture
檢查 component 存在、kind 及逐項 qubit.resonator 對應，無效就拒絕來源。
其他 mux／FTF Forms 共用此入口，精確 import 與呼叫例見
[Measure support](support/measure/README.md#mux-綁定驗證)。
Module 可整組引用或 inline 編輯，tones 使用 picker 逐項編輯。
七個 mux qubit Forms 有 readout slot 時引用 library，完整 tones 與 scalar 設定取自該模組。
缺 slot 時從 blank mux 範本加 group tones，填入 resonator 校準頻率與名稱 label。
Gen_ch 取 `$mux.wiring.gen_ch`，各 tone 的 ro_ch 取 `$mux.wiring.ro_ch[i]`，順序與 group resonators 相同，gain 填 0.1。
Pulse length 與 ro_length 留空必填，頻率缺失或無效時也留空必填。
`mux/onetone` 有 readout slot 時使用副本，以 group tones 與 ringup timing 作初值。
Ringup-copy 路徑保留既有初值：tone gain 為 0.1，ro_ch 按 group 順序從 0 起算。
Mux onetone 缺 slot 時仍使用 blank 範本，ADC window 固定填入 1.0 us，不依賴 kappa 或 time_of_fly。
無 slot 的 mixer 從 `$mux.mixer_freq` 取初值，缺失或 null 時留空必填，不依賴 tone 頻率或 kappa。
Trigger 取 `$mux.time_of_fly` 加五倍 ringup 時間，pulse 長度取 trigger 加 window。
任一 kappa 或 group time_of_fly 缺值時，trigger 與 pulse 長度留空必填，不讀 resonator 的 pulse 路徑 TOF。
上述缺值情境仍保留 1.0 us 的 ADC window。
Readout 欄位說明列來源與未經裝置確認的限制，Run 仍執行 capability 檢查。
Label 可編輯且可重複，不當 identity。固定槽位只限制增刪，不防止整列表交換值。
Library 的 `readout/mux:blank` 不預填必需的通道、gain 或時長。Mux Forms 的 fallback 透過 typed model-list seeds 加入 tones。
九個 FTF Forms 有 readout slot 時同樣引用 library。八張缺 slot 時提供 blank mux 範本，rb 缺 slot 時保持 Invalid。
操作員必須確認通道與功率；範本不是硬體安全預設。

## Mux flux maps

`mux/onetone_flux_global` 與 `mux/twotone_flux_global` 掃 group 的 coil source。
`mux/onetone_flux_local` 與 `mux/twotone_flux_local` 同步掃各 participant 的局部 DC 線。
外層以軟體設值並等待 settle，內層重用既有 onetone host sweep 或 twotone hardware sweep。
Local 每 tone 有自己的 Start／Stop，共用 `flux_expts`。Global 使用一個 `flux` 範圍。
四份 cfg 與 native 保存形狀都是 flat：`reset`、`readout`、`reps`、`rounds`、`initial_delay`、`relax_delay` 在 top-level。
Module 與掃描欄位依 `field_order` 平鋪，不宣告 Modules／Sweep group。
模組仍有 selector 與子表單，掃描仍有 `SweepCfg` 子表單。
Pulse 的 Advanced group 保留 NQZ、Phase、Pre-delay、Post-delay 與 Mixer freq 的既有摺疊。
Schema 為 3.0。Load 舊 major 時保留資料與 metadata，cfg 為 None，不轉換舊巢狀設定。
Onetone 的 `frequency.df` 是共用 MHz offset；twotone 的 `frequency.tones` 保存各 Q 的 drive 與 MHz 範圍，
`frequency.expts` 是共用硬體點數。四張表單依 participants、tones、modules、frequency／DC scan、averaging、timing 排列。
Guide 說明 kickstart、已有頻率、已有 readout slot 與 recipe 的起手步驟，以及 Open／Reset／Refresh 的差異。
`participants` 保存 Q 名與執行順序，tone index 始終是完整 group 的原始序號。
Stop 保留已完成的 flux 點，未完成點留 NaN。Analyze 每次用原始 `tone` index 選一個 participant，
沿既有 half／integer line picker 校準。Local 提各 Q 的 flux scalar，global 提 group `coil_flux` 中該 Q 的校準，
不覆蓋其他 Q。目的地只取 Run／Load capture。四個 guide 列出初值與操作路徑。
Native 保存 tone-major 三維 IQ 與各 tone 的 DC／MHz 座標。Labber 不支援這種三維獨立 grid，
Save 以各 output 的 status 回報其失敗，不影響成功保存的 native 資料。

## Mux qubit numeric cores

`mux/twotone_spectrum` 與 `mux/amp_rabi` 提供可 import 的 typed cfg、public `run`、分析與 native persistence。
兩個 `run()` 各自在 core 直接寫出掃描與 pulse 序列。
Twotone spectrum 用一個 program 的共用硬體 loop 掃 drive DAC 頻率，mux readout 保持固定。
Amp Rabi 仍逐點 compile。
共同 reset 後，既有 Join 讓 drives 同時起跑，短 branch 完成後等待當點最長 branch，
再做共同 mux readout。不補齊末端，不以全 sweep 最大長度 padding。
七個 mux qubit adapters 已註冊在 catalog，透過各 leaf 的 native `form.py` 生成完整
固定具名 tones 及獨立 Participants 輸入。元素 picker 只選編輯對象；名稱 readonly。
普通改名、增刪及重排由公開 FixedElements owner 原子拒絕。
未參與及未顯示元素錯誤照常顯示並擋 Run。七個同名 mux recipes 已明列於
`zcu_lab.recipes.RECIPES`，typed overrides、逐元素來源與操作見
[mux recipe guide](../recipes/mux-guide.md)。
各 leaf guide 說明 GUI、保存來源及 explicit Apply 操作，core 本身仍不依賴 GUI。

Mux 實驗每個參與 tone 各有一張 `measurement_i`，
不把多 tone 疊在同一 axis。每張圖保留自己的 actual x，delay 類使用共用
requested-delay 軸。GE 使用 IQ scatter，以藍／紅區分 probe off/on 兩個 preparation。
`support.singleshot.plot.draw_ge_scatter` 供單 qubit 與 mux GE 共用，先複製 IQ，
再經公開 `Plots.draw` 在 host owner 繪圖。Worker 不持有 active artists，
partial NaN 不補零。

七個 mux qubit cfg 使用完整固定 `tones` 與獨立 `participants: list[int]`。
每個元素有固定 qubit `name` 與 nullable drive；pulse sweep 類另有只含 `start`／`stop` 的 `ToneRangeCfg`，
T1／Ramsey／Echo 使用共用 requested-delay 軸。只針對 participants 驗證 drive、逐 tone sweep 與獨立 generator。未參與欄位保留草稿，結果為 None。全 readout 身分始終驗證。三個 pulse scan 的頂層 `expts` 是共用正整數，
表單在 Tones 後只顯示一個 Expts，預設為頻譜 301、Amp Rabi 51、Len Rabi 101。
每 tone 只有 Start／Stop，不接受舊的每 tone 點數。Len Rabi 參與元素只接受
const／flat_top，不伸縮或裁切 arb asset。
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
`prepare_amplitude_scan(cfg, soccfg=...)` 仍回傳 `support.mux.MuxPulseScan`，
包含原始 `indices`、量化 gain `coordinates (K,P)` 及
固定 numeric `point_pulses (P,K)`。Gain conversion 沿既有 `round_zcu_gain`。
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
它們不讀目前 Form 或 library。Freq 取消保留全 sweep 已完成 rounds 的平均；
Amp Rabi 取消保留已完成點與當點 completed-round 平均，未寫入 IQ 維持 NaN。
Caller 先呼叫 RunContext.scope.stop.raise_if_error()，不把 failure 的 partial data 報為成功。
Numeric core 不保存 writeback destination，也不發 proposal。

## 實驗角色與 device 操作

`support/device.py` 擁有儀器角色 helpers，供實驗 core 與 workflow 使用。DeviceInfo 是資訊，不再提供 set_flux／set_freq／set_power／set_output 控制命令。Device 層只提供儀器本身的操作，不知道 flux 或 RF 角色；實驗層選 driver、明示單位與操作用途。

- `check_flux_unit(cfg: BaseDeviceInfo, unit: str) -> None` 是純檢查，只比 YOKO／Fake Info 儲存的 value_unit。YOKO Info 驗 current／A 與 voltage／V，Fake 為建立時固定的 native、V 或 A。Mismatch 或契約外 unit 報 ValueError，其他 Info 報 TypeError 並指名不支援 flux 設定的類別。不換算、不改 cfg，不讀寫硬體或 cache。
- `set_flux(device, value, *, unit, stop, progress=False)` 委派同一 checker，保既有 mode 查詢、stop、progress 與 driver safety，不新增 get_info 或其他硬體查詢。接受 A、V 或 native。YOKO 的 A 要求 current 模式並呼叫 set_current，V 要求 voltage 模式並呼叫 set_voltage。Fake 比對 driver 實際 get_info() 回報的固定單位，再呼叫 set_value，不新造 native Info。單位或模式不符在寫入前拒絕。Helper 不換算、不切模式、不改 rampstep、不自動開 output，YOKO ramp 使用同一 stop 並沿既有安全限制。
- `set_rf_frequency(device, freq_hz)` 呼叫 SGS100A 的 set_frequency 或 Fake 的 set_value。單位為 Hz，MHz 等換算由 caller 在呼叫處完成。
- `set_rf_power(device, power_dbm)` 呼叫 SGS100A 的 set_power 或 Fake 的 set_value，單位為 dBm。
- `set_output(device, "on" | "off")` 呼叫支援 driver 的 set_output。

Helpers 對其他 device 種類報 TypeError，driver 錯誤原樣傳播。FakeDevice 維持單一 value，不新增 RF 欄位、protocol 或 registry；模擬頻率與功率仍寫同一值。

## 序列與量化入口

- [ZigZag core](twotone/zigzag/core.py) 的 `run()` 明列 register loop 的 pulse 序列，`build_zigzag_modules` 提供離線編譯序列。
- [Actual detune](twotone/time_domain/detune.py) 的公開 docstring 說明 Ramsey／Echo 的 time_scaler、MHz 計算與原生量化失敗語意。

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

`support/resonator.py` 集中單 tone 與 mux 的模型 fitting／畫圖 mechanics 及 route prior 解析，不 import GUI。各 caller 保留選擇、失敗隔離與 writeback 政策。Core 不經 `support/measure/` 取得這些 mechanics。

`definitions.register_all` 明列 measure adapters，從 caller 的 Registry 取得 `templates.TEMPLATES` 持久範本，填入 startup 的 library catalog。組合根以 `Registry(templates=TEMPLATES)` 注入一次，import 不註冊或操作硬體。Reload 重建 leaf、catalog 與 core-typed Notebook helper，保留 measure／singleshot shared support、workflow support、templates 與框架 identity。修改固定依賴需要重啟。

框架 owner 用測試自己定義的最小合成 workflow／adapter 驗證公開接縫，不 import `zcu_lab`。本套件的實驗定義不寫個別測試，也不以 lab registry 選取測試材料。刪掉 `zcu_lab/` 後整套 pytest 仍須通過，包含 slow。Core-only caller 明確 import `core`；GUI callers 明確選擇其附件。
