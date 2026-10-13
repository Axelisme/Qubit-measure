# Measure adapter authoring

**Last updated:** 2026-10-11，參數容器 guide 名稱

`zcu_lab/v2/<實驗>/gui.py` 把同資料夾的 core 包成 GUI adapter，供框架層 `zcu_tools.gui.app.measure` 驅動。Adapter 依賴框架契約 `ExpAdapterProtocol`。框架不 import 使用者套件，也不含實驗領域知識。

---

## 目錄佈局

實驗的 leaf layout 見[套件 README](README.md)。`../definitions.py` 明列可重載的 adapter catalog，`../templates.py` 組合 startup-only template catalog。Private `_support/measure/` 保存跨 adapter 的 builder、typed seeds 與 role defaults。


共用 `BaseAdapter` 位於 `gui/app/measure/adapter/base.py`。它的 authoring hook 回傳 `AdapterCfgDefinition`，並呼叫 instantiate 產生 fresh cfg。靜態 spec 留在 authoring contract，framework 不依賴 builder 或 Seed。
通用 source loader 位於 `gui/app/measure/catalog_loader.py`，由 launcher 宣告來源與重載範圍。
它支援明確的多個 source package，回傳尚未發布的 Registry。框架契約見
[Experiment reload](../../lib/zcu_tools/gui/app/measure/README.md#experiment-reload)。組合根重載 `zcu_lab.v2` 與 `zcu_lab.definitions`，包含 package exports。它保留 `zcu_lab.templates`、`_support` 下的 measure／singleshot 支援家族，以及所有框架 identity。Notebook 的 flux picker helper 引用 core 型別，隨 core 重載。`BaseAdapter` 位於重載範圍之外。新增 adapter 必須明列於 `../definitions.py`，不靠 class 掃描註冊。

Loader 在 GUI 關閉全部 tabs、排除進行中操作後才清除 owned import cache，從已檢查的 source
建立新 modules，並驗證 candidate registry。它不重啟硬體，也不交易回滾任意 import side effect；
可重載程式在 import 時不得啟動 thread、操作硬體或註冊外部 callback。此功能不是 notebook
autoreload，亦不支援保留舊 tabs 執行舊版本。延後 import 使用清除舊 bytecode 後的正常 Python
載入；操作中持續編輯尚未 import 的檔案不提供版本隔離保證。

Reload 是開發便利功能，依賴處理採 best-effort。允許函式內 import，不使用常駐 import guard
或 AST 規則禁止這種寫法。重載結束後的 deferred/dynamic import 可能載入新固定依賴，或造成
新舊程式混用；此時不保證攔截、報錯或辨識靜默失敗，使用者可重啟 GUI 回到乾淨的載入狀態。
已捕捉到的錯誤仍正常回報，不主動吞掉例外。這項取捨只放寬依賴完整性保證，不放寬 tab
關閉確認、RAM snapshot/retry、操作排他或 shutdown lifecycle 的契約。

每個實驗的 `gui.py` 是其 GUI policy 的修改入口。至少被兩個 adapter 共用的 mechanics 才放進 private `_support/measure/`。單一實驗的 policy 留在原檔。

每個 adapter 實作 context-free `cfg_definition()`（static shape + deferred fresh-default recipe）、
`run` / `analyze` / `get_writeback_items`，並在同一 class 以
`guide_text: ClassVar[AdapterGuide]` 宣告 operator-facing prose；framework-facing `guide()`
入口由 `BaseAdapter` 統一提供。每個 adapter 都透過 Base 具備 mandatory
`validate_run_request(req, raw_cfg)` preflight；只有需要 SoC-dependent
但可預測的 run-time cfg 檢查時才覆寫，並維持純 preflight
（例如 `len_rabi` 先確認 length sweep 在 ZCU 時間格點上不會量化成 zero-step）。詳細框架契約見
`../../lib/zcu_tools/gui/app/measure/README.md`。

`GuardService.acquire_run_permit` 在 Run 接受時凍結已呈現的 resolved cfg；
`run(req, raw_cfg)` 不重新讀取 live md/ml 或 lower schema。`BaseAdapter.build_exp_cfg`
把這份 raw cfg 交給 `zcu_tools.experiment.cfg_assembler.assemble_experiment_cfg`，
使用 `ml=None` 與 `RunRequest.device_snapshot`。不要把 active `ml/md` 綁進
長壽 service object，也不要讓 `ModuleLibrary` store 擁有 live device snapshot。
generic model/default/inheritance與validation/lowering直接從`zcu_tools.gui.cfg`匯入；measure
entry point只組current md expression、measure module shape與`SweepCfg` ports。measure adapter
facade只提供framework contract、request/result/writeback/analyze params與session signature
vocabulary，不forward shared generic names。role/module conversion policy仍在使用者的measure
domain，不下沉到shared core。template registration同時攜帶context-free fresh shape factory與eval-aware
value factory；named template沿用`ROLE_FACTORIES.shape`，blank template直接依`experiment.cfg_editing` closed catalog
順序產生。cfg role seed與library adopt policy仍由使用者套件擁有
（ADR-0065；catalog 詳見 `../../lib/zcu_tools/experiment/cfg_editing/README.md`）。

Adapter的module/waveform domain helpers保留可讀名稱，但回傳shared
`ReferenceSpec(kind="module" | "waveform")`與`ReferenceValue`。kind由domain factory顯式
設定，role/default assembly不再依兩套平行shared class分派。

Adapter cfg由context-free `MeasureCfgBuilder`單段宣告；builder不接`SessionEnv`，只有
`MeasureCfgDefinition.instantiate(ctx)`解析fresh defaults。`pulse/readout/reset`的
`ModuleInit.SMART`優先引用ModuleLibrary calibrated entry，required miss退回inline blank、optional
miss產生`None`；`ModuleInit.INLINE`永遠使用fresh blank；`ModuleInit.DISABLED`只允許optional ref且
永遠產生`None`。

module declaration同地保存`blank_overrides`、unconditional `overrides`與`locked`。
`blank_overrides`只修改custom/fallback blank；`overrides`也作用於linked snapshot；所有paths先完整
preflight，locked不得與override重疊，因此失敗不留下partial mount。Spec/Value資料模型仍分離，
但adapter作者只描述一次；shared `CfgSchemaAssembler`同步materialize兩棵樹並對齊literal。

deferred defaults使用小型typed Seed vocabulary：raw/literal、`md`、`scaled_md`、`value_source`、
少數具名range factory與單一純函式`custom` escape hatch。recipe只在fresh cfg建立時解析；restore與
使用者編輯不reseed。不得掃描任意library name或依field name猜domain數值。

Adapter defaults 以 notebook bring-up seed 與目前 parameter context 校準值共同定義：有可信 md 時保持
md-linked expression，缺校準時退回保守 notebook seed；guide prose 需描述這個 operator-facing
fallback policy，而不是臆測固定硬體值。

qubit 類 Pulse role 的 channel seed 依 setup alias fallback `qub_ch → qub_1_4_ch → qub_4_5_ch`
解析，最後才退回 `0`。Rabi 專用 drive pulse 預設採 ModuleLibrary `pi_len → pi_amp`，讓
`len_rabi` / `amp_rabi` 優先沿用已校準的 pi pulse，再由各 adapter 的 spec lock 覆寫 sweep-owned
欄位（`len_rabi` 覆寫 waveform length，`amp_rabi` 覆寫 gain）；library 缺項時才由
`blank_overrides`套用blank pulse的notebook fallback seed。一般 state-prep pi pulse role維持
`pi_amp → pi_len`。Pi/2 pulse role
只採 `pi2_amp → pi2_len`，不降級到 pi pulse role。

`bath_reset` role 的 `cavity_tone_cfg`、`qubit_tone_cfg`、`pi2_cfg` 都是 nested pulse refs：
前兩者預設為 custom pulse 並由 bath reset seed/md 填值，`pi2_cfg` 預設採 pi/2 pulse role
（`pi2_amp → pi2_len`）讓 reset tomography pulse 可直接引用既有校準 module。

`twotone/freq` 的 qubit-drive module 是 spectroscopy probe，不是 state-prep init pulse；
UI label 使用 `Probe Pulse`，欄位 key 仍維持 runtime/notebook contract 的 `qub_pulse`。

adapter-defined top-level knobs以generic scalar/`field` verb按GUI顯示順序宣告；它們可以是正式
ExpCfg欄位，也可以是run-only adapter欄位。正式欄位正常lower到
ExpCfg；run-only 欄位由 adapter 在 `build_exp_cfg()` 或 custom `run()` 內讀取後 pop 掉。
`onetone/freq` 的 `sampling_mode` 是正式 `FreqCfg` 欄位，GUI 維持既有 `sweep.freq`
結構。`homophasal.r_f` / `rf_w` / `theta0` 是正式校正 cfg 欄位，預設為
合法空值；可輸入 direct value 或 expression。linear 模式忽略合法校正值，
但兩種模式都不忽略 expression 解析錯誤。homophasal 模式在 device I/O 前
驗證必要欄位及正值條件，不從 live md 隱藏注入。
`twotone/time_domain/t1` 的 `uniform` 是會隨 RunRecord 保存的 typed cfg 欄位：預設 `True` 使用線性 delay
sweep；設為 `False` 時 adapter 仍保持同一個 cfg start/stop/expts 視窗，底層在硬體量化前
沿 normalized T1 decay curve 等弧長配置 delay。內部 lifetime model 不成為 GUI 欄位；cycle
conversion 保留點數與順序，格點 collision 直接提示擴大 span 或減少 points，而不靜默減點。

跨session狀態的少數default走`value_source(key, target_type, fallback?)` Seed，於definition
instantiate時透過`ctx.values` resolve once，成功後寫入普通direct value，不把lazy ref放進cfg
tree。role default table的同類入口是`Source(key, type_name?)` seed；只用於來源多變、
強型別 helper 反而會拉寬依賴的情境。device source 只發布具名 device 資訊，不推論 flux
語義；`twotone/flux_dep` 的 `dev.flux_dev` 以 `device.flux.name`
為預設（也就是名為 `flux` 的 registered device），無可用來源時 fallback 到 `flux_yoko`。

Role default characterization golden 跟隨 `ROLE_TABLE` 與 `make_default_value` 產出的完整 value tree；
調整 seed 或 default-value policy 時，更新 `tests/zcu_lab/v2/_support/measure/_role_default_golden.json`
才是契約變更的紀錄，不在 runtime builder 內保留舊 fixture 相容邏輯。

`BaseAdapter.load` 是 GUI load path 的 canonical result seam：預設建構 `exp_cls()` 並呼
`exp.load(Path(filepath))`，與 `BaseAdapter.save` 的 canonical persistence 對稱。需要 constructor
參數、非 canonical/manual save、grouped data，或需要額外 metadata 才能安全分析/writeback 的
adapter 必須 override `load()` 或讓預設路徑以明確 `NotImplementedError` fast-fail。
adapter 不提供 legacy 單檔案的轉換或 fallback；canonical `exp.load()` 拒絕的資料
直接回報原始錯誤。`BaseAdapter.load` 只讀取結果，不修改 tab cfg；GUI
`LoadService.load_result` 只從 loaded RunRecord.cfg 嘗試回填 tab 與 Config editor。
缺失或不可採用的 cfg 不撤回有效資料，tab cfg 保持不變。Adapters 用 RunRecord 配對 cfg 與純 Result，分析不讀 current cfg。OneTone／TwoTone FluxDep 的互動分析由 frontend 擁有。

`BaseAdapter` 在 class definition/import 時驗證 `AdapterCapabilities` 與 lifecycle method 是否
一致。`analysis=FIT` 必須實作 `analyze()` 且不得實作 interactive plugin hooks；
`analysis=INTERACTIVE` 必須實作 `make_interactive_plugin()` 與
`make_interactive_frontend()` 且不得實作 `analyze()`；`analysis=NONE` 不得實作
analyze hooks。兩個 flux_dep adapters 共用 `zcu_tools.gui.app.measure.flux_pick_plugin` 的 state/actions/commands，
以及 Qt-only `zcu_tools.gui.app.measure.flux_pick_frontend` 的 artists 與 preview；adapter 傳 FluxPickInputs 與初始數值，service 建立 session，
adapter 不持有分析 operation 或 writeback draft。`get_analyze_params()` 只在 analyze-params **無法全 default 建構**（有欄位無 default）
時才必須覆寫；params 每欄位都有 default（含 `NoAnalyzeParams`、及把常數折進欄位 default 的 adapter）時
沿用 base default（回 `params_cls()`）。`post_analysis=True` 僅允許搭配 primary FIT analyze，並必須實作
`get_post_analyze_params()` / `post_analyze()`；post-analysis 是第二層 CPU-only 探索/比較視圖，
可選地透過 `get_post_writeback_items()` 提出獨立 writeback。`validate_run_request` 不受
capabilities 控制：Base 提供 no-op default，Protocol/Base exact signature 與 registry
conformance tests 共同鎖定 framework mandatory surface。

`singleshot/ge` 的 primary analysis 只產生 fit 產物：operator可選 backend、histogram log scale、
T1 alignment、nullable shared length ratio 與 Initial State（ground / excited，預設 ground）；advanced population priors不進GUI。所選 backend擬合
centres、`ge_s`與initial populations，右側顯示IQ distribution，並經`get_writeback_items()`提出
`fid`、`ge_s`、`g_center`、`e_center`。它的零參數post-analysis是sole confusion路徑：使用primary
fit資料計算`ge_radius`與3×3 confusion matrix，顯示完整confusion diagnostic，並經
`get_post_writeback_items()`提出`ge_radius`、`confusion_matrix`；兩組proposal分屬不同pane的
opaque draft，adapter不接觸Writeback implementation。

`singleshot/len_rabi` 的analysis維持Figure-only summary，不展開額外GUI scalar diagnostics；Figure
上方呈現measured populations與joint-fit curves，左下呈現第一點integrated-bin histogram fit，右下
呈現derived confusion matrix。完整typed numeric fit留在adapter result內。只有backend valid且
`g_center`、`e_center`、`ge_radius`、`confusion_matrix`四項全部finite時，adapter才同時提出四個
獨立`ParameterWriteback` items；任一項無效就全數略過。Adapter只投影同一次domain analysis結果，
不重跑fit、不重算stability，也不直接apply proposal。

`singleshot/amp_rabi` 與 `singleshot/reset_check` 的 `g_center`、`e_center`、`radius` 是正式 experiment cfg
欄位，預設 expression 指向 md 的 `g_center`、`e_center`、`ge_radius`。
缺值保持 invalid；operator 可改用 direct complex／float 值。Run 凍結已解析的
cfg 校正值，沿用 BaseAdapter.run；RunRecord.cfg 保留本次使用的值。
兩者都沿用可調整的 Reps／Rounds；Amp Rabi 串接各 round 的 raw IQ，reset-check 平均各 round 的 populations。

`singleshot/len_rabi`在analysis pane提供`decay: bool`，預設啟用衰減包絡；
此選擇不屬於量測cfg，不改變len Rabi的raw-IQ acquisition。
`singleshot/amp_rabi`共用Len Rabi的raw-IQ joint fit，固定無衰減／零相位，提供Initial State與相同的IQ校準writeback。

Adapter guide 是 prose，不是 machine contract。Guide prose 放在各 adapter 檔案內，避免
新增或刪除實驗時跨檔同步；adapter 以 local `guide_text` class var 提供內容，
GUI / MCP 只呼叫 `guide()`。

ro-optimize 與 reset 的 peak-picking adapter 以 `smooth_method` 提供 `wavelet` / `gaussian`
選擇，預設 `wavelet`；`smooth` 在 GUI 標為 smoothing strength，不再只代表 Gaussian sigma。
twotone `ro_optimize/length` 的 GUI analyze param 對外命名為 `duration_t0`，表示
`SNR/sqrt(length + t0)` 的 duration-normalization term，不是 penalty strength；
`None` / `0.0` 是純 SNR peak，較小的正值比大正值更偏好短 readout。

twotone `ro_optimize` adapters 的 readout spec 一律只接受 pulse readout；writeback
產生兩層 readout writeback：`best_ro_freq` / `best_ro_gain` / `best_ro_length`
仍是 parameter context scalar；當來源 RunRecord 的 `source.cfg` 非 None 且三個 best 值可由本次
analyze result 加上 current parameter context 補齊且皆為 finite number 時，adapter 同時提出
ModuleLibrary `readout_dpm`，並以 writeback `role_id="readout_dpm"` 標示這個
readout role proposal；缺值或 non-finite 值只略過 module writeback。
`readout_dpm` 以 `source.cfg.modules.readout` 作為 template，將 readout
pulse/readout 頻率設為 `best_ro_freq`、pulse gain 設為 `best_ro_gain`、pulse
waveform length 設為 `best_ro_length + READOUT_DPM_PULSE_TAIL_US`（0.1 us tail）、
ADC readout length 設為 `best_ro_length`。

`onetone/freq` analyze 仍寫回 parameter context `r_f` / `rf_w` / `theta0`；當來源 RunRecord 的
`source.cfg` 非 None 且 `source.cfg.modules.readout` 是 pulse readout 時，adapter 也提出
ModuleLibrary `readout_rf`，並以 writeback `role_id="readout"` 標示它是 Pulse
readout role 的 proposal；`readout_rf` 是 target name，不是新的 role id。
`readout_rf` 以該 snapshot readout 作為 template，只用 fitted `r_f` 覆寫
`pulse_cfg.freq` / `ro_cfg.ro_freq`，gain、waveform length/style、channels、ADC
readout length、trigger timing 等欄位都沿用 snapshot。`source.cfg is None` 或
readout 不是 pulse readout 時，module writeback graceful skip，只保留 parameter items。

`onetone/freq` 的 `fit_bg_amp_slope` 預設開啟，`fit_bg_phase_curvature` 預設關閉；
兩者是獨立選項。fake one-tone 的兩個選項預設都關閉。只有啟用的項才進入
raw-complex refinement；未啟用項固定為零。若 Hanger fit 的 `Qi=None`，consumer 依
`qi_status` 呈現診斷，不將它當成數字，也不因 derived Qi 不可解釋就否決整個 fit。

`onetone/freq` 的 electrical-delay analyze policy 提供 `auto` / `calibrated` / `manual`。
`auto` 優先讀取 parameter context 中與本次 pulse generator/readout channel 相符的
`res_edelay_calibration` prior，缺少時以 bounded adaptive global search 建立 branch；`calibrated`
要求相符 prior，`manual` 要求 finite seed。prior/manual 都只固定 branch，後續仍作 local
refinement。只有實際參與 fitting（移除首尾點後）的 nonuniform grid 能自行辨識 absolute
branch；uniform grid 若沒有可信 seed，不會把 local alias 提案寫成 calibration。可持久化
的結果以單一 compound parameter writeback item
`res_edelay_calibration = {edelay, res_ch, ro_ch}` 保存，讓 delay 與 route identity 的選取
all-or-none。只有可辨識 pulse route 的 run snapshot，且有可信 seed 或 fitting grid 為 nonuniform，
才提出此 writeback。沒有 cfg snapshot 的舊資料仍可分析，不提出無 route 的 calibration。
worker 提出 proposal，`ContextService` 在 preview/apply 後負責 parameter context 寫入。
GUI 的 maximum search radius 預設為 `100.0`（MHz 軸時單位為 us）；若已明確固定
`edelay`，不重新搜尋 branch。

---

## Operator workflow guide 約束

`run-measure-gui` skill 與各 adapter guide 描述 operator 流程時，把 mock / real
量測都當黑盒：不假設已知 flux 位置，不用 simulator truth、FakeDevice 內部真值或
predictor 輸出取代掃描與看圖判讀。

標準 bring-up 順序是：`lookback` → `onetone/freq` → `onetone/flux_dep` 找
`flx_int` / period（視 2D map 判讀）→ 將 flux device 移到 `flx_int` →
在該 flux 重跑 `onetone/freq` → `twotone/freq` 寬掃 → `twotone/freq` 窄掃 →
agent / user 審核 figure 與 writeback preview → 後續 Rabi / T1 / T2。

`onetone/freq` 第一次通常用 `linear` sampling 建立 `r_f` / `rf_w` / `theta0`，並在有
pulse-readout snapshot 時提出 `readout_rf`；這些 writeback 齊全後，可切到 `homophasal`
sampling，讓同一個 start/stop/expts 掃描在 resonator circle phase 上等距。

`twotone/flux_dep` 是 readout 與 `q_f` 可信後的後續 qubit model mapping，不是早期找
flux 的工具；readout/qubit 參數還沒處理好時通常看不到可用 arc。Writeback 責任留給
agent / human 判讀，guide 不暗示用自動 fidelity gate 代替判斷。

---

## JPA 校準 family（`jpa/`）

JPA（Josephson parametric amplifier）校準由六個 adapter 構成單一可發現的
family：startup catalog（`../definitions.py` 的 `ADAPTERS`）依 bring-up 順序註冊，generic
tab creation 與 remote adapter listing（`view.adapter_list`）因此直接列出全部六個名稱：

| 順序 | adapter | 角色 |
| --- | --- | --- |
| 1 | `jpa/freq` | 找 JPA pump frequency（FIT analysis，寫回 `best_jpa_freq`） |
| 2 | `jpa/flux` | 找 JPA flux sweet spot（FIT analysis，寫回 `best_jpa_flux`） |
| 3 | `jpa/power` | 找 JPA pump power（FIT analysis，寫回 `best_jpa_power`） |
| 4 | `jpa/auto_optimize` | 聯合最佳化 flux/freq/power（FIT analysis，一次提出三個 `best_jpa_*`） |
| 5 | `jpa/flux_onetone` | flux × readout frequency 2D survey（`analysis=NONE`，無 writeback） |
| 6 | `jpa/check` | pump off/on 比較診斷（figure-only analysis，無 writeback） |

共同 mechanics 在 `_support/measure/jpa_shared.py`：RF 選擇使用 `jpa_rf_dev` role
（`set_freq` / `set_power` / `set_output` knob 能力檢查），flux 選擇使用
`jpa_flux_dev` role（`set_flux` knob）；lowering 與 preflight 只作用於 cached
device snapshot，任何硬體工作前 fast-fail。六個 adapter 都顯式暴露 `reps`、
`rounds`、`relax_delay`，fresh defaults 沿用 notebook bring-up workflow：
`freq` / `flux` / `power` 是 `10000 / 1 / 0.5 us`，`auto_optimize` 是
`1000 / 1 / 30.5 us`，`flux_onetone` 是 `100 / 10 / 0.1 us`，`check` 是
`1000 / 5 / 0.5 us`；`initial_delay` 維持 GUI 隱藏並使用 core 的 `1.0 us`
default。這些是可修改的操作 seed，不是跨 setup 的最佳 SNR 或安全保證。

`auto_optimize` 的三軸 `start` / `stop` 是 search bounds；各 sweep 的 `expts`
是會影響 phase-1 flux-grid 與 per-slice budget allocation 的 relative resolution
hints，不是 Cartesian sample count，總 measurement budget 仍由 `num_points` 表達。
JPA flux 在六個 adapter 一律以中性 device value／`a.u.` 語彙呈現（sweep label、
liveplot 與 canonical persistence 一致，無 `1e3` 縮放或 physical-unit 宣稱）。
capabilities 一致：四個校準 adapter 是 `analysis=FIT` 並提出
`ParameterWriteback`，survey/diagnostic 兩個 adapter 不提供 writeback。每個 adapter
的 operator guide 都以現在式說明 seeded bounds 是 bring-up defaults 而非 certified
safety limits，operator 必須先 review device 與 sweep 再 run；`jpa/check` 特別警告
run 結束時 pump output 保持 ON。真實硬體 acceptance 由 operator 在 device/sweep
review 後執行，不屬於 adapter code。

---

## Reset 校準實驗群（`twotone/reset/`）

對應 notebook `single_qubit.md` 的三種 reset 校準流程。使用者可以重複或不按順序執行。
五個參數實驗把本次分析值累積到 qubit ext。只有指定 finisher 提出模組。

| 流程 | 校準 adapter | Finisher | 固定 library root |
| --- | --- | --- | --- |
| single-tone sideband | `freq`、`length` | `freq` | `$qubit.reset.reset_10` |
| dual-tone | `freq`、`power`、`length` | `power` | `$qubit.reset.reset_120` |
| bath cavity-assisted | `freq_gain`、`length`、`phase` | `phase` | `$qubit.reset.reset_bath` / `reset_bath_e` |
| 共用驗證 | `check`，RabiCheck，contrast / residual analysis | 無 | 無 |

`reset/check`的analysis以reset前分支決定共同IQ投影與Rabi基頻，擬合三條gain sweep
分支；最後一條包含二次諧波。摘要提供前後半峰對峰振幅、相對contrast、相位差、
reset-only殘餘振盪與offset、二次諧波振幅、各分支residual RMS，圖中保留原始資料、
擬合線與殘差。無法解析的相位為`None`；頻率無法解析或資料不足時明確失敗。
這些平均IQ指標不等於reset fidelity，也不唯一識別coherence或population機制；不提出
writeback。cfg只設定一個`rabi_pulse`，兩次pulse使用相同波形與gain。

`singleshot/reset_check`使用硬體gain／branch sweep，保存G/E populations；
analysis可讀取confusion_matrix修正，回報reset-only平均／最大excited及最大Other，
不提供IQ校準writeback。Other不是校準後leakage，reset population不是reset-channel fidelity。

`singleshot/amp_rabi`使用硬體gain sweep保存raw IQ。Reps是每gain每round的shots，
Rounds的shots串接保存；不使用Shots或Shots per batch。本次resolved cfg的g_center、e_center、
radius凍結在snapshot中供live分類，analysis與Len Rabi共用joint fit及三面板診斷圖，重新估計
IQ centers與confusion matrix。Amp固定無衰減／零相位，提供Initial State，保留振幅、
頻率與pi/pi2 gain摘要；四項校準writeback與Len Rabi共用有效性檢查。
舊population-only檔案不符合raw-IQ axes契約，無法據此重建shots或執行joint fit。

### cfg seed 與純量累積

Run cfg 建立時，七個 default readers 從固定 capture 的 `$qubit.ext.*` 降低 seed。
SingleToneFreq 用 reset_f ±1.5 reset_fwhm 建立 201 點頻率窗。SingleToneLength 用
reset_f。DualToneFreq 用 reset_gain1/2 和 reset_f1/2，兩個頻率軸各為中心 ±50 MHz、
201 點。DualToneLength 用兩頻率及兩 gain，DualTonePower 用兩頻率。
BathLength 與 BathPhase 用 reset_bath_freq/gain。BathFreqGain 保留 resonator.freq 與
qubit.ext.rabi_freq 的原 seed。Missing、null 或 invalid scalar 沿既有 lowering 失敗，
不代用 fallback 數字。

本次分析值直接形成 `ParameterWriteback`。五個 caller 保留以下順序：

| Caller | `$qubit.ext.*` targets |
| --- | --- |
| SingleToneFreq | `reset_f`、`reset_fwhm` |
| DualToneFreq | `reset_f1`、`reset_f2` |
| DualTonePower | `reset_gain1`、`reset_gain2` |
| BathFreqGain | `reset_bath_gain`、`reset_bath_freq` |
| BathPhase | `reset_bath_max_phase`、`reset_bath_min_phase` |

### Run-tested finisher

SingleToneFreq、DualTonePower 與 BathPhase 使用該 Run 的完整
`source.cfg.modules.tested_reset`。`module_cfg_to_value` 建立隔離的 Spec/Value，adapter
只替換本次分析欄位，再提出 `ModuleWriteback`。不讀 ext、舊 module 或目前 point/alias
補欄位。手動修改後實際測過的頻率、gain、波形與其他 schema 欄位都保留。

- SingleToneFreq 只換 `pulse_cfg.freq`。FWHM 只寫 ext，不是 reset module 欄位。
- DualTonePower 只換 `pulse1_cfg.gain` 和 `pulse2_cfg.gain`，保留實測兩頻率。
- BathPhase 分別 materialize ground/excited 兩棵獨立樹，只換 `pi2_cfg.phase`。
  Ground 用 max_phase，excited 用 min_phase；兩者保留實測 cavity frequency/gain。

Module targets 使用表中的固定 roots，不經方案 slot 選目的地。已有 root 更新，缺則建立。
Finisher 在 module rows 後提出 `$qubit.module.reset` 的 `ParameterWriteback`。
值由同份 capture 的 `resolve_path` 轉為具體 root，例如 `Q1.reset.reset_10`。
Bath 預設 ground，使用者可以在 draft 中改選 excited。使用者也可以只 Apply library rows，
不選 slot row。逐列 Apply 沿既有 accepted 與成功前綴政策，不新增聯動選取或跨檔交易。

### 實驗責任與不足契約

三個 length adapter 不擬合參數，返回合法空 proposal collection。它們不提出
cfg-only module。DualToneFreq 與 BathFreqGain 只提出純量。框架不提供累積或齊全檢查。
完整合法 tested_reset 不因 ext 缺少而略過 module。公開 nullable cfg/capture 為 None
時，finisher 只返回純量，不讀目前全域來源。Source、alias、lowering、I/O 與 Apply
錯誤仍傳播，不轉成空 proposal 或原因列。

保留 downstream 的 reset/readout/x180/x90/probe slots。Reset 校準不讀
sideband/dual_tone/bath_g/bath_e 方案 slots。Migration profile 將舊 reset 純量映到上述
ext 名稱，resetf_w 映到 reset_fwhm；舊 reset modules 映到四個固定 roots，不猜
module.reset 的目前方案。Mapping revisions 為 fluxonium 1.6、transmon 1.7。

Bath `freq_gain` 保留 single-role 3D HDF5。Tomography phase 是同一 Result 的內部
sweep axis，不拆成四個 phase-resolved sidecar，也不改 Save 政策。

三個 singleshot 分析（ge / len_rabi / amp_rabi）皆提供 `Initial State`，表示 probe / swept drive pulse 之前的主要狀態；Rabi 描述 pulse 前狀態，不是第一個掃描點；啟用 phase offset 時也不等同於模型外推的零 length population。此參數只影響分析，不改量測 cfg 或 raw-IQ persistence。GE primary result 保存使用的初態，Post-Analysis 的 radius、confusion matrix 與繪圖均沿用該 snapshot，不讀取尚未重新分析的表單值。

GE 的主分析與 writeback/post 邊界拒絕非有限 centers/width、重合 centers 與不合法 populations。Optimizer 未收斂不會回傳 initial guess 當作 calibration；失敗不產生新的校準 proposal。


`twotone/rabi/len_rabi` 與 `singleshot/len_rabi` 的 analysis pane 都提供
`Fit phase offset`（`fit_phase: bool = False`）。一般 Len Rabi 啟用後自由擬合相位；
singleshot 在所選 Initial State 方向附近擬合 ±90° offset，結果圖顯示相位。
兩者都可獨立切換 decay；此參數只影響 analysis，不改 acquisition 或 raw data 格式。
一般 Len Rabi 的預設由原本自由 phase 改為固定 0°/180°，校準長度 writeback 沿用
本次模型的結果。Singleshot 保持 Figure-only summary，typed fit result 另提供
`phase` 與 `zero_length_populations`；四項 IQ calibration writeback 仍須通過既有
backend validity 與 finite calibration gate。

`t2ramsey`與`t2echo`的analysis pane提供`Fit phase offset (fringe only)`
（`fit_phase: bool = False`）。開啟時自由擬合fringe phase，關閉保留固定相位；
Ramsey停用Fit fringe或Echo選擇decay時，此選項不影響分析。既有T2與detune相關
writeback沿用本次分析結果，量測設定與資料格式不變。
