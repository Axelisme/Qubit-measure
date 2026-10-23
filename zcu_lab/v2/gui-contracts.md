# Measure GUI contracts

**Last updated:** 2026-10-24

**讀者：** lab-author、framework-dev、agent。查閱 concrete adapter 的來源、分析、writeback 與 capability 契約。

使用者入口見 [tutorial](../tutorial/README.md)。

## 目錄

- [Ownership](#ownership)
- [共用 lifecycle hooks](#共用-lifecycle-hooks)
- [Mux OneTone](#mux-onetone)
- [OneTone、TwoTone 與 Lookback](#onetonetwotone-與-lookback)
- [Reset adapters](#reset-adapters)
- [Singleshot 與 Readout optimize](#singleshot-與-readout-optimize)
- [JPA adapters](#jpa-adapters)
- [分析輸出與 Form cfg](#分析輸出與-form-cfg)
- [Capability 宣告與實作](#capability-宣告與實作)
- [修改原則](#修改原則)

每個實驗的 `core.py` 定義 cfg 與 FieldMeta，`form.py` 提供 `make_form(ctx)`。`gui.py` 接上 Form factory，擁有 run/analyze/writeback policy 與 operator guide 宣告。修改一個實驗時，先讀這三個檔案。共用框架實作位於 `zcu_tools.gui.app.measure.adapter.base`。

Post-analysis adapters 以 `get_post_writeback_items()` 提出 post-owned proposal；framework 將
primary/post 兩組 proposal 放入不同 opaque draft，adapter 不接觸 Writeback 實作。

提供 `SampleWriteback` 的 adapter 宣告元件來源路徑，單 qubit 用
`sample_device_alias = "$qubit.flux_source"`，mux 用 `"$mux.flux_source"`。
Framework 只依宣告從固定 Run／Load capture 與 device snapshot 補入座標。
未宣告時不提供 Sample writeback，Parameter／Module proposals 不受影響。

## Ownership

T1、T2Echo、T2Ramsey、AmpRabi、LenRabi 與 GE 的 summary 包含 core-owned fit_quality。共用 mechanics 只投影數值與具體 summary 路徑，不重算品質。GUI 與 MCP 讀同一份結果，低 r2 不增加 accept 門檻。GE 保留 joint／ground／excited 身分，post 不產生新 fit。

- `zcu_tools.gui.app.measure.adapter.base.BaseAdapter` 擁有共用的 framework implementation，不含特定實驗 policy。此 package 只提供 concrete adapters 與 domain support。
- `lookback/`、`onetone/`、`mux/`、`twotone/`、`singleshot/`、`jpa/` 擁有 concrete experiment definitions。Leaf 的 `gui.py` 對應 `core.py`。專用 helper 留在 leaf；同群組共用的 helper 留在 family。
  `jpa/` 的六個 adapter（`freq` / `flux` / `power` / `auto_optimize` /
  `flux_onetone` / `check`）是單一可發現的 JPA 校準 family，依 bring-up 順序
  註冊於 `../definitions.py`。Concrete adapters 各自擁有 acquisition 初值政策，
  並共同暴露 `reps` / `rounds` / `relax_delay`。四個 qubit SNR adapter 的
  `relax_delay` 採 `1*T1`，透過 `_support/measure/ctx_helpers.py` 的
  `t1_initial_time` 共用固定 capture 的來源分類 mechanics。有效 T1 用捕捉校準，
  缺值用 `30 us` 公允時間尺度，無效值留空，來源故障傳遞 owner 的原錯誤。
  初值可覆寫，不寫回校準。純反射的 `flux_onetone` / `check` 等待維持
  `0.1 us` / `0.5 us`，六個 adapter 的 `reps` / `rounds` 不變；
  `initial_delay` 維持 core-owned hidden default。auto optimizer 的 sweep `expts`
  是 allocation resolution hint，flux 則使用中性 device-value contract。完整家族
  契約見 `measure-authoring.md`。
- `_support/measure/` 是 private package，提供至少被兩個 concrete adapters 共用的 mechanics。
  已定案且供多個 Form 共用的初值政策也可集中在具名 domain helper，目前限 T1 時間尺度選源與 Rabi 的來源、有效性及首次初值。
  各 Form 仍明確選擇倍率或場景，並組合表單結構。校準政策不進入 generic framework，也不擴及 writeback 或實驗順序。
  本 package 不擁有 registry order，也不 import concrete adapter。
- `_support/resonator.py` 不依賴 GUI，供 onetone/freq core 與 mux adapter 共用模型／畫圖 mechanics。Selection、delay mode 與 writeback policy 留 caller，不讓 core 經 `_support/measure/` 載入 GUI。
- `../definitions.py` 明確列出可重載的 adapter catalog；`../components/` 擁有 startup-only component schema 註冊。
- Reload experiments 會重建 concrete adapters 及 family helpers，但保留 framework `BaseAdapter` 與 `_support/measure/`。
  修改這些共用基礎層需重啟 app；concrete module import 不得有硬體或背景工作副作用。

## 共用 lifecycle hooks

下表是 lab 作者實作或沿用的入口。完整簽名與 opaque result 契約見
[`ExpAdapterProtocol`](../../lib/zcu_tools/gui/app/measure/adapter/protocol.py)，
預設委派見 [`BaseAdapter`](../../lib/zcu_tools/gui/app/measure/adapter/base.py)。
RunRecord 是本套件 concrete adapters 採用的結果形式，framework 不從 opaque handle 猜 cfg 或資料。

| 入口 | 來源與結果契約 |
| --- | --- |
| `make_form(ctx)` | 回傳 `Form(core_cfg, tree, ...)`。Open／Reset 用固定 FormContext 建 fresh input；Refresh 也重建定義，但保留輸入與已解析初值。Framework 綁定範本並管理可編輯狀態。 |
| `validate_run_request(req, cfg)` | 選填的純 preflight，Guard 在開 operation 前必呼。cfg 已由 Form 驗證；hook 不碰裝置、不改 cfg 或 state。 |
| `run(req, cfg, *, context)` | 使用 Guard 接受並隔離的 Form 輸出。Base 原樣交給 core；需 RunRecord 的 adapter 自行配對同次 cfg 與純 Result。RunService 提供單次 RunContext，adapter 與核心共用 devices、plots 和取消訊號。 |
| `analyze(req, *, plots)` | FIT 讀取 req.run_result、analyze_params 與固定 Run／Load parameters。具名圖交給本次 Plots，數值結果不含 Figure。`post_analyze` 另接已採用的 primary result，不使用未重新分析的表單。 |
| `restore(stored)` | LoadService 先讀 native data、核對 experiment_tag，再交 StoredRun。Base 委派 `exp_cls().from_stored(stored)`，回傳含 record 與 cfg_issue 的 RestoredRun。Hook 不讀檔、不讀目前表單或重捕歷史。有效資料可在 typed cfg 不相容時保留，缺 cfg envelope 或壞資料仍拒絕。 |
| `data_item(key, result, *, comment)`、`analysis_item(key, result, parameters, *, pane)` | Save 在 GUI owner thread 同步要求內容。Base 從明確 RunRecord/spec 準備 DataItem，從已提交輸出與歷史分析輸入準備 AnalysisItem。Hook 不選路徑、不做 I/O、不讀 live context。缺 cfg 不以表單補值，準備失敗只使該 data output 失敗，獨立圖與分析仍可保存。 |

[`LoadService`](../../lib/zcu_tools/gui/app/measure/services/load.py) 保留目前表單與其固定來源，
不從 loaded record.cfg 回填。分析與 WB 的來源另依檔案 history 固定；同 entry 有 point 時讀取
該歷史 point 的現存持久值，跨 entry 或無 point 時為 None，不回退目前 aliases。
保存路徑、格式與檔案寫入由 framework 的
[`ArtifactHandover`](../../lib/zcu_tools/gui/app/measure/artifacts/handover.py) caller 與
[experiment artifacts owner](../../lib/zcu_tools/experiment/README.md#實驗介面與資料) 負責。

## Mux OneTone

MuxGroup 的必填 resonators 保存有序 resonator 名稱。選填 qubits 由 model 檢查等長與重複 qubit。提供時，Mux OneTone 的 Form 在 Open／Reset 以固定 ParameterContext 逐項檢查 component 存在、qubit／resonator kind 和 qubit.resonator 對應，缺失或 null 同樣拒絕。錯誤指名欄位、index 與實際值，不另加 framework 外部反紅機制。無 qubits 時不套這些跨 component 檢查，也不額外拒絕重複 resonator。

Mux OneTone 的 Readout 初值取 `$mux.module.readout`，沒有來源可手填有效 readout。participants 須非空、唯一且為完整 readout 範圍內的整數，不接受 bool、float 或字串。Open／Reset 依同次 capture 的 `$mux.resonators` 將 `$participants` 名稱轉成 indices，缺列、未綁或空清單選全部。外群名稱保留為可編輯的 invalid draft，整數錯誤標紅並拒絕 cfg。非空清單的未綁元素拒絕 Open／Reset，不 fallback。其他無效 selection 沿宿主 result／Run 診斷定位 participants 並拒絕 Run。Lab 按輸入順序投影 subset 交 program，RunRecord 保存完整 readout、labels 與 selection，不另存 resonators 或 group identity。Live 圖有 K 條線，legend 由 plotting owner callback 設定；原始 tone_indices 與 signals columns 同序，不重新編號。Capability 為 FIT，所有 active tones 共用 onetone/freq 的四欄設定：`model_type`、`fit_bg_amp_slope`、`fit_bg_phase_curvature`、`edelay`，預設依序為 `hm`、`True`、`False`、留空。FIT 核對保存的 cfg、selection、原始座標與資料 shape，每個 column 使用原始 base frequency 加 nominal dfs 並去首尾點。`fit_i` 用原始序號命名，重複 label 不覆蓋圖。每個 active 的 typed outcome 保留成功 scalars 或失敗原因；inactive 不參與分析。缺 cfg 拒絕 FIT，不借目前表單。
Module proposal 從完整 run readout 建立，只換成功 active 的 freq；失敗和 inactive 保留原值，全失敗不提 module。Target 固定 `$mux.readout.readout_rf`。
有 Run／Load capture 時，模組列後新增 `$mux.module.readout` ParameterWriteback，值為同份 capture 解析的 module root。
Apply 模組列與 slot 列後，slot 指向該 library root，下次開表單沿既有 slot 來源讀取。
Capture 為 None 時保留 module 與原有 scalar 提案。
Adapter 從 Run／Load capture 公開 lookup 取得 `$mux.resonators`，長度須等於完整 N，不是 active K。成功 tone i 以 resonators[i] 提 freq、kappa、ext.theta0 和可 persist 的 route-qualified ext.edelay，每個參數獨立一列。Resonators 長度不符時，保留 readout module 與 capture 可解析的 readout slot 提案，省略逐 resonator scalar，summary 顯示原因。Alias／來源故障仍傳播，不以長度不符掩蓋。
Prior 同樣只查 capture 的 resonator.ext.edelay，按原始 tone 的 generator／ADC route 比對。`edelay` 單位為 us，留空時每個 active tone 各自使用有限的 route-matched prior 作 branch seed；沒有匹配時，該 tone 使用內部固定 100 us 上限作 global 搜尋。明填有限數字時，所有 active tones 共用此 branch seed，0 與負數也有效。Seed 是擬合起點，不是固定 delay。非有限或非數字的 `edelay` 輸入會拒絕整次 FIT。Delay 只有 seeded 或非等距 grid 才可 persist，沿單 tone 政策。無 capture 仍可純計算，非空 Apply 沿既有拒絕。Load 使用同 entry 檔案 history 的 point／roles capture，不讀當前 aliases。Duplicate resonators 不去重，由既有 Apply 的重複 destination 驗證處理。

## OneTone、TwoTone 與 Lookback

OneTone Freq 提供 route-qualified delay calibration 與 readout writeback。
有效 pulse-readout module 提案後，有 capture 時新增 `$qubit.module.readout` ParameterWriteback，值為同份 capture 將 `$resonator.readout.readout_rf` 解析的具體 root。
Apply 模組列與 slot 列後，qubit 的 readout slot 指向該 library root。
Capture 為 None 時保留原有 scalar 與 module 提案，alias／來源故障沿 owner 契約傳播。
PowerDep 把 SNR stop control 放進 typed cfg，不提供 analysis。
OneTone FluxDep 的 run 將同次 cfg 與純 Result 配成 RunRecord，interactive plugin 明確取該 source 的資料並捕捉成唯讀 inputs。
GE 的 FIT／post 分別發布 `fit`／`post` 具名圖，post 使用已採用的 primary FIT；
OneTone FluxDep 用具名 2D `measurement` liveplot。互動 Done 從 committed state 呼叫 Qt-free kernel 與原生圖 builder，產生 GUI-owned 數值結果與 `pick` 圖。Qt 畫布只負責預覽。TwoTone FluxDep 的 run 也回傳 RunRecord，plugin 從 explicit source 捕捉 inputs，保留 phase 投影並共用這條終止 renderer。TwoTone Freq 的 FIT 回傳頻率與線寬及其誤差，fit 圖另交 Plots；TwoTone PowerDep 只提供 run 與 canonical records，不提供 analysis。Lookback FIT 只輸出 predict_offset scalar，具名 fit 另由 Plots 發布；GUI ratio0.1／smooth1.0 與 timeFly writeback 不變。其餘 adapters 同樣使用 explicit records，framework 不提供 pyplot 或舊簽名 fallback。
TwoTone AmpRabi／LenRabi 的 run 回傳 RunRecord，FIT 以 explicit source 與 typed options 分析，結果包含 scalar 與 core fit_quality，`fit` 圖交給 Plots。Gain／length／Rabi frequency 的 scalar writeback 不依賴 cfg；校準 pulse module writeback 只從該 source.cfg 複製 qub_pulse，缺 cfg 時略過 module items。
T2Echo／T2Ramsey 在 `form.py` 的 `lower_detune` 將 detune_ratio 降為 cfg.detune，run 配對 cfg／Result，FIT 回傳 scalar 與 core fit_quality，並發布具名 fit。Ramsey q_f writeback 必須具有來源 cfg、實際 detune 與已提交的 fringe fit；canonical load 缺實際 detune 時只提供 t2r。
CKP 的 run 配對 cfg 與純 Result 為 RunRecord，兩張 measurement 熱圖分別呈現 ground／excited。FIT 只提交 chi／kappa／res_freq scalar與具名 fit 圖，保留 chi／rf_w／readout_f writeback。

## Reset adapters

Reset adapters 將同次 cfg／Result 配成 RunRecord，FIT 使用 explicit source 與本次 Plots。
SingleToneFreq、DualToneFreq、DualTonePower、BathFreqGain 與 BathPhase 將本次分析純量
依原順序寫到 `$qubit.ext.*`。七個 default readers 從固定 capture 的 ext 建立 tested_reset
或 sweep，missing/null/invalid 沿既有 lowering 失敗。BathFreqGain 保留 resonator/rabi seed。
SingleToneFreq 只換本次 tested_reset 的 frequency，DualTonePower 只換兩 gain，BathPhase
獨立複製兩 variant，只換 max/ground 與 min/excited phase。其他完整 schema 欄位保留，
不以 ext 或舊 library module 覆蓋實測 cfg，也不加缺 ext gate。
三個 finisher 提出 `$qubit.reset.reset_10`、`reset_120` 或 `reset_bath`/`reset_bath_e`
固定 roots，以及可選的 `$qubit.module.reset` row。Slot 值是同份 capture 解析的具體 root，
bath 預設 ground。Cfg 或 capture 為 None 時只保留純量，source/alias/materialization
錯誤仍傳播。三個 length adapter 無 fit，返回空 proposal collection，不提 cfg-only module。
DualToneFreq 與 BathFreqGain 不提 module。逐列 Apply 與 ledger 失敗政策由 framework owner
維持，不加聯動或跨檔交易。Dual Freq 的 adapter `run` 以 `cfg.with_updates(method="hard")`
建立實際執行的 cfg，交 core 並存入同次 RunRecord，不由 run kwargs 隱藏。

## Singleshot 與 Readout optimize

Singleshot Amp／Len Rabi、Check、ResetCheck、AC Stark 的 run 回傳 RunRecord，FIT 使用 explicit source／typed options／Plots，不在 GUI 數值結果保存 Figure。Rabi 保留完整 numeric fit 供共同 calibration writeback 判斷；ResetCheck 只投影 populations summary，不推導 reset fidelity。Check 的 GE centers 與 AC Stark 的 chi／kappa 仍由分析當次 md 讀取，再交 core options。
Singleshot T1／Tone／Tone sweep adapters 直接組裝含 uniform 的 typed cfg，將同次 cfg／Result 配成 RunRecord。FIT 向本次 Plots 發布 fit 圖；Tone 保留 t1_with_tone writeback，其餘兩者只提交空數值結果。Correction／photon-axis 參數仍從分析當次 md 讀取。
Singleshot MIST Freq／Power／FreqPower adapters 將本次校準 cfg 與純 Result 配成 RunRecord。FIT 使用 explicit source 與 typed options，只提交空數值結果，fit 圖另交 Plots。Confusion matrix 仍讀分析當次 md；只有 Power 使用 ac_stark_coeff／log_scale，FreqPower 不轉交無作用的參數。
Readout AutoOpt 將 num_points 保留在 typed cfg，run 配對 cfg／Result，FIT 只發布三個最佳 scalar 與具名 fit；readout_dpm 由 source.cfg 產生，缺 cfg 時只保留 scalar writeback。Native restore 與保存內容準備沿用核心的 grouped persistence spec。
Readout optimize Freq／FreqGain／Length／Power adapters 將 cfg／Result 配成 RunRecord，FIT 將表單 smoothing／duration／penalty 轉為 core options，只提交 best scalar 與獨立具名 fit。Writeback 保留 best_ro_* scalar；readout_dpm 以 source.cfg 的 pulse readout 配合當次 md 補足缺少的最佳值，cfg=None 時略過 module，不以目前表單補來源。

## JPA adapters

JPA Freq／Flux／Power／Check／OneToneFlux 的 run 配對 cfg／Result。四個 FIT 使用 explicit source 與本次 Plots，前三者只回傳最佳 scalar，Check 為空數值結果。Flux／Power 保留 GUI 圖軸重標政策與 best_jpa_* writeback，缺 cfg 也可離線分析；OneToneFlux 沒有 analysis 或 writeback。AutoOptimize 也以 RunRecord 傳遞 grouped 結果，num_points 進 typed cfg；FIT 保留 phase 色彩與 neutral flux label，三個 best_jpa_* writeback 只取該來源的最佳 scalar。

## 分析輸出與 Form cfg

`RunRequest` 提供 `soc`、`soccfg` 與固定的 `cfg_context`，後者是 library/assets 的
cfg materialization capability。None 只允許無來源依賴的 inline cfg。
Form 以同一份固定 FormEnv 求值、轉換並驗證 typed cfg，Guard 接受隔離副本後，
將它原樣交給 `validate_run_request(req, cfg)` 與 `run(req, cfg, *, context)`。
Framework 不另組 experiment cfg，也不重新 lower。Domain preflight 在 operation 與
硬體 I/O 前拒絕額外的不合法條件；失敗交給 Guard 的 admission error 轉譯。
RunContext 擁有 devices、SoC handles、plots 與 cancel_signal。Device snapshot 由 Run owner
在取得 hardware lease 後、呼叫 adapter.run 前捕捉，供當次 acquisition evidence 使用，
不是 RunRequest 的表單初值來源。Analyze 與 writeback 保持各自的固定來源契約。

插件自行定義 typed 成功輸出的欄位，決定需要保留的選項與重現資訊。
Framework 保存來源、插件輸出與圖，不要求完整 options 或保證可重現性。
`params` 保持表單輸入，互動 Done 不以終態 options 替換它。
GUI 與 remote 讀同一份已提交輸出；取消或失敗不覆蓋前次成功紀錄。

`form.py` 的 `make_form(ctx)` 以 `Form(core_cfg, tree, ...)` 集中表單結構、初值與來源選擇。
`core.py` 的 cfg annotations 與 FieldMeta 定義型別、驗證和欄位顯示資訊。
有 GUI 額外輸入時明列 extras 與具名 lower，`Form.to_cfg` 驗證 lower 的結果。
Generic model 生成、module materialization 與 typed cfg 轉換由
[`zcu_tools.gui.form_adapter`](../../lib/zcu_tools/gui/form_adapter/README.md) 擁有，
通用輸入狀態與驗證由 [`gui.form`](../../lib/zcu_tools/gui/form/README.md) 擁有。
Concrete Form 選擇 `_support/measure/` 的 domain helpers；實驗結構與初值政策仍由 lab 擁有，
不移入 generic framework。

## Capability 宣告與實作

每個 concrete `BaseAdapter` 以一個 `AdapterCapabilities` 宣告 `analysis`（NONE／FIT／INTERACTIVE）、`requires_soc`、`post_analysis`、`load_data`。宣告的是支援範圍，不是目前是否連線、檔案是否可讀或是否已取得 operation lease。實驗側提供符合宣告的 hooks；framework 的 interface 與呼叫端規則見 [measure app](../../lib/zcu_tools/gui/app/measure/README.md) 和 [GUI ADR](../../docs/adr/0067-gui-application.md)。具體實驗的 capability 值以其 adapter class 為準。

`BaseAdapter.__init_subclass__` 在定義 subclass 時檢查宣告與條件式 hooks，不延後到 UI 或 worker 第一次使用。FIT 要有 `analyze()`，INTERACTIVE 要有 `make_interactive_plugin()` 與 `make_interactive_frontend()`，NONE 不接受分析 hooks；非 INTERACTIVE 不接受這兩個 interactive factories。Post-analysis 只配 FIT，並要求其 params 與 analyze hooks。需要值才能建立的 analyze params 必須提供 `get_analyze_params()`，全有預設值則可繼承 base 實作。檢查 method override 時比較沿 MRO 解析後的實作與 `BaseAdapter`，所以中間 base 提供的 override 有效，不維護子類白名單。

`load_data=True` 須有 concrete `restore()`，或有可無參數建立且提供 callable `from_stored()` 的 `exp_cls`；`load_data=False` 不接受 concrete restore override。這是 import-time 宣告一致性檢查，不證明某個檔案能載入、hook 語意正確或操作安全。`validate_run_request()` 是 framework 必呼的 preflight；base 提供 no-op default，不因繼承 no-op 而新增 capability flag。Subclass 拼錯預期要 override 的 no-op hook 名稱，驗證無法辨識其意圖。精確錯誤與預設值以 `zcu_tools.gui.app.measure.adapter.base` 為準。

## 修改原則

1. 單一實驗的初值政策留在其 `form.py`；其他設定、文字與流程 policy 留在 authoritative adapter file。
2. helper 確實出現第二個 caller，且有清楚 mechanics seam 或符合上述共用初值政策邊界時，才移入 `_support/measure/`。
3. 不以 forwarding wrapper 隱藏單行邏輯；不讓 `_support` 解讀 registry key 或實驗順序。
4. 新增 adapter 時同步加入 `../definitions.py` 與 operator guide。實驗定義以直接 review、表單開啟 smoke 與離線合成資料保存／讀回走查驗證，不寫 lab 專屬 pytest，也不建立 `tests/zcu_lab/`。Framework owner 用最小合成 adapter 驗證公開契約，不 import `zcu_lab`；刪掉本套件後整套 pytest 仍須通過，包含 slow。

單一實驗的 range centre/span/fallback 直接寫在其 `form.py`。已定案且多個 Form 共用的初值來源、有效性與首次值，可由上述具名 domain helper 擁有。
writeback target/description/role 仍直接寫在 authoritative adapter；共用 writeback helper 只提供不含這些具體值的 parameterized mechanics。
