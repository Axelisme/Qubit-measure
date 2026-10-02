**Last updated:** 2026-10-02 (Readout AutoOpt records)

# measure experiment adapters

這個 package 是 measure-gui 實驗流程的使用者修改入口。每個 concrete adapter file
同時擁有該實驗的 cfg definition、run/analyze/writeback policy 與 operator guide；修改一個
實驗時，主要閱讀範圍應維持在該檔案及其直接對應的 `experiment/v2/` implementation。

Post-analysis adapters 以 `get_post_writeback_items()` 提出 post-owned proposal；framework 將
primary/post 兩組 proposal 放入不同 opaque draft，adapter 不接觸 Writeback 實作。

## Ownership

- `base.py` 擁有所有 adapter 共用的 framework implementation，不含特定實驗 policy。
- `lookback.py`、`onetone/`、`twotone/`、`singleshot/`、`jpa/`、`fake/` 是 concrete experiment
  definitions；同一實驗專用的 helper 就近放在該檔案或同群組的 `_shared.py`。
  `jpa/` 的六個 adapter（`freq` / `flux` / `power` / `auto_optimize` /
  `flux_onetone` / `check`）是單一可發現的 JPA 校準 family，依 bring-up 順序
  註冊於 `../registry.py`。六個 concrete adapters 各自擁有 notebook-derived
  acquisition defaults，並共同暴露 `reps` / `rounds` / `relax_delay`；
  `initial_delay` 維持 core-owned hidden default。auto optimizer 的 sweep `expts`
  是 allocation resolution hint，flux 則使用中性 device-value contract。完整家族
  契約見 `../README.md`。
- `_support/` 是 private package，只放至少被兩個 concrete adapters 共用的 mechanics；它
  不擁有 registry order，也不 import concrete adapter。
- `../registry.py` 明確列出可重載的 adapter catalog；`../role_registry.py` 擁有 startup-only role composition。
- Reload experiments 會重建 concrete adapters 及 family helpers，但保留 `base.py` 與 `_support/`。
  修改這些共用基礎層需重啟 app；concrete module import 不得有硬體或背景工作副作用。

已遷移adapter的`run(req, raw_cfg, *, context)`使用Guard凍結的resolved cfg，RunService 提供單次 RunContext，adapter 與核心共用其 devices、plots 和取消訊號。
`analyze(req, *, plots)`將圖寫入本次具名Plots，不把Figure塞入數值結果。
T1／GE／Lookback／OneTone Freq、PowerDep／FakeFrequency／Fake stub 的 GUI run 將同次 cfg 與純 Result 配成 RunRecord，analyze 使用該 source。GE post 另接已採用的 primary calibration，不使用未重新分析的表單。
BaseAdapter 的 load/save 直接傳遞核心 records；load 不用目前 cfg 補來源，save 將 explicit source 和 exact path 交給核心，override 可接受 cfg=None。
GUI LoadService 只從 loaded record.cfg 回填，缺失或不可採用時保留 Config/editor，不撤回有效 loaded data。
OneTone Freq 保 route-qualified delay calibration 與 readout writeback。PowerDep 把 SNR stop control 放進 typed cfg，不提供 analysis。
FakeFrequency core 與 adapter 留在 `fake/freq.py`。Core 重用 FreqExp 的資料分析，不讀 simulation truth，圖歸本次 Plots。RunRecord 保留 FakeFreqCfg；readout 使用可解碼的 concrete union。GUI 支援 canonical load，save 使用 exact path；persist_data=False 仍是明確 noop。
`fake/stub.py` 保留未註冊的 hardware-free GUI stub。Core 回傳純 data，GUI 建立 matching cfg／Result record並捕捉 detached device snapshot。Threshold 與 peak 分析由 core 擁有，GUI 投影 numeric peak 並提供 fake_peak writeback。Save 保持明確 noop，load 明確 unsupported，GUI load_data=False。不用固定 samples 捏造 canonical physical axis。
OneTone FluxDep 的 run 將同次 cfg 與純 Result 配成 RunRecord，interactive plugin 明確取該 source 的資料並捕捉成唯讀 inputs。
GE 的 FIT／post 分別發布 `fit`／`post` 具名圖，post 使用已採用的 primary FIT；
OneTone FluxDep 用具名 2D `measurement` liveplot。互動 Done 從 committed state 呼叫 Qt-free kernel 與原生圖 builder，產生 GUI-owned 數值結果與 `pick` 圖。Qt 畫布只負責預覽。TwoTone FluxDep 的 run 也回傳 RunRecord，plugin 從 explicit source 捕捉 inputs，保留 phase 投影並共用這條終止 renderer。TwoTone Freq 的 FIT 回傳頻率與線寬及其誤差，fit 圖另交 Plots；TwoTone PowerDep 只提供 run 與 canonical records，不提供 analysis。Lookback FIT 只輸出 predict_offset scalar，具名 fit 另由 Plots 發布；GUI ratio0.1／smooth1.0 與 timeFly writeback 不變。其餘 adapter 逐項遷移。舊簽名在過渡期可能報錯，framework 不提供 pyplot 或簽名 fallback。
TwoTone AmpRabi／LenRabi 的 run 回傳 RunRecord，FIT 以 explicit source 與 typed options 分析，結果只含 scalar，`fit` 圖交給 Plots。Gain／length／Rabi frequency 的 scalar writeback 不依賴 cfg；校準 pulse module writeback 只從該 source.cfg 複製 qub_pulse，缺 cfg 時略過 module items。
T2Echo／T2Ramsey 在 build_exp_cfg 將 detune_ratio 降為 cfg.detune，run 配對 cfg／Result，FIT 只回傳 scalar 並發布具名 fit。Ramsey q_f writeback 必須具有來源 cfg、實際 detune 與已提交的 fringe fit；canonical load 缺實際 detune 時只提供 t2r。
CKP 的 run 配對 cfg 與純 Result 為 RunRecord，兩張 measurement 熱圖分別呈現 ground／excited。FIT 只提交 chi／kappa／res_freq scalar與具名 fit 圖，保留 chi／rf_w／readout_f writeback。
Bath reset 的 FreqGain／Length／Phase adapter 將同次 cfg／Result 配成 RunRecord，FIT 使用 explicit source 與本次 Plots，僅提交 scalar 或空數值結果。三者的 reset_bath／reset_bath_e module writeback 使用來源 cfg，缺 cfg 時略過；md 欄位與 phase gating 不變。
Single／dual-tone reset 的五個 adapters 將同次 cfg／Result 配成 RunRecord，FIT 僅提交 scalar 或空結果，圖交本次 Plots。Dual Freq 的 GUI hard-sweep policy 在 build_exp_cfg 寫入 cfg.method，不由 run kwargs 隱藏。reset_10／reset_120 module writeback 使用來源 cfg，缺 cfg 時略過，既有 md gating 保留。
Singleshot Amp／Len Rabi、Check、ResetCheck、AC Stark 的 run 回傳 RunRecord，FIT 使用 explicit source／typed options／Plots，不在 GUI 數值結果保存 Figure。Rabi 保留完整 numeric fit 供共同 calibration writeback 判斷；ResetCheck 只投影 populations summary，不推導 reset fidelity。Check 的 GE centers 與 AC Stark 的 chi／kappa 仍由分析當次 md 讀取，再交 core options。
Singleshot T1／Tone／Tone sweep adapters 直接組裝含 uniform 的 typed cfg，將同次 cfg／Result 配成 RunRecord。FIT 向本次 Plots 發布 fit 圖；Tone 保留 t1_with_tone writeback，其餘兩者只提交空數值結果。Correction／photon-axis 參數仍從分析當次 md 讀取。
Singleshot MIST Freq／Power／FreqPower adapters 將本次校準 cfg 與純 Result 配成 RunRecord。FIT 使用 explicit source 與 typed options，只提交空數值結果，fit 圖另交 Plots。Confusion matrix 仍讀分析當次 md；只有 Power 使用 ac_stark_coeff／log_scale，FreqPower 不轉交無作用的參數。
Readout AutoOpt 將 num_points 保留在 typed cfg，run 配對 cfg／Result，FIT 只發布三個最佳 scalar 與具名 fit；readout_dpm 由 source.cfg 產生，缺 cfg 時只保留 scalar writeback。Canonical load/save 委派核心的 grouped record 入口。
Readout optimize Freq／FreqGain／Length／Power adapters 將 cfg／Result 配成 RunRecord，FIT 將表單 smoothing／duration／penalty 轉為 core options，只提交 best scalar 與獨立具名 fit。Writeback 保留 best_ro_* scalar；readout_dpm 以 source.cfg 的 pulse readout 配合當次 md 補足缺少的最佳值，cfg=None 時略過 module，不以目前表單補來源。
JPA Freq／Flux／Power／Check／OneToneFlux 的 run 配對 cfg／Result。四個 FIT 使用 explicit source 與本次 Plots，前三者只回傳最佳 scalar，Check 為空數值結果。Flux／Power 保留 GUI 圖軸重標政策與 best_jpa_* writeback，缺 cfg 也可離線分析；OneToneFlux 沒有 analysis 或 writeback。AutoOptimize grouped 路徑另行遷移。
`RunRequest`只提供SoC handles與detached device snapshot；Base assembler
以此snapshot和`ml=None`建立experiment cfg。自訂builder若委派Base，
須宣告 `ExpCfg_cls`；domain preflight 在硬體 I/O 前拒絕不合法的必要欄位。
Analyze 與 writeback 保持各自的 context 契約。

插件自行定義 typed 成功輸出的欄位，決定需要保留的選項與重現資訊。
Framework 保存來源、插件輸出與圖，不要求完整 options 或保證可重現性。
`params` 保持表單輸入，互動 Done 不以終態 options 替換它。
GUI 與 remote 讀同一份已提交輸出；取消或失敗不覆蓋前次成功紀錄。

`cfg_definition()` 使用 `_support` 提供的 measure-domain builder vocabulary，但結構與預設
policy 留在 concrete adapter，因此使用者不必跨 `spec` / `default_value` 兩個方法理解同一
份設定。generic Spec/Value assembly 由 `zcu_tools.gui.cfg` 擁有，不能搬回本 package。

## Capability 宣告與實作

每個 concrete `BaseAdapter` 以一個 `AdapterCapabilities` 宣告 `analysis`（NONE／FIT／INTERACTIVE）、`requires_soc`、`post_analysis`、`load_data`。宣告的是支援範圍，不是目前是否連線、檔案是否可讀或是否已取得 operation lease。實驗側提供符合宣告的 hooks；framework 的 interface 與呼叫端規則見 [measure app](../../../../gui/app/measure/README.md) 和 [GUI ADR](../../../../../../docs/adr/0067-gui-application.md)。具體實驗的 capability 值以其 adapter class 為準。

`BaseAdapter.__init_subclass__` 在定義 subclass 時檢查宣告與條件式 hooks，不延後到 UI 或 worker 第一次使用。FIT 要有 `analyze()`，INTERACTIVE 要有 `make_interactive_plugin()` 與 `make_interactive_frontend()`，NONE 不接受分析 hooks；非 INTERACTIVE 不接受這兩個 interactive factories。Post-analysis 只配 FIT，並要求其 params 與 analyze hooks。需要值才能建立的 analyze params 必須提供 `get_analyze_params()`，全有預設值則可繼承 base 實作。檢查 method override 時比較沿 MRO 解析後的實作與 `BaseAdapter`，所以中間 base 提供的 override 有效，不維護子類白名單。

`load_data=True` 須有 concrete `load()`，或有可無參數建立且提供 callable `load()` 的 `exp_cls`；`load_data=False` 不接受 concrete load override。這是 import-time 宣告一致性檢查，不證明某個檔案能載入、hook 語意正確或操作安全。`validate_run_request()` 是 framework 必呼的 preflight；base 提供 no-op default，不因繼承 no-op 而新增 capability flag。Subclass 拼錯預期要 override 的 no-op hook 名稱，驗證無法辨識其意圖。精確錯誤與預設值以 `base.py` 為準。

## 修改原則

1. 單一實驗的設定、文字與流程 policy 留在 authoritative adapter file。
2. helper 確實出現第二個 caller，且有清楚 mechanics seam 時，才移入 `_support/`。
3. 不以 forwarding wrapper 隱藏單行邏輯；不讓 `_support` 解讀 registry key 或實驗順序。
4. 新增 adapter 時同步加入 `../registry.py`，並在 `tests/experiment/v2_gui/measure/adapters/`
   對應路徑加入 observable contract tests。

range centre/span/fallback、writeback target/description/role 等 experiment policy 直接寫在
authoritative adapter；`_support` 只提供不含這些具體值的 parameterized mechanics。
