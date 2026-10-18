# v2 experiments

**Last updated:** 2026-10-19，mux 具名 tones／participants 與全元素驗證

ZigZag 的 Run 與離線編譯共用 `build_zigzag_modules`，使用 register loop。
`reset_phase_cycle` 可在偶數 averaging sweeps 間交替最後 reset qubit pulse 的 0/180°
相位，不新增 RF pulse 或 programmed wait。它平均初始化的橫向相干分量，不修正
population 或 repeated-gate 誤差；支援 TwoPulse/Bath reset，需獨立驗證相位及初態投影。

`twotone/time_domain/detune.actual_detune` 集中 Ramsey／Echo 的 actual MHz 計算。Native run 與 workflow callback 共用四個 caller，不各自複製 time／phase rounding 公式。Ramsey 必填 time_scaler=1.0，Echo 必填 0.5，保留兩half-delay的原量化。它使用既有 typed SweepCfg／QickConfig，零 nominal detune 回0.0，其他值保留原 quantization／NumPy failure 語意。不改 fitter、閾值、round_zcu 或 sweep2array。

每個實驗有自己的資料夾。`core.py` 擁有 cfg、Result、量測與分析政策。`gui.py` 是可選的 measure-gui adapter。Leaf 的 `__init__.py` 只提供 core exports，不載入 GUI。

`v2` 延續 program/v2 的版本脈絡，不表示 GUI framework 的第二版。通用 runtime、量化、SNR 與 tracker 工具留在 `zcu_tools.experiment.v2`。Core 不依賴 GUI，import-linter C15 檢查 core 到 GUI framework 的直接與間接依賴。

`mux/onetone` 使用 `readout/mux`，在每個 sweep 點對選定 tone 加上共同的名目頻率偏移。
每條曲線透過原始 tone 序號對應完整 cfg，資料保存共同偏移軸、原始序號與二維 signals。
Tone label 是可重複的顯示文字，不是 component 名稱，也不改變資料軸。
`mux/onetone` adapter 從 group 的 readout slot 建立初值。active_tones=None 量全部，指定非空、唯一的整數 list 則按輸入順序量測原始序號。Lab 將 subset 交 program，不改完整 cfg。RunRecord 與 native cfg restore 保留完整 readout、labels 及 selection，不以目前表單補歷史資料。
Adapter 提供 K 條 active live 曲線及 K 張 `fit_i` 圖，i 是原始 tone 序號。FIT 共用 onetone/freq 的模型與 delay 選項，逐 tone 隔離數值失敗。沒有歷史 cfg 時拒絕 FIT。
Module proposal 只更新成功 active tone 的頻率，保留其他完整設定，固定寫到 `$mux.readout.readout_rf`。
單一參數從 Run／Load capture 的 `$mux.resonators` 依完整原始序號對應。Resonators 數不等於完整 tone 數時只提 module，summary 顯示原因。
MuxGroup 可不填 qubits 做 resonator bring-up，保留手填 readout，不增加完整綁定門檻。
填了則由 model 檢查等長與重複 qubit。Form 在 Open／Reset 呼叫
`_support.measure.mux_bindings.validate_mux_bindings(ctx.parameters)`，以固定 capture
檢查 component 存在、kind 及逐項 qubit.resonator 對應，無效就拒絕來源。
其他 mux／FTF Forms 共用此入口，精確 import 與呼叫例見
[Measure support](_support/measure/README.md#mux-綁定驗證)。
Module 可整組引用或 inline 編輯，tones 使用單行 JSON。
Library 的 `readout/mux:blank` 只給 scalar 編輯初值，作者必須填 tones 並確認接線；範本不是硬體安全預設。

## Mux qubit numeric cores

`mux/freq` 與 `mux/amp_rabi` 提供可 import 的 typed cfg、public `run`、分析與 native persistence。
兩個 run 共用 `_support.mux_acquisition.acquire_mux_scan`，每點 compile 一次。
共同 reset 後，既有 Join 讓 drives 同時起跑，短 branch 完成後等待當點最長 branch，
再做共同 mux readout。不補齊末端，不以全 sweep 最大長度 padding。
Form／recipe／adapter／catalog／writeback／UI 接線另批交付，等待 fixed-list Form Base。
這些 core 入口不註冊 catalog。

七個 mux qubit cfg 使用完整固定 `tones` 與獨立 `participants: list[str]`。
每個元素有固定 qubit `name` 與必填 drive；pulse sweep 類另有自己的 sweep，
T1／Ramsey／Echo 使用共用 requested-delay 軸。所有元素都驗證，未參與者也必須有
有效 drive、獨立 generator；pulse sweep 點數全部相同。Len Rabi 全部元素只接受
const／flat_top，不伸縮或裁切 arb asset。
Participants 非空、不重複，名稱必須存在。Runtime 按名稱順序查完整原始位置，
例如 Q3／Q1 對應 `[2, 0]`。GE raw、analysis 與 native persistence 同樣保留該順序。
RunRecord 保存完整具名 cfg 與 participants，不另保存目的地或 group snapshot。

`prepare_frequency_scan(cfg, soccfg=...)` 與 `prepare_amplitude_scan(cfg, soccfg=...)`
各自按 participants 查出原始 indices，回傳 `_support.mux.MuxPulseScan`。
其 `indices` 保留原始序號，`coordinates` 是各自硬體 grid 的 `(K,P)` 浮點陣列，
`point_pulses` 是 `(P,K)` 固定 numeric pulse cfg，不展開 Cartesian product。
這些入口只準備資料，不 acquire；public run 將其交給上述 common-start zip 排程。
Frequency／gain conversion 分別沿既有 `round_zcu_freq`／`round_zcu_gain`。

`MuxSweepResult` 保存原始 `tone_indices`、各自量化的 `coordinates` 及平均 complex IQ `signals`。
兩個矩陣都是 `(K,P)`。NaN IQ 表示未取得的 partial data，不表示成功。
各 core 的 `analyze(source, options, plots=...)` 使用 explicit RunRecord，允許 cfg=None。
每個原始序號各有 `fit_i` 圖、校準值／誤差／quality，或保留原因的 `MuxFitFailure`。
Freq 沿用單 qubit 的 magnitude/background transform 與 linewidth fitter；Amp 沿用零 gain phase、無衰減 fit。

Native mapping 使用 `coordinates`／`signals` 兩個 variables，共享 Point／Tone axes。
Frequency 座標在磁碟上是 Hz，在記憶體是 MHz。Gain 不縮放。
Analyze／Save 使用 explicit RunRecord；Load 還原同次完整 tones cfg、原始 indices、獨立數值座標與 IQ。
它們不讀目前 Form 或 library。取消保留已完成點與當點 completed-round 平均，未寫入 IQ 維持 NaN。
Caller 先呼叫 RunContext.cancel_signal.raise_if_error()，不把 failure 的 partial data 報為成功。
Numeric core 不保存 writeback destination，也不發 proposal。

## Layout

```text
zcu_lab/
├── definitions.py        # explicit measure catalog
├── templates.py          # startup-only template catalog
└── v2/
    ├── <family>/<experiment>/
    │   ├── core.py
    │   └── gui.py         # optional measure adapter
    └── _support/          # shared domain helpers
```

`_support/resonator.py` 集中單 tone 與 mux 的模型 fitting／畫圖 mechanics 及 route prior 解析，不 import GUI。各 caller 保留選擇、失敗隔離與 writeback 政策。Core 不經 `_support/measure/` 取得這些 mechanics。

`definitions.register_all` 明列 measure adapters，從 caller 的 Registry 取得 `templates.TEMPLATES` 持久範本，填入 startup 的 library catalog。組合根以 `Registry(templates=TEMPLATES)` 注入一次，import 不註冊或操作硬體。Reload 重建 leaf、catalog 與 core-typed Notebook helper，保留 measure／singleshot shared support、workflow support、templates 與框架 identity。修改固定依賴需要重啟。

## Reading routes

- [FTF bring-up 與 CZ 校準](ftf/guide.md) 說明既有 F1/F2 單 qubit 校準、角色與保存方式，以及待交付的 two-qubit 校準路線。
- [Experiment authoring](experiment-authoring.md) 說明核心 records、資料形狀與 run 慣例。
- [Measure authoring](measure-authoring.md) 說明 cfg defaults、operator guides、writeback 與 reload。
- [GUI contracts](gui-contracts.md) 說明 concrete adapter hooks 與 capability。
- [Measure support](_support/measure/README.md) 提供共享 domain mechanics。
- [Framework runtime](../../lib/zcu_tools/experiment/v2/runtime/README.md) 擁有 Schedule、buffer、acquire 與 stop lifecycle。

框架 owner 用測試自己定義的最小合成 workflow／adapter 驗證公開接縫，不 import `zcu_lab`。本套件的實驗定義不寫個別測試，也不以 lab registry 選取測試材料。刪掉 `zcu_lab/` 後整套 pytest 仍須通過，包含 slow。Core-only caller 明確 import `core`；GUI callers 明確選擇其附件。
