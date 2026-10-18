# v2 experiments

**Last updated:** 2026-10-18，FTF CZ common-amplitude 校準、conditional phase 與共用 quadrature 分析

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

- [FTF bring-up 與 CZ 校準](ftf/guide.md) 說明既有 F1/F2 單 qubit 校準、角色與保存方式。已提供四分支 conditional spectroscopy 與 [CZ common-amplitude](ftf/cz_amplitude/guide.md)。後者保留 same-shot raw、擬合零相位完整回返週期，並由操作員逐列接受 coupler 的雙 pulse／雙 slot proposals。[Conditional phase](ftf/conditional_phase/guide.md) 提供獨立 X/Y acquisition、phase 診斷與 n=1 frequency proposal。Family guide 另列其餘待交付的 two-qubit 校準。
- [Experiment authoring](experiment-authoring.md) 說明核心 records、資料形狀與 run 慣例。
- [Measure authoring](measure-authoring.md) 說明 cfg defaults、operator guides、writeback 與 reload。
- [GUI contracts](gui-contracts.md) 說明 concrete adapter hooks 與 capability。
- [Measure support](_support/measure/README.md) 提供共享 domain mechanics。
- [Framework runtime](../../lib/zcu_tools/experiment/v2/runtime/README.md) 擁有 Schedule、buffer、acquire 與 stop lifecycle。

框架 owner 用測試自己定義的最小合成 workflow／adapter 驗證公開接縫，不 import `zcu_lab`。本套件的實驗定義不寫個別測試，也不以 lab registry 選取測試材料。刪掉 `zcu_lab/` 後整套 pytest 仍須通過，包含 slow。Core-only caller 明確 import `core`；GUI callers 明確選擇其附件。
