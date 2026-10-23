**Last updated:** 2026-10-23，mux seed diagnostics

# Measure domain support

**讀者：** lab-author。查閱多個量測 adapter 共用的表單初值、分析載體與 writeback 工具。

這個 private package 擁有至少兩個 measure experiment adapters 共用的 implementation
mechanics：analysis carriers、writeback helpers 與跨實驗的 context utilities。
已定案且供多個 Form 共用的初值政策集中在具名 domain helper，涵蓋 T1 時間尺度選源、Rabi 來源與首次初值，以及 mux readout 的 ringup 等待。
各 Form 仍明確選擇倍率或場景，並組合表單結構。校準政策不進入 generic framework，也不擴及 writeback 或實驗順序。

`analyze_results.py` 把 core 的 named FitQuality map 投影為 JSON-safe summary，並具體化 quality.invalid 的 summary 路徑。Mux caller 指定 `summary.tones[row].fit_quality`，單 qubit 預設維持 `summary.fit_quality`。數值與直接原因由 fitting core 擁有；共用 mechanics 不算品質，也不改 writeback 或 accept policy。

`mux_forms.py` 集中八個 mux Forms 的 readout 初值，以及七個 qubit Forms 的固定名稱與 recovery 初值。
Qubit Forms 先要求完整 qubit/resonator 綁定。缺 readout slot 時從 group resonators 產生清單，每個 resonator 一個槽位。
Group tones 的 freq 取捕捉的 resonator.freq，label 初值為 resonator 名稱。缺 slot 時，gen_ch 取 `$mux.wiring.gen_ch`，第 i 個 tone 的 ro_ch 取 `$mux.wiring.ro_ch[i]`，gain 填 0.1。gen_ch、mixer_freq 與 freq 使用 Expr 初值。各 tone 的 ro_ch 是從固定 capture 的 group.wiring.ro_ch[i] 取出的整數初值，scalar Expr 不支援 indexing。操作員仍須確認通道與振幅。
`mux_readout_initial` 有 captured readout 時回傳 Slot ref，使用 library 完整 tones 與 scalar 設定。沒有時以 `readout/mux:blank` Template 的 scalar 初值為來源，再加 group tones 與 group wiring。Blank 的 mixer_freq 從 `$mux.mixer_freq` 取初值，缺失或 null 時留空必填，不依賴 tone 頻率或 kappa。length 與 ro_length 保持必填且未填，不猜時長。
`mux/onetone` 以 `wait_for_ringup=True` 選擇副本加 group tones，並啟用 timing 初值。
有 captured readout 時，helper 從固定 capture 取完整 group 已設成員的最小 kappa，在來源 trig_offset 上加五倍 ringup 等待。
此路徑忽略缺 leaf 或 null 的 kappa。全未設時不加等待，trig_offset 保留來源值。
Pulse 至少涵蓋 ADC 窗口，原 ro_length 與較長的 pulse 保留。
Mux onetone 缺 slot 時，ro_length 固定填入 1.0 us，不依賴 kappa 或 time_of_fly。
完整 group 的 kappa 都有效且 group 有頂層 time_of_fly，才推導 trigger 與 length。
trig_offset 取 `$mux.time_of_fly` 加 ringup，length 取 trig_offset + ro_length，不讀 resonator 的 pulse 路徑 TOF。
任一 kappa 或 group time_of_fly 缺值時，trig_offset／length 留空必填，mixer_freq 不受影響。
上述缺值情境仍填入 ro_length=1.0 us。
已設的 frequency 須為有限數，time_of_fly 須為有限非負數，否則 Open／Reset 報錯。
候選未經裝置確認，capability 檢查仍在 Run 邊界。Fine 的 Readout FieldMeta.description 逐項列來源。`mux_readout_seed_metadata(initial)` 依實際 Slot／Template overrides 產生 canonical 逐位置 metadata。各 caller 以 `Form(...).with_seed_metadata(...)` 加入來源公式。只有 mixer 初值、推導 timing 與 gain 0.1 標為 uncalibrated，ref-only slot 不標記，未改寫的 captured 欄位保持未知。
每個已設成員的 kappa 都須為有限正數，否則 Open／Reset 報錯。其他 mux Forms 沿用來源 timing。
Onetone 的 Open／Reset 產生 detached、可編輯的初值，cfg 輸出保留操作員之後的編輯。
READOUT_ELEMENTS 使用公開 FixedElements 的無 participants 固定槽位模式，只拒絕增刪，所有 tone 欄位仍可編輯。
Label 不是 identity。這些初值不保證硬體安全，操作員仍須核對接線。缺少必要欄位仍擋 Run。
每個 Form 使用公開 `FixedElements`、native readonly／element_label metadata 與逐位置初值。
所有元素共用宣告，不把逐位置 lock 套成共同限制。Participants 是完整 tones 的原始索引清單。Form 的 FixedElements 宣告 drive、Echo pi_pulse 與逐 tone sweep 的適用性，未參與欄位灰掉且不驗證，readout 身分始終必填。切回 participant 時立即驗證保留草稿。GE 的可選 init_pulse 保持既有驗證，不列入 active 必填或停用集合。
Recovery 採完整名稱清單中最大的五倍 T1；沿用既有 missing／invalid 政策，不重讀來源。
Rabi pulse 初值與 T1 尺度 helper 可用明確 qubit 名稱，單 qubit caller 仍預設 `$qubit`。

`mux_results.py` 集中七個 mux qubit adapters 的固定來源身分核對與 position/name summary。目的地只來自同次 Run／Load capture；缺 capture 不產生 proposal。來源 cfg 有完整名稱時，必須與捕捉的完整 qubits 相同。Summary 的名稱只取來源 cfg，缺 cfg 時保留 None，不讀目前 Form 或 library。
來源名稱可用時，作者附 `ElementIdentity` 給 native summary rows 與候選；原始位置不由
row offset 推測。直接 NumPy scalar 在 summary 邊界轉為 Python scalar，不改數值、fit
或 proposal eligibility。

Concrete adapters 依賴本 package 與 framework contracts，`zcu_lab.definitions` 組合 adapters。
本 package 不 import concrete adapter 或 `zcu_lab.definitions`，也不宣告 experiment order。只屬於
單一實驗的初值政策留在其 `form.py`，其他 domain policy 留在 authoritative file；同一小群組共用但不具全域意義的
helper 留在群組內 `_shared.py`。

表單形狀與初值由各實驗的 `form.py` 提供 `make_form(ctx)`，共用生成器位於
`zcu_tools.gui.form_adapter`。持久範本表由 `zcu_lab.templates.TEMPLATES` 擁有，
組合根注入 Registry 與 Library 新建選單。不再維護 builder、Seed wrapper 或角色 default 表。

`mux_bindings` 擁有固定 ParameterContext 的 qubit／resonator 對應驗證。
Onetone 的缺 qubits bring-up 政策與 qubit 實驗的完整綁定要求分開，兩者不讀 live 來源。
Group model 仍擁有長度與重複 qubit 的結構規則。

`ctx_helpers.t1_initial_time` 集中 T1 衍生初值的選源。它從固定 capture 讀普通 T1，tone 掃描先讀歷史 tone T1，只有缺值才下探普通 T1。無 capture 或所有候選 leaf 缺失／null 時使用 30 µs 時間尺度；已存無效值回 None，讓衍生必填欄位留空。來源與 alias／path 錯誤仍傳遞，不寫回校準。各 Form 明確指定等待或掃描所需倍率，自然等待不使用 tone T1。

`rabi_initial_values.len_rabi_initial_values` 為兩種 Length Rabi 選擇相同的 pulse 來源與掃描上限。
已有 x180 時保留 Slot 與 gain，以有效 pi_len 的四倍作上限。
pi_len leaf 缺失或 null 時上限為 1 µs，已存但無效的 scalar 則留空。
arb 沒有持久 scalar length，使用缺 leaf 的 1 µs 上限，不讀 runtime length。
沒有 x180 時使用 gain=1 的 qub_probe Template 與 1 µs 上限。
它只讀固定 capture 中的 validated Spec，跟隨 waveform key，但不 materialize、不讀 asset、不寫回校準。
原始來源在 owner 驗證前被拒絕或轉型的值，不屬於此 typed capture 的可觀察輸入。

`rabi_initial_values.amp_rabi_initial_pulse` 為兩種 Amplitude Rabi 選擇相同的 pulse 來源與長度。
x180 為 arb waveform 時保留原來源 Slot、形狀與 data，不複製、不覆寫 length，也不讀 asset。gain 仍掃 −0.3～1.0；要調長度須先換 waveform。
有持久 scalar length 的 x180 才透過 Slot(copy=True).set 複製捕捉來源，覆寫 waveform.length=1.5*pi_len*pi_gain（µs）。
pi_len 須為有限正數，pi_gain 須為有限數且在 (0, 1]。無效來源讓必填長度留空。
來源 gain 若為 NaN 或 ±Inf，額外設 gain=None，避免非有限數留在 JSON input。兩個必填欄位都補齊才可提交。
有限的無效 gain 保留原值。沒有 x180 時使用 qub_probe Template，明確設定 gain=1 與 length=1 µs。
Nested waveform key 的複製由 Form adapter 負責。來源與其他欄位不變，不寫回校準。

## Mux 綁定驗證

`mux_bindings.require_mux_group(parameters)` 要求固定 capture 中有 scalar alias `mux`，且指向 `group/mux` 元件。
Mux readout 初值與 `require_mux_qubits` 共用此查詢。缺 capture、alias 未綁或 target 非 group 時，錯誤指出需要 `group/mux` 與 alias `mux`。
此要求不改變下列可選 qubit 配對驗證的 bring-up 政策。

`mux_bindings.validate_mux_bindings(parameters: ParameterContext | None) -> None`
集中 onetone、mux qubit 與 FTF two-qubit Forms 共用的跨 component 驗證。
它只讀借用的固定 capture，不刷新來源、不讀 live 檔案或裝置，也不 materialize library assets。
有 qubits 時，它驗證完整有序清單的 component 存在、kind 與逐項 qubit.resonator 對應。
錯誤保留欄位、原始 index 及實際值。MuxGroup model 仍擁有名稱、等長及重複 qubit 的結構驗證。

None、未綁 mux、不存在或非 MuxGroup 的 target、沒有 qubits 都不增加 bring-up 門檻。
其他 alias 錯誤與來源故障照常傳遞。FTF caller 自行要求 group、qubits 及 F1/F2；
新 mux qubit caller 自行要求完整 tone 數對齊與 active drive。共用函式不選 tone、不猜角色、不寫回。

在 concrete `form.py` 使用下列 import，並在自己的 `make_form(ctx)` 建立 Form 前呼叫。
`TwoQubitCfg` 是該 caller 的既有 core cfg；範例省略 caller 自己的角色要求與 tree 內容。

```python
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2._support.measure.mux_bindings import validate_mux_bindings

from .core import TwoQubitCfg


def make_form(ctx: FormContext) -> Form[TwoQubitCfg]:
    validate_mux_bindings(ctx.parameters)
    # Enforce this caller's required group, qubits and F1/F2 here.
    return Form(TwoQubitCfg, tree={})
```

## 其他 measure helpers

`singleshot_shared.py` 從 AnalyzeRequest 的固定 Run ParameterContext 讀取分類、chi/kappa
與可選分析校準。缺 capture 明示失敗；可選 leaf 缺少或 null 時交給 core 的既有預設。

Measure flux picker 的 plugin、frontend 與結果 carrier 位於 `zcu_tools.gui.app.measure`。Adapter 從固定 Run capture 讀初始參數，交 FluxPickInputs 與普通數值。Lib 不讀具體 Result 或 ctx_helpers。兩個 adapter 保留各自的 magnitude-only 政策，共用原有 kernel／plugin。

`mux_flux_adapter.MuxFluxPickParams` 以原始 `tone` index 選 participant，不把 acquisition row 重編號。
`flux_half`／`flux_int` 是可選的有限 native DC 初線值，每個 None 各自沿用該 tone 的 Run capture，缺校準才用 shared picker 初值。
Recipe 經同一個 Analyze operation 的 typed updates 傳入，不更改 capture 或其他 tone。
Picker done 保留原始 tone／Q 身分並產生該 Q 的 writeback proposal，不直接接受寫回。

package facade 只 re-export concrete adapters 實際共用的 authoring vocabulary。單一 adapter 的
range recipe 與 writeback policy 不經 facade 轉送。共用 readout writeback helper 依 caller 傳入的
target、description、field updates 與 role 建立 proposal，保留既有 DPM 校準政策。

Reset Form 從 Open／Reset 的固定 capture 讀 qubit ext 初值。Expr 與 Range 的求值及驗證由 Form owner 處理。
SingleToneFreq、DualTonePower 與 BathPhase 各自擁有 finisher。它們直接 materialize 該 Run
的完整 tested_reset，只換本次分析欄位，不重讀 ext 或方案 slot。三個 length adapter
不擬合參數，返回空 proposal collection。框架與本 support package 不檢查 reset 校準是否齊全。

`require_mux_qubits(parameters, tone_count)` 是 qubit caller 的嚴格 admission：要求 capture、MuxGroup 及完整 qubits，呼叫公開 validate 契約，再比對完整 numeric tone 數。回傳完整有序 qubit 名稱，不投影 active subset。
