**Last updated:** 2026-10-17，T1 與 Rabi 初值選源

# Measure domain support

這個 private package 擁有至少兩個 measure experiment adapters 共用的 implementation
mechanics：analysis carriers、writeback helpers與跨實驗的 context utilities。

`analyze_results.py` 把 core 的 named FitQuality map 投影為 JSON-safe summary，並具體化 quality.invalid 的 summary 路徑。數值與直接原因由 fitting core 擁有；共用 mechanics 不算品質，也不改 writeback 或 accept policy。

Concrete adapters 依賴本 package 與 framework contracts，`zcu_lab.definitions` 組合 adapters。
本 package 不 import concrete adapter 或 `zcu_lab.definitions`，也不宣告 experiment order。只屬於
單一 adapter 的 domain policy 留在其 authoritative file；同一小群組共用但不具全域意義的
helper 留在群組內 `_shared.py`。

表單形狀與初值由各實驗的 `form.py` 提供 `make_form(ctx)`，共用生成器位於
`zcu_tools.gui.form_adapter`。持久範本表由 `zcu_lab.templates.TEMPLATES` 擁有，
組合根注入 Registry 與 Library 新建選單。不再維護 builder、Seed wrapper 或角色 default 表。

`ctx_helpers.t1_initial_time` 集中 T1 衍生初值的選源。它從固定 capture 讀普通 T1，tone 掃描先讀歷史 tone T1，只有缺值才下探普通 T1。無 capture 或所有候選 leaf 缺失／null 時使用 30 µs 時間尺度；已存無效值回 None，讓衍生必填欄位留空。來源與 alias／path 錯誤仍傳遞，不寫回校準。各 Form 明確指定等待或掃描所需倍率，自然等待不使用 tone T1。

`rabi_initial_values.len_rabi_initial_values` 為兩種 Length Rabi 選擇相同的 pulse 來源與掃描上限。已有 x180 時保留 Slot 與 gain，以有效 pi_len 的四倍作上限；沒有 x180 時使用 gain=1 的 qub_probe Template 與 1 µs 上限。它只讀固定 capture 中的 validated Spec，跟隨 waveform key，但不 materialize、不讀 asset、不寫回校準。原始來源在 owner 驗證前被拒絕或轉型的值，不屬於此 typed capture 的可觀察輸入。

`rabi_initial_values.amp_rabi_initial_pulse` 為兩種 Amplitude Rabi 選擇相同的 pulse 來源與長度。
有 x180 時透過 Slot(copy=True).set 複製捕捉來源，覆寫 waveform.length=1.5*pi_len*pi_gain（µs）。
pi_len 須為有限正數，pi_gain 須為有限數且在 (0, 1]。無效來源讓必填長度留空。
來源 gain 若為 NaN 或 ±Inf，額外設 gain=None，避免非有限數留在 JSON input。兩個必填欄位都補齊才可提交。
有限的無效 gain 保留原值。沒有 x180 時使用 qub_probe Template，明確設定 gain=1 與 length=1 µs。
Nested waveform key 的複製由 Form adapter 負責。來源與其他欄位不變，不寫回校準。

`singleshot_shared.py` 從 AnalyzeRequest 的固定 Run ParameterContext 讀取分類、chi/kappa
與可選分析校準。缺 capture 明示失敗；可選 leaf 缺少或 null 時交給 core 的既有預設。

Measure flux picker 的 plugin、frontend 與結果 carrier 位於 `zcu_tools.gui.app.measure`。Adapter 從固定 Run capture 讀初始參數，交 FluxPickInputs 與普通數值。Lib 不讀具體 Result 或 ctx_helpers。兩個 adapter 保留各自的 magnitude-only 政策，共用原有 kernel／plugin。

package facade 只 re-export concrete adapters 實際共用的 authoring vocabulary。單一 adapter 的
range recipe 與 writeback policy 不經 facade 轉送。共用 readout writeback helper 依 caller 傳入的
target、description、field updates 與 role 建立 proposal，保留既有 DPM 校準政策。

Reset Form 從 Open／Reset 的固定 capture 讀 qubit ext 初值。Expr 與 Range 的求值及驗證由 Form owner 處理。
SingleToneFreq、DualTonePower 與 BathPhase 各自擁有 finisher。它們直接 materialize 該 Run
的完整 tested_reset，只換本次分析欄位，不重讀 ext 或方案 slot。三個 length adapter
不擬合參數，返回空 proposal collection。框架與本 support package 不檢查 reset 校準是否齊全。
