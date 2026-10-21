---
status: accepted
---

# ADR-0074 — 獨立的平鋪 Readout module

## 背景

內嵌 Pulse 與 DirectReadout 讓作者重複填寫 generator 與 ADC 的頻率和通道。
Readout 的引用、表單繼承與校準也各自維護同步規則。

使用者於 2026-10-15 提出：「我在考慮將Readout Module改成獨立一個Module，而不是內嵌Pulse/DirectReadout，同時移除後者。這樣之後MuxPulse/MuxPulseReadout出來後應該比較一致」。

## 決策

### Readout 擁有 generator 與 ADC 配置

持久 `PulseReadoutSpec` 與 materialized `PulseReadoutCfg` 使用 `readout/pulse`。
Generator 通道叫 `gen_ch`，ADC 通道叫 `ro_ch`。
使用者於 2026-10-15 選定：「1保留readout/pulse吧，與readout/mux一致。2叫gen_ch好了」。

Pulse readout 平鋪保存 `gen_ch`、`nqz`、`freq`、`gain`、`phase`、`waveform`、
`mixer_freq`、`pre_delay`、`post_delay`、`ro_ch`、`ro_length`、`trig_offset` 與 `desc`。
它不保存 `pulse_cfg`、`ro_cfg` 或獨立的 `ro_freq`。
MHz 頻率 `freq` 同時決定 generator 與 ADC。時間欄位使用 us，phase 使用 deg，gain 使用 normalized gain。

Pulse readout 的 `set_param` 接受 `freq`、`gain`、`length` 與 `ro_length`。
`length` 委派 waveform，其餘三項更新同名欄位。其他名稱回報 ValueError。

整個 Readout 可以引用 library key。Pulse readout 的 waveform 仍可內嵌或引用 waveform key，
包含 flat-top ramp 的遞迴引用。Readout 不引用 Pulse 條目，Reset 的 Pulse 引用保持不變。

使用者於 2026-10-15 對純 ADC Direct readout 回覆「不保留」，對獨立 ADC 頻率回覆「onetone/sa可以刪除。C」。
因此不提供 `readout/direct`、Direct 範本或 onetone/sa。
Onetone 的 linear 與 homophasal 取樣繼續使用同一個 Readout。

### Mux readout 保存有序 tones

持久 `MuxReadoutSpec` 與 materialized `MuxReadoutCfg` 使用 `readout/mux`。
Module 明填 `gen_ch`、`nqz` 與 `mixer_freq`，共用 const 播放 `length` 與 ADC window。
有序的 1 至 8 筆 tones 各自保存 `freq`、`gain`、`phase` 與 logical ADC channel `ro_ch`。
每筆 tone 的 generator 與 ADC 使用同一個 MHz 頻率，不引用 Pulse 或 waveform。
Mux readout 不接受 sweep 或逐 tone library key，`set_param` 一律拒絕更新，caller 重建 cfg 與 program。

Tone 的選填 `label` 預設空字串，隨 Spec、cfg 與 library 保存。
Label 只作顯示，可以重複，不代表 component；program 不解讀它。
Tone 序號定義資料軸順序，label 與 ADC channel 都不改變這個順序。
Form 沿用整個 module 的 ref／inline 與 Convert to inline，tones 使用 model-list picker 逐項編輯。
每筆 tone 的 freq、gain、phase、ro_ch 與 label 都可編輯。Label 不當固定 identity。

Library 新建入口識別 MuxReadoutSpec，`readout/mux:blank` 只提供 scalar 編輯初值。
Mux 實驗的 lab Form 從固定 capture 的 mux group 產生清單，每個 resonator 一個槽位。
Freq 取該 resonator 校準值，gain 使用 lab 初值，ro_ch 依 group 順序。生成器不猜硬體或校準值。
Lab 以 Template.set／Slot.set 提供依原 Spec schema 驗證的 typed model-list 初值。
Form adapter 集中生成與 nested ref／inline 路徑處理，Core 只處理普通 list input。
FixedElements 的無 participants 模式只固定目前 inline 清單的槽位數，見 [[0079]]。

使用者於 2026-10-16 曾同意 scalar 初值加必填 tones JSON 的方案。
使用者於 2026-10-20 對從 group 產生可逐欄編輯的固定清單回覆：「可以」。
本次取代 JSON leaf 候選，不擴充 InputPolicy，不新增持久 identity，也不提供硬體安全保證。

2026-10-21 註：目前的表單設計採用固定 model-list 與逐欄編輯，JSON leaf 是已被取代的候選。

### Owner 不變，caller 不重建舊配置

`cfg_model` 擁有 Spec、cfg 驗證、引用展開與參數設定。
`program` 負責 native conversion、QICK 宣告與排程。
`resources` 擁有 library 文件與 I/O。
`gui.form_adapter` 從 core cfg 與持久 Spec 推導 Form，session 擁有 Library entry 表單。
這些分工沿用 [[0072]]、[[0065]] 與 [[0073]]，不保留 editable shape catalog。

依賴沿用 `.importlinter` 的 C11、C16、C19 與 `form-adapter-no-host`。
Readout 不新增 registry、factory 或跨 owner resolver。
Core/native factory 使用 caller 捕捉的 resolver 與 asset capability，不查 process-global library。

Pulse readout runtime 可將平鋪 generator 欄位投影成內部 PulseCfg，以重用 waveform 與播放邏輯。
Caller 不維護另一份 ADC cfg。Pulse cfg 不含 `ro_ch`，runtime Pulse 另接收 ADC matching 通道。
PulseRegistry 的硬體定義身分包含該通道，避免相同 pulse cfg 在 ADC 0 與 1 間錯誤共用。

Pulse readout 在 module 起點送出 readout configuration，在 `trig_offset` 觸發 ADC，
在 `pre_delay` 播放 generator。Blocking duration 覆蓋 ADC window 和含 post-delay 的 generator 播放，
各自沿既有硬體 clock rounding。

### 高階頻率與低階 words 分開

`TablePulseReadout` 保留成對的 generator／ADC frequency words。
`freq_val`、`ro_freq_val`、`freq_words` 與 `ro_freq_words` 是 runtime register／table 契約，
不是持久 cfg 的兩個頻率。兩種 words 表示同一語意頻率，由各硬體的 mixer／downconversion 規則編碼。
模擬的 accumulated、raw 與 decimated 路徑使用同一份平鋪 readout plan，數值單位與物理模型不變。

### 表單與持久資料

Pulse readout Form 直接編輯 `gen_ch`、`ro_ch`、`freq`、`gain`、`waveform` 與 ADC window。
`gen_ch`、`ro_ch` 必填且可編輯，不從其他欄位推導或補值。
Form 只在整個 module 和 waveform 的 inline input 加入 wrapper。
例如 recipe 頻率路徑為 `modules.readout.module.freq`，持久 Spec 路徑為 `freq`。
Readout 不使用舊 shape inheritance 或自動補值，表單由共用 Form 生成器建立。

舊 nested Readout 與 Direct 條目不遷移，也不加相容讀取。
正常驗證入口回報具體 library key，作者重建條目。
Artifact restore 的 typed cfg 失效與原始 data 可讀性由 [[0063]] 的 owner 負責。

Module library 版本只跟容器結構改變，不因 module／waveform 欄位變動升版。
使用者於 2026-10-15 定案：「只是改Module而已真的需要升Library版本嗎？就像加一個Component原件定義，不應該讓舊版本不能讀對吧」「1可以，2A，3保留2.0」。
Library 維持既有 2.0，不加 readout 條目版本。
實驗 cfg schema major 與宿主 wire／GUI revision 依各 owner 的不相容契約更新。
Parameter container、native artifact 與 ledger 的外層格式不因 Readout 欄位形狀改變。

## 取捨

共用某個 Pulse 條目的 Readout 改為自行保存 generator 設定，兩者需要分別維護。
換來的是一個頻率來源和一套表單路徑，runtime 仍可重用 Pulse 的播放實作。
純 ADC 與獨立 ADC 頻率不在 Readout 契約內。
Mux readout 共用 generator 與時間欄位，不能逐 tone 設定 waveform 或時間窗。
