# Qubit measurement context

本詞彙表定義 ZCU-Tools 的量測流程、參數容器、保存資料與分析交接用語。

## 參數與環境

**Result Entry**:
一次降溫加一片 chip 的資料單位，包含工作點、參數與量測成果。名稱由使用者決定，不帶可解析的物理意義。
_Avoid_: project identity, chip/qubit scope

**Working Point（工作點）**:
由實驗語意決定的一段穩定環境，擁有獨立參數與 module library。它不綁定特定裝置或物理量。
_Avoid_: device value, bias folder

**Parameter Container（參數容器）**:
保存參數與來源的文件模型，包含 general、components 與 provenance。Setup 元件範本與工作點使用同一種模型，各自保存獨立的值與來源。
_Avoid_: global parameter dict

**Component（元件）**:
參數容器中的具名條目，例如 Q1 或 R1，具有 kind 和該 kind 的欄位。
_Avoid_: role, device

**Role（角色）**:
實驗 adapter 所需的位置，例如 qubit 或 resonator，並帶有要求的 kind。角色不是元件本身。
_Avoid_: component name, module name

**Alias**:
GUI session 中可編輯的名稱對應，值是元件名稱或引用前列的軟連結。它可供多個實驗角色使用。
_Avoid_: focus component, tab-local binding

**Ledger**:
一個 Result Entry 的事件記錄，包含量測、保存與接受結果等事實。參數來源可引用同條目的事件。
_Avoid_: run directory, parameter snapshot

**ModuleLibrary Template**:
建立 module 或 waveform entry 時使用的具名起始 shape 與 defaults。依 template 建立的 entry 彼此獨立。
_Avoid_: experiment role, module role catalog

**SessionEnv**:
GUI session 目前生效的實驗 context 與 live 連線資源。它與預填 Setup 的 Session Preference 不同。
_Avoid_: persisted session preference, complete app state

**Session Preference**:
GUI 記住、用來預填 Setup 的條目名稱與連線位址。它不表示 entry 已開啟或儀器已連線。
_Avoid_: active environment

## 量測與保存

**Experiment Run**:
使用特定 cfg 和硬體狀態執行一次實驗。一次 Run 產生一個 Experiment Result。
_Avoid_: job, task run, measurement job

**Experiment Result**:
一次 Experiment Run 的量測成果，包含分析該次量測所需的資料。
_Avoid_: output blob, save payload

**Complete Experiment Result**:
所需 Data Variables 都齊全的 Experiment Result。完整性不保證全部預定測點都已完成。
_Avoid_: successful operation, fully scanned result

**Experiment Result Identity**:
將保存資料連回同一個 Experiment Result 的共同身分。它不屬於任何單一 dataset。
_Avoid_: primary file identity, sidecar identity

**Experiment Data File**:
一個 Experiment Result 的正規資料檔，稱為 data.h5。多個 Data Variables 屬於同一份正規資料。
_Avoid_: sidecar file, companion file, Labber export

**Labber Dataset**:
以 Labber 資料模型表示 axes 和 measured values 的衍生輸出，供 Log Browser 檢視。它不是 Experiment Result 的正規保存權威。
_Avoid_: canonical data file, HDF5 blob

**Legacy Measurement Artifact**:
在目前 Experiment Data File 格式前保存的量測成果。它屬於明確遷移的來源，不是正常載入格式。
_Avoid_: supported old format, compatibility path

**Data Variable**:
Experiment Result 中具名的量測變數。各變數可以有不同 axes 和 shape，仍共用同一結果身分。
_Avoid_: file suffix, member name

**Grouped Experiment Dataset**:
同一個 Experiment Result 的多個 Data Variables 所形成的資料群。Labber 的 grouped 表示是此資料群的衍生輸出。
_Avoid_: multi-file workaround, sidecar artifact

## 分析與 waveform

**Flux-Dependence Analysis**:
利用隨 flux 變化的 spectra、選定 spectral features 與 flux alignment，產生供後續量測和 dispersive analysis 使用的 fluxonium fit handoff。
_Avoid_: fluxdep widget, spectrum editor, point-picking UI

**Fluxonium Prediction**:
根據擬合後的 fluxonium parameters 與 value-to-flux alignment，計算 transition frequencies、matrix elements 或 dispersive resonator frequencies 的模擬模型。
_Avoid_: GUI predictor service, dispersive tuning widget, plot overlay

**Arbitrary Waveform Asset**:
可被 arbitrary waveform cfg 引用的 qubit-scoped samples 資產。它一定有播放資料，也可以內嵌可編輯的 Formula Recipe。
_Avoid_: waveform cfg, formula file, arbitrary pulse

**Arbitrary Waveform Data Key**:
Arbitrary Waveform Asset 的安全路徑身分，也是 asset filename stem 與 cfg 引用字串。
_Avoid_: filename, module name, waveform name

**Formula Recipe**:
Arbitrary Waveform Asset 內選用的可編輯來源定義，可包含多個 Formula Segments。匯入的 asset 可以只有播放資料。
_Avoid_: cache metadata, formula cfg

**Formula Segment**:
Formula Recipe 中具有獨立 duration 和 expression 的有序片段。片段使用各自的局部時間，串接後形成完整 waveform。
_Avoid_: piece, section, pulse part

## 操作憑證

**Permit**:
呼叫受保護 GUI 操作前，對該次請求已檢查的前置條件憑證。它不表示操作期間的硬體資源仍然可用，也不需釋放。
_Avoid_: lease, operation handle

**Exclusion Lease**:
操作啟動時取得、在終局釋放的動態互斥佔用。它與操作追蹤用的 handle 不同，也不代替請求前置條件檢查。
_Avoid_: permit, operation handle
