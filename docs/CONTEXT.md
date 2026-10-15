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
實驗用途中的位置，例如 qubit 或 resonator。角色不是元件本身，明確的元件 consumer 可以另要求 kind。
_Avoid_: component name, module name

**Alias**:
可編輯的有序名稱對應，值是元件或 device 名稱，也可以引用前列的軟連結。條目保留原始對應，一次 Run 記錄當時的解析名稱。
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

**Simulation Environment（模擬環境）**:
供模擬量測使用的物理模型、模擬控制資源及其關係的組合。它描述模擬樣品與可操作環境，不等於前端的 SessionEnv 或某次實驗 cfg。
_Avoid_: SessionEnv, RunContext

**Simulation Preset（模擬範本）**:
可重用、以參數建立模擬環境的具名組合，例如 FTF preset。它涵蓋整個模擬配置，不是單一 pulse 或 readout module 的起始範本。
_Avoid_: ModuleLibrary Template

## 量測與保存

**Experiment Run**:
使用特定 cfg 和硬體狀態執行一次實驗。一次 Run 產生一個 Experiment Result。
_Avoid_: job, task run, measurement job

**Device Snapshot**:
一次 Experiment Run 開始前的全體已連線儀器設定記錄。它表示實驗前狀態，不是該次 sweep 的實際測點值，也不是模擬每 round 的 Environment Snapshot。
_Avoid_: cached observation, setup result, sweep values

**Simulation Acquire Request（模擬量測要求）**:
一次 acquire 內固定的控制與資料需求，描述要執行的量測，不包含逐 round 改變的環境值。
_Avoid_: live program cfg, environment snapshot

**Simulation Environment Snapshot（模擬環境快照）**:
一個 round 執行時固定的環境值與設定識別。它與該次 acquire 的量測要求及 round 抽樣身份共同構成模擬輸入，見[模擬環境 ADR](adr/draft/simulation-environments.md#每-round-捕捉輸入不承接前次量子態)。
_Avoid_: Run-level Device Snapshot, live device reference

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

## Agent 介面

**Agent 介面層級**:
Agent 可使用的入口依抽象程度分成實驗環境、GUI／Notebook、recipe，以及 workflow 與跨實驗分析四層。層級描述自動化程度，不是權限。
_Avoid_: operation layer, control level

**Recipe**:
把一種常用單次實驗包裝成少量參數的自動 GUI 操作，其餘參數自動推導。Agent 把它當成一個整體，控制輸入、接受輸出，中途只在 recipe 宣告的判斷交接處協作。
_Avoid_: workflow, macro, script

**判斷交接（Decision Handoff）**:
自動流程在預先宣告的位置停下來，把需要判斷的資訊與選項交給 agent 或使用者，例如缺少必要參數、互動分析或寫回確認。
_Avoid_: error, prompt, interrupt

**Workflow**:
一次設定後執行跨多次 Experiment Run 的流程，例如 overnight 或 T1 over flux。中途只在 workflow 宣告的判斷交接處與 agent 協作。
_Avoid_: recipe, batch script, DAG plan

**跨實驗分析**:
整合多次量測，甚至多個 qubit 的已保存資料，做單次實驗無法完成的聯合分析，例如 Flux-Dependence Analysis。
_Avoid_: single-run analysis, tab analysis

## 操作憑證

**Permit**:
呼叫受保護 GUI 操作前，對該次請求已檢查的前置條件憑證。它不表示操作期間的硬體資源仍然可用，也不需釋放。
_Avoid_: lease, operation handle

**Exclusion Lease**:
操作啟動時取得、在終局釋放的動態互斥佔用。它與操作追蹤用的 handle 不同，也不代替請求前置條件檢查。
_Avoid_: permit, operation handle
