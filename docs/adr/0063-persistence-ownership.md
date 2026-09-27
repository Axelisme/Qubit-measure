---
status: accepted
---

# Persistence：保存權威與資料契約

## 脈絡

App memento、Experiment Data File、workflow run artifact、`params.json`、SampleTable 和 waveform asset 都會落盤，但保存目的不同。以檔案格式統一它們，會讓通用讀寫層決定量測語意、完成條件或使用者路徑政策。本決策依保存目的分配權威，不建立統一的 persistence service。格式、欄位和各類資料的操作細節由各 owner 維護。

## 決策

### 保存目的與 owner

- App memento 保存偏好和工作狀態，不是量測備份。各 app 擁有 schema、version、capture／restore 和觸發時機；`gui.session.persistence.SingleFileCaretaker` 只提供單檔 I/O、default degradation 與 replace。measure GUI 目前在 lifecycle close 保存，autofluxdep GUI 另有 debounce 與 terminal 保存；shared caretaker 不規定 close-only，也不訂閱高頻事件代替 app 決策。
- Experiment Data File 保存一個 Experiment Result。`experiment.axes_spec` 定義 typed Result／cfg 與 persisted axes、units、roles 和 metadata 的 mapping；`datafile` 擁有通用檔案格式與讀寫，不反向依賴 experiment。正常載入在 experiment 邊界還原 typed Result，不把 generic role mapping 洩漏給分析端。
- Workflow Run Result Artifact 保存跨 node／flux 的資料與 audit。autofluxdep 的 lifecycle／`RunStore` 擁有 run-scoped artifact；node 經 observer 與 store 提交結果，不直接管理 HDF5。它不是單一 Experiment Result。
- `params.json` 是跨 workflow 的 typed parameter handoff。`QubitParams` 擁有 section 更新與 project identity；不把 sample arrays 或 dense curves 塞進參數檔。generic table storage 不負責這類語意。
- SampleTable v2 是跨 producer／consumer 的座標交換契約。`resources.sample_table.schema` 擁有座標、單位及解析語意；`SampleTable` 只存 schema-free CSV。
- Waveform asset 擁有播放 arrays、reference time axis 和 duration。ModuleLibrary waveform cfg 只保存 asset key，不另存可覆寫的播放長度。program 在完整 asset duration 取樣到硬體 timing；調整時間內容須改 asset 或 recipe，而不是在 cfg 伸縮或任意裁切。

### 資料表示與完整性

Experiment persistence 使用 inner-first axes；disk payload 由 Result-native shape 對應，save／load 不要求 caller 補 transpose。物理單位由 experiment mapping 明示；不能為未定義物理量的 scalar 捏造 A／V。Dataset Role 是結果語意，state／phase 等離散座標仍是 axis。單一 Result 的 canonical one-shot grouped file 使用共同 grid、一份 shared metadata 與明確 role-to-channel mapping；必需 roles 由 experiment 決定，不由 generic writer 猜測。異質 streaming workflow 有獨立 layout 與 completeness 規則，不能因同為 HDF5 就視為同類檔案。

結構完整、量測完成與持久提交不同。Complete Experiment Result 指所需 roles 齊全，不保證所有預定測點完成。Operation stopped／failed 不單獨決定已提交資料能否分析。Autofluxdep 在 node 正常返回、patch 驗證後提交 row；mid-node exception 不把半列視為 committed。已取得 measurement 而 fit／provide 失敗時，保留 raw measurement 並分別標示提供結果的狀態；skip 以 audit event 表示，不以 NaN 推斷。Node-row 與 flux-level commit 是不同進度；partial／stopped／failed run 仍可保有先前已提交的 evidence。Terminal exports／reports 取用 committed journal evidence，不能反向覆寫 canonical row。

### 失敗、路徑與遷移邊界

Memento read／decode 失敗可回 fresh default，並回報 restore outcome；restore／apply 程式錯誤仍傳播。Caretaker 不懂 cfg，也不自行決定何時保存。量測資料、參數和未知格式版本不能沿用 memento 的 whole-file default degradation：必要 role、shape 或 unit 不符時，回報可定位的錯誤而非偽造空結果。

Autofluxdep artifact 建立失敗不啟動 run；canonical row write／flush／journal 失敗會使 run 失敗，並保留此前已 flush 的證據。Terminal export／report 失敗要顯示，但不把成功量測 row 改標為 measurement failure。Stop 仍嘗試 finalize；取消請求不保證 finalize 成功，也不表示 rollback。單檔 replace、HDF5 flush 和 journal append 不是跨檔交易或掉電持久保證；不可從 manifest 的 accepted 標記推導任意 crash 後可安全 resume。跨檔失敗窗口的讀取、修復和 resume 規則尚未驗證。

Experiment saver 寫 caller 指定的 exact path，若檔案已存在則拒絕；需要 unique filename 或覆蓋政策時，由 GUI／runner／notebook 在呼叫 saver 前決定 final path。Run identity 由 run-scoped metadata 記錄，不由可讀 slug 或檔名猜測；manifest 連結 metadata root 和 heavy-data root，不把 exports 當另一份原始權威。

Normal runtime 只接受各自契約指定的格式，不猜測 legacy shape／unit；版本必須與 artifact kind 一起判斷，合法 streaming v1 不因數字小於 one-shot v2 而視為同類 legacy。Experiment 的舊 converter 和 GUI fallback 已退休。SampleTable legacy 資料只能由 caller 明確呼叫 pure DataFrame conversion；此函式不自動改寫 CSV。QubitParams 的 project identity migration 由其 owner 管理，不因本 ADR 授權寫入使用者資料，也不建立跨資料類型的共用 migration layer。

### 共享資料與引用

QubitParams 的 caller 以 typed 語意方法讀寫；更新一個 section 保留其它已知和未知 section。Timestamp 只表示 section 修改時間，不是 dependency provenance 或自動失效。單一 typed module 不等於跨程序互斥，也不代表 mtime 同步可解決多 writer 衝突。

SampleTable v2 明示 `dev_value` 的 A／V base unit，flux 與 row frame 由獨立欄位表示。Flux 解析依 explicit、row-frame、fallback-frame 順序，並標明每列來源；fallback frame 必須與該列 unit 一致。自描述 unit 不保證缺少 flux／frame 的列能不靠外部輸入算出 flux。分析用 correction／alignment 不默默寫回原始樣品資料，也不自動修正 explicit flux。Autofluxdep export 用 run 建立時的 unit snapshot，不以 export 當下的 live device state 重讀舊結果。

Asset repository 的 rename／delete 只處理資產，不掃描或改寫 ModuleLibrary 的 nested references。舊 key 可能因此成為 dangling reference；使用端解析缺失資產時明確失敗，由 caller／使用者修正引用。這避免 repository 反向理解 experiment cfg，也避免跨 owner 批次寫入。凍結 cfg key 不等於凍結該 key 所指的 asset bytes；本決策不引入 asset versioning、copy-on-run 或自動引用遷移。

## 後果

維護者先找保存對象的 owner，再查其就地格式契約：[`datafile`](../../lib/zcu_tools/datafile/README.md)、[`experiment`](../../lib/zcu_tools/experiment/v2/README.md)、[`QubitParams`](../../lib/zcu_tools/resources/qubit_params.md)、[`SampleTable`](../../lib/zcu_tools/resources/sample_table/README.md)、[`waveform assets`](../../lib/zcu_tools/resources/waveform_assets.md)、[`program waveform`](../../lib/zcu_tools/program/v2/README.md)、[main GUI](../../lib/zcu_tools/gui/app/measure/README.md) 與 [autofluxdep](../../lib/zcu_tools/gui/app/autofluxdep/README.md)。本 ADR 不保證多檔 all-or-nothing、crash safety、多 writer 協調、autosave、resume、browser 或任何使用者資料遷移。
