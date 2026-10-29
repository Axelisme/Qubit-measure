# `zcu_tools.resources.context` point-local module library

**Last updated:** 2026-10-29，缺席 library 與明確空庫初始化

**讀者：** framework-dev。Point-local module library 的讀取、編輯與來源契約。

本 package 擁有 point-local `ModuleLibrary`。條目、setup 與 point 參數由 [entry owner](../entry/README.md) 管理；GUI 的切換與命令由 ContextService 編排。Library 使用 typed 文件，呼叫 `refresh()` 明確同步磁碟內容。

## Module library

`library.py` 擁有 `library.yaml` 的文件、查詢與編輯。`library_store.py` 擁有磁碟讀取與 clone 複製。
文件格式維持 `zcu.module-library` 2.0，header-only 文件表示空 library。
`ResultEntry.module_library(label)` 與不傳 label 的 entry library 都允許缺席，回空快照與 revision=None。
讀取不建立檔案或父目錄。
版本只跟容器結構走，不隨 Spec 欄位演進升版，見 ADR-0072。Forward minor 保留容器層未知欄位。

讀取不寫檔、不備份，也不查 ledger。舊 `provenance:` 不論內容都忽略。
Library owner 先驗 header、容器與可定址名稱，再逐 root 驗證 Spec。
無效 Spec 不進入 snapshot，`snapshot.invalid_entries()` 回傳 root 到驗證原因的隔離 mapping。
其他合法 root 照常使用。Header、容器與 I/O 錯誤仍傳遞。
`refresh(strict=True)` 不略過無效 Spec，失敗保留原 snapshot 與 revision。
任何讀取及空 edit 都保留原 bytes。非空 commit 寫回目前序列化，移除舊 provenance 與已略過的無效 Spec。

現行分組為 pulses、readout、reset 與 waveforms，key 是 `Q1.pulses.pi` 這類具體三段路徑。
文件存持久 ModuleSpec／WaveformSpec，保留 inline child 或具體 key 引用，不存 provider 或 asset bytes。
Readout 使用平鋪的 `readout/pulse` Spec，見 ADR-0074。舊 nested readout 依無效 Spec 政策略過。

`LibrarySnapshot` 是不讀檔的隔離投影，也是 core factory 的 ModuleResolver。
Spec 查詢不展開引用；成品查詢使用固定 snapshot 與顯式 asset capability。
`entry_value` 回傳未 materialize 的 Spec mapping。
Replacement、delete 與 rename 返回新候選，不更新原 snapshot。
Rename 不求值 child、不改其他引用，也不 cascade。Library 不產生來源、時間、stderr 或 clone 標記。

`LibrarySnapshot.projection(segments)` 只驗所選 document subtree。選完整 component、partition 或 Spec 時仍驗證該層完整 model，選 leaf 時保留該欄位規則。`summary(describe)` 只驗輸出的 scalar 與描述 sequence 所需的結構，不預驗隱藏的 elements。兩者都不展開 references、不讀 assets，錯誤不變空結果。

`LibraryContent` 保存量測檔中的 opaque library 內容。`LibrarySnapshot.from_content` 不讀檔、不預驗未使用的 Specs。查詢一個 root 時才解碼該 Spec，列出 keys 時只讀取該 family 的名稱。缺少或格式不符的 used section 報錯，不代換成空 library。

Library readout 的可選 `classifier` 保存 circle、linear 或 centroid3 分類校準，mux 各 tone 獨立保存。Centroid3 的 g/e/f 中心與必填 training confusion 隨 snapshot 與重開保留，不另建校準儲存。複數中心沿用 `{__complex__: [re, im]}` tag，library 維持 2.0。舊條目沒有 classifier 時讀為 None，讀取與無關編輯不補寫新 key。

`ModuleLibrary.edit()` 沿用 DocumentStore 的單檔交易。Draft setter 嚴格驗證新值。
Body、schema、conflict 或 I/O 失敗不發布候選。Body 中的 snapshot 反映前項寫入。
`snapshot()` 只複製記憶體，`refresh()` 顯式重載，`subscribe()` 在 publication 後通知。
未變的 bytes 沿用驗證快取，不另查外部依賴。Handle 不監看磁碟。
缺席 library 的空 edit 不建檔，第一次非空 edit 在原鎖與單檔交易內建立文件。
外部先建立文件時，edit 報 ConflictError 且不覆寫外部 bytes。
`refresh(strict=True)` 可安裝外部新建的合法 library，無效候選保留已發布內容。

Caller 透過 `ResultEntry.module_library(label)` 開啟 library，建立 point 由 ResultEntry 負責。
Inherit 與 clone 都經 copy_library 複製合法 Specs 與 forward-minor 容器欄位，不改來源檔、不匯入事件或附件。
來源缺 library 時，目的也保持缺檔。
`library_store.create_empty_library(path)` 明確建立空庫，只供 caller 自有且未發布的目錄使用。既存目的以 FileExistsError 拒絕，不覆寫內容。離線遷移用它固定空庫 baseline；普通 empty point、讀取與空 edit 仍不建立 library。
缺失的 child reference 在使用該欄位時報錯，不 cascade、不補 defaults。
舊 flat 文件由 root `migrations/` 的離線工具轉換後使用。

## Cfg materialization

`LibrarySnapshot.get_module()` 與 `get_waveform()` 透過 core factories，把固定 snapshot 的 Spec 轉成執行用 cfg；asset 讀取使用 snapshot 綁定的 capability，引用或讀取失敗直接傳播。GUI 的 schema／expression lowering 與 core cfg 轉換由 [gui.form_adapter](../../gui/form_adapter/README.md) 負責，使用捕捉的 library 與 device snapshots。

Notebook caller 直接建立實驗的 typed cfg，需要 library 成品時可查詢固定 `LibrarySnapshot`。更新參數或 library 來源後，caller 重新捕捉所需 snapshot 並重建 cfg。Notebook 的 DeviceManager 綁定與 Run 前 device snapshot 見 [Notebook 入口](../../notebook/README.md#共用責任與目前邊界)。`experiment.cfg_assembler` 只提供 Run mapping 的共用值型別 `ExperimentScalar` 與 `ExperimentCfgValue`。
