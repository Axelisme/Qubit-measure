# `zcu_tools.resources.context` point-local module library

**Last updated:** 2026-10-21

**讀者：** framework-dev。Point-local module library 的讀取、編輯與來源契約。

本 package 擁有 point-local `ModuleLibrary`。條目、setup 與 point 參數由 [entry owner](../entry/README.md) 管理；GUI 的切換與命令由 ContextService 編排。Library 使用 typed 文件，呼叫 `refresh()` 明確同步磁碟內容。

## Module library

`library.py` 擁有 `module_cfg.yaml` 的文件、查詢、編輯與來源，`library_store.py` 集中 library handle 與 ResultEntry clone 共用的磁碟讀取政策。文件格式是 `zcu.module-library` 2.0。Header-only 文件表示合法空 library。

Library 版本只跟 header／容器結構走。Module／waveform Spec 欄位演進不自動升版，library 維持 2.0（ADR-0072）。其他 major 仍在 header 拒絕，不遷移、不加相容讀取。Forward minor 保留容器層未知欄位，不放寬 entry 的 extra 規則。

Library 讀取先驗容器，再逐條驗證四個分組的 Spec 與所屬來源。壞條目及其 provenance 被移除，主檔只留合法條目與來源。清理候選完整驗證後，DocumentStore 在同一 sidecar lock 內逐 byte 複製完整原檔成同目錄時間戳 .bak，不覆寫舊備份，再 atomic replace 主檔。每次成功清理只寫一行 warning，包含備份路徑及移除 roots，不新增 GUI／MCP 顯示。

這是一般讀取不寫檔的 library 例外（ADR-0063），涵蓋 open、refresh、edit 進入與 commit 重讀。合法檔與 header-only 不重寫，也不重複備份。Header／容器或不可歸屬的 provenance 錯誤整份拒絕，不備份、不清理。Ledger I/O、schema、附檔與非預期錯誤直接傳遞。備份失敗不替換主檔；replace 失敗保留原主檔及已完成的備份，不發布候選。這不是跨檔交易或掉電恢復。

Readout 使用獨立、平鋪的 `readout/pulse` Spec，waveform 可保留 key 引用，見 [ADR-0074](../../../../docs/adr/0074-flat-readout-module.md)。舊 nested readout 沿壞條目政策備份並移除，不遷移或相容讀取。

Library 的設計是「元件 → 分組 → pydantic model」，分組名稱不是封閉清單。pulses、readout、reset 與 waveforms 是目前使用的四個分組。

- 現行 `ModuleLibraryDocument.components` 與 `library_key_kind` 仍只接受這四個分組。日後需要第五種分組時再放寬實作限制，不另改設計。實際 key 是 `Q1.pulses.pi` 這類三段路徑，不是角色或隱藏 namespace。
- 文件存 `ModuleSpec`／`WaveformSpec`。Spec 保留 inline child 或具體 key 引用，不存 provider、asset bytes 或 runtime object。
- `LibrarySnapshot` 是不讀檔的隔離投影，也是 core factory 的 `ModuleResolver`。Spec 查詢不展開引用。成品查詢以固定 snapshot 與顯式 asset capability materialize cfg。
- `entry_value` 提供未 materialize 的工作單位 Spec mapping。`entry_state` 同時提供相對 leaf 的原始來源，供固定 proposal 比對，不補缺失來源。
- Snapshot replacement、delete 與 rename 都返回新候選，不更新原值或文件。Replacement 接受 manual 或明確 Provenance，接受新 leaves 並清除 clone origin；純快照不查 ledger，寫入 owner 在安裝記憶體候選前驗證本地來源。純 rename 搬移 Spec、reference 字串與全部來源，不求值 child、不掃描其他引用或 cascade。
- `ModuleLibrary.edit()` 使用 `DocumentStore.edit()`。Body 內的 `LibraryEdit.snapshot()` 反映前項寫入。結構操作可合成隔離候選，再由 transaction 驗證並一次替換內容。新 draft 嚴格驗證，不清掉 caller 壞值後當成功。Body、schema、source、conflict 或 I/O 失敗不發布候選，但不回滾先完成的磁碟讀取清理。
- 值與來源同檔提交。普通 replacement 記 manual 並清除 clone origin。明確來源沿 `entry.provenance` 驗證本條目 ledger 事件與附檔，不另設 producer。
- `snapshot()` 只複製已發布記憶體。`refresh()` 顯式重載；`subscribe()` 在 publication 後通知。Handle 不自動監看、mtime sync 或建立缺失文件。

舊 flat 文件由 root `migrations/` 的離線工具轉成現行文件後再使用。Library 與參數文件各自管理 snapshot 與寫入。

Caller 透過 `ResultEntry.module_library(label)` 開啟既有 point 的 library；建立 point、seed 與 clone 由 ResultEntry 負責。Clone 先清理來源，再複製清理後文件，只匯入保留的合法 provenance。合法 Spec 若引用已移除 key，仍在使用該 reference 的原欄位報 missing-key 錯誤，不 cascade、不改 reference、不補 defaults。固定 LibrarySnapshot 不因磁碟清理自行改值。

## Cfg materialization

Experiment cfg materialization 的接合由 `zcu_tools.experiment.cfg_assembler` 負責。Caller 提供固定 library projection 與 device snapshot。GUI 的 schema／expression lowering 由 adapter owner 完成。Notebook 的顯式 `CfgEnv` 選定 library snapshot 與 device manager，讀取失敗直接傳播。更新 library 後，caller 重新捕捉 snapshot 並建立 CfgEnv。
