# `zcu_tools.resources.context` — named experiment work contexts

**Last updated:** 2026-10-11，刪除舊 context 實作

本 package 管理 point-local `ModuleLibrary` 的 typed 文件。參數文件與工作點身分由 `resources.entry` 管理；session 的固定參數讀取由 `ContextService` 提供。

## Module library

`library.py` 擁有 `module_cfg.yaml` 的文件、查詢、編輯與來源。文件格式是 `zcu.module-library` 1.0。Header-only 文件表示合法空 library。

- `ModuleLibraryDocument.components` 依元件存放 pulses、readout、reset 與 waveforms。實際 key 是 `Q1.pulses.pi` 這類三段路徑，不是角色或隱藏 namespace。
- 文件存 `ModuleSpec`／`WaveformSpec`。Spec 保留 inline child 或具體 key 引用，不存 provider、asset bytes 或 runtime object。
- `LibrarySnapshot` 是不讀檔的隔離投影，也是 core factory 的 `ModuleResolver`。Spec 查詢不展開引用。成品查詢以固定 snapshot 與顯式 asset capability materialize cfg。
- `entry_value` 提供未 materialize 的工作單位 Spec mapping。`entry_state` 同時提供相對 leaf 的原始來源，供固定 proposal 比對，不補缺失來源。
- Snapshot replacement、delete 與 rename 都返回新候選，不更新原值或文件。Replacement 接受 manual 或明確 Provenance，接受新 leaves 並清除 clone origin；純快照不查 ledger，寫入 owner 在安裝記憶體候選前驗證本地來源。純 rename 搬移 Spec、reference 字串與全部來源，不求值 child、不掃描其他引用或 cascade。
- `ModuleLibrary.edit()` 使用 `DocumentStore.edit()`。Body 內的 `LibraryEdit.snapshot()` 反映前項寫入。結構操作可合成隔離候選，再由 transaction 驗證並一次替換內容。Body、schema、source、conflict 或 I/O 失敗不發布候選。
- 值與來源同檔提交。普通 replacement 記 manual 並清除 clone origin。明確來源沿 `entry.provenance` 驗證本條目 ledger 事件與附檔，不另設 producer。
- `snapshot()` 只複製已發布記憶體。`refresh()` 顯式重載；`subscribe()` 在 publication 後通知。Handle 不自動監看、mtime sync 或建立缺失文件。

舊 flat 文件必須由顯式 migration 產生新文件。Runtime 不提供 alias bridge。Library 與參數文件各有自己的 snapshot／write owner，不提供雙 store memory 交換。

## Cfg materialization

Experiment cfg materialization 屬於 `zcu_tools.experiment.cfg_assembler`，不是 ModuleLibrary 的責任。Caller 提供固定 library projection 與 device snapshot。GUI 的 schema／expression lowering 由 adapter owner 完成。Notebook 的顯式 `CfgEnv` 選定參數與 device manager，讀取失敗直接傳播。
