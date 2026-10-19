# `migrations.storage_migration` 離線遷移

**Last updated:** 2026-10-20，磁通分組、分析格式檔、樣本發布與 native 歷史保真

本 package 組合 legacy result／Database 與新參數條目。正常 runtime loader 不使用它。公開入口是 `migrate_storage` 與 `load_run_evidence`，型別從 package root 匯入。

- 組合根提供 mapping、entry 的共用 kind 註冊、planning 前的 concrete cfg validation 與發布後的 native validation callback。本 package 不 import `zcu_lab`、experiment 或舊資源容器。`lib` 不 import `migrations`。
- Native schema 與兩個 callbacks 以 `(source tag, cfg_type)` 的明確歷史身分配對。缺 cfg identity、無對應 schema 或 typed-invalid cfg 時只 pending，不回退 tag-only。`native_tags` 只轉換新 RunMetadata 的 tag，不改歷史 raw 或正常 loader。
- `exps/<sub>/` 與舊直屬 child 都轉成同名 point，沿用 parameters mapping 與 pending 規則。兩種來源同名就拒絕，報告列出兩個來源，不合併或覆寫。
- `exps/<sub>/image/*.png` 經 copy、hash、manifest 與 resume 發布到 `result/<entry>/points/<sub>/figures/<原檔名>`。Data-only 不建立缺少的 point，而將圖片列為 located pending；parameters 建 point 後可 resume data 完成發布。`exps` 外的 `image/` 檔案發布到 entry 的 `figures/<來源相對路徑>`。`autofluxdep_runs` 內的圖片與其他 workflow 檔仍進 preserved，不誤掛到 entry figures。
- `params.json` 以本次建立的唯一 qubit component 為對象，找不到唯一對應就整份 pending。Converter 先將原文區塊與 metadata 存入 setup 的 `component.ext.legacy_params`，再建 points，讓 points 繼承完整 YAML。根層與各 point 的 meta_info 值優先，params 只補缺；衝突保留 meta_info 值，並報告兩來源與 pending。
- `fluxdep_fit` 寫入 entry 的 `predictors/<component名>.json`，不寫 component fit 欄位，也不複製到 points。Predictor 格式 owner 是 `resources.predictor_file`。此 converter 在該 owner API 可用前直接寫契約規定的 JSON。檔案使用 `format=zcu.predictor`、整數 `version=1`，params 的欄名與有限數值遵循 simulate Params.from_mapping，完整性與物理限制交給 build_predictor。
- 舊 EJ/EC/EL fit 寫 `model=fluxonium`、`flux_bias=0.0`，`flux_int` 只作 alignment，不寫 predictor。舊 owner 未定義的 transmon 或明示 model fit 保持 pending。predictor 單位只讀 params 根層或 fit 的明示 `flux_unit`。兩處都沒有時才可用 `assume_flux_unit`，報告標「假設單位」。未提供假設，或明示值衝突、不合法時，單位寫 null 並列 pending。meta_info 不能作 predictor 單位證據，也不換算數字。`source.run_id` 固定 null；`source.created_at` 只接受可證實且含時區的 ISO 8601，否則 null 並列 pending。這三個 nullable 欄位仍必須存在；報告保留備份內原文定位。
- `dispersive` 與 `fluxdep_fit.plot_transitions` 經 resources owner 寫入 entry 的 `dispersive/<component名>.json`，`t1_curve_fit` 寫入 `t1_curve/<component名>.json`。只有 transition seed 時也產生 dispersive 檔，不補 g 或 bare_rf。兩檔保留缺項、null 與 active T1 名稱的差異，source 時間只採各自 section 的含時區 timestamp。無法驗證的內容保持 located pending，原文、未知欄位與 metadata 仍留在 setup ext 並由 points 繼承。Parameters 未完成時 resume 經 owner load 比對欄位存在性與內容，衝突就拒絕；完成後保留使用者編輯。Dry-run 列出相同檔案但不寫入。
- Parameters 與 data 可分開執行。第二部分明確 resume 同一 manifest，不合併任意既有條目。完成的 setup／point 仍可編輯；resume 驗證當前文件，不恢復舊值。
- Legacy evidence 1.x 的 params 與來源稽核使用本 package 的 `LegacyParameterSnapshot`、
  `LegacyParameterSource`、`LegacyCloneOrigin`。Wire shape 不變，完整 raw evidence 留在 manifest。
  Converter 產生 native 2.1 簡單 context，不把歷史 params 投影到 native。
- Legacy evidence 的 started_at 與 run_id 可缺省或為 null。開始時間無證據時，native 寫 started_at=None、completion=unknown、finished_at=None，即使原 evidence 留有完成狀態或結束時間也不宣稱已知生命週期；完整 raw 留在 manifest。已知開始時間時保留可證實時間與完成狀態，完成狀態缺省為 unknown。
- 明示可信 run_id 須通過 datafile 的公開 `validate_run_id`。缺 ID 時依 entry 的 `new_run_id` 分配，報告標「遷移賦值」。分配時間不是歷史量測時間。Manifest 在 native publication 前固定來源/hash、ID 與輸出分配時間，resume 不重配。非法的非空時間或 ID 仍拒絕，不當成缺失。
- Manifest 保存固定 entry／run 身分、來源 baseline、完整 evidence raw 與逐檔 publication 狀態。持久化 run 分配即固定該來源的完整 evidence，不必等到 native publication；resume 在任何 checkpoint 或 report 寫入前拒絕變更，避免把遷移 ID 改稱歷史 ID。Report 是累積快照。兩者保留同 major 的未知 JSON 欄位。
- Native 的 `labber_path` 保存相對 entry Database 根的原檔保存路徑，例如 `Labber/2025/10/scan.hdf5`。`Instrument config` 透過 datafile owner 的唯讀 codec 轉為 device_snapshot，只用舊檔中的值。不能表示的 snapshot 列 `device snapshot pending`，該 run snapshot 為 None，但量測資料仍轉 native，原設定留在保存的 Labber。
- Native publication 與驗證完成後才搬移對應 Labber。搬移先複製、驗證雜湊，再刪來源。Source_removed 只驗證目的 hash，舊位置重用不改既有 run，也不刪新檔。除開始時間與 run_id 外，experiment tag+cfg_type、完整歷史 cfg、variable/axes/signal 名稱、單位、dtype、shape 或其他必要 mapping 缺失時，不產生 native 或逐檔搬移，只列 pending，不猜名稱或補量測時間。兩部分都完成後，這些舊檔隨來源清理，只保留在外部備份。
- 新 entry 可與舊資料共用 result／Database root，只要目的與本次兩個舊來源樹互不包含。目的已存在就拒絕，不覆寫其他 entry；report 也不得寫入本次來源樹。
- 正式執行在建立 entry 前完整複製本次舊 result 與 Database 到 repo 外的 `<name>-backup`。`backup_parent`／CLI `--backup-parent` 指定外部上層，預設為工具 checkout 的上層。`repo_root`／`--repo-root` 可明示被保護的 checkout。備份內 `result/` 與 `Database/` 保留原相對路徑與空目錄，`sha256.json` 列出每檔 SHA256。既有備份目錄、repo 內或與來源／目的重疊的路徑一律拒絕。Symlink 與非一般檔案不支援，拒絕後不進入遷移。
- `backup.py` 擁有完整複製、清單驗證與清理。Manifest 綁定清單 SHA256 及來源完整 hashes；每次 resume 驗證既有備份，不重做。已搬移或已清理的來源不必存在。未完成的備份不授權任何遷移；首次備份之後、entry 建立之前若中斷，須人工核對並移走該備份才可重新開始，工具不認領既有目錄。
- Parameters 與 data 都完成、無未完成 publication、無 `point figures` 暫待項且無 failure 後才清理。本次舊 `result/<chip>/<qubit>/` 與 `Database/<chip>/<qubit>/` 會刪除。永久 pending 可留下，但報告位置改指向備份。workflow 與其他 preserved 副本不是正常 runtime 資料；已轉換的 JSONL samples 與 figures 使用各自的新位置。
- 清理前重驗備份與完整來源清單。來源新增、變更或非工具搬移造成的缺檔會拒絕清理。先持久化 `cleaning`，再逐檔核對 hash、刪檔並移除空目錄。中斷後 resume 只收尾清理，允許已刪的備份檔缺席，不重跑轉換。不承諾並行 writer 或掉電交易；來源有其他 writer 時不得執行。Dry-run 只規劃，連備份目錄與清單都不建立。
- Parameters 將根層與 child point 的 module／waveform 轉成 `cfg_model` 的正式 Spec。根庫透過 `ResultEntry.module_library()` 提交到 entry 的 `module_cfg.yaml`，slots 寫 setup。Point 庫透過 `module_library(label)` 提交，slots 寫 point。Mapping 明示來源 family 與 canonical key，typed reference remap 由 `cfg_model` 擁有。新 point 的預設 library 繼承規則不變。
- `qub_ch` 只在同檔沒有 `qub_0_1_ch` 時映射 `Q.wiring.ch`。Pi 脈衝 scalar 找不到成功轉換的對應 module 時，按原 key 保存到 `component.ext`。已轉換的 params 不再複製成 entry 層 `params.json`；原文仍完整留在外部備份。
- Component 的 global 磁通組包含 `cur_A`、`flx_half/period/int`、`flux_bias`，單位寫 `general.ext.flux_unit`。來源順序是同檔 `flux_unit`、根 meta_info 同 key、params 根 `flux_unit`、operator assumption。JPA 組只採 `jpa_flux_unit`，依同檔、根 meta_info、獨立 `assume_jpa_flux_unit` 順序。JPA 假設只接受 A/V，未提供就保持 pending，不借 qubit params 或 global 假設。CLI 這批資料明填 `--assume-flux-unit mA --assume-jpa-flux-unit V`，報告分別標「假設單位」。報告寫實際來源檔。根 meta_info 與 params 同層明示單位衝突時，整組 pending，不用假設解衝突。
- Point 自己明示的單位優先。單位改變時，移除未被 point 覆寫的同組 setup 繼承數值及其 metadata。Nullable typed 值設 null，ext 值從 mapping 移除，metadata 依 owner 契約清為空。候選值先過 owner 驗證，再與新單位同筆 edit 提交。其他組不變。`local_flux_unit` 只照抄到 `Q.flux_unit`，不作任何數值證據或清除依據。Owner 不接受來源單位時，該組 pending，不換算。
- Parameters 將根 `samples.csv` 交給 `sample_csv.plan_sample_csv`。唯一 qubit component 決定 `ResultEntry.samples(component)`，找不到唯一元件就 located pending。Legacy CSV 的 `sample_value_column` 與 `sample_source_unit` 必須獨立宣告，不能借 `assume_flux_unit`。合法列寫入同目錄 temp JSONL，再走 manifest 的 samples publication、hash 與 prepared/published resume；不直接對既有目的地 append。無法轉換的列按資料列號 pending。沒有合法列就不建 JSONL。
- `MigrationReport.declarations` 記錄上述四個 operator 輸入，manifest 保存同一 typed report。Resume 在任何寫入前核對這些宣告，與首次不同就拒絕。已發布 samples 的 hash 改變也拒絕，不覆寫使用者編輯或重複追加。Dry-run 呼叫相同 CSV 規劃與元件驗證，不建 entry、temp、樣本表或備份。
- 未支援 expression、Spec、缺 mapping 或 child reference 不能轉換時保留來源與定位，不 evaluate。Arb asset key 保持原值，不搬 asset，也不保證全部模組已可 materialize。
- Library dry-run 驗證隔離候選，不取得目的 handle。其 report rows 保留 canonical `new_path`，`new_file=None` 本身不表示 pending。正常提交後 manifest 保存來源與目的 hash，resume 核對已記錄完成的 library 而不重做。

`state.py` 集中 report／manifest encoding 與 copy、native、samples、move 共用的 publication recovery。ModuleLibrary 的單檔提交與 manifest 不是跨檔原子。Library 已寫入但未記錄完成的窗口沒有專門 recovery，需取得清理授權後刪目的地重跑。本 package 不提供掉電保證、硬體操作或多程序 writer 協調。

`migrations/migrate_storage.py` 是離線組合根。具體 mapping 與固定 native declarations 位於同目錄上層的 `migrations/storage_mapping.py` 與 `migrations/migration_experiments.py`，沒有動態 discovery 或新 runtime registry。單獨 Parameters 模式不執行 data 搬移；只有同一 manifest 的兩部分都完成才可清理，包含第二部分以 parameters resume 的情況。操作授權、來源保全、pending／resume 與修改 kind 的步驟見 [Agent 離線遷移說明書](../../docs/storage-migration.md)。
