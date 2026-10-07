# Agent 離線遷移說明書

此文件供執行舊 `result/`、Database 遷移，以及修改 component kind 的 agent 使用。入口是永久保留的 `tools/migrate_storage.py`。它不控制硬體，也不改正常 loader。`lib/zcu_tools/resources/storage_migration/README.md` 說明框架邊界，具體對照位於 `zcu_lab/storage_migration.py`。

## 執行前

1. 取得本次操作授權。列明可讀來源、可寫目的地、可刪除的 Labber 來源，以及 report 的位置。開發任務的合成資料驗證不授權遷移真實資料。真實資料複本也需要使用者指定與授權。
2. 保留獨立備份。工具不修改舊 result 樹，但成功的 data 遷移會刪除已確認的 Labber 來源檔。先記錄兩個來源樹的逐檔 SHA256。
3. 固定使用的 checkout 與 Python 環境。使用該 checkout 的 `uv run --directory <worktree> --no-sync --`，不在遷移過程安裝依賴。
4. 明確選擇 `source-chip`、`source-qubit`、新 `name` 與 `qubit-kind`。前兩者只定位目錄，不承載物理語意。Kind 可選 `qubit/fluxonium` 或 `qubit/transmon`，工具不從名稱猜測。
5. 確認 `results/<name>` 與 `Database/<name>` 都不存在。新目的、report 與 Database 寫入或刪除都不能落入整個舊 `result-root`，也不能包住它。不同 entry 或 symlink 別名仍受此限制。一次只允許一個 writer。工具不提供並行遷移、跨檔交易或掉電恢復保證。

完成本節時，授權與備份有明確位置，兩個目的地都空缺，來源 hash 已保存。任何條件不成立就停下來，不清理 caller 的既有檔案來避開碰撞。

## 先 dry-run

以下命令在 worktree 根目錄執行。將示例路徑換成已授權的絕對路徑。

```bash
uv run --directory <worktree> --no-sync -- python tools/migrate_storage.py \
  --result-root /authorized/legacy/result \
  --database-root /authorized/Database \
  --results-root /authorized/results \
  --source-chip TestChip --source-qubit TestQubit --name migrated-entry \
  --qubit-kind qubit/fluxonium --part all --dry-run \
  > /authorized/task/dry-run.json
```

`result-root` 是舊小資料根，來源為其下的 `<source-chip>/<source-qubit>`。`database-root` 同時是舊重資料根與新 Database 根。`results-root` 是新的小資料根，與舊 `result-root` 分開。

Dry-run 只讀來源，將 report 印到 stdout。它會以具體 cfg model 檢查已知 pair 的可轉換性，但不建立條目或 manifest，不搬移 Labber，也不執行 native validation。Report 有 pending 時仍可能 exit 0。逐項檢視 `key_mappings` 與 `pending`，確認新路徑與單位的證據，不能把 exit 0 當成全部資料已遷移。

工具保留未知值與 expression 的原文。Expression 不會 evaluate。Flux 單位必須有明確證據，`cur_A` 的名稱不是單位證據。缺 cfg identity、實驗 schema、歷史 snapshot、量測時間或 completion 的 run 保留來源，列待處理。

完成本節時，report 中每個 pending 都有後續處置或明確保留理由。對照表及真實複本結果由使用者確認，不能用自動驗證代簽。

## 補充歷史 run evidence

需要補充歷史證據時，準備 JSON，再傳 `--run-evidence /authorized/task/evidence.json`。先用公開 `load_run_evidence(Path(...))` 檢查輸入。文件是 `zcu.migration-run-evidence`，`format_version` 為 `1.0`，`entries` 是逐檔證據陣列。同 major 未知欄位保留，較新 major 拒絕。

每筆 evidence 的欄位與型別由 `LegacyRunEvidence`、`LegacySnapshotEvidence` 的 docstring 定義，從 `zcu_tools.resources.storage_migration` 匯入。至少核對以下資訊。

| 欄位 | 核對來源 |
| --- | --- |
| `source`、`source_hash` | 相對 `Database/<source-chip>/<source-qubit>` 的 path 與實際 SHA256。Path 不得逃出此根目錄。 |
| `experiment`、`cfg` | 明確 persisted tag，以及完整歷史 `CfgSnapshot.values`、`cfg_type`、`schema_version`。Schema 依 `(tag, cfg_type)` 配對，同 tag 不代表同 cfg。 |
| `started_at`、`finished_at`、`completion` | 可證明的 UTC ISO 量測時間與 `complete`、`partial` 或 `stopped`。Finished time 可為 null，不用 mtime 或 Labber creation time 補量測時間。 |
| `snapshot` | 當時的 entry_name、point、description、roles 與 params。Point 可明確為 null。不要拿新容器的當前值冒充歷史快照。 |
| `provenance` | `SoftwareProvenance`，缺少證據的欄位只填型別允許的 null，不假造 commit、主機或 SoC。 |

Evidence 必須與來源 hash、檔內 tag 及 cfg 相符。巢狀 JSON 中的 true／1、false／0 不視為相同證據。矛盾不會被補充文件覆蓋。已知 pair 的 cfg 缺必要欄位、版本不是完整 `major.minor` 或 major 不符時，工具在分配 run_id 與規劃 native 前列 pending，保留來源。同 major 的合法 minor 可以轉換。驗證使用隔離副本，不把可推導的欄位回填歷史 cfg。尚未 planned 的 pending evidence 可補正版本後明確 resume，不需手改 manifest。缺 cfg 或 cfg_type 的文件仍保留完整 raw，但該 run pending。缺少必填時間或 snapshot 的 evidence 文件會直接拒絕；缺證據時可先不提供該筆 evidence，保留來源待查。

FreqGainCfg 的來源仍使用歷史 tag `twotone/ge/ro_optimize/freq`。離線 declaration 將新 native tag 寫為 `twotone/ge/ro_optimize/freq_gain`，typed reader 用新 spec 讀取。FreqCfg 的 tag 不變，原 Labber 與完整 evidence 也不改。正常 loader 沒有舊 tag alias。

轉換後 native 的 snapshot.entry_id 是新條目的 UUID，不是舊系統曾記錄的 UUID。Report 明示這項賦值。新 run_id 的時間是遷移賦值時間，也不是歷史量測時間。Migration native 的 labber_path 固定為 null，已成功搬移的 Labber 路徑從 report 的 `moved_labber_files` 查。

完成本節時，每筆證據都有可追溯來源，且 loader 接受整份文件。無法證明的欄位留待處理，不為了消除 pending 補猜測值。

## 正式執行與分段

移除 `--dry-run` 後執行同一命令，保存 stdout report 與 stderr log。預設 report 位於 `results/<name>/records/migration-report.json`。`--report` 可選已授權的獨立檔案，但首次執行拒絕既有檔，resume 也不能改換位置。

`--part parameters` 與 `--part data` 可分開執行，任一部分都會先建立新條目。第二次明確加 `--resume`。兩種順序都有效。

- Parameters 建立 setup 與自足 point，不使用讀時疊層。`params.json` 原樣保存，不推導 Q1.params。
- Data 先發布 native，再用正常 reader 與具體 typed loader 驗證。成功後才複製 Labber、核對 hash、刪除該來源檔。
- Module cfg 在本工具本次交付中只有 header-only seed。Report 保留舊模組到 library 路徑的對照，標明由後續 owner 轉換。不要把 string reference 或 pending mapping 當成已發布的 cfg。

Report 的 part 是已執行部分的聯集。`all` 不代表 pending 已解決。完成 parameters 後，可以經 ResultEntry 公開 edit 修改 setup／point；data resume 驗證當前文件，不還原舊參數。逐檔發布的 params.json、副本、Labber、native 則受 manifest hash 保護，修改它們會報衝突。

完成本節時，每個已確定的 run 能由正常 `load_run_data` 與其具體 experiment 的 `load_run` 讀回，report 的成功與待處理清單一致。

## 中斷後 resume

保留 `results/<name>/records/migration-state.json`。它記錄 entry_id、run_id、來源 baseline、逐檔 publication、完整 evidence 與 report ownership。Resume 使用相同 roots、source-chip、source-qubit、name、kind mapping revision 與 report 位置。Part 可以不同。

1. 核對失敗 log、manifest 與仍在的來源。不要手動把未完成動作標為完成。
2. 用原命令加 `--resume`。只有 manifest 擁有的 temp 可由工具清理或重寫。Final 已存在時必須通過持久 hash 核對。
3. Native writer 或 validation 失敗時，Labber 來源仍在。Resume 先恢復 native，再搬移來源。已 validated 的 native 也必須再核對 hash。
4. Labber 已複製但來源未刪時，工具核對兩端再刪來源。來源已刪但 manifest 尚未更新時，工具依 published 記錄與目的 hash 確認完成。Source_removed 只查目的，不重讀或 resolve 舊位置。舊位置被新普通檔或 symlink 重用時，resume 不讀取或刪除它。
5. Report 更新中斷時，工具從 manifest 重建累積 report。可新增尚未 planned source 的 evidence；已 planned 的證據不得替換，也不能改寫已 source_removed 的 run。

骨架已建立、初始 manifest 尚未發布時，來源未動，但工具不能自動接管殘留條目。停下來記錄目錄與 hash，取得明確清理授權後再重試。碰到目的 hash、身分或 mapping revision 衝突也先停下，不刪 manifest、不改版本來強行續跑。

Exit 0 表示確定的工作完成，pending 可以仍在。Exit 1 是執行、schema 或 I/O 失敗，查 stderr 與 manifest 的 failures。Exit 2 是輸入、目的地或身分衝突。保留失敗 log，不能只保存最後一次成功 report。

## 驗證來源保全

以執行前基線核對整個舊 result 樹的逐檔 hash。它應完全不變。每個 `moved_labber_files` 的目的 hash 應等於來源基線；只刪成功確認的來源檔，不清空日期目錄。

`arb_waveforms/` 原樣保留在 `Database/<name>/arb_waveforms/`。未能證明新格式關係的 samples.csv、image 與 autofluxdep_runs 副本位於 `Database/<name>/migration-preserved/result/`。它們不是新的 SampleTable、workflow run 或 SaveLayout 輸出。原 result 仍保留，其他未辨識檔案由 pending 定位。

交付時列出新條目 UUID、已遷移與待處理數、report／manifest／log 位置、來源 hash 核對結果，以及仍待使用者確認的對照。只有使用者裁決後才安排真實資料遷移，不由本說明書擴大授權。

## 修改 kind 時的遷移步驟

Kind 定義屬於具體 component owner。Framework 沒有通用 kind migration registry。修改定義前，列出受影響的 setup 與所有 point，確認新欄位、型別、單位和 default 的意義。Setup 的修改不會同步更新既有 point。

1. 選擇並取得本次修改策略的授權。若要在 model 定義加入舊欄位相容邏輯，使用 before validator，並明示接受哪些舊輸入。這是 component model 的選擇，不是 framework fallback。
2. 若只需一次性更新資料，使用明確授權的 ruamel script，逐份讀取、修改與驗證 YAML。先在複本比對差異，保留未修改節點的註解與順序。不要只改 kind 字串而留下不相容欄位。
3. 用顯式註冊的新 model，經 ResultEntry.open 驗證 setup，再逐個 use_point 驗證完整 point。開啟、refresh 或空 edit 不會寫回 normalized model；不要把讀取成功當成磁碟已轉換。
4. 寫入授權涵蓋的文件，保留 entry_id、point label、created_at 與來源引用。多個 point 是逐檔更新，不提供整批 all-or-nothing。保留基線、差異與失敗紀錄，以便回復尚未完成的文件。
5. 重讀全部受影響的文件，核對新型別、值、單位與 provenance。若更改 legacy conversion mapping 的 seed、rules、roles、schema 或 native tag 轉換，同時更新 mapping_version。舊 manifest 的 resume 要求完全相等；不要拿新 revision 接管舊計畫。

完成本節時，每個受影響的 setup／point 都由新 model 讀回，既有身分和來源仍可解析，未轉換文件有明確清單。此步驟不授權改寫歷史 native 或 Labber，這些檔案仍保存當時的 cfg 與 snapshot。
