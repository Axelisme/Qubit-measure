---
status: accepted
---

# ADR-0040 — autofluxdep run result artifact

關聯 [[0018]]（autofluxdep resolver-builder boundary）、[[0027]]（一般 experiment data persistence）、[[0038]]（executor workflow collection 不納入單一 Experiment Result）。

## 脈絡

`autofluxdep-gui` 的 run result 目前是 run-lived memory state：每個 placed node 的 `Result` dataclass 預先配置 nan-filled array，worker 在 sweep 中逐 row 填入，UI 由 main thread 讀同一份 Result 更新 plot。關窗、crash 或長掃中斷後，這些 run result 不會落盤。

後續功能需要一個穩定的 run identity 與可稽核資料來源：

- 長掃 crash 後能知道哪些 node/flux row 已完成。
- `fluxdep-gui` / `dispersive-gui` 的 handoff file 不能各自變成事實來源。
- run report 與 agent-memory 記錄需要引用同一個 run artifact。
- resume / repair-run 需要 machine-readable Patch、skip、failure 與 commit state。

## 決策

1. **Run Result Artifact 是 canonical source of truth。** `autofluxdep` 的 canonical persisted output 是一個 run-scoped artifact；fluxdep-compatible spectrum、dispersive input 與 markdown report 都是由它派生的 exchange export / report sidecar，不是 canonical source。

2. **artifact 以 run-scoped metadata/data root 表達。** 每次 run 有一個 sortable slug，對應兩個同 slug root：metadata root 包含 manifest、append-only journal 與 report sidecar；data root 包含每個 placed node 的 result file 與 heavy exports。manifest 擁有 run identity、workflow hash、project result/database roots、metadata/data roots、node file list、terminal status 與 export/report path；journal 擁有 row-level audit events。

3. **result_dir 只放輕量 metadata，heavy HDF5 放 qubit database root。** metadata root 位於 project `result_dir/autofluxdep_runs/<run_slug>/`，不放 active context 的 `exps/` 目錄；data root 位於 `Database/<chip>/<qub>/autofluxdep_runs/<run_slug>/`。若 project `database_path` 是 shared startup resolved dated raw folder `Database/<chip>/<qub>/YYYY/MM/Data_MMDD`，RunStore 會剝掉尾端 Labber 日期三段，再建立 autofluxdep run root。autofluxdep 是 workflow-level sweep，不是 measure-gui tab experiment；其 metadata/report 要被 result tooling 找到，應掛在 qubit scope 的 project result directory；Labber-readable heavy data 則留在 Database tree。

4. **run directory name 以 timestamp 為主，語意 slug 為輔。** 目錄名使用 sortable wall-clock timestamp 加可讀 slug，例如 `20260704-153012_flux-sweep`；真正 identity 是 manifest 內的 `run_id`，包含 timestamp 與短 random suffix，避免同秒 collision。slug 只供人閱讀，不作 stable identity。

5. **manifest / journal 是正式版本化格式。** manifest 包含 `format_version: 1`；journal 每個 event 包含 `event_version: 1`。manifest 的 workflow snapshot 使用 node `to_persisted_raw()`；UI 可用 nested generation groups，但 snapshot 內的 `generation` 仍以 flat logical-key raw map 表達，避免 presentation grouping 改變 artifact contract。result browser、resume、agent report 與 migration tooling 都只能依賴版本化 shape；未知 major version 必須 fast-fail，不以 best-effort parse 假裝可讀。

6. **每個 placed node 一個 streaming HDF5 result file。** canonical node data 不拆成每個 flux 一個檔案，且 node HDF5 寫在 data root 的 `nodes/` 下，不寫在 result_dir metadata root。node file 在 run start 預先建立 full-shape nan-filled datasets；每個 node row 完成後寫入同一個 node file 並 flush。這讓 `qubit_freq` 保持自然的 flux x detune map，也避免大量小檔讓 manifest/journal 變成主要資料庫。

7. **node HDF5 盡量保持 Labber-readable。** 同一 node file 內的各 Result 欄位以多個 Labber log group / dataset role 表達，而不是只把 primary signal 做成 Labber dataset、把其他欄位藏進私有 HDF5 dataset。journal/manifest 才放 Labber 不自然表達的 workflow metadata。

8. **streaming writer 是獨立 public primitive，不改 `save_labber_data` 語意。** `save_labber_data` / `save_grouped_labber_data` 維持 one-shot complete save 與 exact-path ownership。autofluxdep 使用新的 streaming / partial writer primitive：預建 dataset、row write、flush、finalize。此 primitive 可以住在 datasaver low-level module，但 API 與 one-shot save 分開。

9. **Result dataclass 是 typed read/write boundary。** node `produce()` 只填現有 `QubitFreqResult` / `Sweep1DResult` / `Sweep2DResult`。run lifecycle / store layer 從 typed Result 寫 row；reader 從 node HDF5 回復 typed Result。node implementation 不直接操作 h5py 或 persistence writer。

10. **canonical artifact 保存 Result 的全部持久欄位。** 不是只保存 primary signal + scalar summary。第一版 role set：

   - `QubitFreqResult`: `signal`, `fit_curve`, `fit_freq`, `predict_freq`, `snr`
   - `Sweep1DResult`: `signal`, `fit_curve`, `fit_value`, `snr`
   - `Sweep2DResult`: `signal`, `best_freq`, `best_gain`

   role name 是 reader contract；Labber channel name/unit 可由 result kind / node type metadata 提供更具物理語意的名稱。例如 `Sweep1DResult.fit_value` 的 internal role 仍是 `fit_value`，但 metadata 必須能表達它是 T1、pi length、T2 等。

11. **commit 粒度是 node row，另有 flux-level commit marker。** 每個 node 在一個 flux point 完成並正常返回後，store 寫該 node row 並 append `node_row_written` event。當同一 flux point 的 provider loop 結束後，journal append flux-level commit marker。這讓 artifact 同時回答「哪些 measurement attempt 已完成」與「哪些 flux point 整點完成」。

12. **in-progress acquire row 不進 committed data。** round-hook 可以繼續更新 UI live Result row，但 canonical row commit 只在 node `produce()` 正常返回後發生。crash mid-node 時，該 node row 保持未 committed；resume / repair-run 重跑該 row。

13. **fit / provide failure 仍可 committed。** 如果 acquire 完成、Result row 有 raw signal，但 fit gate 失敗導致 Patch 為空，node row 仍算 committed measurement attempt。journal 以 structured status 區分 `measurement_status` 與 `provide_status`；Patch `{}` 是可稽核狀態，不等於未測。

14. **Patch 與 provided modules 進 journal，不進 Labber data channel。** Patch 是 downstream dependency state，不是 node self result channel。`node_row_written` event 記錄 `flux_idx`、node、Patch、provided modules、result file reference 與 provide status；result file reference 是相對 data root 的 path。

15. **resolver skip 是 journal event，不是 HDF5 row。** 被 dependency resolver skip 的 node 不寫 HDF5 row，但 append `node_skipped` event。skip reason 是 structured data，至少包含 missing info keys / missing modules；保留 payload 擴充點。這需要把 resolver 的 implicit `None` skip 語意提升為 typed resolution result 或等價 side-channel。

16. **node exception 同時產生 node-level 與 run-level failure event。** `node_failed` 記錄 flux idx、node、exception type/message 與是否已有 committed row；`run_failed` 記錄 terminal status 與 failed node/flux 摘要。exception traceback 留在 log，不放 manifest。若 `produce()` exception，store 不自動嘗試保存 partially-filled Result row。

17. **partial / stopped / failed run 都是有效 artifact。** terminal finalize 只更新 manifest terminal status、關閉 writers、產生可完成的 exports/report。已 committed row 保留；未 committed row 仍是 nan / absent commit state。

18. **persistence failure policy 採 fast-fail / fail-run / preserve-flushed-data。** run start 前建立 run directory、manifest、journal 或 node writers 任一步失敗，run 不開始；run 中 HDF5 row write、flush 或 journal append 失敗，視為 infrastructure failure 並 fail 整個 run；terminal report / export 失敗記入 manifest 並 surfaced，但不把已 committed measurement row 改成 measurement failure。manifest update 失敗必須 surfaced，但不得刪除或覆蓋已 flush 的 node HDF5 / journal。

19. **markdown report 在 terminal finalize 時產生，不做 live report rewrite。** live safety 由 node HDF5 row flush、journal append 與 manifest update 提供；report 是 terminal summary。第一版 report 只包含 metadata、terminal status、node row/skip/failure summary、metadata/data root paths、fluxdep export path 與 fit/provide summary，不輸出 PNG / MP4，也不要求 headless 重畫 figure。

20. **stop / cancel 必須 finalize artifact。** cooperative stop 不是 failure；terminal path 必須關閉 streaming writers、flush manifest/journal、寫 `status=stopped`，保留已 committed row。若 terminal finalize 本身失敗，錯誤要 surfaced，但不得破壞已 flush 的 node files / journal events。

21. **A1 的第一個可用 implementation slice 包含 exports/report，但不包含 resume/browser。** A1 必須足以直接支援長掃使用：run 自動落盤、per-node HDF5 可用 Labber 開啟、`qubit_freq` 產生 fluxdep-compatible spectrum export、terminal 產生 markdown report 與 result references。Resume、repair-run、result browser UI、PNG / MP4 export 是後續功能；它們依賴同一 artifact，但不拖住 A1 可用版。

22. **Labber Browser sidecars 是 exchange export，不是 canonical artifact。** canonical node data 仍是 `nodes/*.hdf5` grouped artifact；固定軸 sidecar 在 run start 預建 single-log streaming HDF5，node row commit 後同步寫入並 flush，未 committed rows 保持 NaN。`qubit_freq` sidecar 例外維持 terminal finalize 產生，因為它的 Labber Browser / fluxdep-compatible absolute Frequency 軸必須從 journal-committed rows 的 `predict_freq` 推導。sidecar root 位於 run-scoped data root 內的 `labber/YYYY/MM/Data_MMDD/`，日期由 run slug 的 `YYYYMMDD-...` 前綴決定；例如 `Database/<chip>/<qub>/autofluxdep_runs/20260705-223908_flux-sweep-.../labber/2026/07/Data_0705/`。每個 sidecar 都是 `save_labber_data()` 或其 streaming 對偶可讀的 single-log Labber HDF5。manifest 的 `exports["fluxdep_spectrum"]` 保持既有 fluxdep handoff 語意；`exports["labber_browser_root"]` 保存 Labber Browser dated folder 相對 data root 的路徑，`exports["labber_browser_sidecars"]` 保存每個 sidecar 的 node、node type、role 與相對 data root path。第一版 sidecar contract：`qubit_freq` 輸出 `<index>-qubit_freq_qubit_freq.hdf5`，內容等同 fluxdep-compatible qubit frequency spectrum；`lenrabi` 輸出 `<index>-lenrabi_signal.hdf5`；`ro_optimize` 不輸出；`t1` 輸出 `<index>-t1_signal.hdf5` 與 `<index>-t1_t1.hdf5`；`t2ramsey` 輸出 `<index>-t2ramsey_signal.hdf5` 與 `<index>-t2ramsey_t2r.hdf5`；`t2echo` 輸出 `<index>-t2echo_signal.hdf5` 與 `<index>-t2echo_t2e.hdf5`；`mist` 輸出 `<index>-mist_signal.hdf5`。

## 後果

- `autofluxdep` persistence 不是 memento。Memento 只保存 workflow definition / UI preference；Run Result Artifact 保存 run output 與 audit trail。
- `autofluxdep` run lifecycle 需要一個 store layer，負責 create / write node row / commit flux / finalize / load。它是 infrastructure boundary，不屬於 Node produce domain logic。
- `result_dir` 保持輕量，適合 result browser、report 與 agent reference；Labber HDF5 path 必須透過 manifest 的 `paths.data_root` 加 node/export 相對 path 解析。
- `project_snapshot()` / resolver skip path 需要提供 machine-readable skip reason，否則 artifact 無法完整稽核 skipped node。這是 resolver boundary 的顯式化，不改變 orchestrator 的「requirement resolver」角色。
- `save_labber_data` 不承擔 long-lived partial write；新增 streaming primitive 時需保持與 ADR-0027 one-shot Experiment Data File 語意分離。
- `fluxdep` handoff 第一版可由 `qubit_freq` node result 產生 Labber-compatible spectrum export，但 canonical evidence 是 data root 內的 node/export file 加 metadata root 內的 journal/manifest。
- Labber Browser sidecars 讓人工瀏覽使用 dated Labber folder，不改變 run-scoped data root 與 canonical evidence；固定軸 sidecars 可在 run 中直接開啟，修復或重匯出 sidecars 時仍必須以 journal committed rows 為準。

## 拒絕的替代方案

- **terminal 時一次寫完整結果**：簡單，但長掃 crash / GUI 被殺時仍丟失所有 memory-only result，不符合 run artifact 的稽核責任。
- **每個 flux x node 一個 one-shot HDF5**：可直接重用 `save_labber_data`，但會製造大量小檔，破壞 `qubit_freq` 的自然 2D map，也讓拼檔成為正常讀取流程。
- **SQLite / database 作 raw array 主存**：對 journal / manifest 有吸引力，但 raw N-D numeric array 與 Labber-readable handoff 仍需要 HDF5；把 raw data 放 database 會讓 Labber handoff 變成二級 export，增加正常人工檢視成本。
- **把 Patch / skip reason 塞進 Labber data channel**：Patch 是 dependency audit state，不是 measurement channel；塞進 Labber role 會混淆 Result 與 Patch 的 domain boundary。
- **exception 時保存 partially-filled row**：Result dataclass 在 exception path 上沒有 node-specific completion guarantee；自動保存會讓 reader/resume 必須判斷一列半成品是否可信。
