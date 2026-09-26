# Scripts

**Last updated:** 2026-09-27

這裡是使用者入口；repo 品質檢查在 [tools/README.md](../tools/README.md)。
工作站腳本請使用該 worktree 的 interpreter，例如
`uv run --directory <worktree> --no-sync -- python scripts/<name>.py`。
板端入口使用獨立的 PYNQ Python 3.8 環境，不假設工作站依賴可用。
下表只列主要用途；實際參數請用各腳本的說明或閱讀程式確認。

## GUI 與板端 server

- `run_measure_gui.py`：工作站 Python 3.13 GUI profile 的主要量測 GUI。讀取可恢復的 session 與實驗設定；正常關閉會寫回 GUI state，預設在 repo `logs/gui/measure/` 建立 session log。預設啟動 loopback remote-control socket；允許外部連線的選項會改變網路曝露範圍。`--clean` 不恢復先前 session，但不阻止關閉時寫回。量測 GUI 可透過 QICK 連接硬體；啟動器本身只組合 GUI 與 runtime。入口見 `run_measure_gui.py`，GUI 責任見 [gui README](../lib/zcu_tools/gui/README.md)。
- `run_fluxdep_gui.py`、`run_dispersive_gui.py`：工作站 Python 3.13 GUI profile，分別啟動 fluxonium flux-dependence 與 dispersive-shift 分析 GUI。可輸入 chip、qubit、結果目錄與資料庫路徑；傳入的 project 與 repo root 交給 GUI runtime。預設各開 remote-control TCP socket（預設 port 8766、8767，未指定時占用會回退到臨時 port）；GUI 的分析或存檔由各自 app 負責，啟動器不直接改寫原始資料。見 [fluxdep](../lib/zcu_tools/gui/app/fluxdep/README.md) 與 [dispersive](../lib/zcu_tools/gui/app/dispersive/README.md)。
- `run_autofluxdep_gui.py`：工作站 Python 3.13 GUI profile，啟動 autofluxdep workflow GUI；其 runtime 預設寫 `logs/gui/autofluxdep/` 的 session log 並開 read-only remote-control socket（預設 port 8768）。GUI workflow 的資料輸出及執行條件見 [autofluxdep](../lib/zcu_tools/gui/app/autofluxdep/README.md)，不要把入口的啟動誤認為無資料副作用。四個 GUI 的 `.bat` 檔會先切換至腳本目錄，使用 `uv run --extra gui` 啟動相應 Python 檔；repo root 仍由入口檔的 parent 定位。
- `start_server.py`、`start_server.ipynb`：ZCU 板端 PYNQ Python 3.8 啟動 QICK Pyro nameserver 與 proxy server。Python 腳本接收 daemon port、nameserver port、SoC 版本與網卡；它由自身位置將 repo root 加入 `sys.path`，`lib` 仍需由板端環境提供。Notebook 在 `scripts/` 執行，使用相對路徑 `../`、`../lib/` 與 `../qick/qick_lib/`。兩者均需能 import repo-root [`bitfiles`](../bitfiles/README.md) 與 [`zcu_tools.qick_remote`](../lib/zcu_tools/qick_remote/README.md)，會開網路服務並連接板端 SoC；板端需自行安裝或放置 QICK。板端部署、啟動與資產版本相容性**待核實**，尚未在硬體上測試。不要用工作站 Python 執行。

## 資料作業

- `export_autofluxdep_sample_table.py`：工作站 Python 3.13 GUI profile，從 autofluxdep run directory、`manifest.json` 或配對的資料 root 匯出 SampleTable CSV。預設目的地在 data root 的 `exports/sample/samples.csv`，已存在時 append；`--overwrite` 改為重建。缺少有效 flux unit 的 artifact 在寫入前失敗。見 [autofluxdep](../lib/zcu_tools/gui/app/autofluxdep/README.md)。
- `download_result.py`：工作站有 Google Drive client 依賴與 credentials 的環境。輸入 qubit folder 名稱與 `GOOGLE_DRIVE_PARENT_FOLDER_ID`，從 Drive 遞迴下載到 repo `result/<qubit>/`；本地不存在會建目錄，遠端較新會以二進位寫入覆蓋本地檔案並設定 mtime。OAuth token 讀取或更新目前工作目錄的 `token.json`，可能啟動認證流程；會存取網路及使用者資料。
- `upload_result.py`：同上，但需要 Drive 寫入權限；從 repo `result/<qubit>/` 建立或更新 Drive 對應檔案與目錄。明確選用 `--prune-remote` 時，還會刪除遠端缺少本地對應的檔案。OAuth token 存於目前工作目錄的 `token.json`；會存取網路與修改使用者的 Drive 資料。
- `data_server.py`：**用途待確認**。Flask HTTP server 預設監聽 `0.0.0.0:4999`，以 repo `Database/` 為預設 root。`/upload` 接收 h5/hdf5 檔案，經 `reserve_labber_filepath` 決定寫入檔名；`/download` 依所給 path 送出檔案。指定 `--root_dir` 時會將 root 存為字串，上傳與下載用 `/` 組合路徑時會失敗，目前不可用。檔名中的 `..` 未被排除，所給下載相對路徑也未受 root 範圍限制；讀寫可能超出 `Database/`。使用含 Flask 與資料檔依賴的工作站環境。此服務曝露網路與讀寫使用者資料；尚無足夠文件確認部署對象或安全邊界。
- `sync.py`：**用途待確認**。觀察到 `md2nb` 在 `notebook_md/` 到 `notebook/` 間同步，`nb2md` 反向同步；使用 Jupytext、Git 與工作站環境。會建立或覆寫對應檔案，`md2nb` 不同步時可互動詢問，`nb2md` 直接覆寫。未確認此流程是否仍為正式的 notebook 發布程序。

## 模擬資料庫產生

- `generate_fluxonium_sample.py`：工作站有 NumPy、h5py 與 fluxonium 數值依賴的環境；依參數邊界或 preset、樣本數、flux 網格與能階數計算 sample HDF5，供數值搜尋使用。`--output` 指定輸出檔；已存在時預設拒絕，`--overwrite` 才可覆寫；`--dry-run` 使用 `_dryrun.h5` 輸出。可能耗用大量 CPU 與磁碟，不連接量測硬體。資料庫使用方式見 [fluxdep notebook README](../lib/zcu_tools/notebook/analysis/fluxdep/README.md)。
