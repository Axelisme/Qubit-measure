# Data sync

**Last updated:** 2026-11-26

**讀者：** framework-dev、agent。查閱同步 CLI、rclone 安裝與桌面設定的責任分工。

`scripts/sync_data.py` 把程式所在的 checkout 交給 `cli.main(argv, checkout=...)`。同步套件不從安裝位置或 cwd 推導 checkout，也不匯入 scripts。傳輸不解讀量測檔案，依 [ADR-0072](../../../../docs/adr/0072-module-boundaries.md#分層) 留在 `utils`。每個 prefix 都是獨立的 project root，操作與布局見[資料同步指南](../../../../docs/data-sync.md)。

- `cli` 編排設定與傳輸。`push`、`pull`、`sync` 對同一 prefix 使用相同的兩端實體路徑。操作綁定的 prefix 前驗證兩個 checkout 連結。`pull --all` 跳過綁定值。
- `binding` 先檢查兩個連結位置與實體目標。Windows 建 junction，Linux 建目錄符號連結；正確的現有連結不重建。
- `settings` 保存 checkout 的 `.data-sync/sync.json`，內含雲端根、預設 prefix 與本機 Labdata 根。不保存帳號憑證。舊第 1 版設定需重跑 setup。
- `setup` 沿用官方 GUI 授權與唯讀瀏覽。操作者確認兩端資料根、`binding` 驗證兩個連結後，CLI 才保存 checkout 設定。
- `discovery` 從一次遞迴的雲端目錄列舉找出來源。來源直接含 `result/` 或 `Database/`；這兩個目錄下的內容不參與探索。
- `inventory` 批量列出資料檔案與同路徑的目錄，保留缺少大小或 MD5 的檔案，另報重複路徑。它將 RFC3339 時間換成 UTC，保留奈秒精度。只有確認 prefix 目錄不存在，`sync` 才把該端當作空目錄。
- `baseline` 核對兩端大小與 MD5，讀寫 Labdata 下每個雲端根與 prefix 共有的基準。
- `bidirectional` 用完整盤點和基準計畫 `sync`，執行各方向的批次，再管理驗後的基準更新。
- `rclone` 驗證固定版工具，啟動及清理本機 GUI 和 CLI 程序。使用者層的 `rclone.conf` 與工具快取沿用既有的 `qubit-measure/result-transfer` 路徑。

## 傳輸契約

對每個 prefix，rclone 批量複製 `result/` 和 `Database/` 下的檔案，排除 `*.lock`。普通推拉使用 `copy --update`。單一 prefix 的 `--force` 改用 `copy --ignore-times`，以免 rclone 用相同的大小和修改時間略過同名檔案。兩種模式都不刪目的端獨有的檔案。`--dry-run` 不傳輸，也不讀寫基準。

正式推拉先讀取所有選定 prefix 的 `<labdata>/.data-sync/state/<pair-id>.json`。檔名根據雲端根和 prefix 產生，同一 Labdata 的 checkout 共用紀錄。損壞或未知版本的紀錄會阻止複製。每個 prefix 複製成功後，才核對兩端同一路徑的大小和 MD5。無共同 MD5、內容不一致或同一路徑有多個物件時，不推進該檔的基準。基準保存兩端各自的 UTC 修改時間和位元組數，以原子替換更新。傳輸不會上傳這份紀錄。

傳輸測試透過 `cli.main` 使用暫存資料根。執行 `pytest tests/utils/data_sync` 前，將 `DATA_SYNC_TEST_RCLONE` 指向 `rclone.prepare_rclone` 在隔離快取中驗證的執行檔，即可跑實際的本機後端測試。測試不連雲端，也不讀使用者的 rclone 帳號。

## 雙向計畫

`sync` 和 `sync --dry-run` 都先讀基準與兩端盤點。初次比較沒有基準的同路徑檔案時，按 RFC3339 實際時間選較新版本。後續以各端自己的時間與大小判斷修改。需要比較內容時，沒有大小與 MD5 一致的證據就報衝突。單邊刪除、重複路徑或已知目的路徑占用也阻止整個 prefix 傳輸。

計畫先列出每個檔案的動作與原因。開始任何複製前，驗證兩個方向的路徑並備妥清單。每個非空方向各執行一次 rclone `copy`，用 `--files-from0` 傳字面路徑，並以 `--ignore-times` 執行已選擇的覆寫，不再用時間選一次版本。預覽不複製、不保存基準。

複製成功後重新列舉兩端，核對計畫傳輸的路徑。個別檔案未通過驗證時，保留該檔原基準，保存其他已確認的項目並回傳非零。兩端都已刪除的路徑只在成功同步後移除基準。複製、整批核對或保存失敗不更新基準；命令不回滾已複製的檔案。操作員的處理方式見[兩台電腦輪流接手](../../../../docs/data-sync.md#兩台電腦輪流接手)。

## 固定版工具

使用 rclone v1.75.2，內含 Web GUI 1.1.11。下載資產支援 Windows、macOS、Linux 的 x86-64 與 ARM64。不依賴系統 PATH 裡的 rclone、Node 或 Google Python SDK。

[rclone.py](rclone.py) 從官方 release 下載 archive，核對六平台的 SHA-256。每次啟動都核對快取 archive 與 executable；executable 損壞時用驗證過的 archive 修復。首次安裝需能連 GitHub release，授權與傳輸另外需要雲端服務。升版時更新版本、六平台 checksum，並重新驗證官方 GUI 選位置、列舉和 CLI copy。

## 設定服務

官方 GUI、API 與設定 bridge 只綁 `127.0.0.1` 的動態 port。程式確認認證與 HTML readiness 才開瀏覽器。Bridge 只轉送帳號設定和瀏覽所需的請求；轉交選定路徑前會再讀一次，避免保存過期或載入失敗的候選。印出的連結含本機 session 憑證，不要貼到聊天或日誌。
