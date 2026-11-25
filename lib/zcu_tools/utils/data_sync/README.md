# Data sync

**Last updated:** 2026-11-25

**讀者：** framework-dev、agent。查閱同步 CLI、rclone 安裝與桌面設定的責任分工。

`scripts/sync_data.py` 把程式所在的 checkout 交給 `cli.main(argv, checkout=...)`。同步套件不從安裝位置或 cwd 推導 checkout，也不匯入 scripts。傳輸不解讀量測檔案，依 [ADR-0072](../../../../docs/adr/0072-module-boundaries.md#分層) 留在 `utils`。每個 prefix 都是獨立的 project root，操作與布局見[資料同步指南](../../../../docs/data-sync.md)。

- `cli` 顯示兩端資料根，編排設定與傳輸。`push` 和 `pull` 對同一 prefix 使用相同的兩端路徑。操作綁定的 prefix 前驗證兩個 checkout 連結。`pull --all` 跳過綁定值。
- `binding` 先檢查兩個連結位置與實體目標。Windows 建 junction，Linux 建目錄符號連結；正確的現有連結不重建。
- `settings` 保存 checkout 的 `.data-sync/sync.json`，內含雲端根、預設 prefix 與本機 Labdata 根。不保存帳號憑證。舊第 1 版設定需重跑 setup。
- `setup` 沿用官方 GUI 授權與唯讀瀏覽。操作者確認兩端資料根、`binding` 驗證兩個連結後，CLI 才保存 checkout 設定。
- `discovery` 從一次遞迴的雲端目錄列舉找出來源。來源直接含 `result/` 或 `Database/`；這兩個目錄下的內容不參與探索。
- `rclone` 驗證固定版工具，啟動及清理本機 GUI 和 CLI 程序。使用者層的 `rclone.conf` 與工具快取沿用既有的 `qubit-measure/result-transfer` 路徑。

## 傳輸契約

對每個 prefix，rclone 批量複製 `result/` 和 `Database/` 下的檔案，排除 `*.lock`。普通推拉使用 `copy --update`。單一 prefix 的 `--force` 改用 `copy --ignore-times`，以免 rclone 用相同的大小和修改時間略過同名檔案。兩種模式都不刪目的端獨有的檔案。`--dry-run` 不傳輸。

傳輸測試透過 `cli.main` 使用暫存資料根。執行 `pytest tests/utils/data_sync` 前，將 `DATA_SYNC_TEST_RCLONE` 指向 `rclone.prepare_rclone` 在隔離快取中驗證的執行檔，即可跑實際的本機後端測試。測試不連雲端，也不讀使用者的 rclone 帳號。

## 固定版工具

使用 rclone v1.75.2，內含 Web GUI 1.1.11。下載資產支援 Windows、macOS、Linux 的 x86-64 與 ARM64。不依賴系統 PATH 裡的 rclone、Node 或 Google Python SDK。

[rclone.py](rclone.py) 從官方 release 下載 archive，核對六平台的 SHA-256。每次啟動都核對快取 archive 與 executable；executable 損壞時用驗證過的 archive 修復。首次安裝需能連 GitHub release，授權與傳輸另外需要雲端服務。升版時更新版本、六平台 checksum，並重新驗證官方 GUI 選位置、列舉和 CLI copy。

## 設定服務

官方 GUI、API 與設定 bridge 只綁 `127.0.0.1` 的動態 port。程式確認認證與 HTML readiness 才開瀏覽器。Bridge 只轉送帳號設定和瀏覽所需的請求；轉交選定路徑前會再讀一次，避免保存過期或載入失敗的候選。印出的連結含本機 session 憑證，不要貼到聊天或日誌。
