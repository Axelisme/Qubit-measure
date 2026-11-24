# Data sync

**Last updated:** 2026-11-25

**讀者：** framework-dev、agent。查閱同步 CLI、rclone 安裝與桌面設定的責任分工。

`scripts/sync_data.py` 把程式所在的 checkout 交給 `cli.main(argv, checkout=...)`。同步套件不從安裝位置或 cwd 推導 checkout，也不匯入 scripts。傳輸不解讀量測檔案，依 [ADR-0072](../../../../docs/adr/0072-module-boundaries.md#分層) 留在 `utils`。每一個來源 prefix 是獨立的 project root，操作與布局見[資料同步指南](../../../../docs/data-sync.md)。

- `cli` 驗證命令、確認設定並編排 copy。上傳來源只讀 checkout 的 `result/` 與 `Database/`。Pull 比較實體目的地與 checkout 的實體資料根。
- `discovery` 從一次遞迴的雲端目錄列舉找出來源。來源直接含 `result/` 或 `Database/`；這兩個目錄下的內容不參與探索。
- `settings` 保存各 checkout 的 `.data-sync/sync.json`，內含雲端根、來源 prefix 與本地彙整根。帳號憑證不在 checkout 設定裡。
- `rclone` 驗證固定版工具，啟動及清理本機 GUI 和 CLI 程序。使用者層的 `rclone.conf` 與工具快取沿用既有的 `qubit-measure/result-transfer` 路徑。
- `setup` 沿用官方 GUI 授權與唯讀瀏覽。CLI 收到確認過的雲端根後才保存 checkout 設定。

## 固定版工具

使用 rclone v1.75.2，內含 Web GUI 1.1.11。下載資產支援 Windows、macOS、Linux 的 x86-64 與 ARM64。不依賴系統 PATH 裡的 rclone、Node 或 Google Python SDK。

[rclone.py](rclone.py) 從官方 release 下載 archive，核對六平台的 SHA-256。每次啟動都核對快取 archive 與 executable；executable 損壞時用驗證過的 archive 修復。首次安裝需能連 GitHub release，授權與傳輸另外需要雲端服務。升版時更新版本、六平台 checksum，並重新驗證官方 GUI 選位置、列舉和 CLI copy。

## 設定服務

官方 GUI、API 與設定 bridge 只綁 `127.0.0.1` 的動態 port。程式確認認證與 HTML readiness 才開瀏覽器。Bridge 只轉送帳號設定和瀏覽所需的請求；轉交選定路徑前會再讀一次，避免保存過期或載入失敗的候選。印出的連結含本機 session 憑證，不要貼到聊天或日誌。
