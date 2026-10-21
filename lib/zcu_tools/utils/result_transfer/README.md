# Result transfer

**Last updated:** 2026-10-21

**讀者：** framework-dev、agent。維護結果傳輸的 CLI、rclone 安裝與桌面設定流程。

此模組以固定版 rclone 傳輸 entry 資料夾，將資料檔案視為原樣複製的內容。

- `cli`：setup／status／upload／download／prepare 的命令編排與結束碼。
- `rclone`：已驗證的工具安裝與本機 GUI／API 程序生命週期。
- `setup`：沿用官方 GUI 授權、瀏覽及確認雲端父資料夾。
- `settings`：位置保存與 entry 名稱驗證，不保存 OAuth token。`transfer.json` 由入口指定的 checkout 目錄保存（各 checkout 各自的雲端父資料夾）；`rclone.conf` 帳號在使用者設定目錄，所有 checkout 共用。

正式本地布局為 `result/<name>/`。使用者入口 `scripts/transfer_results.py` 呼叫
`cli.main(default_local_root, default_settings_dir)`，由入口提供 `result/` 與 `.result-transfer/`。
操作指南在子命令前明填 `--local-root <checkout>/result`，不依賴入口預設值。
套件本身不推導 repo 路徑，也不在 import 時啟動服務。
三平台命令與操作流程見 [結果傳輸教學](../../../../docs/result-transfer.md)。

## 工具準備與維護

在 repo root 執行下列命令，只安裝與驗證工具：

```bash
uv run --directory . --no-sync -- python scripts/transfer_results.py prepare
```

固定使用 rclone v1.75.2，內含 Web GUI 1.1.11。工具支援 Windows、macOS、Linux 的
x86-64 與 ARM64。安裝使用使用者快取目錄與明確的 executable 路徑，沿用專案的
uv/Python 環境即可，無須管理員權限、系統 PATH 的 rclone、Node 或 Google Python SDK。

`rclone.py` 從官方 GitHub release 下載 archive，核對 repo 固定的六平台 SHA-256。
每次啟動都會核對快取 archive 與 executable。Executable 損壞時，使用已驗證的 archive
修復；無法通過校驗的工具會被拒絕執行。
首次下載需能連線 GitHub release 資產目的地，OAuth 與傳輸另需連線 Google 服務。

## Setup 服務邊界

Setup 的官方 GUI、原生 API 與設定 bridge 只綁本機 `127.0.0.1` 動態 port。
程式確認 API 認證及 HTML readiness 後才開啟瀏覽器。
Bridge 觀察官方 GUI 成功的資料夾讀取，並轉送帳號設定請求。
它依請求順序排除過期的資料夾載入，保存前再確認選定路徑可讀。
資料寫入請求由 setup 拒絕，讓授權與位置選取維持唯讀瀏覽。

`--no-browser setup` 會印出私有 session 連結，其中含本次登入資訊。
連結僅供本機使用，應與聊天內容及日誌隔離。

## 升版與平台驗證

升版時一併更新 [rclone.py](rclone.py) 的版本與六平台 checksum。
重新驗證官方 GUI 的連線、資料夾列舉 API 與 CLI 行為後，才採用新固定版本。
執行時沿用 repo 固定版本。

現有驗證範圍為 Linux x86-64 的官方 GUI 選位置、保存、關閉，以及純 CLI 雙向 copy、
dry-run、中文與空白路徑、更新與排除規則、取消保存、錯誤碼和來源預檢。
Windows、macOS、ARM64 實機，以及真實 Google OAuth 同意畫面與 Drive 傳輸仍需驗證。
平台支援清單描述工具資產範圍，實際驗收範圍以上述紀錄為準。
