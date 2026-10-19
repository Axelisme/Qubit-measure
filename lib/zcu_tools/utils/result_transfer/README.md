# Result transfer

**Last updated:** 2026-10-20，transfer.json 改為每個 checkout 一份

以固定版 rclone 傳輸 entry 資料夾，不解讀量測檔案格式或實驗語意。

- `cli`：setup／status／upload／download／prepare 的命令編排與結束碼。
- `rclone`：已驗證的工具安裝與本機 GUI／API 程序生命週期。
- `setup`：沿用官方 GUI 授權、瀏覽及確認雲端父資料夾。
- `settings`：位置保存與 entry 名稱驗證，不保存 OAuth token。`transfer.json` 由入口指定的 checkout 目錄保存（各 checkout 各自的雲端父資料夾）；`rclone.conf` 帳號在使用者設定目錄，所有 checkout 共用。

正式本地布局為 `result/<name>/`。使用者入口 `scripts/transfer_results.py` 呼叫
`cli.main(default_local_root, default_settings_dir)`，由入口提供 `result/` 與 `.result-transfer/`。
操作指南在子命令前明填 `--local-root <checkout>/result`，不依賴入口預設值。
套件本身不推導 repo 路徑，也不在 import 時啟動服務。
三平台命令與操作流程見 [結果傳輸教學](../../../../docs/result-transfer.md)。
