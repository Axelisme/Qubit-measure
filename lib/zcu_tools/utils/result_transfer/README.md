# Result transfer

**Last updated:** 2026-10-19

以固定版 rclone 傳輸 entry 資料夾，不解讀量測檔案格式或實驗語意。

- `cli`：setup／status／upload／download／prepare 的命令編排與結束碼。
- `rclone`：已驗證的工具安裝與本機 GUI／API 程序生命週期。
- `setup`：沿用官方 GUI 授權、瀏覽及確認雲端父資料夾。
- `settings`：位置保存與 entry 名稱驗證，不保存 OAuth token。

使用者入口 `scripts/transfer_results.py` 呼叫 `cli.main(default_local_root)`，
明確提供該 checkout 的 `results/`。套件本身不推導 repo 路徑，也不在 import 時啟動服務。
三平台命令與操作流程見 [結果傳輸教學](../../../../docs/result-transfer.md)。
