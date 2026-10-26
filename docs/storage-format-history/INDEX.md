**讀者：** framework-dev、agent。按日期查閱存檔形狀、變更理由、辨識方式與遷移邊界。

# 存檔格式歷程

- 2026-10-27，[Labber 目標身分與 caller 命名](2026-10-27-labber-identity.md)。Labber comment 加入 point／roles snapshot，Labber 與 native 路徑改用 caller 的 save_stem，既有檔案不改名。
- 2026-10-26，[移除 acquisition facts](2026-10-26-remove-acquisition-facts.md)。Run metadata、workflow operation／RPC 與 summary 不再保存 acquisition observations，舊 acquisitions key 只讀取忽略。
- 2026-10-11，[舊 result／Database 與 sample CSV 遷入 native entry](2026-10-11-legacy-storage-and-sample-csv.md)。整理 chip／qubit 來源到 entry／native 的對照，以及 CSV 到 record JSONL 的 v2 欄位與離線轉換工具。
