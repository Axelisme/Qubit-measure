# Entry 與 point 文件檔名統一

**讀者：** operator、framework-dev、agent。辨識文件新舊檔名，使用離線工具改名與還原。

2026-10-29，entry 與 point 使用同一組文件檔名：

| 位置與文件 | 舊名 | 新名 |
| --- | --- | --- |
| Entry 參數範本 | `setup.yaml` | `parameters.yaml` |
| Point 參數 | `point.yaml` | `parameters.yaml` |
| 兩層 library | `module_cfg.yaml` | `library.yaml` |
| 兩層 aliases | `aliases.yaml` | `aliases.yaml`，不變 |

`resources.document_names` 擁有檔名。Entry discovery 只認根層 `parameters.yaml`。
Runtime 不讀舊名或自動改名。只有舊 setup 的 entry 開啟時，錯誤會列出舊檔與改名工具。
GUI watcher、MCP files 與 storage migration 產出使用新名。
這次不改 global 工作集、文件內容、schema_version 或容器版本。

## 離線升級

先停止其他 writer，再執行原入口 `scripts/upgrade_parameters.py --scope filenames`。
可傳 entry 目錄或精確舊檔路徑，預設 dry-run 列出改名，不寫檔。
加 `--apply` 才保存外部原 bytes 備份並改名。
內容逐 byte 保持不變，檔內註解與換行也不重寫。
同一目錄新舊並存或多個舊名指向相同目的時，選集在寫入前拒絕。
工具不比較兩份內容，也不覆寫。
Apply 的 I/O 失敗可能留下已完成的前項，沒有跨檔交易或掉電復原保證。

若也需要內容升級，先跑 filenames，再跑預設 components。
Components 只列新檔名；predictor 內容仍由既有獨立 owner 升級。
命令、備份位置與還原順序見[升級指南](../parameter-upgrades.md#先改檔名再升級內容)。

## Wire 與格式版本

Measure `entry.info`、`status` 與 reload 回傳的文件絕對路徑改為新名。
已鎖定的路徑契約改變，因此 WIRE_VERSION 從 131 升為 132，GUI_VERSION 從 164 升為 165。
MCP 使用同一 wire owner 的版本進行握手，不另訂檔名版本。
參數容器仍是 2.0，library 仍是 2.0，aliases 仍是 1.0。
檔名不描述內容 schema，lab component schema_version 維持 3。
