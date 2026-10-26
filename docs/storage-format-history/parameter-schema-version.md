# Component 文件新增 schema_version

**讀者：** operator、lab-author、framework-dev、agent。辨識整份 component 文件版本與明確升級入口。

## 改前與改後

改前只能從容器標頭辨識 YAML 版面，無法判斷 component schema。

```yaml
format: zcu.parameter-container
format_version: '2.0'
components: {}
```

改後每份 setup、point 與 entry template 另有嚴格正整數 `schema_version`。以下只展示相關標頭，其他既有 sections 不變。

```yaml
format: zcu.parameter-container
format_version: '2.0'
schema_version: 1
components: {}
```

Entry template 仍是 `format: zcu.entry-template`、`format_version: '1.0'`，同樣增加 `schema_version: 1`。Module library、alias、record 與量測資料的版本不受影響。這次不改 component 欄位或數值。

## 原因與辨識

容器版本只描述版面。Lab 的 component 結構可獨立演進，因此每份文件共用一條 component schema 線性版本。Lab 宣告目前版本與純函式步驟，lib 只提供版本檢查及通用 runner，不認識 lab 欄位。

Runtime 只接受目前版本，缺版或不同版本時列出來源、文件版本、目前版本與 upgrade 命令，不寫檔。建立寫 caller 宣告的目前版本，保存保留版本，seed／clone 保留已驗證來源的版本。新增有預設值的 optional 欄位不要求升版。

## 升級與相容性

長期入口是 `scripts/upgrade_parameters.py`，預設逐文件、逐欄位 dry-run。`--apply` 先完整備份再原子替換，失敗隔離到單一文件。Current 文件先驗證再跳過，重跑不重寫。

未標版文件不能在 runtime 猜成 v1。只有明確 upgrade，且 lab 的完整舊格式分類確認後，runner 才視為 v1。目前 lab 版本為 1、零個結構步驟，完整 lab 分類尚未接上。現階段命令明確拒絕未標版文件，不改標頭、不把無法分類冒充已證實腐敗。第一個結構步驟交付時需補齊分類，並在結構切換同 commit 宣告對應版本。

操作、輸出、備份還原與 callback 契約見 [參數文件升級](../parameter-upgrades.md)。跨模組邊界見 [ADR-0063](../adr/0063-persistence-ownership.md#component-文件版本與明確升級)。長期鏈不依賴可刪除的 migrations，舊 repo 匯入仍由 caller 直接產生目前版本。
