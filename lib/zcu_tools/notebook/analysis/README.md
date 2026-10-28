# Notebook analysis

**Last updated:** 2026-10-28

**讀者：** lab-author、framework-dev。查找 Notebook 分析入口，區分互動工作流程與共用計算的 owner。

## 分析入口

| 路徑 | 操作與結果 |
| --- | --- |
| [fluxdep](fluxdep/README.md) | 整理光譜、互動選點，組合資料庫搜尋與能譜微調。 |
| [t1_curve](t1_curve/README.md)、[t2_curve](t2_curve/README.md) | 執行各自的曲線擬合與分階段分析，模型與保存條件由各子模組說明。 |
| [fit_tools](fit_tools/README.md) | 為 Notebook 分析提供 flux 校正、樣本接合、權重與 loss 診斷。 |
| [design](design/README.md) | 搜尋候選設計參數，分析 HFSS sweep 與設計需求。 |
| `dispersive.py` | 組合 dispersive 搜尋、人工調整與圖表。 |
| `mist/tool.py` | 計算能量摺疊、不連續處理與碰撞遮罩。 |

Notebook caller 決定操作順序與人工確認時機，本目錄不管理 GUI tabs 或量測 session。
實驗 Run／Analyze／Save 入口由 [notebook adapter](../README.md) 擁有，不在此建立另一組實驗 records。

## 共用計算接縫

通用頻譜型別與整理交給 `zcu_tools.analysis.spectrum`。
Fluxdep 的 transition 型別、光譜集合 I/O 與資料庫搜尋交給
[analysis.fluxdep](../../analysis/fluxdep/README.md)。
診斷圖使用 [plotting.fluxdep](../../plotting/fluxdep/README.md) 的原生 Figure。
物理能譜查詢使用 [simulate.fluxonium](../../simulate/fluxonium/README.md)，參數容器與檔案由 resources 擁有。

Notebook 保留互動入口與工作流程專用計算，不把 GUI 的狀態管理搬到這裡。
套件定位見 [ADR-0072](../../../../docs/adr/0072-module-boundaries.md)。
