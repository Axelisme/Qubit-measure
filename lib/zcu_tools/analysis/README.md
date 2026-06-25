# `zcu_tools.analysis` 模塊重點文檔

**Last updated:** 2026-06-25（fluxdep analysis kernel）

`zcu_tools.analysis` 放置不依賴 notebook widget 或 Qt widget 的分析核心。各 notebook
與 GUI 套件只負責把使用者事件、圖表容器、與 workflow 狀態轉成這裡的純函式或互動狀態機呼叫。

目前子模組：

- `fluxdep/`：Flux-Dependence Analysis 的選點、filtering、line selection、one-tone peak detection
  第一批共用 kernel。資料庫搜尋、診斷圖、與 export pipeline 仍留在既有 notebook/GUI adapters。
