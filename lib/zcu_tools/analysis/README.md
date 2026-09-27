# `zcu_tools.analysis` 模塊重點文檔

**Last updated:** 2026-09-27 — fluxdep transition ownership

`zcu_tools.analysis` 放置不依賴 notebook widget 或 Qt widget 的分析核心。各 notebook
與 GUI 套件只負責把使用者事件、圖表容器、與 workflow 狀態轉成這裡的純函式或互動狀態機呼叫。

`spectrum.py` 提供兩個分析 GUI 共用的原始頻譜資料型別，以及 Hz→GHz 與座標遞增排序。
其中 `fluxs` 是座標軸，不是 flux 校正參數。

子模組：

- `fluxdep/`：Flux-Dependence Analysis 的選點、filtering、line selection、one-tone peak detection
  共用 kernel。`fluxdep/models.py` 擁有頻譜校正／點位資料、transition 型別與能譜到躍遷頻率的共用轉換；
  `fluxdep/io.py` 讀寫 HDF5 頻譜集合。資料庫搜尋、診斷圖與 export workflow
  目前仍由 notebook/GUI adapters 管理。
- `fitting/`：擬合模型、shared/fixed fitting、singleshot 與 resonance fitting。契約見 [fitting README](fitting/README.md)。
