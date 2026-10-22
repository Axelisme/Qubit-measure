# `zcu_tools.analysis` 模塊重點文檔

**Last updated:** 2026-10-22

**讀者：** framework-dev。分析核心、純數值工具與前端 adapter 的分工速查。

`zcu_tools.analysis` 放置不依賴 notebook widget 或 Qt widget 的分析核心。各 notebook
與 GUI 套件只負責把使用者事件、圖表容器、與 workflow 狀態轉成這裡的純函式或互動狀態機呼叫。

Package root 在 caller 選用 `fluxdep` 時才載入它。直接使用 `analysis.fitting` 不載入
fluxdep 的互動 line picker，也不要求 QICK。

`spectrum.py` 提供兩個分析 GUI 共用的原始頻譜資料型別，以及 Hz→GHz 與座標遞增排序。
其中 `fluxs` 是座標軸，不是 flux 校正參數。

`failure.AnalysisFailed` 表示數值 owner 已辨識的無法產出分析結果，與任意程式錯誤分開。
Experiment 公開同一型別，GUI analyze work 經分類後才讓 recipe 交付 failed 值。
分類範圍由各數值 owner 文件列明，不以 ValueError／RuntimeError 的文字猜測。

子模組：

- `fluxdep/`：Flux-Dependence Analysis 的選點、filtering、line selection、one-tone peak detection
  共用 kernel。`fluxdep/models.py` 擁有頻譜校正／點位資料、transition 型別與能譜到躍遷頻率的共用轉換；
  `fluxdep/io.py` 讀寫 HDF5 頻譜集合；`fluxdep/search.py` 負責共用 database search。
  診斷圖由 plotting 提供，顯示與 export workflow 由 notebook/GUI adapters 管理。
- `dispersive/`：色散分析的前處理、electrical-delay refinement 與 auto-tune 數值核心；契約見 [dispersive README](dispersive/README.md)。
- `fitting/`：擬合模型、shared/fixed fitting、singleshot 與 resonance fitting。契約見 [fitting README](fitting/README.md)。

`singleshot.py` 擁有純數值 shot 分類、Rabi 分類比例及 population 換算。QICK raw buffer 的 accumulator normalization 留在 caller 的實驗 helper；本模組不讀 program metadata。
