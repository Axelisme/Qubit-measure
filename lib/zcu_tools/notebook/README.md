# `zcu_tools.notebook`

`zcu_tools.notebook` 提供 Notebook 逐步探索時使用的互動入口、顯示與 widgets，也保留工作流程專用的分析支援。Notebook 工作流程可組合計算與人工確認，不等於 GUI 的量測 session 或狀態管理。實際操作與結果解讀見 [Notebook 內容入口](../../../notebook_md/README.md)；這裡說明支援程式的位置。

## 工作家族

- [fluxdep](analysis/fluxdep/README.md)：通量依賴光譜的資料處理、擬合、圖表與互動選點。`fitting.py` 目前仍實作 database search 與 `fit_spectrum`，search 可回傳診斷圖。
- [t1_curve](analysis/t1_curve/README.md) 與 [t2_curve](analysis/t2_curve/README.md)：各自保留 Notebook 的曲線分析、擬合與分階段工作流程。模型選擇與保存條件見各自的 README。
- [fit_tools](analysis/fit_tools/README.md)：支援 Notebook 分析中的校正、資料接合、loss、weights 與溫度模型；不把這些能力一概視為共用分析核心。
- [design](analysis/design/README.md)：評估模型參數、設計需求與候選組合，並分析 HFSS sweep 資料。
- [mist](analysis/mist/tool.py)：提供能量摺疊、不連續處理與碰撞遮罩的計算工具。
- [circuit_design](circuit_design/README.md)：提供 Qiskit Metal 電路幾何元件；相關 Notebook 展示電路建構及設計檔輸出。
- [`utils.py`](utils.py)：提供 sweep、圖檔保存與設備資訊等 Notebook 輔助函式。
- [`persistance.py`](persistance.py)：目前提供舊版參數結果的讀取、寫入與更新 helper；不負責共用原始頻譜的正規化。

## 共用責任與目前邊界

共用原始頻譜型別與座標整理位於 [`analysis/spectrum.py`](../analysis/spectrum.py)。Fluxdep 的共用 transition 型別、轉換及頻譜集合 I/O 位於 [`analysis/fluxdep`](../analysis/fluxdep/README.md) 的 `models.py`、`io.py`。Notebook 的 fluxdep search 與 `fit_spectrum` 目前仍在上述 Notebook 工作家族內；不要把共用型別的位置當成 search 已經搬移。

物理能譜等計算由 [`simulate/fluxonium`](../simulate/fluxonium/README.md) 提供，參數資料的 typed owner 見 [`resources`](../resources/README.md)。Notebook 可使用這些能力，也保留工作流程所需的專用計算。`TwoLinePicker` 的 Matplotlib rendering 目前仍在 [`analysis/fluxdep/line_picker.py`](../analysis/fluxdep/line_picker.py)，Notebook 的選點介面在 `analysis/fluxdep/interactive/`；此處只描述現有位置，不指定後續搬移。
