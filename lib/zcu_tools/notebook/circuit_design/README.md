# Circuit design

**Last updated:** 2026-09-27

這裡放 Qiskit Metal 的電路幾何元件。`fillet_q.py` 定義帶圓角焊盤的 qubit，`round_tee.py` 定義 T 形耦合元件，`self_define.py` 定義 Cross 等幾何元件。`notebook_md/circuit_design/R59_C1,1_3Q.md` 使用這些元件建構平面電路，透過 Qiskit Metal 的 GDS renderer 輸出 `.gds` 設計檔，並展示 HFSS / Q3D 模擬設定。`notebook_md/circuit_design/utils.md` 示範依目標頻率與線寬估算耦合長度。

此流程負責電路建構與設計檔輸出，與 [`notebook/analysis/design`](../analysis/design/README.md) 的模型參數評估彼此獨立，沒有固定上下游。使用者的 GDS、模擬專案與結果不屬於此程式目錄，也不隨模組搬移。

執行 Qiskit Metal 範例需安裝 `design` optional profile；該 profile 只支援 Python 3.12（`pyproject.toml` 的 qiskit-metal dependency marker 排除 Python 3.13）。待核實：在已安裝 `design` profile 並可使用 Qiskit Metal、ANSYS 的環境中實際執行 GDS 輸出與 HFSS / Q3D 模擬。本 lane 的 Python 3.13 環境未安裝 qiskit-metal。
