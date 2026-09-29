# Notebook 用途入口

`notebook_md/` 收錄 Jupytext Markdown 格式的 Notebook 內容。依工作目的選擇主題，不必為每份 Notebook 尋找同名的 Python helper。這裡不決定 Markdown 與 ipynb 的生成來源，也不承諾同步腳本是正式發布程序。

- 量測與實驗流程：[`single_qubit.md`](single_qubit.md) 包含單量子位量測設定；[`overnight.md`](overnight.md) 是 overnight 實驗流程；[`autofluxdep.md`](autofluxdep.md) 使用 autofluxdep 實驗套件。硬體連接、輸出及執行條件以各 Notebook 及 [實驗套件](../lib/zcu_tools/experiment/v2/README.md)的說明為準。
- 結果分析：[`analysis/`](analysis/) 放分析與擬合 Notebook，包括 flux-dependence 和 T1/T2 主題；Fluxonium 擬合工具的契約見 [fluxdep README](../lib/zcu_tools/notebook/analysis/fluxdep/README.md)。
- 電路設計：[`circuit_design/`](circuit_design/) 收錄設計用 Notebook，與量測流程分開。相關程式見 [circuit design README](../lib/zcu_tools/notebook/circuit_design/README.md)。

Notebook 記錄探索步驟、輸入與結果解讀；支援程式的 README 說明可呼叫的介面和限制。一個主題可以有多份 Notebook，沒有必要逐檔對應同名程式。同步操作及覆寫風險見 [scripts/README.md](../scripts/README.md#資料作業)。
