# Notebook 用途入口

`notebook_md/` 收錄 Jupytext Markdown 格式的 Notebook。Notebook 記錄輸入、探索步驟與結果解讀；支援程式的 README 說明可呼叫的介面和限制。

- [`single_qubit.md`](single_qubit.md) 已改為 FTF 公開 acquisition 範例，示範共用 GUI／Notebook environment、有限 microwave、flux excursion、lookback 與 round 間 bias 修改。它不連硬體，不依賴尚未交付的單顆 preset。`simulation/environment.py` 與相鄰 YAML 提供可執行組裝入口。
- [`analysis/mist/branch.md`](analysis/mist/branch.md) 示範從既有 entry 與 typed fits 讀取一個 point，計算與匯出 branch populations。它不產生 acquisition 或接受校正值。
- [`analysis/`](analysis/) 另有 flux-dependence、T1/T2 等主題；Fluxonium 擬合工具的契約見 [fluxdep README](../lib/zcu_tools/notebook/analysis/fluxdep/README.md)。
- [`circuit_design/`](circuit_design/) 收錄電路設計 Notebook。相關程式見 [circuit design README](../lib/zcu_tools/notebook/circuit_design/README.md)。

需要 entry 的參數範例先在 kernel 顯式執行 `zcu_lab.components.register_all(component_registry)`，再開啟條目與 point。重跑 bootstrap 前重新啟動 kernel，重複註冊會報錯。Existing-data 範例要求已完成離線遷移，不讀舊的 `exps/` 或 `meta_info.json`。Result 與 Database roots 分別傳入，entry 名稱與 point label 都只能是一個路徑段。範例的 Q1、R1 是元件名稱，不是由 entry 名稱推導的身分。

Point 的 flux_value、flux_half 與 flux_period 必須使用相同的顯式 flux_unit。容器不換算數值。新 run 的保存與接受契約見 [Notebook owner](../lib/zcu_tools/notebook/README.md)。

重複或實驗專用的 `test.md`、`develope.md` 與 `analysis/mist/branch_floquet.md` 已退休。前者的環境與實驗接線由 single_qubit 範例取代；後兩者的模擬探索不再提供逐 cell 範例，branch 範例只保留一條離線分析路徑。計算能力仍由原有 simulate、analysis 與使用者實驗套件提供。

Markdown 與 ipynb 的生成來源不由本入口決定，同步與覆寫風險見 [scripts/README.md](../scripts/README.md#資料作業)。
