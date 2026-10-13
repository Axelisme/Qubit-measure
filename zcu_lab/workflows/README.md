# 使用者 workflows

- `coherence_fluxdep/`：逐個 flux 點校準並量測 T1、Ramsey 與 Hahn echo。
- `mist_fluxdep/`：逐個 flux 點校準並掃描 MIST disturbance gain。
- `overnight/`：在固定 flux 下按固定間隔量測 T1 與 Hahn echo，不做 feedback。

每個目錄有 `workflow.py` 與 `form.py`；前兩項另有可直接執行的 `export.py`。
`common/` 放多個 workflow 共用的實驗步驟與表單 helper。
`registry.py` 依上述順序配對 workflow 與表單，交給 scripts 注入 lib 宿主。

科學流程與保存契約見 [definitions](definitions.md)，表單與宿主接線見 [forms](forms.md)。
