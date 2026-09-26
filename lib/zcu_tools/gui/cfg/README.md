# `zcu_tools.gui.cfg` — 共用設定機制

**Last updated:** 2026-09-27 — Cfg 文件分流

此 Qt-free package 擁有 Spec／Value、`CfgSchema`、完整 value tree、codec、paired assembler、binding 與 generic finished-cfg lowering。Spec 是靜態欄位契約，Value 保存可變編輯內容。`LiteralSpec` 的鎖定值由 Spec 定義；assembler 對齊 value，renderer 不另建鎖定權威。Spec 的 fluent 覆寫回傳新 frozen spec；value 的 `with_*` 是可變操作。Role defaults 與 deferred seed 由實驗 adapter 擁有，不由 structural `make_default_value` 猜測業務預設。

每個 Spec 欄位在 value tree 中都有 entry。停用 optional reference 用裸 `None`，未填 scalar 用 `DirectValue(None)` 保留 direct／expression 模式區別。codec 保存編輯狀態，不以 lowering 對 disabled reference 的省略規則寫 memento。`CfgSchema` 可以承載尚未完成的編輯；成品邊界顯式驗證完整性、literal、direct value 型別與 choices。dynamic expression 與 required value 在 lowering 時檢查；不在 `CfgSchema` 建構時拒絕所有編輯中間態。

`binding.CfgDraft` 擁有一份可編輯 field tree，提供 snapshot、valid、refresh 與 nominal `SettableTarget`。scalar 使用 dotted leaf，sweep 使用直接 edge，reference key 使用 `.ref`，子欄位直接下鑽。列舉和解析共用 grammar；不受理未列出的舊 `.sweep.*`／`.value.*` alias。reference refresh 透過 caller 提供的 catalog；現行 modified／missing key 與 lowering 的具體語意不等於已核准的完整 override 目標，見 [待實作設計](../../../../docs/adr/draft/cfg-editing-boundaries.md)。

`lower_finished_cfg` 固定 static、optional dynamic、lower 的順序；consumer 分別提供 expression evaluator、reference shape resolver、range factory，不傳入 broad app environment。linked reference 的 embedded value 與 live-key shape 查詢有不同作用；`EvalValue.resolved` 存在時 lowering 使用該值，沒有才依傳入 resolver 求值。lowering 不更新 draft。program shape／raw policy 由 [`experiment.cfg_editing`](../../experiment/cfg_editing/README.md) 擁有；runtime generation、role policy 與 library writes 不屬本 package。跨 owner 理由見 [ADR-0065](../../../../docs/adr/0065-cfg-editing.md)。
