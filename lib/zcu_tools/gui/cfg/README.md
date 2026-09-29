# `zcu_tools.gui.cfg` — 共用設定機制

**Last updated:** 2026-09-30 — Input-first resource core

此 Qt-free package 擁有 Spec／Value、`CfgSchema`、完整 value tree、codec、paired assembler、binding 與 generic finished-cfg lowering。Spec 是靜態欄位契約，Value 保存可變編輯內容。`LiteralSpec` 的鎖定值由 Spec 定義；assembler 對齊 value，renderer 不另建鎖定權威。Spec 的 fluent 覆寫回傳新 frozen spec；value 的 `with_*` 是可變操作。Role defaults 與 deferred seed 由實驗 adapter 擁有，不由 structural `make_default_value` 猜測業務預設。

每個 Spec 欄位在 value tree 中都有 entry。停用 optional reference 用裸 `None`，未填 scalar 用 `DirectValue(None)` 保留 direct／expression 模式區別。codec 保存編輯狀態，不以 lowering 對 disabled reference 的省略規則寫 memento。`CfgSchema` 可以承載尚未完成的編輯；成品邊界顯式驗證完整性、literal、direct value 型別與 choices。dynamic expression 與 required value 在 lowering 時檢查；不在 `CfgSchema` 建構時拒絕所有編輯中間態。

`CfgResource` 擁有隔離的 input tree 與已發布 observation。Edit 先在候選 input 套用完整 batch，再解析新內容；被完整取代的舊 expression／reference 不會先被求值。Partial reference write、range step 計算與 `$` capture 只讀完成該意圖所需的固定來源。Scalar 文字／型別規則與 range 正規化由共用純函式處理，不依賴 field setter 的求值副作用。Binding 在 resource 中只建立最終解析快照，不是 input mutation 的權威。

既有 `binding.CfgDraft` 仍提供可編輯 field tree，提供 snapshot、valid、refresh 與 nominal `SettableTarget`。scalar 使用 dotted leaf，sweep 使用直接 edge，reference key 使用 `.ref`，子欄位直接下鑽。列舉和解析共用 grammar；不受理未列出的舊 `.sweep.*`／`.value.*` alias。Reference refresh 透過 caller 提供的 catalog。Linked value 跟隨當次 catalog snapshot；override 保留自身內容與 shape，不再解析原 key。Override 內的 nested linked reference 仍更新，明確 relink 才恢復這層的來源依賴。`ReferenceSpec.discriminator` 由 domain builder 明確宣告，指向每個 allowed shape 的唯一 literal 值；generic cfg 不從 literal 欄位排列推測辨識規則。完整 input round trip 與 caller 收斂仍見 [待實作設計](../../../../docs/adr/draft/cfg-editing-boundaries.md)。

`lower_finished_cfg` 固定 static、optional dynamic、lower 的順序；consumer 分別提供 expression evaluator、reference shape resolver、range factory，不傳入 broad app environment。linked reference 的 embedded value 與 live-key shape 查詢有不同作用；`EvalValue.resolved` 存在時 lowering 輸出該值；有 resolver 時 optional dynamic 階段仍先重新求值每個 expression 並轉成欄位型別，失敗即中止，結果不同只記錄 drift；沒有 `resolved` 才輸出 resolver 的求值。lowering 不更新 draft。program shape／raw policy 由 [`experiment.cfg_editing`](../../experiment/cfg_editing/README.md) 擁有；runtime generation、role policy 與 library writes 不屬本 package。跨 owner 理由見 [ADR-0065](../../../../docs/adr/0065-cfg-editing.md)。
