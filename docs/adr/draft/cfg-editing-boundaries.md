# Cfg 編輯接縫與使用邊界

**狀態：部分目標已落實，剩餘範圍待實作。** Measure tab 的 resource lifetime、同步 publication、override、固定 Run acceptance 及 frontend 接線見 [ADR-0065](../0065-cfg-editing.md) 與 [resource contract](cfg-resource-contract.md)。本篇保留 library conversion、selected Apply 與其他 app 的目標，不是它們的現行操作契約，也不把 measure tab 的完成推廣到四個 app。

## 實驗編輯接縫

已移至 `experiment.cfg_editing` 的 catalog／materializer 不等於整條 library-entry 編輯路徑已收斂。目前 measure 的 `services/cfg_editor.py::_initial_schema` 仍直接取 `ModuleLibrary` raw entry，經 app-local `cfg_schemas` 轉換；measure `cfg_schemas` 與 Autofluxdep `cfg/module_adapter.py` 各自正規化 runtime object。generic GUI 尚無由實驗側按 entry identity 提供 schema，並一併收斂 read、validate、commit／lowering 的 library-entry editing port。

目標是由實驗側集中 program conversion、runtime normalization、discriminator 與 missing/default policy，上層 GUI／session／remote 不辨認 program type。library editor 以 entry identity 請實驗側提供可編輯 schema；composition 注入窄能力，converter 不取得 `ContextService` 寫入權。Default、role 與可編輯子集仍由領域 adapter 宣告，不把 program vocabulary 放回 generic cfg。`experiment.cfg_editing` 依已裁決的窄 import 例外可依賴 Qt-free `gui.cfg`；import package root 仍載入 experiment base，不承諾完全輕量。

**轉正條件：** library-entry read／validation／commit／lowering 不再由上層分派 program discriminator；兩 app 的重複 normalization 與 conversion 收斂到實驗 owner；確認 typed inputs、unsupported-shape failure 與寫入 owner，更新各 caller 和模組契約。

## 編輯狀態、刷新與 override

Measure tab 由 `TabCfgResources` 持有 headless resource，Qt detach 不撤銷它。`CfgEditorService` 只管理獨立 library／inspect／writeback draft。Autofluxdep 的 `ui/node_cfg_form.py` 仍在 widget 建立時產生 Default／Generation 兩個局部 draft，關閉時關閉它們。未開啟 form 的 placement 只有保存的 value tree，沒有同等的 resource-owned refresh。兩種用途不需合為一棵樹，也不需共用 measure service。

目標是每個邏輯資源由 app resource owner 持有唯一可編輯 headless model，frontend 共用，widget attach／detach 不支配 model lifetime。source owner 發布變更後由資源 owner 自動 refresh，且可手動刷新指定資源；不可偷偷改用另一 context、輪詢硬體或重跑 fresh defaults。Linked reference 保留 identity，以來源最新內容更新仍跟隨的整份子樹；override 的這一層不再依賴原 entry，刪除原 key 不單憑此使內容失效，但內含 expression、其他 linked reference 和 asset key 仍各自維持依賴。EvalValue 保留 expression 與模式並更新解析／有效性；resolve-once 值不重讀。失敗保留輸入與診斷，stale result 不冒充有效值。Measure resource 已依這些規則處理 override 及 nested dependency；共用 binding 的 override 不再因原 key 消失就改成 custom key。其他 app 是否使用完整 resource refresh 與固定 acceptance，仍須按其 caller 路徑核實。

直接修改 linked 內容應在同一編輯命令內形成整份 reference override，不能把尚未可用的 stale 內容無聲提升為有效副本。Restore defaults 依 owner definition 重新初始化明確範圍；relink 指定 key，先驗證其可建立且 shape 合法，成功才整份替換，失敗保留 override。兩者不是 refresh 或整份 draft discard。Measure resource 的巢狀解除範圍與明確 Custom 命令見 resource contract；其他 app 的接線仍保留自己的驗收義務。

**轉正條件：** 沒有 widget 時仍可刷新／編輯該資源；source change、manual refresh、missing/recovery、override、nested dependency 與 failed relink 的結果已核實；owner 文件定義生命周期、錯誤和具體命令。

## Observation、Run 與 Apply

Measure tab 已使用同 revision 的完整 publication、atomic batch 及 caller 明示 ref 的固定 Run acceptance。它採同步 Valid／Invalid／Unavailable，不使用下段早期目標中的 Pending 狀態。獨立 `CfgDraft` 提供 snapshot／valid；library／writeback 的 `CfgEditorSession.set_fields` 仍逐筆修改 live draft，失敗保留成功前綴。Autofluxdep `NodeCfgSchema.lower` 與獨立 draft 的 lowering 可在使用時讀取 md／ml。Writeback 雖然先準備 selected entries 並只送一次 `ContextWritePort`，單次呼叫本身不能證明所有選中 md／ml writes 的失敗原子性。

目標是 owner 發布同一 revision 的完整編輯輸入、解析結果與有效性；變更或 refresh pending 時不得沿用舊版本。Run／Apply 依 caller 指定的有效 revision 接受輸入，對 stale、pending、invalid 或不符版本明確拒絕，不隱含 refresh，也不靠 lower 時重讀 live source 換值。合法但未完成的編輯可發布 invalid observation，不能當成 run snapshot。接受後的 run snapshot 固定；Autofluxdep 每點仍可套用 workflow 宣告的合法 patches，硬體 guard 不因 cfg 固定而省略。編輯發布、selected Apply、memento 與 Run 有各自的提交目的；不以 memento 的可還原狀態假裝可執行。

同一 cfg 的 edit batch 依序在未發布候選上解析 path，全部可接受才發布一次最終 observation 與 net diff；任何不能套用的操作不發布成功前綴。這不要求每次合法編輯都完成成品驗證。同一 context 的 selected library／writeback Apply 先準備全部選中項目與 guards，任何失敗不修改 live md／ml；成功由既有 write owner 在明確邊界提交。提交後的 observer／durability failure 要與未提交失敗區分；不宣稱跨檔 crash transaction、跨 context 原子性或硬體 transaction。

**轉正條件：** 同版本 observation 與 caller revision guard 可觀察；Run／Apply 不偷換來源；batch failure 無 live prefix 且成功只發布最終結果；selected Apply 的準備／提交失敗與後續通知失敗可區分。版本／wire schema、no-op policy、碰撞、成功後草稿處置及同步機制需另定有界契約，不能由本 draft 猜測。
