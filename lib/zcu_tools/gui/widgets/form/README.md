# gui/widgets/form — Form draft view

**Last updated:** 2026-10-15，單層描述、首次展開建立 controls 與局部子樹更新

`FormEditor` 借用 caller-owned `FormState` 與 host 捕捉的 `FormEnv`。Editor 直接讀 `state.describe(path, env)` 與完整 `preview`，不產生 JSON Schema，也不解析 annotation、union 或選項匹配。Core 擁有 choices、active option、固定欄位與 lone ordinary wrapper 攤平規則。攤平只改版面，controls 的完整 path 仍含 wrapper。

Editor 初次只建立可見層與初始展開的 model rows。Group 預設折疊，model 沿 `Display(collapsed=...)`。Core 的 `can_expand` 只提供結構上的展開能力，首次展開才查下一層；實際沒有可畫 children 時，該列回到普通標題，不留空白區塊。收起已建區塊會保留 QWidget 與未提交文字，不偷寫 state。換 option 或啟停 nullable model 只替換該子樹，無關 rows 與使用者設定的 folds 保持不變。Editor 以已建立的 Qt 子層記錄展開情況，不另持 writable draft。

文字欄位在 Enter 或 focus loss 提交，typing 本身不改 state。Expression action、dropdown、checkbox 與 Enabled 立即提交。未解析或 host 拒絕的輸入保留在 view，`local_parse_errors()` 讓 host 阻止提交。`discard_local_text()` 只丟掉 view-local 輸入，不重設 FormState。外部 publication 仍以 state 的輸入更新已建 controls。完整 preview 與 local errors 決定 validity，折疊、hidden 或尚未建立的 child 不會隱藏錯誤；錯誤標示落在可見祖先或 group 標題。

Host 可成對傳入 `write=` 與 `run_command=`，使每批 parsed input 與選項指令走同一 guard。Editor 不另寫 state。Host 以 `WriteRejected(message)` 拒絕時，文字保留且顯示訊息；dropdown、checkbox 與 Enabled 回到已發布值。其他例外照常拋出。沒有 callback 時，editor 直接呼叫 FormState 的公開寫入與指令介面。

`set_editing_enabled(enabled=False)` 保持表單可讀，不停用整個 widget。文字可選取與複製，fold titles 可展開、折疊與捲動；寫入與 commands 不可用。唯讀時第一次展開的 controls 也遵守這項限制。`refresh(env)` 使用新 capture 重讀已建層的 descriptions，不沿用舊 choices 或 active branch。Detach 與 close 只解除訂閱，不關閉 caller 的 state。

Measure Run panel 使用 tab resource 的 guarded callbacks。Workflows 使用自己的 FormState 與 Apply／Discard／Start policy。Inspect、New Entry 與 writeback 編輯器沿用各自的 entry capture、Apply／Cancel 與 destination guard。共用 editor 不接管這些 lifetime 或提交政策。公開 control object names、keyboard 與 Expression 細節見 `FormEditor` docstring。
