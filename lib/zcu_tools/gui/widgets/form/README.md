# gui/widgets/form — Form draft view

**Last updated:** 2026-10-20，來源拒絕顯示為欄位錯誤

`FormEditor` 借用 caller-owned `FormState` 與 host 捕捉的 `FormEnv`。Editor 直接讀 `state.describe(path, env)` 與完整 `preview`，不產生 JSON Schema，也不解析 annotation、union 或選項匹配。Core 擁有 choices、active option、固定欄位與 lone ordinary wrapper 攤平規則。攤平只改版面，controls 的完整 path 仍含 wrapper。

Editor 初次只建立可見層與初始展開的 model rows。Group 預設折疊，model 沿 `Display(collapsed=...)`。Core 的 `can_expand` 只提供結構上的展開能力，首次展開才查下一層；實際沒有可畫 children 時，該列回到普通標題，不留空白區塊。收起已建區塊會保留 QWidget 與未提交文字，不偷寫 state。實際 branch 改變或啟停普通 nullable model 時只替換該子樹，無關 rows 與使用者設定的 folds 保持不變。沒有 converter 的選項切換只更新 selector，不清除仍相同的 children 與 local parse text。Editor 以 description.branch 判斷子結構，不用 selected 猜 branch。Editor 以已建立的 Qt 子層記錄展開情況，不另持 writable draft。

文字欄位在 Enter 或一般 focus loss 提交，typing 本身不改 state。折疊與移除舊分支引起的 focus loss 不提交，重新展開後的一般 focus loss 仍提交保留文字。Expression action、dropdown、checkbox 與 Enabled 立即提交。未解析或 host 拒絕的輸入保留在 view，`local_parse_errors()` 讓 host 阻止提交。`discard_local_text()` 只丟掉 view-local 輸入，不重設 FormState。外部 publication 仍以 state 的輸入更新已建 controls。完整 preview 與 local errors 決定 validity，折疊、hidden 或尚未建立的 child 不會隱藏錯誤；錯誤標示落在可見祖先或 group 標題。

Host 可訂閱 `errors_changed`，在 `errors()` 結果改變時取得新的 `FormErrors`，全部有效時取得 `None`。本地 parse／拒絕錯誤與 state publication 都會觸發比較；錯誤集合相同時不重送，detach 後不再發出。`validity_changed` 仍只在有效與無效之間切換時發出 bool。

Model-list 的 `form.picker.<path>` 依 Core 的 elements 列出「index: name」。Picker 只控制哪個元素可見，不寫 FormState、不改 participants，也不抓取來源。切走保留該元素的 controls、未提交文字及 parse errors；切回能繼續修正。隱藏元素的錯誤標在 list 標題，row tooltip 指出元素名稱與欄位。Participants 使用獨立 JSON 清單輸入。`Display` 提供元素 label 與 readonly 呈現提示，固定 identity 的拒絕由宣告 owner 提供。唯讀模式仍可切 picker、展開與複製，不能編輯值。

`selection_fixed` 不鎖內層選項。`fixed` option.value 也不凍結 children 或移除 commands；只有固定 scalar 且沒有 command 時畫成唯讀文字。

Host 必須同時傳入 `write=`、`select=` 與 `run_command=`，或全部省略。三者分別接收 parsed input batch、typed path 與 option.value、typed path 與 command 名稱，並使用同一 guard。Dropdown 只呼叫 select，不以值寫入觸發 converter。Editor 不另寫 state。Host 以 `WriteRejected(message)` 拒絕時，文字保留且顯示訊息；dropdown、checkbox 與 Enabled 回到已發布值。各 mutation 路徑都處理明確的 `FormConversionError` 或 `FormInputRejected`。來源拒絕的 `PydanticCustomError` 也顯示為欄位錯誤，不送到 Qt 的 unhandled exception。失敗時顯示 row tooltip，selector 回到已提交的 selection，不重跑 write 或 converter，也不發布 mutation。每個 row 有 `form.row.<canonical path>` objectName，可定位該列及其拒絕原因。其他例外照常拋出。沒有 callback 時，editor 直接呼叫 FormState 的 write、select 與 run_command。

`set_editing_enabled(enabled=False)` 保持表單可讀，不停用整個 widget。文字可選取與複製，fold titles 可展開、折疊與捲動；寫入與 commands 不可用。唯讀時第一次展開的 controls 也遵守這項限制。`refresh(env)` 使用新 capture 重讀已建層的 descriptions，不沿用舊 choices 或 active branch。Detach 與 close 只解除訂閱，不關閉 caller 的 state。

欄位錯誤不增加 row 高度。Direct 文字與 dropdown 以目前 palette 的 Base 混合淡紅底色，checkbox 與 model section 的名稱標紅。Expression 保留原底色，改以紅色邊框標示錯誤。帶色控制項或名稱標籤的 `formError` property 反映錯誤狀態，tooltip 先顯示錯誤再保留說明。錯誤消失時還原顏色與 tooltip；form-wide error label 與 group 的 Contains errors 提示仍保留。Expression 的求值結果以 `= ` 前綴顯示，字串用 JSON 引號。Direct 數值的 authored input 與 preview value 不同時，旁邊以 `Effective: ` 明示有效值；相同時不重複顯示。Step 不整除後的 stop／center／span 使用這條路徑，Qt 不重做採樣公式。

Measure Run panel 使用 tab resource 的 guarded callbacks。Workflows 使用自己的 FormState 與 Apply／Discard／Start policy。Inspect、New Entry 與 writeback 編輯器沿用各自的 entry capture、Apply／Cancel 與 destination guard。共用 editor 不接管這些 lifetime 或提交政策。公開 control object names、keyboard 與 Expression 細節見 `FormEditor` docstring。

文字欄位的右鍵選單保留標準文字操作與 Expression。Owner description 的 field commands 顯示在掛載 row 的標題右鍵選單，包括 sweep 的折疊標題。Editor 不換算座標，只檢查 command.inputs 上的 local parse errors，再送既有 guarded run_command。換分支後保留不參與換算的 local 原文，包括未解析的 n／step。唯讀時 command actions 停用，晚建立的欄位也相同。Selector 是否畫成 dropdown 由 description.selector 決定，不由 Qt 猜 sweep 型別。

共用 scalar 控制項 helpers 位於 `value_widgets.py`，analysis forms 直接重用。Library／Inspect 與 module writeback 不保留第二套 cfg widget。宿主提交邊界見 ADR-0008、ADR-0065。
