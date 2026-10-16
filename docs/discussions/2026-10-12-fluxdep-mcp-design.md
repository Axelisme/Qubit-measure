---
date: 2026-10-12
status: active
superseded_by: null
related:
  - docs/discussions/2026-10-12-agent-interface-layers.md
  - docs/adr/0075-agent-interface-layers.md
  - docs/adr/0068-remote-transport.md
basis:
  - main@0e469d8bc
---

# fluxdep MCP 的定位與介面重設計

## 結論

fluxdep_fit 的 MCP 依「agent 在哪裡做判斷」重新設計，不再鏡像 GUI 的每個機械操作。

- **分層定位**：MCP 面向 agent，設計時要考慮 agent 的使用負擔。RPC 是控制接口，設計上要簡單、基礎、通用。MCP 透過組裝 RPC 兼顧兩邊的需求。這是這次建立的 MCP 設計範式的第一原則。
- **人與 agent 共駕**：兩邊都能做每一步，同一時刻由控制權的 holder 寫入。控制權模型沿用另一個 task 的設計（控制身份），本討論不重談。
- **決策點**：agent 要判斷的地方有六個。每個決策點回一張「證據圖」；程式判斷不了的事才回圖。

  | 決策點 | 判斷什麼 | 證據圖 |
  | --- | --- | --- |
  | J1 資料可用性 | 這張譜要不要用 | 原始譜 |
  | J2 對齊 | 數據完整覆蓋 half／integer 時，設初值後跑 auto_align；只覆蓋部分時，繼承前一張譜的位置或手動設定，不跑 auto_align | 對齊結果 |
  | J3 採樣 | 盡量選上可識別的線；可以帶著猜測選，也可以不知道是哪條躍遷 | 點疊在譜上 |
  | J4 躍遷假設 | 猜躍遷身份，決定 DB 與 bounds，然後 search | 前面各步的圖 |
  | J5 驗收 | 檢查模型曲線疊在資料點上的結果 | 疊圖，點依配到的躍遷上色 |
  | J6 剔除 | 濾掉沒有躍遷能解釋的點或雜訊，不必回單張譜重做 | 對照 J5 的疊圖 |

  各步不一定照順序，也不一定只做一次。後段是 J4 → J5 → J6 的迴圈。
- **能力清單**：MCP 不必涵蓋 GUI 的全部能力。它只提供這些能力：開啟或連上 GUI；讀取、加入、移除數據；錨定對稱點；生成採樣點；Export 與 Restore；Filter；附帶參數擬合；查看結果；匯出擬合參數給 dispersive。
- **每張譜各自保有互動 context**：切換 active 譜只換顯示，不關閉別張譜還沒完成的編輯。active 改成純顯示狀態，agent 操作哪張譜，GUI 就跟過去。
- **tool 即意圖**：MCP 沒有「開啟」這個動作。對一張譜送哪一類命令，它就進入那個階段。
- **命令三原則**：
  1. 沒有隱藏模式，模式設定寫成命令參數。
  2. 能用其他命令表達的就不提供。
  3. 每個階段由編輯命令加上 finish 組成。finish 是決策點的「交卷」動作，回覆帶證據圖。

  依此得到的命令集：

  | 能力 | 命令 |
  | --- | --- |
  | 錨定 | `move_line(role, position, conjugate=False)`、`auto_align`、`finish` |
  | 採樣 onetone | `set_threshold`、`finish` |
  | 採樣 twotone | `stroke`、`set_settings`、`undo`、`finish` |
  | Filter | `stroke`、`set_min_distance`、`undo`、`finish` |
- **回覆**：
  - MCP 不回逐點陣列，改回計數、範圍與差量；摘要同時列出參數和點的變化。
  - 機械命令只回數字摘要；agent 要圖時另外要求。
  - commands schema 寫在工具描述裡，不附在每個回覆。
- **等待**：`auto_align` 和 fit 的 search 都等到完成才回，不設上限。fit 把設參數和啟動 search 合成一步。
- **其他**：
  - 新增 DB 資訊查詢：參數範圍、能階數、flux 範圍。
  - 參數範圍寫進參數規格（minimum／maximum）。
  - Export／Restore 是採點階段結束後的 checkpoint，不保存 Filter 結果。
  - Restore 遇到同名譜就拒絕。

- **失效與品質**：fit 結果依流程依賴模型判定是否過期：對齊 → 該譜的採樣點 → 該譜的 Filter 遮罩 → fit，加減譜與參數也在鏈上。任一上游在結果記錄後改變，結果標示過期並寫出原因，不清除；過期的結果不能匯出、不回疊圖。search 回覆附平均殘差、每條 transition 配到的點數與平均、最大殘差，以及殘差偏大的點數；J5 疊圖標出偏大的點；GUI 的 Show tab 與 MCP 共用同一個渲染。
- **工具與回覆骨架**：工具從 24 個減到 14 個（lifecycle 另計）。`project_info`／`project_setup`／`state_check` 併進 connect；selection 的快照併進 `spectrum_list`；設參數、search 與 operation 工具併成 `fluxdep_fit`。回覆骨架是階段、是否有未 finish 的編輯、目前數值、只列有變欄位的 `changes`（值為 `[before, after]`），以及因此失效的下游。`spectrum_load` 與 `spectrum_snapshot` 預設附原始譜。每個互動命令組裝 RPC 的 open（冪等）加 command 兩次呼叫。fit 的資料庫參數改名 `search_db`，專案的原始資料根目錄改名 `data_root`。
- **估算**：同一份分析約 20k token，是量測到的 17%；呼叫次數從 177 降到約 60。

這些決策目前記在本機 task map。元件定位的裁決會在控制身份 task land 後寫進 ADR。回覆與命令的原則屬於設計範式，寫在 spec，不寫 ADR，也不抽成共用程式碼。

## 問題

使用者認為 fluxdep_fit 的 MCP「對agent不友好」。

當時只有開發測試的紀錄，所以先讓 subagent 用真實的 twotone flux spectrum 跑一次完整分析，再量測它的 transcript。量測結果：
- MCP 回覆約 115k token，佔最終 context 的 48%。
- 逐點陣列約佔回覆的 38%。
- agent 照 server instructions 重讀資源約佔 42%（與陣列部分重疊）。
- 每個互動回覆重複附上的 commands schema 佔 18%。

agent 回報的不順手處，一半出在 GUI 端的行為，不在回覆格式：
- 開新的互動 context 會靜默關掉舊的。
- finish 後 GUI 自動開下一個 picker，但回覆裡看不到。
- `auto_align` 立即回 busy，agent 要另外輪詢。
- 改了對齊之後，舊的 fit 結果仍然顯示。
- 參數範圍不在 schema 裡。

## 考慮過的選項與比較

### 定位：人主導、共駕，或 agent 主導

agent 主導可以讓介面完全為 agent 設計。共駕則要求人看得懂 agent 做了什麼，並能隨時接手。使用者選擇共駕。

### MCP 的粒度：GUI 對等，或能力清單

最初的方向是讓 MCP 至少涵蓋 GUI 的每一步，再加上 MCP 專屬命令。重新檢視 agent 的決策點之後，使用者改成以能力清單定義 MCP，GUI 的其他能力只留在 RPC。

### 回覆精簡在哪一層做

- 在 GUI remote 投影做：wire 只有 MCP 一個使用者，改起來最直接。
- 在 MCP 組裝：符合「RPC 簡單通用、MCP 面向 agent」的分層。也可以只在需要時才請 GUI 出圖，不必讓 GUI 每個命令都渲染 PNG。

使用者先以「只有一個使用方」選了前者，隨後補充分層定位，改成後者。

### finish 的去留

- 每個命令立即發布：agent 少記一個狀態。但 owner 要改成「草稿即狀態」，每一筆 stroke 都觸發下游失效，GUI 與 MCP 的語意也會分歧。
- 呼叫下一階段時隱含 finish：同一張譜內可行。但跨譜的 Filter、fit 無法判斷該發布哪些草稿。
- 保留 finish：實作沿用現有的草稿模型。呼叫次數和其他方案相同，因為證據圖本來就要在 agent 做完那一步時回。

採用保留 finish。

### 每張譜的 context 模型

- 只改用譜名定址，維持單一 context：共駕時，人點一下別張譜，agent 正在選的點就會消失。
- 每張譜各自保有 context：消除靜默關閉，也消除自動開出的 picker 被直接 cancel。

使用者選擇後者。

## 被否決的選項與理由

- **J2 只用一個「設初值加 auto_align」命令**：數據只覆蓋部分 half／integer 範圍時，auto_align 會對齊失敗。
- **在 line picker 回覆附上「線是否落在範圍內」的欄位**：agent 判斷 J2 時不需要圖以外的資訊；第一張譜沒有初值，這個欄位只會是雜訊。
- **selection 圖疊模型曲線並標出殘差大的點**：使用者暫時看不出用處。
- **選點時記錄躍遷猜測的標記**：search 用不到，而且改 search 目標函式超出這次範圍。
- **重新採樣時從已發布的點開始**：使用者的設計是 J3 盡量把可識別的線都選上，之後交給 J6 過濾，不回頭補選。
- **Export 保存 Filter 結果**：Export／Restore 是採點後的 checkpoint。
- **wire 保留完整資料選項（`detail="full"`）**：精確值讀匯出檔；測試直接呼叫 controller。
- **把設計範式寫成 ADR，或抽成共用 helper**：使用者只要建立範式，不需要 ADR 或程式碼。
- **auto_align、search 設等待上限**：使用者選擇等到完成，以簡化操作。

## 討論中使用者的判斷與修正

1. 使用者把互動 context 生命週期的細節定為「旁支末節」，改問兩個根問題：「這整個流程需要agent判斷的決策點在哪？以及gui目前的架構設計合不合理」。隨後給出 J1–J5，並補上 Filter 的作用，成為 J6。
2. 使用者修正了 J2：手勢層的命令有存在的理由，因為部分覆蓋的數據必須保留之前取得的 half／integer 位置，auto_align 反而會失敗。
3. 使用者指出 GUI 的瑕疵：「同時只能開一筆數據，mcp控制也只能控制當前數據，而不是指定名字」。
4. 「對agent來說，tool本身就是意圖，送什麼tool就切換到什麼階段」。
5. 錨定的命令由使用者設計：`move_line(..., conjugate)`、`auto_align`、`finish`。其他能力參照同樣的表達方式。
6. 使用者提議去掉 finish。比較後，使用者以實作負擔為由，選擇保留 finish 作為交卷動作。
7. 補充分層定位：MCP 面向 agent，RPC 是簡單基礎通用的控制接口，MCP 組裝 RPC。
8. 使用者確認 `r_f` 是讀取腔頻率，sideband 是 qubit 躍遷頻率加或減 `r_f`；`sample_f` 是 DAC／ADC 的取樣頻率，mirror 是對 `sample_f` 的鏡像。
9. 失效的判定：「利用流程依賴模型來決定是否失效」。
10. fit 回覆原本提議列出「沒有任何點配到的 transition」。使用者指出：「通常沒用到的transition也會有一兩個點對到，通常是與其他transition交錯的位置會這樣，我認為可以給每個transition親和的點數量」。改為每條 transition 列點數。
11. 範式的修正：「範式7不一定，重點是看agent在過程中能否發揮判斷的影響力，比如measure-gui的run就會回token，過程中可以檢測進度並early_stop。8這點還不確定，先不進範式。10先做簡單設計，再按需做加法。」以及「範式不只是fluxdep，而是全部mcp設計理念」。
12. J2 交給 agent 的理由與判讀依據：「不只要在對稱軸上，這步之所以要agent判斷而不是auto_align，其中一個原因是數據可能覆蓋不夠，另一個是agent才能判讀哪個比較像integer/half」。判讀：「判斷標準通常是看onetone，平坦對稱點通常是integer，陡峭對稱點通常是half flux，而twotone則是plasmon like transition(f01變化平坦且4~6GHz)是integer，fluxon like transition(f01變化快且~0.5GHz)或f03/f13等高能階(~3~7GHz)出現對稱反交錯圖案是half flux」。這段寫進 skill 的 J2。
13. 使用者提出：「我認為有些可以進measure-knowledge，也就是說這個知識庫可能不只有run-measure-gui能用」，並要求「不太希望這個repo與measure-knowledge耦合太嚴重」。判讀依據因此寫進 `measure_knowledge/experiences/fluxdep-fit-analysis/`，skill 只留最小判準與一句入口。

## MCP 設計範式

這次討論建立的範式，適用於為任何 GUI app 設計 agent 用的 MCP，不只 fluxdep。使用者要求寫在這篇，之後可進 ADR 作為全部 MCP 的通用設計理念。

1. **分層**：RPC 是控制接口，簡單、基礎、通用，回完整資料，出圖另有 method。MCP 面向 agent，組裝 RPC、分派階段、做摘要、決定何時出圖。MCP 不保存狀態、不驗證，GUI 是唯一的業務真相。
2. **從決策點出發**：先列出 agent 要判斷的地方與各自的證據。MCP 的範圍是完成這些判斷所需的能力清單，不是 GUI 操作的一對一。
3. **證據圖只在決策點回**：程式判斷不了的事才回圖，機械命令回數字摘要。設計者定預設，agent 可覆寫。人與 agent 看同一張圖，用同一個渲染。
4. **tool 即意圖**：沒有 open、cancel、context id。送哪類命令就進那個階段。每個階段由編輯命令加 finish 組成，finish 是交卷，回證據圖。切換顯示對象不取消別的對象的編輯。
5. **命令三原則**：沒有隱藏模式，模式是命令的參數；能用其他命令表達的就不提供；編輯命令加 finish。
6. **回覆只回判斷需要的**：不回逐點陣列，回計數、範圍、差量。`changes` 只列有變的欄位，參數與資料的變化都列。列出這次變更讓哪些下游失效。schema、範圍、領域詞彙寫在工具描述。
7. **長操作看 agent 在過程中能否發揮判斷的影響力**：有就回 token，給進度與提早停止，例如 measure-gui 的 run 可以看進度並 early_stop。沒有就等到完成，由 MCP 輪詢 RPC。fluxdep 的 search 與 auto_align 等到完成。
8. **人與 agent 共駕**：兩邊都能做每一步，同一時刻由控制權 holder 寫入。狀態、圖、失效標示兩邊一致。
9. **決策點的判讀依據放知識庫**：skill 只放流程、完成條件與每個決策點一行最小判準，加一句指向 `measure_knowledge/` 的入口；完整的判讀依據、反例與圖例放知識庫。知識庫不屬於單一前端。skill 不依賴知識庫才能運作，避免程式 repo 與知識庫耦合。
10. **先做簡單設計，再按需做加法**：第一版只做能力清單需要的最小命令集與回覆。這次的「需」來自讓 agent 用真實資料走一次並量測 token 分布。

失效與品質依流程依賴模型這一條，使用者認為還不確定，暫不進範式，只作為 fluxdep 的 app 行為。

fluxdep 在 agent 介面分層的定位：L3 跨實驗整合分析，獨立、可選的 app。控制粒度是「分析步驟由 agent 驅動，順序不固定；agent 的判斷落在宣告的決策點，程式能判斷的事不交給 agent」，取代分層 ADR draft 的「分析步驟由呼叫者驅動」。定位與範式之後修進分層 ADR draft，並把 ADR-0068 的「Fluxdep MCP 從 GUI declarations 產生 pipeline tools」改為「組裝 GUI 的 RPC method」，不開新 ADR。

## 留下的問題

- server instructions 與 `run-fluxdep-gui` skill 要依決策點重寫。
- 分層 ADR draft 與 ADR-0068 的改動等控制身份 task land 後做，列入實作 task 的交付。
- 殘差偏大的門檻初值是 3 倍平均殘差，spec 可改。
- 驗證時發現三個現行 bug：
  - Filter 遮罩依位置對應整個點雲；點雲總數不變時，會靜默套到錯的點。已交給 storage-redesign 的 gui-uiux 第四輪，改成每張譜一份。
  - transitions 裡拼錯的類別 key 會被靜默忽略，那組躍遷不進 search。修法已定，尚未排程。
  - 改對齊後匯出 params.json，會把舊的 EJ/EC/EL 和新的對齊值寫在一起。由上面的失效規則處理。
- 專案參數的格式依另一個 task 的重構而定，這裡只定原則：用專案識別名稱，不帶路徑。
- 實作要排在控制身份 task 與 gui-uiux 第四輪 land 之後，兩者都在修改 fluxdep GUI。

## 依據

- [ADR-0068 Remote／Transport](../adr/0068-remote-transport.md)
- [ADR-0075 Agent 介面分層](../adr/0075-agent-interface-layers.md)：fluxdep 位於跨實驗整合分析層。
- [fluxdep MCP README](../../lib/zcu_tools/mcp/fluxdep/README.md)
- [fluxdep 分析模組 README](../../lib/zcu_tools/analysis/fluxdep/README.md)
- `lib/zcu_tools/gui/app/fluxdep/`：互動 owner `interactive.py`、remote `remote/`、Filter 遮罩 `state.py` 與 `services/fit.py`
- `lib/zcu_tools/analysis/fluxdep/search_models.py`：躍遷類別與公式
- [run-fluxdep-gui skill](../../.agents/skills/run-fluxdep-gui/SKILL.md)
