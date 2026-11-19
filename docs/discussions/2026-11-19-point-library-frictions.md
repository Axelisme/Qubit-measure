---
date: 2026-11-19
status: active
superseded_by: null
related:
  - ../adr/0075-agent-interface-layers.md
  - ../adr/0037-measure-gui-value-lookup-resolve-once.md
  - ../adr/0065-cfg-editing.md
  - ../adr/0081-point-data-disk-sync.md
  - ../adr/0082-component-flux-frame.md
basis:
  repo:
    main: 2fe51258d42e79cfaf985dff3cf6aae2530032b2
---

# Point 與 library 資料流的摩擦

## 結論

2026-11-19，使用者對一次 FTF 模擬量測中的四項 point／library 摩擦作了裁決：

- Ramsey 頻率寫回不改；
- 回滾不做；
- reload 後的過時 tab 不加標記；
- crosstalk 聯合求解只做成 agent 腳本可呼叫的純函式。

使用者並給出一條取捨原則：便利性的小功能，要等同一個問題多次出現才加。原則寫在 [ADR-0075](../adr/0075-agent-interface-layers.md#先做簡單設計再按需加法)。

## 問題

Agent 經 measure-gui MCP 在 FTF 模擬器上做 CZ 實驗，遇到四項摩擦：

1. Ramsey 寫回只更新 `$qubit.freq`，library 裡 pulse 的 `freq` 不跟著改。Agent 手動改了 library 的 freq。
2. 回到先前設定時，要手動改 library 與 parameters 兩份檔案，再各自帶 revision reload。手上的 submission receipt 不能直接用來還原。
3. 改 point 檔並 reload 之後，已開的 tab 仍沿用舊 capture，表單上看不出過時。
4. 換工作點時，coupler 對 F1、F2 有 crosstalk，三條電流要一起重算，agent 手動求解時漏算過一次。

## 考慮過的選項與比較

1. Ramsey 頻率：
   - A，寫回多列這個 qubit 頻率相同的 drive pulse，各自可選；
   - B，library 欄位持久參照 scalar；
   - C，不改，文件寫明手動同步。
   - Agent 建議 A，理由是 `onetone/freq` 已同時寫 resonator 與 readout module 的頻率。
2. 回滾：
   - A，文件寫明改檔前複製三份 YAML 作為 checkpoint；
   - B，手動建立的具名 snapshot 加明確 restore；
   - C，以 receipt 為 key 保存提交後的完整狀態並提供 restore。
   - Agent 建議 A。
3. 過時 tab：
   - A，唯讀的「來源已變更」標記；
   - B，收到事件自動 Refresh；
   - C，只寫文件。
   - Agent 建議 A。
4. Crosstalk：
   - A，`analysis.flux` 的純計算聯合求解函式加唯讀 RPC；
   - B，A 再加一次設定多個 device 的命令；
   - C，只寫手動步驟。
   - Agent 建議 A。

## 裁決與理由

1. 使用者選 C：「7aC，改了頻率則gain/length也需要聯動改，只改freq只會得到半成品。」兩份 Ramsey guide 已寫明改頻後要用 Rabi 或 zigzag 重新校準 pulse 的 gain 與 length，所以不改文件。Agent 當時只改 library 的 freq，正是使用者所說的半成品。
2. 使用者問：「7b我不懂為什麼不是直接改，而是要複製再改，有什麼顧慮嗎？」Agent 說明直接改檔沒有顧慮，複製只是為了回滾時有原狀可用；point 檔不在 git 中，receipt 只存在 GUI 記憶體。使用者回覆「可以」，接受直接改回、不做任何改動。這也維持 [ADR-0065](../adr/0065-cfg-editing.md) 不做 reverse undo 的裁決。
3. 使用者：「我遇到很多這種小功能，單個看沒什麼問題，但累計起來維護很困難，我沒有多次遇到問題不太想加，讓程式儘可能簡單好維護。」不加標記。[ADR-0065](../adr/0065-cfg-editing.md) 與 [ADR-0081](../adr/0081-point-data-disk-sync.md) 已寫明 reload 不刷新既有 tab。
4. 使用者：「7dA，作為工具可以，讓agent自己寫腳本調用是嗎沒問題的，gui內建機制暫時沒考慮。」只做純函式，不做 RPC 與 GUI 機制。

## 被否決的選項與理由

- 持久參照 scalar（1B）：推翻 [ADR-0037](../adr/0037-measure-gui-value-lookup-resolve-once.md) 的 resolve-once 規則。
- 以 receipt 保存完整狀態並提供 restore（2C）：推翻不做 reverse undo 的裁決。
- 自動 Refresh（3B）：違反 ADR-0065 的固定來源規則。
- 多 device 設定命令（4B）：硬體設定不能原子回滾，而且使用者暫不考慮 GUI 內建機制。

## 留下的問題

- 聯合求解函式的使用方式寫在 measure MCP README，量測 agent 是否會在換工作點時想到它，要看之後的任務。
- 小功能原則中「多次」的次數沒有定數，由使用者在個案判斷。

## 依據

- 本 repo main 2fe51258d：
  - `zcu_lab/v2/twotone/time_domain/t2ramsey/guide.md` 與 `zcu_lab/v2/mux/t2ramsey/guide.md` 的寫回段；
  - `lib/zcu_tools/mcp/measure/README.md` 的 point 檔案讀寫段；
  - `lib/zcu_tools/analysis/flux.py`。
- 調查工作檔未入庫；本文已寫入裁決所需的事實。
