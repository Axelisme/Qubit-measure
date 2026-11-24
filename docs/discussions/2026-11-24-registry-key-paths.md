---
date: 2026-11-24
status: active
superseded_by:
related:
  - ../adr/0072-module-boundaries.md
  - ../adr/0073-measure-form-derivation.md
  - ../../zcu_lab/v2/measure-authoring.md
  - ../../tools/README.md
basis:
  repo_commit: 88e835085
  external_sources: none
---

# 實驗選單 key 跟隨定義目錄

## 結論

使用者採用「目錄為準」：實驗 registry key 等於定義相對 `zcu_lab/v2/` 的目錄，同目錄有多個入口時，各加一段變體名。分組目錄也進 New Tab 選單。改 key 的既有已存 tab 不再用舊名稱開啟，不加相容 key。MCP recipe 只改內部 adapter 字串，工具名不變。Native tag 是存檔身分，本次不改。

## 問題

New Tab 把 registry key 按斜線切成選單。時域實驗放在 `twotone/time_domain/`，卻與其他 two-tone 入口擠在同一層。維護者從選單名稱也找不到實際目錄。使用者指出：「我认为表单的结构应该要跟实际结构对应，或者说一致，这样才好维护啊。」

## 考慮過的選項與比較

時域入口改用 `twotone/time_domain/` 下的完整路徑，讓選單呈現原有分組。兩個名稱都可接受時，保留操作員已用的名稱並改目錄，例如 `twotone/flux_dep`。目錄已說得更清楚時才改 key，例如把 `twotone/reset/check` 改為 `twotone/reset/rabi_check`。使用者答「採用建議吧，對於3，新名字你要叫什麼？」，對新名稱答「可以」。

AC Stark spectroscopy 與 Ramsey 原在同一目錄，但各有自己的 core 和前端接線。Designer 依 [ADR-0072](../adr/0072-module-boundaries.md#實驗計算與-gui-接線) 與 [ADR-0073](../adr/0073-measure-form-derivation.md) 代裁保留 spectroscopy 原目錄，只把 Ramsey 搬到同名平級葉。兩個 key、cfg schema 和 Native tag 不變。兩者不像 ZigZag 的兩個入口共用 core，不需要為了套用變體規則而留在一起。

檢查採用 [Measure MCP 文字預算](../../tools/README.md#measure-mcp-text-budget) 的做法：比較 gate 每次透過 lab assembly 讀實際宣告，pytest 只用合成輸入測規則。這遵循 [ADR-0072](../adr/0072-module-boundaries.md#使用者空間的依賴原則) 的組裝邊界。檢查方法由 designer 定案，不需開啟 GUI 或量測。

## 被否決的選項與理由

一律搬目錄來配合舊 key，會保留原本擠在一起的選單，也要一起搬對應測試。豁免分組目錄則會留下結構不一致的例外。個別改目錄仍可保留合適的公開名稱。

AC Stark 若共用目錄但保留平級 key，Ramsey 得有例外。改成同一目錄下的 `/spectroscopy` 與 `/ramsey`，或新建 AC Stark 分組再各自成葉，都會改動既有公開入口。整個目錄改名則會讓原本已對齊的 spectroscopy 失配。

只提供手動檢查，漏跑就無法發現新 key 偏離。讓預設 pytest 載入真實 catalog 則違反使用者 2026-10-12 的裁決：「如果我刪除zcu_lab測試會報錯，那測試就是寫錯了。」

## 留下的問題

AC Stark 的拆檔方式是 designer 代裁，使用者尚未覆核。公開命名規則與新名稱已有使用者答覆，該答覆沒有指定拆檔方式。

## 依據

命名與 AC Stark 的討論均在 2026-11-24。AC Stark 的靜態比較以本 repo `88e835085` 為基準。原始工作檔未入庫。
