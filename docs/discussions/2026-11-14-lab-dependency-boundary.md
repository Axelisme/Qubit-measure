---
date: 2026-11-14
status: active
superseded_by: null
related:
  - ../adr/0072-module-boundaries.md
  - ../adr/0068-remote-transport.md
  - ../../.importlinter
  - ../../tools/README.md
basis:
  repo_commit: 872f4e53f765850b0b047ca8e6108d0d0c066973
  external_sources: none
---

# Lab 依賴只經 assembly 入口

## 結論

使用者於 2026-11-14 選擇保留直接 Python import，但將 GUI、MCP 組合根與開發工具對 lab 的直接依賴收窄到 assembly。Migrations 保留直接 import 實驗與元件的例外。規則、例子與適用邊界由 [ADR-0072](../adr/0072-module-boundaries.md#使用者空間的依賴原則) 擁有，執行投影是 [`.importlinter`](../../.importlinter) 的 `lab-assembly-only` contract。

`zcu_lab/` 與 `scripts/` 被定位為使用者空間。這次沒有實作刪除 lab 後仍能啟動既有 lab 組合根的完全解耦，也沒有處理 `tests/scripts/` 既有腳本 import 的債務。

## 問題

當時覆核的是量表工具的依賴決策要留在哪裡。原提案只在 [tools README](../../tools/README.md#fixed-scenarios-companions-and-reference-cache) 說明固定場景不依賴 lab。使用者改選寫進 ADR，因為問題不限於量表，而是 repo 對使用者空間的依賴方式。

使用者的原話是：「其实应该是 B， 因为我对这个 repo 的期望是，我把 [zcu_lab] 删掉，也可以正常运作。而 [zcu_lab] 是使用者自由的、控制的、写的、安排结构的模块。」引文中的 `[zcu_lab]` 代替語音輸入的 lab 名稱。

使用者接著區分 Script 與 Tool，並指出 MCP 仍需要正式取得實驗宣告。完全不依賴 lab 與避免複雜註冊機制之間，需要一個可執行的邊界。

## 考慮過的選項與比較

讓 MCP 與入口取得 lab 宣告時，當時比較了三個方向：

| 方向 | 代價與取捨 |
| --- | --- |
| 保留直接 import，只依賴 `zcu_lab.assembly` | 沿用現有 `lab_definitions()` 組裝入口，不新增註冊機制。入口仍依賴選定的 lab，assembly 後方的佈局由 lab 安排。採用。 |
| 在腳本或設定檔填 `zcu_lab.assembly:lab_definitions` 位置字串 | 把模組名稱從 import 搬進設定。刪除 lab 後仍需修改設定，沒有消除入口對特定 lab 的選擇。未採用。 |
| 以 Python entry point 註冊 lab | 需要額外的註冊與發現邏輯。使用者不願為此增加實作行數，未採用。 |

使用者說：「但是呢，我又不想把注册逻辑搞得太复杂，导致实作行数太多。有简单的方法吗？还是我就放弃？我就是让它可以依赖，但是只依赖局部的部分的档案。」因此最後選擇了 assembly，而不是完全解耦。

Migrations 另有甲乙兩案。甲允許直接 import 實驗與元件，因為工具轉換的就是這些實驗保存的舊格式。乙同樣只准經 assembly，必須先擴充 `LabDefinitions`，再改遷移端的實驗 import。使用者對整套邊界與遷移例外回答：「好，甲」。

量表則保留 2026-11-12 的獨立取捨。使用者當時說：「我其實不樂意看到tool依賴zcu_lab，我認為你要測試，那就只複製最核心的pulse序列來模擬啊，又不需要整個experiment」。對工具內建場景的方案，使用者回答：「你這個工具的目的是對優化器做benchmark吧？那場景自然是挑選過的」，並補充：「而且這樣tool也能提高可讀性，我寧可重複也要提高可讀性」。這是固定 benchmark 場景的選擇，不是要求所有開發工具複製 lab 宣告。

### 先例與立場

| 先例 | 當時的立場 |
| --- | --- |
| ADR-0072 的 framework 與 tests 不依賴 lab | 遵循。將使用者空間定位延伸到 scripts，既有 `tests/scripts/` 依賴列為待移除債務。 |
| ADR-0072 的 `lab_definitions()` 唯一組裝入口 | 遵循。將既有約定投影成 import-linter contract，並涵蓋開發工具。 |
| [ADR-0068](../adr/0068-remote-transport.md#分層與連線) 允許 apps 與 mcp_servers 依賴 lab | 遵循。使用者同意將直接依賴收窄到 assembly，沒有取消組合根對 lab 的注入責任。 |
| ADR-0072 將離線遷移定位在 migrations | 遵循。遷移工具必須認得來源格式，因此保留直接 import 例外。 |
| 2026-11-12 的固定量表場景決策 | 遵循。將原本只在 tools README 的規則升入 ADR，沒有改成追蹤真實實驗。 |
| .importlinter 檔頭允許收緊 contract、不准擴大 ignore_imports | 遵循。新增 contract，不改既有例外。 |

## 被否決的選項與理由

只把量表的決策留在 tools README，無法表達使用者對整個 repo 的期望，因此改由 ADR 擁有規則。
位置字串與 entry point 兩案沒有採用，原因分別是只移動依賴名稱，以及增加註冊邏輯。
Migrations 乙案沒有採用，因為遷移工具本來就處理特定實驗的格式，為它擴充 assembly 沒有消除這項知識。

## 留下的問題

當時將 `tests/scripts/` 既有腳本 `main` import 登記為債務，沒有在這次設定與文件修改中清除。
未來若要保證刪除 lab 後所有入口仍可啟動，需要另行決定 lab 選擇與發現機制，這次沒有實作。

## 依據

當時調查確認，非遷移的 GUI、MCP 入口與需要 lab 的品質工具已只直接 import assembly。因此採用的規則只需新增 contract，不需修改產品程式。相關邊界由 [ADR-0072](../adr/0072-module-boundaries.md#lab-定義與組合根)、[ADR-0068](../adr/0068-remote-transport.md#分層與連線) 與 [import-linter 設定](../../.importlinter) 擁有。量表場景的來源與更新方式由 [tools README](../../tools/README.md#fixed-scenarios-companions-and-reference-cache) 擁有。

本文根據 2026-11-12 與 2026-11-14 的使用者裁決，以及當時的設計調查整理。原始工作檔未入庫；選項、理由、原話與先例立場已寫入本文，不依賴原始工作檔。
