---
date: 2026-11-16
status: active
superseded_by: null
related:
  - ../adr/0008-cfg-editor-session.md
  - ../adr/0063-persistence-ownership.md
  - ../adr/0072-module-boundaries.md
  - ../adr/0075-agent-interface-layers.md
  - ../../zcu_lab/v2/onetone/range_find/guide.md
basis:
  repo: 391aa362fcd9e8b9c5c4280ddadef9981df0a55b
  external: none
---

# range_find 的 writeback：沒有就建立，有就覆寫

## 結論

使用者裁定「沒有就建立，有就複寫」。`onetone/range_find` 對每個有名字的 dip 提出 `<name>.freq`。
Apply 時，元件存在就覆寫 `freq`；元件不存在就在同一次 parameter edit 裡新增該名稱的 resonator，
只填 `freq`，其餘欄位用 registered model 的預設值。框架只多一個選填欄位 `create_kind`，kind 由
adapter 給。現行規則由 [ADR-0008〈Parameter upsert〉](../adr/0008-cfg-editor-session.md#parameter-upsert) 擁有。

## 問題

range_find 依 Resonators 名單幫 dip 命名後，原本只對名稱已在本次 Run／Load 參數 capture 裡的元件產生列。
名稱不在 capture 時產生 0 列，Writeback 區塊隱藏。缺名診斷只在存檔後 Results browser 的 Analysis 頁看得到，
即時的分析畫面與 Writeback 區塊看不到。

新裝置第一次跑 range_find 時，使用者要先在 Component tree 新增 resonator，再到 Inspect 抄頻率。
Add 對話框只收必填欄位，所以新增時填不了 `freq`。Analyze 使用 Run 當時凍結的參數，新增元件後要重新 Run
才拿得到該元件的列。使用者：「應該考慮沒有 component 就建議新的 resonator 的行為」。

## 考慮過的選項與比較

- **A：Writeback 多一種「新增元件」列。** 列預設不勾，寫進 capture 的來源文件。審查指出 ADR-0008 規定
  Apply 寫入當下的作用中文件、不取歷史來源，Load 的結果也沒有持久的來源 target。修正成 **A′**：
  寫入 Apply 當下的作用中文件，作用中文件已有同名元件時整批拒絕。
- **B：不加列，顯示原因與步驟。** 對每個缺名在 Writeback 區塊顯示原因與新增步驟，需要新的
  adapter→draft 原因傳遞。變體 **B′** 只在 range_find 自己的分析圖上標出缺名與頻率。
- **C：預填的「新增」按鈕。** 按鈕開既有的 Add 對話框，預填名稱、kind 與 `freq`，按下立即寫入作用中文件。
  這條路只在 GUI 有，MCP 沒有入口。
- **D（使用者提出）：沒有就建立，有就覆寫。** 參數列原本就在同一個 edit draft 裡逐筆設值、整批提交，
  同一個 draft 也能新增 registered 元件。所以 D 只要讓參數列帶一個選填 kind，不需要新的列種類或入口。
  Designer 讀碼推估 3–5 個模組、40–80 行產品碼，未量測。

先例立場：

- ADR-0008：遵循。目的地是 Apply 當下的作用中文件，不需要 capture 的來源 target。
- ADR-0063：遵循。Kind 與欄位由 lab 的 registered model 擁有；框架不寫死 resonator。
- ADR-0075：遵循。列只是候選，Apply 或 accept 才寫入。
- ADR-0072〈Mux onetone〉：區別。那條處理 tone 對應到哪個既有 resonator，對不上時不能猜，要省略並顯示原因。
  range_find 的名稱是使用者輸入的字面目標，沒有別名解析；建立元件由使用者在本次裁決明確要求。

## 被否決的選項與理由

- A／A′：獨立的列種類、預設不勾與同名拒絕都是防禦性設計，不是需求。使用者：「為什麼建議不簡單的做法？
  沒有就建立，有就複寫」。
- B／B′：使用者仍要自己新增元件、自己抄頻率，新裝置第一次跑 range_find 時每次都要做。
- C：只在 GUI 可用；寫入不經 Apply，不出現在 writeback receipt 裡。

## 留下的問題

- Resonators 名單打錯字（例如 `R3`）時，Apply 會建出多餘元件。列上顯示 `R3.freq`，Apply 前可以取消勾選；
  這是已接受的取捨。
- 其他實驗要建立元件時，由各自的 adapter 明確給 `create_kind`；`onetone/freq` 與 mux adapters 維持 None。
- `commit_parameter_updates` 的參數型別寫成 `PointView | SetupView`。使用者要求它保持中立，不為此擴大
  `add_component_to_draft` 的參數型別，所以改由 `EditView.add_component` 提供新增元件。Setup 與 point
  兩種 View 是否收成一種，另案討論。

## 依據

使用者 2026-11-16 提出需求，否決 designer 推薦的 A／A′，提出 D，並對具體做法答「可以，照這樣定」。
實作落地為 main `beafea3bb`。當時的 dossier、角度報告與審查未入庫；本文保留選項、先例立場與採用理由，
不依賴那些工作檔。
