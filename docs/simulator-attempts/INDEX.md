# 模擬器優化嘗試

**讀者：** framework-dev、agent。依主題查找模擬器的效能嘗試、採用狀態與實測範圍。

## 收錄規則

- 每次以速度或資源用量為目標的模擬器嘗試，在落地或放棄時新增一檔，並在下方選一個主題加一列。部分採用、暫停或仍在調查的嘗試也可先登記，結案時更新同一檔。單純的介面或資料格式變更不收。
- 已有討論就連到原文，不搬寫比較過程或量測表。依據只在未入庫的工作檔時，把結論與判斷所需的數字寫進正文，並說明原始工作檔未入庫。
- 索引每列放日期、嘗試檔連結、狀態、一句做法和一個代表數字。每個實測數字旁標明量測 commit、輸入與範圍，例如單輪或總和、實測或外推。資料不全就寫明缺什麼，不以實作 commit 代替量測 commit。
- 狀態記錄這次嘗試的結果，不描述目前的模組契約。若後續嘗試取代這次做法，在 `superseded_by` 填後續嘗試的相對路徑。

## 單檔模板

檔名用 `YYYY-MM-DD-slug.md`。狀態選 `adopted`、`rejected`、`partial`、`paused` 或 `investigating`。`commit` 填實作 commit；原型未入庫則填量測所用的 commit，並在正文說明原型未入庫。`discussion` 填討論的相對路徑，沒有就填 `none`；正文可直連原討論的段落。

```markdown
---
date: YYYY-MM-DD
status: investigating
commit: <commit>
modules:
  - <module path>
discussion: none
superseded_by: null
---

# <嘗試名稱>

**讀者：** framework-dev、agent。<這份紀錄的用途>。

## 核心思路

<想省掉哪一項工作，以及做法。>

## 實測

<連到原討論，或記錄數字、量測 commit、輸入、範圍與限制。>

## 為何採用或否決

<說明結果與決定，或連到原討論的理由。>

## 留下的線索

<後續可重用的做法、待驗證的範圍或不必重試的條件。>
```

## Family 與 guard

- 2026-11-16，[常數雙 tone secular guard](./2026-11-16-constant-multitone-guard-diagnostic.md)，`adopted`。補查被省略的耦合在長脈衝上的累積誤差。未找到可追溯的量測 commit，不列代表數字。
- 2026-10-18，[Two-tone Floquet](./2026-10-18-two-tone-floquet.md)，`adopted`。保留近共振 sidebands，短段另比較 direct 的成本。早期 frame builder 版為 9.214 s（`58ca8b968`，S9 五點、兩 workers，暖機後交錯三次的 acquire wall 中位數）。

## 跨點重用與快取

- 2026-10-18，[Dyson 輸入共用](./2026-10-18-dyson-input-reuse.md)，`partial`。完全相同的前綴只演化一次，後續共用收縮表。未採用的 sparse scan 只省 12.6%（`16b1dde43` 上的原型，R37 201 個 carriers、兩 workers，各版一次含 input capture 的完整 acquire wall）。
- 2026-10-18，[Shaped periods 與 gates](./2026-10-18-shaped-period-and-gates.md)，`partial`。重用 guard 的建表結果，只在窄掃頻組插值。原型觀測少 25.15%（`b0cb487f3` 上的原型，F4 窄頻 2001 點的第一組完整 acquire wall 對照，受其他程序負載干擾，不能判定加速）。

## 參考積分

- 2026-10-18，[Direct integration](./2026-10-18-direct-integration.md)，`partial`。保留自適應控制，合適的單 tone row 改用 interaction picture。原型少 31.50%（`b0cb487f3` 上的 IP 加 gain grouping 原型，F5 201 點、兩 workers 的一組完整 acquire wall 對照，非正式版）。

## 表示與合併

- 2026-10-18，[Constant-period builders](./2026-10-18-constant-period-builders.md)，`partial`。按 key 選 period builder，只沿 gain 軸續算。Carrier continuation 零命中（`b0cb487f3` 上的原型，L7strong 的 1103 次預測全數未通過原檢查）。

## 平行化與排程

- 2026-10-18，[Planner 的 evidence admission](./2026-10-18-planner-evidence-and-overhead.md)，`rejected`。按完整 acquire 評估候選路徑，未通過效能驗收。候選版的確切量測 commit 不明，不列代表數字。

## 環境與讀出

尚無條目。
