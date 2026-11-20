---
date: 2026-11-20
status: active
superseded_by: null
related:
  - 2026-11-20-logical-cfg-paths-and-overrides.md
basis:
  repo:
    main: 4b13489c5
---

# Two-qubit MCP recipes 的首版範圍與輸入

## 結論

2026-11-20，使用者選了七個 two-qubit 實驗。每個實驗各有一個 recipe，每次呼叫做一輪，下一輪由 agent 依結果決定。首版沿用 `two_qubit_` 命名前綴。開 tab 的欄位跟 `overrides` 一樣，由 recipe 自己宣告。這項輸入設計記在 [ADR-0075](../adr/0075-agent-interface-layers.md#agent-介入點)。

使用者原話：「1B 2A 3A,目前共通RecipeInputs到底有什麼？ 4A」（2026-11-20）。追問共通基底的內容後，同日補充：「那reused也從基底移除好了」。

## 問題

11-19 的 FTF CZ 量測以 tab RPC 手動串接七個 two-qubit adapters，共開 11 個 tab、送出 42 次 Run、40 次欄位編輯及 30 次 Save。光 `cz_amplitude` 就在同一張 tab Run 了 17 次。Agent 看上一輪結果才決定下一次怎麼改頻率、增益或脈衝長度。Recipe 要減少重複操作，但當時沒有固定的跨實驗轉移條件可交給腳本。

共通 `RecipeInputs` 的 `target` 只有一個字串。Two-qubit Reset 需要在同一張 tab 選 `f1`、`f2`、`coupler`、`mux` 四個角色。原本就是 recipe 主體把 alias 對照表傳給 `open_tab`，框架不直接讀 `inputs.target`。

## 考慮過的選項

### 1. 第一版包哪些實驗？

- A：先做 `cz_amplitude`、`rb`、`local_z`、`conditional_phase` 四個。
- B，採用：在 A 上加 `rb_heralded`、`conditional_phase_heralded`、`constructive_phase`。下一次 CZ 重做要實測雙線相對相位，以 heralded IRB 的 20 個 seeds 比對單線和雙線，並用 heralded 版量 conditional phase。
- C：涵蓋全部 12 個 two-qubit keys，再加 `chevron`、`conditional_spectroscopy`、`idle_zz`、`joint_readout`、`cz_optimize`。

### 2. 一個工具做一步，還是串成整條流程？

- A，採用：一個 recipe 對一個實驗。每次呼叫做 Run、Analyze、有提案才問 writeback，然後 Save。Agent 再呼叫時可用 `reuse_tab_id` 重用同 adapter 的 idle tab。當時的 43 個 recipes 也各對應一個實驗。
- B：另做一個 recipe 串 conditional phase、CZ amplitude、local Z、RB，由 recipe 判斷下一步。

### 3. 四個角色怎麼填？

- A，採用：沿用 [overrides 的先例](2026-11-20-logical-cfg-paths-and-overrides.md)，讓 recipe 宣告 `target`。單目標維持字串；two-qubit 用 `target: {"f1": "Q3", "f2": "Q5", "coupler": "C0", "mux": "MUX2"}`。省略或填 null 的角色沿用全域 alias。既有 43 個 recipe 加欄位宣告，輸入 schema 不變。
- B：只在 two-qubit recipe 另加 `roles` 物件。
- C：不提供欄位，只用全域 alias。

使用者選 A 時問共通基底還有哪些內容。當時基底含 `target`、`reuse_tab_id` 及三條驗證與省略規則。回覆原建議先保留 `reuse_tab_id`，理由是開 tab 時傳法相同。使用者要求一併移出。重複呼叫需要重用 tab 的 recipe 仍可自行宣告 `reuse_tab_id`。

### 4. 現在就命名，還是等命名批次 N？

- A，採用：沿用現有慣例，以 `two_qubit_` 加上各 key 的末段命名，例如 `two_qubit_cz_amplitude`、`two_qubit_rb_heralded`。前綴避免將來與單 qubit 的 `rb` 撞名。保存以 tag 辨認實驗，工具改名只需改宣告與文件。
- B：先定案命名批次中 two-qubit 的名稱，再做 recipe。
- C：等命名批次排進來再做。

## 被否決的選項與理由

- 範圍 A 少了下一次 CZ 重做會用的三個實驗。範圍 C 多出的五個入口未在 11-19 使用，也不在下一次重做計畫內。
- 流程 B 缺少可預先寫定的轉移條件。當時每輪的調整都取決於上一輪結果。
- 角色 B 會讓 two-qubit schema 留下一個不能用的 `target`。角色 C 換一組目標就得先改全域 alias。`reuse_tab_id` 也不用留在基底，因為開 tab 控制與 `overrides` 可採同樣的欄位宣告方式。
- 命名 B 會把尚未排程的命名工作拉進首版。命名 C 讓下一次 CZ 重做等命名排程。

## 留下的問題

首版沒有定義跨實驗的自動決策。命名批次處理 two-qubit 入口時，再一起核對 recipe 名稱。

## 依據

- [Two-qubit 實驗清單](../../zcu_lab/v2/README.md#two-qubit-實驗清單)列出入口與各自的 guide。
- [Recipe 註冊清單](../../zcu_lab/recipes/__init__.py)是當時 43 個 recipes 的依據，基準為 `4b13489c5`。
- 11-19 的數字按 tab RPC 次數統計。原始量測調查未入庫；上述次數不等於有效量測或成功存檔的筆數。
