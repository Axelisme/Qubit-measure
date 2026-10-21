---
date: 2026-10-21
status: active
superseded_by: null
related:
  - docs/adr/0075-agent-interface-layers.md
  - docs/adr/0068-remote-transport.md
  - docs/adr/0072-module-boundaries.md
  - docs/adr/0073-measure-form-derivation.md
  - docs/discussions/2026-10-16-recipe-workflow-yield-vs-async.md
  - docs/discussions/2026-10-21-three-role-protocol.md
  - zcu_lab/recipes/README.md
  - lib/zcu_tools/mcp/measure/README.md
basis:
  - "repo@2a8a67b71：zcu_lab/recipes 22 個 recipe，RecipeDefinition 手寫 input_schema，mux 系列以 library／inline／overrides 三清單輸入"
  - "設計稿與 prototype（schema 投影、t1／mux_t1／singleshot_ge 改寫版、22 recipe 統計）為研究工作檔，未入庫；結論與參考實作寫在本文"
---

# Recipe 接口重設計：一份型別宣告、兩層執行、作者定義輸入與回覆

## 結論

Recipe 的執行分兩層。驅動引擎只懂 generator 協定（`Wait`、`Ask`、return 值），不知道 GUI 方法、實驗或 Form；便利層是 measure GUI RPC 的 typed 包裝（Form 的 `read`／`set`／`select`、泛用 `call`，以及 `yield from` 糖 `run()`、`analyze(stage)`、`save_raw()`、`ask_writeback()`）。實驗語義的 helper（`choose_frequency`、`require_pi`）留在 `zcu_lab/recipes/helpers.py`。使用者：「recipe執行要分兩層，一層是驅動引擎，與實驗語義解耦，只負責處理yield的流程。另外一層利用前者，提供run/analysis/ask_writeback等便利語法」。

輸入是作者定義的 pydantic `FormModel`，欄位用原生型別（`str`、`int`、`float`、`Literal`），不設框架基底；框架從 model 生成 MCP schema、驗證與欄位說明。可選欄位採兩態：`T | None`，None 表示保留 GUI 現值，不支援經 recipe 清空欄位。宣告用 `@recipe(name, description)` 裝飾函式，`RECIPES` 仍是顯式清單；沒有 adapter 參數與 summary 宣告。回覆是引擎的 `RecipeOutcome(status, report, runs)`，status 由引擎依 generator 結束方式決定，report 是作者 `return` 的小 model。`ask_writeback()` 只問，寫回另以 `tab.accept_writeback(items)` 明確呼叫。

Recipe 是操作 GUI tab 的腳本，可開多個 tab、串多個實驗；「參數少、單次實驗」不再是定位。使用者：「recipe是一組操作gui的定位，其實不只能跑一次實驗，是可以做到類似workflow的行為的」。

## 問題

22 個 recipe 共 3,952 行，每檔約四成是手寫 JSON schema 與 summary 宣告，同一組參數在簽名、docstring、schema、summary 寫四次；`t1` 因此要以 noqa 容忍十參數。單 qubit 系列只曝露作者選的欄位，agent 要多改一個 scalar 就得降到 L1；mux 系列只有 path 清單，agent 必須先知道 Form path，等於 L1 套 L2 的殼。`RecipeTab` 帶 `use_library`、`set_sweep` 等名稱，prototype 一度加入 `require_calibrations`，把實驗詞彙放進 tab。使用者問：「目前recipe定義的代碼片段為何？足夠簡潔明確嗎？足夠靈活嗎？」

## 考慮過的選項與比較

| 題目 | 選項 | 採用與理由 |
| --- | --- | --- |
| schema 來源 | 手寫 JSON schema；dataclass 另闢路徑；pydantic FormModel 投影 | FormModel。workflow 的 plan／tunables 已用 FormModel 與 `Annotated`，三種腳本的輸入宣告只剩一種寫法，GUI Form 或 MCP schema 只是投影 |
| 欄位型別 | `NonBlank`、`PositiveInt` 等 Annotated 別名；原生型別 | 原生型別。Form 自己驗證 ref 與 expts，recipe 再驗是重做別的模組的工作；只有 recipe 自有語義的欄位加約束 |
| 逃生口 | 框架基底帶 `overrides`；作者自定 | 作者自定。使用者：「recipe的輸入欄位因與內部使用的欄位解耦，欄位本身由recipe作者定義意義……overrides不該出現在base class中，超過負責的範疇了」。`FieldEdit` 與 `tab.apply` 由便利層提供但不強制 |
| 可選欄位 | 三態 `Given[T]`（Rust Option 式，MISSING／None／值）；`model_fields_set` 加 `set_given`；兩態 None 保留 | 兩態。清空欄位不是真需求，清空走 L1；`Given[T]` 設計保留，日後需要可 additive 引入。使用者：「好，先用兩態」 |
| overrides 的 value | untagged `{"expr"}`／`{"library"}`／`{"inline"}`；tagged envelope | untagged。tagged 只在 GUI 要畫 recipe inputs 時才需要 |
| 宣告 | `DEFINITION = RecipeDefinition(...)`；`@recipe` 裝飾器 | 裝飾器，與 `@workflow` 同形。使用者：「DEFINITION還需要嗎？還是類似workflow的decorator就好？」 |
| adapter 參數 | 保留供 `recipe_guide` 查 guide；刪除 | 刪除。它只服務 guide 查詢，且多 tab recipe 一個欄位裝不下；guide 改以 adapter 名直接查 |
| 回覆內容 | `summary_parameters`／`summary_estimates` 宣告；作者 return 的 report | report。使用者：「將status從Report中移出，回傳的是Report的包裝，包裝附帶status」 |
| tab 的認知 | 保留 library／sweep／calibration 名稱；只剩 Form 的 typed path 操作 | 後者。使用者：「Form已經去認知化了，tab還留著不太適合」；用詞用 `ref`：「library不是Form的語義吧？ref才是」 |
| 收尾序列 | 維持顯式 20 行；`analyze_steps`／`ask_writeback` 合併相鄰步驟；吃掉整段流程的 helper；取消改例外 | 合併相鄰步驟（4b）。整段 helper 隱藏取消點，例外推翻 2026-10-16 的具名 status 裁決 |

## 被否決的選項與理由

- 框架基底 `RecipeInputsBase(target, reuse_tab_id, overrides)`：欄位意義應由作者定義；`open_tab` 也不接受 inputs，作者自己傳 `aliases={"qubit": inputs.target}`。
- `Given[T]`／MISSING 三態：解的是不存在的清空需求，且 `Missing` 會漏進所有 helper 的型別。
- `set_given` 與 `given()` helper：使用者認為不直觀，傾向明確一行一欄位。
- `NonBlank` 等別名：重做 Form 的驗證。
- `run_step`／`analyze_step` 命名：`_step` 後綴只重複 `yield from` 已說的事。

## 參考實作

```python
class T1Inputs(FormModel):
    target: str | None = Field(None, description="Qubit alias; null uses GUI focus.")
    reuse_tab_id: str | None = Field(None, description="Idle twotone/t1 tab to reset.")
    readout_ref: str | None = Field(None, description="Readout library key.")
    reset_ref: str | Literal["off"] = Field("off", description="Reset library key, or off.")
    pi_ref: str | None = Field(None, description="Calibrated pi-pulse key.")
    max_delay_us: float | None = Field(None, description="Delay sweep stop in us.")
    points: int | None = Field(None, description="Delay sample count.")
    reps: int | None = None
    rounds: int | None = None

class T1Report(FormModel):
    t1_us: float | None = None
    t1_err_us: float | None = None
    writeback_applied: bool = False

@recipe("t1", description="Measure T1 from excited-state decay; save raw, fit, ask before writeback.")
def t1(session: RecipeSession, inputs: T1Inputs) -> RecipeGenerator[T1Report]:
    tab = session.open_tab("twotone/t1", aliases={"qubit": inputs.target}, reuse=inputs.reuse_tab_id)
    select_or_disable(tab, ("modules", "reset"), inputs.reset_ref)
    tab.select(("modules", "readout"), ref=inputs.readout_ref)
    choose_readout_frequency(tab, calibration="$resonator.freq")
    require_pi(tab, inputs.pi_ref)
    tab.set(("sweep", "length", "stop"), inputs.max_delay_us)
    tab.set(("sweep", "length", "expts"), inputs.points)
    tab.set(("reps",), inputs.reps)
    tab.set(("rounds",), inputs.rounds)
    result = yield from tab.run()
    if result.cancelled:
        return None
    tab.save_raw()
    fit = yield from tab.analyze("primary")
    if fit.cancelled:
        return None
    answer = yield from tab.ask_writeback()
    if answer.accepted:
        tab.accept_writeback(answer.items)
    return T1Report(t1_us=fit.result.t1, t1_err_us=fit.result.t1_err, writeback_applied=answer.accepted)
```

約 60 行，對照現行 179 行。agent 只看到作者 `Ask` 的問題與 `return` 的 report；`ask_writeback()` 是 `proposal = tab.writeback_proposal()` 加 `yield from session.ask(WritebackQuestion(proposal))` 的糖，作者可在中間過濾、依條件直接 `accept_writeback` 不問，或自組 question。使用者：「這種糖的定位剛剛好，底層Ask/Wait也是一看就懂」。

## 留下的問題

- MCP 回覆的分層切法（預設回 status、report、runs，細節用 `status(execution, detail="full")`）等 mux 摩擦測試的清單後定。
- 七個 mux recipe 的具名參數與推導清單，等同一份摩擦清單。
- 是否提供寫入非 proposal 值的 `session.context.write(...)`；第一版不提供。
- 多 tab recipe 的回覆以有序 runs 列出 tab、adapter、run_id；`status` 報目前在第幾個 tab。

## 依據

- 現行 recipe 的行數、手寫 schema 行數與收尾行數統計來自固定 commit 的研究工作檔，未入庫；數字：22 個 recipe 3,952 行，收尾序列 16 個為 run→save→primary→ask→accept、2 個多 post 段、4 個不問寫回、1 個只 preview；取消處理 22 個全部是 `return`。
- 兩層架構與三原語的推導見 [三角色協定](2026-10-21-three-role-protocol.md)。
- ADR-0075 的 L2 定義與 recipe 契約段依本文更新，待實作 task 落檔。
