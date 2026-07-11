---
status: accepted
---

# 0047 — Typed expected-error taxonomy

**狀態：** accepted。關聯：[[0005]] 的 application/domain ownership、[[0014]] 的 app-local dispatch policy。

## 背景

GUI 的 Qt view 與 remote driving adapter 都需要辨識 caller 可修正的失敗。只靠
`RuntimeError` 型別或 `reason_code` 字首分類，會把 programmer error、provider failure、
persistence failure 與 session invariant failure誤當成可重試的輸入或狀態問題，也讓
transport adapter持有本應由producer擁有的分類知識。

## 決策

`zcu_tools.gui.expected_error` 定義 remote-independent 的 nominal opt-in seam：

- `ExpectedErrorCategory.INVALID_INPUT` 表示 caller 必須修改輸入才可重試。
- `ExpectedErrorCategory.FAILED_PRECONDITION` 表示 caller 必須先修改 session、resource 或
  environment state 才可重試。
- `ExpectedError` 是顯式 marker/base，提供 `category` 與 optional stable `reason_code`。
- `InvalidInputError` / `FailedPreconditionError` 是保留 `RuntimeError` ancestry 的通用 leaves。

category 是 domain/application semantics，不是 wire `ErrorCode` alias。producer 在 fixed
exception class 或 raise site 明列 category；message 供人閱讀，reason 是 optional stable
machine tag。既有具名 exceptions 保留原有 `RuntimeError` 或 `ValueError` ancestry、constructor
shape、message 與 args。

Opt-in 只發生在已證實為 caller-correctable 的 concrete exception。混合 expected 與 unexpected
children 的 parent 不 opt in；例如 `ValueLookupError` 不屬 taxonomy，只有 `MissingValue`、
`ValueTypeError` 與 `UnavailableValue` 各自 opt in，`ProviderError` 保持 unexpected。SoC/device
async terminal、persistence、I/O、framework invariant 與 programmer failures同樣排除。

目前 measure app 的 remote handlers仍各自把 typed category投影到既有
`INVALID_PARAMS` / `PRECONDITION_FAILED` wire code；handler-local structured mapping（例如
`ArbWaveformError.data`）維持原 owner。`ResultScopeError` handler直接讀 category，不從
reason字串推導。wire envelope、code、reason、data與版本皆不改變。

## 拒絕的替代方案

- **把所有 `RuntimeError` 視為 expected：** programmer error會被降級成可修正失敗並失去正確
  traceback語意。
- **Structural Protocol：** 任何碰巧帶同名 attributes 的 exception 都可能被誤分類，違反
  nominal opt-in。
- **Transport-side type registry：** 新增 domain exception時必須同步 producer與adapter，分類
  locality錯位且容易漂移。
- **讓 lower owner import remote code：** 會讓 domain/service層反向依賴 driving adapter與 wire
  vocabulary，違反 [[0005]]、[[0014]]。

## 後果

- caller-correctable semantics由producer擁有，Qt view與remote adapter可共享同一分類。
- ordinary exceptions不會因 ancestry或 attribute shape自動成為 expected failure。
- closed taxonomy刻意只有兩類；controller/internal/provider/persistence等 unexpected failures不以
  第三個 category包進 shared carrier。
