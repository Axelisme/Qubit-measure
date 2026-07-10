---
status: accepted
---

# Finished-cfg validation — adapter 必回傳完整合法 value 樹，靜態檢查在成品邊界

## 脈絡

ADR-0010 立了「value 樹永遠完整、None 表空」，但**沒有任何地方強制** adapter 的
`make_default_value(self, ctx)`（BaseAdapter 抽象方法，~20 個各寫）真的回傳完整 value 樹。
lookback 在 `modules` 子樹省略 `init_pulse`/`reset`、頂層省略 `reps`（LiteralSpec），產出
結構不完整的 value——上一個 phase 用 codec 回退補丁（`_default_node_value`）掩蓋，但根因是
「完整性靠 adapter 自律、無強制」。

同時 lowering 對 DirectValue scalar **完全不檢查型別 / choices**（只查 `is None`）：UI 路徑有
widget 隔離，但 agent RPC / persist restore / 手構 schema 繞過 UI，非法值（型別不符、不在
choices、LiteralSpec 與 spec 矛盾）能一路通過 lowering 進 exp cfg。

## 決策

**`validate_schema(schema, ml)` 在「成品邊界」顯式呼叫，強制 value 樹完整且靜態合法。**

`CfgSchema` 由 `zcu_tools.gui.cfg.model` 擁有，只保存 spec/value；validation 與
linked-reference resolution 位於 measure-owned `gui.app.main.adapter.lowering`（[[0045]]）。

### validate 的契約（靜態，不需 md）

遞迴查 spec+value，Fast-Fail 第一個錯就 raise（報 path）：
- **結構完整**：每個 `spec.fields` key 在 value 都有 entry。
- **None**：只合法於停用的 optional ref；非 optional ref / 其他 → raise。
- **LiteralSpec**：value 是 `DirectValue` 且 `.value == spec.value`。
- **ScalarSpec（只查 DirectValue）**：型別符 spec.type——**int→float widen OK、float→int reject、
  bool/str 嚴格**；有 choices 則 value ∈ choices。**EvalValue 跳過**（型別在 resolve 時才定）。
- **ref**：用既有 `find_allowed_spec`（lowering.py）從 chosen_key+ml 找 chosen shape，遞迴查 sub-value。
- 簽名帶 `ml`（ref 庫名解析需要）。

### 呼叫點 = 成品邊界（不放 `__post_init__`）

- **`BaseAdapter.make_default_cfg`**：產 schema 後 `validate_schema(schema, ctx.ml)` → adapter 漏/錯當場 raise，
  責任明確指向那 adapter，**框架不補齊**。一處守 ~20 adapter。
- **`schema_to_raw_dict(schema, md, ml)`（lower）**：lower 前先呼叫
  `validate_schema(schema, ml)` → 任何要 lower 的 cfg 先過驗證。

**不放 `CfgSchema.__post_init__`**：`CfgSchema` 被大量「合法的編輯中間態」建構（cfg_form 每次按鍵、
editor draft、codec restore），值可暫時不合法（清空欄位、舊檔）。`__post_init__` 一視同仁會誤傷。
validation 是成品邊界的**顯式動作**，不是型別的隱性負擔。

### 靜態 vs 動態分界

- **靜態**（validate，本 ADR）：結構完整 + LiteralSpec + DirectValue 型別/choices——任何時刻都該成立、不需 md。
- **動態**（lowering 既有 + 未來擴 validate）：required 必須有值、EvalValue 必須 resolve——只有 lower
  那刻（要產 exp cfg）才要求，需 md。lower 一律先 validate（靜態）再自己做動態檢查。

## 否決的選項

- **靠 gate 測試守完整性**：否決。測試是事後防線；validate 在運行時（每次 make_default_cfg/lower）
  Fast-Fail，更強、責任更明確。
- **框架在 `make_default_cfg` 自動補齊缺 key**（用 `inherit_from` 或 merge）：否決。「為 adapter 擦屁股」
  ——讓 adapter 看起來可不完整、框架悄悄掩蓋，違反責任明確。每個 adapter 該自己回傳完整樹。
- **放 `__post_init__`**：否決（誤傷編輯中間態，見上）。
- **LiteralSpec 只查「有 entry」不查值相等**：否決。`lock_literal` 的欄位若 value 帶矛盾值
  （如 onetone/freq freq 鎖 0.0、value 卻 EvalValue("r_f")）是 value 樹自相矛盾的誤導，validate 抓它
  是對的——adapter 在 value 端覆寫成鎖定值（`.with_field("pulse_cfg.freq", 0.0)`）。LiteralSpec 雙重
  來源（值在 spec 也在 value）用此相等檢查綁定，漂移即 raise，化解漂移風險。

## Consequences

- `validate_schema(schema, ml)`（薄 wrapper）→ `lowering.validate_section`（遞迴，復用 `find_allowed_spec`）。
- `make_default_cfg` + `schema_to_raw_dict` 加 static validation 呼叫。
- ~20 adapter：lookback 補 `reps`/`init_pulse`/`reset` entry；onetone/freq 在 value 端 `.with_field`
  覆寫 lock_literal 的 freq/ro_freq=0.0；其餘本就完整不動。
- 移除 codec `_default_node_value` 回退（value 樹保證完整後不需要；非 ref None 改 fast-fail）。
- **lower 加 validate 的行為變嚴**：以前漏網的不合法 cfg（不在 choices、型別不符）現在被擋——
  揪出來逐個判斷（修 cfg/spec 或放寬 validate）。實證現有代碼 100% 型別嚴格，不炸。
- 改一個測試斷言（onetone/freq 不再驗鎖定 freq 是 EvalValue——那是無意義斷言，freq 實驗掃 freq、
  readout freq 被 lock，本不該有人關心其值）。

## 驗證

`tests/gui` + `tests/experiment/v2_gui` + `tests/fluxdep_gui` + `tests/dispersive_gui` **1387 passed**、
pyright 0、ruff clean。新增 `test_validate.py`（20 個：結構完整/缺 key/LiteralSpec/型別 widen-reject/
choices/None-on-ref/ref 遞迴）。live MCP：lookback/twotone tab 新建 cfg + run 過 validate。

與 [[0009]]（spec/value fluent、lock_literal）、[[0010]]（value 樹完整 + None 表空）相關。
第二階段（validate 擴動態檢查、痛點 2 的「有更好 default 值的 helper」）為 backlog。
