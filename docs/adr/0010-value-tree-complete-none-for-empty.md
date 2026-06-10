---
status: accepted
---

# Value 樹永遠完整 + `None` 統一表「空」（停用 optional ref + 未填 scalar）

**狀態：** accepted（已實作）。
**關聯：** 承 [[0009]]（spec/value fluent，「要不要畫是 GUI 決策不是 spec 概念」同源延伸）；persistence 格式見 [[0015]]；成品邊界驗證見 [[0011]]。

## 脈絡

**Bug**：lookback tab cfg 設 `Reset:None` / `Init Pulse:None`，關閉重啟後變 `Reset:NoneReset` / `Init Pulse:Pulse`——停用態持久化後遺失，還原成 enabled 的第一個選項。

**根因（職責層級）**：「optional ModuleRef 停用」這個概念在 value 層曾有**多種互不相容的長相**，散在多層各自漂移——旁路 boolean flag（`is_enabled` 與 `get_value()` 平行、停用態在這層隱形）、「省略 key」（父層 reach-in 子層 `is_enabled` 代為省略，責任倒置）、以及顯式 marker 型別。bug 是這些約定在接縫處漏水：捕獲端用「省略 key」、還原端 `make_default_value` 對缺 key 的 optional ref 回 **enabled** 的 `allowed[0]`。

同源問題在 scalar：「未填」曾用 `DirectValue(value, is_unset)` **雙欄位**表達——`value` 在未填態被覆寫成型別占位（0/""），真相只在 `is_unset` flag，且不同產生點還不一致。

## 決策

把「空」這個概念在整棵 value 樹裡統一：

1. **value 樹永遠完整**：`make_default_value` 等產生 value 的地方，**每個 spec 欄位都產一個 entry**，不再有「缺 key」這個態。每個欄位的狀態就是它的 value 本身（完整自述），不需問 `spec.optional`。
2. **一切「空」統一用 `None`**：
   - **停用 optional ModuleRef/WaveformRef** = `fields[k] = None`（裸 `None`，不記 chosen_key；重新啟用走 helper 預設——停用是純 marker、無 payload）。
   - **未填 scalar** = `DirectValue(value=None)`（**包裝層保留**——scalar 仍須保住 direct vs `EvalValue` 的**模式**身份，裸 `None` 會抹掉這個二元結構）。
3. **檢測「空」一律 value 自述**：ref 空 = `fields[k] is None`；scalar 空 = `dv.value is None`。不再 isinstance sentinel、不再讀 is_unset flag、不再反推 spec。
4. **`make_default_value` 是 helper，可盡量猜合理預設**（scalar 0、sweep 範圍、choices[0]）——對 optional ref 猜「停用（`None`）」是最安全、最不驚訝的預設。有特殊需求的 adapter 用 OO 鏈式/工廠覆寫（`with_field`、`make_*_ref_default(optional=True)` 無 lib 時回 `None`）。
5. **「停用→消失」只在 lowering**（run/save 出口）發生：`_section_to_dict_inner` 對 `None`（停用 ref）省略輸出、對未填 required scalar raise。persist 是 value 樹的忠實序列化，不套用 lowering 的省略語義（停用 ref 序列化為 `{"__kind":"disabled"}`，還原回 `None`）。

### scalar 與 ref 的「空」載體不對稱，且正確

- scalar 有「同格兩種輸入模式（direct/eval）」要保 → 空必須包在 `DirectValue`/`EvalValue` 裡。
- ref 無「模式」概念（狀態就是選哪個模組或停用），停用前的選擇是便利記憶非語義必要 → 停用可裸 `None`。

不對稱是設計依據，非凑合。共同哲學：**狀態由 value 本身表達，不靠並排旁路 flag、不反推 spec。**

## 演化（被取代的設計，保留脈絡）

早先方案（`DisabledRefValue` marker——一個一等的「停用態」value 型別，符合 `CfgNodeValue`）曾被採用，當時明確**否決** 用 `None`，理由＝「`None` 哨兵用『不存在』表達『停用』、不夠強型別」。**該否決的前提是 value 樹會有缺 key（omit key）**，使 `None`＝「不存在」與「停用」混淆。

本決策的 §1（value 樹永遠完整）**消滅了「不存在」這個態**：`None` 不再等於「缺 key」，而是一義的「entry 在、值為 `None`＝停用」。舊否決理由在新前提下不成立——`DisabledRefValue` 已刪除，停用態改用 `None`。當年用 marker 消除 8 個 twotone adapter 各 3 行 `if x is not None` 守衛的成果**保留**（`make_*_ref_default(optional=True)` 改回 `None` 仍是一行放進 fields，因 `CfgSectionValue.fields` 型別放寬成 `dict[str, Optional[CfgNodeValue]]`）。

## 否決的選項

- **保留 `DisabledRefValue` sibling（型別載體）**：否決。多一個型別、與 scalar 的「空＝None」不一致；value 樹完整後 `None` 已足夠一義。
- **scalar 未填用裸 `None`（與 ref 對稱）**：否決。會抹掉 `DirectValue` vs `EvalValue` 的模式身份（eval 寫到一半切走再回來會丟失）。
- **ref 停用記住 chosen_key（`ModuleRefValue(..., enabled=False)`）**：否決。違反「停用無 payload」，製造「停用但背著一份子樹（要不要驗證/序列化？）」的薛丁格態。
- **改 `make_default_value` 全域行為去迁就 lookback 一個 adapter 的偏好**：否決。偏好由各 adapter 用 OO 鏈式/工廠**顯式**聲明。

## Consequences

- 動 value 型別系統：刪 `DisabledRefValue`（從 `CfgNodeValue` union 移除）、`DirectValue` 刪 `is_unset` + value 改 `Optional`、`CfgSectionValue.fields` 改 `dict[str, Optional[CfgNodeValue]]`。
- `live_model`：`ModuleRefLiveField.get_value/set_value` 對停用回/收 `None`；`SectionLiveField` 父層不再 reach-in；`ScalarLiveField` 未填存 `None` 不再填占位。
- `session_codec`：停用 ref↔`{"__kind":"disabled"}`↔`None`；scalar 不寫 is_unset，未填 `value:null`；缺 optional ref key 經完整 `make_default_value` 兜底自動得 `None`（bug 還原端自動修復）。
- `lowering`：`None` 統一處理（optional 省略 / required raise）；邏輯變簡單。
- `make_*_ref_default(optional=True)`（8 個工廠，`experiment/v2_gui`）改回 `None`；lookback **無需改**（省略 key 經 helper 完整兜底自動得 `None`）。
- persistence 格式微調（scalar 去 is_unset、停用 ref 用 `disabled` marker）：舊檔走 strict fallback（[[0015]]），可接受一次性遷移；屬同 GUI_VERSION 內（無 wire/RPC 語意改動）。
