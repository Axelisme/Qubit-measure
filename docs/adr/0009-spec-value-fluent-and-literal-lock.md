---
status: accepted
---

# Spec/Value fluent 覆寫 + LiteralSpec 鎖定 + 每角色 default factory

**狀態：** accepted（已實作）。本檔以現在式描述最終形狀（含試點後收斂與每角色重構）。
**關聯：** value 樹完整性與「空」語義見 [[0010]]；成品邊界驗證見 [[0011]]；value 層組裝 builder 見 [[0012]]。

## 脈絡

系統分析 `single_qubit.md`（~50 experiment cell）後，adapter 撰寫端浮出三個耦合需求：(A) 覆寫 default（fallback 值寫死）；(B) 鏈式構造（取代命令式 `make_*_spec()` + `make_default_value()` + `_patch_*_fields()`）；(C) 鎖定欄位（notebook 高頻 `freq: 0.0, # not used`，被 sweep 軸接管）。grilling 過程中數個初始直覺被代碼證據推翻，故記錄。

## 決策

### 1. 鎖定屬於 Spec 樹，用 `LiteralSpec`

證據：`types.py` 契約「Spec tree — static, never mutated」「Value tree — mutable」；`editable` 在 `ScalarSpec`、value 節點碰不到。`LiteralSpec` 已用於 spec 樹（`"type": LiteralSpec("readout/pulse")`），`LiteralLiveField` 已實作「value 永遠=spec.value、set_value no-op」。

→ 鎖定 = `cfg_spec()` 內把該葉宣告/替換為 `LiteralSpec(value)`。**spec-only、不跨樹、不碰 value**。`lock_literal` **必須在 `cfg_spec()` 內呼叫且結果被 `return`**——鎖是 spec 契約的一部分，在 `cfg_spec()` 外對其回傳值鎖會讓 spec 不再是 SSOT。

**被否決**：`0.0 + editable=False`——須跨兩樹同改、GUI 顯示唯讀框而非藏；`LiteralSpec` 更乾淨。

### 2. Spec fluent 用「frozen + 遞迴 replace 回同型」，非可變 builder

`CfgSectionSpec.lock_literal(path, value)` 掛在 spec 型別上、回**同型新 frozen spec**（遞迴 `replace`）。回同型即天然鏈式，**無 wrapper、無 `.build()`/`.done()`**。path 走 dotted 字串，每種 spec 型別遞迴自己的子結構（`CfgSectionSpec` 走 `fields`、`ModuleRefSpec` 走 `allowed`，duck-type：含該 path 的 allowed shape 就套、全部不含才 raise）。

**被否決**：可變 SpecBuilder 鏡像（`.done()` 是 builder 氣味、與 fluent 矛盾、違反「收斂到既有抽象」）。

### 3. 鎖定/渲染概念只在 spec 層；schema 層獨立覆寫機制不存在

- **`LiteralSpec` 由 widget 隱藏，spec 不帶渲染旗標**：`containers.py` 對**所有** `LiteralSpec` 都 skip widget。鎖定欄位自動消失，無需 spec 帶 `hidden`。「要不要畫」是 GUI 決策、不是 spec 概念。
- **`editable` 保留**——它是**語義**屬性（該不該被改），非渲染指令；要設直接 `ScalarSpec(editable=False)`。無 `readonly` fluent、無 `ScalarSpec.hidden`（純渲染概念，YAGNI）。

> 演化：曾規劃「spec 層 lock（LiteralSpec）／ schema 層 `schema_overrides.py`（合體後 `CfgSchema` 的 editable=False / hidden 覆寫）分層分工」。實作後推翻——`schema_overrides.py` 是零 import 死代碼、且把 GUI 渲染概念編進 schema 層，**已刪除**。收斂為**只有 spec 層**（cfg_spec 內 LiteralSpec）。若日後需要合體後的動態覆寫，再以不含 GUI 概念的形式重建。

### 4. Value OO 覆寫 in-place 回 self —— 刻意不對稱於 spec 層

value 容器（`CfgSectionValue`/`ModuleRefValue`）**維持可變、`with_*` in-place 改 fields 回 self**，而非回新 frozen。證據支撐 in-place 安全：`make_default_value(spec)` 每次從頭新建（不跨呼叫共享）；runtime LiveModel 不 in-place 改傳入 value。

**代價（誠實記錄）**：`spec.lock_literal()` 回新 frozen、`value.with_gain()` 改 self——兩側機制不對稱，是認知負擔點；取捨理由：value 本就可變、改動最小，強收 value frozen 是更大的 YAGNI。

### 5. 每角色一檔的 default factory（單層）

`defaults/<role>.py`（qub_probe / res_probe / pi_pulse / pi2_pulse / readout / reset / qub_waveform / res_waveform）各暴露 `make_<role>_default`（blank）+ `make_<role>_ref_default`（查庫 preferred → fallback blank，optional 無 lib 時回 `None`，見 [[0010]]）。共用 patch helper 收進 `defaults/helpers.py`。RoleCatalog 直接使用 blank factory；adapter 通常透過 [[0012]] 的 `CfgBuilder.role(..., Init.ADOPT/INLINE/DISABLED)` 選 ref / blank / disabled，少數特殊 value 組裝仍可直接呼 L2 factory。

**default factory 零鎖定（職責邊界）**：default factory 只產 value 樹預設，**不預設鎖任何欄位**，即使是高頻場景。鎖定 100% 由 adapter 在 `cfg_spec()` 裡 `lock_literal` 宣告——鎖定屬 spec 層（決策 1）、default factory 屬 value 層；「高頻」不是放進 factory 的理由。

> 演化：曾規劃「角色 wrapper 委派兩層通用 factory」（`default_pi`/`default_qub_probe`…）。後收斂為**每角色一檔的單層結構**（完整對稱矩陣），刪舊三層委派。

## 替代方案（綜述）

| 維度 | 選擇 | 否決 |
| --- | --- | --- |
| 鎖定落點 | spec / LiteralSpec | value / 0.0+editable=False |
| spec 覆寫機制 | frozen + replace 回同型 | 可變 builder + .build() |
| 鎖定/渲染概念 | 只在 spec 層（LiteralSpec by widget 隱藏） | schema 層 `schema_overrides.py`（已刪） |
| value 覆寫 | in-place 回 self | frozen + 回新（對稱但更大改動）|
| 角色 default | 每角色一檔單層 factory | 兩層通用 factory + wrapper |
