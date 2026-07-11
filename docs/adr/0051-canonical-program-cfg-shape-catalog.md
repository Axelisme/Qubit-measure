---
status: accepted
---

# 0051 — Canonical program cfg shape catalog

**狀態：** accepted（2026-07-11）。
**關聯：** [[0020]]、[[0045]]、[[0046]]、[[0050]]。

## 背景

measure-gui 與 autofluxdep-gui 編輯同一套 program/v2 module/waveform cfg，但各自維護
discriminator、label 與完整 Spec factory。兩份結構只有 Arb data option source 與 readout
跨 shape inheritance 兩個真實差異；重複 owner 會讓 runtime vocabulary、reference label 與表單
shape 漂移。

## 決策

`zcu_tools.gui.measure_cfg` 是跨 app、Qt-free 的 program cfg GUI projection owner：

- closed catalog 固定列出七種 module 與六種 waveform discriminator、label、fresh spec factory；
- catalog 不做 runtime registration，也不 import program/v2、app、session、experiment、Qt 或
  `meta_tool`；test 以顯式 runtime cfg class 集合驗 closed discriminator parity；
- 每次 factory call 建立 deep-fresh Spec tree，包括 nested section、scalar choices 與
  `ReferenceSpec.allowed` mutable containers；
- `ProgramSpecPolicy` 只有 `arb_data_choices_source` 與
  `enable_readout_shape_inheritance` 兩欄。main policy 啟用 `arb_waveforms` choices 與 readout
  inheritance；autoflux policy 保持 free-form Arb data 且不掛 inheritance hook；
- explicit unknown discriminator 一律 Fast Fail。raw dict 缺少 waveform style 時既有 Const default
  屬 materialization policy，不等同 unknown explicit style；
- pulse root label 可由 caller factory override，`Pulse 1` / `Pulse 2` 不形成額外 catalog entry。

`gui.cfg` 繼續只擁有 generic Spec/Value、inheritance、binding 與 lowering mechanics，不 import
measure-domain vocabulary。main `specs` 與 autoflux `module_adapter` 只做窄 policy binding；autoflux
node reference allowed subset仍精確限制為 Pulse / Pulse Readout。experiment role seed與預設值仍由
experiment layer擁有，只從 canonical catalog取得 shape。

raw materialization與 RoleEntry/Controller create flow不屬於本階段：materializer收斂時仍須顯式
保存 missing scalar/nested default與 app allowed subset；consumer收斂時直接攜帶 catalog shape，
不得讓 generic cfg core理解 program discriminator。

## 後果

- module/waveform shape、label與factory只有一個 owner，兩個 app 的 normalized spec tree只剩兩個
  可測 policy 差異。
- catalog import保持輕量，runtime vocabulary drift由測試在整合期 Fast Fail。
- unknown explicit style不再在 main blank seam silent fallback成 Const；wire contract不變，GUI code
  revision增加。
- shape catalog完整性不擴張 autoflux可 materialize 或 node reference允許的 module subset。

## 拒絕的替代方案

- **由 program/v2 reflection 自動註冊 GUI fields**：runtime knobs不等於刻意公開的 GUI subset，且會
  把 QICK/Pydantic runtime graph帶入輕量 catalog import。
- **任意 spec mutator callback**：會讓 app重新取得 shape ownership，無法證明只有兩個 policy差異。
- **把 catalog 放入 `gui.cfg` 或 `gui.session`**：generic mechanics與session lifecycle都不擁有
  program domain vocabulary。
- **保留 main unknown → Const compatibility branch**：會把 corrupt explicit discriminator靜默改形。
