# 0043 — Autofluxdep runtime cfg override plan

**狀態：** accepted（2026-07-05）。
**關聯：** [[0018]]、[[0040]]、`new_cfg_design.md`。

## 背景

autofluxdep 的 node cfg editor 顯示 Default cfg 與 Generation overrides。Default cfg 是使用者以 measure-gui typed cfg form 編輯的 run template；Generation overrides 描述跨 flux point 的自動生成策略。沒有顯式 contract 時，runtime 可以在 `make_cfg()` 中覆寫 Default cfg path，造成使用者看見的設定與 run 實際使用的設定不一致。

## 決策

autofluxdep 以 `OverridePlan` 作為 runtime cfg override 的公開契約。每個 `Builder` 宣告一組 `OverridePath(path, mode, source, reason)`；path 指向 Default cfg editor path，mode 是 `after_first_point`、`all_points` 或 `fallback`，source 是機器可比對的 generation/dependency 來源，reason 是人類可讀說明。

`OverridePlan` 屬於 node runtime contract，由 builder 宣告，不放入 `NodeCfgSchema`。`NodeCfgSchema` 仍只負責 typed spec/value tree、logical-key projection、lowering 與 persistence，不承載 experiment-specific generation semantics。

Run start 對每個 enabled runnable node 建立 run-start snapshot：

- `base_cfg`：以當下 md/ml lower 的 Default cfg raw dict，排除 `generation` section。
- `override_plan`：該 node builder 在同一 schema 上宣告的 wire-safe plan。
- `knobs`：同一時間 lower 的 logical-key knob snapshot，供 runtime 讀固定的 generation strategy。

每個 flux point 從 `base_cfg` deep copy 開始，只套用 builder-declared `override_plan` 內允許的 patches。未宣告 path、`after_first_point` 在 flux index 0 被 patch、或 target 不存在都會 Fast Fail。`all_points` path 每個 flux point 都必須提供 patch；`after_first_point` path 在 flux index 0 使用 `base_cfg`，後續 flux point 都必須提供 patch；`fallback` path 可在任一 flux point 省略，省略時保留 `base_cfg` value，提供 patch 時套用 patch。`modules.<name>` whole-module replacement 一律禁止；像 `modules.pi_pulse.waveform` 這類 module 內 discriminated sub-object 可被宣告並整體 patch，避免把 `waveform.style` 改成 `flat_top` 卻漏掉 `raise_waveform` 的非法半狀態。

Remote `node.cfg` 是只讀觀測面，回傳 `{name, type, knobs, override_plan}`。Run artifact manifest 的 workflow node entry 同時保存 persisted workflow `cfg`/`cfg_hash` 與 run truth `base_cfg`/`override_plan`，讓事後審查能分辨「workflow 編輯值」與「本次 run 開始時 lower 出來的 Default cfg」。

Cfg form 的 generic decoration seam 以 full value-tree path 計算 `FieldDecoration`。共用 renderer 不知道 autofluxdep semantics；autofluxdep presenter 由 `OverridePlan` 提供 badge、tooltip、tone 與 enabled state。`all_points` 顯示 `generated` badge 並 disabled；`after_first_point` 顯示 `initial` badge、warning tone 且保持可編輯，表示 flux 0 使用手填值、後續點使用生成值；`fallback` 顯示 `fallback` badge 並保持可編輯，表示 dependency 或 generation source 有值時覆寫，沒有值時使用 Default cfg template。Decoration 是 presentation metadata，runtime enforcement 仍由 `apply_override_patches()` 負責。

## 後果

- 新 node 若會跨 flux point 覆寫 Default cfg path，必須先在 builder `override_plan(schema)` 宣告 path、mode、source、reason。
- `OverridePlan.path`、Default cfg editor path、runtime patch path 使用同一個 dotted path vocabulary。`generation.*` path 不可出現在 plan 中。
- `make_cfg()` 不重新 `schema.lower_raw()` 或整個替換 `raw_cfg["modules"][...]`。它讀 `RunEnv.knobs()` 的 run-start snapshot，計算 patches，透過 builder `point_cfg(env, patches)` 套用到 run-start `base_cfg`。
- `fallback` 是唯一可省略 patch 的 runtime mode；它仍必須由 builder 宣告 path、source、reason，且 path 必須存在於 run-start `base_cfg`。Artifact / remote consumer 必須把它視為 public wire/artifact mode，而不是 UI-only badge。
- Remote/MCP 仍 read-only；`override_plan` 只讓 agent 觀測 contract，不授權 agent 修改 workflow 或 run control。
- Artifact consumer 應以 `base_cfg` + `override_plan` 作為 run-start cfg truth；workflow memento 仍是 GUI 下一次開啟的 editable state。

## 拒絕的替代方案

- **把 override metadata 放進 `NodeCfgSchema`**：會讓通用 cfg/value model 承擔 experiment generation semantics，破壞 cfg seam。
- **允許 whole-module replacement fallback**：會重新引入隱式 path rename/shape mismatch，讓 UI path 與 runtime path 分裂。需要保留 discriminated sub-shape 時只允許宣告到 module 內的子物件 path；`fallback` mode 也必須遵守同一個 leaf/sub-object path contract。
- **remote write tier 順手加入 cfg mutation/run control**：autofluxdep remote bridge 的角色是第二個 read-only view；run control 仍由 GUI 使用者操作。
