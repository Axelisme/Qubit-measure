# 0043 — Autofluxdep runtime cfg override-plan foundation

**狀態：** accepted（2026-07-05）。
**關聯：** [[0018]]、[[0040]]、`new_cfg_design.md`。

## 背景

autofluxdep 的 node cfg editor 顯示 Default cfg 與 Generation overrides。Default cfg 是使用者以 measure-gui typed cfg form 編輯的 run template；Generation overrides 描述跨 flux point 的自動生成策略。沒有顯式 contract 時，runtime 可以在 `make_cfg()` 中覆寫 Default cfg path，造成使用者看見的設定與 run 實際使用的設定不一致。

## 決策

autofluxdep 以 `OverridePlan` 作為 runtime cfg override 的公開基礎契約。每個 `Builder` 可宣告一組 `OverridePath(path, mode, source, reason)`；path 指向 Default cfg editor path，mode 是 `after_first_point` 或 `all_points`，source 是機器可比對的 generation/dependency 來源，reason 是人類可讀說明。

`OverridePlan` 屬於 node runtime contract，由 builder 宣告，不放入 `NodeCfgSchema`。`NodeCfgSchema` 仍只負責 typed spec/value tree、logical-key projection、lowering 與 persistence，不承載 experiment-specific generation semantics。

Run start 對每個 enabled runnable node 建立 run-start snapshot：

- `base_cfg`：以當下 md/ml lower 的 Default cfg raw dict，排除 `generation` section。
- `override_plan`：該 node builder 在同一 schema 上宣告的 wire-safe plan。

controller 在建立 run artifact 前驗證每個 override path 存在於 `base_cfg` 且不是 mapping leaf，避免 whole-module replacement 以相容性 fallback 形式回來。這個驗證是 foundation gate；尚未遷移到 override-plan patch resolver 的既有 node 不因此被宣稱已完全消除所有 runtime overwrite。

Remote `node.cfg` 是只讀觀測面，回傳 `{name, type, knobs, override_plan}`。Run artifact manifest 的 workflow node entry 同時保存 persisted workflow `cfg`/`cfg_hash` 與 run truth `base_cfg`/`override_plan`，讓事後審查能分辨「workflow 編輯值」與「本次 run 開始時 lower 出來的 Default cfg」。

Cfg form 的 generic decoration seam 以 full value-tree path 計算 `FieldDecoration`。共用 renderer 不知道 autofluxdep semantics；autofluxdep-specific presenter 可由 `OverridePlan` 提供 badge、tooltip、tone 或 enabled state。Decoration 是 presentation metadata，不是 domain enforcement。

## 後果

- 新 node 若會跨 flux point 覆寫 Default cfg path，必須先在 builder `override_plan(schema)` 宣告 path、mode、source、reason。
- `OverridePlan.path`、Default cfg editor path、runtime patch path 使用同一個 dotted path vocabulary。`generation.*` path 不可出現在 plan 中。
- `all_points` 不表示 fallback Default cfg；若 generated source 缺失，fallback policy 必須在 Generation 區或 dependency resolver 顯式表達。
- Remote/MCP 仍 read-only；`override_plan` 只讓 agent 觀測 contract，不授權 agent 修改 workflow 或 run control。
- Artifact consumer 應以 `base_cfg` + `override_plan` 作為 run-start cfg truth；workflow memento 仍是 GUI 下一次開啟的 editable state。

## 拒絕的替代方案

- **把 override metadata 放進 `NodeCfgSchema`**：會讓通用 cfg/value model 承擔 experiment generation semantics，破壞 cfg seam。
- **允許 whole-module replacement fallback**：會重新引入隱式 path rename/shape mismatch，讓 UI path 與 runtime path 分裂。
- **remote write tier 順手加入 cfg mutation/run control**：autofluxdep remote bridge 的角色是第二個 read-only view；run control 仍由 GUI 使用者操作。
