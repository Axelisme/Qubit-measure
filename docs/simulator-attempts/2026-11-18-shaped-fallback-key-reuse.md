---
date: 2026-11-18
status: adopted
commit: 547502e65
modules:
  - lib/zcu_tools/simulate/evolution
discussion: ../discussions/2026-11-18-shaped-fallback-key-reuse.md
superseded_by: null
---

# 成形脈衝的 fallback 共用

**讀者：** framework-dev、agent。查 成形脈衝的 fallback 共用 的做法與證據。

## 核心思路

先解析可共用表的 Sambe guard，讓 fallback 可跨 timeline 共建 period table。

## 實測

原討論的合成案例證明 spectral 可共用，沒有落地版的成對速度數字；不把各 duration key 的建構計數當成完整耗時。 見[原討論](../discussions/2026-11-18-shaped-fallback-key-reuse.md#留下的問題)。

## 為何採用或否決

保留原本 timeline 內的 Sambe spectral 共用，避免在不走 fallback 時重建。

## 留下的線索

用 `prepare_key_builds` 分辨共用與重建。舊候選的效能數字不能沿用到審查修正後的版本。
