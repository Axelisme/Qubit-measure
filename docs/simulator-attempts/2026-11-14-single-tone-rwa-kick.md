---
date: 2026-11-14
status: adopted
commit: 72d6381ca
modules:
  - lib/zcu_tools/simulate/evolution
discussion: ../discussions/2026-11-14-single-tone-rwa-kick.md
superseded_by: null
---

# single_tone_rwa 保留 endpoint kick

**讀者：** framework-dev、agent。查 single_tone_rwa 保留 endpoint kick 的做法與證據。

## 核心思路

在兩端補一階 kick 並用 κ2 放行，保住較快的 RWA 路徑。

## 實測

κ2 上限 0.08（設定值）；原討論沒有將每組研究耗時與精度數字釘到量測 commit。 見[原討論](../discussions/2026-11-14-single-tone-rwa-kick.md#依據)。

## 為何採用或否決

先用完整序列誤差核對保留的路徑。初始 kick 後忽略很小的跨 component 項，避免逐對演化。

## 留下的線索

比較價格時要算 kick。保留某個 family 不等於它應先於所有替代路徑。
