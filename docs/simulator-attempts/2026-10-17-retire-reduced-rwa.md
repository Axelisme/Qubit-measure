---
date: 2026-10-17
status: adopted
commit: 02624ff45
modules:
  - lib/zcu_tools/simulate/evolution
discussion: none
superseded_by: null
---

# 退役早期 reduced RWA

**讀者：** framework-dev、agent。查早期單 tone 後援為什麼移除，以及它和後來 RWA family 的區別。

## 核心思路

早期的 constant-tone 路徑在 Dyson 拒收後，還會試一個選擇 sideband 的 reduced RWA。該方法只留下近共振側帶，並用省略側帶的誤差界決定能否放行。這次改動刪掉該分支，改為 Dyson 拒收後直接試 Hilbert period，再退到 T6。

## 實測

沒有找到可核對量測 commit、工作負載及範圍的前後計時。退役由 `02624ff45` 的實作和 `cfdbdd79f` 的合併說明確認，兩者不是計時收據。

## 為何採用或否決

當時的合併說明指出，FTF 使用線性偏振的電荷算符組合。對這種輸入，省略側帶的誤差界只會放行 Dyson 已經接受的旋轉。因此這個後援在目標負載上沒有新增覆蓋。這是當時的退役理由，不是對任意極化的證明。

## 留下的線索

要保留一個後援 family，先找前序 family 不收、它卻能在誤差預算內更快求解的列。這次刪除的是早期獨立的 reduced RWA kernel，不是後來加入的 single-tone 或 multitone RWA。
