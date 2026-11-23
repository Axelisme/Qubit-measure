---
date: 2026-10-21
status: partial
commit: 6def127b6
modules:
  - lib/zcu_tools/simulate/evolution
  - lib/zcu_tools/simulate/adc
discussion: ../discussions/2026-10-21-joint-fock-performance-study.md
superseded_by: null
---

# Joint Fock CKP 的 RK4 與 cutoff

**讀者：** framework-dev、agent。查 Joint Fock CKP 的 RK4 與 cutoff 的做法與證據。

## 核心思路

每個 rf 共用準備產物，用保守步長與自適應 cutoff 壓低兩能階解的成本。

## 實測

原研究列出固定 checkout 量測，但沒釘出確切量測 commit；不能把局部資料的全圖外推當成實跑。 見[原討論](../discussions/2026-10-21-joint-fock-performance-study.md#依據)。

## 為何採用或否決

研究採保守 RK4 加額外 cutoff 餘裕。後續只落地供選用的 prototype 與診斷，未取代 production 半古典路徑。

## 留下的線索

檢查完整 map、不同驅動下的步長穩定性和多能階成本，再考慮推廣。
