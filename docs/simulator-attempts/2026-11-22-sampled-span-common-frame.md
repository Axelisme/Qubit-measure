---
date: 2026-11-22
status: rejected
commit: 47fbcf5e8
modules:
  - lib/zcu_tools/simulate/evolution
discussion: none
superseded_by: null
---

# 讓被拒的 sampled span 改走 common-frame

**讀者：** framework-dev、agent。查放寬 common-frame guard 或加入包絡導數修正是否更快。

## 核心思路

早期 heralded RB 有一段雙 tone sampled CZ 被 common-frame 的 coherent guard 擋下，退到 CPU DP5(4)。調查先強制該列走現有 common-frame，再試把包絡斜率的修正加進 Hamiltonian。兩種原型都希望保留端點精度，同時比 fallback 便宜。

## 實測

在 `47fbcf5e8` 重建 seed 685721 的 heralded RB reference 臂，depth 1、acquire 0、point 1、history `[1]`。四段的輸入簽章、shift 與路由都對上原收據。只量末段 7450–7850 ns，d=24，兩個複數 sampled 包絡帶固定 segment shift。以下是同場非排他的單列 apply 牆鐘，誤差是相對獨立 full-drive Lindblad 參考的端點密度最大元素差。

| 單列路徑 | Apply 牆鐘 | 端點差 |
| --- | ---: | ---: |
| CPU DP5(4) | 0.782 s | `8.81e-5` |
| 強制 common-frame | 0.877 s | `4.94e-6` |
| Common-frame 加斜率修正 | 1.805 s | `4.21e-7` |

原型與原始工作檔未入庫。參考解把固定 shift 併入模型，只驗這一列，不覆蓋一般的 history-dependent shift。

## 為何採用或否決

不改 guard，也不加導數修正。兩個原型都比 fallback 慢。後續的代表性 two-qubit 掃描也沒有重現這條 common-frame 拒收後改走 DP5(4) 的路由，不能拿早期的 workload 成本外推。

## 留下的線索

單列誤差夠低，不能證明要放寬所有同形輸入的 guard。先在當前目標輸入重查真正的 fallback 集合。若仍是 CPU DP5(4) 慢，可先量首步拒絕率，再對照[區間起始步長調整](./2026-11-22-dp54-interval-step-seed.md#實測)。
