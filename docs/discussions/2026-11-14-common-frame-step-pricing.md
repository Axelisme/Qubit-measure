---
date: 2026-11-14
status: active
superseded_by: null
related:
  - docs/adr/0078-evolution-module-boundary.md
  - lib/zcu_tools/simulate/evolution/README.md
basis:
  - "repo base@f9ec2fe2e70418c6af2a54e8b41df603c1bdc651"
---

# common_frame 的步長成本歸 dispatch

## 結論

這次把 `common_frame` 的 2 ns 步長下限改成 dispatch 的步數價格。Kernel 不再用成本拒絕輸入。固定 family 可以執行通過數值 guard 的列，不受載波間距的成本捷徑限制。職責依據是 [ADR-0078](../adr/0078-evolution-module-boundary.md#內部允許實質耦合dispatch-擁有決策知識)，算法與數值限制由 [evolution README](../../lib/zcu_tools/simulate/evolution/README.md#kernel-summary) 說明。

## 問題

O1 的兩個完全重疊 constant tone 使用 20 能階模型。原本外側六點的完整 density 誤差約 `7.9e-5`，卻因步長小於 2 ns 被 kernel 以成本理由拒絕。步長下限只限制載波間距，不衡量一列實際要走多少步。它也讓 diagnostics 把成本略過記成數值拒絕。

## 考慮過的選項與比較

保留門檻不改變現況。調低門檻能放行部分輸入，但仍把成本出口放在 kernel。改成 kernel 內的步數上限會衡量工作量，卻仍違反 dispatch 擁有成本比較的分工。

把 apply 價格改成依 first-order 步數計算，同時保留完整 coefficient-key 的建置攤提，能在 dispatch 比較候選。原本 8 ns 步長時的價格尺度不變。這次遵循 ADR-0078 的成本所有權與 pinned family 契約，不新增載波距離的有效域限制。

## 被否決的選項與理由

只讓 pinned 繞過門檻會讓 kernel 多一個依 caller policy 分支的出口，預設路徑仍會擋掉合格輸入。一次合併 RWA、風險與其他成本捷徑則需要另做定價校準，不在本次範圍。

## 留下的問題

此次精度檢查使用 O1 gain，在 50、72、94、116、138、160 MHz 的六列全部放行，完整 density 最大誤差為 `7.92e-5`。正式 O1 的 24 點在不含 `multitone_rwa` 的候選白名單下全部由 `common_frame` 執行，誤差低於 `2e-3`。O2 約 4.13 GHz 的兩個檢查點被數值 guard 拒絕。這些觀察不保證任意 gain、duration 或模型的精度，也不修正既有精度矩陣的 guard-edge 超標。

## 依據

實作與價格分別由 [common-frame 數值 owner](../../lib/zcu_tools/simulate/evolution/kernels/common_frame.py) 與 [dispatch](../../lib/zcu_tools/simulate/evolution/_common_frame_dispatch.py) 擁有。Pinned 與預設成本略過的行為由 [Engine 路由測試](../../tests/simulate/evolution/test_common_frame_routing.py) 驗證。

一次性精度工作檔未入庫。比較使用獨立 full-Lindblad interaction-picture 參考解，容差為 `2e-10/2e-12` 與 `2e-12/2e-14`。兩級解的最大 density 差約 `1.03e-10`。正式 O1、O2 的 24 點參考解先經 cache-only 核驗，新增間距只求解 cache 未涵蓋的五點。

## 後記

2026-11-14 核對逐列結果時，發現上文把 160 MHz 的誤差當成六列最大值。六列的實際最大值是 72 MHz 的 `8.248844445949352e-5`，不是 `7.92e-5`。正式 O1 的 pinned 與不含 `multitone_rwa` 的候選白名單各有 24 點，兩者最大誤差都是 `8.133469313725601e-5`。這些結果仍低於 `2e-3` 的驗收門檻。
