---
date: 2026-11-21
status: active
superseded_by: null
related:
  - docs/adr/0078-evolution-module-boundary.md
  - lib/zcu_tools/simulators/ftf/environment/README.md
  - lib/zcu_tools/simulators/README.md
basis:
  - "repo@086e9ff124dfd89a4529b9c5ac22c608535d54c8：整合所依據的 FTF 列身分與 capture 實作"
---

# FTF projective 列身分與失敗定位

## 結論

FTF 依原點與依序投影出的狀態，也就是 history，從 SHA-256 取 63 bit 作為 Engine 列 ID。同一組原點與 history 跨輪、跨 preparation 沿用同一 ID。FTF 檢查單次 preparation 的所有點與 span。不同組合若在這個範圍內撞到同一 ID，FTF 以 `ValueError` 指出組合，不合併列，也不回傳部分結果。跨 preparation 不檢查碰撞。

FTF 把原點、絕對 span 與 history 的對照寫進 [`metadata.ftf.projective_spans`](../../lib/zcu_tools/simulators/ftf/environment/README.md#debug-capture)。先用 Engine 記錄的 `capture_id` 找到該次呼叫，再用 Engine 點 ID 找該列。同一 ID 跨 span 或跨輪出現時，也能分清是哪一次呼叫。Engine 的通用 capture 格式不變。

這兩項是 designer 在使用者授權下作的代裁，仍待使用者覆核。

## 問題

FTF 把同一原點、同一 span 的 histories 一起交給 Engine。原做法每批從 0 編號。兩個不同原點的整批數值若一樣，就拿到同一份 Engine content digest，選路用量漏算第二批。`NoValidKernel` 和 capture 也只看得到批內 ID，無法指出哪個原點或 history 出錯。

穩定 ID 解決一般情況，但 Engine 的非負 int64 只留 63 bit 給身分。借用的 routing history 可以跨 preparation、跨 acquire，甚至跨 model source。當時還要決定在多大的範圍內檢查 ID 碰撞。

## 考慮過的選項與比較

身分和定位有三種做法。FTF 可以給穩定 ID 並在自己的 manifest metadata 保存對照。Engine 也可以新增公開 row-provenance API，替各呼叫端傳遞來源。第三種是接受批內 ID。選第一種，因為 Engine 本來就接收呼叫端提供的點 ID，FTF 也已經知道每列的原點和 history。不必拿 preparation key 當身分，或替 Engine 改算用量。

碰撞範圍有四種做法。只檢查單次 preparation，可以在兩列混用前報錯，而且不留跨 preparation 的狀態。由 FTF acquire 保留對照，可以涵蓋同次 acquire 的各輪，但查不到下一次借用同一 routing history 的點。由 routing history 的 owner 保留對照，才涵蓋整段用量生命週期。有界可逆編碼不需要對照，但會限制可表示的點數或 history 長度。

採用單次 preparation 的檢查。若不同原點與 history 的組合只在不同 preparation 出現、ID 相撞，且整批其餘 Engine 輸入逐位相同，Engine 才會把第二批視為重複輸入，少計一次用量。少計可能改變 kernel 選擇；輸出的 density 仍適用同一精度預算。以均勻雜湊估算，一百萬個不同組合中至少有一對 ID 相撞的機率上界約為 5×10⁻⁸。這是 ID 碰撞的上界，不是用量少計的機率。

## 被否決的選項與理由

- Engine 新增 row-provenance API：FTF 已能用既有點 ID 和 caller-owned metadata 定位，不需要把 FTF 的來源資訊放進 Engine。
- 只留批內 ID：數值相同的不同原點仍會少計，錯誤和 capture 也找不到原點。
- FTF acquire 跨輪保留對照：多一份隨輪次與 history 成長的狀態，卻查不到借用同一 routing history 的跨 acquire 碰撞。
- Routing history 的 owner 保留對照：可以涵蓋跨 acquire，但需要擴張 Engine 或新增共享 owner。已知的剩餘風險只影響攤提計數，不值得加這份狀態。
- 有界可逆編碼：為了塞進 int64 而限制投影次數或點數，沒有物理理由。

## 留下的問題

觀察到跨 preparation 的碰撞時，重開作用域的選擇。Engine 若開始用點 ID 識別數值結果或 cache，碰撞就不再只是用量問題，也要重開。若一份 routing history 累積的不同組合接近 2³¹，應重新估計碰撞風險。

## 依據

- [ADR-0078〈黑盒邊界是 simulate.evolution〉](../adr/0078-evolution-module-boundary.md#黑盒邊界是-simulateevolution) 把用量和選路交給 Engine。FTF 只供應列 ID，不維護另一份用量紀錄。
- [ADR-0078〈優化優先於周邊約定；除錯靠 dump，不靠重算〉](../adr/0078-evolution-module-boundary.md#優化優先於周邊約定除錯靠-dump不靠重算) 要求用 capture 留下定位資訊。列來源放進 FTF 的 manifest metadata，不改 Engine codec。
- [ADR-0078〈歷史控制與精度驗收〉](../adr/0078-evolution-module-boundary.md#歷史控制與精度驗收) 允許歷史改變 kernel 選擇，用同一數值精度預算驗收。
- [FTF environment README](../../lib/zcu_tools/simulators/ftf/environment/README.md#cavity-drives) 說明現行續算與診斷契約。[共用 debug capture 文件](../../lib/zcu_tools/simulators/README.md#debug-capture) 說明 manifest 與發佈方式。代裁所依據的原始工作紀錄未入庫；這裡保留比較過的方案和理由。
