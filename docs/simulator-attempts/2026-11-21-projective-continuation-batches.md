---
date: 2026-11-21
status: adopted
commit: 3dd4f94b2
modules:
  - lib/zcu_tools/simulators/ftf/environment
  - lib/zcu_tools/simulate/evolution
discussion: none
superseded_by: null
---

# FTF 同點同 span 的投影續算合批

**讀者：** framework-dev、agent。查投影後如何減少 evolve 呼叫，以及合批測到了什麼。

## 核心思路

投影後的各條 history 要以各自的密度、腔場和 shift 續算。原本逐條呼叫 Engine。改成把同一掃描點、同一絕對 span 的就緒 histories 一起送進 Engine，保留各列身分，再按 ID 回填。下一個 span 仍要等前一段完成。這個改法少付批次固定開銷，不合併不同初態的密度。

## 實測

原排他 ABBA 比較固定的兩態、無驅動 DC 序列。逐列送的 8 次 evolve 改成批寬 2、2、4 的 3 次 evolve。只計 Engine 重播，不含 FTF 組批、回填或讀出。封存摘要有速度比較，但未找到兩側量測 commit 的原始收據，不在此引用時間比。原始工作檔未入庫。

另在 `666eb48ac` 用合成 FTF、seed 123 的四個離線輸入，比較逐列與同點同 span 合批的 family。組合為 reps 512／1536 配點數 1／2，合批寬度 24。四個清冊的 3,072 條帶驅動列都未換 family。其中兩個較高 reps 的清冊與較低 reps 的物理輸入 digest 相同，沿用實際選路的補錄。這是選路核對，不是精度或加速量測。

## 為何採用或否決

採用同點同 span 合批，見 `3dd4f94b2`。保留每列自己的數值輸入和輸出。批內列序號後來改成穩定的 FTF 列 ID，以免不同原點的數值都相同時少算 routing 用量；這是另行記錄的[列身分與診斷決策](../discussions/2026-11-21-ftf-projective-row-provenance.md#結論)。

## 留下的線索

用同一份固定的列清冊隔離呼叫開銷，再分開量真實 FTF Run。合批會改變 Engine 看到的同 key 列數，可能改變成本估價；若實際路由有變，需驗新舊路徑在帶 shift 的驅動列上是否都達精度要求。這與[投影後只續算各 DC 態一次](./2026-11-22-ftf-projective-markov.md#核心思路)是不同做法。
