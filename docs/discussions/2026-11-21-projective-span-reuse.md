---
date: 2026-11-21
status: active
superseded_by: null
related:
  - ../adr/0078-evolution-module-boundary.md
  - 2026-11-21-mist-cavity-reuse.md
  - ../../lib/zcu_tools/simulate/adc/README.md
  - ../../lib/zcu_tools/simulators/ftf/environment/README.md
  - ../../lib/zcu_tools/simulators/ftf/physics/README.md
basis:
  - "本 repo main f05b15d9f，加上 projective-span-reuse 的落地候選 7c82190c8"
  - "計數探針在 a42f7945d；容量矩陣在候選 ed770fb56；真實呼叫者驗收比較 afbed1182（改前）與 aef142668（改後）"
---

# Projective 腔場改走 ADC flat span 合併解

## 結論

FTF 的 projective continuation 改用 ADC 的 `CavityAverageCache.span`，與非 projective（MIST）continuation 走同一條路。一次求解得到 flat 二階矩與各 state 端點場，只在 ADC 的 exact key 命中時重用。FTF 仍逐 history 選態、建 Stark／dephasing shift，並把選中的場傳給下一段。FTF 不合併 histories，不改投影權重或主分支選態。這推翻了 ADR-0078〈算法內聚，共用數值有單一 owner〉原本的邊界句「projective 只重用既有的 flat 二階矩，不收各 state 端點場」，由使用者在 2026-11-21 裁定。

兩種 FTF composition 共用一個 512 MiB 的容量常數，每次 composition 結束清空。ADC 全庫預設仍是 64 MiB。`CavityDrive` 的建構子只留 `cache=` 一個 ADC owner 參數，移除 `span_cache=`，不留別名。容量的選法與兩路共用是 designer 依使用者授權（「我授权你的范围除了 push 还有真实硬体之外都可以做」）代裁；移除 `span_cache=` 是 task 依 repo 規則的判斷。兩者使用者都未覆核。

三個固定前綴的真實呼叫者在改前改後同 seed 的 density 逐元素相等，獨立端點解歸零，`coherent_step` 少 66%–80%。這次沒有量牆鐘，也沒有涵蓋完整 acquire。

## 問題

FTF 模擬器在 main `3d73c92af` 的排他重測批次中，`chain/amp_rabi`、`chain/len_rabi`、`twotone/ro_optimize/chain_freq`、`chain_power` 與 `chain_length` 1.2 µs 五列在 360 s 逾時並送出取消後，30 s 內沒有停下。五列當下的 stack 都在 projective 路徑的腔場計算。`chain_length` 2.2 µs 已記錄的讀出是 358.502 s，演化只有 4.028 s。

當時 projective 路徑的 `CavityDrive.advance` 分開求解兩種產品。二階矩走 ADC 的 average memo，端點場每條 history 都用 `cavity_fields` 重解。非 projective 的 MIST 路徑已用 `CavityAverageCache.span` 一次得到兩者，見 [MIST 腔場重用](2026-11-21-mist-cavity-reuse.md)。`chain/amp_rabi` 與 `chain/len_rabi` 的前綴中，二階矩 memo 命中 434 與 435 次，但每列仍重解端點。ADR-0078 寫明 projective 不收端點場；MIST 的紀錄只寫 projective 路徑沿用既有的二階矩 memo。兩處都沒有寫排除的理由。

各 history 的初始腔場不同，MIST 的命中率不能直接移用。改動前先以計數探針量 projective 的重複。探針在 `a42f7945d` 以 64 MiB 容量旁路呼叫 `span`，只計數，不接產品路徑。

| 前綴（point 0） | exact 命中／span 呼叫 | 跨 history 重複 | 獨立端點解 | `coherent_step` | `expm` |
| --- | ---: | ---: | ---: | ---: | ---: |
| `chain/amp_rabi`，Engine ordinal 0–5 | 434／1,331（32.6%） | 434 | 1,331 → 0 | 2,712 → 922 | 282,024 → 278,444 |
| `chain/len_rabi`，ordinal 0–5 | 435／1,334（32.6%） | 435 | 1,334 → 0 | 2,718 → 924 | 282,636 → 279,048 |
| `chain_length` 2.2 µs，ordinal 0–4 | 5,686／29,921（19.0%），拒收 22,744 | 19,346（64.7%） | 29,921 → 0 | 108,228 → 48,470 | 14,732,256 → 14,637,940 |

端點傳播的重複可以消掉，但 `expm` 只少 0.64%–1.27%。`_flat_integral` 裡的 `expm` 才是腔場成本的大宗。`chain_length` 的跨 history 重複有 64.7%，但 64 MiB 滿載後只命中 19.0%。

## 考慮過的選項與比較

**下一批做什麼。** 使用者答「6A」。

- A：projective 腔場的 span 重用（採用）。
- B：shaped Sambe 路徑。
- C：Floquet preparation。

**ADR-0078 的 projective 邊界句。** 使用者答「2S，我對於模擬器沒那麼多限制，好維護與模擬效率是唯二考量，其他都只是建議」。

- K：維持。projective 每列照舊重解端點。
- S：projective 也呼叫 ADC 的 flat span 合併解（採用）。Owner、容量與一次 composition 的壽命沿用 MIST 的做法。FTF 仍逐 history 選態、建 shift、傳場。只在 exact key 命中時重用，無法證明等價時分開求解、不收錄。

**容量。** 在 64、128、256、512 MiB 下量三個 projective 前綴。選最小的容量，使再加倍多出的命中少於跨 history 重複機會的 10%，且 process peak RSS 增量不超過 512 MiB。套用範圍有兩個選項：

- 只套 projective，MIST 的 `evolve_plans` 留 64 MiB。
- 兩路共用一個值（採用）。條件：選值大於 64 MiB 時，先用一個 MIST 真實呼叫者比較 64 MiB 與選值，process peak RSS 增量不超過 512 MiB 才共用，否則 `evolve_plans` 留 64 MiB。

**`CavityDrive` 的 owner 參數。** 舊建構子有兩個參數：`cache=` 供 average-only 路徑（projective 用），`span_cache=` 供 span 路徑（MIST 用）。統一後 average-only 路徑不再有使用者。

- 保留 `span_cache=` 作 `cache=` 的別名。
- 只留 `cache=`（採用）。它同時供 `average()` 與 `advance()` 的兩種產品。不傳時，`advance` 每次求解、不保留結果。Tracked caller 全部遷移到 `cache=`。

## 被否決的選項與理由

- K：每列重解端點是重複工作，ADR 與 MIST 紀錄都沒有寫排除的理由。
- B、C：本批不做，留作之後的候選。
- 容量只套 projective：兩個常數要兩套說法，日後讀者要追問為何不同。MIST 若撞到 64 MiB，也拿不到較大容量的命中。兩者只差一次輕量量測。
- 不量 MIST 就共用：會把 projective 的 RSS 條件外推到沒量過的路徑。
- 保留 `span_cache=` 別名：兩路已共用同一個 ADC owner，tracked caller 已遷移。Repo 規則在使用者未要求時不加相容邏輯。

## 先例與立場

- ADR-0078〈算法內聚，共用數值有單一 owner〉的邊界句「projective 只重用既有的 flat 二階矩，不收各 state 端點場」：使用者推翻（2026-11-21，「2S，…」）。依舊邊界寫的 projective 分開求解路徑在同一個 task 移除，ADR 條文同步改寫。
- [MIST 腔場重用](2026-11-21-mist-cavity-reuse.md)（designer 代裁，使用者 2026-11-21 追認「其余的第二批追认。」）：遵循。理由相同：ADC 是單一 owner，FTF 不組 key，只用 exact key，壽命是一次 composition。區別在命中率：各 history 的初場不同，所以先量 projective 自己的計數。
- 2026-10-21，使用者對二階矩 memo 答「前兩個採用建議」：遵循。ADC 擁有 lossless key 與有界容量。
- [ADR-0078〈先保證正確，再優化瓶頸〉](../adr/0078-evolution-module-boundary.md#先保證正確再優化瓶頸)：遵循。改動針對有耗時紀錄的列，驗收要求同 seed 輸出相等。
- [ADR-0078〈優化優先於周邊約定；除錯靠 dump，不靠重算〉](../adr/0078-evolution-module-boundary.md#優化優先於周邊約定除錯靠-dump不靠重算)：遵循。相容性配合優化調整，所以 `CavityDrive` 不留舊參數。

## 結果

**容量。** 候選 `ed770fb56` 只替換 FTF 交給 ADC 的容量。每格用新 process，經非排他 `tools/run_heavy.py` 執行。RSS 取 `getrusage` 的 process 峰值，包括依賴、Engine、ADC 與探針，不全歸因於 memo。

| `chain_length` 2.2 µs | 64 MiB | 128 MiB | 256 MiB | 512 MiB |
| --- | ---: | ---: | ---: | ---: |
| exact 命中 | 5,686 | 8,284 | 14,392 | 19,346 |
| 拒收 | 22,744 | 18,654 | 9,563 | 0 |
| peak memo（B） | 67,080,839 | 134,206,547 | 268,412,998 | 475,546,067 |
| peak RSS 相對 64 MiB | 0 | +64.00 MiB | +192.23 MiB | +384.99 MiB |

跨 history 重複機會是 19,346 次，10% 是 1,934.6 次。三次加倍各多 2,598、6,108、4,954 次命中，都高於門檻。512 MiB 收下全部 10,575 個不同 key、零拒收，再加倍不會增加這個前綴的命中，所以沒有量 1,024 MiB。`chain/amp_rabi` 與 `chain/len_rabi` 在四個容量都是 434／1,331 與 435／1,334 次命中、零拒收，RSS 變動在 ±2.70 MiB 內。

MIST 用完整的 `mist/flux_dep` Run，未縮小 sweep，共 101 個 composition、40,804 次 span。64 MiB 與 512 MiB 都是 20,200 次命中、20,604 次 miss、零拒收，peak memo 都是 8,991,628 B。512 MiB 的 process peak RSS 低 3.49 MiB，符合共用條件，所以兩路共用 512 MiB。

**真實呼叫者。** 改前 `afbed1182`、改後 `aef142668`，同 seed 123，輸入與計數探針相同。每場在最後一個納入的 `Engine.evolve` 返回後停止。

| 前綴（point 0） | 最後返回的 density 最大元素差 | 獨立端點解 | `coherent_step` |
| --- | ---: | ---: | ---: |
| `chain/amp_rabi`，ordinal 0–5 | 0 | 1,331 → 0 | 2,712 → 922 |
| `chain/len_rabi`，ordinal 0–5 | 0 | 1,334 → 0 | 2,718 → 924 |
| `chain_length` 2.2 µs，ordinal 0–4 | 0 | 29,921 → 0 | 108,228 → 21,150 |

逐 history 對齊後，batch、span、input rows、完成的 Engine family、fallback 與 CPU 位置都相同。`chain_length` 省下的 `coherent_step` 是計數探針的 1.46 倍，因為探針用 64 MiB，產品用 512 MiB。行為測試另驗 exact key 重複的 span：二階矩逐值相等，主分支端點與分開求解的差在絕對容差 1e-12 內。

## 留下的問題

- 結果只涵蓋三個固定前綴，不代表完整 acquire。後段 spans、readout 與 raw 沒有量。
- 前綴用的是舊版 `chain_length`。main 之後把 pulse length 改成 `window + length_margin`，這裡的 density、命中與 RSS 不外推到新版。
- `_flat_integral` 的 `expm` 仍是腔場成本的大宗，這次只少約 1%，另題處理。
- 沒有量牆鐘。只在 FTF 模擬器上量，沒有接硬體。

## 依據

- 使用者原話：2026-11-21 對下一批答「6A」，對邊界句答「2S，我對於模擬器沒那麼多限制，好維護與模擬效率是唯二考量，其他都只是建議」。
- 計數探針、容量矩陣與真實呼叫者驗收的收據與工作檔未入庫，必要數字寫在正文。
- 現行契約見 [ADC README](../../lib/zcu_tools/simulate/adc/README.md)、[FTF cavity drives](../../lib/zcu_tools/simulators/ftf/environment/README.md#cavity-drives) 與 [FTF cavity drive composition](../../lib/zcu_tools/simulators/ftf/physics/README.md#cavity-drive-composition)。規則見 [ADR-0078〈算法內聚，共用數值有單一 owner〉](../adr/0078-evolution-module-boundary.md#算法內聚共用數值有單一-owner)。
