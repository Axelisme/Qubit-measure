---
date: 2026-11-24
status: rejected
commit: 0f5d83887126c14d7b9af1a21520df44aacb8b41
modules:
  - lib/zcu_tools/simulate/evolution
discussion: none
superseded_by: null
---

# Shaped Sambe 的帶狀 Hermitian 全譜求解

**讀者：** framework-dev、agent。查閱帶狀原型的成本、否決原因，以及量測能支持哪些結論。

## 核心思路

原型把 `shaped_sambe` 的密集 Hermitian 矩陣改成下帶儲存，再用帶狀 Hermitian 求解器取全譜與全部特徵向量。重建 propagator 時，把譜相位改乘到較窄的中央輸入，不再乘到整組特徵向量。原型想省下矩陣儲存、求解與重建的成本，不改 cutoff、guard、dispatch、尾段策略或 constant Sambe。

## 實測

### 先用舊版定位成本

在 `06b5e1966` 用 block1 manifest `931fa4f599f4`、seed 123、離線 FTF 單 worker，對 `chain/amp_rabi` 與 `twotone/ro_optimize/power` 各做一次完整的 11 格 Run。用 py-spy 以 49 Hz 採樣 worker 的 Python 與原生呼叫堆疊，排除主執行緒。每場以包含 `Engine.evolve` 的 stack 為分母，同一函式在一筆 stack 中只算一次。這些比例不是 CPU 計時，巢狀呼叫的比例不可相加。

| 完整 Run | `Engine.evolve` stacks | `shaped_sambe._spectral` 內的 Hermitian eigensolve `scipy_zheevd_` | Single-tone gate builder `_rwa_gate._build_unitaries` |
| --- | ---: | ---: | ---: |
| `chain/amp_rabi` | 2,706 | 1,129（41.7%） | 0 |
| `twotone/ro_optimize/power` | 2,358 | 592（25.1%） | 1,116（47.3%） |

`chain/amp_rabi` 的主要演化成本指向 shaped Sambe 的 Hermitian eigensolve。`twotone/ro_optimize/power` 單場最大的是 single-tone gate 建構。兩場都有 Sambe 求解成本，所以原型只改這一項。這份舊版 profile 用來選改動對象，不是下表的改前基準，也不能用來分攤下表的耗時。

### 再用同基底配對計時

改前是 `addd6c8fb57a55bf36742ec01a5a87e51ca55d1a`，改後是同基底帶狀候選 `0f5d83887126c14d7b9af1a21520df44aacb8b41`。兩側用同一份 block1 manifest，SHA-256 為 `931fa4f599f4db8dc82ac9363cb1068ac6f94ce9286e6e1baec34545bce57a1f`，seed 123，離線 FTF、`time_scale=0`。各側從獨立的空快取開始，按相同順序跑五場。每側各用一個 CPU worker、一個 native thread，不採樣。每場各一次完整 Run，以 11 個 Result 格為分母，不用 acquire 點數。下表是單次配對的每格 core 秒數。

| 場景 | 改前 core s／格 | 改後 core s／格 |
| --- | ---: | ---: |
| `chain/amp_rabi` | 6.294 | 8.912 |
| `mux/amp_rabi` | 2.348 | 2.339 |
| `mux/len_rabi` | 1.091 | 1.099 |
| `twotone/rabi/len_rabi` | 0.101 | 0.123 |
| `twotone/ro_optimize/power` | 3.903 | 4.893 |

若其他階段的成本不變，即使把 evolve 降為零，`chain/amp_rabi` 的改前下限仍是 1.381 core s／格，改後是 1.400。`twotone/ro_optimize/power` 的改前下限是 0.544 core s／格，改後是 0.542。下限用同一批配對的 `(core_s - evolve_s) / 11` 計算，不是另一次測量。

同輸入、同 seed 比對兩側五場的 17 個結果陣列，最大元素絕對差為 0，低於 `1e-2` 的對照門檻。這只檢查候選與舊路徑是否一致，不是獨立精確度驗證。原型、候選測試、原始 profile 與成對收據未入庫。

## 為何採用或否決

兩個主要場景在這次配對中都變慢。不收錄帶狀候選，也不改既有 guard。單次配對不能把變慢歸給某個求解步驟，也不足以判定 mux 的小幅差異有無穩定收益。

## 留下的線索

這次只否決帶狀儲存加全譜求解的候選，不排除其他 Sambe 加速路徑。若再查 `twotone/ro_optimize/power` 的演化成本，先用當時的版本分開量 gate 建構和 Sambe 求解。若目標仍是 0.1 core s／格，這兩場也得降低 evolve 以外的成本：只把 evolve 歸零仍達不到目標。
