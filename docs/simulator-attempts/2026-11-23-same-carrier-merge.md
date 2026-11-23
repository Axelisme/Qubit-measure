---
date: 2026-11-23
status: adopted
commit: 65a294445efebd3e8dd6e20110256cd623f0dab9
modules:
  - lib/zcu_tools/simulate/evolution
discussion: ../discussions/2026-11-23-same-carrier-merge.md
superseded_by: null
---

# 同頻多線摺成單 tone

**讀者：** framework-dev、agent。記錄 Evolution 同形摺疊的准入條件、收益與驗證範圍。

## 核心思路

Evolution 原本按 active drive slot 數選路。兩條同載波的 CZ drive 即使同形，也會進多 tone 路徑。現在 planner 只摺疊有號載波頻率、起點、時長與 `sample_ns` 全相同，且樣本互為複數常數倍的線。它把各線的 gain 與相位吸收進一對合成的 Hermitian I/Q 算符，把共同包絡交給現有單 tone family。取樣軸不同時不摺，即使插值後成比例。

`single_tone_gate` 可接複數包絡。當省略載波對角項的 trace-distance 上界超過原誤差預算時，該 family 拒收。這個上界只約束被省略的項，不代表整個 gate 的誤差。公開的 drive 與 capture 輸入不變。現行契約見 [Evolution README](../../lib/zcu_tools/simulate/evolution/README.md)。

## 實測

以下比較 `2b8f628c3` 與同形摺疊實作 `0d851985`。每場用相同輸入和 seed、CPU 單 worker，各暖機一次後量一次正式 Run。時間是該次 Run 的 process CPU，誤差欄是對齊的完整 density 最大元素絕對差，不是對參考解的誤差。

| 離線輸入 | Process CPU s，前 → 後 | 最大 density 差 | 路由變化 |
| --- | ---: | ---: | --- |
| Plain RB，depth `[0,4,8]`、seed 701、sim seed 123、2 reps、1 round | 16.401 → 0.892 | `2.39846e-3` | 54 列從 `common_frame` 轉到 `shaped_sambe` |
| CZ optimize 的 E0，24 態、raw gain | 3.108 → 0.140 | `3.73687e-3` | 32 列從 `multitone_gate` 轉到 `shaped_sambe`，16 列從 `multitone_rwa` 轉到 `single_tone_rwa` |
| Chevron 2×2，gg herald、100 reps、1 round | 143.688 → 143.713 | `2.65213e-5` | 64 列從 `multitone_rwa` 轉到 `single_tone_rwa`，128 列 `dop853` 不變 |

三場都在 `1e-2` 內。參考解補查只涵蓋 `2b8f628c3` 的第一個 plain RB 基線呼叫。對 fine production DOP853 的最大差是 `1.84269e-5`，coarse/fine 兩次參考解之間為 `3.13111e-7`。這不涵蓋摺後 RB 或其他兩場，也不是獨立求解器的驗證。

審查修正另用同一組 plain RB 輸入比對 `60fb524c` 與 `65a29444`。兩側 CPU 單 worker，暖機後各一次。兩次 evolve 的完整 density 最大差為 0，family 列數、14 個 round-stat 陣列和 3 個結果陣列都相同。修正後未重量 E0 或 Chevron。原始工作檔與收據未入庫。

### 前期原型

下列未入庫原型從 `Engine.evolve` 與 model source 接縫替換同形輸入。每側是一次 CPU 單 worker 的 profile process，與上面的正式實作驗收分開。

| 量測 commit 與輸入 | Process CPU s，前 → 後 | 最大 density 差 |
| --- | ---: | ---: |
| `3e86bba6d`，plain RB 的 reference 與 CZ-interleaved 兩臂、depth `[0,4,8]`、seed 701、2 reps、1 round | 27.693 → 1.731 | `2.398e-3`，2 次 evolve |
| `819180c9b`，FTF Chevron 2×2、gg herald | 202.452 → 203.914 | `2.652e-5`，26 次 evolve |
| `819180c9b`，CZ 候選 E0、24 態 | 4.889 → 0.250 | `3.737e-3`，2 次 evolve |

## 為何採用

同形摺疊在測過的 RB 與 E0 輸入省下多 tone 路徑的成本，且不需要改各 family 的輸入表示。沒有把收益外推到未摺的線。取捨見[同頻多線的討論](../discussions/2026-11-23-same-carrier-merge.md)。

## 留下的線索

- Chevron 整次 CPU 時間近乎不變。同形 CZ 轉到單 tone 後，不同載波的雙 tone 段仍走直接積分。下次先查這些段的成本，不用把同形 RB 的收益外推到 Chevron。
- 異形輸入保留原多 tone 路由。[異形同載波的 shaped Sambe 原型](2026-11-24-same-carrier-shaped-sambe.md)在誤差檢查被拒收，未合入。
