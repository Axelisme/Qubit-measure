---
date: 2026-11-23
status: investigating
commit: 3e86bba6d
modules:
  - lib/zcu_tools/simulate/evolution
discussion: none
superseded_by: null
---

# 同頻多線摺成單 tone

**讀者：** framework-dev、agent。區分同形合併原型的收益與不同包絡的支援範圍。

## 核心思路

當時 Evolution 按 active drive slot 數分類。兩條同載波的 CZ drive 即使同頻，還是會進多 tone 路徑。若兩線同時起止、包絡只差一個固定複數倍，可把各線的 gain 與相位吸收進一對合成的 Hermitian I/Q 算符，再把共同包絡交給現有單 tone family。只共用載波並不夠。不同包絡通常需要保留多組 I/Q 算符和各自的包絡。

## 實測

未入庫原型從 `Engine.evolve` 與 model source 接縫替換同形輸入，不改 tracked 實作。下表是各輸入在同一 commit 上的原路徑與原型各一次、CPU 單 worker 的 profile process CPU 時間。不把這些對照當成後續實作的驗收。

| 量測 commit 與輸入 | Process CPU | 路由變化 |
| --- | ---: | --- |
| `3e86bba6d`，plain RB 與 CZ-interleaved 兩臂，depth `[0,4,8]`、seed 701、2 reps、1 round | 27.693 → 1.731 s | `common_frame` 的 54 列轉到 `shaped_sambe` |
| `819180c9b`，FTF `two_qubit/chevron`，2×2 gg herald、26 次 evolve | 202.452 → 203.914 s | 64 列從 `multitone_rwa` 轉到 `single_tone_rwa`，128 列 `dop853` 不變 |
| `819180c9b`，CZ 候選 E0 的模擬器呼叫，d=24、2 次 evolve | 4.889 → 0.250 s | `multitone_gate` 的 32 列轉到 `shaped_sambe`，`multitone_rwa` 的 16 列轉到 `single_tone_rwa` |

用相同輸入對齊演化輸出，兩條路徑的最大完整 density 差依表序為 `2.399e-3`、`2.652e-5`、`3.737e-3`。這是原路徑對原型的差，不是相對獨立參考的精度。原始工作檔未入庫。Frontmatter 的 commit 是首列 RB 的量測版本。

## 為何採用或否決

同形原型在 RB 與單個 E0 候選有收益，沒有加快這個 chevron。後續獲准做分階段方案 S：在 planner 保留載波群與 slot 的對照，把同形群摺成一個 slot，讓單 tone gate 接受複數包絡；不同形群先留原多 tone 路由，另做快速路徑原型。這份紀錄只涵蓋前期調查，S 的實作與驗收不在表中。

## 留下的線索

數值上能化成單 tone，不代表原本的主要成本會消失。先看原本哪些列走慢路徑，再比完整工作負載。對包絡不同的線，應在保留各線通道的前提下比較同一載波座標中的快速積分，不能從同形的數字外推。合併時仍要區分完整的通道身分，讓準備 key 和前綴簽章不會把不同的輸入視為同一列。
