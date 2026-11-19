---
date: 2026-11-20
status: active
superseded_by: null
related:
  - zcu_lab/v2/two_qubit/rb/scheduling.py
  - zcu_lab/v2/two_qubit/rb/sequence.py
  - zcu_lab/v2/two_qubit/rb/guide.md
  - zcu_lab/v2/two_qubit/rb/guide_heralded.md
  - docs/discussions/2026-11-15-two-qubit-rb-simultaneous-gates.md
  - docs/adr/0083-program-macros-and-timelines.md
basis:
  - "repo@95064aa80：2026-11-15 的並行排程，RBSequence 只提供攤平的 body"
  - "Magesan et al., Phys. Rev. Lett. 109, 080505 (2012)"
  - "使用者 2026-11-20 裁決；原始工作紀錄與驗收檔未入庫，必要原話、時長數據與結論寫在正文"
---

# Two-qubit RB 在 Clifford 邊界對齊兩條時間線

## 結論

從 2026-11-20 起，普通與 heralded two-qubit RB 的 reference、interleaved 兩組，都在每個 Clifford 結束時對齊 F1、F2 兩條時間線，recovery 之前也對齊。Clifford 內部仍然並行，CZ 仍然同步兩條線。

這修正了 2026-11-15 並行排程的一部分。原本每個 qubit 各自盡早開始，而且可以跨過 Clifford 邊界；現在不能跨過。

Interleaved 組的 pulse 時間不變，reference 組變長。以 seed 2 為例，reference 組的 shot body（含終端 readout）在 depth 1、2、4 分別多了 0.040、0.042、0.079 µs。

## 問題

討論在板上掃 depth 的跳轉寫法時，使用者看到示意碼的兩條時間線各自累計，提出（語音輸入原樣，「clear for gate」指 Clifford gate）：「对于一个 clear for gate 来说，它应该在。时间上是对齐的，跨 clear for gate 不会出现重叠。你的时速看起来可以重叠一样。」

查證後，現行程式確實如此。`emit_sequence` 只在 CZ 處同步兩條線，而 `RBSequence.body` 是攤平的 gate 序列，排程時已經沒有 Clifford 邊界。因此 Clifford k 結尾在 F1 上的 X90，可以和 Clifford k+1 開頭在 F2 上的 X90 同時播放。

這讓兩組不對稱：

- Interleaved 組在每個 random Clifford 後插一個 CZ。CZ 會 join 兩條線，所以這組其實每個邊界都對齊。
- Reference 組沒有這個 CZ，相鄰的 Clifford 可以重疊，總長較短。
- IRB 把兩組的差全部算給 CZ，所以 CZ 的誤差多算了一段「被迫對齊的閒置時間」。

2026-11-15 的紀錄寫「Reference與interleaved使用相同單qubit排法」，在 Clifford 邊界上並不成立。

## 考慮過的選項與比較

1. 兩組都在 Clifford 邊界加 barrier：每個 Clifford 播完就 join 兩條線，下一個 Clifford 從共同結尾開始。Interleaved 組的時序不變，只有 reference 組變長，兩組在邊界的對齊方式一致。
2. 維持 2026-11-15 的排程：reference 組較短，但兩組的對齊方式不同。

建議第 1 項，使用者回「都采用建议」。

對先例的立場：

- 部分推翻 2026-11-15 的[並行排程](2026-11-15-two-qubit-rb-simultaneous-gates.md)，由使用者本人裁決。Clifford 內部並行、CZ 同步、frame 與 reset 政策都保留，只推翻跨 Clifford 邊界的「各自盡早開始」。依舊排程寫的程式只有共用的 `emit_sequence`，這次一起改掉。
- 遵循 [ADR-0083](../adr/0083-program-macros-and-timelines.md)：join 取兩條線結尾的較大者，不移動 reference。
- 遵循 Magesan 的 interleaved 比較目的：兩組除了插入的 CZ，其餘排法相同。

## 被否決的選項與理由

維持 2026-11-15 的排程：兩組的邊界對齊方式不同，對齊造成的閒置會被算進 CZ 的誤差。

## 留下的問題

- 這次改動後的錯誤率，不能和舊資料直接混比，包括 2026-11-15 到 2026-11-20 之間用並行排程得到的結果。FTF 模擬上那次 CZ 量測的 IRB 4.1% 就屬於這一類。
- Clifford 內部仍有兩個 qubit 同時打 X90 的串擾。兩組都有這個串擾，但比較本身不保證它會抵銷。
- 時長增量只在 FTF 的編譯結果上量過，對硬體 fidelity 的影響要實際量測才知道。

## 依據

- 排程：[scheduling.py](../../zcu_lab/v2/two_qubit/rb/scheduling.py)。Clifford 分塊：[sequence.py](../../zcu_lab/v2/two_qubit/rb/sequence.py) 的 `RBSequence.body_blocks` 與 `clifford_blocks`；攤平的 `body`、`gates`、隨機抽樣與分解結果不變。
- 操作說明與新舊資料的比較限制：[普通 guide](../../zcu_lab/v2/two_qubit/rb/guide.md)、[heralded guide](../../zcu_lab/v2/two_qubit/rb/guide_heralded.md)。
- 驗收（工作檔未入庫，結論如下）：
  - 從編譯結果檢查：改動前，reference 組有 3 個案例在 Clifford 邊界重疊，例如下一塊在 tick 60840 開始，上一塊到 tick 64100 才結束。改動後沒有重疊。
  - Interleaved 組每個 pulse 的時間與讀出 trigger 的 tick，改動前後都相同。
  - FTF 上的普通 RB 改動前後都跑完，`sums` 都是 float64，形狀 `(4, 1, 2, 1, 3, 3)`。
  - Reference 組 shot body 時長（seed 2，含終端 readout，不含 relax delay，單位 µs）：

    | Depth | 改動前 | 改動後 | 增量 |
    | ---: | ---: | ---: | ---: |
    | 0 | 0.655227 | 0.655227 | 0 |
    | 1 | 1.024926 | 1.064453 | 0.039528 |
    | 2 | 1.062128 | 1.103981 | 0.041853 |
    | 4 | 1.399275 | 1.478330 | 0.079055 |

- [Magesan et al., Efficient Measurement of Quantum Gate Error by Interleaved Randomized Benchmarking](https://doi.org/10.1103/PhysRevLett.109.080505)，2012。
