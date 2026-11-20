---
date: 2026-11-15
status: active
superseded_by: null
related:
  - zcu_lab/v2/two_qubit/rb/core.py
  - zcu_lab/v2/two_qubit/rb/core_heralded.py
  - zcu_lab/v2/two_qubit/rb/scheduling.py
  - zcu_lab/v2/two_qubit/rb/guide.md
  - zcu_lab/v2/two_qubit/rb/guide_heralded.md
  - docs/adr/0083-program-macros-and-timelines.md
  - docs/adr/0074-flat-readout-module.md
basis:
  - "repo@90c6b48cd91f96d7c3069e3d1ebdf63316e8b7f9：原有RB序列、兩版core與timeline契約"
  - "Magesan et al., Phys. Rev. Lett. 109, 080505 (2012)"
  - "Gambetta et al., Phys. Rev. Lett. 109, 240504 (2012)"
  - "使用者2026-11-15裁決；原始工作紀錄未入庫，必要原話與結論寫在正文"
---

# Two-qubit RB 的兩組都同時打單 qubit gates

## 結論

使用者在2026-11-15選擇讓普通RB與heralded RB的reference、interleaved兩組都改成並行單qubit排程。每個qubit保持自己的gate順序，在CZ處共同同步。兩個core在同一個RB實驗資料夾共用這份排程，不跨實驗import。

這是既有[具名timeline契約](../adr/0083-program-macros-and-timelines.md)在實驗端的應用，不改framework介面。操作機制、日期與新舊資料比較限制由[普通guide](../../zcu_lab/v2/two_qubit/rb/guide.md)及[heralded guide](../../zcu_lab/v2/two_qubit/rb/guide_heralded.md)擁有。

## 問題

原core把兩個qubit的X90放在同一條default timeline，結果輪流播放。只有雙線CZ用fork／join同時播放。使用者問：「难道现在不是同时打吗？我预期他应该也是同时打。」

調查也區分了兩種比較。Interleaved RB比較reference與插入CZ的序列。Simultaneous single-qubit RB則比較單獨跑與同時跑的各自隨機序列。把前者的reference設成串行、interleaved設成並行，會同時改變CZ數量、串擾條件與閒置時間，不能只把差異歸給CZ。

## 考慮過的選項與比較

1. 兩組都保持串行。無需修改，但不符合使用者對同時打的預期，序列也較長。
2. Reference串行、interleaved並行。能觀察排程差異，但把不同問題混進CZ的interleaved比較。
3. 兩組都並行。每個qubit各自盡早開始，CZ等兩邊後共同開始，CZ後從共同結尾繼續。不同qubit的邏輯gate互相對易，同qubit順序與各自frame相位保持不變。

採用第三項。兩組共有的X90重疊與串擾不再是額外的排程差異，但這不保證所有串擾都能在error ratio中抵銷。

既有先例採取以下立場：

- 遵循ADR-0083。Fork複製end，join取max，不移動reference；CZ與終端readout等待共同邊界。
- 遵循[readout／heralding契約](../adr/0074-flat-readout-module.md)。本次只改gate排程，不改取數分類、herald篩選或可選reset。
- 遵循Magesan的interleaved比較目的。Reference與interleaved使用相同單qubit排法，保留各自recovery。
- 區別Gambetta的simultaneous RB。單獨對同時的比較是另一項實驗需求，不在本次新增。

使用者曾問：「作为 reference，可能分开打，然后待测 gate 同时打，是这样子吗？……你怎么看？」在上述區分後，提案是現在讓two-qubit RB兩組都同時打，simultaneous single-qubit RB等有需求才開。使用者答「A」。

Heralded部分一度暫停以確認保留範圍。使用者同日答：「那我认为如果可以取代 reset 的效果，那可以保留。单 Q Big Gate 的可以考虑加回来。」當次確認保留九個two-qubit heralded變體，rb_heralded一起修改。這不是heralding等同物理reset的驗證，也沒有授權改reset政策或加回單qubit實驗。

## 被否決的選項與理由

維持串行不符合使用者預期。只讓interleaved並行會混入另一種比較，故不採。沒有為這次排程新增framework API，也沒有跨實驗抽取helper；兩個現存RB core就是本地排程的consumer。

## 留下的問題

改動後的錯誤率不能直接與舊串行資料混比。離線編譯與probe只能驗排程、取數形狀與繪圖，不能驗收硬體串擾、物理fidelity或reset效果。

Simultaneous single-qubit RB與單qubit heralding是否另開，留待使用者指定需求，不從本次保留變體推定啟動。

## 依據

- [原序列產生與recovery](../../zcu_lab/v2/two_qubit/rb/sequence.py)不變；播放排程由同資料夾的[scheduling.py](../../zcu_lab/v2/two_qubit/rb/scheduling.py)擁有。
- [Magesan et al., Efficient Measurement of Quantum Gate Error by Interleaved Randomized Benchmarking](https://doi.org/10.1103/PhysRevLett.109.080505)，2012。
- [Gambetta et al., Characterization of Addressability by Simultaneous Randomized Benchmarking](https://doi.org/10.1103/PhysRevLett.109.240504)，2012。

## 後記

2026-11-20：本篇「Reference與interleaved使用相同單qubit排法」一句，在 Clifford 邊界上不成立。Interleaved 組每個 random Clifford 後的 CZ 讓兩條時間線在邊界對齊，reference 組沒有，相鄰 Clifford 可以重疊。使用者同日改為兩組都在 Clifford 邊界對齊，見 [Two-qubit RB 在 Clifford 邊界對齊兩條時間線](2026-11-20-rb-clifford-barrier.md)。

2026-11-20，使用者收窄本篇沿用的九個獨立 heralded 入口。RB 的衰減參數不受 gg、ee 準備誤差影響，因此 `rb_heralded` 保留；排程裁決不因這次取捨而變。見[只為共用參數保留獨立 heralded 實驗](2026-11-20-heralded-experiment-pruning.md)。
