---
date: 2026-11-20
status: active
superseded_by: null
related:
  - 2026-11-20-timeline-forward-jumps.md
  - 2026-11-20-rb-clifford-barrier.md
basis:
  repo:
    previous_program: c333da7d8
  external: []
---

# 板上 RB 程式改寫

## 結論

Plain 與 heralded two-qubit RB 每個 seed、arm 編一支板上 depth program。Run 在取數前編完所有 program，PMEM 超量就報錯。每個 seed 只抽一次最深序列，兩臂與各 depth 共用其 Clifford 前綴。Recovery 放在對應前綴之後，只從該處跳一次到共同讀取。最深點直接讀取。

兩版 RB 都在 cfg 驗證時把 depth、seed 去重排序，讓保存的 cfg 對齊資料軸。兩版 RB 與 CZ 形狀優化都新增兩支校準 X180。各 Clifford 的 index 不變，分解先最少 CZ，再最少實體 X 脈衝。CZ 形狀優化沿用同一個序列產生方法，但只播放一個固定 depth，不加跳轉。

2026-11-20，使用者看過第三版草稿與 X180 建議後答「其他采用建议」。現行取捨規則見 [ADR-0083](../adr/0083-program-macros-and-timelines.md#量測能力不抵可讀性成本時不加分支)。

## 問題

上一版把共用前綴寫在前面，recoveries 集中寫在後面。較淺的 shot 要跳到另一段打 recovery，再跳到共同讀取。PMEM 超量時，主機按 depth 對半重編。取數要先迴圈走分塊，再把資料寫到原 depth 軸的 slice。使用者說「我觉得你这一版的 RB 的实作可读性不高，我看不太懂」。

## 考慮過的選項與比較

- **就地播放 recovery。** 使用者問「为什么不能在 Clearfield 的中间直接插入 recovery，然后跳转一次到 real？」（Clearfield 指 Clifford，real 指 readout）。只有 depth 索引相符的 shot 播放 recovery。其餘 shot 跳過 recovery，接著播前綴。進入讀取前，各路徑用 `delay_by_timeline` 同步到自己的結尾。這沿用 [ADR-0083](../adr/0083-program-macros-and-timelines.md#往後跳轉與-label-合併) 的合流規則。延續 [Clifford 邊界對齊](2026-11-20-rb-clifford-barrier.md)：每段 Clifford 結束時兩個 qubit 都要等到較晚的一條線。
- **每 seed 抽一次，而非每個 depth 重抽。** 使用者說「应该是每一个 seed 先决定，呃，最深的 clearful gate 序列。然后再用这个序列去生成 pair sequence」。舊法重建各 depth 的序列，結果仍是同一個前綴。直接抽最深序列保留既有 `random.Random` 的 index，也只走一次每臂的累積乘積，不必事後把獨立建出的序列對回最深的一條。
- **將 X180 當成一支實體脈衝。** 使用者說「你只需要 x 九十吗？你不需要 x 一百八吗？我记得 single qubit 两个都给吧。这样的话，你就不用两个 x 九十去等效一个 x 一百八了。」維持原本 X90／Z90／CZ 的 BFS index，只在最短分解中加入 X180。X90 和 X180 都依設定相位減掉 virtual-Z frame。
- **Cfg 接受重複和亂序。** 使用者說 depth「在 Run 的前面呢，先把它 Unicode 跟排序一下就可以了」（Unicode 指 unique），再說「重复的 seed 也是比照办理」。在 cfg 欄位驗證時做，不只在 Run 的區域變數做。若只改迴圈，保存的 cfg 仍有重複或順序不同，資料軸就不相符。

## 被否決的選項與理由

- **PMEM 放不下就對半切。** 現有稀疏網格以合成脈衝、無 reset、4096 words 的 PMEM 離線編譯，21 點不分塊估算最大 depth 約 460，分塊約 490。其他稀疏網格的提升約 15–60 個 depth。這不足以保留重編譯迴圈和 Schedule 的 slice 寫入。使用者說「拿掉粉块，它带来的效益没有比可读性的破坏来得多」。
- **保留集中 recovery，或在執行期查表算 recovery。** 集中寫法需要從前綴跳出去再回到讀取。查表要另處理讀取位址和相位。現有靜態前綴已知各點要播放的 recovery，只保留用到的 gate，放在實際執行的位置。
- **舊 cfg 缺 X180 時回退到兩支 X90。** 新 cfg 需要獨立校準的 X180，沒有預設值或遷移。Plain RB、heralded RB、CZ 形狀優化的 cfg schema major 分別升至 6、3、2。這延續 [ADR-0069](../adr/0069-experiment-core-and-frontends.md#cfg-欄位改名與-schema-版本) 對不相容 cfg 的處理。舊 Results 與 metadata 在同樣可讀時照常 Load，`CfgUnavailableWarning` 指出 cfg 無法還原，`RunRecord.cfg` 為 None。需要 cfg 的分析與寫回不會用現在的預設值繼續。

## 留下的問題

尚未量測實板的 tProc latency。合成脈衝的 PMEM 用量不能替代以自己的 gate 與 reset 設定編譯。若稀疏取點仍達不到實際量測需要的深度，再比較增加掃描範圍和維護額外路徑的成本。

## 依據

- 前一版 program 與 cfg 以 repo `c333da7d8` 為基準。[普通 RB guide](../../zcu_lab/v2/two_qubit/rb/guide.md#sequences-and-cost) 描述新 program 與操作方式。
- 同一份無 reset 合成設定、seed 7 下，201 點 depth 0–200 的 reference／interleaved 用量從 3750／4145 降到 3728／4114 words。21 點 depth 0–400 從 2803／3612 降到 2777／3586 words。兩組條件都以記憶體內提高上限的方式取得超量用量。
- 原始草稿與編譯量測工作檔未入庫。上文保留裁決、條件與比較數字。

## 後記

2026-11-20，[RB 取數歸約](2026-11-20-rb-round-reduction-and-fit-quality.md)保留本文的板上 depth program，改由 acquire 加總各程式已完成的 rounds。兩個 two-qubit RB Result 都不再保存 round 軸，Load 不接受帶 round 軸的舊檔。
