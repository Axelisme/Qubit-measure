---
date: 2026-10-17
status: active
superseded_by: null
related:
  - docs/discussions/2026-10-18-dyson-input-reuse.md
  - docs/discussions/2026-10-18-shaped-period-and-gates.md
  - docs/discussions/2026-10-18-two-tone-floquet.md
  - docs/discussions/2026-10-16-ftf-simulation-speedup.md
  - docs/adr/draft/simulation-environments.md
  - lib/zcu_tools/simulate/evolution/README.md
basis:
  - "repo@11f28b048：FTF 分段引擎落地（含 Dyson 門檻）"
  - "repo@f8193252c：模擬加速的設計決策寫入 ADR draft 與 README"
  - "repo@6848f3217：撰寫時的 main HEAD"
---

# FTF 分段引擎：Dyson 門檻、分類成本與事後檢查

## 結論

- 分類要輕量快速，不跑 kernel 的主要數值計算。使用者的理由：「分類要保持輕量，因為分類要做完才能開始批次計算」。
- 同一段不演化兩次。使用者：「把最慢的步驟算兩次真的很蠢」。
- Dyson 被拒時不重算。使用者：「就不重試了，誤差大就大吧，重試很浪費時間」。|Δp| 改成 apply 之後從結果 density 計算，超過門檻只記錄並發 warning，不保證。
- 治本的方式是分類時用便宜的解析指標，預先避開可能超標的段，並可評估其他算法。使用者：「我認為這問題治根要在分類時就用一些簡單指標來避開，也可以考慮引入其他算法」。
- 不用 Rust 重寫。瓶頸在 Numba 編譯的 Dyson 計算，不在 Python。
- Threading pool 隨每次 evolve 開關，不跨 acquire 保留。使用者：「不是瓶頸就沒關係」。
- 門檻公式：
  - |Δp| ≤ min(0.02, (1.5·budget)^(2/3))
  - r ≤ min(0.1, 0.25·budget^(1/3))

  兩個係數都是用合成模型掃描校準的，不是多能階的嚴格證明。

撰寫時，事後檢查與分類指標的 spec 尚在交使用者確認。

## 問題

1. sim-speed 的 land review 找到反例：共振、gain 0.04、τ = 1 ns 時，Dyson 的三個寫死門檻都通過，單段 coherence 誤差卻是 1.32e-3，超過每段預算 1e-3。
2. 使用者規模（301 flux × 2001 freq）的完整 acquire 約 1023 秒，其中 kernel 分類約 371 秒，在 caller thread 上單執行緒執行。

## 考慮過的選項與比較

### Dyson 門檻

| 嘗試 | 結果 |
| --- | --- |
| 可嚴格證明的餘項上界 exp(a)·a³/6 | 不看 detuning。遠離共振的點誤差只有 9.7e-6，上界卻是 2.9e6；所有 twotone 點被推到 T6，慢 119 倍 |
| 由預算推出 \|Δp\| 門檻 | twotone 不受影響（+0.45%）；3456 列掃描仍有 51 列超預算，最大 4.5e-3。population 轉移小，不代表 coherence 誤差小 |
| 再加由預算推出的 r 門檻（採用） | 誤差與 r³ 成正比。係數取 0.25 而非剛好壓線的 0.3，留約 1.7 倍邊際。掃描中三條路徑都在預算內；twotone 1 flux 慢 9.9% |

### 分類成本

可行性量測：分類樣本有 94% 落在 Dyson guard 的 Numba `_evaluate_batch`，只有約 2.2% 持有 GIL。

| 選項 | 結果 |
| --- | --- |
| Threading 分類 | 4 threads 分類快 2.49 倍，結果逐 bit 相同；完整 acquire 約快 1.28 倍 |
| Rust 改寫決策樹 | 消除全部 Python 也只快約 1.06 倍 |
| 事後檢查（採用方向） | guard 與 apply 對同一批 Dyson 段跑同一個 path 評估。把 \|Δp\| 移到 apply 之後，分類只剩 r、r₂。prototype：分類快 21 倍，1 flux 完整 acquire 快 1.55 倍 |

一開始否決「加上 Dyson 重用指數」，理由是整體只多約 3%。這個估計只算到 guard，沒算到 guard 與 apply 跑的是同一個函式。使用者指出後改變方向。

### 被拒的段怎麼處理

| 選項 | 結果 |
| --- | --- |
| apply 階段當場改走 RWA／T6 重算 | 要加 chunk 內 fallback、lazy prepare、跨 chunk 去重，實作重，也浪費時間 |
| 只報告、不重算（採用） | 超過門檻的段照用 Dyson 結果，記進 diagnostics，每次 acquire 發一次 warning |
| 完全拿掉 \|Δp\| | 失去可見性 |

## 被否決的選項與理由

- 保留嚴格上界、接受慢 119 倍：等於放棄 twotone 加速。
- 在 spec 註明 Dyson 每段誤差可到約 1.3e-3 或 4.5e-3：開預算例外，而且不能隨預算收緊。
- Rust 改寫：瓶頸不在 Python。
- 跨 acquire 保留 worker pool：threading 開一次約 10 µs，301 個 acquire 合計 3 ms；保留 pool 反而要管理關閉。
- 被拒的段在 apply 階段重算：見上表。

## 留下的問題

- 分類用的解析指標要多保守、會多拒絕多少段，待 prototype。已知前面有 flux 段時，要用「任意輸入態」的上界才不漏放。
- 被指標拒絕的段先走既有的 RWA → T6。若之後出現大量新 T6 key，下一個候選是共振小 block 的精確 rotating-frame 解，再加微擾修正。
- Dyson apply 的加速：剪 path、重用指數、蒙地卡羅評估，三者都在評估中。
- |Δp| 改成只報告後，`lib/zcu_tools/simulate/evolution/README.md` 的 Dispatch 與 Accuracy 節要改寫。

## 依據

量測數字來自 sim-speed 的票 19 驗收、land review 掃描與 sim-classify-speed 的可行性 prototype。原始報告在本機工作目錄，未入庫。已落地的門檻與流程以 `lib/zcu_tools/simulate/evolution/README.md` 為準。

## 後記，2026-10-18

上文的 371 s 分類成本、待確認 spec 與 RWA → T6 次序，是 2026-10-17 的狀態，不代表 main `f152eff67` 的完整決策圖。

後續採用了不演化 density 的解析 B2、實際共同前綴輸入與 contracted paths。不共用輸入的短 plateau 則比較 period 成本，詳見 [Dyson input reuse](2026-10-18-dyson-input-reuse.md)。apply 後 |Δp| 仍只報告，不重算。

「分類要保持輕量」不能解讀成所有 family 都不准 build-handoff。[Shaped preparation](2026-10-18-shaped-period-and-gates.md)與 [two-tone Floquet](2026-10-18-two-tone-floquet.md)記錄 guard 需要中間數值資料的原因，以及已建 payload 共用如何避免算兩次。那些 guards 的校準與未認證限制各自列出，沒有借用本文的 Dyson 門檻宣稱更強保證。
