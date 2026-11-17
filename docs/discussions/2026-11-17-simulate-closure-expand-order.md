---
date: 2026-11-17
status: active
superseded_by: null
related:
  - ../adr/0072-module-boundaries.md
  - 2026-11-15-simulator-closed-submodule.md
basis:
  repo_commit: c927e635e8a35a4b30081539793bcc6f2061d5a6
  historical_repo_commit: f13505159ed2b5280032cdacff04f8b7710f0f8b
  external_sources: none
---

# 封閉 simulate 的搬遷順序

## 結論

Expand 放在 contract 允許的方向。先在新位置 `simulate/boundary.py` 建 19 行 stub，轉出舊定義，也就是 C19 允許的 `simulate → core`。分兩批改 importer，最後一票才把定義搬進 `simulate`、刪舊檔、加 `simulate-closed`。Stub 只在 task integration 存在，main 從未出現 re-export。封閉的長期規則由 [ADR-0072](../adr/0072-module-boundaries.md#模擬器是封閉的子模塊模型由使用方聲明) 擁有，原則的討論見[模擬器封閉成子模塊](2026-11-15-simulator-closed-submodule.md)。

## 問題

原規劃是在新位置定義、舊位置轉出，形成 `core → simulate`。C19 禁止 core import simulate，且 ignore_imports 只減不增。照原順序，第一票到刪舊檔前 lint-imports 都會失敗；每票 gate 都要全綠，就收不了票。

在 main `f13505159ed2b5280032cdacff04f8b7710f0f8b` 量到：

- Importer 有 105 檔、118 條 import。Lib 37 檔包含 experiment 2、program 9、simulate 8、simulators 18，另有 tests 65 檔與 zcu_lab 3 檔。
- 邊界模組有 19 個公開名稱，沒有私有名稱。兩個測試用 `from zcu_tools.core import simulation`，測試對象就是邊界 module。
- C8 把 core 與 simulate 放在同一列，冒號表示同層可互 import，所以 `simulate → core` 不違反 contract。
- `simulate/__init__.py` 只 import numpy。Repo 內沒有以模組路徑持久化這些型別。

## 考慮過的選項與比較

| 欄位 | Keep，原順序 | A，反轉轉出 | B，單票切換 | B′，cache 一票加切換一票 |
| --- | --- | --- | --- | --- |
| 改什麼 | 不改票 | 先收 cache root、19 行新位置 stub 與 simulate 內 8 檔；再改 32 個非測試 importer 與 65 個測試檔；最後用原 836 行換掉 stub、刪舊檔、加 contract 與文件 | 第一票吸收後三票，git mv 加腳本改 118 條 import、contract 與文件 | 第一票只收 cache root；第二票吸收邊界搬移、105 個 importer、contract 與文件 |
| 驗收改動 | 無 | 第一票只允許 boundary.py → core.simulation 這一條對外依賴，完整封閉移到最後一票，暫時轉出反向 | 第一票合併原後三票驗收 | 第一票刪封閉項，第二票承接全部邊界搬遷驗收 |
| 收票單位／依賴深度 | 4／3 | 4／3，中間兩批可平行 | 1／1 | 2／2 |
| Gate 範圍 | 不變 | 前三票各跑 affected，最後一票跑全庫 pytest 與全樹 Pyright ratchet | 一次全庫 pytest 與全樹 ratchet | Cache 票已有綠證據，切換票跑全庫 |
| 故障定位 | 不變 | 每票只碰一批檔，收票核對範圍小 | 一票約 110 檔，失敗時整票重看 | 同 B，cache 已隔開 |
| 一次性程式 | Core 轉出 stub | 19 行 stub，最後一票刪除 | 無 | 無 |
| 失敗模式 | 刪舊檔前 lint-imports 必敗 | 中間狀態的 simulate 尚未封閉；最後從 stub 換全文，當時未驗證 Git 是否視為 rename | 未實測單 lane context 能否放下約 110 檔，腳本可降低負荷 | 同 B |

另比較了 A′，按 package 把 importer 切成 5 票，總共 7 個收票單位、依賴深度仍為 3。C 是放寬 C19 或加 ignore_imports。還有一案是新舊位置各放一份 dataclass。

先例立場：

- 遵循 ADR-0072、C19 與 ignore_imports 只減不增。
- 遵循封閉原則。原則只規定終點，不要求每個中間 commit 都封閉。
- 遵循使用者 2026-11-16 核可的分批實作計畫，當時寫的是「99 個 importer 改 import（37 lib、62 tests），分批」；重新量到的 105 檔不改分批決定。
- 區別先前 owner 搬遷的使用者確認「caller 改用新 owner，舊路徑不以 re-export 保留」。那管的是 main 上留給 caller 的長期路徑；本案 stub 在新路徑，只活在 integration，終點刪除。
- 遵循 expand → 分批 migrate → contract 的 wide refactor 做法，expand 要服從現有 import contract。
- 區別三個單票切換先例：sim-framework 的 136 檔切換 `4847bd319accbbc4e84383ca28bfd1c05252c5e6`、WaveformSamples 的 31 檔切換 `cf662fcd49b795527687b5c18924849deab63ef6`，以及附「放不下才分批」退路的一次中性 metadata 搬遷。它們沒有使用者核可的分批計畫，前兩者也沒有擋住舊位置轉出的 contract。
- 沒有找到「新位置轉出舊定義」的直接先例。

A 的成本是 19 行一次性 stub、4 個收票單位、依賴深度 3，順序為第一票 → 兩批 importer 平行 → 最後一票，與原規劃相同。原放在第一票的 836 行搬移改到最後一票。

## 被否決的選項與理由

- Keep：lint-imports 必敗。
- B、B′：偏離使用者核可的分批計畫，單 lane 能否放下約 110 檔也未實測。
- A′：多 3 個收票單位，review 沒變小。
- C：放寬 C19 會推翻有拘束力的先例，只有使用者能決定。
- 兩份 dataclass：新舊 importer 混用時，isinstance 與物件身份會分裂。

## 留下的問題

當時未驗證 Git 是否把最後一票的搬移偵測成 rename，這不影響驗收，不把歷史版面當作行為契約。

## 依據

歷史量測取自 frontmatter 的 historical_repo_commit。原始的兩份角度報告、提案與 challenge 未入庫；本文保留了各選項的 gate、成本、失敗模式與先例立場。

## 後記

2026-11-17 升級文件時補記：最後一票把定義原樣搬入，與 main 原檔 bytes 相同。Reconcile main 時新進的 `core/simulation.py` docstring 更新也帶入，boundary 共 839 行。本文核對的實作 commit 是 `c927e635e8a35a4b30081539793bcc6f2061d5a6`，尚未合併到 main，不把它記為 main 落地 SHA。
