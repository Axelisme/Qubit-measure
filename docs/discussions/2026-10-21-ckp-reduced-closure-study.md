---
date: 2026-10-21
status: active
superseded_by: null
related:
  - zcu_lab/simulation_envs/FTF/field-guide.md
  - lib/zcu_tools/experiment/ftf/README.md
  - docs/discussions/2026-10-16-ftf-simulation-speedup.md
  - docs/discussions/2026-10-17-ftf-dyson-guard-and-classification.md
basis:
  - "repo@14c13fbea：半古典 finite-drive 腔模型 partial land，CKP 對完整 Lindblad 的驗收維持未通過"
  - "研究工作檔（歷史項分解、候選 closure、成本計時）未入庫；結論與必要數字寫在本文"
---

**讀者：** framework-dev。記錄 FTF 模擬腔驅動模型為何停在半古典近似、哪些 reduced closure 已被量化否決，以及接下來為何走 joint Fock 性能研究。

# CKP reduced closure 研究：精度過但成本 42 倍

## 結論

目前沒有同時滿足精度與成本的 reduced closure。Dense displaced-Fock（DF）closure 在總激發數 cutoff N=2 時六點 Pe 差 0.00144，低於 shot 標準差 0.0040，但成本是現行半古典的 41.95 倍，2,550 點 CKP map 外推 4,466 s（目標 200 s）。單 Gaussian closure 因 coherence block 的 trace 零點沒有有限表示，否決。保存的 joint qubit-cavity Fock Lindblad reference 全圖 714 s，比 DF N2 便宜。

使用者 2026-10-21 裁定：先以半古典模型 partial land（「我認為可以先partial land，有真實物理對應的算法獨立研究，不擋其他task」），研究結束後選 joint Fock 性能研究（「2」）。半古典模型的已知偏差與適用條件寫在 [field-guide 的 Cavity 段](../../zcu_lab/simulation_envs/FTF/field-guide.md#cavity)。

## 問題

CKP（conditional phase 校準）在 FTF 模擬上的驗收要求半古典腔模型與獨立的兩能階 × 兩腔 Fock Lindblad 在同 κ、χ、drive、Γ1/Γ2、時序與溫度下中心差落在 scaled SE 內。實測 Run 中心 −23.6 kHz，reference −134.6 kHz，差 110.9 kHz，為 18.57 倍 scaled SE。診斷確認這不是實作 bug，而是模型不等價：

- Spec 的 Stark 頻移是 Σ(χe|αe|²−χg|αg|²)，Lindblad 的 coherent-overlap 相位是 Re Σ(χe−χg)αe*αg，兩者不相等。
- Spec 的外線 dephasing 係數是 Lindblad 的一半。
- 只取消段平均仍留 0.03 至 0.04 的局部 Pe 差。

問題變成：保留哪些 finite-drive 腔歷史，才能同時達到 Lindblad 精度與 ≤200 s 的 map 成本。

## 哪些腔歷史必須保留

用保存的單 rf、三個 q、兩種 preparation 共六點，對 Lindblad reference 做受控變體（reps 4096 的逐點 shot σ 為 0.0040 至 0.0053）：

| 歷史項 | 控制定義 | 最大 ΔPe | 超過 σ 的點 | 判定 |
| --- | --- | ---: | ---: | --- |
| Qubit flip 留下的非真空腔激發 | Γ1=0 的 exact 減 polaron-vacuum | 0.0374 | 5/6 | 必須保留 |
| Jump 後的腔 transient | full 減「jump 時腔重置到 pointer」 | 0.0000072 | 0/6 | 六點門檻下可捨棄 |
| 有限驅動高階項 | full 減 Ω² 弱響應 | 0.0062 | 2/6 | 不能只留 Ω² |
| Γ1 布居演化 | full 減 no-jump exact | 0.0057 | 不作獨立判定 | 必須保留 |

三項重疊，不可相加。中心差只有單 rf 三 q，無法識別五參數 line fit，只能給解析尺度（約 −12.5 kHz 與 +6.1 kHz 的單事件相位代理），不能宣稱 CKP 中心契約通過或失敗。

## 考慮過的候選

| 候選 | 保留的歷史 | 六點精度 | 成本倍數 | 2,550 點外推 | 狀態 |
| --- | --- | --- | ---: | ---: | --- |
| Polaron-vacuum projection | coherent cross phase、兩 port dephasing、瞬時 overlap | 最大 ΔPe 0.037，中心不可辨識 | 1.19 | 未外推 | 否決：丟掉 flip 非真空激發 |
| DF N1（n1+n2≤1） | flip residual、完整 Ω、Γ1、jump transient | 最大 ΔPe 0.0089（1.95σ） | 10.7 | 1,118 s | 精度與成本都不過 |
| DF N2（n1+n2≤2） | 同上 | 最大 ΔPe 0.00144（0.36σ） | 41.9 | 4,466 s | 精度過，成本不過 |
| DF N3（n1+n2≤3） | 同上，cutoff 診斷 | 最大 ΔPe 0.0002 | 143.3 | 15,236 s | 只作 cutoff 收斂診斷 |
| 單 Gaussian closure | mean、covariance、flip/jump source | 無法定義 | 未定義 | 未定義 | 否決 |

DF 的做法：兩條 no-flip pointer 從 cavity onset 連續演化，以 displacement 把密度矩陣移到 pointer 框架後投影到總激發數 ≤N 的 Fock 子空間，再演化有限維 Lindblad。投影後的 displacement 不再 unitary，Γ1 通道的 tail 也被截斷。Dense 顯式 DOP853 的瓶頸是 off-resonant 第二腔帶來的 690 rad/µs 最大虛頻率；interaction picture、implicit 或 sparse solver 未實作。

成本比較的分母是固定 commit 的 16 能階半古典 production 入口，分子是兩能階 DF，同一 heavy-slot 單核、BLAS=1、三輪 median。這不是同維度 solver microbenchmark。

## 被否決的方案與理由

- **Polaron-vacuum projection**：六點 Pe 差 0.037，與 Lindblad 的差距主要來自丟掉 flip 產生的非真空腔激發；擴樣不會修復缺口。
- **單 Gaussian closure**：coherence block ρ_ge = c(|α⟩⟨α|−|β⟩⟨β|) 的 trace 為零而一階矩不為零，單一指數型 Gaussian 無法表示；ground 初態的 coherence trace 本來就是零。沒有為它補 epsilon 湊 timing。
- **DF N2 性能研究**：起點 41.9 倍，比 joint Fock 的 714 s 更不利。
- **接受半古典限制**：使用者未接受以現有精度收尾。

## 留下的問題

- Joint Fock 路線能否從 714 s 壓到 200 s 級：sparse Liouvillian 加 Krylov、每 rf 共用 preparation、自適應 cutoff、16 能階擴充的成本都未量。
- 其他 rf、完整 q 線形、κ 與 fit-window 穩定性、其他溫度與強驅動都未驗證；六點 Pe 通過不等於 CKP 中心契約通過。
- Jump transient 在六點門檻下可捨棄，但更嚴的門檻或其他條件下是否仍可捨棄未知。

## 依據

- 研究於 2026-10-21 在固定 commit 上以三條平行 lane 完成（歷史項分解、候選推導、成本估計），工作檔與 JSON 輸出未入庫；本文數字取自其綜合報告。
- 半古典模型與 Cavity 段文字：[field-guide](../../zcu_lab/simulation_envs/FTF/field-guide.md#cavity)、[FTF engine README](../../lib/zcu_tools/experiment/ftf/README.md)。
- 先前的速度工作：[FTF 模擬加速](2026-10-16-ftf-simulation-speedup.md)、[Dyson guard 與分類](2026-10-17-ftf-dyson-guard-and-classification.md)。
