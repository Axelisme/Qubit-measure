---
date: 2026-10-21
status: active
superseded_by: null
related:
  - docs/discussions/2026-10-21-ckp-reduced-closure-study.md
  - zcu_lab/simulation_envs/FTF/field-guide.md
  - lib/zcu_tools/experiment/ftf/README.md
  - lib/zcu_tools/simulate/evolution/README.md
basis:
  - "repo@14c13fbea：半古典 finite-drive 腔模型 partial land；joint qubit-cavity Fock Lindblad reference 為研究基準"
  - "研究腳本、計時原始資料與 reference 未入庫；結論與必要數字寫在本文"
---

# Joint Fock CKP 性能研究：保守 RK4 加 N+1 cutoff 外推 129 s，16 能階不可行

## 結論

兩能階 joint qubit-cavity Fock Lindblad 模型有一個候選同時通過局部精度與 200 s 成本目標：fixed RK4 1 ns 步長，總激發數 cutoff 取 pointer 自適應值再加一（三個受測 rf 為 N2／N3／N2）。單核 2,550 點 CKP map 外推 129 s，含 cutoff selection。六點 Pe 與保存 reference 最大差 0.005 shot σ；三個 rf、兩種 preparation 的六條 qubit line-centre 偏移最大 0.06 倍 reference 的 residual-scaled SE。較快的 RK4 2 ns 加 cutoff N 外推 20 s，但一條 line centre 偏移已達 0.92 SE，餘裕太薄，不採。

這是三個完整 51 q 行的外推，不是全圖實跑；其餘 22 個 rf 與完整 cavity-centre 未以候選驗證。使用者 2026-10-21 裁決下一步：以保守版建 prototype，接全圖驗證與 production wall 量測（「前兩個採用建議」）。

16 能階擴充不可行：既有公開模型 4／8 能階各實測一次 batch（23 s／93 s），多能階 DOP853 原型外推全圖 42 到 66 小時。後續多能階研究改為動態 [0,2] 加最低四階截斷診斷，保留完整 16 能階 virtual sum；物理截斷未驗證。

## 問題

CKP reduced closure 研究（見 related）結束於「reduced closure 沒有同時滿足精度與成本」，保存的 joint Fock reference 全圖 714 s。本研究問：從 joint Fock reference 出發，solver、排程重用與 cutoff 能否把兩能階 2,550 點壓到 200 s 級；16 能階成本如何。

精度門檻：六點 Pe 差在 shot σ 內；另抽三個 rf 各跑完整 51 q 行，候選與 reference 的 qubit line-centre 差在 reference 逐行 residual-scaled SE 內。完整 cavity-centre 只以 hybrid map 作敏感度診斷，不算通過。

## 考慮過的方案

| 方案 | 2,550 點外推 | 六點精度 | 中心 gate | 備註 |
| --- | ---: | --- | --- | --- |
| 保存 reference（CSR expm_multiply，sparse Krylov） | 1,708 s | 基準 | 基準 | 同機重測；先前 714 s 為另一設定 |
| dense DOP853（控制組） | 9,434 s | 通過 | 未測 | 只作同 generator 對照 |
| SciPy／Numba CSR DOP853 | 831／690 s | 通過 | 未測 | 容差控制，成本未達 |
| fixed RK4 2 ns | 238 s | 通過 | 未測 | 單獨最快 solver |
| CSR Krylov 加自適應 cutoff | 543 s（加權） | 0.06 σ | 通過 | cutoff 單獨不足 |
| RK4 2 ns 加 cutoff N（快版） | 20 s | 0.06 σ | 最大 0.92 SE | 餘裕薄 |
| RK4 1 ns 加 cutoff N+1（保守版） | 129 s | 0.005 σ | 最大 0.06 SE | 採用 |
| 現行半古典（production） | 274 s 同機 | 不通過（10 σ） | 物理不同 | 只作成本分母 |

排程重用：每 rf 共用 preparation，相鄰 rf 的腔 pointer 只作線性方程的數值 warm start，不改每個 rf 的物理初態。

多能階：模型取 simulate 引擎既有的 16 能階 JointModel 與逐能階腔頻移，旋轉框架走 `multitone_rwa` 公開的 coefficient owner，不重做 kernel 內部的 admission 與 pruning。4→8 能階六點 Pe 差 4e−13，8 能階中 level 4 以上總 population 最大 5e−82；這些是 RWA mask 下的數字，不是 lab-frame leakage 上界。

## 否決的方案與理由

- 快版（20 s）：rf12 ground line centre 偏移 0.92 SE，為未經全圖驗證的速度先用掉精度餘裕。
- 多能階優先：16 能階無實測成本，原型外推數十小時；先確認兩能階路線再談。
- 停止擴充：保守版已過局部門檻，值得一輪 prototype。
- dense 顯式解作主要 reference：保存 reference 已是 sparse Krylov，dense 只能作控制組。

## 留下的問題

- 候選在其餘 22 個 rf 與完整 cavity-centre 的全圖精度。
- 跨核心與 production runtime guard 的實際 wall。
- RK4 固定步長在其他驅動條件下的穩定性。
- 多能階的 RWA omission、kernel admission 與驅動後 leakage 是否有界。
- 若 joint Fock 兩能階進入 production CKP，半古典腔模型保留給哪些用途。

## 依據

- 保存 reference、六點 Pe 與逐行 fit 的 SE 定義沿 CKP reduced closure 研究的工作檔，未入庫。
- 成本數字為同一台機器單核、固定 commit 的 detached worktree 量測，單次配對、未取中位數；外推公式為三個完整 rf 行的加權。
- 16 能階 JointModel、逐能階 chi 與 RWA coefficient owner 的契約見 related 的 README 與 field-guide。
