---
date: 2026-11-23
status: active
superseded_by: null
related:
  - ../adr/0069-experiment-core-and-frontends.md#呈現規範
  - ../adr/0071-coherence-units.md#pair-lifetime-與-manifold-model-t1
  - ../adr/0073-measure-form-derivation.md#form-的形狀與初值
  - 2026-11-23-singleshot-t1-relaxation-times.md
basis:
  repo_commit: d7911319985a4bfe17aec50b7ecb60953caa88b0
  external_sources: none
---

# Singleshot T1 以振幅分 T1／T1b

## 結論

2026-11-23，使用者要求按振幅區分主副模。本次選「準備 e、量 e 布居」曲線。絕對振幅較大的模，其衰減時間是 `t1`，另一個是 `t1_b`。振幅相等時取較長時間為 `t1` 並記 log。兩版 Analyze 圖改寫成 `T1 = …, T1b = … µs`，保留 reduced χ²。

固定 tone 版仍提案寫入 `$qubit.ext.t1_with_tone`，值改為主模 `t1`。Tone Stop 初值仍是保存值的五倍，舊值不遷移。普通版新增 `$qubit.t1` 的寫回提案，也取主模。

## 問題

夜間代裁把兩個模型衰減時間由短到長列出，不將其中一個稱為 T1。固定 tone 版仍寫回較短值，普通版不提寫回。時間長短不表示該模在觀測的 e 布居曲線上有多大振幅。

## 選項與取捨

- 沿用短到長，不用選觀測量，但不符合使用者要以主模寫回 T1 的要求。
- 以 e 準備的 e 布居選主模。用明確的初態與可觀察量解讀兩個模型係數，採此做法。
- 以所有準備與布居曲線的總振幅選模是使用者允許的替代。這次不採用，因 e 準備的 e 布居就是要解讀的 T1 曲線，不需加總其他曲線。

## 留下的問題

Stop 的五倍是掃描初值，不保證曲線達到指定 plateau。舊保存值不重算；下次接受寫回時才更新。

## 依據

- 2026-11-23，使用者：「但我就是要寫回t1啊，你應該把振幅較大者視為t1，較小者視為副t1，我記得我都寫t1b之類的」；「3b要，a可以，也可以用總振幅而不只是e」。原始問答未入庫。
- 推翻[按短到長命名的夜間代裁](2026-11-23-singleshot-t1-relaxation-times.md)，保留其 Stop 倍數。沿用 [ADR-0069](../adr/0069-experiment-core-and-frontends.md#呈現規範) 的數值分析與寫回分工，由 core 計算，adapter 提案。這是模型衰減模，不用 [ADR-0071](../adr/0071-coherence-units.md#pair-lifetime-與-manifold-model-t1) 的 pair lifetime 取代。
- 操作與欄位見[普通 T1](../../zcu_lab/v2/singleshot/t1/guide.md)與[固定 tone T1](../../zcu_lab/v2/singleshot/t1_tone/guide.md)。
