---
date: 2026-11-22
status: active
superseded_by: null
related:
  - ../../lib/zcu_tools/analysis/fitting/README.md
  - ../../lib/zcu_tools/analysis/fitting/spectrum.py
  - ../adr/0085-numerical-fitting-owners.md
  - ../../zcu_lab/v2/figure-style.md
basis:
  repo_commit: a90cb66bbaa7e5269572d375fef321936dbeb3cc
  implementation_baseline: b3d508a0d
  external_sources: none
---

# 頻譜擬合模型的求值 owner

## 結論

`SpectrumFit.evaluate` 用同次擬合係數在給定的窗內 MHz 座標求值。Two-tone Freq 與 Chain 各自選 200 點網格，保留原觀測點。數值留 fitting，畫圖留實驗。公開接縫記在 [fitting README](../../lib/zcu_tools/analysis/fitting/README.md) 與 docstring。這次由 designer 代裁，當時未經使用者覆核，沒有改寫 ADR-0085。

## 問題

原模型線用了擬合輸入的位置與相應模型值。局部窗口可能只有十幾個樣本，線看起來像原點相連，並非以密網格呈現連續模型。

## 選項與取捨

保留 Lorentzian 或 sinc 的同次係數，讓 caller 傳入窗內座標。這延續 ADR-0085 的數值與呈現分工；Freq 原本的局部峰窗口、Chain 原本的中點窗口和選峰政策不變。用兩套 81 點輸入核對估計、原點、窗口與輸出的 200 點，不為畫線再擬合。

## 未採用

插值不等於模型求值。讓 fitting 決定網格點數會讓畫圖政策進入 fitter。公開係數也可行，但兩個 caller 都得配對模型與參數並驗窗口；當時沒有用係數做其他事的 caller。

## 留下的問題

不承諾窗口外預測、不新增係數格式。將來若有實際需要讀參數的 caller，再比較是否公開係數。

## 依據

沿用 [R2](../../zcu_lab/v2/figure-style.md#r2同時畫量測點與密網格的-fit)、[ADR-0072](../adr/0072-module-boundaries.md) 與 [ADR-0085](../adr/0085-numerical-fitting-owners.md)。原前後圖和 fixture 未入庫；數值接縫的測試在 [test_spectrum.py](../../tests/analysis/fitting/test_spectrum.py)。
