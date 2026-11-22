---
date: 2026-11-22
status: active
superseded_by: null
related:
  - ../../zcu_lab/v2/experiment-authoring.md#圖的建立顯示與更新
  - ../../zcu_lab/v2/support/live.py
  - ../adr/0072-module-boundaries.md
basis:
  repo_commit: a90cb66bbaa7e5269572d375fef321936dbeb3cc
  external_sources: Matplotlib 3.11.0
---

# 非均勻熱圖在建構時選色盤

## 結論

`live_heatmap` 接已註冊的 Matplotlib 色盤名稱，省略仍是 `RdBu_r`。五個原本晚設 `viridis` 的實驗改在建構時傳入，保留原色棒、座標、資料更新和設定。作者用法見 [實驗寫法](../../zcu_lab/v2/experiment-authoring.md#圖的建立顯示與更新)。

## 問題

Matplotlib 的 `NonUniformImage` 裝入資料後拒絕改色盤。Helper 在建立時已裝全 NaN 的 2×2 placeholder；就算把 setter 搬到第一次 leaf `set` 之前也太晚。

## 選項與取捨

把名稱交給原建構器，在 placeholder 之前套用。維持 uniform 與 nonuniform 的設定同義；未知名稱由 Matplotlib 拒絕，不掛圖。依 ADR-0072，這是 lab 的可選 artist 工具，不是通用框架政策。例：CKP 的不等距熱圖與 0–1 色限不變，只調整色盤設定時點。

## 未採用

刪除 setter 而改用預設會悄悄換掉五案配色。每個 leaf 自建 artist 會各自重做降序軸、NaN、色棒和更新。先把未裝資料的 artist 交給 caller 會增加呼叫時序約束，且無需為配色改成 `pcolormesh`。

## 留下的問題

不提供後設色盤 API 或任意 `Colormap` 物件契約。已驗原尺寸圖，不把這個變更等同 Data／Save pane 驗收。

## 依據

原始對照圖未入庫。Agg 公開接縫核對兩模式、錯名、升降序、缺值、色限與同一 artist 更新；個別 leaf 另用公開 stage 驗證。
