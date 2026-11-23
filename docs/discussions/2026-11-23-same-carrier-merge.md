---
date: 2026-11-23
status: active
superseded_by: null
related:
  - ../adr/0078-evolution-module-boundary.md
  - 2026-11-18-sampled-multitone-rwa-admission.md
  - 2026-11-19-shared-multitone-pricing.md
  - ../simulator-attempts/2026-11-23-same-carrier-merge.md
  - ../simulator-attempts/2026-11-24-same-carrier-shaped-sambe.md
basis:
  repo_commit: 3e86bba6d8c9e3e9cacf25599c7bb82566a82050
---

# 同頻多線在 Evolution 的表示

## 結論

使用者於 2026-11-23 裁定：「S 直接實作加異形快速路徑prototype」。當時選擇在 Evolution planner 保留載波群與原 slots 的對照，同形群才摺成一個虛擬 tone。不同形的群仍用原有多 tone 路由，另做不合入的快速路徑原型。公開 `DriveTrack`、capture 輸入與 FTF 介面不改。[Evolution README](../../lib/zcu_tools/simulate/evolution/README.md)記錄實作規則，[嘗試紀錄](../simulator-attempts/2026-11-23-same-carrier-merge.md)保存前後對照。

## 問題

當時 planner 一條 active drive 用一個 slot，選路依 slot 數分單 tone 與多 tone。同頻 CZ 的兩條線即使共用包絡，也不能用既有的單 tone family。直接把不同相位和算符的 gain 當純量相加會改掉 Hamiltonian。另一方面，兩條線可能用不同波形，不能只看頻率就摺。

## 考慮過的選項與比較

- **S，載波群 sidecar。** 保留 slot 軸，以 sidecar 記錄載波群與原通道。同形時把 gain 與相位合進一對虛擬 I/Q 算符，由既有 family 處理；異形時保留各線的包絡。挑戰後選 sidecar，不改 slot 軸，否則第一階段就得修改每個 kernel 的輸入。
- **F，虛擬 port 捷徑。** 同樣合成 I/Q 算符，但只保留摺後的 port，不建立載波群對照。它能取得同形原型的收益，後續加異形 family 時要重做群組表示。
- **維持現狀。** 不動選路或 prepare identity。同形 CZ 繼續負擔多 tone 的準備與演化成本。
- **一次改成多通道 family。** 各 family 直接消費同載波多通道。準備、guard 和 telemetry 都要改，且要處理包絡不同時的交叉項。當時沒有異形快路徑的量測證據支持這個範圍。

## 被否決的選項與理由

F 省下的主要是載波群對照，沒有省掉合成算符或 cache identity 的工作。考慮已知呼叫端能產生異形線，當時不選將來要換表示的捷徑。不改則留下已在 RB 與 E0 原型觀察到的多 tone 成本。一次改完所有 family 會延後同形收益；保留原多 tone 路徑能先處理異形輸入，沒有必要等快路徑。

## 留下的問題

落地審查於 2026-11-24 釐清兩個邊界。只有 `sample_ns` 逐項相同才摺，兩線插值成比例也不例外，免於選一條新取樣軸。依 recipe 計算的準備品和跨 evolve 的選路記錄要區分 recipe。準備完整模型、apply 才選 port 的 `dop853` 和 `flux_dop853` 則按 model identity 共用，不按 channel 集合重建模型。

[異形同載波原型](../simulator-attempts/2026-11-24-same-carrier-shaped-sambe.md)在受測輸入未通過誤差檢查。異形列繼續用現有多 tone 路由。若以後的慢場景主要花在異形列，再按實測路由和 prepare 次數判斷要加哪種快速 family。

## 依據

遵循 [ADR-0078](../adr/0078-evolution-module-boundary.md) 的數值 owner、支援範圍、preparation lifetime 與先量效率原則。合併只在 Evolution 判斷；不摺的列保留既有求解路徑。先前的 sampled multitone RWA 放行與成形雙 tone 比價都保留多 tone 表示，與這次只對同載波同形線消去一個 slot 不同。原始調查、挑戰與原型工作檔未入庫。
