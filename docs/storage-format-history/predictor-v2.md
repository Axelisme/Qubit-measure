# Predictor v2

**讀者：** operator、lab-author、framework-dev、agent。辨識 predictor v2、升級舊檔並還原備份。

## 格式與座標

`zcu.predictor` 的 `version` 現為整數 2。Fluxonium 的 `params` 只有
`EJ/EC/EL`，單位 GHz。Transmon 的 `params` 只有 `EJmax/EC/d`，
前兩者用 GHz，`d` 是無因次 junction asymmetry。省略或 null 的能量仍是未知值。

`flux` 保存該批資料的 offset、每台儀器的 signed period 與單位。
`offset` 用 Phi0，不做 mod 1。`period` 與 `unit` 的儀器名稱必須完全相同，
period 必須有限且非零，unit 必須是非空白字串。空的 period/unit 表示固定 flux。
原本的頂層 `flux_unit` 與 params 中的仿射欄位不再存在。

```json
{
  "format": "zcu.predictor",
  "version": 2,
  "model": "fluxonium",
  "params": {"EJ": 5.0, "EC": 1.0, "EL": 0.5},
  "flux": {
    "offset": 0.25,
    "period": {"bias": -0.4, "coil": 2.0},
    "unit": {"bias": "mA", "coil": "A"}
  },
  "source": {"run_id": null, "created_at": null}
}
```

座標是 `φ = offset + Σ value_d / period_d`。換算由 `analysis.flux` 擁有，
數值 predictor 只收 reduced flux φ。檔案的 offset 記錄該批資料的對齊，
可包含 flux jump，不代表 component 的目前校準。Source 保留作者給定的 run ID 與含時區時間，
未知值寫 null，讀寫不補目前時間或查 ledger。

## 明確升級

讀取端只認 v2。V1 讀取失敗時會列出 entry 升級命令，不會改檔。
在 checkout 根目錄執行同一個 component 升級入口，預設只列預覽：

```bash
uv run --no-sync -- python scripts/upgrade_parameters.py Result/<entry>
uv run --no-sync -- python scripts/upgrade_parameters.py Result/<entry> --apply
```

Entry 範圍包含 setup、各 point 的 component YAML 與 `predictors/*.json`。
Record 範圍 `--scope records` 不受此變更影響。

V1 換算保留能量與 source，使用下式：

```text
offset = 0.5 + (flux_bias - flux_half) / flux_period
period = {"unnamed_device": flux_period}
unit = {"unnamed_device": flux_unit 或 "native"}
```

升級不猜儀器名稱，操作者使用前應把 `unnamed_device` 換成實際名稱，
並同步修改 period 與 unit 的 key。缺少或 null 的 flux_unit 採 native。
半通量、period 或 bias 缺失、非有限，以及 period 為零的 v1 都會拒絕升級，
不替未知校準補預設值。已是 v2 的檔案驗證後跳過，未來版本拒絕處理。

`--apply` 在替換每個有效舊檔前，先建立 `<file>.<unique>.bak` 保存原始 bytes。
單檔替換採 sibling temporary 與 atomic replace。每份檔案獨立報告
dry-run、applied、skipped 或 rejected，一份失敗不阻止其他檔案。
這不是多檔交易或掉電持久保證。還原時停止使用該檔，將報告中的 backup 複製回原路徑。
還原成 v1 後，v2 runtime 仍會拒讀，需重新明確升級。
