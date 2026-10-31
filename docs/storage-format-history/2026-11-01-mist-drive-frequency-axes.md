# MIST drive-frequency native 軸順序

**讀者：** operator、lab-author、framework-dev、agent。辨識 MIST drive-frequency 的舊方陣檔與修正後的 native 軸順序。

## 記憶體與磁碟版面

`mist/power_dep/drive_freq` 的 `signals` 一直使用 `[frequency, gain]`。
修正只調整 `AXES_SPEC` 的 inner-first 軸順序，不轉置 Result，不改分析、pulse 序列或軸 label／unit。

| 宣告／版面 | 修正前 | 修正後 |
| --- | --- | --- |
| `AXES_SPEC`，inner-first | `freqs`, `gains` | `gains`, `freqs` |
| Native `vars/data` 的 NumPy 維度軸 | `Pulse gain` (a.u.), `Pulse frequency` (Hz) | `Pulse frequency` (Hz), `Pulse gain` (a.u.) |
| Result `signals` | `[frequency, gain]` | `[frequency, gain]` |

例如 4 個 frequency 與 7 個 gain 會產生 `(4, 7)` 的 `signals`。
舊宣告要求 `(7, 4)`，因此 writer 在寫入前拒絕。
新宣告保存 `(4, 7)`，第一維為 frequency，第二維為 gain。
記憶體頻率使用 MHz，磁碟使用 Hz，比例仍為 `1e6`。

## 辨識舊方陣檔

舊 writer 在 frequency 與 gain 點數相同時可以保存方陣，但磁碟軸與 Result 的語意錯置。
不要只用方陣 shape 判斷方向，也不要只因軸長相同就交換軸。

辨識時核對 native `docs/metadata.experiment` 為 `mist/`，以及 `vars/data` 的維度軸。
舊 DriveFreq 方陣檔的 frequency／gain 點數相同，且磁碟第一維是 `Pulse gain` (a.u.)，第二維是 `Pulse frequency` (Hz)。
修正後的第一維是 frequency，第二維是 gain。

例如舊檔可包含 frequency `[101, 103, 105, 107]` MHz、gain `[0.11, 0.21, 0.31, 0.41]`，及非對稱的 `(4, 4)` complex IQ。
舊 writer 仍會依錯置宣告，把 gain 軸放在磁碟第一維，把頻率軸放在第二維。
新版 Load 核對軸 label／unit 時明確拒絕這類檔案，錯誤指出 `Pulse frequency`／`Hz` 與預期的 `Pulse gain`／`a.u.` 不符。
它不在 runtime 轉置陣列，也不自動重排歷史軸。

## 舊資料處理

目前沒有這個格式的離線轉換工具。
保留原檔，不改寫、不刪除，也不要把只換軸名稱的檔案當成已驗證的轉換。
未來的轉換工具必須先辨識原始軸與資料方向，再在副本中轉換並逐點核對。
通用 container 版本及 cfg schema 版本沒有因這次軸順序修正而改變。
