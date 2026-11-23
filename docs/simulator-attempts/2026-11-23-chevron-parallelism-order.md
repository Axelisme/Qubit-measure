---
date: 2026-11-23
status: paused
commit: ffd2a0c6c
modules:
  - lib/zcu_tools/simulate/evolution
  - lib/zcu_tools/simulators/ftf/environment
discussion: none
superseded_by: null
---

# Chevron 先減少 caller 準備成本，再評估程序池

**讀者：** framework-dev、agent。查 heralded chevron 的執行緒收益，以及為什麼暫緩程序池。

## 核心思路

FTF 會把 worker 數傳給 Engine。Engine 明確用 joblib `threading`，並非嘗試 `loky` 失敗才降級。有 herald 時，FTF 逐 sweep point 建投影鏈，每次只把該點當前 span 的狀態列送進 evolve。調查先比較既有 worker 數，再試直接換成程序 backend。另一個方案是在 FTF 層把整個 point chain 交給一個程序，讓各程序從凍結模型建立自己的 Engine。

## 實測

在 `ffd2a0c6c` 用離線 FTF `two_qubit/chevron`。gg herald，頻率 5795–5805.5 MHz 與時長 0.06–0.10 µs 各取 2 點，100 reps、1 round、seed 123、24 retained levels、projection floor `1e-5`、合成分類器。非排他 CPU 0–3，native 與 Numba 各限 1 thread。1 worker 與 7 workers 各跑一次，完整 Run 牆鐘分別是 413.00 s 與 323.08 s，約快 1.28 倍，兩側都沒開子程序。同一對 Run 的 `multitone_rwa` preparation 分別用了 269.14 與 258.89 CPU s。CPU 加總不可當成可直接從牆鐘扣掉的時間。

同一 commit 上，把無 herald 的同尺寸 7-worker 對照只換成 `loky`。送出共用準備工作時，執行緒鎖無法 pickle，Run 未完成，沒有程序池速度數字。原始工作檔與覆寫原型未入庫。

## 為何採用或否決

先減少 caller 端的重複準備工作。只換 pool 不會把原本在 caller 做的工作平行化，且現有 job 借用不能直接序列化的狀態。調查之後[在同次 preparation 重用探針結果](./2026-11-23-ftf-rwa-probe-memo.md#核心思路)已採用；程序池本身則暫緩。

## 留下的線索

若再試程序池，先定義可序列化的整點工作單位。父程序按原掃描順序合併結果後再做 ADC 與 seeded draws。跨程序的 Stop、routing history、debug capture 各由誰持有要先定，並量 worker 建模時間及記憶體。不要用四點的 worker 比值推算 31×31，也不要假定跨點合批一定保留相同的 guard 分組。
