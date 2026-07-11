---
status: accepted
---

# 0052 — Event meta 與多前端 attribution

**狀態:** accepted(2026-07-11)。
**關聯:** [[0004]]、[[0021]]、[[0047]]、[[0048]]、[[0049]]。

## 背景

GUI 已同時存在兩個 peer 前端:Qt View 與 MCP agent,未來將加入 Web View(見
`web_view_evaluation.md`)。[[0048]] 讓事件攜帶 closed domain fact、[[0049]] 讓 wire push
subscriber-aware,但事件仍缺三個多前端必需的屬性:

1. **attribution**:訂閱者無法分辨事件由誰引發(使用者操作、agent 命令、或某個
   operation 的生命週期)。agent 看不出「是我改的還是使用者改的」;Qt View 無法
   顯示「agent 正在做什麼」的 presence 提示。
2. **順序性**:事件沒有全域序號,斷線重連的 client 無法判斷自己漏了什麼,只能
   全量重新定向。
3. **高頻收合**:cfg 表單每個 keystroke 觸發全樹 snapshot 並整份 commit 進
   State(`widgets/cfg/form.py` schema_changed),無收合機制。

先例依據(調研見 `web_view_evaluation.md` 附錄 B):Jupyter IOPub 的
`parent_header` attribution、Syncthing 的單調 event id、OBS 的高頻 channel 顯式
opt-in、Home Assistant 的 per-tick coalescing 教訓。

## 決策

### EventMeta 與蓋章

`gui.event_bus.BaseEventBus` 在 emit 時為每則事件產生 `EventMeta`:

```python
@dataclass(frozen=True)
class EventOrigin:
    kind: Literal["user", "agent", "system"]   # 原始發起者
    client_id: str | None = None               # kind=agent 時的 RPC client
    operation_id: str | None = None            # 由某 operation 生命週期引發時

@dataclass(frozen=True)
class EventMeta:
    seq: int          # bus 單調遞增,process 內唯一
    origin: EventOrigin
```

- payload dataclass **完全不動**;meta 是 bus 蓋章的側車,不是 payload 欄位。
- 既有 `subscribe(payload_type, cb)` 簽名與行為不變;需要 meta 的訂閱者改用新的
  `subscribe_with_meta(payload_type, cb)`(callback 收 `(payload, meta)`)。
- `seq` 由 bus 維護,單調遞增、不保證連續(未來 coalescing 合併時允許跳號)。

### Origin 的設定與巢狀規則

- origin 由 **dispatch 邊界**以 context manager 宣告:remote handler 執行包在
  `with bus.origin(EventOrigin(kind="agent", client_id=...))`;GUI 端命令不宣告,
  預設 `kind="user"`;背景/自發行為(persistence flush、shutdown)為 `kind="system"`。
- **巢狀規則(對應 Jupyter parent_header 鏈):** operation 生命週期內 emit 的事件
  (worker 完成回主線程後的 terminal/content 事件)同時攜帶「原始發起者 kind/client_id」
  與該 operation 的 `operation_id`——發起者資訊在 operation 建立時捕捉並存於
  operation 記錄,terminal 路徑 emit 時還原,不靠 thread-local 跨線程傳遞。
- context manager 使用 ContextVar 實作,但**不跨 thread 傳播**(Python thread 起始
  為空 context);跨 worker 的 origin 一律走上述 operation 記錄顯式攜帶。

### Subscriber-side coalescing

bus 本體維持同步、Qt-free、零 timer([[0049]] 的 lazy 選擇不變)。收合由訂閱端
adapter 各自負責:

- **UI coordinator**:同一 event-loop tick 內對同一 (payload_type, tab_id) 的多次
  事件,以 0 ms `QTimer` 合併為一次 reaction(fact 反應矩陣語意不變,只是同 tick
  去重)。
- **cfg 表單 snapshot(E7)**:`CfgFormWidget` 的 `schema_changed` 全樹 snapshot 改為
  0 ms 合併——同一 tick 多次 field 變更只 materialize 一次 `CfgSchema` 並 commit
  一次 State。逐鍵 validity 回饋**不受影響**(validity event 與 snapshot 分離)。
- **remote broadcaster**:沿用 [[0049]] 的 lazy line factory;是否加批次 flush 由
  wire 量測決定,本 ADR 不強制。

### Wire 穿透

- remote event 封套由 `{"event", "payload"}` 擴為
  `{"event", "payload", "seq", "origin"}`(additive,不破壞既有 client;
  `origin` 為 `{kind, operation_id}`,**不含 client_id**——client 身分屬
  RPC↔MCP bookkeeping,agent 只需分辨 user/agent/system)。
- MCP 層 pass-through 暴露 origin,使 agent 能分辨事件來源(批次 2 驗收標準)。
- `WIRE_VERSION` 因封套加欄位而 bump 與否,依既有 wire 政策(additive 欄位若政策
  允許不 bump,則不 bump)。

### 高頻 channel 原則(規格,先行宣告)

未來的高頻資料流(live plot patch、progress tick)一律**顯式 opt-in 訂閱**,與低頻
state event 分流;預設訂閱集不含高頻 channel(OBS 模式)。本 ADR 只立原則,
plot stream 的具體 channel 設計屬 plot document 階段(路線批次 5 / Web Phase 3)。

### Replay(僅規格,不實作)

重連補齊機制規格如下,**於 Web adapter 需要時才實作**:bus(或 wire adapter)保留
最近 N 則事件的 `(seq, wire line)` 環形 buffer;client 重連帶 `since=<last-seen-seq>`,
gap 可補則回放、不可補(seq 已滾出 buffer)則回覆「需全量 re-snapshot」。在此之前,
斷線 client 的既定行為維持現狀:以 `gui_overview` / snapshot 全量重新定向。

### Stale 衝突 code(承 D.1#4)

stale-version 衝突維持 `PRECONDITION_FAILED` + `data={"stale":[...]}`,**不新增**
`CONFLICT` code:MCP 層的 re-snapshot contract 已提供 reread-retry 語意,closed
enum 不為此擴張。Web adapter 落地時若 HTTP 慣例需要 409 對映,由 Web adapter 在
自己的翻譯層對映,不回頭改 wire enum。

## 後果

- agent 與未來 Web View 可靠 origin 分辨事件來源;presence 顯示(「另一方正在…」)
  的資料基礎就緒。
- 事件獲得全域順序,重連補齊有明確規格可循。
- keystroke 級的 State commit 從 per-event 降為 per-tick,且不犧牲逐鍵 validity。
- payload 與既有 `subscribe` 完全不變,遷移零強迫;`subscribe_with_meta` 只在需要
  meta 的邊界(remote broadcaster、presence UI)使用。
- bus 仍是同步、Qt-free 的純機制模組,批次 3 的 OwnerScheduler 引入不受本 ADR 牽制。

## 非目標

- 不做 bus-level 非同步/deferred publish;不做 CRDT 或 view 端樂觀更新(明文禁止,
  pending 態由 core 廣播);不實作 replay buffer;不設計 plot stream channel;
  不動 [[0048]] 的 fact 分類與反應矩陣語意。
