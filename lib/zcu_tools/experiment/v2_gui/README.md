# `zcu_tools.experiment.v2_gui` — GUI 實驗接入家族

**Last updated:** 2026-09-27 — measure 與 autofluxdep 分支

`v2` 延續 `experiment/v2` 與 `program/v2` 的版本脈絡，不表示 GUI framework 的第二版。
這個 package 容納兩種不同的 GUI 實驗接入契約：

- [measure](measure/README.md) 實作 measure-gui 的 `ExpAdapterProtocol`，擁有 concrete
  adapters、role 與 adapter registry，以及 catalog reload policy。
- [autofluxdep](autofluxdep/README.md) 實作 Autofluxdep app 的 Builder／Node／RunEnv
  workflow 契約，擁有 concrete 節點、catalog 與 `_support`；不依賴 measure adapters。
  具體節點的 authoring 說明留在該分支，不放入 measure adapter 文件。

兩個分支不共用 private adapter base，也不共用 reload catalog。GUI apps 擁有各自的
workflow 與 UI；實驗分支提供具體的領域定義。
