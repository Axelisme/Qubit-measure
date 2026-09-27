# Autofluxdep experiment tests

**Last updated:** 2026-09-27 — experiment ownership relocation

本目錄驗證 `experiment/v2_gui/autofluxdep/` 的 concrete Builder／Node、cfg、acquire、
fit、Result、Patch 和 catalog 契約。App 的 orchestrator、workflow persistence、
artifact 與 UI 行為仍由 `tests/gui/app/autofluxdep/` 擁有。

`conftest.py` 在這組案例的最小目錄範圍提供 offscreen QApplication、Qt event draining
和 mock fake_flux 清理，維持搬遷前的 fixture scope 與生命週期。共用 mock builder／
recording helpers 仍由可 import 的 `tests.gui.app.autofluxdep._helpers` 提供，
不從另一個 test module 或 conftest 匯入。
