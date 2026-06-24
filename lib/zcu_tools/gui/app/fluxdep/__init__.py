"""fluxdep-gui — a standalone Qt GUI for fluxonium flux-dependence fitting.

An optional analysis tool, fully independent of the measure-gui (``zcu_tools.gui``)
package: its own state / services / interactive widgets / MCP server / skill.
The two share no runtime coupling — common scaffolding (version table, NDJSON-TCP
RPC mechanism, mpl/progress backends) is *copied and rewritten* here rather than
imported, so each app evolves independently until a shared core is later
extracted under ``gui/app/{main,fluxdep}/`` (see .agent_state/plans/tool_gui/).

Pipeline (v1, the "front-half skeleton"): load an hdf5 spectrum → pick the half /
integer flux lines → select spectral points → (accumulate several spectra) →
cross-spectrum point filtering → export ``spectrums.hdf5``. Database search +
scipy fit + result visualisation are deferred to v2.
"""
