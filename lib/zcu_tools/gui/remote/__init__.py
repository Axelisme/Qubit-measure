"""Shared transport mechanism for the GUI apps' RemoteControlAdapters.

App-agnostic, import-clean (no Qt, no matplotlib): the NDJSON framing, error
envelopes, the request/response wire types + field-validation primitives, and the
ParamSpec schema engine. Both ``app/main`` and ``app/fluxdep`` import these
verbatim; each app keeps only its own domain (dispatch tables, method specs, the
RemoteControlAdapter wiring) and its per-app wire version constants.
"""
