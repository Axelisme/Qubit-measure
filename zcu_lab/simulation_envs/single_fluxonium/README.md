# Single-fluxonium demo

This directory defines one fluxonium, one DC source, one microwave line,
one fast-flux line and one readout cavity. It has no coupler, second qubit or mux.

`default.yaml` contains physical settings. `entry-template.yaml` contains
the Q1/R1/flux_bias components, wiring and initial aliases.
The files have separate loaders and do not supply calibrated pulse modules.

`startup.py` creates or reopens `single-fluxonium-demo/initial`, then installs
the mock SoC and the fixed-unit DC driver through the GUI's normal startup
facets. New entries receive the template aliases once. Existing entries retain
their edited values. The script uses this directory's files, not the shell cwd.

Acquisition is not connected. Any acquire raises `NotImplementedError`
until the single-cavity ADC integration is complete. Startup assembly has not
been accepted through MCP, and no measurement result is claimed here.

Once readout integration is available, select this trusted script with
`connect(launch='new', script='<checkout>/zcu_lab/simulation_envs/single_fluxonium/startup.py')`.
The GUI does not choose a simulation mode or load model truth.

See [field-guide.md](field-guide.md) for units and approximation boundaries.
