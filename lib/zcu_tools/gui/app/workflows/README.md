# gui/app/workflows — Workflow form submission

**Last updated:** 2026-10-09，bounded Start／Stop composition

TunablesHost owns accepted expression input and Apply/Discard for one native run. GUI and external edits submit complete current ordinary values through the same FormState/Engine seam. External edits use the same parsed GUI draft, without a GUI revision conflict guard. Engine owns numeric validation, revision, step capture and journal publication.

Control identity will reject non-holder writes after it lands. This transition does not check holder and does not introduce a workflows-specific holder mechanism.

WorkflowController owns pending Start admission, worker startup/execute and cooperative Stop through OperationRunner. The worker holds the run lease until owner terminal delivery. SessionDevices borrows the shared device service's public drivers; native setup and driver limits remain with their existing owners. Concrete T1WorkflowWindow and T1TuningPanel attachments stay in zcu_lab and use the same captured inputs. Pause/Resume, RPC/MCP and source reload remain unimplemented. Offline Start/Stop evidence does not establish full F5/P2, CP or hardware acceptance.
