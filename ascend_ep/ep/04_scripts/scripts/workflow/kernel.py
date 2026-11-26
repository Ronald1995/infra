"""Phase 4: kernel build.

Four native projects can be needed by an L0 case:

* ``kernels/asc_comm``                -- the AIV / CCU device data planes,
* ``baselines/l0/transport/native``   -- the host transport bridge,
* ``baselines/l0/shmem/native``       -- the UDMA kernel,
* ``baselines/l0/memory/native``      -- the memory copy kernel.

The rule that matters: **a missing device kernel blocks only the cases that need it.** It
never turns an AIV/CCU measurement into a host NBI measurement, and it never blocks the
rest of the matrix. A build runs only when the source and the toolchain are both present;
``--clean`` is never used, because deleting a working artifact to rebuild it is not a
diagnostic step.
"""
from pathlib import Path

from scripts.workflow.phases import PhaseOutcome

#: Native projects that are built with their own script, keyed by a stable name.
NATIVE_PROJECTS = {
    "transport_bridge": "baselines/l0/transport/native",
    "shmem_udma": "baselines/l0/shmem/native",
    "memory_kernel": "baselines/l0/memory/native",
}
ASC_COMM = "kernels/asc_comm"


def _kernel_artifacts():
    from baselines.l0.transport import kernel_artifacts
    return kernel_artifacts


def engine_state(root):
    """``{engine: probe dict}`` for the AIV and CCU device planes."""
    module = _kernel_artifacts()
    state = {}
    for engine in ("AIV", "CCU"):
        probe = module.probe(engine)
        record = probe.as_dict()
        record["project"] = str(module.PROJECT)
        record["source_files"] = _source_files(module.PROJECT / engine.lower())
        state[engine] = record
    return state


def _source_files(directory):
    """Device sources for one engine. Only a README means "source is missing"."""
    directory = Path(directory)
    if not directory.is_dir():
        return []
    return sorted(path.name for path in directory.iterdir()
                  if path.is_file() and path.suffix in {".c", ".cc", ".cpp", ".h", ".hpp", ".asc"})


def native_state(root):
    """``{project: {...}}`` for the native helper projects."""
    state = {}
    for name, relative in NATIVE_PROJECTS.items():
        directory = Path(root) / relative
        scripts = sorted(path.name for path in directory.glob("build*.sh")) \
            if directory.is_dir() else []
        sources = sorted(path.name for path in directory.iterdir()
                         if path.suffix in {".c", ".cc", ".cpp", ".h", ".hpp"}) \
            if directory.is_dir() else []
        state[name] = {"path": relative, "source_present": directory.is_dir(),
                       "sources": sources, "build_scripts": scripts}
    return state


def _toolchain_ready(root):
    """Whether the ASC-COMM build has what it needs, from its own probe."""
    module = _kernel_artifacts()
    toolchain = module.toolchain()
    build_script = Path(root) / ASC_COMM / "build.sh"
    toolchain["build_script"] = str(build_script) if build_script.is_file() else None
    toolchain["complete"] = bool(toolchain.get("complete") and toolchain["build_script"])
    return toolchain


def plan_build(root):
    """What can and cannot be built here, with the reason for each decision."""
    engines = engine_state(root)
    toolchain = _toolchain_ready(root)
    buildable, blocked, ready = [], {}, {}
    for engine, record in engines.items():
        if record["status"] == "ready":
            ready[engine] = record
            continue
        if not record["source_present"]:
            blocked[engine] = ("the ASC-COMM kernel project is missing at "
                               f"{record['project']}")
            continue
        if not record["source_files"]:
            blocked[engine] = (f"{engine} device source files are missing "
                               f"({record['project']}/{engine.lower()} has no source)")
            continue
        if not toolchain["complete"]:
            blocked[engine] = ("the CANN toolchain is incomplete: "
                               + ", ".join(name for name in ("ascend_home", "cmake", "cxx",
                                                             "build_script")
                                           if not toolchain.get(name)))
            continue
        buildable.append(engine)
    return {"toolchain": toolchain, "engines": engines, "ready": ready,
            "buildable": buildable, "blocked": blocked,
            "native": native_state(root)}


def handler(ctx):
    request = ctx.request
    root = ctx.root
    plan = plan_build(root)
    native = plan["native"]
    path = ctx.write_json("kernel_report.json", {
        "schema_version": 1, "workflow_id": ctx.state.workflow_id,
        "soc": request.environment.soc,
        **plan,
    })

    if request.environment.device == "cpu":
        return PhaseOutcome.skipped(
            "a CPU run needs no Ascend device kernel; the AIV/CCU data planes stay "
            "outside the measured scope")

    if not plan["buildable"]:
        if plan["ready"] and not plan["blocked"]:
            return PhaseOutcome.skipped("every kernel the run needs is already built")
        reason = ("no kernel can be built here: "
                  + "; ".join(f"{engine}: {why}" for engine, why in plan["blocked"].items()))
        # A missing device kernel blocks *those cases*, not the workflow: the rest of
        # the matrix is still measurable and the affected L0 tasks stay blocked in the
        # capability probe.
        return PhaseOutcome.partial(
            reason + ". The cases that need those engines stay blocked; nothing falls "
            "back to the host data plane.", artifacts=[path],
            detail={"blocked_engines": plan["blocked"]})

    soc = request.environment.soc or (
        (ctx.data.get("doctor") or {}).get("platform") or {}).get("soc") or ""
    argv = ["bash", str(Path(root) / ASC_COMM / "build.sh")]
    if soc:
        argv += ["--soc", soc]
    argv.append("--strict")
    result = ctx.commands.run(argv, log_name="asc_comm-build.log")
    ctx.record("kernel_build", result, engines=plan["buildable"])
    after = plan_build(root)
    built = {engine: record for engine, record in after["engines"].items()
             if record["status"] == "ready"}
    ctx.write_json("kernel_report.json", {
        "schema_version": 1, "workflow_id": ctx.state.workflow_id,
        "soc": request.environment.soc, "before": plan, "after": after,
        "build_exit_code": result.exit_code, "build_log": str(result.log_path),
    })
    if result.exit_code != 0:
        return PhaseOutcome.partial(
            f"the kernel build exited {result.exit_code}; the cases that need it stay "
            f"blocked (see {result.log_path})", artifacts=[path],
            detail={"build_exit_code": result.exit_code})
    still_blocked = {engine: record["reason"] or record["status"]
                     for engine, record in after["engines"].items()
                     if record["status"] != "ready"}
    if still_blocked:
        return PhaseOutcome.partial(
            "the build ran but these engines are still unavailable: "
            + "; ".join(f"{engine}: {why}" for engine, why in still_blocked.items()),
            artifacts=[path], detail={"built": sorted(built)})
    return PhaseOutcome.passed([path], detail={"built": sorted(built),
                                               "native": native})
