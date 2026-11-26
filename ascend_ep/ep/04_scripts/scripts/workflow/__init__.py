"""Prompt-driven Ascend EP benchmark workflow.

The package turns one structured request into an auditable sequence of phases that call
the documented ``run_benchmark.py`` commands. It contains no benchmark logic of its own:
the plan, scheduler, controller, runner, analysis and reporting stay where they were.

Entry points:

* :func:`handlers` -- the phase name to handler mapping used by the orchestrator,
* :mod:`scripts.workflow.request` -- natural-language interpretation and the request
  schema,
* :mod:`scripts.workflow.state` -- the persisted state machine.
"""
from scripts.workflow import (
    analysis, capability, dependency, environment, execution, kernel, planning, profiling,
    reporting,
)

#: Phase name -> handler. ``request_resolution`` is handled by the orchestrator because it
#: runs before a workflow directory exists.
PHASE_HANDLERS = {
    "environment_discovery": environment.handler,
    "dependency_check": dependency.handler_check,
    "dependency_install": dependency.handler_install,
    "capability_probe": capability.handler,
    "kernel_build": kernel.handler,
    "plan_generation": planning.handler_plan,
    "resource_estimation": planning.handler_estimate,
    "benchmark_execution": execution.handler_run,
    "result_validation": execution.handler_validate,
    "profiling": profiling.handler,
    "analysis": analysis.handler,
    "report_generation": reporting.handler_generate,
    "final_validation": reporting.handler_validate,
}


def handlers():
    """The complete handler mapping, including the request resolution phase."""
    from scripts.workflow import request as request_module
    return {"request_resolution": request_module.handler, **PHASE_HANDLERS}


__all__ = ["PHASE_HANDLERS", "handlers"]
