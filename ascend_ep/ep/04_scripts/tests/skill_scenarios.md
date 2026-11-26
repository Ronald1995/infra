# Skill acceptance scenarios

The ten prompts from the skill specification, the decision each one must produce, and where
that decision is tested. The tests are offline: they drive the real orchestrator against a
modelled host (`tests/conftest.py::FakeBenchmark`), so a scenario can be asserted without an
Ascend device.

Run them:

```sh
python -m pytest tests/test_workflow_execution.py tests/test_workflow_gates.py \
                 tests/test_workflow_resume.py tests/test_workflow_state.py -q
```

## A — full automation

> 在 Ascend 950 上运行完整 EP Benchmark，EP8、EP64、EP128，覆盖 L0-L4 和所有算子。
> 如果缺少依赖，尝试安装；如果需要编译 Kernel，尝试编译。完成后进行性能分析和
> Profiling，生成完整报告。

Expected: `full_benchmark`, the plan's EP8/64/128 and L0-L4, `install_missing`, SoC
`Ascend950`; doctor, capabilities, plan, one `run`, `profile`, `analyze`, `report`.

Tested by `test_scenario_a_full_benchmark_is_interpreted_and_run`,
`test_scenario_a_install_is_only_attempted_when_authorised`.

## B — partial scope

> 只跑 EP64 的 L1、L2，比较所有支持的 Operator。

Expected: `partial_benchmark`, EP64 only, L1+L2 only, `operators: all` preserved (the
operator set is never narrowed by a level request), other EP sizes absent from the manifest.

Tested by `test_scenario_b_partial_scope_keeps_operator_coverage`.

## C — plan and estimate only

> 先帮我估算完整 Benchmark 的任务数和运行时间，不要执行测试。

Expected: `estimate`; doctor and plan may run; nothing installed; no benchmark worker
started; a runtime estimate with its `basis` and `uncertainty`.

Tested by `test_scenario_c_estimate_does_not_execute_or_install`.

## D — resume

> 继续上次没跑完的 EP Benchmark。

Expected: `resume`; the latest unfinished workflow is selected; request hash and plan hash
are checked; passed tasks are not re-measured (the run continues with `resume --run-id`); the
downstream phases are completed.

Tested by `test_scenario_d_resume_picks_the_unfinished_workflow` and
`tests/test_workflow_resume.py`.

## E — incomplete dependencies

> 完成完整 Benchmark，如果缺少依赖就尝试安装。

Expected: installable items are attempted **only** when the network or a wheelhouse was
authorised; a host-provided dependency (driver, CANN, device) is blocked with its manual
step; operators that depend on a blocked item stay blocked while the rest runs.

Tested by `test_scenario_e_installable_and_host_dependencies_are_separated`,
`test_installable_dependency_without_authorization_is_skipped`,
`test_install_runs_when_the_network_is_allowed`.

## F — incomplete kernels

> 完成所有可运行的 L0-L4 测试。

Expected: the AIV/CCU kernel sources are checked; when they are missing the affected cases are
blocked, no host-NBI measurement is produced in their place, and the rest of the matrix still
runs.

Tested by `test_scenario_f_missing_kernel_does_not_fake_a_measurement`,
`test_missing_device_kernel_blocks_only_those_cases`,
`test_kernel_build_is_never_told_to_clean`.

## G — profiling

> 对 EP8、EP64、EP128 的主要算子执行 Profiling。

Expected: `profile`; a recorded selection (one task per EP size × level × operator), every
requested EP size covered, only `passed` tasks, and a profile summary with collected/failed
counts. Per-EP `profiling.enabled` flags are recorded but do not decide coverage.

Tested by `test_scenario_g_profiling_covers_every_ep_size`,
`test_profiling_selection_is_deterministic_and_passing_only`.

## H — no duplicate run

> 完整跑一次 Benchmark 并生成报告。

Expected: exactly one `run` command, no `pipeline`, one run id across every phase, and a
second attempt on a completed run is `skipped`, not re-executed.

Tested by `test_scenario_h_one_run_only`, `test_a_full_npu_workflow_runs_every_phase_once`,
`test_a_completed_run_is_not_executed_again`.

## I — deliver after failures

> (a run where tasks report `dependency_missing`, `runtime_failed`, `timeout`)

Expected: finished results are kept; blocked tasks are never counted as passed; a retryable
failure can be resumed; the report is still generated and `final_validation` is `partial`
with each failure named.

Tested by `test_scenario_i_a_failing_run_still_delivers`,
`test_a_partial_delivery_reports_what_failed`,
`test_non_zero_exit_with_terminal_tasks_still_delivers`.

## J — CPU host

> 在当前 CPU 环境测试整个框架。

Expected: `smoke_test` on `configs/smoke_cpu.yaml`; CPU/Gloo tasks run; the kernel phase is
`skipped` (no device kernel needed); profiling is `none`; the environment records
`device: cpu` and a CPU process capacity source; no Ascend performance claim is made.

Tested by `test_scenario_j_cpu_run_is_labelled_as_framework_validation`,
`test_cpu_capacity_is_a_process_policy_not_a_device_claim`.

## Recorded real runs

`docs/workflow.md` section "Recorded runs" keeps the output of two executed workflows on the
development host: a complete CPU smoke flow, and an NPU request that stops honestly because
the host has no Ascend device.
