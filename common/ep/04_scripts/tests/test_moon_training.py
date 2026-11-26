"""CPU checks of planner offsets and differentiable weight/gradient plumbing."""
from types import SimpleNamespace

import pytest

from common.config import DEFAULT, merge

torch = pytest.importorskip("torch", exc_type=ImportError)
from common.validation import CorrectnessError, validate_moe
from common.workload import make_workload
from operators.moonep_dynamic_planner.training import local_expert_counts, run_training_step


def test_plan_prefix_maps_home_and_slot_groups():
    assert local_expert_counts([0, 0, 2, 5, 9, 9], experts=4, rank=1, local=2) == (2, 3, 4, 0)
    with pytest.raises(CorrectnessError, match="non-local"):
        local_expert_counts([1, 1, 2, 5, 8, 8], experts=4, rank=1, local=2)
    with pytest.raises(CorrectnessError, match="number"):
        local_expert_counts([0], experts=4, rank=1, local=2)


def test_planner_training_adjoint_and_replica_gradients_on_cpu():
    config = merge(DEFAULT, {
        "device": "cpu", "model": {"num_experts": 2, "top_k": 2, "hidden_size": 8,
                                    "intermediate_size": 8, "dtype": "float32"},
        "workload": {"tokens_per_rank": 4},
    })
    ctx = SimpleNamespace(rank=0, size=1, device=torch.device("cpu"), config=config, torch=torch)
    workload = make_workload(ctx)
    home = (workload.w_gate, workload.w_up, workload.w_down)

    class LocalProjections:
        weights = home
        received = None

        def grad_inputs(self):
            # Exercise the bookkeeping by assigning the second logical expert
            # to replica slot 0. Empty home and slot groups get zero gradients.
            slots = [torch.stack((value[1].detach(), torch.zeros_like(value[0]))) for value in home]
            return tuple(value.detach().requires_grad_() for value in (*home, *slots))

        def pack_gradients(self, gradients):
            self.received = gradients

        def home_gradients(self):
            assert self.received is not None
            reduced = tuple(
                torch.stack((local[0], local[1] + slot[0]))
                for local, slot in zip(self.received[:3], self.received[3:])
            )
            return reduced

    class LocalBuffer:
        def dispatch(self, x, weights=None, *, plan):
            indices = plan.permutation
            return x[indices // 2], None if weights is None else weights.flatten()[indices], None, plan

        def combine(self, plan, rows, route_weights_nvs=None):
            indices = plan.permutation
            value = rows.new_zeros(workload.hidden_states.shape).index_add(0, indices // 2, rows)
            weights = (None if route_weights_nvs is None else
                       route_weights_nvs[indices.argsort()].view_as(workload.topk_weights))
            return value, weights, None

        def prefetch_projections(self, plan, weights):
            assert plan.experts_to_copy.tolist() == [[1, -1]]

        def reduce_projection_grads(self, plan, grad_pairs):
            assert plan.experts_to_copy.tolist() == [[1, -1]]

    buffer = LocalBuffer()
    projections = LocalProjections()
    projections.grad_pairs = ()

    def dispatch(work):
        tokens = len(work.hidden_states)
        plan = SimpleNamespace(permutation=work.topk_ids.flatten().argsort(stable=True),
                               experts_to_copy=torch.tensor([[1, -1]]))
        rows, weights, _, _ = buffer.dispatch(work.hidden_states, work.topk_weights, plan=plan)
        return rows, weights, torch.tensor([tokens, tokens, 2 * tokens, 2 * tokens]), plan

    backend = SimpleNamespace(context=ctx, buffer=buffer, projections=projections, planned_dispatch=dispatch)
    result = run_training_step(backend, workload)
    checks = validate_moe(ctx, workload, result.output, result.gradients, use_router=True)
    assert result.received_rows == 8
    assert checks["scope"] == "cpu_fp32_oracle"
    assert result.timing_ns() == {}


def test_mindspeed_bridge_returns_native_leaf_gradients():
    from operators.hierarchical_alltoallv_mindspeed.native import OverlapLayer

    hidden = torch.randn(3, 2, requires_grad=True)
    probabilities = torch.rand(3, 1, requires_grad=True)
    weights = [torch.randn(1, 2, 2, requires_grad=True) for _ in range(3)]

    def layer(x, _, probs, first, second):
        gate, up = (x @ first[0]).chunk(2, dim=-1)
        return (torch.nn.functional.silu(gate) * up) @ second[0] * probs

    ids = torch.zeros_like(probabilities, dtype=torch.long)
    native = SimpleNamespace(native_forward=layer)
    actual = OverlapLayer.apply(hidden, ids, probabilities, *weights, native)
    first = torch.cat(weights[:2], dim=1).transpose(-1, -2)
    expected = layer(hidden, ids, probabilities, first, weights[2].transpose(-1, -2))
    leaves = (hidden, probabilities, *weights)
    torch.testing.assert_close(actual, expected)
    for got, want in zip(torch.autograd.grad(actual.sum(), leaves), torch.autograd.grad(expected.sum(), leaves)):
        torch.testing.assert_close(got, want)
