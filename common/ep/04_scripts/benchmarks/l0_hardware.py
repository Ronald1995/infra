"""Standalone collective, copy, GMM, activation and concurrency cases."""
import torch

from baselines.reference.collectives import CollectiveBackend, scatter_sum
from baselines.reference.experts import activate, gmm1, gmm2, grouped_linear_gradient
from common.backend import BenchmarkUnavailable
from common.validation import CorrectnessError, error


def hardware_case(ctx, case, work, size, variant):
    op, n, rank = case.operation, ctx.size, ctx.rank
    if op.startswith("shmem"):
        from baselines.l0.shmem import prepare
        run, validate, cleanup = prepare(ctx, size, variant)
        return run, validate, cleanup, size * n

    if op == "grouped_matmul" or op == "swiglu" or op == "concurrent_gmm_communication":
        ref = CollectiveBackend(None, "alltoall")
        ref.initialize(ctx)
        dispatched = ref.dispatch(work)
        rows, counts = dispatched.expert_input.detach().requires_grad_(), dispatched.expert_token_counts
        gate_up = gmm1(work, rows, counts, ctx).detach().requires_grad_()
        activation = activate(gate_up, work.limit).detach().requires_grad_()
        if op == "swiglu":
            act = lambda: activate(gate_up, work.limit)
            upstream = torch.ones_like(activation)
            expected = torch.nn.functional.silu(gate_up.float().chunk(2, dim=-1)[0].clamp(max=work.limit))
            expected = expected * gate_up.float().chunk(2, dim=-1)[1].clamp(-work.limit, work.limit)
            if variant == "backward":
                run = lambda: torch.autograd.grad(act(), gate_up, upstream)[0]
                expected = torch.autograd.grad(expected, gate_up, upstream.float())[0]
            else:
                run = act
        elif op == "concurrent_gmm_communication":
            if n < 2 or ctx.device.type == "cpu":
                raise BenchmarkUnavailable("unsupported", "multi-stream NPU baseline requires at least two devices")
            source = torch.full((size,), rank % 127, dtype=torch.uint8, device=ctx.device)
            dest = torch.empty_like(source)
            stream = torch.npu.Stream(device=ctx.device)

            def run():
                stream.wait_stream(torch.npu.current_stream())
                with torch.npu.stream(stream):
                    handle = ctx.dist.all_to_all_single(dest, source, group=ctx.ep_group, async_op=True)
                value = gmm1(work, rows, counts, ctx)
                handle.wait()
                torch.npu.current_stream().wait_stream(stream)
                return value

            expected = gmm1(work, rows, counts, ctx)
        else:
            first = variant.startswith("gmm1")
            x = rows if first else activation
            weights = (work.w_gate, work.w_up) if first else (work.w_down,)
            fwd = lambda: (gmm1 if first else gmm2)(work, x, counts, ctx)
            cpu_x = x.detach().cpu().float().requires_grad_()
            cpu_weights = [w.detach().cpu().float().requires_grad_() for w in weights]
            packed = torch.cat(cpu_weights, dim=1) if first else cpu_weights[0]
            expected = torch.cat([torch.nn.functional.linear(piece, weight)
                                  for piece, weight in zip(cpu_x.split(counts), packed)]) if counts else (
                                      cpu_x.new_empty((0, packed.shape[1])) +
                                      (cpu_x.sum() + packed.sum()) * 0)
            run = fwd
            if variant.endswith(("dgrad", "wgrad")):
                derivative = variant.split("_")[-1]
                reference_inputs = (cpu_x,) if derivative == "dgrad" else (packed,)
                native_weights = torch.cat(weights, dim=1) if first else weights[0]
                upstream = x.new_ones(expected.shape)
                run = lambda: grouped_linear_gradient(x, native_weights, counts, upstream, derivative, ctx)
                expected = torch.autograd.grad(expected.sum(), reference_inputs)

        def validate():
            got = run()
            expected_values = expected if isinstance(expected, tuple) else (expected,)
            got_values = got if isinstance(got, tuple) else (got,)
            threshold = 0.04 if work.hidden_states.dtype != torch.float32 else 2e-5
            errors = [error(a, b, threshold) for a, b in zip(got_values, expected_values)]
            if op == "concurrent_gmm_communication":
                stripe = size // n
                peer_values = (torch.arange(n, device=ctx.device) % 127).to(torch.uint8)
                if not torch.equal(dest, peer_values.repeat_interleave(stripe)):
                    raise CorrectnessError("concurrent AllToAll payload did not match")
            return {"relative_l2": errors}

        return run, validate, lambda: None, size * (n - 1) if op.startswith("concurrent") else None

    value = torch.full((size,), rank % 127 + 1, dtype=torch.uint8, device=ctx.device)
    if op == "device_copy":
        result = torch.empty_like(value)
        run = lambda: result.copy_(value)
        expected = value
        payload = size * 2 * n
    else:
        if n < 2:
            raise BenchmarkUnavailable("unsupported", "communication baseline needs at least two ranks")
        if op in ("p2p_unidirectional", "p2p_bidirectional", "multi_peer"):
            return p2p(ctx, op, value)
        if op == "alltoall":
            if size % n:
                raise BenchmarkUnavailable("unsupported", "AllToAll message must divide EP size")
            result = torch.empty_like(value)
            run = lambda: ctx.dist.all_to_all_single(result, value, group=ctx.ep_group)
            expected = torch.arange(1, n + 1, device=ctx.device).sub(1).remainder(127).add(1)
            expected = expected.to(torch.uint8).repeat_interleave(size // n)
            payload = (size - size // n) * n
        elif op == "alltoallv":
            # Rotate a deliberately skewed split while preserving the total bytes per rank.
            shares = [size * (i + 1) // (n * (n + 1) // 2) for i in range(n)]
            shares[-1] += size - sum(shares)
            send = [shares[(peer - rank) % n] for peer in range(n)]
            recv = [shares[(rank - peer) % n] for peer in range(n)]
            result = torch.empty_like(value)
            run = lambda: ctx.dist.all_to_all_single(result, value, recv, send, group=ctx.ep_group)
            expected = torch.cat([value.new_full((count,), peer % 127 + 1)
                                  for peer, count in enumerate(recv)])
            payload = (size - shares[0]) * n
        elif variant == "allgather":
            buffers = [torch.empty_like(value) for _ in range(n)]
            run = lambda: ctx.dist.all_gather(buffers, value, group=ctx.ep_group)
            result = buffers
            expected = [value.new_full(value.shape, peer % 127 + 1) for peer in range(n)]
            payload = size * n * (n - 1)
        else:
            inputs = value.repeat(n).view(n, -1)
            result = torch.empty_like(value)
            if ctx.device.type == "npu":
                run = lambda: ctx.dist.reduce_scatter_tensor(result, inputs.flatten(), group=ctx.ep_group)
            else:
                run = lambda: result.copy_(scatter_sum(inputs, ctx))
            expected = value.new_full(value.shape, sum(peer % 127 + 1 for peer in range(n)) % 256)
            payload = size * n * (n - 1)

    def validate():
        run()
        got = result if isinstance(result, list) else [result]
        want = expected if isinstance(expected, list) else [expected]
        if any(not torch.equal(a, b) for a, b in zip(got, want)):
            raise CorrectnessError(f"{case.id} output bytes disagree with expected collective payload")
        return {"checked": "byte_exact"}

    return run, validate, lambda: None, payload


def p2p(ctx, operation, value):
    n, rank, dist = ctx.size, ctx.rank, ctx.dist
    if operation == "multi_peer":
        sends = receives = [r for r in range(n) if r != rank]
    elif operation == "p2p_bidirectional":
        sends = receives = [1 - rank] if rank < 2 else []
    else:
        sends, receives = ([1] if rank == 0 else []), ([0] if rank == 1 else [])
    buffers = {peer: torch.empty_like(value) for peer in receives}

    def run():
        requests = [dist.P2POp(dist.irecv, buffers[p], p, ctx.ep_group) for p in receives]
        requests.extend(dist.P2POp(dist.isend, value, p, ctx.ep_group) for p in sends)
        if requests:
            for request in dist.batch_isend_irecv(requests):
                request.wait()

    def validate():
        run()
        for peer, buffer in buffers.items():
            if not torch.equal(buffer, value.new_full(value.shape, peer % 127 + 1)):
                raise CorrectnessError(f"P2P received corrupt payload from rank {peer}")
        return {"checked": "byte_exact", "receivers": receives}

    transfers = {"multi_peer": n * (n - 1), "p2p_bidirectional": 2, "p2p_unidirectional": 1}
    return run, validate, lambda: None, value.numel() * transfers[operation]
