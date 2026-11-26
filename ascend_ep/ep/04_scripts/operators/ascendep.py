"""Pinned ascendep native forward adapters, vendored under ``operators/vendor/ascendep``.

The adapter modules themselves live in ``operators/vendor/ascendep/ep/`` as a byte-exact
copy of ``https://github.com/penn513/ascendep`` at :data:`UPSTREAM_COMMIT`, so a benchmark
run needs no external checkout. ``operators/vendor/ascendep/VENDOR.json`` records that
provenance and ``scripts/vendor_ascendep.py --check`` verifies it.

``EP_ASCENDEP_ROOT`` is still honoured as a development override. It is validated the same
way (``ep/runner.py`` present, clean git checkout at the pinned commit) and reported
separately, so a result can never confuse a vendored adapter with an external one:
:func:`source_info` returns ``{"origin": "vendored" | "override", ...}``.

Importing this module stays CPU-only: the adapter modules, ``torch_npu`` and every native
library are imported lazily from :meth:`AscendEPBackend.initialize`, never at import time.
"""
from dataclasses import dataclass
from functools import lru_cache
from importlib import import_module
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from common.backend import BenchmarkUnavailable, CombineResult, DispatchResult, EPBackend

UPSTREAM_URL = "https://github.com/penn513/ascendep"
UPSTREAM_COMMIT = "24ff84d3ab8c01e1e2071802465173e624e1f89d"
OVERRIDE_ENV = "EP_ASCENDEP_ROOT"

OPERATORS = Path(__file__).resolve().parent
VENDORED_ROOT = OPERATORS / "vendor" / "ascendep"
VENDORED_MANIFEST = VENDORED_ROOT / "VENDOR.json"
ENTRYPOINT = Path("ep") / "runner.py"


@dataclass(frozen=True)
class SourceInfo:
    """Which ascendep adapter tree a backend imports, and where that tree came from."""

    origin: str
    commit: str
    path: str
    upstream: str = UPSTREAM_URL
    pinned: str = UPSTREAM_COMMIT
    manifest: str | None = None

    def as_dict(self):
        """JSON friendly form, suitable as a field of a benchmark result."""
        return {
            "origin": self.origin,
            "commit": self.commit,
            "pinned": self.pinned,
            "path": self.path,
            "upstream": self.upstream,
            "manifest": self.manifest,
        }


def git_revision(root):
    """Commit of a git checkout, refusing one with local changes under ``ep/``."""
    try:
        revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"],
                                           encoding="ascii", timeout=10).strip()
        dirty = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain",
                                         "--untracked-files=no", "--", "ep"],
                                        encoding="utf-8", timeout=10).strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise BenchmarkUnavailable(
            "source_unavailable",
            f"cannot identify the ascendep checkout selected by {OVERRIDE_ENV} at {root}: {exc}") from exc
    if dirty:
        raise BenchmarkUnavailable(
            "source_unavailable",
            f"{OVERRIDE_ENV}={root} has uncommitted changes under ep/: {dirty.splitlines()[0]}") from None
    return revision


def _vendored_commit(root):
    """Commit recorded by the vendored manifest, after checking that it describes the tree."""
    manifest = root / VENDORED_MANIFEST.name
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise BenchmarkUnavailable(
            "source_unavailable",
            f"cannot read the vendored ascendep provenance {manifest}: {exc}; "
            "run `python scripts/vendor_ascendep.py --check`") from exc
    commit = payload.get("upstream", {}).get("commit")
    if not commit:
        raise BenchmarkUnavailable("source_unavailable",
                                   f"{manifest} records no upstream commit for the vendored adapters")
    absent = [entry.get("path") for entry in payload.get("files", [])
              if not (root / str(entry.get("path"))).is_file()]
    if absent:
        raise BenchmarkUnavailable(
            "source_unavailable",
            f"the vendored ascendep copy is incomplete, {len(absent)} file(s) missing "
            f"({', '.join(str(path) for path in absent[:4])}); run "
            "`python scripts/vendor_ascendep.py --check`") from None
    return commit


@lru_cache(maxsize=None)
def _resolve(override):
    """Locate the adapter tree once: vendored by default, an explicit override when given."""
    if override:
        root = Path(override).expanduser().resolve()
        if not (root / ENTRYPOINT).is_file():
            raise BenchmarkUnavailable(
                "source_unavailable",
                f"{OVERRIDE_ENV}={root} has no {ENTRYPOINT.as_posix()}; point it at a pinned "
                "ascendep checkout or unset it to use the vendored copy") from None
        commit = git_revision(root)
        if commit != UPSTREAM_COMMIT:
            raise BenchmarkUnavailable(
                "source_unavailable",
                f"{OVERRIDE_ENV}={root} is at commit {commit}, not the pinned {UPSTREAM_COMMIT}")
        return SourceInfo("override", commit, str(root))
    if not (VENDORED_ROOT / ENTRYPOINT).is_file():
        raise BenchmarkUnavailable(
            "source_unavailable",
            f"the vendored ascendep adapter tree is missing at {VENDORED_ROOT}; reinstall the "
            "package or run `python scripts/vendor_ascendep.py --check`") from None
    commit = _vendored_commit(VENDORED_ROOT)
    if commit != UPSTREAM_COMMIT:
        raise BenchmarkUnavailable(
            "source_unavailable",
            f"{VENDORED_MANIFEST} pins {commit}, but this module expects {UPSTREAM_COMMIT}; "
            "refresh the vendored copy with `python scripts/vendor_ascendep.py --refresh`") from None
    return SourceInfo("vendored", commit, str(VENDORED_ROOT), manifest=str(VENDORED_MANIFEST))


def _override(value):
    if value is None:
        value = os.getenv(OVERRIDE_ENV, "")
    return str(value).strip() or None


def reset_source_cache():
    """Forget the memoised resolution so the next call re-reads ``EP_ASCENDEP_ROOT``."""
    _resolve.cache_clear()


def source_info(value=None):
    """Report the adapter source as a dict: ``origin``, ``commit``, ``path`` and provenance.

    ``origin`` is ``"vendored"`` for the in-tree copy and ``"override"`` when
    ``EP_ASCENDEP_ROOT`` selects a checkout. Raises :class:`BenchmarkUnavailable` with
    status ``source_unavailable`` when the tree is missing, incomplete or off-commit.
    """
    return _resolve(_override(value)).as_dict()


def source_checkout(value=None):
    """Absolute path of the ascendep tree whose ``ep.<name>.demo`` modules are imported."""
    return Path(_resolve(_override(value)).path)


def adapter_module(spec):
    """Dotted module the registry selects for one operator, e.g. ``ep.deepep_ascend.demo``."""
    return f"ep.{spec.upstream or spec.id}.demo"


def _import_demo(name, source):
    """Import one vendored adapter, naming the real blocker when it cannot be imported."""
    try:
        return import_module(name)
    except ImportError as exc:
        missing = getattr(exc, "name", None) or ""
        if missing == "ep" or missing.startswith("ep."):
            raise BenchmarkUnavailable(
                "source_unavailable",
                f"the {source['origin']} ascendep tree {source['path']} does not provide {name} "
                f"({exc}); run `python scripts/vendor_ascendep.py --check`") from exc
        raise BenchmarkUnavailable(
            "dependency_missing",
            f"ascendep adapter {name} needs the native module {missing or exc!r}, which is not "
            f"installed: {exc}. Native libraries are listed in dependencies/manifest.yaml") from exc


class AscendEPBackend(EPBackend):
    def initialize(self, context):
        super().initialize(context)
        self.layer = None
        self.source = source_info()
        root = Path(self.source["path"])
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        name = adapter_module(self.spec)
        self.demo = _import_demo(name, self.source)
        located = Path(self.demo.__file__).resolve()
        if not located.is_relative_to(root):
            raise BenchmarkUnavailable(
                "source_unavailable",
                f"{name} resolved to {located}, outside the {self.source['origin']} ascendep tree "
                f"{root}; another ep package shadows it") from None

    def prepare(self, work):
        torch = self.context.torch
        self.layer = SimpleNamespace(
            down_proj=work.w_down, hidden_dim=work.hidden_states.shape[1],
            num_global_experts=len(work.route.placement), limit=work.limit,
            act_fn=torch.nn.functional.silu,
            expert_ids_per_ep_rank=torch.arange(len(work.local_experts), device=self.context.device).repeat(
                self.context.size
            ),
        )
        if self.spec.id in {"moonep", "deepep_cann"} and (
            sum(work.route.tokens) * work.topk_ids.shape[1] % len(work.route.placement)
        ):
            raise BenchmarkUnavailable("unsupported", "native adapter requires equal expert receive counts")

    def moe_forward(self, work):
        # Packing is required on each call so the weight gradients retain their graph.
        torch = self.context.torch
        self.layer.gate_up_proj = torch.cat((work.w_gate, work.w_up), dim=1)
        return self.demo.forward(self.layer, work.hidden_states, work.topk_ids, work.topk_weights)

    def finalize(self):
        if self.layer is not None:
            release = getattr(self.layer, "release_buffer", None)
            if release is not None:
                release()


class MC2V2Backend(AscendEPBackend):
    def prepare(self, work):
        super().prepare(work)
        self.group = None

    def exchange(self, work):
        if len(work.hidden_states) > 512:
            raise BenchmarkUnavailable("unsupported", "published 950DT V2 adapter limit is 512 tokens per native invocation")
        if self.group is None:
            self.group = self.context.dist.new_group(backend="hccl")
        return self.demo.Exchange(work.topk_ids, len(work.route.placement), self.group)

    def dispatch(self, work, cached=None):
        exchange = self.exchange(work)
        rows = self.demo.Dispatch.apply(work.hidden_states, exchange)
        return self.result(work, rows, exchange.sizes, exchange.slots, exchange)

    def result(self, work, rows, counts, slots, handle):
        torch = self.context.torch
        experts = torch.repeat_interleave(torch.tensor(work.local_experts, device=rows.device),
                                         torch.tensor(counts, device=rows.device))
        return DispatchResult(rows, tuple(counts), handle, "local_expert_contiguous", experts, slots)

    def quantized_dispatch(self, work):
        import torch_npu

        exchange = self.exchange(work)
        return torch_npu.npu_moe_distribute_dispatch_v2(
            work.hidden_states.contiguous(), exchange.ids, *exchange.config,
            expert_scales=exchange.labels, shared_expert_num=0, global_bs=exchange.batch,
            expert_token_nums_type=1, quant_mode=2,
        )

    def validate_quantized_dispatch(self, work, native, trace):
        from common.validation import CorrectnessError, validate_dispatch

        rows, scales, inverse, counts, _, _, labels = native
        counts = counts.tolist()
        used = sum(counts)
        if rows.dtype != self.context.torch.int8 or scales[:used].numel() != used:
            raise CorrectnessError("quantization mode 2 must return INT8 rows and one scale per route")
        dequantized = rows[:used].float() * scales[:used].reshape(used, 1).float()
        result = self.result(work, dequantized, counts, labels[:used].long(), inverse)
        return validate_dispatch(work, result, self.context, trace, threshold=.015)

    def combine(self, work, dispatched, rows):
        from baselines.reference.collectives import Gather

        exchange = dispatched.routing_handle
        weights = Gather.apply(work.topk_weights, self.context).flatten()[exchange.slots]
        output = self.demo.Combine.apply(rows * weights[:, None], exchange)
        return CombineResult(output)

    def finalize(self):
        try:
            super().finalize()
        finally:
            if getattr(self, "group", None) is not None:
                self.context.dist.destroy_process_group(self.group)


class FusedGMMBackend(AscendEPBackend):
    def route(self, work):
        from baselines.reference.collectives import exchange

        ctx = self.context
        torch = ctx.torch
        ids = work.topk_ids.flatten()
        order = ids.argsort(stable=True)
        send = torch.bincount(ids, minlength=len(work.route.placement))
        n = len(work.local_experts)
        recv = exchange(send, [n] * ctx.size, [n] * ctx.size, ctx)
        counts = recv.view(ctx.size, -1).sum(0)
        return order, send, recv, counts

    def dispatch_gmm(self, work):
        order, send, recv, counts = self.route(work)
        torch = self.context.torch
        w = torch.cat((work.w_gate, work.w_up), dim=1)
        x = work.hidden_states[order // work.topk_ids.shape[1]]
        ctx = self.context
        if self.spec.id == "alltoallv_gmm_cann":
            name = ctx.ep_group._get_backend(x.device).get_hccl_comm_name(ctx.rank)
            return self.demo.DispatchLinear.apply(x, w, name, ctx.size, send.tolist(), recv.tolist(), counts)
        self.ensure_udma()
        route = self.demo.Route(work.topk_ids, len(work.route.placement))
        return self.demo.DispatchLinear.apply(x, w, route)

    def gmm_combine(self, work, activation):
        order, send, recv, counts = self.route(work)
        ctx = self.context
        if self.spec.id == "alltoallv_gmm_cann":
            name = ctx.ep_group._get_backend(activation.device).get_hccl_comm_name(ctx.rank)
            result = self.demo.CombineLinear.apply(
                activation, work.w_down, name, ctx.size, recv.tolist(), send.tolist(), counts
            )
        else:
            self.ensure_udma()
            route = self.demo.Route(work.topk_ids, len(work.route.placement))
            result = self.demo.CombineLinear.apply(activation, work.w_down, route)
        result = result[order.argsort()].reshape(*work.topk_ids.shape, work.w_down.shape[1])
        return (result * work.topk_weights.unsqueeze(-1)).sum(1)

    def tuned_dispatch(self, work, variant):
        from deep_ep_experimental.ep_gmm_fused._tiling import default_tiling, validate_tiling

        self.ensure_udma()
        ctx = self.context
        hidden = work.hidden_states.shape[-1]
        columns = 2 * work.w_gate.shape[1]
        case = {
            "operator": "all_to_allv_grouped_mat_mul",
            "shape": {"m": work.topk_ids.numel(), "n": columns, "k": hidden},
            "expert_num": len(work.route.placement), "world_size": ctx.size, "output_all_to_allv": False,
        }
        tiling = default_tiling(case)
        if variant.startswith("tile_"):
            tiling.update(zip(("m0", "n0", "k0"), (int(v) for v in variant.split("_")[1:])))
        else:
            tiling["self_copy_cores"] = int(variant.rsplit("_", 1)[1])
        try:
            tiling = validate_tiling(case["operator"], tiling, **case["shape"], world_size=ctx.size)
        except ValueError as exc:
            raise BenchmarkUnavailable("unsupported", f"native tuning contract rejected {variant}: {exc}") from exc

        def execute(value):
            order, send, recv, counts = self.route(value)
            x = value.hidden_states[order // value.topk_ids.shape[1]].contiguous()
            w = ctx.torch.cat((value.w_gate, value.w_up), dim=1).transpose(1, 2).contiguous()
            return self.demo.alltoallv_gmm(
                x, w, send, recv, received_rows=sum(counts.tolist()),
                ep_size=ctx.size, expert_num=case["expert_num"], tiling=tiling,
            )
        return execute

    def ensure_udma(self):
        if self.layer is not None and not hasattr(self.layer, "release_buffer"):
            self.demo.init_udma_group(group=self.context.ep_group)
            self.layer.release_buffer = self.demo.destroy_udma_group
