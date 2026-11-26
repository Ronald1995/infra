"""Slow-rank (straggler) injection: who is slow, where, how much, and did it happen.

A straggler test is not "make one rank do more work": routing imbalance changes the
*amount* of work and a straggler changes the *capability* of a rank.  This controller
owns the second one, and it is deliberately split from :mod:`common.routing` so the
two can be combined later without either one faking the other.

What is real here (CPU-testable, no device):

* selection -- fixed / random / rotating / fraction / server / hotspot / non_hotspot,
  seeded and therefore identical on every rank;
* injection -- ``host_delay`` and ``stage_delay`` at a declared stage boundary, with
  the delay placed *inside* the timed region and after the barrier, so it cannot be
  absorbed by rank alignment;
* observation -- every injection records the requested and the measured delay plus
  whether calibration passed, and the record is part of the rank result.

What is *not* implemented is named, not approximated: ``device_delay`` and the three
contention methods raise ``BenchmarkUnavailable("not_implemented", ...)`` when a
request asks for them, so the test point is recorded instead of silently running
without a straggler.
"""
from dataclasses import dataclass, field
from time import perf_counter_ns, sleep

import numpy as np

from common.backend import BenchmarkUnavailable

#: Injection methods with a real implementation today.
METHODS = ("host_delay", "stage_delay")
#: Methods that require a device-side implementation (device kernel / stream event /
#: resource contention).  They are declared so a request is rejected explicitly.
UNIMPLEMENTED_METHODS = ("device_delay", "compute_contention", "memory_contention",
                         "communication_contention")
#: Selection policies.
SELECTIONS = ("fixed", "random", "rotating", "fraction", "server", "hotspot",
              "non_hotspot")
#: Stage boundaries an adapter may expose.  A stage that the adapter does not expose
#: is ``unsupported_stage_injection``: the fused execution path is never split apart
#: just to inject a delay.
STAGES = ("before_operation", "before_router", "before_dispatch", "before_gmm",
          "before_combine", "after_combine")
DEFAULT_TOLERANCE = 0.10


@dataclass
class StragglerObservation:
    iteration: int
    selected_ranks: list
    stage: str
    method: str
    requested_delay_us: float
    actual_delay_us: float
    injected: bool
    calibration_passed: bool | None
    reason: str | None = None

    def as_dict(self):
        return {"iteration": self.iteration, "selected_ranks": self.selected_ranks,
                "stage": self.stage, "method": self.method,
                "requested_delay_us": self.requested_delay_us,
                "actual_delay_us": self.actual_delay_us, "injected": self.injected,
                "calibration_passed": self.calibration_passed, "reason": self.reason}


def block(config):
    """The ``benchmark.straggler`` block of one effective configuration."""
    return (config.get("benchmark") or {}).get("straggler") or {}


def validate(config):
    """Validate the straggler block; return it (``{}`` when disabled)."""
    settings = block(config)
    if not settings:
        return {}
    if not isinstance(settings, dict):
        raise ValueError("benchmark.straggler must be a mapping")
    enabled = settings.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("benchmark.straggler.enabled must be a boolean")
    selection = settings.get("selection") or {}
    mode = selection.get("mode", "fixed")
    if mode not in SELECTIONS:
        raise ValueError(f"benchmark.straggler.selection.mode must be one of {SELECTIONS}")
    if mode == "fraction":
        fraction = selection.get("fraction")
        if not isinstance(fraction, (int, float)) or not 0 < fraction <= 1:
            raise ValueError("benchmark.straggler.selection.fraction must be in (0, 1]")
    if mode in ("fixed", "server", "hotspot", "non_hotspot"):
        ranks = selection.get("ranks", [])
        if not isinstance(ranks, list) or any(type(rank) is not int or rank < 0
                                              for rank in ranks):
            raise ValueError("benchmark.straggler.selection.ranks must be a list of ranks")
    injection = settings.get("injection") or {}
    method = injection.get("method", "host_delay")
    if method not in METHODS + UNIMPLEMENTED_METHODS:
        raise ValueError(f"benchmark.straggler.injection.method must be one of "
                         f"{METHODS + UNIMPLEMENTED_METHODS}")
    stage = injection.get("stage", "before_operation")
    if stage not in STAGES:
        raise ValueError(f"benchmark.straggler.injection.stage must be one of {STAGES}")
    delay = injection.get("delay_us")
    ratio = injection.get("delay_ratio")
    if delay is None and ratio is None:
        raise ValueError("benchmark.straggler.injection needs delay_us or delay_ratio")
    if delay is not None and (isinstance(delay, bool) or not isinstance(delay, (int, float))
                              or delay < 0):
        raise ValueError("benchmark.straggler.injection.delay_us must be non-negative")
    if ratio is not None and (isinstance(ratio, bool) or not isinstance(ratio, (int, float))
                              or ratio < 0):
        raise ValueError("benchmark.straggler.injection.delay_ratio must be non-negative")
    trigger = settings.get("trigger") or {}
    probability = trigger.get("probability", 1.0)
    if not isinstance(probability, (int, float)) or not 0 <= probability <= 1:
        raise ValueError("benchmark.straggler.trigger.probability must be in [0, 1]")
    every = trigger.get("every_n_iterations", 1)
    if type(every) is not int or every < 1:
        raise ValueError("benchmark.straggler.trigger.every_n_iterations must be >= 1")
    tolerance = (settings.get("calibration") or {}).get("relative_tolerance",
                                                        DEFAULT_TOLERANCE)
    if not isinstance(tolerance, (int, float)) or tolerance < 0:
        raise ValueError("benchmark.straggler.calibration.relative_tolerance must be >= 0")
    return settings


class StragglerController:
    """One rank's view of the injection plan.

    Every rank constructs the controller with the same configuration and its own
    rank id; the selection decisions only depend on the seed and the iteration, so
    the ranks agree on who is slow without a collective.
    """

    def __init__(self, config, rank, size):
        self.config = config
        self.rank = int(rank)
        self.size = int(size)
        settings = validate(config) if block(config) else {}
        self.settings = settings
        self.enabled = bool(settings.get("enabled"))
        selection = settings.get("selection") or {}
        injection = settings.get("injection") or {}
        trigger = settings.get("trigger") or {}
        calibration = settings.get("calibration") or {}
        self.mode = selection.get("mode", "fixed")
        self.fraction = selection.get("fraction", 0.125)
        self.static_ranks = list(selection.get("ranks") or [])
        self.method = injection.get("method", "host_delay")
        self.stage = injection.get("stage", "before_operation")
        self.delay_us = injection.get("delay_us")
        self.delay_ratio = injection.get("delay_ratio")
        self.baseline_us = injection.get("baseline_us")
        self.probability = float(trigger.get("probability", 1.0))
        self.every = int(trigger.get("every_n_iterations", 1))
        self.tolerance = float(calibration.get("relative_tolerance", DEFAULT_TOLERANCE))
        self.seed = int((settings.get("seed") if settings.get("seed") is not None
                         else (config.get("workload") or {}).get("seed", 42)))
        self.observations = []
        if not self.enabled:
            return
        if self.method in UNIMPLEMENTED_METHODS:
            raise BenchmarkUnavailable(
                "not_implemented",
                f"straggler method {self.method!r} needs a device-side implementation; "
                f"only {METHODS} are implemented (host-side) today")
        if self.delay_us is None:
            if self.baseline_us is None:
                raise BenchmarkUnavailable(
                    "not_applicable",
                    "a relative straggler delay needs injection.baseline_us (the measured "
                    "stage latency); no adapter supplies one yet")
            self.delay_us = float(self.delay_ratio) * float(self.baseline_us)
        if self.mode in ("server", "hotspot", "non_hotspot") and not self._has_context():
            raise BenchmarkUnavailable(
                "not_applicable",
                f"straggler selection {self.mode!r} needs the deployment context "
                f"(topology.server_ids / workload.hot_rank)")

    # -- selection ---------------------------------------------------------
    def _has_context(self):
        topology = (self.config.get("topology") or {})
        if self.mode == "server":
            return bool(topology.get("server_ids"))
        return (self.config.get("workload") or {}).get("hot_rank") is not None

    def selected_ranks(self, iteration):
        """The ranks that are slow in ``iteration`` (sorted, deterministic)."""
        if not self.enabled:
            return []
        if self.mode == "fixed":
            return sorted({rank for rank in self.static_ranks if 0 <= rank < self.size})
        if self.mode == "random":
            rng = np.random.default_rng([self.seed, iteration, 991])
            count = max(1, int(round(self.fraction * self.size)))
            return sorted(int(rank) for rank in rng.choice(self.size, count, replace=False))
        if self.mode == "rotating":
            return [(iteration + rank) % self.size for rank in self.static_ranks or [0]]
        if self.mode == "fraction":
            count = max(1, int(round(self.fraction * self.size)))
            step = max(1, self.size // count)
            return sorted({(iteration + index * step) % self.size for index in range(count)})
        if self.mode == "server":
            servers = (self.config.get("topology") or {}).get("server_ids") or []
            anchor = (self.static_ranks or [0])[0]
            domain = servers[anchor] if anchor < len(servers) else None
            return [rank for rank, value in enumerate(servers) if value == domain]
        hot = int((self.config.get("workload") or {}).get("hot_rank", 0))
        if self.mode == "hotspot":
            return [hot]
        return [rank for rank in range(self.size) if rank != hot]

    def should_inject(self, iteration):
        if not self.enabled:
            return False
        if iteration % self.every:
            return False
        if self.probability >= 1.0:
            return True
        if self.probability <= 0.0:
            return False
        rng = np.random.default_rng([self.seed, iteration, 17])
        return bool(rng.random() < self.probability)

    # -- injection ---------------------------------------------------------
    def _inject(self, iteration, stage, reason=None):
        """Sleep on a selected rank, inside the timed region, and record the result."""
        selected = self.selected_ranks(iteration)
        if not self.enabled or self.rank not in selected or not self.should_inject(iteration):
            return None
        started = perf_counter_ns()
        sleep(float(self.delay_us) / 1e6)
        actual = (perf_counter_ns() - started) / 1000.0
        requested = float(self.delay_us)
        tolerance = self.tolerance * requested if requested else 0.0
        observation = StragglerObservation(
            iteration=iteration, selected_ranks=selected, stage=stage, method=self.method,
            requested_delay_us=requested, actual_delay_us=actual, injected=True,
            calibration_passed=abs(actual - requested) <= max(tolerance, 50.0),
            reason=reason)
        self.observations.append(observation)
        return observation.as_dict()

    def before_operation(self, iteration):
        """Called after the barrier and the synchronize, before the operation."""
        if not self.enabled:
            return None
        if self.stage != "before_operation":
            return None
        if self.method != "host_delay":
            return None
        return self._inject(iteration, "before_operation")

    def before_stage(self, stage, iteration):
        """Called by an adapter at a declared stage boundary."""
        if not self.enabled:
            return None
        if stage not in STAGES:
            raise ValueError(f"unknown straggler stage {stage!r}")
        if self.stage != stage or self.method != "stage_delay":
            return None
        return self._inject(iteration, stage)

    def unsupported_stage(self):
        """The record an adapter uses when it cannot expose the requested stage."""
        if not self.enabled:
            return None
        return {"status": "unsupported_stage_injection",
                "stage": self.stage,
                "reason": (f"the adapter does not expose a {self.stage} boundary; the fused "
                           f"execution path is not split just to inject a delay")}

    def summary(self):
        injected = [item for item in self.observations if item.injected]
        return {
            "enabled": self.enabled,
            "mode": self.mode if self.enabled else None,
            "method": self.method if self.enabled else None,
            "stage": self.stage if self.enabled else None,
            "requested_delay_us": (float(self.delay_us) if self.enabled else None),
            "rank": self.rank,
            "injections": len(injected),
            "actual_delay_us_sum": sum(item.actual_delay_us for item in injected),
            "calibration_failures": sum(1 for item in injected
                                        if item.calibration_passed is False),
            "observations": [item.as_dict() for item in self.observations],
        }
