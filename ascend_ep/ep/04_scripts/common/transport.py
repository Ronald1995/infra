"""The single transport-mechanism registry.

Every other module reads mechanism metadata from here: configuration validation
(which mechanisms exist), variant expansion (which dimensions a mechanism sweeps) and
the tests (does every mechanism have a case, and every case a mechanism).  Before this
module the same list lived in ``config.L0_MECHANISMS`` and a second, hand-maintained
table in ``variants.MECHANISM_DIMENSIONS``, so adding a mechanism meant editing two
places and forgetting one was invisible.

A mechanism is described on four axes, because they are *not* the same abstraction:

* ``programming_model`` -- how the transfer is programmed (ub memory semantics, URMA,
  SHMEM);
* ``protocol``          -- the wire protocol when the mechanism names one;
* ``execution_engine``  -- which engine issues the transfer (AIV, AICPU+TS, CCU, host);
* ``physical_link``     -- the link the bytes actually traverse.

Only ``shmem_udma``'s device variant exposes core-count and async-submit controls in
the current runtime; every other mechanism declares them ``not_applicable`` with the
reason, so no task ever carries a parameter the backend cannot honor.
"""
#: ``id -> metadata``.  Order is the order validation reports.
MECHANISMS = {
    "ub_memory": {
        "programming_model": "ub_memory", "protocol": "UB_MEM", "physical_link": "ub",
        "execution_engines": ("aiv",), "dimensions": {"cores": "not_applicable",
                                                     "quiet_mode": "not_applicable"},
        "link_cases": ("LINK-001",), "overlap_cases": ("OVERLAP-001", "OVERLAP-005"),
        "claim": "memory-semantic access reaches the peer's memory without a collective",
    },
    "urma": {
        "programming_model": "urma", "protocol": "UB_CTP", "physical_link": "ub",
        "execution_engines": ("aiv", "aicpu_ts", "ccu"),
        "dimensions": {"cores": "not_applicable", "quiet_mode": "not_applicable"},
        "link_cases": ("LINK-002", "LINK-005", "LINK-008"),
        "overlap_cases": ("OVERLAP-002", "OVERLAP-003", "OVERLAP-004", "OVERLAP-006"),
        "claim": "URMA remote-memory transfer on the CT protocol path",
    },
    "ub_ctp": {
        "programming_model": "ub_ctp", "protocol": "UB_CTP", "physical_link": "ub",
        "execution_engines": ("aiv", "aicpu_ts"),
        "dimensions": {"cores": "not_applicable", "quiet_mode": "not_applicable"},
        "link_cases": ("LINK-003", "LINK-006"), "overlap_cases": (),
        "claim": "CTP protocol path measured on its own engine pair",
    },
    "ub_rtp": {
        "programming_model": "ub_rtp", "protocol": "UB_RTP", "physical_link": "ub",
        "execution_engines": ("aiv", "aicpu_ts"),
        "dimensions": {"cores": "not_applicable", "quiet_mode": "not_applicable"},
        "link_cases": ("LINK-004", "LINK-007"), "overlap_cases": (),
        "claim": "RTP protocol path measured on its own engine pair",
    },
    "hccs": {
        "programming_model": "hccs", "protocol": "HCCS", "physical_link": "hccs",
        "execution_engines": ("aicpu_ts",),
        "dimensions": {"cores": "not_applicable", "quiet_mode": "not_applicable"},
        "link_cases": ("LINK-009",), "overlap_cases": (),
        "claim": "HCCS link throughput without a collective library in the path",
    },
    "roce": {
        "programming_model": "roce", "protocol": "ROCE", "physical_link": "roce",
        "execution_engines": ("aicpu_ts",),
        "dimensions": {"cores": "not_applicable", "quiet_mode": "not_applicable"},
        "link_cases": ("LINK-010",), "overlap_cases": (),
        "claim": "RDMA-over-Converged-Ethernet network transfer",
    },
    "uboe": {
        "programming_model": "uboe", "protocol": "UBOE", "physical_link": "uboe",
        "execution_engines": ("aicpu_ts",),
        "dimensions": {"cores": "not_applicable", "quiet_mode": "not_applicable"},
        "link_cases": ("LINK-011",), "overlap_cases": (),
        "claim": "UB-over-Ethernet path, kept separate from RoCE and UB",
    },
    "shmem_mte": {
        "programming_model": "shmem", "protocol": None, "physical_link": "ub",
        "execution_engines": ("host_cpu",),
        "dimensions": {"cores": "not_applicable", "quiet_mode": "not_applicable"},
        "link_cases": ("LINK-012",), "overlap_cases": (),
        "claim": "SHMEM MTE data movement through the ACLSHMEM runtime",
    },
    "shmem_udma": {
        "programming_model": "shmem", "protocol": None, "physical_link": "ub",
        "execution_engines": ("aiv",),
        "dimensions": {"cores": "sweep", "quiet_mode": "sweep",
                       "variants": {"device": "sweep", "host": "fixed"}},
        "link_cases": ("LINK-013",), "overlap_cases": ("OVERLAP-007",),
        "claim": "SHMEM UDMA submission with a measurable QP count and async-submit flag",
    },
}

IDS = tuple(MECHANISMS)


def mechanism(identifier):
    if identifier not in MECHANISMS:
        raise ValueError(f"unknown transport mechanism {identifier!r}; expected one of {IDS}")
    return MECHANISMS[identifier]


def dimensions(identifier):
    return dict(mechanism(identifier)["dimensions"])


def dimension(identifier, name, variant=None):
    """``sweep`` / ``fixed`` / ``not_applicable`` for one mechanism dimension."""
    record = mechanism(identifier)["dimensions"]
    kind = record.get(name, "not_applicable")
    if kind == "sweep" and variant is not None:
        kind = (record.get("variants") or {}).get(variant, kind)
    return kind


def cases(identifier=None):
    """All case ids that drive a mechanism (or every mechanism's cases)."""
    if identifier is None:
        return {case for item in MECHANISMS.values()
                for case in (*item["link_cases"], *item["overlap_cases"])}
    item = mechanism(identifier)
    return (*item["link_cases"], *item["overlap_cases"])


def case_mechanism(case_id):
    for identifier, item in MECHANISMS.items():
        if case_id in item["link_cases"] or case_id in item["overlap_cases"]:
            return identifier
    return None
