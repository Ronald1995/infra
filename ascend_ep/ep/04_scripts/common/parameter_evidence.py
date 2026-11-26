"""Requested vs observed values for the parameters a task claims to sweep.

A plan can say ``cores: [1, 2, 4]`` and ``outstanding: [1, 4]``; that is a claim about
what the *runtime* will do.  This module records, per task, what was requested and what
the runtime actually used, and marks each dimension ``verified`` or ``unverified``:

* ``verified``   -- the handler recorded the value it used at the runtime boundary
  (an API argument, a session/channel setting, a kernel parameter);
* ``unverified`` -- no runtime record exists for that parameter, so the plan's value is
  an intention, not evidence.

Copying the configuration into ``observed`` is explicitly *not* verification: a
dimension only becomes verified when the handler itself reports the value it used.
A task that sweeps no audited dimension claims nothing and is trivially consistent.
"""
#: Parameters whose runtime consumption is audited.
DIMENSIONS = ("message_bytes", "outstanding", "peer_count", "cores", "quiet_mode",
              "accelerator", "mechanism")
_MISSING = object()


def requested(params):
    """The audited parameters a task asks for (``None`` values are not claims)."""
    return {name: value for name, value in (params or {}).items()
            if name in DIMENSIONS and value is not None}


def observed_from(params, **values):
    """Build the ``observed`` mapping from the values a handler actually used."""
    return {name: value for name, value in values.items()
            if name in DIMENSIONS and name in requested(params) and value is not None}


def record(params, observed=None):
    """``{requested, observed, verification}`` per audited dimension, plus a verdict."""
    wanted = requested(params)
    observed = observed if isinstance(observed, dict) else {}
    entries = {}
    for name, value in wanted.items():
        actual = observed.get(name, _MISSING)
        if actual is _MISSING:
            entries[name] = {"requested": value, "observed": None,
                             "verification": "unverified",
                             "reason": "the runtime did not record this parameter, so the "
                                       "configured value is an intention, not evidence"}
        elif actual == value:
            entries[name] = {"requested": value, "observed": actual,
                             "verification": "verified"}
        else:
            entries[name] = {"requested": value, "observed": actual,
                             "verification": "unverified",
                             "reason": "the runtime recorded a different value"}
    unverified = sorted(name for name, entry in entries.items()
                        if entry["verification"] == "unverified")
    return {"parameters": entries, "verified": not unverified, "unverified": unverified}
