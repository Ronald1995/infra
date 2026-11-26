"""Topology discovery: turn observed host names into the five-state provenance.

The plan can *assume* a layout (`server_size` -> `synthetic`) or be given one
(`user_provided`).  At run time the workers can do better: every rank knows the host it
runs on, and gathering those names is a real observation, not a guess.  This module
turns ``rank -> host`` into ``topology.server_ids`` with ``source: discovered`` and
``verified: true``.

Precedence is deliberate:

* a ``user_provided`` map always wins (the deployment says what the physical layout is;
  the workers only see host names);
* a ``synthetic``/``unknown``/absent map is replaced by the discovered one, because a
  measurement should use the layout it actually ran on.

When every rank reports the same host the result is still a discovery: one server.  It
is recorded as such instead of being confused with "no topology".
"""

#: States a discovery may replace.
REPLACEABLE = ("synthetic", "unknown", "inferred", None)


def gather_hosts(dist, size, group=None, hostname=None):
    """``rank -> host`` from the process group, or ``None`` when it cannot be read."""
    if hostname is None:
        import socket

        hostname = socket.gethostname()
    if size <= 1:
        return [hostname]
    if dist is None or not getattr(dist, "is_initialized", lambda: False)():
        return None
    try:
        hosts = [None] * size
        dist.all_gather_object(hosts, hostname, group=group)
    except (RuntimeError, ValueError, TypeError):
        return None
    if len(hosts) != size or any(not isinstance(host, str) or not host for host in hosts):
        return None
    return hosts


def apply_discovered_topology(config, hosts):
    """Return the configuration with a discovered topology, or an unchanged one.

    ``hosts`` is ``rank -> host`` or ``None``.  The input configuration is not mutated.
    """
    current = dict(config.get("topology") or {})
    if (not hosts or len(hosts) != config["distributed"]["ep_size"]
            or any(not isinstance(host, str) or not host for host in hosts)):
        return config
    source = current.get("source")
    if source == "user_provided" or (current.get("server_ids") and source == "discovered"):
        return config
    if source not in REPLACEABLE:
        return config
    order, domains = [], {}
    for host in hosts:
        if host not in domains:
            domains[host] = len(order)
            order.append(host)
    server_ids = [domains[host] for host in hosts]
    counts = {domain: server_ids.count(domain) for domain in set(server_ids)}
    topology = {**current, "source": "discovered", "verified": True,
                "server_ids": server_ids, "server_size": max(counts.values()),
                "host_ids": order, "rank_to_host": list(hosts)}
    return {**config, "topology": topology}
