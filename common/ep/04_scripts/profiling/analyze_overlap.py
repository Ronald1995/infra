"""Interval arithmetic for concurrent streams on the same rank."""
from collections import Counter, defaultdict


def union(intervals):
    output = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if output and start <= output[-1][1]:
            output[-1] = (output[-1][0], max(end, output[-1][1]))
        else:
            output.append((start, end))
    return output


def length(intervals):
    return sum(end - start for start, end in intervals)


def intersection(a, b):
    i, j, total = 0, 0, 0
    while i < len(a) and j < len(b):
        left, right = a[i], b[j]
        total += max(0, min(left[1], right[1]) - max(left[0], right[0]))
        if left[1] <= right[1]:
            i += 1
        else:
            j += 1
    return total


def analyze(events):
    ranks = defaultdict(list)
    for event in events:
        ranks[event.rank].append(event)
    report = {}
    for rank, rows in sorted(ranks.items()):
        per_category = defaultdict(list)
        kernels = Counter()
        host_by_id = {}
        for row in sorted(rows, key=lambda e: e.start_ns):
            if row.kernel_category == "host":
                if row.correlation_id:
                    host_by_id[row.correlation_id] = row.start_ns
            else:
                per_category[row.kernel_category].append((row.start_ns, row.start_ns + row.duration_ns))
                kernels[row.kernel_name] += row.duration_ns
        comm = union(per_category["communication"])
        compute = union(per_category["compute"])
        busy = union(interval for intervals in per_category.values() for interval in intervals)
        span = busy[-1][1] - busy[0][0] if busy else None
        overlap = intersection(comm, compute)
        lengths = (length(comm), length(compute))
        gaps = [row.start_ns - host_by_id[row.correlation_id] for row in rows
                if row.kernel_category != "host" and row.correlation_id in host_by_id
                and row.start_ns >= host_by_id[row.correlation_id]]
        report[str(rank)] = {
            "source": "profiler_trace", "unit": "ns", "span_ns": span,
            "visible_communication_ns": lengths[0] if busy else None,
            "visible_compute_ns": lengths[1] if busy else None,
            "visible_overlap_ns": overlap if busy else None,
            "unoverlapped_communication_ns": lengths[0] - overlap if busy else None,
            "visible_overlap_ratio": overlap / lengths[0] if lengths[0] else None,
            "internal_overlap_ratio": None, "device_idle_ns": span - length(busy) if span else None,
            "device_idle_ratio": (span - length(busy)) / span if span else None,
            "host_to_device_start_ns": sum(gaps) / len(gaps) if gaps else None,
            "top_kernels_by_cumulative_ns": kernels.most_common(10),
        }
    return report
