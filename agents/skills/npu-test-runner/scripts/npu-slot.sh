#!/usr/bin/env bash
#
# npu-slot.sh - hand out an isolated "session slot" on a shared Ascend NPU host.
#
# Why this exists
# ---------------
# Several agent/test sessions share one long-lived container (liurong_vllm).
# That container shares these resources host-wide:
#
#   1. the git working tree  - /home is bind-mounted, so every session edits
#                              the same /home/liurong/vllm-ascend checkout
#   2. the NPU chips         - 8 cards x 2 dies = /dev/davinci0..15, all exposed
#   3. TCP ports             - the container runs with --net=host, and the rlhf
#                              e2e conftest defaults to port 8770
#   4. tmp/cache dirs        - /tmp is bind-mounted from the host
#   5. the process namespace - "docker exec" shares one pid space, so
#                              "pkill -f python" from one session kills another's
#                              vllm server
#
# A slot is one {worktree, chip set, port block, tmpdir} lease per session,
# recorded under $NPU_SLOT_STATE_DIR and reclaimable after a TTL. Chips and
# ports are allocated under a lock, so two sessions cannot accidentally pick the
# same card or the same port.
#
# Usage
# -----
#   eval "$(npu-slot.sh acquire --session alice --chips 2 --ref origin/rl-e2e)"
#   npu-slot.sh run -- pytest -sv tests/e2e/pull_request/one_card/rlhf/
#   npu-slot.sh status
#   npu-slot.sh release --session alice --rm-worktree
#
# Commands: acquire | env | status | renew | run | kill | release | gc
# Env knobs: NPU_SLOT_STATE_DIR NPU_SLOT_WT_ROOT NPU_SLOT_REPO NPU_SLOT_TOTAL_CHIPS
#            NPU_SLOT_CHIP_BUSY_MB NPU_SLOT_PORT_START NPU_SLOT_PORT_BLOCK
#            NPU_SLOT_TTL_HOURS
#
# See SKILL.md ("Session isolation on a shared container") for the full workflow.

set -euo pipefail

STATE_DIR=${NPU_SLOT_STATE_DIR:-/home/liurong/.npu-slots}
LEASE_DIR=$STATE_DIR/leases
WT_ROOT=${NPU_SLOT_WT_ROOT:-/home/liurong/ws}
PRIMARY_REPO=${NPU_SLOT_REPO:-/home/liurong/vllm-ascend}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

TOTAL_CHIPS=${NPU_SLOT_TOTAL_CHIPS:-16}
CHIP_BUSY_MB=${NPU_SLOT_CHIP_BUSY_MB:-8192}   # idle baseline is ~3.1 GB per die
PORT_START=${NPU_SLOT_PORT_START:-8800}
PORT_BLOCK=${NPU_SLOT_PORT_BLOCK:-64}        # stride between per-session port blocks
LEASE_TTL_HOURS=${NPU_SLOT_TTL_HOURS:-8}

log()  { printf '[npu-slot] %s\n' "$*" >&2; }
warn() { printf '[npu-slot] WARNING: %s\n' "$*" >&2; }
die()  { printf '[npu-slot] ERROR: %s\n' "$*" >&2; exit 1; }

now() { date +%s; }

init_state() {
    mkdir -p "$LEASE_DIR" "$WT_ROOT"
    touch "$STATE_DIR/.lock"
}

with_lock() {  # run a command with the global allocator lock held (fd 9)
    local timeout=${NPU_SLOT_LOCK_TIMEOUT:-30}
    mkdir -p "$STATE_DIR"   # the lock file lives here, so create it before opening
    exec 9>"$STATE_DIR/.lock"
    flock -w "$timeout" 9 || die "another npu-slot.sh holds the lock (waited ${timeout}s)"
    "$@"
}

lease_file() { printf '%s/%s.lease' "$LEASE_DIR" "$1"; }

lease_get() {  # lease_get <session> <KEY>
    local f; f=$(lease_file "$1")
    [ -f "$f" ] || return 1
    sed -n "s/^$2=//p" "$f" | tail -1
}

lease_set() {  # lease_set <session> <KEY> <VALUE>
    local f tmp; f=$(lease_file "$1"); tmp=$f.tmp
    if grep -q "^$2=" "$f" 2>/dev/null; then
        sed "s|^$2=.*|$2=$3|" "$f" >"$tmp"
    else
        cp "$f" "$tmp"; printf '%s=%s\n' "$2" "$3" >>"$tmp"
    fi
    mv "$tmp" "$f"
}

lease_valid() {  # 0 = exists and not expired
    local sid=$1 exp
    [ -f "$(lease_file "$sid")" ] || return 1
    exp=$(lease_get "$sid" EXPIRES_EPOCH) || return 1
    [ "$(now)" -lt "$exp" ]
}

# ---------------------------------------------------------------- chip inventory
# Prints "<chip> <used_mb> <total_mb>" for every die reported by npu-smi.
# $4 of a chip row looks like " 0           0    / 0          3114 / 65536   "
# (AICore / mem-used / mem-total, then HBM-used / HBM-total): the HBM pair is
# the last two "/" fields, and only the trailing number of each is meaningful.
chip_hbm() {
    command -v npu-smi >/dev/null 2>&1 || return 1
    npu-smi info 2>/dev/null | awk -F'|' '
        /^\|[[:space:]]*[0-9]+[[:space:]]+[0-9]+[[:space:]]*\|/ {
            head=$2; tail=$4
            gsub(/^[ \t]+|[ \t]+$/, "", head)
            split(head, a, /[ \t]+/)
            chip=a[2]
            n=split(tail, b, "/")
            if (n >= 2) {
                u=b[n-1]; t=b[n]
                sub(/[ \t]*$/, "", u)
                match(u, /[0-9]+$/); u=substr(u, RSTART, RLENGTH)
                match(t, /[0-9]+/);  t=substr(t, RSTART, RLENGTH)
                if (chip != "" && u != "" && t != "") print chip, u+0, t+0
            }
        }'
}

chips_in_use_by_leases() {  # comma-joined chips across all live leases
    local f sid chips out=""
    for f in "$LEASE_DIR"/*.lease; do
        [ -e "$f" ] || continue
        sid=$(basename "$f" .lease)
        lease_valid "$sid" || continue
        chips=$(lease_get "$sid" CHIPS) || continue
        out="$out,$chips"
    done
    printf '%s' "${out#,}"
}

snapshot_chip_hbm() { CHIP_SNAPSHOT=$(chip_hbm 2>/dev/null || true); }

chip_is_busy_now() {  # chip_is_busy_now <chip>  (uses $CHIP_SNAPSHOT)
    local chip=$1 used
    used=$(printf '%s\n' "${CHIP_SNAPSHOT:-}" | awk -v c="$chip" '$1==c {print $2; exit}')
    [ -n "$used" ] || return 1
    [ "$used" -gt "$CHIP_BUSY_MB" ]
}

# ---------------------------------------------------------------- allocation
allocate_chips() {  # allocate_chips <count> -> echoes "c1,c2"
    local want=$1 leased c skip free=()
    leased=$(chips_in_use_by_leases)
    snapshot_chip_hbm
    for ((c = 0; c < TOTAL_CHIPS; c++)); do
        skip=0
        case ",$leased," in *",$c,"*) skip=1 ;; esac
        if [ "$skip" -eq 0 ] && chip_is_busy_now "$c"; then
            warn "chip $c has >${CHIP_BUSY_MB}MB HBM in use and is not leased here - skipping"
            skip=1
        fi
        [ "$skip" -eq 0 ] && free+=("$c")
        [ "${#free[@]}" -ge "$want" ] && break
    done
    [ "${#free[@]}" -ge "$want" ] || die "only ${#free[@]} idle chip(s) available, need $want (see: npu-slot.sh status)"
    local out=""
    for ((i = 0; i < want; i++)); do out="$out,${free[$i]}"; done
    printf '%s' "${out#,}"
}

port_free() {  # port_free <port>
    local p=$1
    command -v python3 >/dev/null 2>&1 || return 0   # cannot probe: assume free
    python3 - "$p" >/dev/null 2>&1 <<'PY'
import socket, sys
s = socket.socket()
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try:
    s.bind(("0.0.0.0", int(sys.argv[1])))
except OSError:
    sys.exit(1)
finally:
    s.close()
PY
}

allocate_port_base() {
    local leased="" f sid pb base=$PORT_START tries=0
    for f in "$LEASE_DIR"/*.lease; do
        [ -e "$f" ] || continue
        sid=$(basename "$f" .lease)
        lease_valid "$sid" || continue
        pb=$(lease_get "$sid" PORT_BASE) || continue
        leased="$leased $pb"
    done
    while [ "$tries" -lt 64 ]; do
        if ! printf '%s\n' $leased | grep -qx "$base"; then
            if port_free "$base"; then printf '%s' "$base"; return 0; fi
            warn "port $base is bound by a process outside the leasing scheme - skipping"
        fi
        base=$((base + PORT_BLOCK))
        tries=$((tries + 1))
    done
    die "no free port block found from $PORT_START (step $PORT_BLOCK)"
}

# ---------------------------------------------------------------- worktree
seed_ops_artifacts() {  # make a fresh worktree usable without rebuilding custom ops
    local wt=$1 src=$PRIMARY_REPO/vllm_ascend/_cann_ops_custom dst=$wt/vllm_ascend/_cann_ops_custom
    [ -d "$src" ] || return 0
    [ -e "$dst" ] || return 0
    [ -L "$dst" ] && return 0
    [ "$(find "$dst" -name '*.so' 2>/dev/null | wc -l)" -eq 0 ] || return 0
    # only replace a pristine checkout artefact (tracked .gitkeep and nothing else)
    [ "$(find "$dst" -mindepth 1 -maxdepth 1 ! -name '.gitkeep' | wc -l)" -eq 0 ] || return 0
    rm -rf "$dst"
    ln -s "$src" "$dst"
    log "linked prebuilt custom ops: $dst -> $src"
}

apply_port_patch() {  # apply_port_patch <worktree>
    local wt=$1
    [ -f "$SCRIPT_DIR/patch_rlhf_port.py" ] || { warn "patch_rlhf_port.py not found next to npu-slot.sh"; return 0; }
    command -v python3 >/dev/null 2>&1 || { warn "python3 not found - skipped rlhf port patch"; return 0; }
    # stdout is reserved for the eval-able export snippet, so send everything to stderr
    python3 "$SCRIPT_DIR/patch_rlhf_port.py" "$wt" >&2 || warn "rlhf port patch not applied"
}

# ---------------------------------------------------------------- env snippet
print_env() {  # print_env <session>; stdout must stay eval-able
    local sid=$1 wt chips port tmp
    wt=$(lease_get "$sid" WORKTREE)
    chips=$(lease_get "$sid" CHIPS)
    port=$(lease_get "$sid" PORT_BASE)
    tmp=$(lease_get "$sid" TMPDIR)
    printf "export NPU_SLOT_SESSION=%q\n" "$sid"
    printf "export NPU_SLOT_LEASE=%q\n" "$(lease_file "$sid")"
    printf "export NPU_SLOT_WORKTREE=%q\n" "$wt"
    printf "export NPU_SLOT_CHIPS=%q\n" "$chips"
    printf "export ASCEND_RT_VISIBLE_DEVICES=%q\n" "$chips"
    printf "export VLLM_RL_TEST_PORT=%q\n" "$port"
    printf "export NPU_SLOT_PORT_BASE=%q\n" "$port"
    printf "export TMPDIR=%q\n" "$tmp"
    printf "export TORCHINDUCTOR_CACHE_DIR=%q\n" "$wt/.cache/inductor"
    printf "export VLLM_CACHE_ROOT=%q\n" "$wt/.cache/vllm"
    printf "export PYTHONPATH=%q\n" "$wt${PYTHONPATH:+:$PYTHONPATH}"
}

# ---------------------------------------------------------------- commands
cmd_acquire() {
    local sid="" chips=2 ref="HEAD" repo=$PRIMARY_REPO worktree=1 portpatch=1 force=0
    while [ $# -gt 0 ]; do
        case $1 in
            --session) sid=$2; shift 2 ;;
            --chips) chips=$2; shift 2 ;;
            --ref) ref=$2; shift 2 ;;
            --repo) repo=$2; shift 2 ;;
            --no-worktree) worktree=0; shift ;;
            --no-port-patch) portpatch=0; shift ;;
            --force) force=1; shift ;;
            *) die "acquire: unknown option '$1'" ;;
        esac
    done
    [ -n "$sid" ] || sid=${NPU_SLOT_SESSION:-}
    [ -n "$sid" ] || sid="s$(date +%m%d-%H%M%S)-$$"
    case $sid in *[!A-Za-z0-9._-]*) die "session id '$sid' must match [A-Za-z0-9._-]" ;; esac
    case $chips in ''|*[!0-9]*) die "--chips must be a positive integer" ;; esac
    [ "$chips" -ge 1 ] || die "--chips must be >= 1"

    init_state

    if lease_valid "$sid"; then
        if [ "$force" -eq 1 ]; then
            warn "stealing the live lease of session '$sid' (--force)"
            rm -f "$(lease_file "$sid")"
        else
            warn "session '$sid' already holds a live lease - reusing it"
            print_env "$sid"
            return 0
        fi
    fi
    if [ -f "$(lease_file "$sid")" ]; then
        log "reclaiming expired lease '$sid'"
        rm -f "$(lease_file "$sid")"
    fi

    local mychips myport wt tmpdir
    mychips=$(allocate_chips "$chips")
    myport=$(allocate_port_base)

    if [ "$worktree" -eq 1 ]; then
        wt=$WT_ROOT/$sid
        if [ -e "$wt" ]; then
            warn "worktree path $wt already exists - reusing it as-is"
        else
            { [ -d "$repo/.git" ] || [ -f "$repo/.git" ]; } || die "$repo is not a git checkout"
            log "creating worktree $wt (detached at $ref)"
            git -C "$repo" worktree add --detach "$wt" "$ref" >&2
            seed_ops_artifacts "$wt"
            [ "$portpatch" -eq 1 ] && apply_port_patch "$wt"
        fi
    else
        wt=$repo
    fi
    tmpdir=$wt/.tmp
    mkdir -p "$tmpdir" "$wt/.cache/inductor" "$wt/.cache/vllm"

    {
        printf 'SESSION=%s\n' "$sid"
        printf 'CHIPS=%s\n' "$mychips"
        printf 'PORT_BASE=%s\n' "$myport"
        printf 'WORKTREE=%s\n' "$wt"
        printf 'TMPDIR=%s\n' "$tmpdir"
        printf 'REPO=%s\n' "$repo"
        printf 'HAS_WORKTREE=%s\n' "$worktree"
        printf 'CREATED_EPOCH=%s\n' "$(now)"
        printf 'EXPIRES_EPOCH=%s\n' "$(($(now) + LEASE_TTL_HOURS * 3600))"
    } >"$(lease_file "$sid")"

    log "slot acquired: session=$sid chips=$mychips port=$myport worktree=$wt"
    if [ "$worktree" -eq 0 ]; then
        warn "--no-worktree: you are working in the shared checkout '$repo'."
        warn "the port patch was NOT applied there; expect collisions with other sessions."
    fi
    print_env "$sid"
}

resolve_session() {  # resolve_session <cmd> <sid>
    local cmd=$1 sid=$2
    [ -n "$sid" ] || sid=${NPU_SLOT_SESSION:-}
    [ -n "$sid" ] || die "$cmd: pass --session or export NPU_SLOT_SESSION"
    printf '%s' "$sid"
}

cmd_env() {
    local sid=""
    while [ $# -gt 0 ]; do
        case $1 in --session) sid=$2; shift 2 ;; *) die "env: unknown option '$1'" ;; esac
    done
    sid=$(resolve_session env "$sid")
    lease_valid "$sid" || die "no live lease for session '$sid'"
    print_env "$sid"
}

cmd_status() {
    init_state
    printf '%-24s %-12s %-8s %-30s %s\n' SESSION CHIPS PORT WORKTREE EXPIRES
    local f sid exp
    for f in "$LEASE_DIR"/*.lease; do
        [ -e "$f" ] || continue
        sid=$(basename "$f" .lease)
        if lease_valid "$sid"; then
            exp=$(lease_get "$sid" EXPIRES_EPOCH)
            printf '%-24s %-12s %-8s %-30s %s\n' "$sid" "$(lease_get "$sid" CHIPS)" \
                "$(lease_get "$sid" PORT_BASE)" "$(lease_get "$sid" WORKTREE)" "in $(( (exp - $(now)) / 60 ))m"
        else
            printf '%-24s %-12s %-8s %-30s %s\n' "$sid" "$(lease_get "$sid" CHIPS)" \
                "$(lease_get "$sid" PORT_BASE)" "$(lease_get "$sid" WORKTREE)" "EXPIRED"
        fi
    done
    echo
    if snapshot_chip_hbm && [ -n "${CHIP_SNAPSHOT:-}" ]; then
        echo "NPU HBM in use (MiB); 'busy' above ${CHIP_BUSY_MB}MiB usually means an unleased process:"
        printf '%s\n' "$CHIP_SNAPSHOT" | awk -v lim="$CHIP_BUSY_MB" \
            '{ flag = ($2 > lim) ? "   <-- busy (not leased?)" : ""; printf "  chip %-3s %8s / %-8s%s\n", $1, $2, $3, flag }'
    else
        warn "npu-smi not available here - chip usage not shown"
    fi
}

cmd_release() {
    local sid="" rmwt=0
    while [ $# -gt 0 ]; do
        case $1 in
            --session) sid=$2; shift 2 ;;
            --rm-worktree) rmwt=1; shift ;;
            *) die "release: unknown option '$1'" ;;
        esac
    done
    sid=$(resolve_session release "$sid")
    init_state
    [ -f "$(lease_file "$sid")" ] || { warn "no lease for session '$sid'"; return 0; }

    local pgid; pgid=$(lease_get "$sid" RUN_PGID) || true
    if [ -n "${pgid:-}" ]; then
        kill -TERM "-$pgid" 2>/dev/null || true
        sleep 2
        kill -KILL "-$pgid" 2>/dev/null || true
    fi

    if [ "$rmwt" -eq 1 ]; then
        local wt repo; wt=$(lease_get "$sid" WORKTREE); repo=$(lease_get "$sid" REPO)
        if [ -n "$wt" ] && [ "$wt" != "$repo" ] && [ -d "$wt" ]; then
            log "removing worktree $wt"
            git -C "$repo" worktree remove --force "$wt" >&2 || true
            git -C "$repo" worktree prune >&2 || true
        fi
    fi
    rm -f "$(lease_file "$sid")"
    log "released slot '$sid'"
}

cmd_renew() {
    local sid="" hours=$LEASE_TTL_HOURS
    while [ $# -gt 0 ]; do
        case $1 in --session) sid=$2; shift 2 ;; --hours) hours=$2; shift 2 ;; *) die "renew: unknown option '$1'" ;; esac
    done
    sid=$(resolve_session renew "$sid")
    init_state
    [ -f "$(lease_file "$sid")" ] || die "no lease for session '$sid'"
    lease_set "$sid" EXPIRES_EPOCH "$(($(now) + hours * 3600))"
    log "lease '$sid' renewed for ${hours}h"
}

cmd_run() {
    local sid="" log_file=""
    while [ $# -gt 0 ]; do
        case $1 in --session) sid=$2; shift 2 ;; --log) log_file=$2; shift 2 ;; --) shift; break ;; *) break ;; esac
    done
    [ $# -gt 0 ] || die "run: nothing to run (usage: npu-slot.sh run -- pytest ...)"
    sid=$(resolve_session run "$sid")
    lease_valid "$sid" || die "no live lease for session '$sid'"

    local wt; wt=$(lease_get "$sid" WORKTREE)
    [ -d "$wt" ] || die "worktree '$wt' is gone"
    [ -n "$log_file" ] || log_file="$wt/.npu-slot-$(date +%m%d-%H%M%S)-$$.log"

    eval "$(cmd_env --session "$sid")"
    log "running in $wt (chips=$ASCEND_RT_VISIBLE_DEVICES port=$VLLM_RL_TEST_PORT)"
    log "log -> $log_file"
    # Rebuild the command line without losing argument boundaries. A single
    # argument is passed through verbatim, so shell syntax still works:
    #   npu-slot.sh run -- 'pytest -q ... | tail -20'
    local quoted
    if [ $# -eq 1 ]; then
        quoted=$1
    else
        quoted=$(printf '%q ' "$@")
    fi
    cd "$wt"
    # Own session => own process group, so cleanup can never touch another session.
    setsid bash -lc "$quoted" >"$log_file" 2>&1 &
    local pid=$! pgid rc=0
    pgid=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ') || pgid=""
    [ -n "$pgid" ] && lease_set "$sid" RUN_PGID "$pgid"
    lease_set "$sid" RUN_LOG "$log_file"
    wait "$pid" || rc=$?
    lease_set "$sid" RUN_PGID ""
    log "command finished with exit code $rc"
    tail -n 5 "$log_file" >&2 || true
    return "$rc"
}

cmd_kill() {
    local sid=""
    while [ $# -gt 0 ]; do
        case $1 in --session) sid=$2; shift 2 ;; *) die "kill: unknown option '$1'" ;; esac
    done
    sid=$(resolve_session kill "$sid")
    local pgid; pgid=$(lease_get "$sid" RUN_PGID) || true
    [ -n "${pgid:-}" ] || { warn "session '$sid' has no recorded process group"; return 0; }
    log "killing process group $pgid (session '$sid')"
    kill -TERM "-$pgid" 2>/dev/null || true
    sleep 2
    kill -KILL "-$pgid" 2>/dev/null || true
    lease_set "$sid" RUN_PGID ""
}

cmd_gc() {
    init_state
    local f sid
    for f in "$LEASE_DIR"/*.lease; do
        [ -e "$f" ] || continue
        sid=$(basename "$f" .lease)
        if ! lease_valid "$sid"; then
            log "dropping expired lease '$sid'"
            rm -f "$f"
        fi
    done
    if [ -d "$PRIMARY_REPO/.git" ]; then git -C "$PRIMARY_REPO" worktree prune >&2 || true; fi
}

usage() {
    cat >&2 <<'EOF'
npu-slot.sh - isolated session slots for the shared liurong_vllm container

  acquire [--session ID] [--chips N] [--ref REF] [--repo PATH]
          [--no-worktree] [--no-port-patch] [--force]
                          lease chips + port, create a git worktree, print env
                          (use as: eval "$(npu-slot.sh acquire --session ID)")
                          --force steals a live lease from a crashed session
                          with the same id; expired leases are reclaimed anyway
  env     [--session ID]  re-print the export snippet for an existing slot
  status                  list leases + current NPU HBM usage per die
  renew   [--session ID] [--hours H]
                          extend the lease TTL (default 8h)
  run     [--session ID] [--log FILE] -- CMD...
                          run CMD inside the slot worktree/env, in its own
                          process group, with output logged to FILE
  kill    [--session ID]  kill only this session's recorded process group
  release [--session ID] [--rm-worktree]
                          drop the lease (and optionally the worktree)
  gc                      drop expired leases, prune git worktree metadata
EOF
}

main() {
    local cmd=${1:-}
    [ $# -gt 0 ] && shift
    case $cmd in
        acquire) cmd_acquire "$@" ;;
        env)     cmd_env "$@" ;;
        status)  cmd_status "$@" ;;
        renew)   cmd_renew "$@" ;;
        run)     cmd_run "$@" ;;
        kill)    cmd_kill "$@" ;;
        release) cmd_release "$@" ;;
        gc)      cmd_gc "$@" ;;
        ""|-h|--help|help) usage ;;
        *) die "unknown command '$cmd' (try: npu-slot.sh --help)" ;;
    esac
}

# Mutating commands take the allocator lock; read-only ones do not.
case ${1:-} in
    acquire|release|renew|gc) with_lock main "$@" ;;
    *) main "$@" ;;
esac
