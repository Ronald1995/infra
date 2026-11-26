#!/usr/bin/env bash
# Install the third-party dependencies declared in dependencies/manifest.yaml.
#
# This script owns no dependency knowledge: URLs, commits, tags, build steps and verify
# commands all come from the manifest through scripts/dependency_plan.py.  The script's
# job is to ask for a plan, show it, authorize it, execute it step by step with a log, and
# then prove the result with the doctor.
#
#   bash scripts/install_deps.sh --profile cpu
#   bash scripts/install_deps.sh --profile communication --dry-run
#   bash scripts/install_deps.sh --profile all --offline \
#       --source-cache /data/ep-sources --wheelhouse /data/ep-wheels
#   bash scripts/install_deps.sh --check-only --profile all
#
# Profiles: cpu | npu | communication | profiling | multinode | all
#
# Safety: drivers, firmware, CANN and device nodes are never installed here; the plan
# lists them with the exact manual step and the doctor keeps them blocked.  Nothing in
# this script deletes an existing checkout or build artifact.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
PROFILE="all"
DRY_RUN=0
CHECK_ONLY=0
OFFLINE=0
ALLOW_NETWORK=0
JSON=0
SOURCE_CACHE="${EP_SOURCE_CACHE:-$HOME/.cache/ep-benchmark/deps}"
WHEELHOUSE="${EP_WHEELHOUSE:-}"
ENVIRONMENTS=()
LOG_DIR="${EP_INSTALL_LOG_DIR:-$ROOT/.install-logs}"

usage() {
  sed -n '2,22p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  cat <<'EOF'

Options:
  --profile NAME        cpu | npu | communication | profiling | multinode | all
  --dry-run             print the plan and the commands; change nothing
  --check-only          verify what is installed; install nothing
  --offline             never fetch: require the pinned checkouts in --source-cache
  --source-cache DIR    where pinned checkouts live (default $HOME/.cache/ep-benchmark/deps)
  --wheelhouse DIR      install python wheels only from this directory
  --allow-network       authorize fetching pinned sources from the network
  --environment NAME    restrict the plan to one dependency environment (repeatable)
  --json                print the machine-readable plan
  -h, --help            this text
EOF
}

die() { printf 'BLOCKED: %s\n' "$*" >&2; exit 1; }
note() { printf '%s\n' "$*"; }

while [ "$#" -gt 0 ]; do
  case "$1" in
    --profile) [ "$#" -ge 2 ] || die "--profile needs a value"; PROFILE="$2"; shift 2 ;;
    --profile=*) PROFILE="${1#*=}"; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --check-only) CHECK_ONLY=1; shift ;;
    --offline) OFFLINE=1; shift ;;
    --source-cache) [ "$#" -ge 2 ] || die "--source-cache needs a directory"
                    SOURCE_CACHE="$2"; shift 2 ;;
    --source-cache=*) SOURCE_CACHE="${1#*=}"; shift ;;
    --wheelhouse) [ "$#" -ge 2 ] || die "--wheelhouse needs a directory"
                  WHEELHOUSE="$2"; shift 2 ;;
    --wheelhouse=*) WHEELHOUSE="${1#*=}"; shift ;;
    --allow-network) ALLOW_NETWORK=1; shift ;;
    --environment) [ "$#" -ge 2 ] || die "--environment needs a name"
                   ENVIRONMENTS+=("$2"); shift 2 ;;
    --environment=*) ENVIRONMENTS+=("${1#*=}"); shift ;;
    --json) JSON=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option '$1' (see --help)" ;;
  esac
done

case "$PROFILE" in
  cpu|npu|communication|profiling|multinode|all) ;;
  *) die "unknown profile '$PROFILE' (cpu|npu|communication|profiling|multinode|all)" ;;
esac

if [ -n "$WHEELHOUSE" ] && [ ! -d "$WHEELHOUSE" ]; then
  die "--wheelhouse '$WHEELHOUSE' is not a directory"
fi
if [ "$OFFLINE" = 1 ] && [ "$ALLOW_NETWORK" = 1 ]; then
  die "--offline and --allow-network contradict each other"
fi

cd "$ROOT" || die "cannot enter $ROOT"
mkdir -p "$LOG_DIR" || die "cannot create $LOG_DIR"
PLAN_JSON="$LOG_DIR/plan-$PROFILE.json"
STEPS_SH="$LOG_DIR/plan-$PROFILE.sh"
STEPS_LOG="$LOG_DIR/plan-$PROFILE.steps.log"

export EP_SOURCE_CACHE="$SOURCE_CACHE"
export EP_ENV_ROOT="${EP_ENV_ROOT:-$ROOT/.envs}"
export EP_BASE_PYTHON="${EP_BASE_PYTHON:-$PYTHON}"
mkdir -p "$EP_ENV_ROOT" 2>/dev/null || true

plan_args=(--profile "$PROFILE" --source-cache "$SOURCE_CACHE")
[ "$OFFLINE" = 1 ] && plan_args+=(--offline)
[ "$CHECK_ONLY" = 1 ] && plan_args+=(--check-only)
[ -n "$WHEELHOUSE" ] && plan_args+=(--wheelhouse "$WHEELHOUSE")
for environment in "${ENVIRONMENTS[@]:-}"; do
  [ -n "$environment" ] && plan_args+=(--environment "$environment")
done

note "install_deps: profile=$PROFILE offline=$OFFLINE check_only=$CHECK_ONLY"
note "install_deps: source cache $SOURCE_CACHE"
[ -n "$WHEELHOUSE" ] && note "install_deps: wheelhouse $WHEELHOUSE"

"$PYTHON" scripts/dependency_plan.py "${plan_args[@]}" --out "$PLAN_JSON" >/dev/null 2>"$LOG_DIR/plan.err" \
  || { cat "$LOG_DIR/plan.err" >&2; die "the dependency plan could not be built"; }

if [ "$JSON" = 1 ]; then
  "$PYTHON" scripts/dependency_plan.py "${plan_args[@]}" --json --no-fail
  exit 0
fi

"$PYTHON" scripts/dependency_plan.py "${plan_args[@]}" --no-fail

needs_fetch() {
  "$PYTHON" - "$PLAN_JSON" <<'PY'
import json, sys
plan = json.load(open(sys.argv[1], encoding="utf-8"))
print(sum(1 for step in plan["steps"] if step["kind"] == "fetch"))
PY
}

fetch_count="$(needs_fetch 2>/dev/null || echo 0)"

if [ "$DRY_RUN" = 1 ]; then
  note ""
  note "--- commands that would run ---"
  "$PYTHON" scripts/dependency_plan.py "${plan_args[@]}" --script --no-fail | sed 's/^/  /'
  note ""
  if [ "$fetch_count" != "0" ] && [ "$OFFLINE" != 1 ] && [ "$ALLOW_NETWORK" != 1 ]; then
    note "NOTE: $fetch_count fetch step(s) need --allow-network (or use --offline with a source cache)."
  fi
  # Exit non-zero only when the plan has an actionable blocker (an unpinned installable
  # package, a missing wheelhouse); host-provided blockers are expected on this machine.
  "$PYTHON" scripts/dependency_plan.py "${plan_args[@]}" >/dev/null
  exit $?
fi

if [ "$fetch_count" != "0" ] && [ "$OFFLINE" != 1 ] && [ "$ALLOW_NETWORK" != 1 ]; then
  die "$fetch_count step(s) would download pinned sources. Pass --allow-network, or use --offline with a prepared --source-cache."
fi
if [ "$fetch_count" != "0" ] && [ "$OFFLINE" != 1 ] && ! command -v git >/dev/null 2>&1; then
  die "fetching pinned sources needs git, which is not on PATH"
fi

"$PYTHON" scripts/dependency_plan.py "${plan_args[@]}" --script --no-fail >"$STEPS_SH" \
  || die "the install script could not be generated"
note "install_deps: plan $PLAN_JSON"
note "install_deps: steps $STEPS_SH"

EP_STEP_LOG="$STEPS_LOG" bash "$STEPS_SH"
steps_status=$?
if [ "$steps_status" -ne 0 ]; then
  note ""
  note "install_deps: some steps failed (exit $steps_status); the exact failure is above."
  note "install_deps: repair or authorize it, then re-run this command; it is idempotent."
fi

note ""
note "install_deps: verifying with the doctor"
doctor_args=(--profile "$PROFILE")
[ -n "$WHEELHOUSE" ] && doctor_args+=(--offline-wheelhouse "$WHEELHOUSE")
"$PYTHON" scripts/doctor.py "${doctor_args[@]}"
doctor_status=$?

if [ "$doctor_status" -ne 0 ]; then
  note ""
  note "install_deps: the profile is not satisfied; the doctor above names each missing"
  note "item. Host-provided items (driver, firmware, CANN, device nodes) must be installed"
  note "outside this container; their manual step is in the doctor output."
fi
exit "$doctor_status"
