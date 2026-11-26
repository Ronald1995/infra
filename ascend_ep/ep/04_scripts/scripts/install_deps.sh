#!/usr/bin/env bash
# Third-party dependency installer for the Ascend EP benchmark.
#
# Why this script exists: installing these dependencies by hand produces a
# container that looks ready and is not -- a pip torch that disagrees with
# torch_npu, a DeepEP built against another CANN release, an "installed" driver
# that is only a library without devices.  The installer therefore does the small
# number of things a container is actually allowed to do (wheels, native builds
# against the installed toolkit), prints the exact manual step for everything that
# belongs to the host, and ends by running scripts/doctor.py so "done" is a
# verdict rather than an assumption.
#
# What this script will NOT do:
#   * install, upgrade or reload the NPU driver, the firmware or /dev/davinci*
#     (host state; the exact manual step is printed instead);
#   * run sudo without --yes, and never with an interactive password prompt;
#   * delete anything it did not create in this run;
#   * report success when a native build could not run -- a missing toolchain is a
#     BLOCKED line and a non-zero exit, never a warning.
#
# Usage:
#   bash scripts/install_deps.sh --profile cpu
#   bash scripts/install_deps.sh --profile npu --wheelhouse /data/wheels
#   bash scripts/install_deps.sh --profile communication --dry-run
#   bash scripts/install_deps.sh --check-only --profile all
#
# Keep the profile -> category table below in sync with PROFILES in
# scripts/doctor.py; the manifest declares the same table, and a mismatch would
# make the installer verify a different set than it installed.
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
SCRIPTS_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
PYTHON=${PYTHON:-python3}
DEPS_ROOT=${DEPS_ROOT:-"$HOME/.cache/ep-benchmark/deps"}
DOCTOR="$SCRIPTS_ROOT/scripts/doctor.py"

PROFILE=cpu
WHEELHOUSE=
CHECK_ONLY=0
DRY_RUN=0
ASSUME_YES=0
BLOCKERS=0
TMP_STATE=
TMP_LOG=

# ---------------------------------------------------------------------------
# Output and bookkeeping
# ---------------------------------------------------------------------------
say() { printf '%s\n' "$*"; }
step() { printf '\n== %s\n' "$*"; }
warn() { printf 'WARNING: %s\n' "$*" >&2; }
blocker() { BLOCKERS=$((BLOCKERS + 1)); printf 'BLOCKED: %s\n' "$*" >&2; }
die() { printf 'error: %s\n' "$*" >&2; exit 2; }

cleanup() {
    [ -z "$TMP_STATE" ] || rm -f "$TMP_STATE"
    [ -z "$TMP_LOG" ] || rm -f "$TMP_LOG"
}
trap cleanup EXIT

usage() {
    cat <<'USAGE'
Install the third-party dependencies this benchmark needs, then verify them.

Options:
  --profile cpu|npu|communication|profiling|multinode|all
                                   dependency categories to install (default: cpu)
  --wheelhouse DIR                 offline mode: install only from wheels in DIR
  --check-only                     install nothing; verify and print what is missing
  --dry-run                        print every command instead of running it
  --yes, -y                        allow sudo for package-manager installs
  --help, -h                       this text

Profiles (dependency categories from dependencies/manifest.yaml):
  cpu            A  python base stack; must work on an offline CPU-only machine
  npu            A+B  plus torch_npu (wheels only; CANN and the driver stay on the host)
  communication  A+B+C  plus SHMEM / DeepEP / npu_patch / MoonEP / ASC-COMM kernels
  profiling      A+B+E  plus msprof / hccl_test (host components; never installed here)
  multinode      A+B+D  plus MPI; the deployment variables are printed, not set
  all            A+B+C+D+E

Exit codes: 0 ready, 1 not ready (blockers or doctor failures), 2 usage/pre-flight.
USAGE
}

# ---------------------------------------------------------------------------
# Profile -> categories (mirrors PROFILES in scripts/doctor.py)
# ---------------------------------------------------------------------------
profile_has() {
    case $PROFILE in
        all) return 0 ;;
        cpu) [ "$1" = A ] ;;
        npu) [ "$1" = A ] || [ "$1" = B ] ;;
        communication) [ "$1" = A ] || [ "$1" = B ] || [ "$1" = C ] ;;
        profiling) [ "$1" = A ] || [ "$1" = B ] || [ "$1" = E ] ;;
        multinode) [ "$1" = A ] || [ "$1" = B ] || [ "$1" = D ] ;;
        *) return 1 ;;
    esac
}

# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------
parse_args() {
    while [ "$#" -gt 0 ]; do
        case $1 in
            --profile)
                [ "$#" -ge 2 ] || die "--profile needs a value"
                PROFILE=$2
                shift 2
                ;;
            --profile=*)
                PROFILE=${1#*=}
                shift
                ;;
            --wheelhouse)
                [ "$#" -ge 2 ] || die "--wheelhouse needs a directory"
                WHEELHOUSE=$2
                shift 2
                ;;
            --wheelhouse=*)
                WHEELHOUSE=${1#*=}
                shift
                ;;
            --check-only)
                CHECK_ONLY=1
                shift
                ;;
            --dry-run)
                DRY_RUN=1
                shift
                ;;
            --yes|-y)
                ASSUME_YES=1
                shift
                ;;
            --help|-h)
                usage
                exit 0
                ;;
            *)
                usage >&2
                die "unknown argument: $1"
                ;;
        esac
    done
    case $PROFILE in
        cpu|npu|communication|profiling|multinode|all) : ;;
        *) die "unknown profile '$PROFILE' (cpu|npu|communication|profiling|multinode|all)" ;;
    esac
    if [ -n "$WHEELHOUSE" ] && [ "$DRY_RUN" -eq 0 ]; then
        [ -d "$WHEELHOUSE" ] || die "--wheelhouse '$WHEELHOUSE' is not a directory"
    fi
}

# ---------------------------------------------------------------------------
# Pre-flight
# ---------------------------------------------------------------------------
preflight() {
    step "pre-flight (profile=$PROFILE)"
    [ -f "$SCRIPTS_ROOT/run_benchmark.py" ] || die "run from the SCRIPTS root (run_benchmark.py not found in $SCRIPTS_ROOT)"
    [ -f "$SCRIPTS_ROOT/dependencies/manifest.yaml" ] || die "dependencies/manifest.yaml is missing"
    [ -f "$DOCTOR" ] || die "scripts/doctor.py is missing"
    command -v "$PYTHON" >/dev/null 2>&1 || die "python '$PYTHON' not found; set PYTHON=/path/to/python"
    if ! "$PYTHON" -c 'import sys
raise SystemExit(0 if (3, 10) <= sys.version_info[:2] < (3, 13) else 1)'; then
        die "'$PYTHON' is not CPython 3.10-3.12; install a supported interpreter first"
    fi
    if [ "$CHECK_ONLY" -eq 0 ] && [ "$DRY_RUN" -eq 0 ]; then
        if ! "$PYTHON" -m pip --version >/dev/null 2>&1; then
            die "'$PYTHON -m pip' is unavailable; run '$PYTHON -m ensurepip' or set PYTHON to an interpreter that has pip"
        fi
    elif ! "$PYTHON" -m pip --version >/dev/null 2>&1; then
        warn "'$PYTHON -m pip' is unavailable; an install run would stop here ('$PYTHON -m ensurepip')"
    fi
    say "python       $("$PYTHON" -c 'import sys; print(sys.version.split()[0], sys.executable)')"
    say "scripts root $SCRIPTS_ROOT"
    say "wheelhouse   ${WHEELHOUSE:-<none: pip may use the network>}"
    case "$(uname -s 2>/dev/null || echo unknown)" in
        Linux) : ;;
        *)
            if profile_has B; then
                warn "this host is not Linux; the NPU categories cannot be installed or used here."
                warn "use --profile cpu for a CPU-only check, or run this inside the Ascend container."
            fi
            ;;
    esac
    if [ "$DRY_RUN" -eq 1 ]; then
        say "mode         dry-run (nothing is installed or executed)"
    elif [ "$CHECK_ONLY" -eq 1 ]; then
        say "mode         check-only (nothing is installed)"
    else
        say "mode         install"
    fi
    TMP_STATE=$(mktemp "${TMPDIR:-/tmp}/ep-deps-state.XXXXXX")
    TMP_LOG=$(mktemp "${TMPDIR:-/tmp}/ep-deps-log.XXXXXX")
}

# ---------------------------------------------------------------------------
# pip
# ---------------------------------------------------------------------------
pip_install() {
    if [ "$CHECK_ONLY" -eq 1 ]; then
        printf '  [check-only] %s -m pip install %s\n' "$PYTHON" "$*"
        return 0
    fi
    if [ "$DRY_RUN" -eq 1 ]; then
        printf '  [dry-run] %s -m pip install %s\n' "$PYTHON" "$*"
        return 0
    fi
    if [ -n "$WHEELHOUSE" ]; then
        "$PYTHON" -m pip install --disable-pip-version-check --no-index \
            --find-links "$WHEELHOUSE" "$@"
    else
        "$PYTHON" -m pip install --disable-pip-version-check "$@"
    fi
}

install_python_base() {
    requirement="$SCRIPTS_ROOT/dependencies/python/base.txt"
    step "category A: python base stack"
    [ -f "$requirement" ] || die "missing $requirement"
    if [ -n "$WHEELHOUSE" ]; then
        say "offline: every wheel and sdist must already be in $WHEELHOUSE"
    fi
    # The pins live in the requirement file.  This script never passes --upgrade,
    # so an already-satisfied pin is a no-op and is never quietly replaced.
    pip_install -r "$requirement"
}

install_torch_npu() {
    requirement="$SCRIPTS_ROOT/dependencies/python/npu.txt"
    step "category B: torch_npu wheel"
    [ -f "$requirement" ] || die "missing $requirement"
    say "the wheel must match the installed torch and the CANN row recorded in"
    say "dependencies/manifest.yaml compatibility.rows (never 'latest')"
    pip_install -r "$requirement"
}

# ---------------------------------------------------------------------------
# Root commands (never interactive)
# ---------------------------------------------------------------------------
run_root() {
    if [ "$DRY_RUN" -eq 1 ]; then
        printf '  [dry-run] (root) %s\n' "$*"
        return 0
    fi
    if [ "$(id -u)" -eq 0 ]; then
        "$@"
        return $?
    fi
    if [ "$ASSUME_YES" -eq 1 ]; then
        # -n so a host without passwordless sudo fails loudly instead of hanging.
        if ! sudo -n "$@"; then
            blocker "root command failed (no passwordless sudo): sudo $*"
            return 1
        fi
        return 0
    fi
    blocker "needs root: sudo $*   (rerun with --yes to allow sudo, or run it by hand)"
    return 1
}

package_manager() {
    for candidate in apt-get dnf yum apk zypper; do
        if command -v "$candidate" >/dev/null 2>&1; then
            printf '%s' "$candidate"
            return 0
        fi
    done
    return 1
}

install_system_packages() {
    # install_system_packages <why> <proving binary> <package list>
    local why=$1
    local probe=$2
    local packages=$3
    local pm
    if command -v "$probe" >/dev/null 2>&1; then
        say "$why already present: $(command -v "$probe")"
        return 0
    fi
    if [ "$CHECK_ONLY" -eq 1 ]; then
        say "$why is missing (check-only: not installing)"
        return 0
    fi
    if ! pm=$(package_manager); then
        blocker "$why is missing and no supported package manager was found; install ${packages} by hand"
        return 0
    fi
    say "$why is missing; installing with $pm: $packages"
    case $pm in
        apt-get) run_root apt-get install -y --no-install-recommends $packages || return 0 ;;
        dnf|yum) run_root "$pm" install -y $packages || return 0 ;;
        apk) run_root apk add --no-cache $packages || return 0 ;;
        zypper) run_root zypper --non-interactive install $packages || return 0 ;;
        *) blocker "$why is missing and '$pm' is not supported; install ${packages} by hand"; return 0 ;;
    esac
    if command -v "$probe" >/dev/null 2>&1; then
        say "$why installed: $(command -v "$probe")"
    else
        blocker "$why is still missing after '${pm}' installed ${packages}"
    fi
}

# ---------------------------------------------------------------------------
# Native builds
# ---------------------------------------------------------------------------
require_toolchain() {
    # require_toolchain <dependency> <binary>...
    local dependency=$1
    shift
    local missing=
    local binary
    for binary in "$@"; do
        if ! command -v "$binary" >/dev/null 2>&1; then
            missing="$missing $binary"
        fi
    done
    if [ -n "$missing" ]; then
        blocker "$dependency cannot be built: missing$missing (CANN toolkit / compiler; see dependencies/system/cann.yaml)"
        return 1
    fi
    return 0
}

doctor_status() {
    # doctor_status <check-id>: pass/warn/fail/skip from the pre-install snapshot.
    [ -s "$TMP_STATE" ] || return 1
    "$PYTHON" -c '
import json, sys
data = json.load(open(sys.argv[1], encoding="utf-8"))
print({check["id"]: check["status"] for check in data["checks"]}.get(sys.argv[2], "missing"))
' "$TMP_STATE" "$1" 2>/dev/null || return 1
}

already_present() {
    local status
    status=$(doctor_status "$1") || return 1
    [ "$status" = pass ] || [ "$status" = warn ]
}

delegate_build() {
    # delegate_build <id> <source dir> <build script|-> <source url> <pin> <pip mode>
    local dependency=$1 directory=$2 script=$3 url=$4 pin=$5 mode=$6
    if [ "$DRY_RUN" -eq 1 ]; then
        printf '  [dry-run] (cd %s && bash %s)  # %s @ %s\n' "$directory" "$script" "$url" "$pin"
        return 0
    fi
    if [ ! -d "$directory" ]; then
        blocker "$dependency: no checkout at $directory. Pinned source: $url @ $pin"
        return 0
    fi
    if [ "$script" != "-" ]; then
        if [ ! -f "$directory/$script" ]; then
            blocker "$dependency: $directory/$script is absent; this script does not guess the build entry point"
            return 0
        fi
        say "building $dependency from $directory ($script)"
        if ! ( cd "$directory" && bash "$script" ); then
            blocker "$dependency: $script failed; nothing was installed"
            return 0
        fi
    fi
    case $mode in
        none) : ;;
        wheel)
            if ls "$directory"/build/*.whl >/dev/null 2>&1; then
                pip_install --no-build-isolation "$directory"/build/*.whl
            else
                blocker "$dependency: the build produced no wheel under $directory/build"
            fi
            ;;
        editable) pip_install --no-build-isolation -e "$directory" ;;
        sub:*) pip_install --no-build-isolation "$directory/${mode#sub:}" ;;
        *) pip_install --no-build-isolation "$directory" ;;
    esac
}

install_native_communication() {
    step "category C: communication runtimes (source builds)"
    say "pinned sources (single source of truth: dependencies/manifest.yaml):"
    say "  shmem              https://gitcode.com/cann/shmem            release tag matching the CANN row (commit: null)"
    say "  deep-ep            https://github.com/lueelu/DeepEP          83252a54d2363734a706a7dd8368bebc3094af5a"
    say "  deep-ep-v2         https://gitcode.com/lishaoxun/DeepEP_9283 69090a120cfee06be431a79873692677bc07b042"
    say "  npu-patch/moonep   https://gitcode.com/Ascend/MoonEP         b4127baa1bec0bcccd49b48c2dc2a3be70da9a0f"
    say "  l0 bridges         baselines/l0/{transport,memory,shmem}/native/  (this repository)"
    if [ "$CHECK_ONLY" -eq 1 ]; then
        say "check-only: native builds were not attempted (scripts/doctor.py reports what is missing)"
        return 0
    fi
    if already_present shmem; then
        say "shmem is already importable; skipping its build"
    elif require_toolchain shmem ccec; then
        delegate_build shmem "$DEPS_ROOT/shmem" build.sh \
            https://gitcode.com/cann/shmem "release tag matching the CANN row" wheel
    fi
    if already_present deep-ep || already_present deep-ep-v2; then
        say "deep_ep is already importable; skipping the DeepEP build (one fork per environment)"
    elif require_toolchain deep-ep ccec; then
        delegate_build deep-ep "$DEPS_ROOT/DeepEP" build.sh \
            https://github.com/lueelu/DeepEP 83252a54d2363734a706a7dd8368bebc3094af5a pip
    fi
    if already_present npu-patch; then
        say "npu_patch is already importable; skipping"
    elif require_toolchain npu-patch ccec; then
        delegate_build npu-patch "$DEPS_ROOT/MoonEP" - \
            https://gitcode.com/Ascend/MoonEP b4127baa1bec0bcccd49b48c2dc2a3be70da9a0f sub:npu_patch
    fi
    if already_present moonep-runtime; then
        say "moonep is already importable; skipping"
    elif require_toolchain moonep-runtime ccec; then
        delegate_build moonep-runtime "$DEPS_ROOT/MoonEP" - \
            https://gitcode.com/Ascend/MoonEP b4127baa1bec0bcccd49b48c2dc2a3be70da9a0f editable
    fi
    build_l0_bridges
}

build_l0_bridges() {
    step "category C: L0 native bridges (this repository)"
    local base="$SCRIPTS_ROOT/baselines/l0"
    if already_present l0-native-bridges; then
        say "L0 bridges are already built; keeping the existing artifacts"
        return 0
    fi
    if ! require_toolchain l0-native-bridges cmake; then
        return 0
    fi
    local entry component rest script artifact directory
    for entry in "transport:native/build.sh:libl0_transport_bridge.so" \
                 "memory:native/build_mem.sh:libl0_mem.so" \
                 "shmem:native/build_udma.sh:libep_l0_udma.so"; do
        component=${entry%%:*}
        rest=${entry#*:}
        script=${rest%%:*}
        artifact=${rest#*:}
        directory="$base/$component/native"
        if [ ! -f "$directory/$script" ]; then
            blocker "l0-native-bridges: $directory/$script is absent"
            continue
        fi
        if [ "$DRY_RUN" -eq 1 ]; then
            printf '  [dry-run] (cd %s && bash %s)\n' "$directory" "$script"
            continue
        fi
        if ( cd "$directory" && bash "$script" ); then
            say "$component bridge built"
        else
            blocker "l0-native-bridges: the $component build failed (L0 cases needing $artifact stay dependency_missing)"
        fi
    done
}

install_asc_comm() {
    step "category C/E: ASC-COMM kernels -- BLOCKED"
    say "status: BLOCKED (dependencies/system/asc_comm.yaml records the exact blockers)"
    local source_dir="$SCRIPTS_ROOT/kernels/asc_comm"
    local build_script="$source_dir/build.sh"
    if [ "$CHECK_ONLY" -eq 1 ]; then
        say "check-only: not attempting the ASC-COMM build"
        return 0
    fi
    if [ ! -d "$source_dir" ]; then
        blocker "kernels/asc_comm/ is absent; there is nothing to build"
        return 0
    fi
    if [ ! -f "$build_script" ]; then
        blocker "$build_script is absent; this script does not guess the ASC-COMM build entry point"
        return 0
    fi
    if ! command -v cmake >/dev/null 2>&1 || ! command -v g++ >/dev/null 2>&1; then
        blocker "the ASC-COMM build needs cmake and g++ (install them with the image's package manager)"
        return 0
    fi
    if [ -z "${ASCEND_HOME_PATH:-}" ] || [ ! -d "${ASCEND_HOME_PATH}" ]; then
        blocker "ASCEND_HOME_PATH is unset or invalid; the ASC-COMM kernels need the CANN toolkit headers (host)"
        return 0
    fi
    if [ ! -f "$source_dir/aiv/write_nbi/kernel.cpp" ] || [ ! -f "$source_dir/ccu/dsl/kernel.cpp" ]; then
        blocker "the ASC-COMM device sources are not written yet (aiv/write_nbi/kernel.cpp, ccu/dsl/kernel.cpp)"
        return 0
    fi
    if [ "$DRY_RUN" -eq 1 ]; then
        printf '  [dry-run] (cd %s && bash build.sh)\n' "$source_dir"
        return 0
    fi
    if ( cd "$source_dir" && bash build.sh ); then
        say "ASC-COMM kernels built"
    else
        blocker "the ASC-COMM build failed; see dependencies/system/asc_comm.yaml for the blocker list"
    fi
}

# ---------------------------------------------------------------------------
# Multinode
# ---------------------------------------------------------------------------
install_mpi() {
    step "category D: MPI launcher"
    install_system_packages "mpirun" mpirun "openmpi-bin libopenmpi-dev"
    if command -v mpirun >/dev/null 2>&1; then
        say "mpirun implementation: $(mpirun --version 2>&1 | sed -n '1,2p' | tr '\n' ' ')"
    fi
    say "the deployment variables (MASTER_ADDR, MASTER_PORT, ASCEND_RT_VISIBLE_DEVICES,"
    say "HCCL_IF_BASE_PORT) and the HCCL_* configuration are site state; see"
    say "dependencies/system/mpi.yaml.  This script prints them, it does not set them."
}

# ---------------------------------------------------------------------------
# Host boundary: printed, never executed
# ---------------------------------------------------------------------------
host_boundary_notice() {
    step "host components: printed, never installed by this script"
    cat <<'NOTICE'
These are host state.  A container cannot install, upgrade or reload them, and a
copy inside the container shadows the host libraries.  Run them on the host as an
administrator:

  1) NPU driver + firmware  -- this script refuses to touch them
       Ascend-hdk-<soc>-npu-driver_<ver>_linux-<arch>.run --full
       Ascend-hdk-<soc>-npu-firmware_<ver>_linux-<arch>.run --full
       reboot   (or: npu-smi set -t reset -i <id>)
       npu-smi info          # must list every chip before the benchmark runs

  2) CANN toolkit  (driver bind mount, HCCL, HCOMM, ccec, msprof, npu-smi)
       Ascend-cann-toolkit_<ver>_linux-<arch>.run --install
       Ascend-cann-kernels-<soc>_<ver>_linux-<arch>.run --install
       source /usr/local/Ascend/ascend-toolkit/set_env.sh

  3) Device permissions
       usermod -aG HwHiAiUser <user>        # on the host, then re-login
       docker run --device=/dev/davinci0 --device=/dev/davinci_manager \
                  --device=/dev/devmm_svm ...

  4) hccl_test (CANN oam-tools, built on the host)
       export EP_HCCL_TEST_BINARY=<oam-tools>/tools/hccl_test/bin/alltoall_test

The container may only: install wheels (dependencies/python/*.txt), build
extensions against the installed toolkit (SHMEM, DeepEP, npu_patch, ASC-COMM),
and install cmake/g++ from its own package manager.
NOTICE
}

# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
precheck_snapshot() {
    if [ "$DRY_RUN" -eq 1 ]; then
        return 0
    fi
    step "pre-install snapshot (idempotency: what is already present)"
    if ! "$PYTHON" "$DOCTOR" --profile "$PROFILE" --json >"$TMP_STATE" 2>"$TMP_LOG"; then
        warn "the pre-install probe reported problems; continuing (log: $TMP_LOG)"
    fi
}

verify() {
    step "verify: scripts/doctor.py --profile $PROFILE"
    local status=0
    if [ -n "$WHEELHOUSE" ]; then
        "$PYTHON" "$DOCTOR" --profile "$PROFILE" --offline-wheelhouse "$WHEELHOUSE" || status=$?
    else
        "$PYTHON" "$DOCTOR" --profile "$PROFILE" || status=$?
    fi
    if [ "$BLOCKERS" -gt 0 ]; then
        printf '\n%d blocker(s) above were not resolved; this environment is not ready.\n' "$BLOCKERS" >&2
    fi
    if [ "$status" -ne 0 ]; then
        printf 'doctor exit=%s: profile %s is not ready (see the FAIL lines)\n' "$status" "$PROFILE" >&2
        exit "$status"
    fi
    if [ "$BLOCKERS" -gt 0 ]; then
        exit 1
    fi
    say "profile $PROFILE is ready"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
main() {
    parse_args "$@"
    preflight
    precheck_snapshot

    install_python_base
    if profile_has B; then
        install_torch_npu
    fi
    if profile_has D; then
        install_mpi
    fi
    if profile_has C; then
        install_native_communication
        install_asc_comm
    fi
    if profile_has E; then
        step "category E: profiling tools and build toolchain"
        say "msprof and hccl_test are host components shipped inside a CANN release;"
        say "they are never installed here.  The build toolchain is:"
        say "  hccl_test: build CANN oam-tools on the host, then"
        say "             export EP_HCCL_TEST_BINARY=<oam-tools>/tools/hccl_test/bin/alltoall_test"
        install_system_packages "cmake" cmake "cmake"
        install_system_packages "g++" g++ "g++"
        if ! command -v ccec >/dev/null 2>&1; then
            blocker "ccec (AscendC compiler) is missing; it ships inside the CANN toolkit on the host"
        fi
    fi
    host_boundary_notice
    verify
}

main "$@"
