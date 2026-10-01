#!/usr/bin/env bash
# Run one job inside a user-systemd scope with a hard memory cap, so a runaway job is killed alone (exit 137)
# instead of taking the WSL VM down. Added 2026-09-28 after the R0 SLT run built a 26.5 GB results pickle and WSL
# went down at 15:21 (outputs/recent_baselines_2026-09-28/HANDOFF.md). The cap counts the job's RSS plus the page
# cache it charges. Before the scope exits, that page cache is reclaimed so vmmemWSL can hand it back to Windows.
# Caps: SIGNGEN_MEM_MAX (default 16G) and SIGNGEN_SWAP_MAX (default 2G).
# If the user systemd bus is unreachable, exit 97 (the chains' retry-in-5-min code), so a job never runs uncapped.
# light.sh and heavy.sh call this inside their locks. usage: memcap.sh <command...>
set -u
export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}
if ! systemctl --user show-environment > /dev/null 2>&1; then
  echo "MEMCAP FAIL: user systemd bus unreachable, job not started uncapped" >&2; exit 97
fi
cap=${SIGNGEN_MEM_MAX:-16G}
exec systemd-run --user --scope --collect --quiet --slice=signgen.slice \
  -p MemoryMax="$cap" -p MemorySwapMax="${SIGNGEN_SWAP_MAX:-2G}" -p OOMPolicy=continue \
  bash -c '"$@"; rc=$?
cg=/sys/fs/cgroup$(sed -n "s/^0:://p" /proc/self/cgroup)
peak=$(( $(cat "$cg/memory.peak" 2>/dev/null || echo 0) / 1048576 ))
echo 64G > "$cg/memory.reclaim" 2>/dev/null
echo "[memcap] exit $rc, peak ${peak} MiB of cap '"$cap"', $(date -Is)" >&2
exit $rc' _ "$@"
