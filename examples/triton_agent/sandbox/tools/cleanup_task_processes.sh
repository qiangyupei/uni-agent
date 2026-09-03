#!/usr/bin/env bash
set -uo pipefail

state_dir="${PWD}/.triton_verify_processes"
shopt -s nullglob

pid_alive() {
  local state
  state="$(ps -o stat= -p "$1" 2>/dev/null | awk 'NR == 1 { print $1 }')"
  [[ -n "${state}" && "${state:0:1}" != Z ]]
}

group_alive() {
  ps -eo pgid=,stat= 2>/dev/null | awk -v pgid="$1" '
    $1 == pgid && substr($2, 1, 1) != "Z" { found = 1; exit }
    END { exit !found }
  '
}

signal_registered() {
  local signal=$1 state wrapper kind child
  for state in "${state_dir}"/*.state; do
    read -r wrapper kind child <"${state}" || continue
    [[ "${wrapper}" =~ ^[0-9]+$ && "${wrapper}" -gt 1 ]] && kill "-${signal}" "${wrapper}" 2>/dev/null || true
    [[ "${kind}" == pgid && "${child}" =~ ^[0-9]+$ && "${child}" -gt 1 ]] \
      && kill "-${signal}" -- "-${child}" 2>/dev/null || true
  done
  command -v pkill >/dev/null 2>&1 && pkill "-${signal}" -x claude 2>/dev/null || true
}

registered_alive() {
  local state wrapper kind child
  for state in "${state_dir}"/*.state; do
    read -r wrapper kind child <"${state}" || continue
    [[ "${wrapper}" =~ ^[0-9]+$ ]] && pid_alive "${wrapper}" && return 0
    [[ "${kind}" == pgid && "${child}" =~ ^[0-9]+$ ]] && group_alive "${child}" && return 0
  done
  return 1
}

signal_registered TERM
for _ in {1..50}; do
  registered_alive || break
  sleep 0.1
done
if registered_alive; then
  signal_registered KILL
  for _ in {1..20}; do
    registered_alive || break
    sleep 0.1
  done
fi

registered_alive && exit 1

rm -f "${state_dir}"/*.state
rmdir "${state_dir}" 2>/dev/null || true
